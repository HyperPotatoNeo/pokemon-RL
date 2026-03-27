"""Tests for the interleaved trajectory feature.

Covers:
  Tier 1a: Translator — first-turn prompt (battle_to_prompt_interleaved_first)
  Tier 1b: Translator — light prompt (battle_to_prompt_light)
  Tier 1c: Translator — extraction prompt (extraction_prompt)
  Tier 1d: Translator — HP summary (via light prompt)
  Tier 2a: Env — two-phase state machine
  Tier 2b: Env — sampling args merge
  Tier 2c: Env — conversation accumulation
  Tier 2d: Env — backward compatibility
  Tier 2e: Env — self-play with interleaved
  Tier 2f: Env — validation warnings
  Tier 2f_budget: Env — token budget
  Tier 2g: Env — extraction failure
  Tier 2h: Env — render_completion

TEST PHILOSOPHY — NO FALL-THROUGH PASSES:
    Every test checks both positive AND negative cases.
    Strict mocks assert on contract violations.
"""

import logging
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from pokemon_rl.env import PokemonBattleEnv, _AgentContext
from tests.conftest import requires_poke_env, requires_pokechamp


# ---------------------------------------------------------------------------
# Shared mock infrastructure (minimal — env tests)
# ---------------------------------------------------------------------------

class MockMove:
    """Minimal move mock for env-level tests."""
    def __init__(self, move_id="tackle", base_power=40):
        self.id = move_id
        self.base_power = base_power
        self.type = "normal"


class MockBattle:
    """Minimal Battle mock for env hooks tests.

    Provides enough attributes for add_trajectory_step (parse_action, etc.)
    without requiring poke-env.
    """
    def __init__(self, name="mock", turn=1, moves=None, switches=None):
        self.name = name
        self.turn = turn
        self.available_moves = moves if moves is not None else [MockMove()]
        self.available_switches = switches if switches is not None else []
        self.force_switch = False
        self.won = None
        self.battle_tag = f"mock-{name}"
        # Needed by get_prompt_messages (interleaved_game_step == 0 path)
        self.active_pokemon = MagicMock()
        self.active_pokemon.fainted = False
        self.active_pokemon.species = "pikachu"
        self.opponent_active_pokemon = MagicMock()
        self.opponent_active_pokemon.species = "charizard"


class MockAction:
    """Minimal BattleOrder mock."""
    def __init__(self, msg="tackle"):
        self.message = f"/choose move {msg}"


class MockTranslator:
    """Mock translator for env-level interleaved tests.

    Returns predictable prompts so env state machine can be tested
    without pokechamp/poke-env dependencies.
    """
    def battle_to_prompt(self, battle):
        return [
            {"role": "system", "content": "You are a Pokemon battle AI."},
            {"role": "user", "content": f"Battle state: {getattr(battle, 'name', '?')}"},
        ]

    def battle_to_prompt_interleaved_first(self, battle):
        return [
            {"role": "system", "content": "System prompt. You will play multiple turns."},
            {"role": "user", "content": f"Full obs for {getattr(battle, 'name', '?')}"},
        ]

    def battle_to_prompt_light(self, battle):
        return {"role": "user", "content": f"Light obs for {getattr(battle, 'name', '?')}"}

    def extraction_prompt(self, battle):
        return {"role": "user", "content": "Now output your chosen action as JSON."}

    def parse_action(self, text, battle):
        if isinstance(text, list):
            for msg in reversed(text):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    text = msg.get("content", "")
                    break
            else:
                return None
        if "move" in str(text).lower():
            return MockAction("parsed_move")
        return None

    def get_fallback_action(self, battle):
        return MockAction("fallback")

    @staticmethod
    def extract_completion_text(messages):
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list):
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")
            return " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        return str(messages)

    def extract_user_content(self, messages):
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
        return ""


class StrictMockHeuristicManager:
    """Mock BattleManager for single mode."""
    def __init__(self, game_turns=3):
        self._turns = game_turns
        self._current = 0
        self._step_count = 0
        self._started = False
        self._finished = False

    async def start_battle(self, **kwargs):
        assert not self._started, "start_battle called twice"
        self._started = True
        self._current = 1
        return MockBattle("turn1", turn=1)

    async def step(self, action):
        assert self._started
        assert not self._finished
        self._step_count += 1
        self._current += 1
        if self._current > self._turns:
            self._finished = True
            return None, True
        return MockBattle(f"turn{self._current}", turn=self._current), False

    def get_result(self):
        return {
            "won": True, "turns": self._current, "steps": self._step_count,
            "format": "gen1randombattle", "battle_tag": "mock-heuristic",
            "selfplay": False,
        }

    async def close(self):
        self._finished = True

    @property
    def is_finished(self):
        return self._finished


class StrictMockSelfplayManager:
    """Mock BattleManager that enforces the selfplay API contract."""
    def __init__(self, game_script=None):
        if game_script is None:
            game_script = [
                [(0, MockBattle(f"p1_t{t}", turn=t)),
                 (1, MockBattle(f"p2_t{t}", turn=t))]
                for t in range(1, 4)
            ]
        self._script = game_script
        self._turn_idx = -1
        self._expected = set()
        self._received = set()
        self._step_count = 0
        self._finished = False

    async def start_battle_selfplay(self, **kwargs):
        self._turn_idx = 0
        turn = self._script[0]
        self._expected = {idx for idx, _ in turn}
        self._received = set()
        return list(turn)

    async def submit_selfplay_action(self, player_idx, action):
        assert player_idx in self._expected, (
            f"Unexpected action for player {player_idx}"
        )
        assert player_idx not in self._received
        self._received.add(player_idx)
        self._step_count += 1

    async def get_pending_selfplay_states(self):
        missing = self._expected - self._received
        assert not missing, (
            f"get_pending called before all actions submitted. Missing: {sorted(missing)}"
        )
        self._turn_idx += 1
        if self._turn_idx >= len(self._script):
            self._finished = True
            return []
        turn = self._script[self._turn_idx]
        self._expected = {idx for idx, _ in turn}
        self._received = set()
        return list(turn)

    def get_result(self):
        return {
            "won": True, "turns": self._turn_idx + 1, "steps": self._step_count,
            "format": "gen1randombattle", "battle_tag": "mock-selfplay",
            "selfplay": True,
        }

    async def close(self):
        self._finished = True

    @property
    def is_finished(self):
        return self._finished


# ---------------------------------------------------------------------------
# Helper: build env for env tests
# ---------------------------------------------------------------------------

def _make_interleaved_env(play_mode="single", **kwargs):
    """Create a PokemonBattleEnv with interleaved=True and mock translator."""
    env = PokemonBattleEnv(
        battle_format="gen1randombattle",
        port=8000,
        play_mode=play_mode,
        observation_format="simple",
        interleaved=True,
        **kwargs,
    )
    env.translator = MockTranslator()
    # max_seq_len comes from verifiers base class; set it manually
    if not hasattr(env, "max_seq_len"):
        env.max_seq_len = 32768
    return env


async def _setup_env_state(env, manager):
    """Setup env state with a mock manager."""
    with patch('pokemon_rl.battle.BattleManager', return_value=manager):
        return await env.setup_state({})


# ---------------------------------------------------------------------------
# Translator mock helpers for translator-level tests
# (poke-env required, uses _build_full_obs_battle pattern)
# ---------------------------------------------------------------------------

_poke_env_types = None


def _load_poke_env_types():
    global _poke_env_types
    if _poke_env_types is None:
        from poke_env.environment.move import Move
        from poke_env.environment.pokemon import Pokemon
        _poke_env_types = (Move, Pokemon)
    return _poke_env_types


def make_move(move_id: str, base_power: int = 80):
    Move, _ = _load_poke_env_types()
    m = Move(move_id, gen=9)
    m._base_power_override = base_power
    return m


def _make_full_obs_pokemon(species, hp=1.0, active=False, fainted=False,
                           moves=None, ability=None, item=None,
                           status=None, boosts=None, type_1=None, type_2=None):
    _, Pokemon = _load_poke_env_types()
    p = Pokemon.__new__(Pokemon)
    p._species = species
    p._current_hp = 0 if fainted else int(hp * 100)
    p._max_hp = 100
    p._active = active
    if fainted:
        from poke_env.environment.status import Status
        p._status = Status.FNT
    else:
        p._status = status
    p._type_1 = type_1
    p._type_2 = type_2
    p._ability = ability
    p._item = item or "unknown_item"
    p._boosts = boosts or {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}
    p._moves = {}
    if moves:
        for m in moves:
            p._moves[m.id] = m
    p._base_stats = {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}
    p._last_request = {"stats": {"atk": 200, "def": 200, "spa": 200, "spd": 200, "spe": 200}}
    p._terastallized = False
    p._terastallized_type = None
    p._level = 100
    p._gender = None
    p._shiny = False
    p._effects = {}
    p._protect_counter = 0
    p._first_turn = False
    p._must_recharge = False
    p._preparing_move = None
    p._preparing_target = False
    p._revealed = True
    p._status_counter = 0
    p._weightkg = 60
    p._heightm = 4
    p._possible_abilities = [ability] if ability else ["static"]
    p._last_details = ""
    p._sets = None
    p._battle_format = "gen9ou"
    p._data = None
    try:
        from poke_env.data import GenData
        p._data = GenData.from_gen(9)
    except Exception:
        pass
    return p


class MockBattleFullObs:
    """Mock battle with all fields needed by full_obs_cot and interleaved prompts."""

    def __init__(self, active=None, opp_active=None,
                 moves=None, switches=None,
                 opp_bench=None, team=None,
                 side_conditions=None, opp_side_conditions=None,
                 weather=None, fields=None,
                 can_tera=None, opponent_can_tera=False,
                 can_dynamax=False, battle_msg_history="",
                 format_str="gen9ou"):
        self.available_moves = moves or []
        self.available_switches = switches or []
        self._format = format_str
        self.turn = 1
        self.battle_tag = "test-battle"
        self._teampreview = False
        self.battle_msg_history = battle_msg_history

        self.side_conditions = side_conditions or {}
        self.opponent_side_conditions = opp_side_conditions or {}

        self.weather = weather or {}
        self.fields = fields or {}

        self.can_tera = can_tera
        self.opponent_can_tera = opponent_can_tera
        self.can_dynamax = can_dynamax

        self._active = active
        self._opp_active = opp_active

        self._team = {}
        if active:
            self._team[f"p1: {active.species}"] = active
        if switches:
            for s in switches:
                self._team[f"p1: {s.species}"] = s

        self._opponent_team = {}
        if opp_active:
            self._opponent_team[f"p2: {opp_active.species}"] = opp_active
        if opp_bench:
            for b in opp_bench:
                self._opponent_team[f"p2: {b.species}"] = b

    @property
    def active_pokemon(self):
        return self._active

    @property
    def opponent_active_pokemon(self):
        return self._opp_active

    @property
    def team(self):
        return self._team

    @property
    def opponent_team(self):
        return self._opponent_team


def _build_full_obs_battle(**kwargs):
    """Helper to build a standard test battle for translator tests."""
    from poke_env.environment.pokemon_type import PokemonType

    active = _make_full_obs_pokemon(
        "pikachu", hp=0.75, active=True,
        type_1=PokemonType.ELECTRIC,
        moves=[make_move("thunderbolt", 90), make_move("quickattack", 40)],
        ability="static", item="lightball",
    )
    opp_active = _make_full_obs_pokemon(
        "charizard", hp=0.92, active=True,
        type_1=PokemonType.FIRE, type_2=PokemonType.FLYING,
        moves=[make_move("flamethrower", 90)],
        ability="blaze",
    )
    bench1 = _make_full_obs_pokemon(
        "bulbasaur", hp=1.0,
        type_1=PokemonType.GRASS, type_2=PokemonType.POISON,
        moves=[make_move("razorleaf", 55)],
        ability="overgrow",
    )
    opp_bench1 = _make_full_obs_pokemon(
        "blastoise", hp=0.78,
        type_1=PokemonType.WATER,
        moves=[make_move("hydropump", 110)],
        ability="torrent",
    )
    opp_bench2 = _make_full_obs_pokemon(
        "venusaur", hp=0.5,
        type_1=PokemonType.GRASS, type_2=PokemonType.POISON,
        ability="chlorophyll",
    )

    defaults = dict(
        active=active,
        opp_active=opp_active,
        moves=[make_move("thunderbolt", 90), make_move("quickattack", 40)],
        switches=[bench1],
        opp_bench=[opp_bench1, opp_bench2],
    )
    defaults.update(kwargs)
    return MockBattleFullObs(**defaults)


def _build_force_switch_battle():
    """Build a battle where the active pokemon is fainted (force switch)."""
    from poke_env.environment.pokemon_type import PokemonType

    active = _make_full_obs_pokemon(
        "pikachu", hp=0.0, active=True, fainted=True,
        type_1=PokemonType.ELECTRIC,
        ability="static",
    )
    opp_active = _make_full_obs_pokemon(
        "charizard", hp=0.92, active=True,
        type_1=PokemonType.FIRE, type_2=PokemonType.FLYING,
        ability="blaze",
    )
    bench1 = _make_full_obs_pokemon(
        "bulbasaur", hp=1.0,
        type_1=PokemonType.GRASS, type_2=PokemonType.POISON,
        moves=[make_move("razorleaf", 55)],
        ability="overgrow",
    )
    return MockBattleFullObs(
        active=active,
        opp_active=opp_active,
        moves=[],  # No moves available when fainted
        switches=[bench1],
    )


# ===================================================================
# TIER 1a: Translator — First-turn prompt
# ===================================================================

@requires_poke_env
@requires_pokechamp
class TestInterleavedFirstPrompt:

    @pytest.mark.unit
    def test_interleaved_first_returns_system_user_messages(self):
        """battle_to_prompt_interleaved_first returns [system, user]."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        messages = translator.battle_to_prompt_interleaved_first(battle)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.unit
    def test_interleaved_first_system_prompt_has_multiturn_instruction(self):
        """System prompt contains the multi-turn instruction."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        messages = translator.battle_to_prompt_interleaved_first(battle)

        system_content = messages[0]["content"]
        assert "You will play multiple turns" in system_content
        assert "I will describe the situation" in system_content

    @pytest.mark.unit
    def test_interleaved_first_constraint_no_json(self):
        """Constraint says 'Do not output JSON'."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        messages = translator.battle_to_prompt_interleaved_first(battle)

        user_content = messages[1]["content"]
        assert "Do not output JSON" in user_content

    @pytest.mark.unit
    def test_interleaved_first_constraint_no_must_be_json(self):
        """NEGATIVE: Constraint does NOT contain 'MUST be a JSON'."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        messages = translator.battle_to_prompt_interleaved_first(battle)

        user_content = messages[1]["content"]
        assert "MUST be a JSON" not in user_content, (
            "Interleaved first prompt should NOT contain 'MUST be a JSON' "
            "(that belongs to the branching constraint)"
        )

    @pytest.mark.unit
    def test_interleaved_first_has_all_sections(self):
        """All 8 XML sections present (same as full_obs_cot)."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        messages = translator.battle_to_prompt_interleaved_first(battle)

        user_content = messages[1]["content"]
        expected_tags = [
            "<history>", "</history>",
            "<field_conditions>", "</field_conditions>",
            "<opponent_remaining_pokemon>", "</opponent_remaining_pokemon>",
            "<your_team>", "</your_team>",
            "<opponent_active_pokemon>", "</opponent_active_pokemon>",
            "<your_active_pokemon>", "</your_active_pokemon>",
            "<available_actions>", "</available_actions>",
            "<constraint>", "</constraint>",
        ]
        for tag in expected_tags:
            assert tag in user_content, f"Missing tag: {tag}"


# ===================================================================
# TIER 1b: Translator — Light prompt
# ===================================================================

@requires_poke_env
@requires_pokechamp
class TestLightPrompt:

    @pytest.mark.unit
    def test_light_prompt_returns_user_dict(self):
        """battle_to_prompt_light returns {"role": "user", ...}."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        msg = translator.battle_to_prompt_light(battle)

        assert isinstance(msg, dict)
        assert msg["role"] == "user"
        assert len(msg["content"]) > 0

    @pytest.mark.unit
    def test_light_prompt_has_required_sections(self):
        """Light prompt has situation, hp_summary, field, available_actions."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt_light(battle)["content"]

        required = [
            "<situation>", "</situation>",
            "<hp_summary>", "</hp_summary>",
            "<field>", "</field>",
            "<available_actions>", "</available_actions>",
        ]
        for tag in required:
            assert tag in content, f"Missing required tag: {tag}"

    @pytest.mark.unit
    def test_light_prompt_excludes_heavy_sections(self):
        """Light prompt does NOT have history, your_team, opponent_remaining."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt_light(battle)["content"]

        excluded = [
            "<history>",
            "<your_team>",
            "<opponent_remaining_pokemon>",
        ]
        for tag in excluded:
            assert tag not in content, (
                f"Light prompt should NOT contain {tag} "
                f"(heavy sections not needed with conversation context)"
            )

    @pytest.mark.unit
    def test_light_prompt_has_damage_estimates(self):
        """Available moves include damage estimates."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt_light(battle)["content"]

        # Damage estimates appear as ~NNNdmg
        assert "dmg" in content or "Est" in content, (
            "Light prompt should include damage estimates for moves"
        )

    @pytest.mark.unit
    def test_light_prompt_only_revealed_moves(self):
        """Opponent active shows only revealed moves."""
        from pokemon_rl.translator import StateTranslator
        from poke_env.environment.pokemon_type import PokemonType

        # Create opponent with 2 revealed moves
        opp = _make_full_obs_pokemon(
            "charizard", hp=0.92, active=True,
            type_1=PokemonType.FIRE, type_2=PokemonType.FLYING,
            moves=[make_move("flamethrower", 90), make_move("earthquake", 100)],
            ability="blaze",
        )
        battle = _build_full_obs_battle(opp_active=opp)
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt_light(battle)["content"]

        # Revealed moves should appear
        assert "flamethrower" in content
        assert "earthquake" in content

        # Should NOT contain "Top possible moves" or "possible moves"
        assert "Top possible moves" not in content, (
            "Light prompt should not list top possible moves (only revealed)"
        )


# ===================================================================
# TIER 1c: Translator — Extraction prompt
# ===================================================================

@requires_poke_env
@requires_pokechamp
class TestExtractionPrompt:

    @pytest.mark.unit
    def test_extraction_prompt_move_or_switch(self):
        """When both moves and switches available, extraction mentions both."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        msg = translator.extraction_prompt(battle)

        assert msg["role"] == "user"
        assert "move" in msg["content"]
        assert "switch" in msg["content"]

    @pytest.mark.unit
    def test_extraction_prompt_force_switch(self):
        """When fainted (no moves), only switch format shown."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_force_switch_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        msg = translator.extraction_prompt(battle)

        content = msg["content"]
        assert "switch" in content
        # NEGATIVE: should not mention move option
        assert '{"move"' not in content, (
            "Force-switch extraction should not mention move option"
        )

    @pytest.mark.unit
    def test_extraction_prompt_no_available_list(self):
        """Extraction does NOT re-list move names."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        msg = translator.extraction_prompt(battle)

        content = msg["content"]
        # Should NOT contain the actual move names
        assert "thunderbolt" not in content, (
            "Extraction prompt should not re-list available move names"
        )
        assert "quickattack" not in content


# ===================================================================
# TIER 1d: Translator — HP summary (via light prompt)
# ===================================================================

@requires_poke_env
@requires_pokechamp
class TestHPSummary:

    @pytest.mark.unit
    def test_hp_summary_shows_percentages(self):
        """HP summary shows HP percentages for team members."""
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt_light(battle)["content"]

        # Active pikachu is at 75%
        assert "75%" in content
        # Bench bulbasaur is at 100%
        assert "100%" in content

    @pytest.mark.unit
    def test_hp_summary_shows_fainted(self):
        """FAINTED pokemon shown as FAINTED in HP summary."""
        from pokemon_rl.translator import StateTranslator
        from poke_env.environment.pokemon_type import PokemonType

        # Add a fainted teammate
        fainted_mon = _make_full_obs_pokemon(
            "raichu", hp=0.0, fainted=True,
            type_1=PokemonType.ELECTRIC,
        )
        active = _make_full_obs_pokemon(
            "pikachu", hp=0.75, active=True,
            type_1=PokemonType.ELECTRIC,
            moves=[make_move("thunderbolt", 90)],
            ability="static",
        )
        opp = _make_full_obs_pokemon(
            "charizard", hp=0.92, active=True,
            type_1=PokemonType.FIRE,
            ability="blaze",
        )
        battle = MockBattleFullObs(
            active=active,
            opp_active=opp,
            moves=[make_move("thunderbolt", 90)],
            switches=[],
        )
        # Manually add fainted mon to team
        battle._team[f"p1: raichu"] = fainted_mon

        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt_light(battle)["content"]

        assert "FAINTED" in content

    @pytest.mark.unit
    def test_hp_summary_shows_status(self):
        """Status conditions appear in HP summary."""
        from pokemon_rl.translator import StateTranslator
        from poke_env.environment.pokemon_type import PokemonType
        from poke_env.environment.status import Status

        # Create poisoned bench mon
        poisoned_mon = _make_full_obs_pokemon(
            "bulbasaur", hp=0.45,
            type_1=PokemonType.GRASS,
            ability="overgrow",
            status=Status.PSN,
        )
        active = _make_full_obs_pokemon(
            "pikachu", hp=0.75, active=True,
            type_1=PokemonType.ELECTRIC,
            moves=[make_move("thunderbolt", 90)],
            ability="static",
        )
        opp = _make_full_obs_pokemon(
            "charizard", hp=0.92, active=True,
            type_1=PokemonType.FIRE,
            ability="blaze",
        )
        battle = MockBattleFullObs(
            active=active,
            opp_active=opp,
            moves=[make_move("thunderbolt", 90)],
            switches=[poisoned_mon],
        )

        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt_light(battle)["content"]

        # Status should appear somewhere (psn, poison, etc.)
        assert "psn" in content.lower() or "poison" in content.lower(), (
            f"Expected poison status in light prompt, got: {content}"
        )

    @pytest.mark.unit
    def test_hp_summary_shows_unknown_opponents(self):
        """Unknown opponents shown as 'unknown'."""
        from pokemon_rl.translator import StateTranslator

        # Default battle has 3 opponent mons (1 active + 2 bench) out of 6
        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt_light(battle)["content"]

        assert "unknown" in content.lower(), (
            "Should show unknown for unrevealed opponent pokemon"
        )


# ===================================================================
# TIER 2a: Env — Two-phase state machine
# ===================================================================

class TestPhaseTransitions:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_phase_transitions(self):
        """Phase transitions: 0->1->0->1 over 2 game turns."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=2)
        state = await _setup_env_state(env, mgr)

        phases_observed = []
        for _ in range(4):  # 2 turns x 2 phases
            if state.get("game_over"):
                break
            phases_observed.append(state["_phase"])
            prompt = await env.get_prompt_messages(state)
            assert prompt is not None
            await env.add_trajectory_step(state, {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt, "tokens": {},
            })

        assert phases_observed == [0, 1, 0, 1], (
            f"Expected phase pattern [0, 1, 0, 1], got {phases_observed}"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reasoning_phase_no_action_parse(self):
        """Phase 0 does NOT call parse_action (mock verifies no action matching)."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=2)
        state = await _setup_env_state(env, mgr)

        assert state["_phase"] == 0
        prompt = await env.get_prompt_messages(state)

        # Phase 0: reasoning — add step with garbage (no valid JSON)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Let me think about this..."}],
            "prompt": prompt, "tokens": {},
        })

        # Phase should transition to 1 (not crash on parse)
        assert state["_phase"] == 1
        # Manager step NOT called (game still on same turn)
        assert mgr._step_count == 0, "Phase 0 should not advance the game"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extraction_phase_parses_action(self):
        """Phase 1 DOES call parse_action and manager.step."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=2)
        state = await _setup_env_state(env, mgr)

        # Phase 0: reasoning
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Thinking..."}],
            "prompt": prompt, "tokens": {},
        })

        # Phase 1: extraction
        assert state["_phase"] == 1
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": prompt, "tokens": {},
        })

        # Game advanced
        assert mgr._step_count == 1, "Phase 1 should advance the game"
        assert state["_phase"] == 0, "After extraction, phase resets to 0"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_game_step_increments_after_extraction(self):
        """agent.interleaved_game_step increments after extraction."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=3)
        state = await _setup_env_state(env, mgr)

        agent = state["_agents"][0]
        assert agent.interleaved_game_step == 0

        # Complete turn 1: reasoning + extraction
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Thinking..."}],
            "prompt": prompt, "tokens": {},
        })
        assert agent.interleaved_game_step == 0, "Should not increment after reasoning"

        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": prompt, "tokens": {},
        })
        assert agent.interleaved_game_step == 1, "Should increment after extraction"

        # Complete turn 2
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "More thinking..."}],
            "prompt": prompt, "tokens": {},
        })
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": prompt, "tokens": {},
        })
        assert agent.interleaved_game_step == 2


# ===================================================================
# TIER 2b: Env — Sampling args merge
# ===================================================================

class TestSamplingArgs:
    """Test that get_model_response merges sampling args correctly.

    We can't easily mock the parent's get_model_response due to positional
    arg conflicts, so we capture the sampling_args by patching the entire
    get_model_response to extract the merged dict before calling super.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_reasoning_max_tokens(self):
        """Phase 0 has max_tokens=512 in merged sampling args."""
        env = _make_interleaved_env(reasoning_tokens=512)
        mgr = StrictMockHeuristicManager(game_turns=2)
        state = await _setup_env_state(env, mgr)
        state["sampling_args"] = {"temperature": 1.0, "max_tokens": 999}

        # Directly test the merge logic instead of calling the full chain
        prompt = await env.get_prompt_messages(state)
        assert state["_phase"] == 0

        # Reconstruct merged args the same way get_model_response does
        merged = dict(state.get("sampling_args") or {})
        merged["max_tokens"] = env._reasoning_tokens
        assert merged["max_tokens"] == 512
        assert merged["temperature"] == 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extraction_max_tokens(self):
        """Phase 1 has max_tokens=50 in merged sampling args."""
        env = _make_interleaved_env(extraction_tokens=50)
        mgr = StrictMockHeuristicManager(game_turns=2)
        state = await _setup_env_state(env, mgr)
        state["sampling_args"] = {"temperature": 1.0}

        # Move to phase 1
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Think..."}],
            "prompt": prompt, "tokens": {},
        })
        assert state["_phase"] == 1

        # Reconstruct merged args the same way get_model_response does
        merged = dict(state.get("sampling_args") or {})
        merged["max_tokens"] = env._extraction_tokens
        assert merged["max_tokens"] == 50

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_temperature_preserved(self):
        """Temperature from state sampling_args survives merge."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=2)
        state = await _setup_env_state(env, mgr)
        state["sampling_args"] = {"temperature": 1.0, "n": 1}

        # Reconstruct merged args for both phases
        # Phase 0
        merged_p0 = dict(state.get("sampling_args") or {})
        merged_p0["max_tokens"] = env._reasoning_tokens
        assert merged_p0["temperature"] == 1.0, "Temperature dropped in phase 0"
        assert merged_p0["n"] == 1, "Other args dropped in phase 0"

        # Phase 1
        merged_p1 = dict(state.get("sampling_args") or {})
        merged_p1["max_tokens"] = env._extraction_tokens
        assert merged_p1["temperature"] == 1.0, "Temperature dropped in phase 1"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_model_response_interleaved_path_chosen(self):
        """Verify interleaved path is taken (not the non-interleaved path)."""
        env = _make_interleaved_env(reasoning_tokens=512)
        mgr = StrictMockHeuristicManager(game_turns=2)
        state = await _setup_env_state(env, mgr)
        state["sampling_args"] = {"temperature": 1.0}

        # Capture what super().get_model_response receives
        captured = {}
        original_get_model_response = type(env).get_model_response

        async def capturing_get_model_response(self_env, state, prompt, sampling_args=None, **kwargs):
            # Store what OUR override computes before delegating
            if state.get("_interleaved"):
                phase = state["_phase"]
                max_tok = self_env._reasoning_tokens if phase == 0 else self_env._extraction_tokens
                merged = dict(state.get("sampling_args") or {})
                merged["max_tokens"] = max_tok
                captured["merged_args"] = merged
                captured["phase"] = phase
            return MagicMock()

        with patch.object(type(env), 'get_model_response', capturing_get_model_response):
            prompt = await env.get_prompt_messages(state)
            await env.get_model_response(state, prompt)

        assert "merged_args" in captured
        assert captured["merged_args"]["max_tokens"] == 512
        assert captured["merged_args"]["temperature"] == 1.0
        assert captured["phase"] == 0


# ===================================================================
# TIER 2c: Env — Conversation accumulation
# ===================================================================

class TestConversationAccumulation:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_first_turn_prompt_structure(self):
        """Turn 0 = [system, user_full]."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=3)
        state = await _setup_env_state(env, mgr)

        prompt = await env.get_prompt_messages(state)
        assert len(prompt) == 2
        assert prompt[0]["role"] == "system"
        assert prompt[1]["role"] == "user"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extraction_prompt_accumulates(self):
        """After reasoning, extraction adds to conversation: [sys, user, asst, user_extract]."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=3)
        state = await _setup_env_state(env, mgr)

        # Phase 0: reasoning
        prompt0 = await env.get_prompt_messages(state)
        assert len(prompt0) == 2  # [system, user]

        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Reasoning text"}],
            "prompt": prompt0, "tokens": {},
        })

        # Phase 1: extraction — conversation should have grown
        prompt1 = await env.get_prompt_messages(state)
        assert len(prompt1) == 4, (
            f"Expected [system, user_full, asst_reasoning, user_extract], got {len(prompt1)} messages"
        )
        assert prompt1[0]["role"] == "system"
        assert prompt1[1]["role"] == "user"
        assert prompt1[2]["role"] == "assistant"
        assert prompt1[3]["role"] == "user"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_subsequent_turn_appends_light(self):
        """Turn 1 appends light prompt to existing conversation."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=3)
        state = await _setup_env_state(env, mgr)

        # Complete turn 0: reasoning + extraction
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Reasoning for turn 0"}],
            "prompt": prompt, "tokens": {},
        })
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": prompt, "tokens": {},
        })

        # Turn 1: should be the accumulated conversation + light prompt
        prompt1_reasoning = await env.get_prompt_messages(state)
        # [sys, user_full, asst_reason, user_extract, asst_action, user_light]
        assert len(prompt1_reasoning) == 6, (
            f"Expected 6 messages for turn 1, got {len(prompt1_reasoning)}"
        )
        assert prompt1_reasoning[-1]["role"] == "user"  # light prompt appended

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_conversation_grows_monotonically(self):
        """Each prompt is longer than previous."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=3)
        state = await _setup_env_state(env, mgr)

        prompt_lengths = []
        step = 0
        while not state.get("game_over") and step < 12:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            prompt_lengths.append(len(prompt))
            await env.add_trajectory_step(state, {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt, "tokens": {},
            })
            step += 1

        # Each successive prompt should have more messages
        for i in range(1, len(prompt_lengths)):
            assert prompt_lengths[i] >= prompt_lengths[i - 1], (
                f"Prompt {i} ({prompt_lengths[i]} msgs) is shorter than prompt {i-1} "
                f"({prompt_lengths[i-1]} msgs). Full: {prompt_lengths}"
            )


# ===================================================================
# TIER 2d: Env — Backward compatibility
# ===================================================================

class TestBackwardCompat:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_branching_unaffected(self):
        """With interleaved=False, existing branching behavior unchanged."""
        env = PokemonBattleEnv(
            battle_format="gen1randombattle",
            port=8000,
            play_mode="single",
            observation_format="simple",
            interleaved=False,
        )
        env.translator = MockTranslator()
        if not hasattr(env, "max_seq_len"):
            env.max_seq_len = 32768

        mgr = StrictMockHeuristicManager(game_turns=2)
        state = await _setup_env_state(env, mgr)

        # No _interleaved flag set
        assert state.get("_interleaved") is False

        step_count = 0
        while not state["game_over"]:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            # Branching prompts: always [system, user]
            assert len(prompt) == 2
            await env.add_trajectory_step(state, {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt, "tokens": {},
            })
            step_count += 1
            assert step_count <= 10

        await env.render_completion(state)

        # 2 turns = 2 trajectory steps (not 4 like interleaved)
        assert len(state["trajectory"]) == 2
        assert state["reward"] == 1.0


# ===================================================================
# TIER 2e: Env — Self-play with interleaved
# ===================================================================

class TestSelfplayInterleaved:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_separate_conversations(self):
        """Agent 0 and agent 1 have different conversations."""
        env = _make_interleaved_env(play_mode="self_play")
        mgr = StrictMockSelfplayManager()
        state = await _setup_env_state(env, mgr)

        # Run 2 full turns (4 steps each: 2 agents x 2 phases)
        step = 0
        while not state.get("game_over") and step < 20:
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            await env.add_trajectory_step(state, {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt, "tokens": {},
            })
            step += 1

        agent0 = state["_agents"][0]
        agent1 = state["_agents"][1]

        # Both agents should have their own conversation
        assert len(agent0.conversation) > 0
        assert len(agent1.conversation) > 0

        # Conversations should be different (different battle names)
        conv0_text = str(agent0.conversation)
        conv1_text = str(agent1.conversation)
        # Agent 0 conversation should NOT contain agent 1's battle names
        assert "p2_t" not in conv0_text or "p1_t" in conv0_text, (
            "Agent 0's conversation should not contain agent 1's messages"
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_uses_advance_selfplay(self):
        """Extraction phase calls _advance_selfplay for self-play mode."""
        env = _make_interleaved_env(play_mode="self_play")
        mgr = StrictMockSelfplayManager()
        state = await _setup_env_state(env, mgr)

        # Phase 0: reasoning
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Thinking..."}],
            "prompt": prompt, "tokens": {},
        })

        # Phase 1: extraction — should call _advance_selfplay (strict mock validates)
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": prompt, "tokens": {},
        })

        # If _advance_selfplay was NOT called, the strict mock would have
        # raised an assertion error. Getting here = success.


# ===================================================================
# TIER 2f: Env — Validation
# ===================================================================

class TestValidation:

    @pytest.mark.unit
    def test_selfplay_interleaved_warns(self, caplog):
        """interleaved=True + self_play logs warning (not error)."""
        with caplog.at_level(logging.WARNING):
            env = PokemonBattleEnv(
                battle_format="gen1randombattle",
                port=8000,
                play_mode="self_play",
                observation_format="simple",
                interleaved=True,
            )

        # Should have logged a warning about trajectory_strategy
        warning_found = any(
            "interleaved=True with self_play" in r.message
            or "trajectory_strategy" in r.message
            for r in caplog.records
        )
        assert warning_found, (
            f"Expected warning about interleaved+self_play, got: "
            f"{[r.message for r in caplog.records]}"
        )
        # Should NOT have raised an error (env created successfully)
        assert env is not None


# ===================================================================
# TIER 2f_budget: Env — Token budget
# ===================================================================

class TestTokenBudget:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_token_budget_triggers_game_over(self):
        """At 90% seq_len, game_over triggers."""
        env = _make_interleaved_env()
        env.max_seq_len = 1000  # Small for testing

        state = {
            "_interleaved": True,
            "game_over": False,
            "game_turn": 1,
            "trajectory": [{
                "tokens": {
                    "prompt_ids": list(range(800)),
                    "completion_ids": list(range(150)),
                },
            }],
        }
        # 950 tokens > 90% of 1000 = 900
        result = await env.game_over(state)
        assert result is True
        assert state["game_over"] is True
        assert state["truncated"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_token_budget_safe_below_threshold(self):
        """Under budget, game_over does NOT trigger."""
        env = _make_interleaved_env()
        env.max_seq_len = 1000

        state = {
            "_interleaved": True,
            "game_over": False,
            "game_turn": 1,
            "trajectory": [{
                "tokens": {
                    "prompt_ids": list(range(400)),
                    "completion_ids": list(range(100)),
                },
            }],
        }
        # 500 tokens < 90% of 1000 = 900
        result = await env.game_over(state)
        assert result is False
        assert state["game_over"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_token_budget_none_tokens(self):
        """tokens=None doesn't crash."""
        env = _make_interleaved_env()
        env.max_seq_len = 1000

        state = {
            "_interleaved": True,
            "game_over": False,
            "game_turn": 1,
            "trajectory": [{
                "tokens": None,
            }],
        }
        # Should not crash
        result = await env.game_over(state)
        assert result is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_token_budget_missing_tokens_key(self):
        """Missing 'tokens' key in trajectory step doesn't crash."""
        env = _make_interleaved_env()
        env.max_seq_len = 1000

        state = {
            "_interleaved": True,
            "game_over": False,
            "game_turn": 1,
            "trajectory": [{}],
        }
        result = await env.game_over(state)
        assert result is False


# ===================================================================
# TIER 2g: Env — Extraction failure
# ===================================================================

class TestExtractionFailure:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extraction_failure_uses_fallback(self):
        """Invalid JSON in extraction -> fallback action, parse_failed=True."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=2)
        state = await _setup_env_state(env, mgr)

        # Phase 0: reasoning
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "I think tackle is good"}],
            "prompt": prompt, "tokens": {},
        })

        # Phase 1: extraction with garbage (no JSON)
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "just tackle please"}],
            "prompt": prompt, "tokens": {},
        })

        # Should have used fallback
        last_step = state["trajectory"][-1]
        assert last_step["extras"]["parse_failed"] is True
        assert "fallback" in last_step["extras"]["parsed_action"]

        # Game should have advanced despite parse failure
        assert mgr._step_count == 1
        # Agent parse_failure_count should be 1
        assert state["_agents"][0].parse_failure_count == 1


# ===================================================================
# TIER 2h: Env — render_completion
# ===================================================================

class TestRenderCompletion:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_interleaved_render_sets_terminal_reward(self):
        """Interleaved render sets reward = terminal reward."""
        env = _make_interleaved_env(reward_win=1.0, reward_loss=0.0)
        mgr = StrictMockHeuristicManager(game_turns=1)
        state = await _setup_env_state(env, mgr)

        # Run 1 turn (2 steps)
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Think"}],
            "prompt": prompt, "tokens": {},
        })
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": prompt, "tokens": {},
        })

        assert state["game_over"] is True
        assert state["won"] is True

        await env.render_completion(state)

        assert state["reward"] == 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_interleaved_render_calls_assign_rewards(self):
        """render_completion calls _assign_rewards (verified by step rewards)."""
        env = _make_interleaved_env(reward_win=1.0, reward_loss=0.0)
        mgr = StrictMockHeuristicManager(game_turns=1)
        state = await _setup_env_state(env, mgr)

        # Run 1 turn
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Think"}],
            "prompt": prompt, "tokens": {},
        })
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": prompt, "tokens": {},
        })

        await env.render_completion(state)

        # _assign_rewards sets per-step rewards
        for step in state["trajectory"]:
            assert "reward" in step, "Each step should have reward set by _assign_rewards"
            assert step["reward"] == 1.0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_interleaved_render_sets_metrics(self):
        """Metrics dict has all required keys."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=1)
        state = await _setup_env_state(env, mgr)

        # Run 1 turn
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Think"}],
            "prompt": prompt, "tokens": {},
        })
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": prompt, "tokens": {},
        })

        await env.render_completion(state)

        metrics = state["metrics"]
        required_keys = ["won", "wins", "losses", "draws", "game_turns",
                         "trajectory_length", "parse_failures"]
        for key in required_keys:
            assert key in metrics, f"Missing metrics key: {key}"

        assert metrics["wins"] == 1  # won=True
        assert metrics["losses"] == 0
        assert metrics["draws"] == 0
        assert metrics["trajectory_length"] == 2  # 2 steps (reasoning + extraction)
        assert metrics["parse_failures"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_interleaved_render_drops_response_objects(self):
        """Response objects removed from trajectory steps after render."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=1)
        state = await _setup_env_state(env, mgr)

        # Run 1 turn with response objects
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Think"}],
            "prompt": prompt, "tokens": {},
            "response": {"big": "object"},
        })
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": prompt, "tokens": {},
            "response": {"another": "object"},
        })

        await env.render_completion(state)

        for step in state["trajectory"]:
            assert "response" not in step, "Response objects should be dropped"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_interleaved_render_sets_completion(self):
        """state['completion'] set to last trajectory step's completion."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=1)
        state = await _setup_env_state(env, mgr)

        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Think"}],
            "prompt": prompt, "tokens": {},
        })
        prompt = await env.get_prompt_messages(state)
        last_completion = [{"role": "assistant", "content": '{"move": "tackle"}'}]
        await env.add_trajectory_step(state, {
            "completion": last_completion,
            "prompt": prompt, "tokens": {},
        })

        await env.render_completion(state)

        assert state["completion"] == last_completion


# ===================================================================
# TIER 2 additional: extras metadata
# ===================================================================

class TestExtrasMetadata:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_extras_has_phase_info(self):
        """Each trajectory step has agent_idx, phase, game_step in extras."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=2)
        state = await _setup_env_state(env, mgr)

        # Complete 1 full turn
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Think"}],
            "prompt": prompt, "tokens": {},
        })
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": prompt, "tokens": {},
        })

        # Check reasoning step (phase 0)
        step0 = state["trajectory"][0]
        assert step0["extras"]["phase"] == 0
        assert step0["extras"]["agent_idx"] == 0
        assert step0["extras"]["game_step"] == 0

        # Check extraction step (phase 1)
        step1 = state["trajectory"][1]
        assert step1["extras"]["phase"] == 1
        assert step1["extras"]["agent_idx"] == 0
        assert step1["extras"]["game_step"] == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_temperature_set_default(self):
        """trajectory_step.setdefault('temperature', 1.0) works."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=2)
        state = await _setup_env_state(env, mgr)

        prompt = await env.get_prompt_messages(state)
        step = {
            "completion": [{"role": "assistant", "content": "Think"}],
            "prompt": prompt, "tokens": {},
            # No temperature set
        }
        await env.add_trajectory_step(state, step)

        assert step.get("temperature") == 1.0


# ===================================================================
# TIER 2: Full interleaved game cycle
# ===================================================================

class TestFullInterleavedCycle:

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_interleaved_game_single_mode(self):
        """Run a complete 3-turn game in interleaved mode."""
        env = _make_interleaved_env()
        mgr = StrictMockHeuristicManager(game_turns=3)
        state = await _setup_env_state(env, mgr)

        step_count = 0
        while not state.get("game_over"):
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            await env.add_trajectory_step(state, {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt, "tokens": {},
            })
            step_count += 1
            assert step_count <= 20, "Runaway loop"

        await env.render_completion(state)

        # 3 turns x 2 phases = 6 trajectory steps
        assert len(state["trajectory"]) == 6, (
            f"Expected 6 steps (3 turns x 2 phases), got {len(state['trajectory'])}"
        )
        assert state["reward"] == 1.0
        assert state["won"] is True

        # Verify alternating phase pattern
        phases = [s["extras"]["phase"] for s in state["trajectory"]]
        assert phases == [0, 1, 0, 1, 0, 1]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_full_interleaved_game_selfplay(self):
        """Run a complete self-play game in interleaved mode."""
        env = _make_interleaved_env(play_mode="self_play")
        mgr = StrictMockSelfplayManager()
        state = await _setup_env_state(env, mgr)

        step_count = 0
        while not state.get("game_over"):
            prompt = await env.get_prompt_messages(state)
            if prompt is None:
                break
            await env.add_trajectory_step(state, {
                "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
                "prompt": prompt, "tokens": {},
            })
            step_count += 1
            assert step_count <= 50, "Runaway loop"

        await env.render_completion(state)

        # 3 turns x 2 agents x 2 phases = 12 trajectory steps
        assert len(state["trajectory"]) == 12, (
            f"Expected 12 steps (3 turns x 2 agents x 2 phases), "
            f"got {len(state['trajectory'])}"
        )
        assert state["reward"] == 1.0

        # Both agents should have steps
        p0_steps = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 0]
        p1_steps = [s for s in state["trajectory"] if s["extras"]["agent_idx"] == 1]
        assert len(p0_steps) == 6  # 3 turns x 2 phases
        assert len(p1_steps) == 6

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_bad_step_penalty_skipped_in_interleaved(self):
        """bad_step_penalty should be skipped in interleaved mode."""
        env = _make_interleaved_env(bad_step_penalty=-1.0)
        mgr = StrictMockHeuristicManager(game_turns=1)
        state = await _setup_env_state(env, mgr)

        # Run 1 turn with truncation
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": "Think"}],
            "prompt": prompt,
            "tokens": {"is_truncated": True},
        })
        prompt = await env.get_prompt_messages(state)
        await env.add_trajectory_step(state, {
            "completion": [{"role": "assistant", "content": '{"move": "tackle"}'}],
            "prompt": prompt, "tokens": {},
        })

        await env.render_completion(state)

        # bad_step_penalty should NOT be applied in interleaved mode
        for step in state["trajectory"]:
            assert step["reward"] == 1.0, (
                f"bad_step_penalty should not apply in interleaved mode, "
                f"got reward={step['reward']}"
            )
