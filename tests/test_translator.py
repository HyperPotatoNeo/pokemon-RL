"""Tests for Layer 3: StateTranslator.

Unit tests for parse_action (no external deps).
Integration tests for battle_to_prompt (need poke-env + Showdown).

M8 fix: poke-env imports are deferred inside helper functions and test
classes, so pytest can collect unit tests on login nodes without poke-env.
"""

import pytest

from tests.conftest import requires_poke_env, requires_pokechamp, requires_showdown


# ---- Unit tests: action parsing (mock battle objects) ----
# poke-env types deferred to avoid import-time failure on login nodes.

_poke_env_types = None


def _load_poke_env_types():
    """Lazy-load poke-env types. Returns (Move, Pokemon) or raises."""
    global _poke_env_types
    if _poke_env_types is None:
        from poke_env.environment.move import Move
        from poke_env.environment.pokemon import Pokemon
        _poke_env_types = (Move, Pokemon)
    return _poke_env_types


def make_move(move_id: str, base_power: int = 80):
    """Create a real Move object for testing (bypasses data file lookups)."""
    Move, _ = _load_poke_env_types()
    m = Move(move_id, gen=1)
    m._base_power_override = base_power
    return m


def make_pokemon(species: str, hp: float = 1.0):
    """Create a real Pokemon object for testing (bypasses data file lookups)."""
    _, Pokemon = _load_poke_env_types()
    p = Pokemon.__new__(Pokemon)
    p._species = species
    p._current_hp = int(hp * 100)
    p._max_hp = 100
    p._status = None
    p._type_1 = None
    p._type_2 = None
    return p


class MockBattle:
    """Mock battle that doesn't require poke-env at import time."""
    def __init__(self, moves=None, switches=None, format_str=None):
        self.available_moves = moves or []
        self.available_switches = switches or []
        self.turn = 1
        self.battle_tag = "test-battle"
        if format_str is not None:
            self._format = format_str
        # These are set lazily only when poke-env is available
        self._active = None
        self._opponent_active = None

    @property
    def active_pokemon(self):
        if self._active is None:
            try:
                self._active = make_pokemon("pikachu")
            except ImportError:
                return None
        return self._active

    @property
    def opponent_active_pokemon(self):
        if self._opponent_active is None:
            try:
                self._opponent_active = make_pokemon("charizard")
            except ImportError:
                return None
        return self._opponent_active


@requires_poke_env
class TestParseAction:
    @pytest.mark.unit
    def test_parse_move(self):
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(
            moves=[make_move("thunderbolt"), make_move("quickattack")]
        )
        translator = StateTranslator(format_style="simple")

        order = translator.parse_action('{"move": "thunderbolt"}', battle)
        assert order is not None
        assert "thunderbolt" in order.message

    @pytest.mark.unit
    def test_parse_move_case_insensitive(self):
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(moves=[make_move("thunderbolt")])
        translator = StateTranslator(format_style="simple")

        order = translator.parse_action('{"move": "Thunder Bolt"}', battle)
        assert order is not None
        assert "thunderbolt" in order.message

    @pytest.mark.unit
    def test_parse_switch(self):
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(switches=[make_pokemon("charizard")])
        translator = StateTranslator(format_style="simple")

        order = translator.parse_action('{"switch": "charizard"}', battle)
        assert order is not None
        assert "charizard" in order.message

    @pytest.mark.unit
    def test_parse_invalid_move(self):
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(moves=[make_move("thunderbolt")])
        translator = StateTranslator(format_style="simple")

        order = translator.parse_action('{"move": "flamethrower"}', battle)
        assert order is None  # flamethrower not available

    @pytest.mark.unit
    def test_parse_garbage(self):
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(moves=[make_move("thunderbolt")])
        translator = StateTranslator(format_style="simple")

        assert translator.parse_action("garbage text", battle) is None
        assert translator.parse_action("", battle) is None
        assert translator.parse_action("{invalid json", battle) is None

    @pytest.mark.unit
    def test_parse_json_with_reasoning(self):
        """LLM often outputs reasoning before JSON."""
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(moves=[make_move("thunderbolt")])
        translator = StateTranslator(format_style="simple")

        response = (
            "I should use an electric move since Charizard is flying type. "
            '{"move": "thunderbolt"}'
        )
        order = translator.parse_action(response, battle)
        assert order is not None
        assert "thunderbolt" in order.message

    @pytest.mark.unit
    def test_parse_last_json_wins(self):
        """When multiple JSON objects, take the last one."""
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(
            moves=[make_move("thunderbolt"), make_move("quickattack")]
        )
        translator = StateTranslator(format_style="simple")

        response = '{"thought": "hmm"} {"move": "quickattack"}'
        order = translator.parse_action(response, battle)
        assert order is not None
        assert "quickattack" in order.message

    @pytest.mark.unit
    def test_parse_dynamax(self):
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(moves=[make_move("thunderbolt")])
        translator = StateTranslator(format_style="simple")

        order = translator.parse_action('{"dynamax": "thunderbolt"}', battle)
        assert order is not None
        assert order.dynamax is True

    @pytest.mark.unit
    def test_parse_nested_json_works(self):
        """Nested JSON should be parsed correctly.

        M6 fix: The JSON extractor now handles nested objects by trying
        progressively larger substrings, not just flat regex.
        """
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(moves=[make_move("thunderbolt")])
        translator = StateTranslator(format_style="simple")

        response = '{"move": "thunderbolt", "reasoning": {"step": 1}}'
        order = translator.parse_action(response, battle)

        assert order is not None, (
            "Nested JSON should now be parsed correctly (M6 fix)."
        )
        assert "thunderbolt" in order.message

    @pytest.mark.unit
    def test_parse_extra_keys_no_interference(self):
        """JSON with extra unknown keys should still match move/switch."""
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(moves=[make_move("thunderbolt")])
        translator = StateTranslator(format_style="simple")

        response = '{"reasoning": "water is weak to electric", "move": "thunderbolt"}'
        order = translator.parse_action(response, battle)
        assert order is not None, "Extra keys should not prevent move matching"
        assert "thunderbolt" in order.message

    @pytest.mark.unit
    def test_parse_no_available_moves_returns_none(self):
        """Parsed move not in available_moves → None."""
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(moves=[])  # no available moves
        translator = StateTranslator(format_style="simple")

        order = translator.parse_action('{"move": "thunderbolt"}', battle)
        assert order is None, "Should return None when move not in available_moves"

    @pytest.mark.unit
    def test_parse_dynamax_blocked_in_gen1(self):
        """M7 fix: Dynamax should not be parsed in gen1 format."""
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(
            moves=[make_move("thunderbolt")], format_str="gen1randombattle"
        )
        translator = StateTranslator(format_style="simple")
        order = translator.parse_action('{"dynamax": "thunderbolt"}', battle)
        assert order is None, "Dynamax should be blocked in gen1"

    @pytest.mark.unit
    def test_parse_move_preferred_over_dynamax_gen1(self):
        """In gen1, regular move key should still work even if dynamax is present."""
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(
            moves=[make_move("thunderbolt")], format_str="gen1randombattle"
        )
        translator = StateTranslator(format_style="simple")
        order = translator.parse_action('{"move": "thunderbolt"}', battle)
        assert order is not None
        assert "thunderbolt" in order.message

    @pytest.mark.unit
    def test_parse_both_move_and_switch_move_wins(self):
        """When JSON has both move and switch keys, move is tried first."""
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(
            moves=[make_move("thunderbolt")],
            switches=[make_pokemon("charizard")],
        )
        translator = StateTranslator(format_style="simple")
        order = translator.parse_action(
            '{"move": "thunderbolt", "switch": "charizard"}', battle
        )
        assert order is not None
        assert "thunderbolt" in order.message



class TestExtractLastJson:
    """Tests for _extract_last_json — pure Python, no poke-env needed."""

    @pytest.mark.unit
    def test_no_json_returns_none(self):
        """_extract_last_json with no JSON returns None."""
        from pokemon_rl.translator import StateTranslator

        result = StateTranslator._extract_last_json("no json here at all")
        assert result is None

    @pytest.mark.unit
    def test_deeply_nested(self):
        """_extract_last_json handles deeply nested JSON."""
        from pokemon_rl.translator import StateTranslator

        text = '{"a": {"b": {"c": 1}}, "move": "thunderbolt"}'
        result = StateTranslator._extract_last_json(text)
        assert result is not None
        assert result["move"] == "thunderbolt"

    @pytest.mark.unit
    def test_last_json_wins(self):
        """When multiple JSON objects exist, the last one is returned."""
        from pokemon_rl.translator import StateTranslator

        text = '{"first": 1} some text {"second": 2}'
        result = StateTranslator._extract_last_json(text)
        assert result == {"second": 2}

    @pytest.mark.unit
    def test_empty_string(self):
        from pokemon_rl.translator import StateTranslator
        assert StateTranslator._extract_last_json("") is None

    @pytest.mark.unit
    def test_nested_with_reasoning(self):
        """Real-world LLM output with nested reasoning object."""
        from pokemon_rl.translator import StateTranslator

        text = 'I think the best move is {"move": "thunderbolt", "reasoning": {"step": 1, "why": "super effective"}}'
        result = StateTranslator._extract_last_json(text)
        assert result is not None
        assert result["move"] == "thunderbolt"
        assert "reasoning" in result


@requires_poke_env
class TestFallbackAction:
    @pytest.mark.unit
    def test_fallback_picks_random_action(self):
        """C1 fix: Fallback now picks a random legal action, not max power.

        Verifies the fallback returns a valid action from available moves
        (no longer always the highest base power).
        """
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(
            moves=[make_move("quickattack", 40), make_move("thunderbolt", 90)]
        )
        translator = StateTranslator(format_style="simple")

        order = translator.get_fallback_action(battle)
        assert order is not None
        # Should be one of the available moves (random, not always thunderbolt)
        assert "quickattack" in order.message or "thunderbolt" in order.message

    @pytest.mark.unit
    def test_fallback_is_not_deterministic_highest_power(self):
        """C1 fix: Verify fallback is NOT always highest power.

        Run 50 times — if fallback were max_power, it would always pick
        thunderbolt. With random, P(all same) = (1/2)^49 ≈ 2e-15.
        """
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(
            moves=[make_move("quickattack", 40), make_move("thunderbolt", 90)]
        )
        translator = StateTranslator(format_style="simple")

        actions = set()
        for _ in range(50):
            order = translator.get_fallback_action(battle)
            actions.add(order.message)

        assert len(actions) > 1, (
            f"Fallback always picks the same action: {actions}. "
            f"Should be random (C1 fix)."
        )

    @pytest.mark.unit
    def test_fallback_switch_when_no_moves(self):
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(switches=[make_pokemon("bulbasaur")])
        translator = StateTranslator(format_style="simple")

        order = translator.get_fallback_action(battle)
        assert order is not None
        assert "bulbasaur" in order.message


@requires_poke_env
class TestSimplePrompt:
    @pytest.mark.unit
    def test_simple_prompt_structure(self):
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(
            moves=[make_move("thunderbolt", 90)],
            switches=[make_pokemon("bulbasaur")],
        )
        translator = StateTranslator(format_style="simple")

        messages = translator.battle_to_prompt(battle)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "thunderbolt" in messages[1]["content"]
        assert "bulbasaur" in messages[1]["content"]
        assert "pikachu" in messages[1]["content"]


@requires_poke_env
class TestBattleToPrompt:
    @pytest.mark.unit
    def test_unknown_format_raises(self):
        """Unknown format_style must raise ValueError."""
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator(format_style="nonexistent")
        battle = MockBattle(moves=[make_move("thunderbolt")])
        with pytest.raises(ValueError, match="Unknown format_style"):
            translator.battle_to_prompt(battle)


# ---- Integration tests: full state translation ----


@requires_poke_env
@requires_showdown
class TestPokechampPromptIntegration:
    """Test state translation with real Battle objects."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_simple_prompt_real_battle(self, showdown_port):
        """Generate simple prompts during a real battle."""
        from pokemon_rl.adapter import BattleAdapter
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator(format_style="simple")
        prompts_generated = []

        def prompt_capturing_action(battle):
            from poke_env.player.battle_order import BattleOrder

            messages = translator.battle_to_prompt(battle)
            prompts_generated.append(messages)

            if battle.available_moves:
                return BattleOrder(battle.available_moves[0])
            return BattleOrder(battle.available_switches[0])

        adapter = BattleAdapter(
            port=showdown_port, battle_format="gen1randombattle"
        )
        await adapter.run_battle(action_fn=prompt_capturing_action)

        assert len(prompts_generated) > 0
        for msgs in prompts_generated:
            assert len(msgs) == 2
            assert msgs[0]["role"] == "system"
            assert msgs[1]["role"] == "user"
            assert len(msgs[1]["content"]) > 50  # Should have meaningful content

    @requires_pokechamp
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pokechamp_io_prompt(self, showdown_port):
        """Generate pokechamp_io prompts during a real battle."""
        from pokemon_rl.adapter import BattleAdapter
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator(format_style="pokechamp_io")
        prompts_generated = []
        errors = []

        def prompt_capturing_action(battle):
            from poke_env.player.battle_order import BattleOrder

            try:
                messages = translator.battle_to_prompt(battle)
                prompts_generated.append(messages)
            except Exception as e:
                errors.append(str(e))

            if battle.available_moves:
                return BattleOrder(battle.available_moves[0])
            return BattleOrder(battle.available_switches[0])

        adapter = BattleAdapter(
            port=showdown_port, battle_format="gen1randombattle"
        )
        await adapter.run_battle(action_fn=prompt_capturing_action)

        # All turns should produce prompts without errors
        if errors:
            pytest.fail(
                f"pokechamp_io had {len(errors)} errors "
                f"(first: {errors[0][:200]}). Use 'simple' format."
            )

        assert len(prompts_generated) > 0
        for msgs in prompts_generated:
            assert len(msgs) == 2
            # pokechamp prompts are typically very detailed
            assert len(msgs[1]["content"]) > 200


# ---- full_obs_cot tests ----


def _make_full_obs_pokemon(species, hp=1.0, active=False, fainted=False,
                           moves=None, ability=None, item=None,
                           status=None, boosts=None, type_1=None, type_2=None):
    """Create a Pokemon-like mock with all fields needed by full_obs_cot."""
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
    """Mock battle with all fields needed by full_obs_cot."""

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

        # Side conditions
        self.side_conditions = side_conditions or {}
        self.opponent_side_conditions = opp_side_conditions or {}

        # Weather / terrain
        self.weather = weather or {}
        self.fields = fields or {}

        # Tera / dynamax
        self.can_tera = can_tera
        self.opponent_can_tera = opponent_can_tera
        self.can_dynamax = can_dynamax

        # Active pokemon
        self._active = active
        self._opp_active = opp_active

        # Build team dicts
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
    """Helper to build a standard full_obs_cot test battle."""
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


@requires_poke_env
@requires_pokechamp
class TestFullObsCotPrompt:
    """Tests for the full_obs_cot prompt format."""

    @pytest.mark.unit
    def test_full_obs_cot_returns_two_messages(self):
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        messages = translator.battle_to_prompt(battle)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert len(messages[0]["content"]) > 0
        assert len(messages[1]["content"]) > 0

    @pytest.mark.unit
    def test_full_obs_cot_has_all_tags(self):
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt(battle)[1]["content"]

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
            assert tag in content, f"Missing tag: {tag}"

    @pytest.mark.unit
    def test_full_obs_cot_tag_ordering(self):
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt(battle)[1]["content"]

        ordered_tags = [
            "<history>",
            "<field_conditions>",
            "<opponent_remaining_pokemon>",
            "<your_team>",
            "<opponent_active_pokemon>",
            "<your_active_pokemon>",
            "<available_actions>",
            "<constraint>",
        ]
        positions = [content.index(tag) for tag in ordered_tags]
        assert positions == sorted(positions), (
            f"Tags not in expected order. Positions: "
            f"{list(zip(ordered_tags, positions))}"
        )

    @pytest.mark.unit
    def test_full_obs_cot_opponent_bench_included(self):
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt(battle)[1]["content"]

        # Extract opponent_remaining section
        start = content.index("<opponent_remaining_pokemon>")
        end = content.index("</opponent_remaining_pokemon>")
        section = content[start:end]

        assert "blastoise" in section, "Bench Pokemon blastoise should appear"
        assert "venusaur" in section, "Bench Pokemon venusaur should appear"
        assert "charizard" not in section, "Active opponent should NOT be in bench section"

    @pytest.mark.unit
    def test_full_obs_cot_opponent_bench_excludes_fainted(self):
        from pokemon_rl.translator import StateTranslator
        from poke_env.environment.pokemon_type import PokemonType

        fainted_mon = _make_full_obs_pokemon(
            "magikarp", fainted=True,
            type_1=PokemonType.WATER,
        )
        battle = _build_full_obs_battle(opp_bench=[fainted_mon])
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt(battle)[1]["content"]

        start = content.index("<opponent_remaining_pokemon>")
        end = content.index("</opponent_remaining_pokemon>")
        section = content[start:end]

        assert "magikarp" not in section, "Fainted Pokemon should NOT appear in bench"

    @pytest.mark.unit
    def test_full_obs_cot_opponent_bench_excludes_active(self):
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt(battle)[1]["content"]

        # Active opponent in its own section, not in remaining
        opp_start = content.index("<opponent_active_pokemon>")
        opp_end = content.index("</opponent_active_pokemon>")
        opp_section = content[opp_start:opp_end]
        assert "charizard" in opp_section

        bench_start = content.index("<opponent_remaining_pokemon>")
        bench_end = content.index("</opponent_remaining_pokemon>")
        bench_section = content[bench_start:bench_end]
        assert "charizard" not in bench_section

    @pytest.mark.unit
    def test_full_obs_cot_constraint_format(self):
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle()
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt(battle)[1]["content"]

        start = content.index("<constraint>")
        end = content.index("</constraint>")
        section = content[start:end]

        assert "JSON" in section
        assert "thought" in section
        assert "move" in section
        assert "switch" in section
        # No sentence limit
        assert "3 sentences" not in section

    @pytest.mark.unit
    def test_full_obs_cot_fainted_active_forces_switch(self):
        from pokemon_rl.translator import StateTranslator
        from poke_env.environment.pokemon_type import PokemonType
        from poke_env.environment.status import Status

        active = _make_full_obs_pokemon(
            "pikachu", fainted=True, active=True,
            type_1=PokemonType.ELECTRIC,
        )
        bench = _make_full_obs_pokemon(
            "bulbasaur", hp=1.0,
            type_1=PokemonType.GRASS,
        )
        opp = _make_full_obs_pokemon(
            "charizard", hp=0.9, active=True,
            type_1=PokemonType.FIRE,
        )
        battle = MockBattleFullObs(
            active=active, opp_active=opp,
            moves=[], switches=[bench],
        )
        translator = StateTranslator(format_style="full_obs_cot")
        messages = translator.battle_to_prompt(battle)

        # System prompt should mention fainted
        assert "fainted" in messages[0]["content"].lower()

        # Constraint should only allow switch
        content = messages[1]["content"]
        start = content.index("<constraint>")
        end = content.index("</constraint>")
        section = content[start:end]
        assert "switch" in section
        # Should NOT have move option in constraint when fainted
        assert '"move"' not in section

    @pytest.mark.unit
    def test_full_obs_cot_empty_opponent_bench(self):
        from pokemon_rl.translator import StateTranslator

        battle = _build_full_obs_battle(opp_bench=[])
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt(battle)[1]["content"]

        start = content.index("<opponent_remaining_pokemon>")
        end = content.index("</opponent_remaining_pokemon>")
        section = content[start:end]

        assert "no revealed" in section.lower() or "No revealed" in section

    @pytest.mark.unit
    def test_full_obs_cot_is_default(self):
        from pokemon_rl.translator import StateTranslator

        translator = StateTranslator()
        assert translator.format_style == "full_obs_cot"

    @pytest.mark.unit
    def test_full_obs_cot_no_revealed_moves(self):
        from pokemon_rl.translator import StateTranslator
        from poke_env.environment.pokemon_type import PokemonType

        # Opponent bench with no moves revealed
        opp_bench = _make_full_obs_pokemon(
            "gyarados", hp=1.0,
            type_1=PokemonType.WATER, type_2=PokemonType.FLYING,
            moves=[],  # empty — just switched in
            ability="intimidate",
        )
        battle = _build_full_obs_battle(opp_bench=[opp_bench])
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt(battle)[1]["content"]

        start = content.index("<opponent_remaining_pokemon>")
        end = content.index("</opponent_remaining_pokemon>")
        section = content[start:end]

        assert "gyarados" in section
        assert "none" in section.lower() or "No revealed" in section

    @pytest.mark.unit
    def test_full_obs_cot_opponent_status_shown(self):
        from pokemon_rl.translator import StateTranslator
        from poke_env.environment.pokemon_type import PokemonType
        from poke_env.environment.status import Status

        burned_mon = _make_full_obs_pokemon(
            "tyranitar", hp=0.6,
            type_1=PokemonType.ROCK, type_2=PokemonType.DARK,
            status=Status.BRN,
            ability="sandstream",
        )
        battle = _build_full_obs_battle(opp_bench=[burned_mon])
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt(battle)[1]["content"]

        start = content.index("<opponent_remaining_pokemon>")
        end = content.index("</opponent_remaining_pokemon>")
        section = content[start:end]

        assert "tyranitar" in section
        assert "burnt" in section.lower() or "burn" in section.lower()

    @pytest.mark.unit
    def test_full_obs_cot_weather_terrain(self):
        from pokemon_rl.translator import StateTranslator
        from poke_env.environment.weather import Weather
        from poke_env.environment.field import Field

        battle = _build_full_obs_battle(
            weather={Weather.SANDSTORM: 1},
            fields={Field.ELECTRIC_TERRAIN: 1},
        )
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt(battle)[1]["content"]

        start = content.index("<field_conditions>")
        end = content.index("</field_conditions>")
        section = content[start:end]

        assert "sandstorm" in section.lower()
        assert "electric" in section.lower()

    @pytest.mark.unit
    def test_full_obs_cot_boost_stages_shown(self):
        from pokemon_rl.translator import StateTranslator
        from poke_env.environment.pokemon_type import PokemonType

        opp_active = _make_full_obs_pokemon(
            "charizard", hp=0.92, active=True,
            type_1=PokemonType.FIRE, type_2=PokemonType.FLYING,
            boosts={"atk": 2, "def": 0, "spa": 0, "spd": 0, "spe": -1, "accuracy": 0, "evasion": 0},
            ability="blaze",
        )
        battle = _build_full_obs_battle(opp_active=opp_active)
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt(battle)[1]["content"]

        start = content.index("<opponent_active_pokemon>")
        end = content.index("</opponent_active_pokemon>")
        section = content[start:end]

        assert "+2" in section, "Should show +2 boost stage for Atk"
        assert "-1" in section, "Should show -1 boost stage for Spe"

    @pytest.mark.unit
    def test_full_obs_cot_tera_action_shown(self):
        from pokemon_rl.translator import StateTranslator
        from poke_env.environment.pokemon_type import PokemonType

        battle = _build_full_obs_battle(
            can_tera=PokemonType.FAIRY,
            opponent_can_tera=True,
        )
        translator = StateTranslator(format_style="full_obs_cot")
        content = translator.battle_to_prompt(battle)[1]["content"]

        # Constraint should include terastallize option
        c_start = content.index("<constraint>")
        c_end = content.index("</constraint>")
        constraint = content[c_start:c_end]
        assert "terastallize" in constraint

        # Field conditions should show tera info
        fc_start = content.index("<field_conditions>")
        fc_end = content.index("</field_conditions>")
        fc = content[fc_start:fc_end]
        assert "fairy" in fc.lower() or "Fairy" in fc
        assert "terastallize" in fc.lower()
