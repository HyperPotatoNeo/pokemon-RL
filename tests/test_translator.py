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
