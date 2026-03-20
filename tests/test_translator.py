"""Tests for Layer 3: StateTranslator.

Unit tests for parse_action (no external deps).
Integration tests for battle_to_prompt (need poke-env + Showdown).
"""

import pytest

from tests.conftest import requires_poke_env, requires_pokechamp, requires_showdown


# ---- Unit tests: action parsing (mock battle objects) ----
# Use real poke-env types so BattleOrder.message isinstance checks pass.

from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon


def make_move(move_id: str, base_power: int = 80) -> Move:
    """Create a real Move object for testing (bypasses data file lookups)."""
    m = Move(move_id, gen=1)
    m._base_power_override = base_power
    return m


def make_pokemon(species: str, hp: float = 1.0) -> Pokemon:
    """Create a real Pokemon object for testing (bypasses data file lookups)."""
    p = Pokemon.__new__(Pokemon)
    p._species = species
    p._current_hp = int(hp * 100)
    p._max_hp = 100
    p._status = None
    p._type_1 = None
    p._type_2 = None
    return p


class MockBattle:
    def __init__(self, moves=None, switches=None):
        self.available_moves = moves or []
        self.available_switches = switches or []
        self.active_pokemon = make_pokemon("pikachu")
        self.opponent_active_pokemon = make_pokemon("charizard")
        self.turn = 1
        self.battle_tag = "test-battle"


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
    def test_parse_nested_json_matches_inner(self):
        """Nested JSON: regex matches inner object, not outer.

        Documents a known limitation of the flat JSON regex r"\\{[^{}]*\\}".
        If the LLM nests objects, the inner one is matched instead.
        """
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(moves=[make_move("thunderbolt")])
        translator = StateTranslator(format_style="simple")

        response = '{"move": "thunderbolt", "reasoning": {"step": 1}}'
        order = translator.parse_action(response, battle)

        # The regex can't match the outer object (contains inner braces).
        # It matches {"step": 1} which has no "move" key → returns None.
        assert order is None, (
            "Nested JSON should fail because the flat regex matches the "
            "inner object. If this passes, the regex was improved."
        )

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


class TestFallbackAction:
    @pytest.mark.unit
    def test_fallback_picks_highest_power(self):
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(
            moves=[make_move("quickattack", 40), make_move("thunderbolt", 90)]
        )
        translator = StateTranslator(format_style="simple")

        order = translator.get_fallback_action(battle)
        assert order is not None
        assert "thunderbolt" in order.message

    @pytest.mark.unit
    def test_fallback_switch_when_no_moves(self):
        from pokemon_rl.translator import StateTranslator

        battle = MockBattle(switches=[make_pokemon("bulbasaur")])
        translator = StateTranslator(format_style="simple")

        order = translator.get_fallback_action(battle)
        assert order is not None
        assert "bulbasaur" in order.message


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
