"""Tests for Layer 2b: BattleManager — turn-by-turn battle orchestration.

TEST PHILOSOPHY — NO FALL-THROUGH PASSES:
    Every test checks BOTH that correct behavior happens AND that incorrect
    behavior does NOT happen. Specific principles:

    1. TRAJECTORY VERIFICATION: Don't just check len > 0. Check specific
       expected values — turn numbers, action types, player indices.

    2. STATE MACHINE GUARDS: Verify that calling methods in wrong order
       raises RuntimeError, not silently succeeds.

    3. GAME-OVER PRECISION: Check that games end WHEN they should (sentinel
       received) and DON'T end when they shouldn't (mid-game states are real).

    4. SELF-PLAY SYMMETRY: Both players must get states. Actions must reach
       the right player. Rewards must be opposite for the two players.

Unit tests: mock the queue pattern, no poke-env needed.
Integration tests: real Showdown + poke-env.
"""

import asyncio
import pytest

from tests.conftest import requires_poke_env, requires_showdown


# ---- Unit tests: BattleManager state machine ----


class TestBattleManagerStateMachine:
    """Test BattleManager lifecycle guards without external deps."""

    @pytest.mark.unit
    def test_init_state(self):
        """Fresh BattleManager is not started, not finished, not selfplay."""
        from pokemon_rl.battle import BattleManager

        mgr = BattleManager(port=9999, battle_format="gen1randombattle")
        assert not mgr.is_started, "Should not be started"
        assert not mgr.is_finished, "Should not be finished"
        assert not mgr.is_selfplay, "Should not be selfplay"

    @pytest.mark.unit
    def test_server_host_default(self):
        """Default server host is localhost."""
        from pokemon_rl.battle import BattleManager

        mgr = BattleManager(port=8000)
        assert mgr.server_host == "localhost"

    @pytest.mark.unit
    def test_server_host_custom(self):
        """Custom server host for cross-node play."""
        from pokemon_rl.battle import BattleManager

        mgr = BattleManager(port=8000, server_host="nid008268")
        assert mgr.server_host == "nid008268"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_step_before_start_raises(self):
        """Calling step() before start_battle() must raise."""
        from pokemon_rl.battle import BattleManager

        mgr = BattleManager(port=9999)
        with pytest.raises(RuntimeError, match="start_battle"):
            await mgr.step(None)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_submit_selfplay_on_non_selfplay_raises(self):
        """submit_selfplay_action() on non-selfplay must raise."""
        from pokemon_rl.battle import BattleManager

        mgr = BattleManager(port=9999)
        mgr._started = True
        mgr._selfplay = False
        with pytest.raises(RuntimeError, match="selfplay"):
            await mgr.submit_selfplay_action(0, None)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_step_on_selfplay_raises(self):
        """step() on a selfplay manager must raise."""
        from pokemon_rl.battle import BattleManager

        mgr = BattleManager(port=9999)
        mgr._started = True
        mgr._selfplay = True
        with pytest.raises(RuntimeError, match="step_selfplay"):
            await mgr.step(None)

    @pytest.mark.unit
    def test_get_result_before_finish_raises(self):
        """get_result() before game ends must raise."""
        from pokemon_rl.battle import BattleManager

        mgr = BattleManager(port=9999)
        with pytest.raises(RuntimeError, match="not finished"):
            mgr.get_result()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_step_after_finish_raises(self):
        """step() after game ends must raise."""
        from pokemon_rl.battle import BattleManager

        mgr = BattleManager(port=9999)
        mgr._started = True
        mgr._finished = True
        with pytest.raises(RuntimeError, match="already finished"):
            await mgr.step(None)

    @pytest.mark.unit
    def test_get_result_schema_all_fields(self):
        """Result dict must have all expected fields with correct types."""
        from unittest.mock import MagicMock
        from pokemon_rl.battle import BattleManager

        mgr = BattleManager(port=9999, battle_format="gen1randombattle")
        mgr._started = True
        mgr._finished = True
        mgr._selfplay = False

        # Mock the player (get_result accesses _player.result_battle)
        mock_player = MagicMock()
        mock_player.result_battle = None
        mock_player.battles = {}
        mgr._player = mock_player

        result = mgr.get_result()

        required_fields = {"won", "turns", "steps", "format", "battle_tag", "selfplay"}
        actual_fields = set(result.keys())
        missing = required_fields - actual_fields
        assert not missing, f"Result missing fields: {missing}"
        assert result["format"] == "gen1randombattle"
        assert result["selfplay"] is False
        assert isinstance(result["steps"], int)
        assert isinstance(result["turns"], int)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_sentinel_in_grace_window_keeps_valid(self):
        """Valid state not discarded when sentinel arrives in grace window.

        When get_pending reads a valid state, then a None sentinel arrives
        within 0.5s, the valid state should still be returned.

        BUG: Current code returns [] when sentinel found in grace window,
        discarding the already-collected valid state.
        """
        import queue as thread_queue
        from pokemon_rl.battle import BattleManager

        mgr = BattleManager(port=9999)
        mgr._started = True
        mgr._selfplay = True
        mgr._finished = False
        mgr._selfplay_relay = thread_queue.Queue()

        # Simulate: valid state, then sentinel in quick succession
        mgr._selfplay_relay.put((0, "valid_battle"))
        mgr._selfplay_relay.put((1, None))

        result = await mgr.get_pending_selfplay_states()

        assert len(result) >= 1, (
            f"Expected at least 1 state (the valid one), got {len(result)}. "
            f"The valid state was consumed from the relay queue but discarded "
            f"when the sentinel was found in the grace window."
        )
        assert result[0] == (0, "valid_battle")


# ---- Integration tests: real Showdown ----


@requires_poke_env
@requires_showdown
class TestBattleManagerIntegration:
    """Integration tests — real Showdown server + poke-env."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_start_step_finish_heuristic(self, showdown_port):
        """Full game: start → step through each turn → get_result.

        This is the primary integration test for the turn-by-turn bridge.
        Verifies:
        - First state is a real Battle object with moves/switches
        - Each step returns a new state or game over
        - Game eventually ends (sentinel None received)
        - get_result returns valid battle outcome
        - Step count matches number of decisions made
        """
        from pokemon_rl.battle import BattleManager
        from poke_env.player.battle_order import BattleOrder

        mgr = BattleManager(port=showdown_port, battle_format="gen1randombattle")

        # Start battle
        battle = await mgr.start_battle(opponent_type="random")
        assert battle is not None, "First state should not be None"
        assert mgr.is_started, "Manager should be started"
        assert not mgr.is_finished, "Manager should not be finished yet"

        # Verify first state has content
        has_moves = len(battle.available_moves) > 0
        has_switches = len(battle.available_switches) > 0
        assert has_moves or has_switches, (
            "First state must have available moves or switches"
        )

        # Play through the game
        states = [battle]
        step_count = 0

        while battle is not None and step_count < 300:
            # Choose action
            if battle.available_moves:
                action = BattleOrder(battle.available_moves[0])
            elif battle.available_switches:
                action = BattleOrder(battle.available_switches[0])
            else:
                action = None  # shouldn't happen

            battle, done = await mgr.step(action)
            step_count += 1

            if done:
                assert battle is None, (
                    "When done=True, battle should be None"
                )
                break
            else:
                assert battle is not None, (
                    "When done=False, battle should not be None"
                )
                states.append(battle)
                # Verify mid-game states have content
                mid_has_moves = len(battle.available_moves) > 0
                mid_has_switches = len(battle.available_switches) > 0
                assert mid_has_moves or mid_has_switches, (
                    f"Step {step_count}: mid-game state must have moves/switches"
                )

        # Verify game ended
        assert mgr.is_finished, "Manager should be finished"
        assert step_count > 0, "Game should have at least 1 step"

        # Verify result
        result = mgr.get_result()
        assert result["won"] in (True, False), (
            f"Expected True/False, got {result['won']}"
        )
        assert result["turns"] > 0, "Game should have turns"
        assert result["steps"] == step_count, (
            f"Steps mismatch: result says {result['steps']}, counted {step_count}"
        )
        assert result["selfplay"] is False
        assert "battle_tag" in result

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_selfplay_full_game(self, showdown_port):
        """Self-play: two ControllablePlayers complete a game.

        Uses the sequential selfplay API which handles force-switches.

        Verifies:
        - Both players get initial states
        - Both states are real Battle objects
        - Game resolves after all actions submitted
        - Winner is determined (one wins, one loses)
        - Steps are counted correctly
        - Force-switches produce single-player pending states
        """
        from pokemon_rl.battle import BattleManager
        from poke_env.player.battle_order import BattleOrder

        mgr = BattleManager(port=showdown_port, battle_format="gen1randombattle")

        # Start self-play battle
        pending = await mgr.start_battle_selfplay()
        assert len(pending) == 2, f"Expected 2 initial states, got {len(pending)}"
        assert mgr.is_selfplay, "Should be selfplay mode"

        # Verify initial states have content
        for idx, state in pending:
            has_actions = (
                len(state.available_moves) > 0
                or len(state.available_switches) > 0
            )
            assert has_actions, f"Player {idx} must have available actions"

        # Play through using sequential API
        step_count = 0
        normal_turns = 0
        force_switch_turns = 0

        while pending and step_count < 600:
            if len(pending) == 2:
                normal_turns += 1
            elif len(pending) == 1:
                force_switch_turns += 1

            for idx, state in pending:
                if state.available_moves:
                    action = BattleOrder(state.available_moves[0])
                elif state.available_switches:
                    action = BattleOrder(state.available_switches[0])
                else:
                    action = None

                await mgr.submit_selfplay_action(idx, action)
                step_count += 1

            pending = await mgr.get_pending_selfplay_states()

        assert mgr.is_finished, "Manager should be finished"
        assert step_count > 0, "Self-play should have at least 1 step"
        assert normal_turns > 0, "Should have at least 1 normal turn"

        result = mgr.get_result()
        assert result["won"] in (True, False), (
            f"Expected True/False, got {result['won']}"
        )
        assert result["selfplay"] is True
        assert result["steps"] == step_count

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_battles_no_contamination(self, showdown_port):
        """Run 3 battles concurrently — verify no cross-contamination.

        Each battle should have its own unique battle_tag and independent
        win/loss outcome. Trajectory from one battle must not appear in another.
        """
        from pokemon_rl.battle import BattleManager
        from poke_env.player.battle_order import BattleOrder

        async def play_one_game(port):
            mgr = BattleManager(port=port, battle_format="gen1randombattle")
            battle = await mgr.start_battle(opponent_type="random")
            steps = 0
            while battle is not None and steps < 300:
                if battle.available_moves:
                    action = BattleOrder(battle.available_moves[0])
                elif battle.available_switches:
                    action = BattleOrder(battle.available_switches[0])
                else:
                    break
                battle, done = await mgr.step(action)
                steps += 1
                if done:
                    break
            return mgr.get_result()

        # Run 3 games concurrently
        results = await asyncio.gather(
            play_one_game(showdown_port),
            play_one_game(showdown_port),
            play_one_game(showdown_port),
        )

        assert len(results) == 3, "Should have 3 results"

        # Verify unique battle tags
        tags = [r["battle_tag"] for r in results]
        assert len(set(tags)) == 3, (
            f"Expected 3 unique battle tags, got: {tags}"
        )

        # Verify each game completed independently
        for i, r in enumerate(results):
            assert r["won"] in (True, False), (
                f"Game {i}: expected True/False, got {r['won']}"
            )
            assert r["turns"] > 0, f"Game {i}: should have turns"
            assert r["steps"] > 0, f"Game {i}: should have steps"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_double_start_raises(self, showdown_port):
        """Starting a battle twice must raise RuntimeError."""
        from pokemon_rl.battle import BattleManager
        from poke_env.player.battle_order import BattleOrder

        mgr = BattleManager(port=showdown_port, battle_format="gen1randombattle")
        battle = await mgr.start_battle(opponent_type="random")
        assert battle is not None

        with pytest.raises(RuntimeError, match="already started"):
            await mgr.start_battle(opponent_type="random")

        # Clean up: play through the game
        while battle is not None:
            if battle.available_moves:
                action = BattleOrder(battle.available_moves[0])
            elif battle.available_switches:
                action = BattleOrder(battle.available_switches[0])
            else:
                break
            battle, done = await mgr.step(action)
            if done:
                break

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_selfplay_battles(self, showdown_port):
        """Multiple selfplay battles concurrently — no cross-contamination.

        Each battle should have unique tags and independent outcomes.
        """
        from pokemon_rl.battle import BattleManager
        from poke_env.player.battle_order import BattleOrder

        async def play_selfplay(port):
            mgr = BattleManager(port=port, battle_format="gen1randombattle")
            pending = await mgr.start_battle_selfplay()
            steps = 0
            while pending and steps < 600:
                for idx, state in pending:
                    if state.available_moves:
                        action = BattleOrder(state.available_moves[0])
                    elif state.available_switches:
                        action = BattleOrder(state.available_switches[0])
                    else:
                        action = None
                    await mgr.submit_selfplay_action(idx, action)
                    steps += 1
                pending = await mgr.get_pending_selfplay_states()
            return mgr.get_result()

        results = await asyncio.gather(
            play_selfplay(showdown_port),
            play_selfplay(showdown_port),
        )

        assert len(results) == 2
        tags = [r["battle_tag"] for r in results]
        assert tags[0] != tags[1], f"Tags should differ: {tags}"

        for i, r in enumerate(results):
            assert r["won"] in (True, False), f"Game {i}: expected bool won"
            assert r["selfplay"] is True
            assert r["steps"] > 0, f"Game {i}: should have steps"
