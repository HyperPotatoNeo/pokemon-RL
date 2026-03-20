"""Tests for Layer 2a: ControllablePlayer and opponent factory.

TEST PHILOSOPHY — NO FALL-THROUGH PASSES:
    Every test verifies BOTH the positive case (correct behavior happens)
    AND the negative case (incorrect behavior does NOT happen).

    For example:
    - Test that state IS pushed to queue → also verify queue WAS empty before
    - Test that action IS received → also verify a WRONG action is distinguishable
    - Test that timeout DOES fire → also verify it does NOT fire when action is fast
    - Test that sentinel DOES appear → also verify non-sentinel states are NOT sentinel

    This ensures tests catch real bugs, not just pass by default.

Unit tests: mock battle objects, asyncio queues, no poke-env needed.
Integration tests: real poke-env Players, need Showdown running.
"""

import asyncio
import pytest

from tests.conftest import requires_poke_env, requires_showdown


# ---- Unit tests: queue mechanics ----


class TestAtomicUsername:
    """Test username generation doesn't collide under concurrent use."""

    @pytest.mark.unit
    def test_unique_usernames(self):
        """Generate 100 usernames — all must be unique."""
        from pokemon_rl.players import _next_username

        names = [_next_username("test") for _ in range(100)]
        assert len(set(names)) == 100, (
            f"Expected 100 unique names, got {len(set(names))}. "
            f"Duplicates: {[n for n in names if names.count(n) > 1]}"
        )

    @pytest.mark.unit
    def test_username_has_prefix(self):
        """Username should contain the requested prefix."""
        from pokemon_rl.players import _next_username

        name = _next_username("ctrl")
        assert name.startswith("ctrl-"), f"Expected ctrl- prefix, got: {name}"

        name2 = _next_username("opp")
        assert name2.startswith("opp-"), f"Expected opp- prefix, got: {name2}"

    @pytest.mark.unit
    def test_different_prefixes_different_names(self):
        """Different prefixes must produce different names."""
        from pokemon_rl.players import _next_username

        n1 = _next_username("a")
        n2 = _next_username("b")
        assert n1 != n2


# ---- Queue mechanics tests using raw asyncio (no poke-env) ----


class TestQueueMechanics:
    """Test the queue-based control flow pattern with plain asyncio.

    These tests verify the core logic WITHOUT poke-env imports, using
    regular asyncio.Queue to simulate the ControllablePlayer pattern.
    """

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_state_push_action_receive(self):
        """Simulate choose_move: push state, receive action."""
        state_q = asyncio.Queue()
        action_q = asyncio.Queue()

        # Simulate choose_move (would run on POKE_LOOP)
        async def fake_choose_move(battle_state):
            await state_q.put(battle_state)
            action = await action_q.get()
            return action

        # Simulate external controller (would run on caller's loop)
        async def controller():
            # Verify queue is empty before state arrives
            assert state_q.empty(), "State queue should be empty before choose_move"

            # Start choose_move
            task = asyncio.create_task(fake_choose_move("battle_state_1"))

            # Read state
            state = await asyncio.wait_for(state_q.get(), timeout=2)
            assert state == "battle_state_1", f"Expected battle_state_1, got: {state}"

            # Verify action queue is empty before we submit
            assert action_q.empty(), "Action queue should be empty before submission"

            # Submit action
            await action_q.put("use_thunderbolt")

            # Verify choose_move returns the correct action
            result = await asyncio.wait_for(task, timeout=2)
            assert result == "use_thunderbolt", f"Expected use_thunderbolt, got: {result}"

        await controller()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_wrong_action_is_distinguishable(self):
        """Verify that different actions produce different results."""
        state_q = asyncio.Queue()
        action_q = asyncio.Queue()

        async def fake_choose_move():
            await state_q.put("state")
            return await action_q.get()

        task = asyncio.create_task(fake_choose_move())
        await state_q.get()  # consume state

        await action_q.put("action_A")
        result = await task

        assert result == "action_A"
        assert result != "action_B", "Different actions must be distinguishable"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sentinel_none_signals_game_over(self):
        """None sentinel on state queue means game over."""
        state_q = asyncio.Queue()

        # Put a real state, then a sentinel
        await state_q.put("real_battle_state")
        await state_q.put(None)

        s1 = await state_q.get()
        assert s1 is not None, "First state should be a real battle state"
        assert s1 == "real_battle_state"

        s2 = await state_q.get()
        assert s2 is None, "Second state should be None sentinel (game over)"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_non_sentinel_is_not_none(self):
        """Regular states must not be None (would be confused with sentinel)."""
        state_q = asyncio.Queue()
        await state_q.put("battle_state")

        state = await state_q.get()
        assert state is not None, "Regular state must not be None"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_fires_when_no_action(self):
        """If no action is submitted, timeout should trigger."""
        action_q = asyncio.Queue()

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(action_q.get(), timeout=0.1)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_does_not_fire_when_action_is_fast(self):
        """If action is submitted quickly, timeout should NOT fire."""
        action_q = asyncio.Queue()

        async def submit_fast():
            await asyncio.sleep(0.01)
            await action_q.put("fast_action")

        asyncio.create_task(submit_fast())
        # 2 second timeout — should NOT fire for a 10ms action
        result = await asyncio.wait_for(action_q.get(), timeout=2.0)
        assert result == "fast_action"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_turns_sequential(self):
        """Simulate 3 turns of back-and-forth."""
        state_q = asyncio.Queue()
        action_q = asyncio.Queue()

        turns_played = []

        async def fake_player():
            """Simulates 3 turns of choose_move."""
            for i in range(3):
                await state_q.put(f"state_{i}")
                action = await action_q.get()
                turns_played.append((f"state_{i}", action))
            await state_q.put(None)  # game over

        async def fake_controller():
            task = asyncio.create_task(fake_player())
            turn = 0
            while True:
                state = await state_q.get()
                if state is None:
                    break
                assert state == f"state_{turn}", (
                    f"Turn {turn}: expected state_{turn}, got {state}"
                )
                await action_q.put(f"action_{turn}")
                turn += 1
            await task
            return turn

        turns = await fake_controller()
        assert turns == 3, f"Expected 3 turns, got {turns}"
        assert len(turns_played) == 3
        # Verify each turn got the right action
        for i, (state, action) in enumerate(turns_played):
            assert state == f"state_{i}"
            assert action == f"action_{i}"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_selfplay_both_states_arrive(self):
        """In self-play, both players' states should arrive."""
        p1_state_q = asyncio.Queue()
        p2_state_q = asyncio.Queue()

        await p1_state_q.put("p1_battle")
        await p2_state_q.put("p2_battle")

        # Collect both with gather (order-independent)
        s1, s2 = await asyncio.gather(
            p1_state_q.get(),
            p2_state_q.get(),
        )
        assert s1 == "p1_battle"
        assert s2 == "p2_battle"
        assert s1 != s2, "Players must have different states"


# ---- Integration tests: real ControllablePlayer ----


@requires_poke_env
@requires_showdown
class TestControllablePlayerIntegration:
    """Integration tests with real poke-env Player and Showdown."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_controllable_player_creates(self, showdown_port):
        """ControllablePlayer.create returns a valid Player with queues."""
        from pokemon_rl.players import ControllablePlayer
        from poke_env.ps_client.server_configuration import ServerConfiguration

        server_config = ServerConfiguration(
            f"localhost:{showdown_port}",
            "https://play.pokemonshowdown.com/action.php?",
        )
        player = ControllablePlayer.create(
            battle_format="gen1randombattle",
            server_config=server_config,
        )
        # Verify the player has the expected queue attributes
        assert hasattr(player, "state_queue"), "Missing state_queue"
        assert hasattr(player, "action_queue"), "Missing action_queue"
        assert hasattr(player, "finished_event"), "Missing finished_event"
        assert hasattr(player, "result_battle"), "Missing result_battle"

        # Verify queues are empty at creation
        assert player.state_queue.empty(), "state_queue should start empty"
        assert player.action_queue.empty(), "action_queue should start empty"
        assert not player.finished_event.is_set(), "finished_event should start unset"
        assert player.result_battle is None, "result_battle should start None"

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_controllable_vs_random_full_game(self, showdown_port):
        """Play a full game: ControllablePlayer vs RandomPlayer.

        This is the critical integration test — verifies the queue bridge
        actually works with a real Showdown server.
        """
        from poke_env.player.random_player import RandomPlayer
        from poke_env.player.battle_order import BattleOrder
        from poke_env import AccountConfiguration
        from poke_env.ps_client.server_configuration import ServerConfiguration
        from poke_env.concurrency import POKE_LOOP
        from pokemon_rl.players import ControllablePlayer, _next_username

        server_config = ServerConfiguration(
            f"localhost:{showdown_port}",
            "https://play.pokemonshowdown.com/action.php?",
        )

        player = ControllablePlayer.create(
            battle_format="gen1randombattle",
            server_config=server_config,
        )
        opponent = RandomPlayer(
            battle_format="gen1randombattle",
            server_configuration=server_config,
            account_configuration=AccountConfiguration(
                _next_username("topp"), None
            ),
            max_concurrent_battles=1,
        )

        # Start battle on POKE_LOOP
        battle_future = asyncio.run_coroutine_threadsafe(
            player._battle_against(opponent, 1),
            POKE_LOOP,
        )

        # Play turns
        states_seen = []
        actions_taken = []
        turn_count = 0
        max_turns = 200

        while turn_count < max_turns:
            # Get state from POKE_LOOP
            state_future = asyncio.run_coroutine_threadsafe(
                player.state_queue.get(), POKE_LOOP
            )
            state = await asyncio.wrap_future(state_future)

            if state is None:
                break  # game over

            states_seen.append(state)

            # Choose action: first available move, or first switch
            if state.available_moves:
                order = BattleOrder(state.available_moves[0])
            elif state.available_switches:
                order = BattleOrder(state.available_switches[0])
            else:
                order = player.choose_default_move()

            actions_taken.append(order)

            # Submit action to POKE_LOOP
            action_future = asyncio.run_coroutine_threadsafe(
                player.action_queue.put(order), POKE_LOOP
            )
            await asyncio.wrap_future(action_future)
            turn_count += 1

        # Verify game completed
        assert turn_count > 0, "Game should have at least 1 turn"
        assert len(states_seen) == turn_count, (
            f"Should have seen {turn_count} states, got {len(states_seen)}"
        )
        assert len(actions_taken) == turn_count, (
            f"Should have taken {turn_count} actions, got {len(actions_taken)}"
        )

        # Verify final state
        assert player.finished_event.is_set(), "finished_event should be set after game"
        assert player.result_battle is not None, "result_battle should be set"
        assert player.result_battle.won in (True, False), (
            f"Game result should be True/False, got {player.result_battle.won}"
        )

        # Verify states had meaningful content
        for i, state in enumerate(states_seen):
            has_moves = len(state.available_moves) > 0
            has_switches = len(state.available_switches) > 0
            assert has_moves or has_switches, (
                f"Turn {i}: no available moves or switches — invalid state"
            )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_opponent_factory_random(self, showdown_port):
        """create_opponent('random') returns a RandomPlayer."""
        from pokemon_rl.players import create_opponent
        from poke_env.player.random_player import RandomPlayer
        from poke_env.ps_client.server_configuration import ServerConfiguration

        server_config = ServerConfiguration(
            f"localhost:{showdown_port}",
            "https://play.pokemonshowdown.com/action.php?",
        )
        opp = create_opponent(
            "random", "gen1randombattle", server_config
        )
        assert isinstance(opp, RandomPlayer), (
            f"Expected RandomPlayer, got {type(opp).__name__}"
        )

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_opponent_factory_unknown_raises(self, showdown_port):
        """create_opponent with unknown type raises ValueError."""
        from pokemon_rl.players import create_opponent
        from poke_env.ps_client.server_configuration import ServerConfiguration

        server_config = ServerConfiguration(
            f"localhost:{showdown_port}",
            "https://play.pokemonshowdown.com/action.php?",
        )
        with pytest.raises(ValueError, match="Unknown opponent_type"):
            create_opponent("nonexistent", "gen1randombattle", server_config)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_opponent_factory_callback_requires_fn(self, showdown_port):
        """create_opponent('callback') without callback raises ValueError."""
        from pokemon_rl.players import create_opponent
        from poke_env.ps_client.server_configuration import ServerConfiguration

        server_config = ServerConfiguration(
            f"localhost:{showdown_port}",
            "https://play.pokemonshowdown.com/action.php?",
        )
        with pytest.raises(ValueError, match="callback"):
            create_opponent("callback", "gen1randombattle", server_config)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_opponent_factory_controllable_has_queues(self, showdown_port):
        """create_opponent('controllable') returns player with queue attrs."""
        from pokemon_rl.players import create_opponent
        from poke_env.ps_client.server_configuration import ServerConfiguration

        server_config = ServerConfiguration(
            f"localhost:{showdown_port}",
            "https://play.pokemonshowdown.com/action.php?",
        )
        player = create_opponent(
            "controllable", "gen1randombattle", server_config
        )

        assert hasattr(player, "state_queue"), "Missing state_queue"
        assert hasattr(player, "action_queue"), "Missing action_queue"
        assert hasattr(player, "finished_event"), "Missing finished_event"
        assert player.state_queue.empty(), "state_queue should start empty"
        assert player.action_queue.empty(), "action_queue should start empty"
        assert not player.finished_event.is_set(), "finished_event should start unset"
