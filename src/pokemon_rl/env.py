"""Layer 4: Pokemon battle environment — MultiTurnEnv interface.

Implements the 4-hook interface from prime-rl's verifiers framework:
    setup_state          — Start a new battle
    get_prompt_messages  — Return current player's state as LLM prompt
    add_trajectory_step  — Parse action, advance game
    render_completion    — Assign terminal rewards

Supports two control modes:
    "full_battle" — Runs complete battle via BattleAdapter (callback-driven)
    "turn_by_turn" — Step-by-step via BattleManager (imperative control)

And two opponent modes:
    "heuristic" — Opponent auto-responds (random, max_damage, etc.)
    "self_play" — Both sides controlled by caller (LLM vs itself)

Usage (standalone testing):
    env = PokemonBattleEnv(adapter=adapter, translator=translator)
    result = await env.run_standalone()

Usage (turn-by-turn testing):
    env = PokemonBattleEnv(translator=translator, control_mode="turn_by_turn",
                           port=8000, battle_format="gen1randombattle")
    result = await env.run_turn_by_turn()

Usage (future verifiers integration):
    # PokemonBattleEnv inherits from vf.MultiTurnEnv
    # Orchestrator calls the 4 hooks automatically
"""

from __future__ import annotations

from typing import Any, Callable


def _passthrough_reward(state: dict, **kwargs) -> float:
    """Passthrough rubric function for verifiers integration.

    Returns pre-computed reward from render_completion.
    Prevents verifiers' rubric from overwriting our rewards.
    """
    return state.get("reward", 0.0)


class PokemonBattleEnv:
    """Pokemon Showdown RL environment.

    Each trajectory step = one game turn decision.

    For heuristic opponent mode:
        - Each turn generates 1 trajectory step (training player only)
        - Opponent acts automatically via poke-env

    For self-play mode:
        - Each game turn generates 2 trajectory steps (one per player)
        - Both actions collected before the turn resolves
        - state["current_player"] tracks which player's turn it is

    Args:
        adapter: BattleAdapter instance (for full_battle mode). Can be None
            if using turn_by_turn mode.
        translator: StateTranslator instance for prompt/action conversion
        control_mode: "full_battle" or "turn_by_turn"
        opponent_mode: "heuristic" or "self_play"
        opponent_type: Specific opponent (for heuristic): "random", "max_damage"
        max_game_turns: Maximum game turns before truncation
        port: Showdown server port (for turn_by_turn mode)
        battle_format: Pokemon format string (for turn_by_turn mode)
        server_host: Showdown host (for cross-node play)
        reward_win: Terminal reward for wins (default 1.0)
        reward_loss: Terminal reward for losses (default 0.0)
        reward_draw: Terminal reward for draws/truncations/crashes (default 0.0)
        step_reward_fn: Optional per-step reward callback.
            Signature: (battle_before, battle_after, action, player_idx) -> float.
            Called after each game advancement. battle_after is None on game-over
            or in self-play (where the turn hasn't resolved yet). Result stored
            as step["step_reward"], separate from terminal step["reward"].
    """

    def __init__(
        self,
        adapter: Any = None,
        translator: Any = None,
        control_mode: str = "full_battle",
        opponent_mode: str = "heuristic",
        opponent_type: str = "random",
        max_game_turns: int = 200,
        port: int = 8000,
        battle_format: str = "gen1randombattle",
        server_host: str = "localhost",
        reward_win: float = 1.0,
        reward_loss: float = 0.0,
        reward_draw: float = 0.0,
        step_reward_fn: Callable | None = None,
    ):
        self.adapter = adapter
        self.translator = translator
        self.control_mode = control_mode
        self.opponent_mode = opponent_mode
        self.opponent_type = opponent_type
        self.max_game_turns = max_game_turns
        self.port = port
        self.battle_format = battle_format
        self.server_host = server_host
        self.reward_win = reward_win
        self.reward_loss = reward_loss
        self.reward_draw = reward_draw
        self.step_reward_fn = step_reward_fn

        if control_mode not in ("full_battle", "turn_by_turn"):
            raise ValueError(f"Unknown control_mode: {control_mode}")
        if opponent_mode not in ("heuristic", "self_play"):
            raise ValueError(f"Unknown opponent_mode: {opponent_mode}")
        if control_mode == "full_battle" and opponent_mode == "self_play":
            raise ValueError(
                "Self-play requires turn_by_turn control_mode. "
                "full_battle mode only supports heuristic opponents."
            )

    # ------------------------------------------------------------------
    # MultiTurnEnv hooks (match verifiers interface exactly)
    # ------------------------------------------------------------------

    async def setup_state(self, state: dict) -> dict:
        """Initialize a new battle.

        Called once at the start of each rollout.
        For turn_by_turn mode, creates a BattleManager and starts the battle.
        """
        # Clean up any previous manager (H7: prevents leak on retry)
        old_manager = state.get("manager")
        if old_manager is not None and hasattr(old_manager, 'close'):
            try:
                await old_manager.close()
            except Exception:
                pass

        state["trajectory"] = []
        state["game_over"] = False
        state["turn"] = 0
        state["decision_count"] = 0  # includes force-switches
        state["truncated"] = False
        state["won"] = None
        state["parse_failure_count"] = 0
        state["battle"] = None
        state["manager"] = None

        if self.control_mode == "turn_by_turn":
            from pokemon_rl.battle import BattleManager

            manager = BattleManager(
                port=self.port,
                battle_format=self.battle_format,
                server_host=self.server_host,
            )
            state["manager"] = manager

            if self.opponent_mode == "self_play":
                pending = await manager.start_battle_selfplay()
                # pending is [(idx, battle), ...] — unpack tuples
                state["_pending_states"] = list(pending)
                state["current_player"] = pending[0][0] if pending else 0
                state["battle"] = pending[0][1] if pending else None
                # Check if any battle is None (failed start)
                if not pending or any(b is None for _, b in pending):
                    state["game_over"] = True
            else:
                battle = await manager.start_battle(
                    opponent_type=self.opponent_type
                )
                state["battle"] = battle
                if battle is None:
                    state["game_over"] = True

        return state

    async def get_prompt_messages(self, state: dict) -> list[dict] | None:
        """Return LLM prompt for current game state.

        Returns None when the game is over (signals end of rollout).
        """
        if state["game_over"]:
            return None
        if state["turn"] >= self.max_game_turns:
            state["game_over"] = True
            state["truncated"] = True
            # Truncation is not a loss — mark as draw/unknown
            if "won" not in state or state["won"] is None:
                state["won"] = None
            return None

        battle = state.get("battle")
        if battle is None:
            return None

        return self.translator.battle_to_prompt(battle)

    async def add_trajectory_step(
        self, state: dict, trajectory_step: dict
    ) -> None:
        """Process LLM response and advance the game.

        Parses the LLM's text into a BattleOrder, submits it to the battle,
        records the step in the trajectory.

        For turn_by_turn mode, advances the game via BattleManager.step().
        For full_battle mode, just records the step (battle runs externally).
        """
        response_text = trajectory_step.get("completion", "")
        battle = state.get("battle")

        # Parse action
        if battle is not None:
            action = self.translator.parse_action(response_text, battle)
            if action is None:
                action = self.translator.get_fallback_action(battle)
                trajectory_step["parse_failed"] = True
                state["parse_failure_count"] = state.get("parse_failure_count", 0) + 1
            else:
                trajectory_step["parse_failed"] = False
            trajectory_step["parsed_action"] = (
                action.message if hasattr(action, "message") else str(action)
            )
        else:
            action = None
            trajectory_step["parsed_action"] = "no_battle"

        # Determine player index
        if self.opponent_mode == "self_play":
            player_idx = state.get("current_player", 0)
        else:
            player_idx = 0
        trajectory_step["player_idx"] = player_idx

        # Record force_switch flag
        if battle is not None and hasattr(battle, "force_switch"):
            trajectory_step["force_switch"] = bool(battle.force_switch)
        else:
            trajectory_step["force_switch"] = False

        # Record turn number from battle (doesn't increment on force-switch)
        if battle is not None and hasattr(battle, "turn"):
            trajectory_step["game_turn"] = battle.turn

        state["trajectory"].append(trajectory_step)
        state["decision_count"] += 1

        # Advance game state
        battle_before = battle  # Capture for step_reward_fn
        manager = state.get("manager")
        next_battle = None
        if manager is not None and action is not None:
            if self.opponent_mode == "self_play":
                await self._advance_selfplay(state, action, player_idx)
                # In self-play, post-resolution state isn't available per-player
                next_battle = None
            else:
                next_battle, done = await manager.step(action)
                state["battle"] = next_battle
                if done:
                    state["game_over"] = True
                    result = manager.get_result()
                    state["won"] = result["won"]
                elif next_battle is not None:
                    state["turn"] = next_battle.turn

        # Step-level reward (optional, separate from terminal reward)
        if self.step_reward_fn is not None and battle_before is not None:
            trajectory_step["step_reward"] = self.step_reward_fn(
                battle_before, next_battle, action, player_idx
            )
        else:
            trajectory_step["step_reward"] = 0.0

    async def _advance_selfplay(
        self, state: dict, action: Any, player_idx: int
    ) -> None:
        """Handle self-play turn advancement using sequential API.

        The hooks model calls add_trajectory_step once per LLM response,
        but self-play needs ALL pending actions submitted before the turn
        resolves. This method buffers pending states and only calls
        get_pending_selfplay_states after the last buffered action.

        Flow for a normal turn with 2 pending states:
            1. Hook call #1: P1's action → submit, pop P2 from buffer
            2. Hook call #2: P2's action → submit, buffer empty → get_pending
            3. New pending states arrive for next turn
        """
        manager = state["manager"]
        pending = state.get("_pending_states", [])

        # Submit action for current player
        await manager.submit_selfplay_action(player_idx, action)

        # Remove the current player's entry from the pending buffer
        pending = [(idx, b) for idx, b in pending if idx != player_idx]

        if pending:
            # More players need to act before the turn resolves.
            # Set the next buffered player's state for the next prompt.
            next_idx, next_battle = pending[0]
            state["current_player"] = next_idx
            state["battle"] = next_battle
            state["_pending_states"] = pending
            if next_battle is not None:
                state["turn"] = next_battle.turn
        else:
            # All buffered actions submitted — now ask BattleManager for
            # the next turn's states (this is when Showdown resolves).
            new_pending = await manager.get_pending_selfplay_states()

            if not new_pending:
                state["game_over"] = True
                result = manager.get_result()
                state["won"] = result["won"]
                state["_pending_states"] = []
                return

            next_idx, next_battle = new_pending[0]
            state["current_player"] = next_idx
            state["battle"] = next_battle
            state["_pending_states"] = list(new_pending)
            if next_battle is not None:
                state["turn"] = next_battle.turn

    # ------------------------------------------------------------------
    # Reward computation — single source of truth
    # ------------------------------------------------------------------

    def _compute_terminal_reward(self, won: bool | None) -> float:
        """Compute terminal reward from game outcome.

        All reward paths call this. Configurable via constructor args.
        """
        if won is None:
            return self.reward_draw
        return self.reward_win if won else self.reward_loss

    def _assign_rewards(
        self, trajectory: list, won: bool | None
    ) -> float:
        """Assign terminal rewards to all trajectory steps. Returns the reward.

        For heuristic mode: all steps get the same reward.
        For self-play: P1 steps get winner's reward, P2 steps get loser's.
        """
        if self.opponent_mode == "self_play":
            if won is None:
                p1_reward = self.reward_draw
                p2_reward = self.reward_draw
            elif won:  # P1 won
                p1_reward = self.reward_win
                p2_reward = self.reward_loss
            else:  # P2 won
                p1_reward = self.reward_loss
                p2_reward = self.reward_win

            for step in trajectory:
                step["reward"] = (
                    p1_reward if step["player_idx"] == 0 else p2_reward
                )
            return p1_reward  # state-level reward tracks P1
        else:
            reward = self._compute_terminal_reward(won)
            for step in trajectory:
                step["reward"] = reward
            return reward

    async def render_completion(self, state: dict) -> None:
        """Assign terminal rewards to all trajectory steps.

        Uses configurable reward_win / reward_loss / reward_draw.
        Self-play uses explicit per-player rewards (not `1.0 - reward`).
        """
        won = state.get("won")  # Don't default to False — None is meaningful
        truncated = state.get("truncated", False)

        reward = self._assign_rewards(state["trajectory"], won)

        state["reward"] = reward
        state["metrics"] = {
            "won": int(won) if won is not None else -1,
            "truncated": int(truncated),
            "turns": state["turn"],
            "decision_count": state["decision_count"],
            "trajectory_length": len(state["trajectory"]),
            "parse_failure_count": state.get("parse_failure_count", 0),
        }

    # ------------------------------------------------------------------
    # Standalone modes (for testing without verifiers)
    # ------------------------------------------------------------------

    async def run_standalone(
        self,
        action_fn: Callable | None = None,
    ) -> dict:
        """Run a complete game loop using BattleAdapter (full-battle mode).

        Uses the adapter's callback player. For testing state translation
        and basic battle flow.

        Args:
            action_fn: fn(battle) -> BattleOrder. If None, uses
                translator.get_fallback_action (random legal action).

        Returns:
            dict with keys: trajectory, won, turns, reward, battle_tag
        """
        if self.adapter is None:
            raise RuntimeError("run_standalone requires adapter to be set.")

        trajectory = []

        def capturing_callback(battle):
            """Wraps action_fn, captures prompt + action for trajectory."""
            try:
                prompt = self.translator.battle_to_prompt(battle)
            except Exception as e:
                prompt = [{"role": "error", "content": str(e)}]

            if action_fn is not None:
                order = action_fn(battle)
            else:
                order = self.translator.get_fallback_action(battle)

            trajectory.append({
                "turn": battle.turn,
                "prompt_length": sum(len(m["content"]) for m in prompt),
                "prompt_messages": len(prompt),
                "action": order.message if order else "/choose default",
                "player_idx": 0,
            })
            return order

        result = await self.adapter.run_battle(action_fn=capturing_callback)

        won = result["won"]
        reward = self._assign_rewards(trajectory, won)

        return {
            "trajectory": trajectory,
            "won": won,
            "turns": result["turns"],
            "reward": reward,
            "battle_tag": result.get("battle_tag"),
        }

    async def run_turn_by_turn(
        self,
        action_fn: Callable | None = None,
    ) -> dict:
        """Run a complete game using BattleManager step-by-step.

        This tests the turn-by-turn control path without needing an LLM.
        Uses a callback function to decide actions at each step.

        Args:
            action_fn: fn(battle) -> BattleOrder. If None, uses
                translator.get_fallback_action (random legal action).

        Returns:
            dict with keys: trajectory, won, turns, reward, battle_tag,
            decision_count, selfplay
        """
        from pokemon_rl.battle import BattleManager

        async with BattleManager(
            port=self.port,
            battle_format=self.battle_format,
            server_host=self.server_host,
        ) as manager:
            trajectory = []

            if self.opponent_mode == "self_play":
                return await self._run_selfplay_standalone(
                    manager, action_fn, trajectory
                )

            # Heuristic opponent mode
            battle = await manager.start_battle(
                opponent_type=self.opponent_type
            )
            turn_count = 0

            while battle is not None and turn_count < self.max_game_turns:
                # Generate prompt
                try:
                    prompt = self.translator.battle_to_prompt(battle)
                    prompt_length = sum(len(m["content"]) for m in prompt)
                except Exception:
                    prompt_length = 0

                # Get action
                if action_fn is not None:
                    order = action_fn(battle)
                else:
                    order = self.translator.get_fallback_action(battle)

                trajectory.append({
                    "turn": battle.turn,
                    "prompt_length": prompt_length,
                    "action": order.message if order else "/choose default",
                    "player_idx": 0,
                    "force_switch": bool(
                        getattr(battle, "force_switch", False)
                    ),
                })

                battle, done = await manager.step(order)
                turn_count += 1
                if done:
                    break

            result = manager.get_result()
            won = result["won"]
            truncated = turn_count >= self.max_game_turns
            reward = self._assign_rewards(trajectory, won)

            return {
                "trajectory": trajectory,
                "won": won,
                "turns": result["turns"],
                "reward": reward,
                "truncated": truncated,
                "battle_tag": result.get("battle_tag"),
                "decision_count": len(trajectory),
                "selfplay": False,
            }

    async def _run_selfplay_standalone(
        self,
        manager: Any,
        action_fn: Callable | None,
        trajectory: list,
    ) -> dict:
        """Self-play standalone: both sides use action_fn.

        Uses the sequential selfplay API which naturally handles
        force-switches (asymmetric state count per cycle).
        Manager cleanup handled by caller's `async with`.
        """
        pending = await manager.start_battle_selfplay()
        step_count = 0

        while pending and step_count < self.max_game_turns * 2:
            for idx, state in pending:
                # Decide action
                if action_fn is not None:
                    order = action_fn(state)
                else:
                    order = self.translator.get_fallback_action(state)

                trajectory.append({
                    "turn": state.turn,
                    "action": order.message if order else "/choose default",
                    "player_idx": idx,
                    "force_switch": bool(getattr(state, "force_switch", False)),
                })

                await manager.submit_selfplay_action(idx, order)
                step_count += 1

            pending = await manager.get_pending_selfplay_states()

        result = manager.get_result()
        won = result["won"]
        reward = self._assign_rewards(trajectory, won)

        return {
            "trajectory": trajectory,
            "won": won,
            "turns": result["turns"],
            "reward": reward,
            "battle_tag": result.get("battle_tag"),
            "decision_count": len(trajectory),
            "selfplay": True,
        }
