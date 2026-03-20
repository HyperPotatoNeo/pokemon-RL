"""Layer 4: Pokemon battle environment — MultiTurnEnv skeleton.

Implements the 4-hook interface from prime-rl's verifiers framework:
    setup_state          — Start a new battle
    get_prompt_messages  — Return current player's state as LLM prompt
    add_trajectory_step  — Parse action, advance game
    render_completion    — Assign terminal rewards

Currently standalone. Future: inherits from vf.MultiTurnEnv for
full prime-rl integration with branching trajectory strategy.

Usage (standalone testing):
    env = PokemonBattleEnv(adapter, translator)
    result = await env.run_standalone()

Usage (future verifiers integration):
    # PokemonBattleEnv will inherit from vf.MultiTurnEnv
    # The orchestrator calls the 4 hooks automatically
"""

from __future__ import annotations

from typing import Any, Callable


def _passthrough_reward(state: dict, **kwargs) -> float:
    """Passthrough rubric function for verifiers integration.

    Returns pre-computed reward from render_completion.
    Prevents verifiers' rubric from overwriting our rewards.
    """
    return state.get("reward", 0.0) or 0.0


class PokemonBattleEnv:
    """Pokemon Showdown RL environment.

    This class defines the MultiTurnEnv interface for Pokemon battles.
    Each trajectory step = one game turn decision.

    For heuristic opponent mode:
        - Each turn generates 1 trajectory step (training player only)
        - Opponent acts automatically inside add_trajectory_step

    For self-play mode (planned):
        - Each turn generates 2 trajectory steps (one per player)
        - Players alternate: even steps = P1, odd steps = P2
        - Both actions must be collected before the turn resolves

    Args:
        adapter: BattleAdapter instance for running battles
        translator: StateTranslator instance for prompt/action conversion
        opponent_mode: "heuristic" (default) or "self_play" (planned)
        max_game_turns: Maximum turns before game is truncated
    """

    def __init__(
        self,
        adapter: Any,
        translator: Any,
        opponent_mode: str = "heuristic",
        max_game_turns: int = 200,
    ):
        self.adapter = adapter
        self.translator = translator
        self.opponent_mode = opponent_mode
        self.max_game_turns = max_game_turns

        if opponent_mode == "self_play":
            raise NotImplementedError(
                "Self-play mode is planned but not yet implemented. "
                "Use opponent_mode='heuristic' for now."
            )

    # ------------------------------------------------------------------
    # MultiTurnEnv hooks (match verifiers interface exactly)
    # ------------------------------------------------------------------

    async def setup_state(self, state: dict) -> dict:
        """Initialize a new battle.

        Called once at the start of each rollout.
        Sets up tracking state for the game loop.
        """
        state["trajectory"] = []
        state["game_over"] = False
        state["turn"] = 0
        state["winner"] = None
        state["battle"] = None  # Set by run_standalone or turn-by-turn control
        return state

    async def get_prompt_messages(self, state: dict) -> list[dict] | None:
        """Return LLM prompt for current game state.

        Returns None when the game is over (signals end of rollout).

        In verifiers integration, the orchestrator generates a completion
        from these messages and passes it to add_trajectory_step.
        """
        if state["game_over"] or state["turn"] >= self.max_game_turns:
            return None

        battle = state.get("battle")
        if battle is None:
            return None

        return self.translator.battle_to_prompt(battle)

    async def add_trajectory_step(
        self, state: dict, trajectory_step: dict
    ) -> None:
        """Process LLM response and advance the game.

        Parses the LLM's text response into a BattleOrder, records
        it in the trajectory, and advances the game state.

        In full-battle mode (run_standalone), this is handled by the
        adapter's callback. In turn-by-turn mode (future), this method
        sends the action to the adapter and gets the new state.
        """
        response_text = trajectory_step.get("completion", "")
        battle = state.get("battle")

        if battle is not None:
            action = self.translator.parse_action(response_text, battle)
            if action is None:
                action = self.translator.get_fallback_action(battle)
            trajectory_step["parsed_action"] = str(action)
        else:
            trajectory_step["parsed_action"] = "no_battle"

        trajectory_step["player_idx"] = 0  # Training player
        state["trajectory"].append(trajectory_step)
        state["turn"] += 1

    async def render_completion(self, state: dict) -> None:
        """Assign terminal rewards to all trajectory steps.

        Binary reward: 1.0 for win, 0.0 for loss.
        All steps in the trajectory get the same terminal reward.

        For GRPO: with multiple concurrent games, advantage is computed
        across games (each game's win/loss is the reward signal).

        Future: per-step shaped rewards (damage dealt, pokemon fainted)
        can be added here by inspecting the battle state at each step.
        """
        won = state.get("won", False)
        reward = 1.0 if won else 0.0

        for step in state["trajectory"]:
            step["reward"] = reward

        state["reward"] = reward
        state["metrics"] = {
            "won": int(won),
            "turns": state["turn"],
            "trajectory_length": len(state["trajectory"]),
        }

    # ------------------------------------------------------------------
    # Standalone mode (for testing without verifiers)
    # ------------------------------------------------------------------

    async def run_standalone(
        self,
        action_fn: Callable | None = None,
    ) -> dict:
        """Run a complete game loop without verifiers.

        Uses the adapter's full-battle mode. The callback player captures
        the trajectory. State translation is tested via the callback.

        Args:
            action_fn: fn(battle) -> BattleOrder. If None, uses
                translator.get_fallback_action (highest power move).

        Returns:
            dict with keys: trajectory, won, turns, reward
        """
        trajectory = []

        def capturing_callback(battle):
            """Wraps action_fn, captures prompt + action for trajectory."""
            # Generate prompt (validates state translation)
            try:
                prompt = self.translator.battle_to_prompt(battle)
            except Exception as e:
                prompt = [{"role": "error", "content": str(e)}]

            # Get action
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

        # Run the battle
        result = await self.adapter.run_battle(action_fn=capturing_callback)

        # Assign rewards
        won = result["won"]
        reward = 1.0 if won else 0.0
        for step in trajectory:
            step["reward"] = reward

        return {
            "trajectory": trajectory,
            "won": won,
            "turns": result["turns"],
            "reward": reward,
            "battle_tag": result.get("battle_tag"),
        }
