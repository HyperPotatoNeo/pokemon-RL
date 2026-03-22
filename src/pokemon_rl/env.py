"""Layer 4: Pokemon battle environment — MultiTurnEnv interface.

Implements the hook interface from prime-rl's verifiers framework:
    setup_state          — Start a new battle, create _AgentContext(s)
    get_prompt_messages  — Build prompt for current agent via _build_agent_prompt
    add_trajectory_step  — Parse action, advance game, update agent state
    render_completion    — Assign rewards/advantages via _assign_rewards

Two play modes:
    "single"    — One agent vs heuristic opponent (random, max_damage, etc.)
    "self_play" — Both sides produce training trajectories

Reward/advantage flow (verified against verifiers + prime-rl source):
    render_completion → _assign_rewards sets step["reward"] + step["advantage"]
    → score_group: skips pre-set values (only sets if None)
    → extract_result: copies reward, advantage (NOT extras — IPC boundary)
    → branch_rollout: TrainingSample.reward/advantage from step
    → orchestrator: skips pre-set advantages (only sets if None)
    Note: step_reward_fn output is folded into step["reward"] by _assign_rewards
    so it survives extract_result. Original value kept in extras for logging.

Usage (verifiers integration):
    env = PokemonBattleEnv(battle_format="gen1randombattle", port=8000,
                           play_mode="self_play")
    # Orchestrator calls hooks automatically via @final rollout()

Usage (standalone testing):
    env = PokemonBattleEnv(battle_format="gen1randombattle", port=8000,
                           play_mode="single")
    result = await env.run_turn_by_turn()
"""

from __future__ import annotations

import asyncio
import logging
import os
import random as _random
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

from pokemon_rl.translator import StateTranslator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Team pool factory
# ---------------------------------------------------------------------------

def random_team_pool(team_dir: str) -> Callable[[], str]:
    """Load team files from a directory and return a callable that picks randomly.

    Args:
        team_dir: Path to directory containing .txt team files (Showdown paste format).
            Relative paths are resolved against the pokemon-rl package root
            (so TOML configs work regardless of the orchestrator's CWD).

    Returns:
        Callable that returns a random team string on each call.

    Raises:
        FileNotFoundError: If team_dir does not exist.
        ValueError: If team_dir contains no .txt files.
    """
    if not os.path.isabs(team_dir):
        # Resolve relative paths against the pokemon-rl package root
        # (2 levels up from src/pokemon_rl/env.py)
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        team_dir = os.path.join(pkg_root, team_dir)

    if not os.path.isdir(team_dir):
        raise FileNotFoundError(f"Team directory not found: {team_dir}")

    teams: list[str] = []
    for fname in sorted(os.listdir(team_dir)):
        if fname.endswith(".txt"):
            path = os.path.join(team_dir, fname)
            with open(path) as f:
                content = f.read().strip()
            if content:
                teams.append(content)

    if not teams:
        raise ValueError(f"No .txt team files found in {team_dir}")

    def _pick() -> str:
        return _random.choice(teams)

    return _pick

# ---------------------------------------------------------------------------
# Conditional verifiers import — works with and without verifiers installed
# ---------------------------------------------------------------------------
try:
    import verifiers as vf
    _HAS_VERIFIERS = True
except ImportError:
    _HAS_VERIFIERS = False

_EnvBase = vf.MultiTurnEnv if _HAS_VERIFIERS else object
_RubricBase = vf.Rubric if _HAS_VERIFIERS else object
_vf_stop = vf.stop if _HAS_VERIFIERS else lambda fn: fn
_vf_cleanup = vf.cleanup if _HAS_VERIFIERS else lambda fn: fn
_VfError = vf.Error if _HAS_VERIFIERS else RuntimeError


# ---------------------------------------------------------------------------
# Module-level passthrough (kept for backward compat with older imports)
# ---------------------------------------------------------------------------
def _passthrough_reward(state: dict, **kwargs) -> float:
    """Passthrough rubric function for verifiers integration."""
    reward = state.get("reward")
    if reward is None:
        return 0.0
    return reward


# ---------------------------------------------------------------------------
# PokemonRubric — passthrough reward + game metrics
# ---------------------------------------------------------------------------
class PokemonRubric(_RubricBase):
    """Passthrough reward + Pokemon-specific metrics for the scoring pipeline.

    Rubric methods must be explicitly registered (C13 — framework does not
    auto-discover methods). passthrough_reward prevents score_group from
    overwriting our env-computed rewards. Game metrics survive score_group's
    overwrite of state["metrics"].
    """

    def __init__(self):
        if _HAS_VERIFIERS:
            super().__init__()
            self.add_reward_func(self.passthrough_reward)
            self.add_metric(self.won)
            self.add_metric(self.game_turns)
            self.add_metric(self.parse_failures)

    def _passthrough_reward_sync(self, state: dict) -> float:
        """Synchronous passthrough. Returns state['reward'] or 0.0 if None/missing."""
        reward = state.get("reward")
        if reward is None:
            return 0.0
        return reward

    async def passthrough_reward(self, state, **kwargs):
        """Async wrapper for rubric framework. Returns float, not coroutine."""
        return self._passthrough_reward_sync(state)

    async def won(self, state):
        w = state.get("won")
        return int(w) if w is not None else -1

    async def game_turns(self, state):
        return state.get("game_turn", 0)

    async def parse_failures(self, state):
        return sum(
            1 for s in state.get("trajectory", [])
            if s.get("extras", {}).get("parse_failed")
        )


# ---------------------------------------------------------------------------
# _AgentContext — passive per-agent state during a rollout
# ---------------------------------------------------------------------------
@dataclass
class _AgentContext:
    """Per-agent state during a rollout. Data only, no behavior."""
    agent_idx: int
    battle: Any = None
    steps: list = field(default_factory=list)
    message_history: list = field(default_factory=list)
    parse_failure_count: int = 0
    force_switch_count: int = 0


# ---------------------------------------------------------------------------
# PokemonBattleEnv
# ---------------------------------------------------------------------------
class PokemonBattleEnv(_EnvBase):
    """Pokemon Showdown RL environment.

    Each trajectory step = one agent decision. For self-play, interleaved
    steps are tagged with agent_idx in step["extras"]. With branching
    trajectory strategy, each step becomes a separate TrainingSample.

    Args:
        battle_format: Pokemon Showdown format string
        port: Showdown server port
        server_host: Showdown hostname (for cross-node play)
        play_mode: "single" (vs heuristic) or "self_play" (both train)
        opponent_type: Heuristic type for single mode: "random", "max_damage"
        observation_format: "pokechamp_io" or "simple"
        system_prompt: Override system prompt (None = use translator's)
        reward_win: Terminal reward for wins (default 1.0)
        reward_loss: Terminal reward for losses (default 0.0)
        reward_draw: Terminal reward for draws/truncations (default 0.0)
        step_reward_fn: Optional per-step callback:
            (battle_before, battle_after, action, agent_idx) -> float
        max_game_turns: Max game turns before truncation
        num_battles: Dataset size (number of battle placeholders)
    """

    # Parameters that PokemonBattleEnv explicitly handles.
    # Anything not in this set (and not consumed by the verifiers base class)
    # will trigger a warning so that TOML typos are caught early.
    _KNOWN_KWARGS = frozenset({
        "battle_format", "port", "server_host", "play_mode", "opponent_type",
        "observation_format", "system_prompt", "reward_win", "reward_loss",
        "reward_draw", "step_reward_fn", "max_game_turns", "num_battles",
        "team_dir", "team_fn", "score_rollouts", "max_concurrent_battles",
    })

    def __init__(
        self,
        battle_format: str = "gen1randombattle",
        port: int = 8000,
        server_host: str = "localhost",
        play_mode: str = "single",
        opponent_type: str = "random",
        observation_format: str = "pokechamp_io",
        system_prompt: str | None = None,
        reward_win: float = 1.0,
        reward_loss: float = 0.0,
        reward_draw: float = 0.0,  # deliberately same as loss — no draw exploitation
        step_reward_fn: Callable | None = None,
        max_game_turns: int = 200,
        num_battles: int = 1000,
        max_concurrent_battles: int = 8,
        team_dir: str | None = None,
        team_fn: Callable[[], str] | None = None,
        **kwargs,
    ):
        # Backward compat: opponent_type="ladder" -> "kakuna"
        if opponent_type == "ladder":
            warnings.warn(
                'opponent_type="ladder" is deprecated. '
                'Use the specific opponent name (e.g., "kakuna") instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            opponent_type = "kakuna"

        if play_mode not in ("single", "self_play"):
            raise ValueError(f"Unknown play_mode: {play_mode}")

        # --- Warn about unrecognized kwargs ---
        for key in kwargs:
            if key not in self._KNOWN_KWARGS:
                warnings.warn(
                    f"PokemonBattleEnv received unrecognized keyword argument: "
                    f"'{key}'. Check for typos in your config.",
                    UserWarning,
                    stacklevel=2,
                )

        self.battle_format = battle_format
        self.port = port
        self.server_host = server_host
        self.play_mode = play_mode
        self.opponent_type = opponent_type
        self.reward_win = reward_win
        self.reward_loss = reward_loss
        self.reward_draw = reward_draw
        self.step_reward_fn = step_reward_fn
        self.max_game_turns = max_game_turns
        self.max_concurrent_battles = max_concurrent_battles
        # None = preserve translator's system prompt (e.g. pokechamp's rich prompt)
        # Explicit string = override with custom prompt
        self._system_prompt = system_prompt
        self.translator = StateTranslator(format_style=observation_format)

        # --- Team handling ---
        # team_fn takes priority over team_dir.
        # team_dir creates a team_fn via random_team_pool().
        # Neither provided → team_fn is None (OK for random battle formats).
        if team_fn is not None:
            self.team_fn = team_fn
        elif team_dir is not None:
            self.team_fn = random_team_pool(team_dir)
        else:
            self.team_fn = None
            # Warn if non-random format has no team source (B4)
            if "random" not in battle_format.lower():
                logger.warning(
                    f"battle_format='{battle_format}' is not a random format "
                    f"but no team_dir or team_fn was provided. "
                    f"Showdown may reject games or assign random teams."
                )

        if _HAS_VERIFIERS:
            # Prevent score_rollouts=False from being passed via kwargs (C7)
            kwargs.pop("score_rollouts", None)
            super().__init__(
                max_turns=-1,
                dataset=self._make_battle_dataset(num_battles, battle_format),
                rubric=PokemonRubric(),
                system_prompt=None,  # we manage prompts ourselves
                score_rollouts=True,  # MANDATORY (C7)
                **kwargs,
            )
            self.score_rollouts = True

    # ------------------------------------------------------------------
    # Dataset & Registration
    # ------------------------------------------------------------------

    @staticmethod
    def _make_battle_dataset(num_battles: int, battle_format: str):
        """Create placeholder dataset. Actual battles generated in setup_state."""
        from datasets import Dataset
        return Dataset.from_dict({
            "question": [
                f"Play a {battle_format} Pokemon battle."
                for _ in range(num_battles)
            ],
            "answer": ["" for _ in range(num_battles)],
        })

    # ------------------------------------------------------------------
    # Hook: setup_state
    # ------------------------------------------------------------------

    async def setup_state(self, state: dict) -> dict:
        """Initialize a new battle and create _AgentContext(s).

        Called once at the start of each rollout.
        """
        if _HAS_VERIFIERS:
            state = await super().setup_state(state)

        # Clean up any previous manager (H7: prevents leak on retry)
        old_manager = state.get("manager")
        if old_manager is not None and hasattr(old_manager, "close"):
            try:
                await old_manager.close()
            except Exception:
                pass

        # Acquire a battle slot (blocks if at max concurrent battles)
        from pokemon_rl.coordinator import BattleCoordinator
        coordinator = BattleCoordinator.get(self.max_concurrent_battles)
        await coordinator._semaphore.acquire()
        state["_has_coordinator_slot"] = True

        state["game_over"] = False
        state["game_turn"] = 0
        state["won"] = None
        state["truncated"] = False
        state["trajectory"] = []  # Always reset (stale steps from retries)
        # Fields required by branch_rollout (prime-rl/orchestrator/trajectories.py)
        state.setdefault("error", None)
        state.setdefault("example_id", "battle")

        try:
            from pokemon_rl.battle import BattleManager

            manager = BattleManager(
                port=self.port,
                battle_format=self.battle_format,
                server_host=self.server_host,
            )
            state["manager"] = manager

            if self.play_mode == "self_play":
                # Each player gets an independently-sampled team
                p1_team = self.team_fn() if self.team_fn else None
                p2_team = self.team_fn() if self.team_fn else None
                pending = await manager.start_battle_selfplay(
                    player1_team=p1_team,
                    player2_team=p2_team,
                )
                agents = [_AgentContext(0), _AgentContext(1)]
                state["_agents"] = agents
                state["_pending_states"] = list(pending)
                state["_current_agent_idx"] = 0  # always set
                if not pending or any(b is None for _, b in pending):
                    state["game_over"] = True
                else:
                    state["_current_agent_idx"] = pending[0][0]
                    agents[pending[0][0]].battle = pending[0][1]
                    if len(pending) > 1:
                        agents[pending[1][0]].battle = pending[1][1]
            else:
                # Single-agent mode — route via opponent registry
                from pokemon_rl.opponents import get_opponent_spec
                spec = get_opponent_spec(self.opponent_type)
                player_team = self.team_fn() if self.team_fn else None

                if spec.kind == "direct":
                    # In-process heuristic opponent
                    opponent_team = (
                        self.team_fn() if self.team_fn else None
                    )
                    battle = await manager.start_battle(
                        opponent_type=spec.opponent_type,
                        player_team=player_team,
                        opponent_team=opponent_team,
                    )
                elif spec.kind == "external":
                    # External process — serialized ladder to prevent
                    # env workers from matching each other.
                    # NOTE: serialization only works within a single process.
                    # Use workers_per_env=1 for external opponents.
                    battle = await manager.start_battle_ladder(
                        player_team=player_team,
                        serialize_matching=True,
                    )
                else:
                    raise ValueError(
                        f"Unknown opponent kind: '{spec.kind}' "
                        f"for opponent_type='{self.opponent_type}'"
                    )

                state["_agents"] = [_AgentContext(0)]
                state["_agents"][0].battle = battle
                state["_current_agent_idx"] = 0
                if battle is None:
                    state["game_over"] = True

        except Exception as e:
            if state.get("manager"):
                await state["manager"].close()
                state["manager"] = None
            raise _VfError(
                f"Battle start failed: {type(e).__name__}: {e}"
            ) from e

        return state

    # ------------------------------------------------------------------
    # Hook: @vf.stop game_over
    # ------------------------------------------------------------------

    @_vf_stop
    async def game_over(self, state: dict) -> bool:
        """Stop condition: game ended or max turns reached."""
        if state.get("game_over", False):
            return True
        if state.get("game_turn", 0) >= self.max_game_turns:
            state["game_over"] = True
            state["truncated"] = True
            return True
        return False

    # ------------------------------------------------------------------
    # Hook: @vf.cleanup cleanup_battle
    # ------------------------------------------------------------------

    @_vf_cleanup
    async def cleanup_battle(self, state: dict) -> None:
        """Clean up BattleManager on any exit path. Must not raise."""
        manager = state.get("manager")
        if manager is not None:
            try:
                await manager.close()
            except Exception:
                pass  # Cleanup must not propagate exceptions
            state["manager"] = None
        # Release coordinator slot so another battle can start
        if state.pop("_has_coordinator_slot", False):
            from pokemon_rl.coordinator import BattleCoordinator
            coordinator = BattleCoordinator.get(self.max_concurrent_battles)
            coordinator._semaphore.release()

    # ------------------------------------------------------------------
    # Hook: env_response (required abstract, unused by our override)
    # ------------------------------------------------------------------

    async def env_response(self, messages, state) -> list:
        """Required abstract method stub. Unused — we override get_prompt_messages."""
        return []

    # ------------------------------------------------------------------
    # Hook: get_prompt_messages
    # ------------------------------------------------------------------

    async def get_prompt_messages(self, state: dict) -> list[dict] | None:
        """Build prompt for current agent via _build_agent_prompt."""
        agent = state["_agents"][state["_current_agent_idx"]]
        assert agent.battle is not None, (
            "get_prompt_messages called with no active battle"
        )
        return self._build_agent_prompt(agent, state)

    def _build_agent_prompt(
        self, agent: _AgentContext, state: dict
    ) -> list[dict]:
        """Build prompt for this agent's current turn.

        Default: fresh prompt from current battle state.
        Override for episodic (full history) or windowed (last N turns) modes.
        """
        try:
            messages = self.translator.battle_to_prompt(agent.battle)
        except Exception as e:
            raise _VfError(
                f"Prompt build failed: {type(e).__name__}: {e}"
            ) from e

        if self._system_prompt is not None:
            if messages and messages[0].get("role") == "system":
                messages[0] = {
                    "role": "system",
                    "content": self._system_prompt,
                }
            else:
                messages.insert(
                    0, {"role": "system", "content": self._system_prompt}
                )
        return messages

    # ------------------------------------------------------------------
    # Hook: add_trajectory_step
    # ------------------------------------------------------------------

    async def add_trajectory_step(
        self, state: dict, trajectory_step: dict
    ) -> None:
        """Parse action from LLM response, advance game, update agent state."""
        agent_idx = state["_current_agent_idx"]
        agent = state["_agents"][agent_idx]
        battle = agent.battle

        # 1. Extract text from completion (handles both string and Messages)
        completion = trajectory_step.get("completion", "")
        response_text = self.translator.extract_completion_text(completion)

        # Probe: dump first turn's prompt + response to file for inspection
        probe_path = os.environ.get("POKEMON_RL_PROBE_PATH")
        if probe_path and not os.path.exists(probe_path):
            try:
                prompt = trajectory_step.get("prompt", [])
                with open(probe_path, "w") as f:
                    f.write("=== POKEMON RL PROBE: First Turn ===\n\n")
                    f.write(f"Agent index: {agent_idx}\n")
                    f.write(f"Game turn: {battle.turn if battle else '?'}\n\n")
                    f.write("--- PROMPT (system + user messages) ---\n\n")
                    for msg in prompt:
                        if isinstance(msg, dict):
                            f.write(f"[{msg.get('role', '?')}]\n")
                            f.write(f"{msg.get('content', '')}\n\n")
                    f.write("--- LLM RESPONSE ---\n\n")
                    f.write(f"{response_text}\n\n")
                    f.write("--- PARSE RESULT ---\n\n")
            except Exception:
                pass  # Probe must never break the pipeline

        # 2. Parse action
        try:
            action = (
                self.translator.parse_action(response_text, battle)
                if battle
                else None
            )
        except Exception:
            action = None

        parse_failed = action is None
        if parse_failed and battle:
            action = self.translator.get_fallback_action(battle)
            agent.parse_failure_count += 1

        # 3. Agent-centric metadata in extras (update, don't replace)
        extras = trajectory_step.get("extras", {})
        extras.update({
            "agent_idx": agent_idx,
            "game_turn": battle.turn if battle else 0,
            "force_switch": bool(getattr(battle, "force_switch", False)),
            "parsed_action": (
                action.message if action and hasattr(action, "message")
                else str(action)
            ),
            "parse_failed": parse_failed,
        })
        trajectory_step["extras"] = extras

        # 4. Ensure temperature is set (required by branch_rollout)
        trajectory_step.setdefault("temperature", 1.0)

        # 5. Record in trajectory + agent's step list
        state["trajectory"].append(trajectory_step)
        agent.steps.append(trajectory_step)

        # 5. Record conversation history (enables future episodic mode)
        prompt_messages = trajectory_step.get("prompt", [])
        agent.message_history.append({
            "role": "user",
            "content": self.translator.extract_user_content(prompt_messages),
        })
        agent.message_history.append({
            "role": "assistant",
            "content": response_text,
        })

        # 6. Advance game
        battle_before = battle
        manager = state.get("manager")
        next_battle = None
        if manager and action:
            try:
                if self.play_mode == "self_play":
                    await self._advance_selfplay(state, action, agent_idx)
                    # Turn hasn't resolved yet — both players must act first
                    next_battle = None
                else:
                    next_battle, done = await manager.step(action)
                    agent.battle = next_battle
                    if done:
                        state["game_over"] = True
                        result = manager.get_result()
                        state["won"] = result["won"]
                    elif next_battle:
                        state["game_turn"] = max(
                            state.get("game_turn", 0), next_battle.turn
                        )
            except Exception as e:
                state["game_over"] = True
                raise _VfError(
                    f"Battle step failed: {type(e).__name__}: {e}"
                ) from e

        # 7. Optional per-step reward shaping
        if self.step_reward_fn and battle_before:
            trajectory_step["extras"]["step_reward"] = self.step_reward_fn(
                battle_before, next_battle, action, agent_idx
            )

    # ------------------------------------------------------------------
    # Self-play turn advancement
    # ------------------------------------------------------------------

    async def _advance_selfplay(
        self, state: dict, action: Any, agent_idx: int
    ) -> None:
        """Handle self-play turn advancement using sequential API.

        Buffers pending states. Calls get_pending_selfplay_states only after
        ALL buffered actions submitted (prevents deadlock).
        """
        manager = state["manager"]
        pending = state.get("_pending_states", [])

        await manager.submit_selfplay_action(agent_idx, action)
        pending = [(idx, b) for idx, b in pending if idx != agent_idx]

        if pending:
            next_idx, next_battle = pending[0]
            state["_current_agent_idx"] = next_idx
            state["_agents"][next_idx].battle = next_battle
            state["_pending_states"] = pending
            if next_battle:
                state["game_turn"] = max(
                    state.get("game_turn", 0), next_battle.turn
                )
        else:
            new_pending = await manager.get_pending_selfplay_states()
            if not new_pending:
                state["game_over"] = True
                result = manager.get_result()
                state["won"] = result["won"]
                state["_pending_states"] = []
                return

            next_idx, next_battle = new_pending[0]
            state["_current_agent_idx"] = next_idx
            state["_agents"][next_idx].battle = next_battle
            state["_pending_states"] = list(new_pending)
            for idx, b in new_pending[1:]:
                state["_agents"][idx].battle = b
            if next_battle:
                state["game_turn"] = max(
                    state.get("game_turn", 0), next_battle.turn
                )

    # ------------------------------------------------------------------
    # Hook: render_completion
    # ------------------------------------------------------------------

    async def render_completion(self, state: dict) -> None:
        """Assign rewards/advantages and set framework-required fields.

        Metrics are set here for standalone/integration testing. When running
        through the verifiers pipeline, PokemonRubric also provides metrics
        via score_group (which overwrites state["metrics"]).
        """
        self._assign_rewards(state)

        trajectory = state["trajectory"]
        # State-level reward: use P0's perspective (consistent with state["won"])
        if trajectory:
            p0_steps = [
                s for s in trajectory
                if s.get("extras", {}).get("agent_idx", 0) == 0
            ]
            state["reward"] = (
                p0_steps[0]["reward"] if p0_steps
                else self._compute_terminal_reward(state.get("won"))
            )
        else:
            state["reward"] = 0.0
        state["completion"] = (
            trajectory[-1]["completion"] if trajectory else []
        )

        won = state.get("won")
        state["metrics"] = {
            "won": int(won) if won is not None else -1,
            "game_turns": state.get("game_turn", 0),
            "trajectory_length": len(trajectory),
            "parse_failures": sum(
                1 for s in trajectory
                if s.get("extras", {}).get("parse_failed")
            ),
        }

    # ------------------------------------------------------------------
    # Reward computation — single source of truth
    # ------------------------------------------------------------------

    def _compute_terminal_reward(self, won: bool | None) -> float:
        """Map game outcome to reward value."""
        if won is None:
            return self.reward_draw
        return self.reward_win if won else self.reward_loss

    def _assign_rewards(self, state: dict) -> None:
        """Assign per-step rewards and advantages from game outcome.

        Sets step["reward"] for every step (terminal + step_reward_fn).
        When rewards vary within the rollout (self-play with a winner, or
        step rewards), also pre-sets step["advantage"] to prevent the
        framework from assigning uniform state-level values.

        When rewards are uniform (single-agent terminal-only, or self-play
        draw), leaves advantage=None so score_group fills cross-rollout
        normalized advantage.
        """
        trajectory = state["trajectory"]
        if not trajectory:
            return

        won = state.get("won")

        # --- Per-step terminal rewards ---
        if self.play_mode == "self_play":
            if won is None:
                p0_reward, p1_reward = self.reward_draw, self.reward_draw
            elif won:  # P0 won (state["won"] is from P0's perspective)
                p0_reward, p1_reward = self.reward_win, self.reward_loss
            else:      # P1 won
                p0_reward, p1_reward = self.reward_loss, self.reward_win

            for step in trajectory:
                aidx = step.get("extras", {}).get("agent_idx", 0)
                step["reward"] = p0_reward if aidx == 0 else p1_reward
        else:
            reward = self._compute_terminal_reward(won)
            for step in trajectory:
                step["reward"] = reward

        # --- Fold step_reward_fn output into step["reward"] ---
        # extras["step_reward"] is set by add_trajectory_step but extras
        # are NOT copied through prime-rl's IPC (extract_result only copies
        # reward, advantage, prompt, completion, tokens, temperature).
        # Folding here ensures step rewards reach training.
        # The original value remains in extras for logging/analysis.
        for step in trajectory:
            step_r = step.get("extras", {}).get("step_reward")
            if step_r is not None:
                step["reward"] += step_r

        # --- Per-step advantages (only when rewards are non-uniform) ---
        rewards = [s["reward"] for s in trajectory]
        if all(r == rewards[0] for r in rewards):
            return  # uniform → score_group fills cross-rollout advantage

        # Use config-derived baseline, not within-rollout mean.
        # Within-rollout mean is skewed by step-count asymmetry:
        # a winner with more steps gets lower per-step advantage.
        # The midpoint (reward_win + reward_loss) / 2 is deterministic
        # and gives uniform ±magnitude regardless of step counts.
        baseline = (self.reward_win + self.reward_loss) / 2
        for step in trajectory:
            step["advantage"] = step["reward"] - baseline

    # ------------------------------------------------------------------
    # Standalone modes (for testing without verifiers)
    # ------------------------------------------------------------------

    async def run_standalone(
        self,
        adapter: Any = None,
        action_fn: Callable | None = None,
    ) -> dict:
        """Run a complete game loop using BattleAdapter (full-battle mode).

        Args:
            adapter: BattleAdapter instance.
            action_fn: fn(battle) -> BattleOrder. If None, uses random.

        Returns:
            dict with keys: trajectory, won, turns, reward, battle_tag
        """
        if adapter is None:
            raise RuntimeError("run_standalone requires adapter to be set.")

        trajectory = []

        def capturing_callback(battle):
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
                "extras": {"agent_idx": 0},
            })
            return order

        result = await adapter.run_battle(action_fn=capturing_callback)
        won = result["won"]

        state = {"won": won, "trajectory": trajectory}
        self._assign_rewards(state)

        return {
            "trajectory": trajectory,
            "won": won,
            "turns": result["turns"],
            "reward": trajectory[0]["reward"] if trajectory else 0.0,
            "battle_tag": result.get("battle_tag"),
        }

    async def run_turn_by_turn(
        self,
        action_fn: Callable | None = None,
    ) -> dict:
        """Run a complete game using BattleManager step-by-step.

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

            if self.play_mode == "self_play":
                return await self._run_selfplay_standalone(
                    manager, action_fn, trajectory
                )

            # Single-agent mode — route via opponent registry
            from pokemon_rl.opponents import get_opponent_spec
            spec = get_opponent_spec(self.opponent_type)
            player_team = self.team_fn() if self.team_fn else None

            if spec.kind == "direct":
                opponent_team = self.team_fn() if self.team_fn else None
                battle = await manager.start_battle(
                    opponent_type=spec.opponent_type,
                    player_team=player_team,
                    opponent_team=opponent_team,
                )
            elif spec.kind == "external":
                battle = await manager.start_battle_ladder(
                    player_team=player_team,
                    serialize_matching=True,
                )
            else:
                raise ValueError(
                    f"Unknown opponent kind: '{spec.kind}' "
                    f"for opponent_type='{self.opponent_type}'"
                )
            turn_count = 0

            while battle is not None and turn_count < self.max_game_turns:
                try:
                    prompt = self.translator.battle_to_prompt(battle)
                    prompt_length = sum(len(m["content"]) for m in prompt)
                except Exception:
                    prompt_length = 0

                if action_fn is not None:
                    order = action_fn(battle)
                else:
                    order = self.translator.get_fallback_action(battle)

                trajectory.append({
                    "turn": battle.turn,
                    "prompt_length": prompt_length,
                    "action": order.message if order else "/choose default",
                    "extras": {
                        "agent_idx": 0,
                        "force_switch": bool(
                            getattr(battle, "force_switch", False)
                        ),
                    },
                })

                battle, done = await manager.step(order)
                turn_count += 1
                if done:
                    break

            result = manager.get_result()
            won = result["won"]
            truncated = turn_count >= self.max_game_turns

            state = {"won": won, "trajectory": trajectory}
            self._assign_rewards(state)

            return {
                "trajectory": trajectory,
                "won": won,
                "turns": result["turns"],
                "reward": trajectory[0]["reward"] if trajectory else 0.0,
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
        """Self-play standalone: both sides use action_fn."""
        p1_team = self.team_fn() if self.team_fn else None
        p2_team = self.team_fn() if self.team_fn else None
        pending = await manager.start_battle_selfplay(
            player1_team=p1_team,
            player2_team=p2_team,
        )
        step_count = 0

        while pending and step_count < self.max_game_turns * 2:
            for idx, battle_state in pending:
                if action_fn is not None:
                    order = action_fn(battle_state)
                else:
                    order = self.translator.get_fallback_action(battle_state)

                trajectory.append({
                    "turn": battle_state.turn,
                    "action": (
                        order.message if order else "/choose default"
                    ),
                    "extras": {
                        "agent_idx": idx,
                        "force_switch": bool(
                            getattr(battle_state, "force_switch", False)
                        ),
                    },
                })

                await manager.submit_selfplay_action(idx, order)
                step_count += 1

            pending = await manager.get_pending_selfplay_states()

        result = manager.get_result()
        won = result["won"]

        state = {"won": won, "trajectory": trajectory}
        self._assign_rewards(state)

        return {
            "trajectory": trajectory,
            "won": won,
            "turns": result["turns"],
            "reward": trajectory[0]["reward"] if trajectory else 0.0,
            "battle_tag": result.get("battle_tag"),
            "decision_count": len(trajectory),
            "selfplay": True,
        }
