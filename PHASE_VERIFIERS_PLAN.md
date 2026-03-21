# Phase: Verifiers Integration Plan

## Overview

Transform `PokemonBattleEnv` from a standalone env with verifiers-like hooks into a proper
`vf.MultiTurnEnv` subclass. The env becomes the **single authority on the agent's interface**:
what it sees (observations), what it does (actions), and what it learns (rewards/trajectories).

All changes work within the `@final rollout()` loop — we override hooks, not the loop itself.

**Scope**: Infrastructure only. No RL training in this phase, but the design must be fully
compatible with prime-rl's orchestrator, trajectory strategies, and GRPO training.

**Reviewed**: Three rounds of adversarial review against actual verifiers source code.
All findings integrated into this document.

---

## Part 1: Hard Constraints from Verifiers Framework

Verified against source in `prime-rl/.venv/lib/python3.12/site-packages/verifiers/`.

| # | Constraint | Source | Impact |
|---|-----------|--------|--------|
| C1 | `rollout()` is `@final` | `multiturn_env.py:128` | Cannot override. Must work through hooks. |
| C2 | `env_response()` is `@abstractmethod` | `multiturn_env.py:42` | Must implement, even if unused by our override. |
| C3 | `state["completion"]` must be set | `multiturn_env.py:83-95` | Framework extracts for IPC. |
| C4 | `add_trajectory_step` receives `TrajectoryStep` TypedDict | `multiturn_env.py:115-126` | Has `prompt`, `completion` (Messages), `tokens`, `extras`. |
| C5 | Fresh prompts break interleaved prefix-sharing | `trajectories.py:81` | Must use `trajectory_strategy="branching"`. |
| C6 | `branch_rollout` reads `step.get("reward")`, `step.get("advantage")` | `trajectories.py:137-138` | Pre-set in `_assign_rewards` to preserve through pipeline. |
| C7 | `score_rollouts=False` zeroes rewards | RSAgent bug pattern | Must use `score_rollouts=True` with passthrough rubric. |
| C8 | Passthrough rubric prevents reward overwrite | RSAgent pattern | `PokemonRubric` with registered passthrough function. |
| C9 | `max_turns` counts trajectory steps, not game turns | `multiturn_env.py:60-61` | Use `max_turns=-1` + custom `@vf.stop`. |
| C10 | `score_group` overwrites `state["metrics"]` | `rubric.py:324-326` | Use rubric metric functions, not manual dict. |
| C11 | `score_group` fills `t["advantage"]` before orchestrator | `rubric.py:319-321` | Must pre-set per-step advantage when rewards vary. |
| C12 | `vf.register_environment` does not exist | Verified — uses `importlib` | Must expose `load_environment(**kwargs)` function. |
| C13 | Rubric methods NOT auto-discovered | `rubric.py:40` | Must explicitly call `add_reward_func`/`add_metric`. |

---

## Part 2: The `@final rollout()` Loop

The entire design fits within this loop (cannot be changed):

```python
# multiturn_env.py:128-155
state = await self.init_state(input, client, model, sampling_args)
state = await self.setup_state(state)                     # ← we override

while not await self.is_completed(state):                 # checks @vf.stop conditions
    prompt_messages = await self.get_prompt_messages(state)  # ← we override
    if state.get("final_env_response") is not None:
        continue
    response = await self.get_model_response(state, prompt_messages)  # framework: LLM call
    await self.add_model_response(state, prompt_messages, response)   # calls add_trajectory_step
        # add_model_response creates TrajectoryStep with tokens/logprobs,
        # then calls add_trajectory_step(state, trajectory_step)         # ← we override

await self.render_completion(state)                       # ← we override
```

**Each loop iteration = one LLM call = one agent decision.** For self-play, alternating
iterations serve alternating agents. The game advances inside `add_trajectory_step`.

---

## Part 3: Design Decisions

### D1: Work Within Hooks (Not Custom Rollout)

Five override points + two decorators:

| Hook | Role |
|------|------|
| `setup_state(state)` | Create BattleManager, start battle, init `_AgentContext`(s) |
| `get_prompt_messages(state)` | Build prompt for current agent via `_build_agent_prompt` |
| `add_trajectory_step(state, step)` | Parse action, advance game, update agent state |
| `render_completion(state)` | Delegate to `_assign_rewards`, set framework fields |
| `env_response(messages, state)` | Required abstract; returns `[]` (unused by our override) |
| `@vf.stop game_over` | Signal battle end or max-turn truncation |
| `@vf.cleanup cleanup_battle` | Close BattleManager on any exit path |

### D2: Prompt Construction via `_build_agent_prompt` (Extensible)

Override `get_prompt_messages` to route through a single internal method:

```python
async def get_prompt_messages(self, state):
    agent = state["_agents"][state["_current_agent_idx"]]
    assert agent.battle is not None, "get_prompt_messages called with no active battle"
    return self._build_agent_prompt(agent, state)
```

**Default (this phase)**: Fresh prompt each turn. Requires `trajectory_strategy="branching"`.

**Future episodic mode**: Override `_build_agent_prompt` to return
`[system] + agent.message_history + [current_state]`. Single-agent can switch to
interleaved strategy. Self-play keeps branching (per-agent histories don't prefix-share).

**Future windowed mode**: Return `[system] + agent.message_history[-2*N:] + [state]`.

Enablers already built in: `_AgentContext.message_history` always records the conversation.
`env_response()` can be activated later for default framework accumulation mode.

Note: overrides of `_build_agent_prompt` are responsible for prompt length management
(truncating old history before exceeding `max_seq_len`).

### D3: Keep StateTranslator as Utility

StateTranslator stays as a class encapsulating format-specific logic:
- `battle_to_prompt(battle)` — pokechamp_io or simple format
- `parse_action(text, battle)` — JSON extraction + move matching + mechanic validation
- `get_fallback_action(battle)` — random legal action
- `extract_completion_text(messages)` — **new**: inverse of prompt production (Messages → str)

**The env decides WHEN** to build prompts (each turn, for which agent).
**The translator decides HOW** to format battle state into text.

### D4: Rename `opponent_mode` → `play_mode`

| Current | Proposed | Rationale |
|---------|----------|-----------|
| `opponent_mode="heuristic"` | `play_mode="single"` | "heuristic" is misleading — opponent could be checkpoint, LLM, human |
| `opponent_mode="self_play"` | `play_mode="self_play"` | Both sides produce training trajectories |

`opponent_type` is orthogonal — only relevant in single mode.

**Subclass threshold**: If a third `play_mode` is ever added (e.g., team battle), refactor
to subclasses. At 3+ modes, branching complexity exceeds shared logic benefit.

### D5: Self-Play as Interleaved Steps in One Trajectory

One rollout produces interleaved steps tagged with `agent_idx` in `step["extras"]`.
With branching, each becomes a separate `TrainingSample` with its own reward/advantage.

### D6: Branching Strategy (Hard Requirement for Fresh Prompts)

`trajectory_strategy = "branching"` is mandatory. Each turn → one `TrainingSample`.

### D7: `rollouts_per_example ≥ 4` for GRPO Signal

Each "example" is a placeholder triggering a random battle. With `rollouts_per_example=1`,
advantage = 0. With 4-8, rewards ∈ {0.0, 1.0}, mean ≈ 0.5, advantages ≈ ±0.5.

### D8: Showdown Server Lifecycle is External

Pre-start externally. Host/port via constructor args. Multiple EnvWorker processes
share one server per node.

### D9: `_AgentContext` — Passive Dataclass

```python
@dataclass
class _AgentContext:
    """Per-agent state during a rollout. Data only, no behavior."""
    agent_idx: int
    battle: Any = None
    steps: list = field(default_factory=list)
    message_history: list = field(default_factory=list)
    parse_failure_count: int = 0
    force_switch_count: int = 0
```

No methods — logic lives in the env, data lives in the context.

---

## Part 4: Architecture

```
AFTER:
  L4: PokemonBattleEnv(vf.MultiTurnEnv)
      ├── Overrides: setup_state, get_prompt_messages, add_trajectory_step, render_completion
      ├── Prompt: _build_agent_prompt (extensible override point)
      ├── Rewards: _assign_rewards (single override for rewards + advantages)
      ├── Internal: _AgentContext passive dataclass
      ├── Scoring: PokemonRubric (passthrough reward + metrics)
      ├── Stop: @vf.stop game_over
      ├── Cleanup: @vf.cleanup cleanup_battle
      └── Uses: StateTranslator as utility
  L3: StateTranslator    ─── utility (interface + extract_completion_text)
  L2: BattleManager      ─── unchanged
  L1: ShowdownEngine     ─── unchanged
```

BattleManager, ControllablePlayer, ShowdownEngine, concurrency model: all untouched.

---

## Part 5: Detailed Hook Implementations

### `PokemonRubric` (Scoring)

Rubric methods must be explicitly registered (C13). Combines passthrough reward
(prevents framework from overwriting our rewards) with game-specific metrics
(survive `score_group` overwrite of `state["metrics"]`).

```python
class PokemonRubric(vf.Rubric):
    """Passthrough reward + Pokemon-specific metrics for the scoring pipeline."""

    def __init__(self):
        super().__init__()
        self.add_reward_func(self.passthrough_reward)
        self.add_metric(self.won)
        self.add_metric(self.game_turns)
        self.add_metric(self.parse_failures)

    async def passthrough_reward(self, state, **kwargs):
        """Return env-computed reward. Prevents score_group from overwriting."""
        return state.get("reward", 0.0) or 0.0

    async def won(self, state):
        w = state.get("won")
        return int(w) if w is not None else -1

    async def game_turns(self, state):
        return state.get("game_turn", 0)

    async def parse_failures(self, state):
        return sum(1 for s in state["trajectory"]
                   if s.get("extras", {}).get("parse_failed"))
```

Note: `MultiTurnEnv.__init__` also adds `MultiTurnMonitorRubric` (with `num_turns` metric).
Final rubric is `RubricGroup([PokemonRubric(), MultiTurnMonitorRubric()])`. `num_turns`
counts trajectory steps (framework), `game_turns` counts actual game turns (ours).

### `__init__`

```python
class PokemonBattleEnv(MultiTurnEnv):
    def __init__(
        self,
        # Battle config
        battle_format: str = "gen1randombattle",
        port: int = 8000,
        server_host: str = "localhost",
        # Play mode
        play_mode: str = "single",         # "single" or "self_play"
        opponent_type: str = "random",      # only for single mode
        # Observation
        observation_format: str = "pokechamp_io",  # or "simple"
        system_prompt: str | None = None,
        # Rewards
        reward_win: float = 1.0,
        reward_loss: float = 0.0,
        reward_draw: float = 0.5,
        step_reward_fn: Callable | None = None,
        # Game
        max_game_turns: int = 200,
        # Dataset
        num_battles: int = 1000,
        **kwargs,
    ):
        if play_mode not in ("single", "self_play"):
            raise ValueError(f"Unknown play_mode: {play_mode}")

        self._system_prompt = system_prompt or self._default_system_prompt()

        super().__init__(
            max_turns=-1,                 # disable framework step-count; use @vf.stop
            dataset=self._make_battle_dataset(num_battles, battle_format),
            rubric=PokemonRubric(),       # passthrough reward + game metrics (C8, C10, C13)
            system_prompt=None,           # we manage prompts ourselves (not framework)
            score_rollouts=True,          # MANDATORY (C7)
            **kwargs,
        )
        self.translator = StateTranslator(format_style=observation_format)
        # ... store all config attributes ...
```

Key choices:
- `system_prompt=None` to super: prevents framework from prepending to dataset rows.
  We own prompt construction entirely via `_build_agent_prompt`.
- `max_turns=-1`: disables framework step counting. We use `@vf.stop game_over`
  which counts game turns (not trajectory steps — self-play doubles steps/turn).

### `setup_state`

```python
async def setup_state(self, state):
    state = await super().setup_state(state)

    state["game_over"] = False
    state["game_turn"] = 0
    state["won"] = None
    state["truncated"] = False

    try:
        manager = BattleManager(
            port=self.port, battle_format=self.battle_format,
            server_host=self.server_host,
        )
        state["manager"] = manager

        if self.play_mode == "self_play":
            pending = await manager.start_battle_selfplay()
            agents = [_AgentContext(0), _AgentContext(1)]
            state["_agents"] = agents
            state["_pending_states"] = list(pending)
            if not pending or any(b is None for _, b in pending):
                state["game_over"] = True
            else:
                state["_current_agent_idx"] = pending[0][0]
                agents[pending[0][0]].battle = pending[0][1]
                if len(pending) > 1:
                    agents[pending[1][0]].battle = pending[1][1]
        else:
            battle = await manager.start_battle(opponent_type=self.opponent_type)
            state["_agents"] = [_AgentContext(0)]
            state["_agents"][0].battle = battle
            state["_current_agent_idx"] = 0
            if battle is None:
                state["game_over"] = True
    except Exception as e:
        if state.get("manager"):
            await state["manager"].close()
            state["manager"] = None     # prevent double-close in @vf.cleanup
        raise vf.Error(f"Battle start failed: {type(e).__name__}: {e}") from e

    return state
```

### `@vf.stop game_over`

```python
@vf.stop
async def game_over(self, state):
    if state.get("game_over", False):
        return True
    if state.get("game_turn", 0) >= self.max_game_turns:
        state["game_over"] = True
        state["truncated"] = True
        return True
    return False
```

### `get_prompt_messages`

Contract: only called when game is active. `setup_state` raises `vf.Error` on failed
start; `add_trajectory_step` sets `game_over=True` on game end. The stop condition fires
before the next `get_prompt_messages` call.

```python
async def get_prompt_messages(self, state):
    agent = state["_agents"][state["_current_agent_idx"]]
    assert agent.battle is not None, "get_prompt_messages called with no active battle"
    return self._build_agent_prompt(agent, state)

def _build_agent_prompt(self, agent, state):
    """Build prompt for this agent's current turn.

    Default: fresh prompt from current battle state.
    Override for episodic (full history) or windowed (last N turns) modes.
    """
    try:
        messages = self.translator.battle_to_prompt(agent.battle)
    except Exception as e:
        raise vf.Error(f"Prompt build failed: {type(e).__name__}: {e}") from e

    if self._system_prompt:
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": self._system_prompt}
        else:
            messages.insert(0, {"role": "system", "content": self._system_prompt})
    return messages
```

### `env_response` (required abstract, unused by our override)

```python
async def env_response(self, messages, state):
    return []
```

### `add_trajectory_step`

```python
async def add_trajectory_step(self, state, trajectory_step):
    agent_idx = state["_current_agent_idx"]
    agent = state["_agents"][agent_idx]
    battle = agent.battle

    # 1. Extract text from TrajectoryStep completion (Messages format)
    response_text = self.translator.extract_completion_text(trajectory_step["completion"])

    # 2. Parse action
    try:
        action = self.translator.parse_action(response_text, battle) if battle else None
    except Exception:
        action = None

    parse_failed = action is None
    if parse_failed and battle:
        action = self.translator.get_fallback_action(battle)
        agent.parse_failure_count += 1

    # 3. Agent-centric metadata in extras
    trajectory_step["extras"] = {
        "agent_idx": agent_idx,
        "game_turn": battle.turn if battle else 0,
        "force_switch": bool(getattr(battle, "force_switch", False)),
        "parsed_action": action.message if action and hasattr(action, "message") else str(action),
        "parse_failed": parse_failed,
    }

    # 4. Record in trajectory + agent's step list
    state["trajectory"].append(trajectory_step)
    agent.steps.append(trajectory_step)

    # 5. Record conversation history (enables future episodic mode)
    prompt_messages = trajectory_step.get("prompt", [])
    agent.message_history.append({"role": "user", "content": self.translator.extract_user_content(prompt_messages)})
    agent.message_history.append({"role": "assistant", "content": response_text})

    # 6. Advance game
    manager = state["manager"]
    if manager and action:
        try:
            if self.play_mode == "self_play":
                await self._advance_selfplay(state, action, agent_idx)
            else:
                next_battle, done = await manager.step(action)
                agent.battle = next_battle
                if done:
                    state["game_over"] = True
                    result = manager.get_result()
                    state["won"] = result["won"]
                elif next_battle:
                    state["game_turn"] = next_battle.turn
        except Exception as e:
            state["game_over"] = True
            raise vf.Error(f"Battle step failed: {type(e).__name__}: {e}") from e

    # 7. Optional per-step reward shaping
    if self.step_reward_fn and battle:
        trajectory_step["extras"]["step_reward"] = self.step_reward_fn(
            battle, agent.battle, action, agent_idx
        )
```

### `_advance_selfplay` (internal)

```python
async def _advance_selfplay(self, state, action, agent_idx):
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
            state["game_turn"] = next_battle.turn
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
            state["game_turn"] = next_battle.turn
```

### `render_completion`

Delegates to single override point, then sets framework-required fields.
Metrics come from `PokemonRubric` — not set manually here.

```python
async def render_completion(self, state):
    """Assign rewards/advantages and set framework-required fields."""
    self._assign_rewards(state)

    trajectory = state["trajectory"]
    state["reward"] = trajectory[0]["reward"] if trajectory else 0.0
    state["completion"] = trajectory[-1]["completion"] if trajectory else []
```

### `_assign_rewards` (Single Override Point for Rewards + Advantages)

This is the most important method for training correctness. Full verified data flow:

```
render_completion → _assign_rewards sets step["reward"] and step["advantage"]
    ↓
score_group (rubric.py:319-323):
    state["reward"] = passthrough_reward(state)     ← reads our state["reward"]
    state["advantage"] = state["reward"] - avg      ← cross-rollout (state-level only)
    for t in trajectory:
        if t["advantage"] is None → set to state["advantage"]   ← SKIPPED (we pre-set)
        if t["reward"] is None → set to state["reward"]         ← SKIPPED (we pre-set)
    ↓
extract_result copies step["reward"], step["advantage"]
    ↓
branch_rollout → TrainingSample.reward = step["reward"], .advantage = step["advantage"]
    ↓
orchestrator:
    if te.reward is None → set rollout reward         ← SKIPPED (pre-set)
    if te.advantage is None → compute_advantages      ← SKIPPED (pre-set for self-play)
```

Pre-set values survive the entire pipeline because every downstream consumer
checks `is None` before overwriting.

```python
def _assign_rewards(self, state):
    """Assign per-step rewards and advantages from game outcome.

    Sets step["reward"] for every step. When rewards vary within the
    rollout (self-play, shaped rewards), also pre-sets step["advantage"]
    to prevent the framework from assigning uniform state-level values.

    When rewards are uniform (single-agent terminal-only), leaves
    advantage=None. score_group then fills in cross-rollout normalized
    advantage via the passthrough rubric (state-level, applied to all steps).

    Override for shaped per-step rewards, custom advantage baselines, etc.
    """
    trajectory = state["trajectory"]
    if not trajectory:
        return
    won = state.get("won")

    # --- Per-step rewards ---
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

    # --- Per-step advantages (only when rewards are non-uniform) ---
    rewards = [s["reward"] for s in trajectory]
    if all(r == rewards[0] for r in rewards):
        return  # uniform → score_group fills cross-rollout advantage

    mean_reward = sum(rewards) / len(rewards)
    for step in trajectory:
        step["advantage"] = step["reward"] - mean_reward

def _compute_terminal_reward(self, won):
    """Map game outcome to reward value."""
    if won is None:
        return self.reward_draw
    return self.reward_win if won else self.reward_loss
```

### `@vf.cleanup cleanup_battle`

```python
@vf.cleanup
async def cleanup_battle(self, state):
    manager = state.get("manager")
    if manager is not None:
        await manager.close()
        state["manager"] = None
```

---

## Part 6: Self-Play

### Turn Flow

```
Iteration 1:
  is_completed? → No
  get_prompt_messages → Agent 0's battle state
  LLM generates → Agent 0's action text
  add_trajectory_step:
    Parse action, submit to BattleManager
    Pop Agent 0 from pending, set Agent 1 as current

Iteration 2:
  is_completed? → No
  get_prompt_messages → Agent 1's battle state
  LLM generates → Agent 1's action text
  add_trajectory_step:
    Parse action, submit to BattleManager
    Buffer empty → call get_pending_selfplay_states()
    Showdown resolves → new pending states
    Set first pending agent as current

...continues until game_over...
```

### Force-Switch Handling

When a pokemon faints, only that player gets a force-switch.
`get_pending_selfplay_states()` returns 1 state instead of 2. The loop handles this
naturally — one iteration for the force-switch, then both on the next normal turn.
Agent ordering in pending is not assumed — code uses `pending[0][0]` as agent index.

### Reward Assignment

```
P0 wins:  P0 steps → reward_win,  P1 steps → reward_loss
P1 wins:  P0 steps → reward_loss, P1 steps → reward_win
Draw:     all steps → reward_draw
```

`state["won"]` is from P0's perspective (BattleManager uses P0's `_player.result_battle.won`).

### GRPO Advantage

With `rollouts_per_example=4`, self-play rewards are non-uniform within each rollout
(P0 ≠ P1). `_assign_rewards` pre-sets per-step advantage = `reward - mean(all rewards)`.

Example: P0 wins with reward_win=1.0, reward_loss=0.0:
- mean = (1.0 × p0_count + 0.0 × p1_count) / total ≈ 0.5
- P0 steps: advantage = +0.5, P1 steps: advantage = -0.5

Same pattern as RSAgent's `per_step_grpo` — validated in production.

---

## Part 7: Future Context Modes

Three modes via `_build_agent_prompt` (only Fresh implemented this phase):

**Fresh** (default): `translator.battle_to_prompt(battle)`. Independent per step.
Requires branching.

**Episodic** (future): `[system] + agent.message_history + [current_state]`.
Single-agent can use interleaved. Self-play keeps branching.

**Windowed** (future): `[system] + agent.message_history[-2*N:] + [current_state]`.

Enablers: `_AgentContext.message_history` always recorded. `_build_agent_prompt`
is the single override point.

---

## Part 8: Dataset & Registration

### Battle Dataset

```python
def _make_battle_dataset(self, num_battles, battle_format):
    from datasets import Dataset
    return Dataset.from_dict({
        "question": [f"Play a {battle_format} Pokemon battle." for _ in range(num_battles)],
        "answer": ["" for _ in range(num_battles)],
    })
```

Placeholder rows. Actual battles generated in `setup_state()`.

### Environment Discovery

Verifiers uses `importlib.import_module(env_id)` then calls `module.load_environment(**env_args)`.

```python
# In pokemon_rl/__init__.py
def load_environment(**kwargs):
    """Verifiers env discovery entry point."""
    from pokemon_rl.env import PokemonBattleEnv
    return PokemonBattleEnv(**kwargs)
```

### TOML Config

```toml
[[orchestrator.env]]
id = "pokemon_rl"               # must match importlib.import_module("pokemon_rl")

[orchestrator.env.args]
battle_format = "gen1randombattle"
port = 8000
server_host = "localhost"
play_mode = "self_play"
observation_format = "pokechamp_io"
reward_win = 1.0
reward_loss = 0.0
reward_draw = 0.0
max_game_turns = 200
num_battles = 10000

[orchestrator]
trajectory_strategy = "branching"   # MANDATORY for fresh-prompt mode
rollouts_per_example = 4            # Minimum for GRPO signal
```

---

## Part 9: Operational Concerns

### Showdown Server
- Pre-start externally. Health check in `setup_state`.
- Multiple EnvWorker processes share one server per node.
- Multi-node: one server per node, `server_host` points to correct node.

### Event Loop Isolation
- Each EnvWorker = separate process with its own POKE_LOOP.
- BattleManager cross-loop bridge works per-process.

### Error Boundary
Every BattleManager and translator call in hooks is wrapped in try/except → `vf.Error`.
The framework catches `vf.Error`, stores in `state["error"]`, and the `has_error` stop
condition terminates cleanly. `@vf.cleanup` then runs. Raw exceptions from poke-env
would crash the worker, so wrapping is mandatory at every external call site.

### `extract_result` Compatibility
`env_worker.py:extract_result()` must be updated to include `"extras": step.get("extras", {})`
in the extracted trajectory step. One-line change in prime-rl. `extras` is needed for
logging/metrics (agent_idx, parse_failed), not for training — `TrainingSample` does not
include extras.

---

## Part 10: File Changes

| File | Change | Scope |
|------|--------|-------|
| `env.py` | Major refactor: inherit `vf.MultiTurnEnv`, all hooks, `_AgentContext`, `_build_agent_prompt`, `_assign_rewards`, `PokemonRubric`, `@vf.stop`/`@vf.cleanup`, error wrapping | ~400 lines |
| `translator.py` | Add `extract_completion_text(messages)` static method, update docstring | ~20 lines |
| `__init__.py` | Add `load_environment(**kwargs)` entry point, update exports | ~10 lines |
| `pyproject.toml` | Add `verifiers` as optional dependency | ~5 lines |
| `battle.py` | No changes | 0 |
| `players.py` | No changes | 0 |
| `engine.py` | No changes | 0 |
| `data.py` | Minor: ensure `agent_idx` in logged fields | ~5 lines |
| `prime-rl env_worker.py` | Add `"extras": step.get("extras", {})` to `extract_result` | 1 line |
| `tests/test_env.py` | Major update: verifiers integration tests | ~300 lines |
| `tests/test_hooks.py` | Update for new hook signatures | ~150 lines |
| `tests/helpers.py` | New: standalone test utilities (from env.py) | ~200 lines |
| `docs/` | Update architecture.md, new verifiers.md | |

---

## Part 11: Test Plan

### Design Principles

**No-fall-through**: Every test verifies both correct AND incorrect behavior.
Tests target reward hacking, framework constraint violations, and self-play invariants.

### Unit Tests (No External Dependencies)

#### T1: TrajectoryStep Format Contract

```
T1.1: add_trajectory_step extracts text from Messages completion
T1.2: Empty completion → fallback action (no crash)
T1.3: Multi-message completion → extracts last assistant content
T1.4: extras has ALL keys: agent_idx, game_turn, force_switch, parsed_action, parse_failed
```

#### T2: Reward Assignment (Single Agent)

```
T2.1: Win → reward_win; loss → reward_loss
      NEGATIVE: win reward != loss reward
T2.2: Draw/truncation → reward_draw
      NEGATIVE: draw != win AND != loss
T2.3: Custom values (10, -5, 0) propagated correctly
T2.4: Empty trajectory → state["reward"] = 0.0
T2.5: Parse-failed steps get SAME terminal reward as successful steps
```

#### T3: Reward Assignment (Self-Play)

```
T3.1: P0 wins → P0=reward_win, P1=reward_loss
T3.2: P1 wins → P0=reward_loss, P1=reward_win
      NEGATIVE: T3.1 and T3.2 produce DIFFERENT assignments
T3.3: Draw → BOTH players get reward_draw
T3.4: Custom asymmetric (win=1.0, loss=-1.0):
      P1's reward is reward_loss, NOT (1.0 - reward_win)
T3.5: won=None → both get reward_draw
      NEGATIVE: won=None does NOT default to False
T3.6: Every step has agent_idx ∈ {0, 1}
T3.7: Cross-check: extras.agent_idx matches reward value
```

#### T4: Reward Hack Prevention

```
T4.1: Garbage text → fallback is RANDOM (50 calls → ≥2 distinct actions)
      NEGATIVE: not deterministic max-power
T4.2: PokemonRubric.passthrough_reward reads state["reward"], not computes own
T4.3: score_rollouts=True verified in __init__
T4.4: step["reward"] from _assign_rewards survives branch_rollout
```

#### T5: Stop Conditions

```
T5.1: game_over=True → is_completed True
T5.2: game_turn >= max_game_turns → game_over, truncated, is_completed True
T5.3: game_turn < max → is_completed False
T5.4: Self-play: max_game_turns counts GAME turns not trajectory steps
T5.5: Truncation → reward_draw (NOT reward_loss)
```

#### T6: _AgentContext

```
T6.1: Single mode → 1 agent. Self-play → 2 agents.
T6.2: message_history grows by 2 entries per step
T6.3: Each agent's steps contain only that agent's trajectory steps
T6.4: parse_failure_count correct (0 after 5 successes, >0 after failure)
```

#### T7: Self-Play Turn Alternation

```
T7.1: Initial pending has both agents
T7.2: After agent 0 acts, current switches to 1
T7.3: get_pending NOT called until ALL buffered actions submitted
T7.4: Force-switch → only one agent acts, then get_pending
T7.5: Game over → get_pending returns [] → game_over set
```

#### T8: Cleanup

```
T8.1: Normal completion → manager.close() called
T8.2: Error in add_trajectory_step → cleanup still runs
T8.3: Error in setup_state → manager closed, state["manager"]=None
T8.4: cleanup_battle idempotent (twice → no crash)
```

#### T9: Prompt Construction

```
T9.1: _build_agent_prompt returns Messages format
T9.2: System prompt override replaces default system message
T9.3: Different agents get different prompts (different battle states)
T9.4: System prompt insertion works even if translator returns no system message
```

#### T10: Dataset, Config & Discovery

```
T10.1: _make_battle_dataset returns Dataset with "question" column
T10.2: Dataset has correct number of rows
T10.3: play_mode="invalid" raises ValueError
T10.4: load_environment(battle_format="gen1randombattle") returns PokemonBattleEnv
       with correct attributes
```

#### T11: PokemonRubric

```
T11.1: PokemonRubric().funcs contains passthrough_reward
       NEGATIVE: funcs is not empty
T11.2: passthrough_reward returns state["reward"] value
T11.3: passthrough_reward with state["reward"]=None returns 0.0 (not TypeError)
T11.4: Metric functions (won, game_turns, parse_failures) are registered
```

#### T12: Advantage Differentiation

```
T12.1: Self-play → P0 and P1 steps have DIFFERENT advantages within one rollout
       NEGATIVE: not all the same value
T12.2: Pre-set step["advantage"] survives (not overwritten when non-None)
T12.3: Single-agent terminal-only → step["advantage"] remains None
T12.4: Non-uniform rewards (e.g., step_reward_fn) → advantages pre-set
```

#### T13: Error Handling

```
T13.1: BattleManager.step raises → vf.Error, cleanup runs
T13.2: setup_state with error → vf.Error, manager cleaned up, state["manager"]=None
T13.3: Translator exception → vf.Error (not raw exception crash)
```

### Integration Tests (Require Showdown Server + poke-env)

#### T14: Full Hooks Cycle (Single Agent)

```
T14.1: Full cycle with mock LLM → trajectory > 0, reward set, game_over, completion set
T14.2: Real BattleManager + random opponent → game completes, unique battle_tag
T14.3: Parse failure → fallback used, game continues, parse_failure_count > 0
```

#### T15: Full Hooks Cycle (Self-Play)

```
T15.1: Full game → both agents have steps, rewards opposite, game completes
T15.2: Force-switch → pending states handle asymmetry
T15.3: p0_steps + p1_steps == len(trajectory)
```

#### T16: Concurrent Battles

```
T16.1: 3 concurrent single-agent → unique battle_tags, no contamination
T16.2: 2 concurrent self-play → independent outcomes
```

#### T17: Message History Recording

```
T17.1: 5-turn game → 10 history entries per agent (5 user + 5 assistant)
T17.2: Self-play → each agent's history is independent
```

#### T18: Trajectory Format Compatibility

```
T18.1: step["reward"] is float after render_completion
T18.2: step["extras"]["agent_idx"] is int ∈ {0, 1}
T18.3: state["completion"] is valid Messages format
```

#### T19: Truncation Mid-Self-Play

```
T19.1: max_game_turns reached mid-turn → game_over, both get reward_draw
T19.2: Asymmetric step counts handled correctly in reward assignment
```

#### T20: score_group Survival (Critical Pipeline Test)

```
T20.1: After render_completion + score_group → state["reward"] equals env-computed value
T20.2: Per-step rewards unchanged (score_group only sets if None)
T20.3: Per-step advantages unchanged for self-play (pre-set, non-None)
T20.4: state["metrics"] contains won, game_turns, parse_failures (from PokemonRubric)
```

---

## Part 12: Implementation Order

```
Step 1: Core scaffolding
  - PokemonBattleEnv(MultiTurnEnv) shell
  - __init__ with dataset, PokemonRubric, score_rollouts
  - PokemonRubric class with registered funcs
  - env_response stub
  - _AgentContext dataclass
  - _build_agent_prompt (fresh mode)
  - _compute_terminal_reward
  - StateTranslator.extract_completion_text

Step 2: Single-agent hooks
  - setup_state (create BattleManager, start battle, error wrapping)
  - get_prompt_messages (route to _build_agent_prompt)
  - add_trajectory_step (parse action, advance game, error wrapping)
  - render_completion → _assign_rewards
  - @vf.stop game_over
  - @vf.cleanup cleanup_battle

Step 3: Self-play hooks
  - setup_state selfplay path
  - _advance_selfplay (pending buffer, turn resolution)
  - Per-agent reward + advantage assignment in _assign_rewards

Step 4: Registration + config
  - load_environment in __init__.py
  - _make_battle_dataset
  - TOML config template

Step 5: prime-rl change
  - extract_result: add "extras" field

Step 6: Move standalone test utilities
  - run_standalone → tests/helpers.py
  - run_turn_by_turn → tests/helpers.py
  - Backward-compat imports

Step 7: Unit tests (T1-T13)

Step 8: Integration tests (T14-T20)

Step 9: Documentation
  - Update architecture.md
  - New verifiers.md guide
  - Update TODO.md / PROGRESS.md
```

---

## Part 13: Open Questions (Deferred)

| # | Question | Recommendation |
|---|----------|----------------|
| Q1 | When to add episodic context mode? | RL training phase — fresh is sufficient initially |
| Q2 | Parse failure handling in training | Train on fallback (random prevents reward hacking). Track metrics. |
| Q3 | Single class vs subclasses for play modes | Single class until 3+ modes. Document threshold. |
| Q4 | Per-step shaped rewards | Terminal only this phase. step_reward_fn implementations in RL phase. |
| Q5 | Curriculum: heuristic → self-play warmup | Safer than direct self-play. Training-phase decision. |
| Q6 | Custom trajectory converter for self-play episodic | Defer until episodic mode. |
