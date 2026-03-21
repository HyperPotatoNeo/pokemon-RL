# Testing Protocol: Phase 4 Verifiers Integration

**Audience**: Claude implementation agent executing Phase 4.
**Authority**: This protocol is the testing authority. If implementation conflicts
with these tests, fix the implementation — not the tests (unless CODE_REVIEW.md
lists the test as explicitly wrong, in which case update both).

---

## 1. Test Files & Organization

| File | Type | Deps | Purpose |
|------|------|------|---------|
| `tests/test_phase4_unit.py` | Unit | None (login node) | Core logic: rubric, rewards, advantages, contexts |
| `tests/test_phase4_integration.py` | Integration | Showdown + poke-env | Real battles with hooks |
| `tests/test_phase4_gpu.py` | GPU | Showdown + poke-env + vLLM | LLM battles |
| `tests/test_env.py` | Existing unit | None | Pre-Phase-4 env tests (update for renames) |
| `tests/test_hooks.py` | Existing unit | None | Pre-Phase-4 hooks tests (update for renames) |

### Test Markers

```python
@pytest.mark.unit          # No external deps — runs on login node
@pytest.mark.integration   # Needs Showdown + poke-env — compute node
@pytest.mark.gpu           # Needs GPU + vLLM — compute node with GPU
@pytest.mark.multinode     # Needs 2+ compute nodes
```

---

## 2. Critical Anti-Reward-Hacking Rules

**DO NOT** optimize the implementation to make tests pass by:
1. **Hardcoding expected values** — e.g., returning 1.0 from `_assign_rewards` for
   tests that check `reward == 1.0`. The function must compute rewards from game state.
2. **Making fallback deterministic** — `get_fallback_action` MUST be random. A test
   verifies this by calling 50 times and expecting >= 2 distinct actions.
3. **Computing rewards in the rubric** — `PokemonRubric.passthrough_reward` MUST
   return `state["reward"]` (the env-computed value), not compute its own.
4. **Skipping advantage pre-setting** — Self-play rewards are non-uniform. If you
   leave `step["advantage"]` as None, `score_group` will assign the same advantage
   to winner and loser steps. This is WRONG for GRPO.
5. **Bypassing score_rollouts** — `score_rollouts=True` is mandatory. Setting it
   to False zeroes all rewards.

If a test fails, trace the root cause through the data flow. Do not patch symptoms.

---

## 3. Implementation Steps & Test Checkpoints

### Step 1: Core Scaffolding

**Implement**:
- `PokemonRubric` class with `_passthrough_reward_sync` and async wrapper
- `_AgentContext` dataclass
- `_build_agent_prompt` method
- `_compute_terminal_reward` (already exists, verify)

**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestPokemonRubric -v`
**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestAgentContext -v`
**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestPromptConstruction -v`

**Must pass**: All tests in these 3 classes.

### Step 2: Reward + Advantage Assignment

**Implement**:
- `_assign_rewards(self, state)` — new signature, reads state dict
- Per-step advantage pre-setting for non-uniform rewards
- leave advantage=None for uniform rewards

**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestAssignRewardsSingleAgent -v`
**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestAssignRewardsSelfPlay -v`
**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestAdvantagePipeline -v`

**Must pass**: All tests. Pay special attention to:
- `test_selfplay_advantages_preset` — advantages MUST be non-None for self-play
- `test_selfplay_advantages_opposite_sign` — P0 and P1 advantages differ
- `test_uniform_rewards_leave_advantage_none` — single agent leaves None
- `test_preset_advantage_survives_score_group_simulation` — THE critical pipeline test

### Step 3: Stop, Cleanup, Error Handling

**Implement**:
- `@vf.stop game_over` decorator
- `@vf.cleanup cleanup_battle` decorator
- Error wrapping: BattleManager/translator exceptions → `vf.Error`

**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestStopConditions -v`
**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestCleanup -v`
**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestErrorHandling -v`

**Must pass**: All. Note: ErrorHandling tests require verifiers installed.

### Step 4: Hooks Cycle (Mock BattleManager)

**Implement**:
- `setup_state` with `_AgentContext` creation, BattleManager start
- `get_prompt_messages` routing through `_build_agent_prompt`
- `add_trajectory_step` with Messages extraction, extras dict, game advancement
- `render_completion` with state["completion"] and _assign_rewards
- `_advance_selfplay` with pending buffer management

**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestHooksCycleSingleAgent -v`
**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestHooksCycleSelfPlay -v`
**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestStateFieldNames -v`

**Must pass**: All. Pay special attention to:
- `test_trajectory_step_has_extras` — extras dict with all required fields
- `test_messages_completion_extracted` — Messages format handling
- `test_advance_does_not_call_get_pending_prematurely` — strict mock validation
- `test_force_switch_one_player` — force-switch produces correct step count

### Step 5: Config + Discovery

**Implement**:
- `play_mode` parameter (replacing `opponent_mode`)
- `load_environment()` in `__init__.py`
- `_make_battle_dataset` method
- Verifiers inheritance (`PokemonBattleEnv(vf.MultiTurnEnv)`)

**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestDatasetConfigDiscovery -v`
**Run**: `.venv/bin/python -m pytest tests/test_phase4_unit.py::TestDecoratorRegistration -v`

### Step 6: Update Existing Tests

The following existing tests will need updates for Phase 4 renames:

1. **`test_env.py`**: Update `opponent_mode="heuristic"` → `play_mode="single"`,
   `opponent_mode="self_play"` → `play_mode="self_play"`. Update `state["turn"]`
   → `state["game_turn"]`. Update `_assign_rewards(trajectory, won)` → `_assign_rewards(state)`.

2. **`test_hooks.py`**: Same renames. The mock managers and test structure should
   remain the same — they're testing the SAME behavior, just with new parameter names.

**Run**: `.venv/bin/python -m pytest tests/test_env.py tests/test_hooks.py -v`

**Must pass**: All existing tests (updated for renames).

### Step 7: Full Unit Test Suite

**Run**: `.venv/bin/python -m pytest -m unit -v`

**Must pass**: ALL unit tests (new + existing).

### Step 8: Integration Tests (Compute Node)

**Setup** (on compute node in container):
```bash
source .venv/bin/activate
# Start Showdown if not running
node /pscratch/sd/s/siddart2/pokechamp/pokemon-showdown/pokemon-showdown start \
    --no-security --port 8000 &
sleep 5
```

**Run**:
```bash
python -m pytest tests/test_phase4_integration.py -v
```

**Must pass**: All integration tests.

### Step 9: GPU Tests (Optional but Recommended)

**Setup** (on GPU node in container):
```bash
source .venv/bin/activate
# Start Showdown
node /pscratch/sd/s/siddart2/pokechamp/pokemon-showdown/pokemon-showdown start \
    --no-security --port 8000 &
# Start vLLM
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8001 --max-model-len 4096 \
    --no-enable-log-requests &
sleep 30
```

**Run**:
```bash
VLLM_HOST=localhost VLLM_PORT=8001 MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
    python -m pytest tests/test_phase4_gpu.py -v -m gpu
```

### Step 10: Multi-Node Tests (Optional)

Requires 2 nodes from `_CAP_tinker` reservation.

**Node A** (Showdown server):
```bash
ssh nid008205 "bash /pscratch/sd/s/siddart2/pokemon-rl/scripts/setup_node.sh"
```

**Node B** (run tests):
```bash
ssh nid008268
cd /pscratch/sd/s/siddart2/pokemon-rl
source .venv/bin/activate
REMOTE_NODE=nid008205 python -m pytest tests/test_phase4_gpu.py -m multinode -v
```

---

## 4. Naming Convention Reference

| Before (Phase 3) | After (Phase 4) | Files Affected |
|-------------------|-----------------|----------------|
| `opponent_mode="heuristic"` | `play_mode="single"` | env.py, test_env.py, test_hooks.py |
| `opponent_mode="self_play"` | `play_mode="self_play"` | env.py, test_env.py, test_hooks.py |
| `state["turn"]` | `state["game_turn"]` | env.py, test_env.py, test_hooks.py |
| `state["current_player"]` | `state["_current_agent_idx"]` | env.py, test_hooks.py |
| `_assign_rewards(trajectory, won)` | `_assign_rewards(state)` | env.py, test_env.py |
| `step["player_idx"]` (direct) | `step["extras"]["agent_idx"]` | env.py, all tests |

---

## 5. Key Interfaces the Tests Expect

### PokemonRubric

```python
class PokemonRubric(vf.Rubric):  # or standalone if verifiers not available
    def __init__(self): ...
    def _passthrough_reward_sync(self, state: dict) -> float:
        """Synchronous passthrough. Returns state['reward'] or 0.0 if None/missing."""
    # Also has async version for rubric framework
```

### _AgentContext

```python
@dataclass
class _AgentContext:
    agent_idx: int
    battle: Any = None
    steps: list = field(default_factory=list)
    message_history: list = field(default_factory=list)
    parse_failure_count: int = 0
    force_switch_count: int = 0
```

### PokemonBattleEnv

```python
class PokemonBattleEnv:  # or (vf.MultiTurnEnv) with verifiers
    def __init__(self, battle_format, port, play_mode, observation_format, ...): ...

    # Attributes the tests read:
    self.play_mode: str          # "single" or "self_play"
    self.battle_format: str
    self.port: int
    self.reward_win: float
    self.reward_loss: float
    self.reward_draw: float
    self.translator: StateTranslator
    self.score_rollouts: bool    # True (with verifiers)

    # Methods:
    async def setup_state(self, state: dict) -> dict
    async def get_prompt_messages(self, state: dict) -> list | None
    async def add_trajectory_step(self, state: dict, step: dict) -> None
    async def render_completion(self, state: dict) -> None
    async def game_over(self, state: dict) -> bool  # @vf.stop
    async def cleanup_battle(self, state: dict) -> None  # @vf.cleanup
    async def env_response(self, messages, state) -> list  # required abstract
    def _assign_rewards(self, state: dict) -> None
    def _build_agent_prompt(self, agent: _AgentContext, state: dict) -> list[dict]
    def _compute_terminal_reward(self, won: bool | None) -> float
    def _make_battle_dataset(self, num_battles, battle_format) -> Dataset
    async def _advance_selfplay(self, state, action, agent_idx) -> None

# Module-level:
def load_environment(**kwargs) -> PokemonBattleEnv
```

### State Dict Fields (After Phase 4)

```python
state = {
    # Framework fields
    "task": str,
    "prompt": list,
    "completion": list,          # Set by render_completion (CR-2)
    "trajectory": list,
    "reward": float,
    "metrics": dict,
    "error": Exception | None,
    "is_truncated": bool,

    # Pokemon-specific fields
    "game_over": bool,
    "game_turn": int,            # Renamed from "turn" (CR-5)
    "won": bool | None,
    "truncated": bool,
    "manager": BattleManager | None,
    "_agents": list[_AgentContext],
    "_current_agent_idx": int,
    "_pending_states": list,     # Self-play only
}
```

### Trajectory Step Fields (After Phase 4)

```python
trajectory_step = {
    # Framework fields (set by add_model_response)
    "prompt": Messages,
    "completion": Messages,
    "tokens": dict,

    # Pokemon fields (set by add_trajectory_step → extras)
    "extras": {
        "agent_idx": int,        # 0 or 1
        "game_turn": int,
        "force_switch": bool,
        "parsed_action": str,
        "parse_failed": bool,
        "step_reward": float,    # Optional per-step reward
    },

    # Reward fields (set by render_completion → _assign_rewards)
    "reward": float,
    "advantage": float | None,   # Pre-set for non-uniform, None for uniform
}
```

---

## 6. What Counts as "Passing"

### Unit Tests (Step 7)
ALL must pass with zero failures, zero errors. No skips except for
`@requires_verifiers` tests when verifiers not installed.

### Integration Tests (Step 8)
ALL must pass. Flaky failures (timeouts, Showdown restarts) may be retried
once. If a test fails consistently, it's a real bug.

### GPU Tests (Step 9)
Best effort. LLM output is stochastic. Tests verify game completion and
trajectory integrity, not specific LLM responses. All should pass if
vLLM is serving and Showdown is running.

### Existing Tests (Step 6)
ALL existing tests must still pass after updates. If a test fails,
the Phase 4 changes broke something — fix the implementation.

---

## 7. Debugging Tips

**Test fails with `ImportError: verifiers`**: Run unit tests without
verifiers-dependent tests: `pytest -m "unit and not requires_verifiers"`

**Test fails with `AssertionError: get_pending called before all actions`**:
`_advance_selfplay` is calling `get_pending_selfplay_states()` when the
pending buffer still has entries. Check the buffer management logic.

**Test fails with `step['advantage'] is None` for self-play**:
`_assign_rewards` is not pre-setting advantages for non-uniform rewards.
Check the "all rewards equal" detection at the end of `_assign_rewards`.

**Test fails with `state['completion'] not in state`**:
`render_completion` is not setting `state["completion"]`. Add:
`state["completion"] = trajectory[-1]["completion"] if trajectory else []`

**Integration test hangs**: Battle deadlock. Check that `_advance_selfplay`
correctly handles the force-switch case (1 pending state instead of 2).
Set `max_game_turns=50` for faster debugging.

**GPU test fails with `ConnectionError`**: vLLM not running or wrong port.
Check: `curl http://localhost:8001/v1/models`
