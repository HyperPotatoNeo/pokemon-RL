# Eval Feature — Testing Protocol

This document is a step-by-step guide for testing and debugging the pokemon-rl eval feature. It covers every test tier (unit → integration → GPU → production), exact commands, expected outputs, and a debugging playbook for common failures.

**Read BEFORE starting:**
- `docs/architecture.md` — 4-layer design (ShowdownEngine → BattleManager → StateTranslator → PokemonBattleEnv)
- `docs/concurrency.md` — POKE_LOOP bridging (critical for understanding LLMPlayer)
- `docs/testing.md` — No-fall-through philosophy, mock patterns, marker conventions
- `CLAUDE.md` — Project venv, running tests, current status

---

## What Was Added

### New Package: `src/pokemon_rl/eval/`

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `__init__.py` | Package init | Exports `LLMPlayer`, `OpponentConfig`, `PokemonEvalConfig` |
| `config.py` | TOML config parsing | `OpponentConfig`, `PokemonEvalConfig.from_toml()`, `compute_node_share()` |
| `llm_player.py` | LLM opponent via vLLM API | `LLMPlayer.create()` → returns poke-env Player whose `choose_move` calls vLLM |
| `runner.py` | Main eval loop | `run_pokemon_eval(config)`, `start_vllm_server()`, `wait_for_health()` |
| `report.py` | Results aggregation | `compute_stats()`, `save_results()`, `generate_summary()`, `merge_node_results()` |

### Modified Files (minimal, ~21 lines total)

| File | Change | What to verify |
|------|--------|----------------|
| `opponents.py` | Added `"llm"` to `_REGISTRY` | `get_opponent_spec("llm")` returns `OpponentSpec(kind="direct", opponent_type="llm")` |
| `players.py` | Added `"llm"` case to `create_opponent()` | `create_opponent("llm", ..., llm_kwargs={...})` returns `_LLMPlayerImpl` |
| `env.py` | Added `llm_opponent_kwargs` param + threading through `setup_state` and `run_turn_by_turn` | PokemonBattleEnv accepts and passes through LLM config to BattleManager |

### Config and Scripts

| File | Purpose |
|------|---------|
| `configs/pokemon/eval_example.toml` | Example eval config with all 4 opponent types |
| `scripts/launch_eval.sh` | Generic launcher: Showdown → agent vLLM → metamon (if needed) → runner |
| `local_scripts/launch_eval_prod.sh` | NERSC sbatch: container setup, `_CAP_tinker` reservation, inner script pattern |

### Test Files

| File | Tier | Count | Requirements |
|------|------|-------|-------------|
| `tests/test_eval_unit.py` | Unit | 42 | Login node only. No Showdown, no GPU, no poke-env. |
| `tests/test_eval_integration.py` | Integration | 8 | Compute node + Showdown running on port 8000. |
| `tests/test_eval_gpu.py` | GPU | 3 | Compute node + Showdown + agent vLLM (+ optionally opponent vLLM). |

---

## Opponent Type Taxonomy

The eval config uses semantic types, not raw opponent names:

| Config `type` | Required sub-field | GPU Needed | Maps to `opponent_type` in env |
|---------------|-------------------|------------|-------------------------------|
| `"heuristic"` | `heuristic = "abyssal"` | No | `"abyssal"` (direct, in-process) |
| `"heuristic"` | `heuristic = "max_damage"` | No | `"max_damage"` (direct, in-process) |
| `"heuristic"` | `heuristic = "random"` | No | `"random"` (direct, in-process) |
| `"metamon"` | `agent = "kakuna"` | Yes (1 GPU) | `"kakuna"` (external, ladder) |
| `"llm"` | `model_name`, `base_url` | Yes (2 GPUs) | `"llm"` (direct, LLMPlayer) |

The mapping lives in `OpponentConfig.opponent_type_for_env` property.

---

## How LLMPlayer Works (read before debugging)

LLMPlayer is a poke-env `Player` subclass whose `choose_move(battle)` returns an **awaitable**. Poke-env's internal handler (line 760-761 of `vendor/pokechamp/poke_env/player/player.py`) checks `if isinstance(message, Awaitable): message = await message`, so this runs on POKE_LOOP.

**Critical path per move:**
1. `battle_to_prompt(battle)` via `asyncio.to_thread` (offloads CPU-bound pokechamp prompt building)
2. `AsyncOpenAI.chat.completions.create()` via `asyncio.wait_for(timeout=60)` (non-blocking HTTP)
3. `parse_action(text, battle)` → BattleOrder or None
4. On None: `get_fallback_action(battle)` → random legal action
5. On exception/timeout: `choose_default_move()` + increment failure counter
6. After 3 consecutive failures: `DefaultBattleOrder()` (forfeit)

**Lazy client creation:** `AsyncOpenAI` is created on first `choose_move` call, not in `__init__`. This ensures the httpx client binds to POKE_LOOP's event loop. If you see `RuntimeError: Event loop is closed` or similar, check that the client isn't being created on the wrong loop.

---

## How the Runner Works

`runner.py:run_pokemon_eval(config)` iterates over opponents **sequentially**:

1. For each opponent:
   - If `type="llm"` + `gpu_ids`: spawn opponent vLLM via `start_vllm_server()`, wait for health
   - Create `PokemonBattleEnv(opponent_type=..., llm_opponent_kwargs=...)`
   - Get eval dataset (N placeholder examples)
   - `asyncio.gather` N rollouts through verifiers pipeline (`_generate_rollout`)
   - Save JSONL results, compute stats
   - If LLM opponent: terminate opponent vLLM subprocess
2. Generate summary table + JSON

**Data flow through existing code:**
```
runner → PokemonBattleEnv(**env_kwargs)
       → setup_state() → get_opponent_spec("llm") → kind="direct"
       → manager.start_battle(opponent_type="llm", llm_kwargs={...})
       → create_opponent("llm", ..., llm_kwargs={...})
       → LLMPlayer.create(base_url=..., model_name=..., ...)
       → _LLMPlayerImpl(Player)
```

The kwargs chain is: `env.llm_opponent_kwargs` → `start_battle(**opp_kwargs)` → `create_opponent(**kwargs)` → `kwargs.get("llm_kwargs")`. Each hop was verified in the plan phase.

---

## Testing Tiers

### Tier 1: Unit Tests (login node)

**No external dependencies.** Tests config parsing, report math, opponent registry, and LLMPlayer logic with mocks.

```bash
cd /pscratch/sd/s/siddart2/pokemon-rl
.venv/bin/python -m pytest tests/test_eval_unit.py -v -m unit
```

**Expected:** 42 passed in <1s.

**Also verify no regressions in existing tests:**
```bash
.venv/bin/python -m pytest tests/ -v -m unit
```

**Expected:** 340 passed (298 existing + 42 new).

**What to look for if tests fail:**
- Import errors → check `src/pokemon_rl/eval/__init__.py` exports
- Config parsing errors → check TOML format in test fixtures
- `get_opponent_spec("llm")` fails → check `opponents.py` has the `"llm"` entry

### Tier 2: Integration Tests (compute node + Showdown)

Requires Showdown running on port 8000. No GPU needed — battles use random/heuristic actions.

**Step 1: Get a compute node**
```bash
salloc -A m5017 -C "gpu&hbm80g" --reservation=_CAP_tinker --qos=interactive --time 2:00:00 --gpus-per-node 4
# Note the node name (e.g., nid008205)
ssh nid008205
```

**Step 2: Enter container**
```bash
export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd $HOME
podman-hpc run --rm -it \
  --user "$(id -u):$(id -g)" --replace --name skyrl \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH=$HOME -e HOME=$HOME \
  -v "$HOME":"$HOME" -v "/global/homes/s/siddart2":"/global/homes/s/siddart2" \
  -w "$HOME/pokemon-rl" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 /bin/bash
```

**Step 3: Start Showdown**
```bash
unset NCCL_SOCKET_IFNAME
cd /pscratch/sd/s/siddart2/pokemon-rl
node vendor/pokechamp/pokemon-showdown/pokemon-showdown start --no-security --port 8000 &
# Wait ~5s, verify:
nc -z localhost 8000 && echo "Showdown OK"
```

**Step 4: Run integration tests**
```bash
source .venv/bin/activate
python -m pytest tests/test_eval_integration.py -v -m integration
```

**Expected:** All tests pass. Each test runs 3-10 real battles against random/abyssal via Showdown.

**What to look for if tests fail:**
- `Showdown not running` skip → Showdown didn't start. Check `node` binary path, port 8000.
- Timeout/hang → likely a poke-env WebSocket issue. Check Showdown process is alive.
- `BattleCoordinator` errors → the `_reset_coordinator` fixture in `conftest.py` should handle this. If not, add it to `test_eval_integration.py`.
- `won` field assertions → if all battles draw, may indicate action parsing issues. Check `observation_format`.

### Tier 3: GPU Tests (compute node + Showdown + vLLM)

Tests real LLM inference. Requires agent vLLM (and optionally opponent vLLM).

**Step 1-3:** Same as Tier 2 (compute node + container + Showdown).

**Step 4: Start agent vLLM**
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 \
    --port 8001 \
    --trust-remote-code \
    --data-parallel-size 2 &

# Wait ~2-3 min for model load, then verify:
curl -s http://localhost:8001/v1/models | python -m json.tool
```

**Step 5: Run GPU tests (heuristic opponents)**
```bash
VLLM_PORT=8001 MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
    python -m pytest tests/test_eval_gpu.py::TestEvalGPUvsHeuristic -v
```

**Expected:** Agent wins at least 1/5 vs random, completes 3 games vs abyssal.

**Step 6: Start opponent vLLM (for LLM-vs-LLM tests)**
```bash
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --port 8002 \
    --trust-remote-code \
    --data-parallel-size 2 &

# Wait ~1-2 min, verify:
curl -s http://localhost:8002/v1/models | python -m json.tool
```

**Step 7: Run LLM-vs-LLM GPU tests**
```bash
VLLM_PORT=8001 OPP_VLLM_PORT=8002 \
    MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507 \
    OPP_MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
    python -m pytest tests/test_eval_gpu.py::TestEvalGPUvsLLM -v
```

**Expected:** 3 battles complete, both sides produce moves, game turns > 0.

**What to look for if GPU tests fail:**
- `vLLM not running` skip → model not loaded yet. Wait longer or check CUDA_VISIBLE_DEVICES.
- Parse failure rate > 50% → model not generating valid JSON moves. Check `observation_format` (try `"simple"` for debugging — simpler prompts, easier to parse).
- LLMPlayer forfeiting → 3 consecutive failures. Check opponent vLLM port/model is correct.
- `asyncio.TimeoutError` in LLMPlayer → vLLM too slow. Increase `timeout` in LLMPlayer.create() or reduce concurrent battles.

### Tier 4: Full Pipeline Test (runner end-to-end)

Tests the actual `runner.py` entry point with a real config.

**Step 1-3:** Same setup (compute node + container + Showdown).

**Step 4: Create a minimal test config**
```bash
cat > /tmp/eval_test.toml << 'EOF'
agent_model = "Qwen/Qwen3-4B-Instruct-2507"
agent_base_url = "http://localhost:8001/v1"
battle_format = "gen1randombattle"
n_battles_per_opp = 5
max_concurrent_battles = 4
max_game_turns = 100
observation_format = "simple"
showdown_port = 8000
output_dir = "/tmp/eval_test_output"
sampling_max_tokens = 256
sampling_temperature = 0.7

[[opponents]]
name = "random"
type = "heuristic"
heuristic = "random"

[[opponents]]
name = "abyssal"
type = "heuristic"
heuristic = "abyssal"
EOF
```

**Step 5: Start agent vLLM (if not already running)**
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B-Instruct-2507 --port 8001 --trust-remote-code --data-parallel-size 2 &
# Wait for ready...
```

**Step 6: Run the eval runner**
```bash
python -m pokemon_rl.eval.runner /tmp/eval_test.toml
```

**Expected output:**
```
--- Evaluating vs random (type=heuristic, opp_type=random) ---
Running 5 battles vs random...
vs random: wins=X losses=Y draws=Z win_rate=XX.X% +/- X.X%

--- Evaluating vs abyssal (type=heuristic, opp_type=abyssal) ---
Running 5 battles vs abyssal...
vs abyssal: wins=X losses=Y draws=Z win_rate=XX.X% +/- X.X%

=== Eval Summary ===
Opponent                   Win%    Loss%    Draw%       SE    Turns      N
------------------------------------------------------------------------
random                     XX.X%   XX.X%    XX.X%    XX.XX%    XX.X      5
abyssal                    XX.X%   XX.X%    XX.X%    XX.XX%    XX.X      5
```

**Verify output files:**
```bash
ls /tmp/eval_test_output/
# Expected: random/  abyssal/  summary.json

cat /tmp/eval_test_output/random/results.jsonl | python -m json.tool --no-ensure-ascii | head
# Expected: JSONL with example_id, opponent, reward, won, game_turns, parse_failures

cat /tmp/eval_test_output/summary.json | python -m json.tool
```

**Step 7: Test with LLM opponent (requires 2 vLLM servers)**
```bash
cat > /tmp/eval_llm_test.toml << 'EOF'
agent_model = "Qwen/Qwen3-4B-Instruct-2507"
agent_base_url = "http://localhost:8001/v1"
battle_format = "gen1randombattle"
n_battles_per_opp = 3
max_concurrent_battles = 2
max_game_turns = 100
observation_format = "simple"
showdown_port = 8000
output_dir = "/tmp/eval_llm_test_output"
sampling_max_tokens = 256
sampling_temperature = 0.7

[[opponents]]
name = "qwen2.5-1.5b"
type = "llm"
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
base_url = "http://localhost:8002/v1"
gpu_ids = [2, 3]
max_tokens = 256
temperature = 0.7
observation_format = "simple"
EOF

# Start opponent vLLM manually (or let runner do it — but manual is easier to debug)
CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct --port 8002 --trust-remote-code --data-parallel-size 2 &

# Run eval (comment out gpu_ids in TOML to skip auto-start since we started manually)
python -m pokemon_rl.eval.runner /tmp/eval_llm_test.toml
```

### Tier 5: Launch Script Test

Tests the full `scripts/launch_eval.sh` end-to-end.

```bash
# Inside container on compute node:
bash scripts/launch_eval.sh /tmp/eval_test.toml
```

This starts Showdown, agent vLLM, and runs the eval. Verify the cleanup trap kills all subprocesses on exit.

---

## Debugging Playbook

### Problem: "Showdown not running" skip on all integration tests

**Cause:** Showdown server not started or wrong port.
**Fix:**
```bash
node vendor/pokechamp/pokemon-showdown/pokemon-showdown start --no-security --port 8000 &
sleep 5
nc -z localhost 8000  # Should succeed
```

### Problem: All battles result in draws / truncation

**Cause:** Actions not being parsed correctly → fallback to random → games hit max_game_turns.
**Debug:**
1. Check `observation_format`. Use `"simple"` for debugging (less complex prompts).
2. Add logging: `LOGLEVEL=DEBUG python -m pytest ...` — LLMPlayer logs parse failures.
3. Run a single battle manually to see prompts:
```python
from pokemon_rl.env import PokemonBattleEnv
env = PokemonBattleEnv(battle_format="gen1randombattle", port=8000,
                        play_mode="single", opponent_type="random",
                        observation_format="simple")
result = await env.run_turn_by_turn()
print(result)
```

### Problem: LLMPlayer forfeiting (3 consecutive failures)

**Cause:** vLLM not serving, wrong model name, or timeout too short.
**Debug:**
1. Verify vLLM is serving: `curl http://localhost:8002/v1/models`
2. Verify model name matches: the model name in the config must match what vLLM reports
3. Check timeout: default is 60s. If model is slow, increase in config.
4. Check LLMPlayer logs for the specific exception type (APIError, TimeoutError, etc.)

### Problem: `RuntimeError: Event loop is closed` in LLMPlayer

**Cause:** AsyncOpenAI client created on wrong event loop (not POKE_LOOP).
**Debug:** This should not happen with the lazy creation pattern. If it does:
1. Verify `_client` is `None` before first `choose_move` call
2. Verify `choose_move` is running on POKE_LOOP (add `import asyncio; print(asyncio.get_running_loop())`)
3. Check that no test creates the client outside of a battle context

### Problem: Runner hangs after opponent eval

**Cause:** Opponent vLLM subprocess not terminating.
**Debug:**
1. Check `opp_proc.poll()` — if None, process is still running
2. Try `opp_proc.kill()` instead of `terminate()`
3. Check if vLLM spawned child processes that need killing

### Problem: `KeyError: 'llm_kwargs'` in create_opponent

**Cause:** `llm_opponent_kwargs` not threaded through `env.py` → `start_battle` → `create_opponent`.
**Debug:**
1. Verify `env.py` passes `llm_kwargs=self.llm_opponent_kwargs` in the `opp_kwargs` dict
2. Verify `battle.py:start_battle` has `**opponent_kwargs` in its signature
3. Verify `players.py:create_opponent` reads `kwargs.get("llm_kwargs", {})`

### Problem: `verifiers` import fails

**Cause:** pokemon-rl not installed in prime-rl venv.
**Fix:**
```bash
cd /pscratch/sd/s/siddart2/prime-rl
source .venv/bin/activate
pip install -e /pscratch/sd/s/siddart2/pokemon-rl/vendor/pokechamp
pip install -e /pscratch/sd/s/siddart2/pokemon-rl
python -c "import verifiers; print('OK')"
```

### Problem: Existing unit tests regress after changes

**Fix:** Run the full unit suite to identify which test broke:
```bash
.venv/bin/python -m pytest tests/ -v -m unit --tb=short
```
The 3 modified files are minimal (opponents.py +1 line, players.py +12 lines, env.py +8 lines). Regressions would come from:
- Accidentally modifying the wrong line in `env.py`
- Breaking the `create_opponent` dispatch (wrong elif ordering)
- Breaking `_KNOWN_KWARGS` (typo in the new entry)

---

## Reserved Compute Nodes

The `_CAP_tinker` SLURM reservation has 6 GPU nodes for our account:
- Nodes: `nid008205, nid008268, nid008297, nid008304, nid008448, nid008480`
- Features: `hbm80g` (A100-80GB)
- Account: `m5017_g`, Partition: `gpu_ss11`
- Expires: 2026-03-29

```bash
salloc -A m5017 -C "gpu&hbm80g" --reservation=_CAP_tinker --qos=interactive --time 2:00:00 --gpus-per-node 4
```

---

## Port Layout

| Port | Service | Always running? |
|------|---------|----------------|
| 8000 | Showdown | Yes (started by launch script) |
| 8001 | Agent vLLM | Yes (started by launch script) |
| 8002 | Opponent vLLM (1st LLM opp) | Only during LLM opponent eval |
| 8003 | Opponent vLLM (2nd LLM opp) | Only if 2+ LLM opponents in sequence |

---

## GPU Allocation (4 GPUs/node, all DP)

| Opponent Type | Agent GPUs | Opponent GPUs | Agent DP |
|---------------|-----------|---------------|----------|
| Heuristic | 0, 1, 2, 3 | N/A | DP=4 |
| Metamon | 0, 1, 2 | 3 | DP=3 |
| LLM | 0, 1 | 2, 3 | DP=2 |

The runner can restart agent vLLM between opponents to change GPU count. For simplicity in testing, starting agent with DP=2 on GPUs 0,1 works for all opponent types.

---

## File Inventory (complete)

### Source files
```
src/pokemon_rl/eval/
├── __init__.py          # Package init, exports
├── config.py            # OpponentConfig, PokemonEvalConfig, compute_node_share
├── llm_player.py        # LLMPlayer.create() → _LLMPlayerImpl(Player)
├── runner.py            # run_pokemon_eval(), start_vllm_server(), main()
└── report.py            # compute_stats(), save_results(), generate_summary()
```

### Modified files
```
src/pokemon_rl/opponents.py   # +1 line: "llm" in _REGISTRY
src/pokemon_rl/players.py     # +12 lines: "llm" case in create_opponent
src/pokemon_rl/env.py         # +8 lines: llm_opponent_kwargs param + threading
```

### Configs
```
configs/pokemon/eval_example.toml   # Full example with heuristic + metamon + 2 LLM opponents
```

### Scripts
```
scripts/launch_eval.sh              # Generic: Showdown + vLLM + runner
local_scripts/launch_eval_prod.sh   # NERSC sbatch with container + _CAP_tinker reservation
```

### Tests
```
tests/test_eval_unit.py             # 42 unit tests (login node)
tests/test_eval_integration.py      # Integration tests (compute + Showdown)
tests/test_eval_gpu.py              # GPU tests (compute + Showdown + vLLM)
```
