# RL Training Guide

pokemon-rl integrates with [prime-rl](https://github.com/HyperPotatoNeo/prime-rl) via the verifiers framework. `PokemonBattleEnv` implements `vf.MultiTurnEnv` hooks, giving the orchestrator turn-by-turn control over Pokemon battles while the trainer updates model weights via GRPO.

## How It Works

```
prime-rl process layout:
┌─────────────────────────────────────────────┐
│ Inference Server (vLLM)  — GPU 0-2          │
│   Serves model completions via OpenAI API   │
├─────────────────────────────────────────────┤
│ Orchestrator  — CPU                         │
│   Calls PokemonBattleEnv hooks:             │
│   setup_state → get_prompt → add_step → ... │
│   Connects to Showdown server (port 8000)   │
├─────────────────────────────────────────────┤
│ Trainer  — GPU 3                            │
│   GRPO weight updates from rollout data     │
├─────────────────────────────────────────────┤
│ Showdown Server  — CPU (background)         │
│   Node.js Pokemon battle simulator          │
└─────────────────────────────────────────────┘
```

Each training step: orchestrator plays N battles, collects TrainingSamples, trainer runs GRPO update, weights broadcast to inference server, repeat. In branching mode each turn becomes a separate TrainingSample; in interleaved mode the full conversation becomes one TrainingSample.

## Prompt Construction

### Prompt Formats

The default `full_obs_cot` format (used by interleaved mode) and the `pokechamp_io` format both produce rich prompts with damage calculations, type effectiveness, speed comparisons, and move effects. They are built from three components:

**1. System Prompt** — Strategic battle role:
```
You are a pokemon battler in generation {gen} OU format Pokemon Showdown
that targets to win the pokemon battle. You can choose to take a move or
switch in another pokemon. Here are some battle tips: [strategic advice
about boosting moves, traps, speed, switching costs...]
```

**2. State Prompt** — Full battle state from pokechamp's `state_translate()`:
- Battle history (last 5 turns, with player perspective translation)
- Opponent pokemon: species, type, HP%, stats, boosts, ability, known moves, possible moves, type effectiveness
- Your active pokemon: same detail level + your actual move power calculations
- Your bench pokemon: species, type, HP%, stats, speed comparison, move type effectiveness
- Side conditions (hazards, screens)

**3. Action Constraint + CoT Prompt** — Forces JSON output with brief reasoning:
```
Choose the best action by thinking step by step. Your thought should be
no more than 3 sentences. Your output MUST be a JSON like:
{"thought":"<step-by-step-thinking>", "move":"<move_name>"} or
{"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}
```

This constraint adapts to the battle situation:
- Pokemon fainted → switch-only constraint
- No switches available → move-only constraint
- Both available → move-or-switch constraint

The constraint wraps the chain-of-thought reasoning inside JSON, ensuring parseable output without requiring `json_format=True` on the API. This matches pokechamp's `llm_player.io()` method.

**Example model response:**
```json
{"thought":"Gholdengo has high speed and defenses, and Ariados' Bug-type
moves are highly ineffective. Knockoff is super-effective and removes the
item.","move":"knockoff"}
```

### simple Format

Minimal text format with no pokechamp dependency. Lists active pokemon, HP, moves, and switches. Useful for smoke testing and debugging.

### Prompt Code Location

- `src/pokemon_rl/translator.py` — `StateTranslator._pokechamp_io_prompt()` builds the full prompt
- `vendor/pokechamp/pokechamp/prompts.py` — `state_translate()` produces the battle state text
- The CoT constraint is appended in `translator.py` (not in pokechamp)

## Configuration Reference

RL training configs live in `configs/pokemon/`. Each is a prime-rl TOML config with pokemon-rl environment args.

### Available Configs

| Config | Play Mode | Strategy | Description |
|--------|-----------|----------|-------------|
| `rl_test.toml` | self_play | branching | Integration testing — 3 steps, batch_size=4, gen9randombattle |
| `rl_interleaved.toml` | single | interleaved | Production interleaved vs abyssal — 600 steps, batch_size=64, gen9ou, seq_len=32768 |
| `rl_vs_abyssal_600_4x4.toml` | single | branching | Production branching vs abyssal — 600 steps, multi-node, gen9ou |
| `inference_node1.toml` | -- | -- | Standalone inference for Node 1 (DP=4, host=0.0.0.0) |
| `inference_interleaved.toml` | -- | -- | Standalone inference for interleaved (DP=4, 32K context, prefix caching) |

### Config Sections

```toml
max_steps = 100             # Training steps
seq_len = 4096              # Max sequence length (prompt + completion)

[model]
name = "Qwen/Qwen3-4B-Instruct-2507"

[wandb]
project = "pokemon-rl"
name = "my-run-name"

[orchestrator]
batch_size = 16             # Games per training step
rollouts_per_example = 16   # MUST equal batch_size (batch-level GRPO normalization)
trajectory_strategy = "branching"   # or "interleaved" — see below

[orchestrator.client]
base_url = ["http://localhost:8001/v1"]   # Must match inference.server.port

[orchestrator.sampling]
max_tokens = 800            # Per-turn completion budget (must fit CoT + JSON)
temperature = 1.0

[[orchestrator.env]]
id = "pokemon_rl"           # Registered via load_environment()
name = "pokemon-selfplay"

[orchestrator.env.args]
battle_format = "gen9ou"    # Showdown format string
play_mode = "self_play"     # "self_play" or "single"
port = 8000                 # Showdown server port
observation_format = "full_obs_cot"  # "full_obs_cot", "pokechamp_io", or "simple"
reward_win = 1.0
reward_loss = 0.0
reward_draw = 0.0           # Deliberate: draws = losses
max_game_turns = 200        # Truncation limit
num_battles = 10000         # Total battles before env stops

# Optional:
# opponent_type = "max_damage"  # For play_mode = "single"
# team_dir = "vendor/pokechamp/poke_env/data/static/teams/gen9ou"
# max_concurrent_battles = 8    # Throttle concurrent games per worker (default: 8)

[inference]
# Must be present — prime-rl starts vLLM automatically

[inference.server]
port = 8001                 # Must NOT be 8000 (Showdown uses 8000)

[trainer.model.ac]
freq = 1                    # Gradient checkpointing (prevents OOM on 1 GPU)

[trainer.optim]
lr = 3e-6
betas1 = 0.9                # AdamW beta1 (default for all pokemon configs)
betas2 = 0.9                # AdamW beta2 (lower than default 0.999 — reduces momentum lag)

[ckpt]
interval = 10               # Checkpoint every N steps
```

### Critical Settings

**`trajectory_strategy`** — Controls how game turns map to TrainingSamples. Two options:
- `"branching"` — Each turn becomes a separate TrainingSample. Used for self-play or when sequences must stay short.
- `"interleaved"` — The full conversation becomes one TrainingSample via `interleave_rollout`. Requires `interleaved=true` in env args and a large `seq_len` (32768). See [Interleaved Trajectory Mode](#interleaved-trajectory-mode) below.

**`rollouts_per_example = batch_size`** — All games form one GRPO group. Since every game starts from a random state (random teams, random matchup), no within-game grouping makes sense. Batch mean is the correct baseline.

**`[inference]` section** — Must exist even if empty. prime-rl only starts the vLLM inference server when this section is present.

**`inference.server.port ≠ 8000`** — Port 8000 is reserved for Showdown. The inference server must use a different port (8001 recommended).

**`max_tokens = 800`** — Must be large enough for the CoT reasoning (3 sentences) plus the JSON action. 400 tokens is too small; the model hits the limit before producing JSON. 800 works well.

### Environment Args Reference

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `battle_format` | str | required | Showdown format (gen9ou, gen9randombattle, etc.) |
| `play_mode` | str | `"single"` | `"self_play"` (both train) or `"single"` (one trains) |
| `port` | int | `8000` | Showdown server port |
| `server_host` | str | `"localhost"` | Showdown server hostname (for multi-node) |
| `observation_format` | str | `"full_obs_cot"` | Prompt format: `"full_obs_cot"`, `"pokechamp_io"`, or `"simple"` |
| `opponent_type` | str | `"random"` | Opponent to train against (see Opponent Types below) |
| `reward_win` | float | `1.0` | Reward for winning |
| `reward_loss` | float | `0.0` | Reward for losing |
| `reward_draw` | float | `0.0` | Reward for draw (deliberate: same as loss) |
| `max_game_turns` | int | `200` | Max turns before truncation |
| `num_battles` | int | `1000` | Total battles before env exhausts |
| `team_dir` | str | None | Path to team .txt files (relative to pokemon-rl root) |
| `max_concurrent_battles` | int | `8` | Max battles running concurrently per worker process |
| `score_rollouts` | bool | `True` | Whether prime-rl scores rollouts (keep True) |
| `interleaved` | bool | `False` | Enable two-phase interleaved generation (reasoning + extraction) |
| `reasoning_tokens` | int | `512` | Token budget for reasoning phase per turn (interleaved only) |
| `extraction_tokens` | int | `50` | Token budget for JSON extraction phase (interleaved only) |

### Opponent Types

The opponent registry (`src/pokemon_rl/opponents.py`) routes each opponent type automatically:

| Type | Kind | Description |
|------|------|-------------|
| `random` | direct | Random legal action (in-process) |
| `max_damage` | direct | Always picks highest base power move (in-process) |
| `abyssal` | direct | Strong heuristic with type/switch logic (in-process) |
| `kakuna` | external | Metamon's best RL agent, 7.8M self-play battles (separate process) |

**Direct opponents** run in-process as poke-env Players. No setup needed — just set `opponent_type` in the config.

**External opponents** run as separate processes connecting to the same Showdown server. `launch_rl.sh` auto-starts one Kakuna instance per game in the batch (matching `batch_size`) for full parallelism. Each instance plays one game at a time; with N instances, N games run concurrently.

The system serializes ladder matching within each worker process so that concurrent rollouts don't match each other — they match successive Kakuna instances instead. Kakuna is small (~1.6 GB per instance, 142M params) and runs on the GPUs not used by vLLM/trainer (typically GPUs 2-3).

**Important**: For external opponents, prime-rl must use a single env worker process (`workers_per_env = 1`, the default). The serialized matching only works within one process.

To add a new external opponent, add an entry to the `_REGISTRY` in `opponents.py`.

### Reward System

- **Self-play**: Both players' turns become TrainingSamples. Winner turns get `reward_win`, loser turns get `reward_loss`. Advantages are pre-set using config-derived baseline: `(reward_win + reward_loss) / 2`.
- **vs heuristic**: Only the trained agent's turns become TrainingSamples.
- **Draws = losses** (`reward_draw = 0.0`): Deliberate design choice. Prevents the model from learning to stall games.
- **Parse failures** → random fallback action. The fallback is random (not max-damage) to prevent reward hacking: a model that always outputs garbage would otherwise get the strongest heuristic move for free.

See [rewards.md](rewards.md) for the full reward system documentation.

## Launching Training

### Quick Start

```bash
# Inside the prime-rl environment (Showdown must be running):
bash /path/to/pokemon-rl/scripts/launch_rl.sh

# With a specific config:
bash scripts/launch_rl.sh configs/pokemon/rl_interleaved.toml
```

The `launch_rl.sh` script handles everything: starts Showdown, installs pokemon-rl, runs training.

### Manual Launch

```bash
# 1. Start Showdown
node vendor/pokechamp/pokemon-showdown/pokemon-showdown start --no-security --port 8000 &

# 2. Activate prime-rl venv
cd /path/to/prime-rl
source .venv/bin/activate

# 3. Install pokemon-rl (once)
pip install -e /path/to/pokemon-rl/vendor/pokechamp
pip install -e /path/to/pokemon-rl

# 4. Symlink poke_env data (pokechamp uses relative paths)
ln -sfn /path/to/pokemon-rl/vendor/pokechamp/poke_env poke_env

# 5. Run
rl @ /path/to/pokemon-rl/configs/pokemon/rl_test.toml
```

### GPU Distribution

prime-rl distributes across GPUs:

- **Inference (vLLM)**: Serves model completions. Supports tensor-parallel (TP, splits one model across GPUs) and data-parallel (DP, runs independent replicas). DP doubles throughput for the same model size.
- **Trainer**: Runs GRPO weight updates. Uses FSDP2 for model sharding if multiple GPUs.
- **External opponents** (Kakuna): Run on remaining GPUs. Tiny models (~1.6 GB each).

**Single-node layouts (4× A100-80GB):**

```
Self-play / heuristic:
  GPU 0-2: vLLM inference (TP=3 or DP=3)
  GPU 3:   Trainer

Vs Kakuna:
  GPU 0-1: vLLM inference (DP=2)
  GPU 2:   Trainer
  GPU 3:   Kakuna instances (batch_size instances, ~1.6 GB each)
```

```bash
# Explicit GPU assignment:
rl @ config.toml --inference_gpu_ids 0,1 --trainer_gpu_ids 2

# Or via launch_rl.sh env vars:
INFERENCE_GPUS="0,1" TRAINER_GPUS="2" bash scripts/launch_rl.sh
```

### Concurrency Throttle

`max_concurrent_battles` (default: 8) limits how many games run simultaneously within a single worker process. With `batch_size=16` and `max_concurrent_battles=8`, 8 games run at once and the remaining 8 are queued. This controls memory and CPU pressure from Showdown connections.

For external opponents like Kakuna, set `max_concurrent_battles` to match the number of Kakuna instances (which defaults to `batch_size`).

```toml
[orchestrator.env.args]
max_concurrent_battles = 8   # At most 8 games running at once
```

### Cluster Launch

For HPC clusters (SLURM, containers), create cluster-specific scripts in `local_scripts/` (gitignored). See `local_scripts/README.md` for examples.

## Multi-Node Training

For large batch sizes (128+), distribute inference across two nodes:

```
2-node layout (4× A100-80GB per node):
┌─────────────────────────────┐  ┌─────────────────────────────┐
│ Node 1 (inference only)     │  │ Node 2 (rl + trainer)       │
│   GPU 0-3: vLLM DP=4       │  │   GPU 0-1: vLLM DP=2        │
│   Port 8001, host=0.0.0.0  │  │   GPU 2-3: Trainer (FSDP)   │
│                             │  │   Showdown + Orchestrator    │
└─────────────────────────────┘  └─────────────────────────────┘
        Total inference: DP=6 (4+2) — 6× throughput
```

### Config Setup

Multi-node configs use `__INFERENCE_NODE__` as a placeholder for Node 1's hostname. The launch script resolves it at runtime:

```toml
[orchestrator.client]
# Two inference servers: Node 1 (DP=4) + local Node 2 (DP=2)
base_url = ["http://__INFERENCE_NODE__:8001/v1", "http://localhost:8001/v1"]
```

A separate `inference_node1.toml` config runs the standalone inference server on Node 1 (DP=4, host=0.0.0.0 so Node 2 can reach it).

### Launch

```bash
# Submit 2-node batch job with any config using __INFERENCE_NODE__:
sbatch local_scripts/launch_2node_prod.sh configs/pokemon/rl_vs_abyssal_600_4x4.toml

# Or the abyssal-specific launcher:
sbatch local_scripts/launch_abyssal_2node.sh
```

### Requirements

- Both containers must use `--net=host` (the pokechamp poke-env fork patches ws:// for non-localhost)
- `setup_node_hostnet.sh` creates host-networking containers
- Inference server must bind to `0.0.0.0` (not localhost) so the other node can connect

## Performance Optimizations

### `_copy_battle=False` (11x speedup)

The `pokechamp_io` prompt builder calls `LocalSim(battle, ...)` for damage calculations. By default, `LocalSim.__init__` does `deepcopy(battle)`, which is extremely expensive for rich Battle objects. Passing `_copy_battle=False` (in `translator.py`) eliminates this copy.

This was the single biggest performance bottleneck: **69 min/step down to 6.4 min/step** with batch_size=128.

**Safety**: Requires the mutation fix in `vendor/pokechamp/pokechamp/prompts.py` — three locations where `pokemon._max_hp = 1` was changed to use a local variable `max_hp = pokemon.max_hp if pokemon.max_hp != 0 else 1` to prevent battle state corruption.

### `asyncio.to_thread` in `get_prompt_messages` (not yet implemented)

Offloading `_build_agent_prompt` (pokechamp's CPU-bound damage calculations) to a thread pool via `asyncio.to_thread()` would allow other battles to progress concurrently instead of blocking on one battle's prompt construction. This optimization was considered but is not currently implemented.

### Step Time Estimates

| Setup | Batch Size | Strategy | Step Time |
|-------|-----------|----------|-----------|
| 1-node, DP=3 (self-play) | 16 | branching | ~3-4 min |
| 1-node, DP=3 (vs heuristic) | 16 | branching | ~2-3 min |
| 2-node, DP=6 (vs abyssal) | 128 | branching | ~6-7 min |
| 2-node (interleaved vs abyssal) | 64 | interleaved | ~10-25 min |

## Monitoring

### wandb

All training runs log to Weights & Biases. Key metrics:
- `reward_mean` — should be ~0.5 for self-play (balanced wins/losses)
- `wins`, `losses`, `draws` — separate game outcome counters
- `won` — 1=win, 0=loss, -1=draw/crash/truncation (legacy, kept for backward compat)
- `loss` — GRPO policy loss
- `entropy` — action distribution entropy
- `grad_norm` — gradient norm

### Rollout Inspection

Rollouts are saved in `outputs/run_default/rollouts/step_N/rollouts.bin` (msgpack format). Each file contains TrainingSamples with prompt IDs, completion IDs, advantages, and rewards.

### Probe File

Set `POKEMON_RL_PROBE_PATH` to capture the first turn's full prompt and response:
```bash
POKEMON_RL_PROBE_PATH=/tmp/probe.txt rl @ config.toml
# Then inspect: cat /tmp/probe.txt
```

## Architecture Notes

- **Branching trajectory**: Each turn is a separate TrainingSample. A 30-turn game produces 30 samples (or 60 in self-play, since both players' turns are included).
- **Interleaved trajectory**: The full conversation (all turns) becomes one TrainingSample via `interleave_rollout`. Two LLM calls per turn (reasoning + extraction) are concatenated. See [Interleaved Trajectory Mode](#interleaved-trajectory-mode) below.
- **GRPO normalization**: Batch-level. All games in a batch form one GRPO group. Advantages are normalized across the batch, with self-play baseline pre-setting.
- **Weight broadcast**: After each training step, updated weights are broadcast to the inference server via filesystem checkpoint.
- **Showdown lifecycle**: One Showdown server runs for the entire training. Battles are created/destroyed within it. No restart needed between steps.

## Interleaved Trajectory Mode

Interleaved mode replaces the single-turn "branching" approach with a full-conversation approach. Instead of each turn producing one TrainingSample, the entire game conversation accumulates and becomes one TrainingSample via `interleave_rollout`.

### How It Works

Each game turn produces **two LLM calls**:

1. **Reasoning phase** (up to `reasoning_tokens`, default 512) -- the model thinks through the battle state with chain-of-thought reasoning.
2. **Extraction phase** (up to `extraction_tokens`, default 50) -- the model outputs a JSON action based on its reasoning.

Turn 1 gets a comprehensive prompt (~2600 tokens) with full battle context. Subsequent turns get a lighter prompt (~500 tokens) that relies on the accumulated conversation history. The per-agent conversation accumulates across turns, supporting both single-agent and self-play modes.

vLLM's Automatic Prefix Caching reuses KV entries across turns within the same conversation, giving O(N) compute instead of O(N^2) for N turns.

### Branching vs Interleaved Comparison

| Aspect | Branching | Interleaved |
|--------|-----------|-------------|
| TrainingSamples per game | 1 per turn (~30) | 1 per game |
| LLM calls per turn | 1 | 2 (reasoning + extraction) |
| Prompt per turn | Full (~2600 tokens) | Turn 1 full, then light (~500) |
| Sequence length | Short (~800-2K per sample) | Long (~17-20K per game) |
| Required `seq_len` | 4096-8192 | 32768 |
| KV cache efficiency | No reuse across turns | Prefix caching (O(N) vs O(N^2)) |
| Self-play support | Yes (primary) | Yes (via per-agent accumulation) |
| `trajectory_strategy` | `"branching"` | `"interleaved"` |

### Config Reference

Interleaved mode requires three settings in `[orchestrator.env.args]`:

```toml
[orchestrator]
trajectory_strategy = "interleaved"   # Activates interleave_rollout

[orchestrator.env.args]
interleaved = true                    # Two-phase generation in the env
reasoning_tokens = 512                # Reasoning phase budget per turn
extraction_tokens = 50                # Extraction phase budget per turn
```

Additionally:
- `seq_len = 32768` -- conversations accumulate to 17-20K tokens; 32K provides headroom.
- `[trainer.model] attn = "flash_attention_2"` -- recommended for long sequences.
- `observation_format = "full_obs_cot"` -- the interleaved-aware prompt format.

See `configs/pokemon/rl_interleaved.toml` for the canonical production config.

### When to Use Each Strategy

**Use interleaved** when:
- Training single-agent vs heuristic (the primary use case)
- You want the model to learn from full game context, not isolated turns
- You have sufficient `seq_len` budget (32K)
- You want the entropy-decline signal that indicates actual learning

**Use branching** when:
- Running self-play (both players train, interleaved works but branching is simpler)
- You need short sequences (limited GPU memory)
- Integration testing (faster iteration)

### Token Budget Analysis

A typical 25-turn game in interleaved mode:
- Turn 1 prompt: ~2600 tokens
- Turn 1 reasoning: ~512 tokens
- Turn 1 extraction: ~50 tokens
- Turns 2-25 prompt: ~500 tokens each = ~12000
- Turns 2-25 reasoning: ~512 each = ~12288
- Turns 2-25 extraction: ~50 each = ~1200
- **Total: ~17-20K tokens per sample** (fits comfortably in 32K seq_len)

Peak GPU memory: ~31.2 GiB (well within 80 GiB A100).

### Monitoring (W&B)

Key differences from branching runs:
- **Entropy**: Should decline over training (e.g., 0.837 to 0.763). In branching mode, entropy often stays flat (~0.92) because isolated turns don't provide enough learning signal.
- **Sequence length**: Expect 17-19.5K tokens/sample. If it grows beyond 25K, the model may be generating excessively long reasoning.
- **Step time**: Longer than branching (10-25 min trainer, 7-12 min orchestrator) due to full-conversation processing.
- **Grad norm**: Comparable to branching (~0.076).
