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

Each training step: orchestrator plays N battles → collects TrainingSamples (one per turn) → trainer runs GRPO update → weights broadcast to inference server → repeat.

## Prompt Construction

### pokechamp_io Format (Recommended)

The `pokechamp_io` observation format produces rich prompts with damage calculations, type effectiveness, speed comparisons, and move effects. It is built from three components:

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

| Config | Play Mode | Description |
|--------|-----------|-------------|
| `rl_test.toml` | self_play | Integration testing — 3 steps, batch_size=4, gen9randombattle |
| `rl_selfplay.toml` | self_play | Production self-play — 100 steps, batch_size=16, gen9ou |
| `rl_vs_heuristic.toml` | single | Production vs heuristic bot — 100 steps, batch_size=16, gen9ou (default: max_damage, also: random, abyssal) |

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
trajectory_strategy = "branching"   # CRITICAL — see below

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
observation_format = "pokechamp_io"  # "pokechamp_io" or "simple"
reward_win = 1.0
reward_loss = 0.0
reward_draw = 0.0           # Deliberate: draws = losses
max_game_turns = 200        # Truncation limit
num_battles = 10000         # Total battles before env stops

# Optional:
# opponent_type = "max_damage"  # For play_mode = "single"
# team_dir = "vendor/pokechamp/poke_env/data/static/teams/gen9ou"

[inference]
# Must be present — prime-rl starts vLLM automatically

[inference.server]
port = 8001                 # Must NOT be 8000 (Showdown uses 8000)

[trainer.model.ac]
freq = 1                    # Gradient checkpointing (prevents OOM on 1 GPU)

[trainer.optim]
lr = 3e-6

[ckpt]
interval = 10               # Checkpoint every N steps
```

### Critical Settings

**`trajectory_strategy = "branching"`** — Each turn becomes a separate TrainingSample. Without this, prime-rl concatenates all turns into one sequence (100+ turns × 800 tokens = 80K+ tokens), which exceeds seq_len and corrupts training. This is the single most important config setting.

**`rollouts_per_example = batch_size`** — All games form one GRPO group. Since every game starts from a random state (random teams, random matchup), no within-game grouping makes sense. Batch mean is the correct baseline.

**`[inference]` section** — Must exist even if empty. prime-rl only starts the vLLM inference server when this section is present.

**`inference.server.port ≠ 8000`** — Port 8000 is reserved for Showdown. The inference server must use a different port (8001 recommended).

**`max_tokens = 800`** — Must be large enough for the CoT reasoning (3 sentences) plus the JSON action. 400 tokens is too small; the model hits the limit before producing JSON. 800 works well.

### Environment Args Reference

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `battle_format` | str | required | Showdown format (gen9ou, gen9randombattle, etc.) |
| `play_mode` | str | `"self_play"` | `"self_play"` (both train) or `"single"` (one trains) |
| `port` | int | `8000` | Showdown server port |
| `server_host` | str | `"localhost"` | Showdown server hostname (for multi-node) |
| `observation_format` | str | `"pokechamp_io"` | Prompt format: `"pokechamp_io"` or `"simple"` |
| `opponent_type` | str | `"random"` | Opponent to train against (see Opponent Types below) |
| `reward_win` | float | `1.0` | Reward for winning |
| `reward_loss` | float | `0.0` | Reward for losing |
| `reward_draw` | float | `0.0` | Reward for draw (deliberate: same as loss) |
| `max_game_turns` | int | `200` | Max turns before truncation |
| `num_battles` | int | `1000` | Total battles before env exhausts |
| `team_dir` | str | None | Path to team .txt files (relative to pokemon-rl root) |
| `score_rollouts` | bool | `True` | Whether prime-rl scores rollouts (keep True) |

### Opponent Types

The opponent registry (`src/pokemon_rl/opponents.py`) routes each opponent type automatically:

| Type | Kind | Description |
|------|------|-------------|
| `random` | direct | Random legal action (in-process) |
| `max_damage` | direct | Always picks highest base power move (in-process) |
| `abyssal` | direct | Strong heuristic with type/switch logic (in-process) |
| `kakuna` | external | Metamon's best RL agent, 7.8M self-play battles (separate process) |

**Direct opponents** run in-process as poke-env Players. No setup needed — just set `opponent_type` in the config.

**External opponents** run as separate processes connecting to the same Showdown server. The system serializes ladder matching within each worker process so that concurrent rollouts don't match each other. `launch_rl.sh` auto-starts Kakuna if `opponent_type = "kakuna"` and the launcher script exists at `local_scripts/launch_kakuna_opponent.sh`.

**Important**: For external opponents, prime-rl must use a single env worker process (`workers_per_env = 1`, the default). The serialized matching only works within one process. Multiple worker processes would each get their own semaphore and could match against each other.

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
bash scripts/launch_rl.sh configs/pokemon/rl_vs_heuristic.toml
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
rl @ /path/to/pokemon-rl/configs/pokemon/rl_selfplay.toml
```

### GPU Assignment

```bash
# Explicit GPU assignment:
rl @ config.toml --inference_gpu_ids 0 1 2 --trainer_gpu_ids 3

# Or via launch_rl.sh env vars:
INFERENCE_GPUS="0 1 2" TRAINER_GPUS="3" bash scripts/launch_rl.sh
```

### Cluster Launch

For HPC clusters (SLURM, containers), create cluster-specific scripts in `local_scripts/` (gitignored). See `local_scripts/README.md` for examples.

## Monitoring

### wandb

All training runs log to Weights & Biases. Key metrics:
- `reward_mean` — should be ~0.5 for self-play (balanced wins/losses)
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
- **GRPO normalization**: Batch-level. All games in a batch form one GRPO group. Advantages are normalized across the batch, with self-play baseline pre-setting.
- **Weight broadcast**: After each training step, updated weights are broadcast to the inference server via filesystem checkpoint.
- **Showdown lifecycle**: One Showdown server runs for the entire training. Battles are created/destroyed within it. No restart needed between steps.
