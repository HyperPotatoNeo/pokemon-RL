# TODO

## Done: High-Level Skeleton

- [x] Project structure + pyproject.toml
- [x] Layer 1: ShowdownEngine (start/stop/health_check)
- [x] Layer 2: BattleAdapter (full-battle mode with callback player)
- [x] Layer 3: StateTranslator (simple + pokechamp_io formats)
- [x] Layer 4: PokemonBattleEnv (MultiTurnEnv skeleton)
- [x] Unit tests + integration tests (37 total)

## Done: Turn-by-Turn Control + Self-Play

- [x] ControllablePlayer (queue-based external control)
- [x] BattleManager (heuristic + selfplay modes)
- [x] Self-play with force-switch handling (sequential API)
- [x] Enhanced PokemonBattleEnv (full_battle + turn_by_turn modes)
- [x] Opponent factory (random, max_damage, callback, controllable)
- [x] TrajectoryLogger (JSONL battle data collection)

## Done: Adversarial Review + Fixes

- [x] Configurable rewards (reward_win, reward_loss, reward_draw constructor args)
- [x] Step-level reward callback (step_reward_fn)
- [x] Centralized reward computation (_compute_terminal_reward, _assign_rewards)
- [x] Random fallback action (prevents reward hacking via max-power heuristic)
- [x] BattleManager.close() + async context manager (resource cleanup)
- [x] Zombie loop prevention (consecutive timeout → forfeit)
- [x] Atomic JSONL writes, socket leak fix, PIPE → DEVNULL
- [x] 149 tests (122 unit + 27 integration)

## Done: Verifiers Integration Plan

- [x] Full design plan (3 rounds adversarial review) → `PHASE_VERIFIERS_PLAN.md`

## Done: Verifiers Integration (Layer 4) — Implementation

- [x] PokemonBattleEnv(vf.MultiTurnEnv) with hook overrides
- [x] PokemonRubric (passthrough reward + metrics, explicitly registered)
- [x] _AgentContext passive dataclass
- [x] _build_agent_prompt (fresh mode, extensible for episodic/windowed)
- [x] _assign_rewards (single override: rewards + config-derived advantage baseline)
- [x] @vf.stop game_over, @vf.cleanup cleanup_battle
- [x] Error boundary: all BattleManager/translator calls wrapped → vf.Error
- [x] load_environment() entry point in __init__.py
- [x] StateTranslator.extract_completion_text (Messages → str)
- [x] Unit tests T1-T13 (216 passing)
- [x] Integration tests T14-T20 (16 passing, +12 old integration)
- [x] GPU tests G1-G4 (6 passing, including cross-node)
- [x] Documentation updated (all docs, README, CLUSTER.md)
- [x] pokechamp submodule at vendor/pokechamp (pokechamp_io format works)

## Next: GPU Smoke Tests

- [ ] run_selfplay.sh: LLM self-play on GPU node via vLLM
- [ ] run_crossnode.sh: cross-node battle (2 GPU nodes)
- [ ] LLM vs heuristic (vLLM + Showdown)

## Future

- [ ] Opponent curriculum (heuristic → self-play → frozen checkpoint)
- [ ] Shaped step_reward_fn implementations (damage dealt, pokemon fainted)
- [ ] Team pool / format selection
- [ ] Elo tracking
- [ ] Multi-node deployment scripts
- [ ] pokechamp_io format for all generations
- [ ] Replay logging for analysis
- [ ] Human player interface (websocket or terminal)
- [ ] Online tournament connection (PokéAgent challenge)
