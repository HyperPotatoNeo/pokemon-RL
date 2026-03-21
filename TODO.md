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

## Next: Verifiers Integration (Layer 4)

- [ ] Inherit from vf.MultiTurnEnv
- [ ] Integrate with prime-rl orchestrator
- [ ] Branching trajectory strategy
- [ ] Passthrough rubric (score_rollouts=True)
- [ ] load_environment() discovery
- [ ] Per-step GRPO advantage computation
- [ ] Tests with mock vLLM client

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
