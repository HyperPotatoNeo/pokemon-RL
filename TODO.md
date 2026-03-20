# TODO

## Current: High-Level Skeleton

- [x] Project structure + pyproject.toml
- [x] Layer 1: ShowdownEngine (start/stop/health_check)
- [x] Layer 2: BattleAdapter (full-battle mode with callback player)
- [x] Layer 3: StateTranslator (simple + pokechamp_io formats)
- [x] Layer 4: PokemonBattleEnv (MultiTurnEnv skeleton)
- [x] Unit tests (no external deps)
- [x] Integration tests (Showdown + poke-env)
- [x] Scripts (allocate, setup, run_tests)
- [ ] Run unit tests on login node
- [ ] Run integration tests on compute node
- [ ] Fix any issues found during testing

## Next: Turn-by-Turn Control (Layer 2)

- [ ] ControllablePlayer (choose_move blocks until external action)
- [ ] BattleAdapter.start_battle() / submit_actions() / get_winner()
- [ ] Thread-safe bridge between poke-env event loop and env hooks
- [ ] Tests for turn-by-turn control

## Next: Verifiers Integration (Layer 4)

- [ ] Inherit from vf.MultiTurnEnv
- [ ] Integrate with prime-rl orchestrator
- [ ] branching trajectory strategy
- [ ] Passthrough rubric
- [ ] load_environment() discovery
- [ ] Tests with mock vLLM client

## Next: Self-Play Mode

- [ ] Two ControllablePlayers per battle
- [ ] Alternating trajectory steps (P1/P2)
- [ ] Both players' trajectories captured
- [ ] Simultaneous action resolution
- [ ] Per-player reward assignment

## Future

- [ ] Opponent curriculum (heuristic → self-play → frozen checkpoint)
- [ ] Shaped rewards (damage, pokemon fainted)
- [ ] Team pool / format selection
- [ ] Elo tracking
- [ ] Multi-node deployment scripts
- [ ] pokechamp_io format for all generations
- [ ] Replay logging for analysis
