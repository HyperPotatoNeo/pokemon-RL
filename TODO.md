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

## Done: Phase 5 RL Training Integration — Plan

- [x] Comprehensive plan: `PHASE5_RL_PLAN.md` (v2, 2 adversarial reviews + consistency check)
- [x] Sequence length analysis (4096 for testing, 8192 for production)
- [x] TOML config design: self-play, heuristic, test configs
- [x] Multi-node architecture: 1-node (3 inf + 1 train) and 2-node layouts
- [x] Advantage computation analysis (pre-set for self-play, batch-level for single-agent)
- [x] Team handling design (team_fn callable, random_team_pool factory)
- [x] Kakuna opponent integration via ladder + separate metamon process
- [x] Testing plan: T1-T7b (unit), T8-T12 (battle flow), T16-T19 (verifiers), T20-T21 (RL loop), T22-T24 (multi-node)

## Current: Phase 5A — Implementation (Test Cases)

- [ ] Write Phase 5 test cases per PHASE5_TEST_AGENT_INSTRUCTIONS.md
- [ ] Write testing protocol for implementation agent

## Next: Phase 5B — Implementation (Code)

- [ ] Team handling: team_fn + random_team_pool in PokemonBattleEnv (~100 lines)
- [ ] Ladder mode: BattleManager.start_battle_ladder() (~15 lines)
- [ ] AbyssalPlayer in opponent factory (~10 lines)
- [ ] LocalSim format fix (~3 lines)
- [ ] Arg validation (warn on unrecognized kwargs)
- [ ] TOML configs: selfplay, heuristic, test, 2-node variant
- [ ] Launch scripts: 1-node, 2-node, Kakuna launcher

## Next: Phase 5C — Testing + Validation

- [ ] Run unit tests (T1-T7b) on login node
- [ ] Run battle flow tests (T8-T12) on compute node
- [ ] Run verifiers pipeline tests (T16-T19) on GPU node
- [ ] Run full RL loop tests (T20) on GPU node + Showdown
- [ ] Run multi-node tests (T22-T24) on 2 GPU nodes
- [ ] Run Kakuna opponent tests on GPU node + metamon process
- [ ] 10-step training validation run

## Future: Phase 6 — Population Training

- [ ] Opponent curriculum (heuristic → self-play → frozen checkpoint → Kakuna)
- [ ] Shaped step_reward_fn (damage dealt, pokemon fainted)
- [ ] Team pool expansion / format selection
- [ ] Elo tracking
- [ ] Deterministic Kakuna pairing (send_challenges/accept_challenges)
- [ ] Multi-opponent configs (population of opponents per training step)
- [ ] pokechamp_io format for all generations
- [ ] Replay logging for analysis
- [ ] Human player interface (websocket or terminal)
- [ ] Online tournament connection (PokéAgent challenge)
