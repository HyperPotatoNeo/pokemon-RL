# Testing

235 tests (207 unit + 28 integration). All tests follow the "no fall-through" philosophy.

## Test Philosophy: No Fall-Through Passes

Every test verifies both the **positive case** (correct behavior happens) AND the **negative case** (incorrect behavior does NOT happen).

Examples:
- Test that wins get reward 1.0 AND losses get reward 0.0 AND they are different
- Test that game_over=True returns None AND game_over=False does NOT return None
- Test that valid JSON parses AND invalid JSON returns None AND they are distinguishable
- Test that timeout fires when no action AND does NOT fire when action is fast

A test that only checks `assert result is not None` passes by accident if the function always returns something. A no-fall-through test checks `assert result == expected_value AND result != wrong_value`.

## Test Markers

```python
@pytest.mark.unit         # No external deps. Runs anywhere with .venv active.
@pytest.mark.integration  # Needs Showdown server + poke-env (compute node).
```

**Skip markers** in `tests/conftest.py`:
- `@requires_poke_env` — skips if poke-env not importable
- `@requires_pokechamp` — skips if pokechamp not importable
- `@requires_showdown` — skips if Showdown not running on SHOWDOWN_PORT
- `@requires_verifiers` — skips if verifiers framework not importable

## Running Tests

```bash
# Unit tests (no Showdown needed):
.venv/bin/python -m pytest -m unit -v

# All tests (Showdown must be running on SHOWDOWN_PORT):
bash scripts/run_tests.sh -v
bash scripts/run_tests.sh -m integration -v
```

## Test File Map

| File | Tests | What it covers |
|------|-------|---------------|
| `tests/test_engine.py` | 5 unit, 2 integration | ShowdownEngine init, health check, start/stop, external detection |
| `tests/test_adapter.py` | 2 unit, 4 integration | Imports, full battle with random/default actions, trajectory, multiple battles |
| `tests/test_players.py` | 12 unit, 6 integration | Atomic usernames, queue mechanics (state/action/sentinel/timeout), ControllablePlayer creation, opponent factory |
| `tests/test_battle.py` | 16 unit, 5 integration | BattleManager state machine guards, close/cleanup, exception propagation, sentinel handling, full heuristic + selfplay games, concurrent battles |
| `tests/test_translator.py` | 21 unit, 2 integration | parse_action (move/switch/dynamax/nested JSON/garbage), _extract_last_json, fallback action randomness, simple prompt structure, format validation, pokechamp_io prompts |
| `tests/test_env.py` | 33 unit, 8 integration | State machine, reward computation (win/loss/draw/truncation/custom/selfplay), step rewards, parse failure tracking, manager cleanup, full game loops (heuristic + selfplay), hooks integration, trajectory integrity |
| `tests/test_hooks.py` | 23 unit | Hooks cycle (heuristic + selfplay), StrictMockSelfplayManager contract enforcement, setup_state type verification, advance buffering, standalone contract |
| `tests/test_phase4_unit.py` | 88 unit | Phase 4 verifiers integration: PokemonRubric, _AgentContext, advantage pre-setting, game_over/@vf.stop, cleanup_battle/@vf.cleanup, env_response, conditional verifiers inheritance |
| `tests/test_phase4_integration.py` | 16 integration | Phase 4 end-to-end: full games through verifiers pipeline, rubric scoring, advantage flow, self-play with verifiers |
| `tests/test_data.py` | 7 unit | TrajectoryLogger create/roundtrip/multi-line/step, concurrent writes (4 threads) |

## Mock Patterns

### MockBattle (test_translator.py, test_hooks.py)

Minimal mock that works without poke-env. Key: it must NOT be a tuple (tests verify `state["_agents"][idx].battle` is not a tuple — a real bug that was caught).

```python
class MockBattle:
    def __init__(self, name="mock", turn=1, moves=None, switches=None):
        self.name = name
        self.turn = turn
        self.available_moves = moves or [MockMove()]
        self.available_switches = switches or []
        self.force_switch = False
```

### StrictMockSelfplayManager (test_hooks.py)

**Enforces the BattleManager API contract**. If `get_pending_selfplay_states()` is called before all actions are submitted, it raises `AssertionError` with a diagnostic message. This catches the hooks buffering bug (submitting only P1's action then calling get_pending) without deadlocking.

```python
async def get_pending_selfplay_states(self):
    missing = self._expected - self._received
    assert not missing, f"get_pending called before all actions submitted! Missing: {missing}"
```

### Real poke-env Types (test_translator.py)

For tests that need `BattleOrder.message` isinstance checks to pass, use real poke-env types with lazy loading:

```python
def make_move(move_id, base_power=80):
    Move, _ = _load_poke_env_types()  # Lazy import
    m = Move(move_id, gen=1)
    m._base_power_override = base_power
    return m
```

These tests are in `@requires_poke_env` classes so they skip on login nodes.

### TestExtractLastJson (test_translator.py)

Pure-Python tests for `_extract_last_json` that need NO poke-env. In their own class without `@requires_poke_env` so they run everywhere, including login nodes.

## Adding New Tests

1. **Unit tests**: Use mock objects. No poke-env imports at module level.
2. **Integration tests**: Add `@pytest.mark.integration` + `@requires_poke_env` + `@requires_showdown`.
3. **Both markers**: Every test must have `@pytest.mark.unit` or `@pytest.mark.integration`.
4. **No fall-through**: Assert positive AND negative cases.
5. **Deferred imports**: Inside test functions, not module level. Exception: classes with `@requires_poke_env`.
