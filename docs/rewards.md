# Reward System

All reward computation is centralized in two methods in `env.py`. Every code path that assigns rewards calls these methods — there are no hardcoded reward values elsewhere.

## Terminal Rewards

Configurable via constructor arguments:

```python
PokemonBattleEnv(
    reward_win=1.0,    # Agent's team KOs opponent
    reward_loss=0.0,   # Agent's team KO'd
    reward_draw=0.0,   # Truncation, crash, disconnect, timeout
)
```

**Defaults** are simple binary: win=1, everything else=0. This is the standard starting point for GRPO.

### Single Source of Truth

`_compute_terminal_reward(won)` at `env.py:331-338`:

```python
def _compute_terminal_reward(self, won: bool | None) -> float:
    if won is None:
        return self.reward_draw
    return self.reward_win if won else self.reward_loss
```

`_assign_rewards(trajectory, won)` at `env.py:340-368` handles both modes:
- **Heuristic**: All steps get the same reward.
- **Self-play**: P1 and P2 get explicit per-player rewards (not `1 - reward`).

These two methods are called by:
- `render_completion` (hooks path)
- `run_standalone` (full-battle testing)
- `run_turn_by_turn` (turn-by-turn testing)
- `_run_selfplay_standalone` (self-play testing)

### Self-Play Reward Assignment

```python
if won is None:
    p1_reward = self.reward_draw
    p2_reward = self.reward_draw
elif won:  # P1 won
    p1_reward = self.reward_win
    p2_reward = self.reward_loss
else:      # P2 won
    p1_reward = self.reward_loss
    p2_reward = self.reward_win
```

This works for any reward scale:
- Binary: `reward_win=1, reward_loss=0` → P1=1.0, P2=0.0
- Symmetric: `reward_win=1, reward_loss=-1` → P1=1.0, P2=-1.0
- Custom: `reward_win=10, reward_loss=-10, reward_draw=0.5`

The old code used `1.0 - reward` which only works for [0, 1] scale.

### Game Outcomes

| Outcome | `state["won"]` | `state["truncated"]` | Default Reward |
|---------|----------------|---------------------|----------------|
| Win | `True` | `False` | `reward_win` (1.0) |
| Loss | `False` | `False` | `reward_loss` (0.0) |
| Truncation | `None` | `True` | `reward_draw` (0.0) |
| Crash/error | `None` | `False` | `reward_draw` (0.0) |

Truncation occurs when `state["turn"] >= max_game_turns` (default 200). The `truncated` flag is set in `get_prompt_messages` at `env.py:184-190`.

### Metrics

`render_completion` writes a metrics dict to `state["metrics"]`:

```python
{
    "won": 1 | 0 | -1,       # 1=win, 0=loss, -1=draw/crash/truncation
    "truncated": 0 | 1,
    "turns": int,
    "decision_count": int,
    "trajectory_length": int,
    "parse_failure_count": int,
}
```

## Step-Level Rewards

Optional per-step reward callback for credit assignment and reward shaping.

```python
PokemonBattleEnv(
    step_reward_fn=my_step_fn,  # or None (default)
)
```

### Callback Signature

```python
def step_reward_fn(
    battle_before: Battle,    # State the player saw when deciding
    battle_after: Battle | None,  # State after action resolved
    action: BattleOrder,      # The action taken
    player_idx: int,          # 0 or 1
) -> float:
```

- **Heuristic mode**: `battle_after` is the next Battle state (or `None` on game over).
- **Self-play mode**: `battle_after` is always `None` because the turn hasn't resolved yet (both players must act before Showdown advances).

### Storage

Step rewards are stored as `step["step_reward"]` — **separate from** `step["reward"]` (terminal reward). They are never mixed:

```python
step["reward"] = 1.0          # Terminal (from render_completion)
step["step_reward"] = 0.15    # Step-level (from step_reward_fn)
```

Training code decides how to combine them (sum, discount, separate advantage, etc.).

### Example: Faint Reward

```python
def faint_reward(before, after, action, player_idx):
    """Give +0.15 for each opponent pokemon KO'd this turn."""
    if after is None:
        return 0.0
    opp_fainted_before = sum(1 for p in before.opponent_team.values() if p.fainted)
    opp_fainted_after = sum(1 for p in after.opponent_team.values() if p.fainted)
    return (opp_fainted_after - opp_fainted_before) * 0.15
```

## Parse Failure Tracking

When the LLM outputs unparseable text, `add_trajectory_step` records:
- `step["parse_failed"] = True` on the trajectory step
- `state["parse_failure_count"]` incremented
- `metrics["parse_failure_count"]` in render_completion output

The fallback action is a **random legal action** (not max-power), preventing reward hacking where a model learns to output garbage to get the strongest heuristic action for free.

## Passthrough Rubric

`_passthrough_reward` at `env.py:36-42` is used for prime-rl's verifiers integration:

```python
def _passthrough_reward(state, **kwargs):
    return state.get("reward", 0.0)
```

This returns the pre-computed reward from `render_completion`, preventing verifiers' own rubric from overwriting pokemon-rl's rewards. It correctly handles negative rewards (no `or 0.0` that would zero them out).
