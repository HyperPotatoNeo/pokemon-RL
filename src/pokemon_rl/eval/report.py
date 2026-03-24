"""Results aggregation and reporting for pokemon-rl eval."""

from __future__ import annotations

import json
import math
from pathlib import Path


def compute_stats(states: list[dict]) -> dict:
    """Compute win/loss/draw statistics from rollout states.

    Args:
        states: List of verifiers State dicts with "metrics" containing
            "won", "game_turns", "parse_failures".

    Returns:
        Dict with win_rate, loss_rate, draw_rate, stderr, avg_game_turns,
        avg_parse_failures, total.
    """
    if not states:
        return {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "total": 0,
            "win_rate": 0.0,
            "loss_rate": 0.0,
            "draw_rate": 0.0,
            "win_rate_stderr": 0.0,
            "avg_game_turns": 0.0,
            "avg_parse_failures": 0.0,
        }

    wins = 0
    losses = 0
    draws = 0
    game_turns_sum = 0
    parse_failures_sum = 0

    for state in states:
        metrics = state.get("metrics", {})
        won = metrics.get("won", -1)
        if won == 1:
            wins += 1
        elif won == 0:
            losses += 1
        else:
            draws += 1
        game_turns_sum += metrics.get("game_turns", 0)
        parse_failures_sum += metrics.get("parse_failures", 0)

    total = len(states)
    win_rate = wins / total
    loss_rate = losses / total
    draw_rate = draws / total

    # Standard error for a proportion: sqrt(p * (1 - p) / n)
    win_rate_stderr = math.sqrt(win_rate * (1 - win_rate) / total) if total > 0 else 0.0

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total": total,
        "win_rate": win_rate,
        "loss_rate": loss_rate,
        "draw_rate": draw_rate,
        "win_rate_stderr": win_rate_stderr,
        "avg_game_turns": game_turns_sum / total,
        "avg_parse_failures": parse_failures_sum / total,
    }


def save_results(
    states: list[dict],
    opponent_name: str,
    output_dir: str,
    node_rank: int = 0,
) -> Path:
    """Save per-rollout results to JSONL.

    Returns the path to the saved file.
    """
    output_path = Path(output_dir) / opponent_name
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"results_{node_rank}.jsonl" if node_rank > 0 else "results.jsonl"
    filepath = output_path / filename

    with open(filepath, "w") as f:
        for i, state in enumerate(states):
            metrics = state.get("metrics", {})
            row = {
                "example_id": state.get("example_id", i),
                "opponent": opponent_name,
                "reward": state.get("reward"),
                "won": metrics.get("won", -1),
                "game_turns": metrics.get("game_turns", 0),
                "parse_failures": metrics.get("parse_failures", 0),
                "wins": metrics.get("wins", 0),
                "losses": metrics.get("losses", 0),
                "draws": metrics.get("draws", 0),
            }
            f.write(json.dumps(row) + "\n")

    return filepath


def generate_summary(
    all_results: dict[str, dict],
    output_dir: str,
    node_rank: int = 0,
) -> str:
    """Generate a summary table and save to summary.json.

    Args:
        all_results: {opponent_name: stats_dict}
        output_dir: Output directory
        node_rank: Node rank (for multi-node)

    Returns:
        Formatted summary string.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save JSON
    filename = f"summary_{node_rank}.json" if node_rank > 0 else "summary.json"
    with open(output_path / filename, "w") as f:
        json.dump(all_results, f, indent=2)

    # Format table
    lines = []
    header = f"{'Opponent':<25} {'Win%':>8} {'Loss%':>8} {'Draw%':>8} {'SE':>8} {'Turns':>8} {'N':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for opp_name, stats in all_results.items():
        wr = stats.get("win_rate", 0.0) * 100
        lr = stats.get("loss_rate", 0.0) * 100
        dr = stats.get("draw_rate", 0.0) * 100
        se = stats.get("win_rate_stderr", 0.0) * 100
        turns = stats.get("avg_game_turns", 0.0)
        n = stats.get("total", 0)
        lines.append(
            f"{opp_name:<25} {wr:>7.1f}% {lr:>7.1f}% {dr:>7.1f}% {se:>7.2f}% {turns:>7.1f} {n:>6d}"
        )

    summary = "\n".join(lines)
    return summary


def merge_node_results(output_dir: str, opponent_name: str, n_nodes: int) -> list[dict]:
    """Merge results from multiple nodes for a single opponent.

    Reads results.jsonl and results_{rank}.jsonl files.

    Returns:
        List of all result dicts across nodes.
    """
    results = []
    opp_dir = Path(output_dir) / opponent_name

    for rank in range(n_nodes):
        filename = "results.jsonl" if rank == 0 else f"results_{rank}.jsonl"
        filepath = opp_dir / filename
        if filepath.exists():
            with open(filepath) as f:
                for line in f:
                    if line.strip():
                        results.append(json.loads(line))

    return results
