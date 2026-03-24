"""Eval configuration — parsed from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OpponentConfig:
    """Configuration for a single opponent in an eval run.

    Three opponent types:
        "heuristic" — in-process poke-env Player (abyssal, max_damage, random).
            Requires `heuristic` field.
        "metamon" — external RL agent via Showdown ladder (kakuna, alakazam).
            Requires `agent` field and `gpu_ids`.
        "llm" — LLM served via vLLM, uses LLMPlayer.
            Requires `model_name`, `base_url`, and `gpu_ids`.
    """

    name: str
    type: str  # "heuristic", "metamon", or "llm"
    # Heuristic-specific
    heuristic: str | None = None
    # Metamon-specific
    agent: str | None = None
    # Shared
    gpu_ids: list[int] | None = None
    # LLM-specific
    model_name: str | None = None
    base_url: str | None = None
    max_tokens: int = 800
    temperature: float = 1.0
    observation_format: str = "full_obs_cot"

    def validate(self) -> None:
        """Validate that required fields are present for each type."""
        if self.type == "heuristic":
            if not self.heuristic:
                raise ValueError(
                    f"Opponent '{self.name}': type='heuristic' requires "
                    f"'heuristic' field (e.g., 'abyssal', 'max_damage', 'random')"
                )
        elif self.type == "metamon":
            if not self.agent:
                raise ValueError(
                    f"Opponent '{self.name}': type='metamon' requires "
                    f"'agent' field (e.g., 'kakuna', 'alakazam')"
                )
        elif self.type == "llm":
            if not self.model_name:
                raise ValueError(
                    f"Opponent '{self.name}': type='llm' requires 'model_name'"
                )
            if not self.base_url:
                raise ValueError(
                    f"Opponent '{self.name}': type='llm' requires 'base_url'"
                )
        else:
            raise ValueError(
                f"Opponent '{self.name}': unknown type '{self.type}'. "
                f"Expected: 'heuristic', 'metamon', or 'llm'"
            )

    @property
    def opponent_type_for_env(self) -> str:
        """Return the opponent_type string for PokemonBattleEnv."""
        if self.type == "heuristic":
            return self.heuristic  # type: ignore[return-value]
        elif self.type == "metamon":
            return self.agent  # type: ignore[return-value]
        else:
            return "llm"


@dataclass
class PokemonEvalConfig:
    """Top-level eval configuration."""

    # Agent
    agent_model: str
    agent_base_url: str

    # Opponents
    opponents: list[OpponentConfig] = field(default_factory=list)

    # Battle settings
    battle_format: str = "gen9ou"
    n_battles_per_opp: int = 100
    max_concurrent_battles: int = 8
    max_game_turns: int = 200
    observation_format: str = "full_obs_cot"
    team_dir: str | None = None

    # Sampling for agent
    sampling_max_tokens: int = 800
    sampling_temperature: float = 1.0

    # Showdown
    showdown_port: int = 8000

    # Output
    output_dir: str = "eval_outputs"

    # Multi-node
    node_rank: int = 0
    n_nodes: int = 1

    def validate(self) -> None:
        """Validate the full config."""
        if not self.opponents:
            raise ValueError("At least one opponent is required")
        for opp in self.opponents:
            opp.validate()

    @classmethod
    def from_toml(cls, path: str | Path) -> PokemonEvalConfig:
        """Parse configuration from a TOML file."""
        path = Path(path)
        with open(path, "rb") as f:
            raw = tomllib.load(f)

        opponents_raw = raw.pop("opponents", [])
        opponents = []
        for opp_dict in opponents_raw:
            # Convert gpu_ids from TOML array to list[int]
            if "gpu_ids" in opp_dict and opp_dict["gpu_ids"] is not None:
                opp_dict["gpu_ids"] = list(opp_dict["gpu_ids"])
            opponents.append(OpponentConfig(**opp_dict))

        config = cls(opponents=opponents, **raw)
        config.validate()
        return config


def compute_node_share(total: int, node_rank: int, n_nodes: int) -> int:
    """Compute how many battles this node should run.

    Distributes evenly, with earlier nodes getting the remainder.
    """
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}")
    if node_rank < 0 or node_rank >= n_nodes:
        raise ValueError(f"node_rank {node_rank} out of range [0, {n_nodes})")

    base = total // n_nodes
    remainder = total % n_nodes
    if node_rank < remainder:
        return base + 1
    return base
