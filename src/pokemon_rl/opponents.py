"""Opponent registry — maps opponent names to routing behavior.

The registry determines how each opponent type is matched:
- "direct": In-process poke-env Player (random, max_damage, abyssal).
  Uses BattleManager.start_battle() with _battle_against.
- "external": Separate process connecting to Showdown's ladder.
  Uses BattleManager.start_battle_ladder() with serialized matching
  to prevent env workers from matching each other.

Users just set `opponent_type = "kakuna"` in their config — the routing
is handled automatically.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpponentSpec:
    """Specification for how to match against an opponent.

    Attributes:
        kind: "direct" (in-process) or "external" (separate process, ladder).
        opponent_type: For direct opponents, the type string passed to
            create_opponent(). None for external opponents.
    """
    kind: str  # "direct" or "external"
    opponent_type: str | None = None

    def __post_init__(self):
        if self.kind not in ("direct", "external"):
            raise ValueError(f"Unknown opponent kind: {self.kind}")
        if self.kind == "direct" and self.opponent_type is None:
            raise ValueError("Direct opponents must specify opponent_type")


# ---- Registry ----
# Add new opponents here. The key is the user-facing name used in TOML configs.
_REGISTRY: dict[str, OpponentSpec] = {
    # Direct opponents — built-in poke-env Players, no external process
    "random": OpponentSpec(kind="direct", opponent_type="random"),
    "max_damage": OpponentSpec(kind="direct", opponent_type="max_damage"),
    "abyssal": OpponentSpec(kind="direct", opponent_type="abyssal"),

    # External opponents — separate process, matched via Showdown ladder
    "kakuna": OpponentSpec(kind="external"),

    # LLM opponents — in-process Player that calls vLLM API
    "llm": OpponentSpec(kind="direct", opponent_type="llm"),
}


def get_opponent_spec(opponent_type: str) -> OpponentSpec:
    """Look up the routing spec for an opponent type.

    Args:
        opponent_type: The user-facing opponent name (e.g., "kakuna", "random").

    Returns:
        OpponentSpec describing how to match against this opponent.

    Raises:
        ValueError: If the opponent type is not in the registry.
    """
    if opponent_type in _REGISTRY:
        return _REGISTRY[opponent_type]
    raise ValueError(
        f"Unknown opponent_type: '{opponent_type}'. "
        f"Available: {sorted(_REGISTRY.keys())}"
    )


def is_external_opponent(opponent_type: str) -> bool:
    """Check if an opponent type requires an external process."""
    spec = _REGISTRY.get(opponent_type)
    return spec is not None and spec.kind == "external"


def list_opponents() -> dict[str, str]:
    """Return a dict of opponent_type -> kind for all registered opponents."""
    return {name: spec.kind for name, spec in _REGISTRY.items()}
