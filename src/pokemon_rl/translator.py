"""Layer 3: State translation — battle state <-> LLM text.

Converts poke-env Battle objects into LLM-friendly text prompts and
parses LLM text responses back into BattleOrders.

Supports pluggable format styles:
- "pokechamp_io": Full pokechamp prompt with damage calcs (default)
- "simple": Minimal text representation (no external deps)

Dependencies: pokechamp (installed via `uv pip install -e /path/to/pokechamp`).

Usage:
    translator = StateTranslator(format_style="pokechamp_io")
    messages = translator.battle_to_prompt(battle)
    order = translator.parse_action('{"move": "thunderbolt"}', battle)
"""

from __future__ import annotations

import json
import re
from typing import Any


class StateTranslator:
    """Converts between battle state and LLM text format.

    Args:
        format_style: Prompt format to use. "pokechamp_io" uses pokechamp's
            full state_translate with damage calcs. "simple" uses a minimal
            text representation with no external deps.
    """

    def __init__(
        self,
        format_style: str = "pokechamp_io",
    ):
        self.format_style = format_style

    def battle_to_prompt(self, battle: Any) -> list[dict[str, str]]:
        """Convert a poke-env Battle object into chat messages.

        Returns:
            List of message dicts: [{"role": "system", ...}, {"role": "user", ...}]
        """
        if self.format_style == "pokechamp_io":
            return self._pokechamp_io_prompt(battle)
        elif self.format_style == "simple":
            return self._simple_prompt(battle)
        else:
            raise ValueError(f"Unknown format_style: {self.format_style}")

    def parse_action(self, response_text: str, battle: Any) -> Any | None:
        """Parse LLM text response into a BattleOrder.

        Extracts the last JSON object from the response and matches
        move/switch names against available actions.

        Returns:
            BattleOrder if successfully parsed, None otherwise.
        """
        from poke_env.player.battle_order import BattleOrder

        # Find the last JSON object in the response
        json_match = None
        for m in re.finditer(r"\{[^{}]*\}", response_text):
            json_match = m

        if json_match is None:
            return None

        try:
            action_json = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            return None

        keys_lower = {k.lower(): k for k in action_json.keys()}

        # Try move / dynamax / terastallize
        for action_type in ("move", "dynamax", "terastallize"):
            if action_type in keys_lower:
                original_key = keys_lower[action_type]
                move_name = str(action_json[original_key]).strip()
                move_id = move_name.lower().replace(" ", "")

                for move in battle.available_moves:
                    if move.id.lower().replace(" ", "") == move_id:
                        return BattleOrder(
                            move,
                            dynamax=(action_type == "dynamax"),
                            terastallize=(action_type == "terastallize"),
                        )

        # Try switch
        if "switch" in keys_lower:
            original_key = keys_lower["switch"]
            switch_name = str(action_json[original_key]).strip()
            switch_id = switch_name.lower().replace(" ", "")

            for pokemon in battle.available_switches:
                if pokemon.species.lower().replace(" ", "") == switch_id:
                    return BattleOrder(pokemon)

        return None

    def get_fallback_action(self, battle: Any) -> Any:
        """Fallback: pick highest base power move, or first switch."""
        from poke_env.player.battle_order import BattleOrder

        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda m: m.base_power)
            return BattleOrder(best_move)
        elif battle.available_switches:
            return BattleOrder(battle.available_switches[0])
        return BattleOrder(None)

    # ------------------------------------------------------------------
    # Format implementations
    # ------------------------------------------------------------------

    def _pokechamp_io_prompt(self, battle: Any) -> list[dict[str, str]]:
        """Use pokechamp's state_translate for rich prompts with damage calcs."""
        try:
            # poke_env must be fully loaded before pokechamp.prompts to avoid
            # circular import: pokechamp.prompts → poke_env → baselines → pokechamp.prompts
            import poke_env  # noqa: F401
            from poke_env.player.local_simulation import LocalSim
            from pokechamp.prompts import state_translate
            from pokechamp.data_cache import (
                get_cached_move_effect,
                get_cached_pokemon_move_dict,
                get_cached_ability_effect,
                get_cached_pokemon_ability_dict,
                get_cached_item_effect,
                get_cached_pokemon_item_dict,
            )
        except ImportError as e:
            raise ImportError(
                "pokechamp_io format requires pokechamp installed. "
                "Run: uv pip install -e /path/to/pokechamp"
            ) from e

        # LocalSim requires data dicts that pokechamp loads from JSON caches
        battle_format = getattr(battle, '_format', 'gen1ou')
        gen_data = poke_env.data.GenData.from_format(battle_format)
        dynamax_disable = "gen1" in battle_format or "gen2" in battle_format or "gen3" in battle_format
        sim = LocalSim(
            battle,
            get_cached_move_effect(),
            get_cached_pokemon_move_dict(),
            get_cached_ability_effect(),
            get_cached_pokemon_ability_dict(),
            get_cached_item_effect(),
            get_cached_pokemon_item_dict(),
            gen_data,
            dynamax_disable,
        )
        system_prompt, state_prompt, action_prompt = state_translate(sim, battle)

        # Combine state and action prompts for the user message
        user_prompt = state_prompt + "\n" + action_prompt

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _simple_prompt(self, battle: Any) -> list[dict[str, str]]:
        """Minimal text prompt — no pokechamp dependency.

        Includes: active pokemon, HP, available moves, available switches.
        Good for smoke testing and simple baselines.
        """
        lines = []

        # Active pokemon
        if battle.active_pokemon:
            p = battle.active_pokemon
            lines.append(f"Your active pokemon: {p.species}")
            lines.append(f"  HP: {p.current_hp_fraction * 100:.0f}%")
            if hasattr(p, "types") and p.types:
                types = [str(t).split(".")[-1] for t in p.types if t is not None]
                lines.append(f"  Types: {', '.join(types)}")

        # Opponent
        if battle.opponent_active_pokemon:
            o = battle.opponent_active_pokemon
            lines.append(f"Opponent active pokemon: {o.species}")
            lines.append(f"  HP: {o.current_hp_fraction * 100:.0f}%")

        # Available moves
        if battle.available_moves:
            lines.append("Available moves:")
            for m in battle.available_moves:
                lines.append(f"  - {m.id} (power: {m.base_power}, type: {m.type})")

        # Available switches
        if battle.available_switches:
            lines.append("Available switches:")
            for p in battle.available_switches:
                lines.append(
                    f"  - {p.species} (HP: {p.current_hp_fraction * 100:.0f}%)"
                )

        # Action format
        lines.append("")
        lines.append(
            'Choose an action as JSON: {"move": "<name>"} or {"switch": "<name>"}'
        )

        system_prompt = (
            "You are a Pokemon battle AI. Choose the best action each turn."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(lines)},
        ]
