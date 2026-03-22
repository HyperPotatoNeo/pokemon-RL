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
import logging
import random
import re
from typing import Any

logger = logging.getLogger(__name__)


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

        Extracts the last valid JSON object from the response (supports
        nested JSON) and matches move/switch names against available actions.

        Returns:
            BattleOrder if successfully parsed, None otherwise.
        """
        from poke_env.player.battle_order import BattleOrder

        action_json = self._extract_last_json(response_text)
        if action_json is None:
            return None

        keys_lower = {k.lower(): k for k in action_json.keys()}

        # Detect battle format for mechanic validation
        battle_format = getattr(battle, '_format', '') or ''
        format_lower = battle_format.lower()

        # Try move / dynamax / terastallize
        for action_type in ("move", "dynamax", "terastallize"):
            if action_type in keys_lower:
                # Validate mechanic against format
                if action_type == "dynamax" and any(
                    g in format_lower for g in ("gen1", "gen2", "gen3", "gen4", "gen5", "gen6", "gen7", "gen9")
                ):
                    continue  # dynamax only exists in gen8
                if action_type == "terastallize" and "gen9" not in format_lower:
                    continue  # terastallize only exists in gen9

                original_key = keys_lower[action_type]
                move_name = str(action_json[original_key]).strip()
                move_id = re.sub(r'[^a-z0-9]', '', move_name.lower())

                for move in battle.available_moves:
                    if re.sub(r'[^a-z0-9]', '', move.id.lower()) == move_id:
                        return BattleOrder(
                            move,
                            dynamax=(action_type == "dynamax"),
                            terastallize=(action_type == "terastallize"),
                        )

        # Try switch
        if "switch" in keys_lower:
            original_key = keys_lower["switch"]
            switch_name = str(action_json[original_key]).strip()
            switch_id = re.sub(r'[^a-z0-9]', '', switch_name.lower())

            for pokemon in battle.available_switches:
                if re.sub(r'[^a-z0-9]', '', pokemon.species.lower()) == switch_id:
                    return BattleOrder(pokemon)

        return None

    @staticmethod
    def _extract_last_json(text: str) -> dict | None:
        """Extract the last valid JSON object from text.

        Handles nested JSON by trying progressively larger substrings
        from each closing brace backwards to each opening brace.
        """
        for i in range(len(text) - 1, -1, -1):
            if text[i] == '}':
                for j in range(i, -1, -1):
                    if text[j] == '{':
                        try:
                            obj = json.loads(text[j:i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            continue
        return None

    def get_fallback_action(self, battle: Any) -> Any:
        """Fallback: random legal action.

        Uses random instead of max-base-power to prevent reward hacking:
        a model that always outputs garbage would otherwise get the
        strongest heuristic move for free via this fallback.

        Returns a BattleOrder subclass with robust .message (handles both
        real poke-env types and mock objects used in testing).
        """
        from poke_env.player.battle_order import BattleOrder

        class _RobustOrder(BattleOrder):
            """BattleOrder with fallback message for non-poke-env types."""
            @property
            def message(self) -> str:
                msg = super().message
                if msg:
                    return msg
                if hasattr(self.order, "id"):
                    return f"/choose move {self.order.id}"
                if hasattr(self.order, "species"):
                    return f"/choose switch {self.order.species}"
                return "/choose default"

        actions = []
        for m in battle.available_moves:
            actions.append(_RobustOrder(m))
        for p in battle.available_switches:
            actions.append(_RobustOrder(p))
        if actions:
            return random.choice(actions)
        return BattleOrder(None)

    # ------------------------------------------------------------------
    # Completion text extraction (Phase 4: Messages → str)
    # ------------------------------------------------------------------

    @staticmethod
    def extract_completion_text(messages: Any) -> str:
        """Extract text from a completion, handling both string and Messages format.

        In verifiers, add_model_response creates a TrajectoryStep where
        'completion' is a Messages list (list of dicts with role/content),
        not a plain string. This method converts to string for parse_action.

        Args:
            messages: Either a string (legacy) or a list of message dicts
                     (verifiers Messages format).

        Returns:
            The extracted text string.
        """
        if isinstance(messages, str):
            return messages
        if isinstance(messages, list):
            # Extract last assistant message content
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content")
                    # Handle multimodal content blocks (thinking models)
                    if isinstance(content, list):
                        return " ".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict)
                            and block.get("type") == "text"
                        )
                    return content or ""
            # No assistant message — concatenate all content
            parts = []
            for m in messages:
                if isinstance(m, dict):
                    c = m.get("content", "")
                    if isinstance(c, list):
                        c = " ".join(
                            block.get("text", "")
                            for block in c
                            if isinstance(block, dict)
                            and block.get("type") == "text"
                        )
                    parts.append(c)
            return " ".join(parts)
        return str(messages)

    @staticmethod
    def extract_user_content(messages: Any) -> str:
        """Extract user content from a Messages list.

        Used for recording conversation history in _AgentContext.

        Args:
            messages: List of message dicts (Messages format).

        Returns:
            The user message content, or empty string if not found.
        """
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
        return ""

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
        battle_format = getattr(battle, '_format', None)
        if battle_format is None:
            battle_format = 'gen1ou'
            logger.warning(
                "battle._format not found (poke-env API may have changed). "
                "Defaulting to 'gen1ou'. This may produce incorrect damage calcs."
            )
        gen_data = poke_env.data.GenData.from_format(battle_format)
        dynamax_disable = "gen8" not in battle_format
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
            format=battle_format,
        )
        system_prompt, state_prompt, action_prompt = state_translate(sim, battle)

        # Build constraint prompt matching pokechamp's llm_player CoT variant.
        # Uses the "cot" constraint which wraps thought inside JSON, ensuring
        # the model outputs parseable JSON even without json_format=True.
        if battle.active_pokemon.fainted or len(battle.available_moves) == 0:
            constraint = (
                'Choose the most suitable pokemon to switch by thinking '
                'step by step. Your thought should be no more than 3 sentences. '
                'Your output MUST be a JSON like: '
                '{"thought":"<step-by-step-thinking>", '
                '"switch":"<switch_pokemon_name>"}\n'
            )
        elif len(battle.available_switches) == 0:
            constraint = (
                'Choose the best action by thinking step by step. '
                'Your thought should be no more than 3 sentences. '
                'Your output MUST be a JSON like: '
                '{"thought":"<step-by-step-thinking>", '
                '"move":"<move_name>"}\n'
            )
        else:
            constraint = (
                'Choose the best action by thinking step by step. '
                'Your thought should be no more than 3 sentences. '
                'Your output MUST be a JSON like: '
                '{"thought":"<step-by-step-thinking>", '
                '"move":"<move_name>"} or '
                '{"thought":"<step-by-step-thinking>", '
                '"switch":"<switch_pokemon_name>"}\n'
            )

        # Combine: state + action choices + JSON constraint
        user_prompt = state_prompt + "\n" + action_prompt + constraint

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
