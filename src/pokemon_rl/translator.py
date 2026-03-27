"""Layer 3: State translation — battle state <-> LLM text.

Converts poke-env Battle objects into LLM-friendly text prompts and
parses LLM text responses back into BattleOrders.

Supports pluggable format styles:
- "full_obs_cot": XML-tagged prompt with full team observability (default)
- "pokechamp_io": Full pokechamp prompt with damage calcs
- "simple": Minimal text representation (no external deps)

Dependencies: pokechamp (installed via `uv pip install -e /path/to/pokechamp`).

Usage:
    translator = StateTranslator(format_style="full_obs_cot")
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
        format_style: Prompt format to use. "full_obs_cot" (default) uses
            XML-tagged sections with full team info. "pokechamp_io" uses
            pokechamp's state_translate. "simple" uses minimal text.
    """

    def __init__(
        self,
        format_style: str = "full_obs_cot",
    ):
        self.format_style = format_style

    def battle_to_prompt(self, battle: Any) -> list[dict[str, str]]:
        """Convert a poke-env Battle object into chat messages.

        Returns:
            List of message dicts: [{"role": "system", ...}, {"role": "user", ...}]
        """
        if self.format_style == "full_obs_cot":
            return self._full_obs_cot_prompt(battle)
        elif self.format_style == "pokechamp_io":
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
            _copy_battle=False,
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

    def _full_obs_cot_prompt(self, battle: Any) -> list[dict[str, str]]:
        """Structured prompt with XML tags and full team observability.

        Includes opponent bench Pokemon, weather/terrain, terastallize info,
        and estimated damage calculations. Requires pokechamp.
        """
        try:
            import poke_env  # noqa: F401
            from poke_env.player.local_simulation import LocalSim, move_type_damage_wrapper
            from poke_env.environment.side_condition import SideCondition
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
                "full_obs_cot format requires pokechamp installed. "
                "Run: uv pip install -e vendor/pokechamp"
            ) from e

        battle_format = getattr(battle, '_format', None) or 'gen9ou'
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
            _copy_battle=False,
        )

        # --- helpers ---
        def _safe_stats(mon: Any) -> dict:
            """Get stats for a Pokemon, caching and handling errors."""
            cache_key = id(mon)
            if cache_key not in _stats_cache:
                # Own team has .stats from request; opponent needs calculate_stats
                own_stats = getattr(mon, 'stats', None)
                if own_stats and own_stats.get('atk') is not None:
                    _stats_cache[cache_key] = own_stats
                else:
                    try:
                        _stats_cache[cache_key] = mon.calculate_stats(
                            battle_format=battle_format
                        )
                    except Exception:
                        _stats_cache[cache_key] = mon.base_stats
            return _stats_cache[cache_key]

        def _hp_pct(mon: Any) -> str:
            max_hp = mon.max_hp if mon.max_hp != 0 else 1
            return f"{round(mon.current_hp / max_hp * 100)}%"

        def _type_str(mon: Any) -> str:
            parts = []
            if mon.type_1:
                parts.append(mon.type_1.name.capitalize())
            if mon.type_2:
                parts.append(mon.type_2.name.capitalize())
            return " and ".join(parts) if parts else "Unknown"

        def _status_str(mon: Any) -> str:
            s = sim.check_status(mon.status)
            return s if s else "none"

        def _ability_str(mon: Any) -> str:
            ab = mon.ability
            if not ab:
                return "unknown"
            try:
                info = sim.ability_effect[ab]
                return f"{info['name']} ({info['effect']})"
            except (KeyError, TypeError):
                return ab

        def _item_str(mon: Any) -> str:
            item = mon.item
            if not item or item == "unknown_item":
                return "unknown"
            try:
                info = sim.item_effect[item]
                return f"{info['name']} ({info['effect']})"
            except (KeyError, TypeError):
                return item

        def _boost_str(boosts: dict, stat: str) -> str:
            """Format stat with boost annotation, e.g. '588 (+2)'."""
            lvl = boosts.get(stat, 0)
            if lvl == 0:
                return ""
            return f" ({'+' if lvl > 0 else ''}{lvl})"

        def _est_dmg(atk_mon: Any, def_mon: Any, move: Any,
                     atk_stats: dict, def_stats: dict,
                     atk_boosts: dict, def_boosts: dict) -> int:
            """Estimated damage: atk/def ratio * base_power."""
            if move.base_power == 0:
                return 0
            cat = getattr(move, 'category', None)
            cat_name = cat.name if cat else ""
            if cat_name == "SPECIAL":
                a = atk_stats.get('spa', 100) * sim.boost_multiplier('spa', atk_boosts.get('spa', 0))
                d = def_stats.get('spd', 100) * sim.boost_multiplier('spd', def_boosts.get('spd', 0))
            elif cat_name == "PHYSICAL":
                a = atk_stats.get('atk', 100) * sim.boost_multiplier('atk', atk_boosts.get('atk', 0))
                d = def_stats.get('def', 100) * sim.boost_multiplier('def', def_boosts.get('def', 0))
            else:
                return 0
            d = d if d != 0 else 1
            return round(a / d * move.base_power)

        _stats_cache: dict = {}

        # --- data collection ---
        is_p1 = "p1" in list(battle.team.keys())[0] if battle.team else True
        active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon
        active_stats = _safe_stats(active)
        active_boosts = getattr(active, '_boosts', {}) if active else {}
        opp_stats = _safe_stats(opp_active) if opp_active else {}
        opp_boosts = getattr(opp_active, '_boosts', {}) if opp_active else {}
        opp_speed = round(
            opp_stats.get('spe', 0) * sim.boost_multiplier('spe', opp_boosts.get('spe', 0))
        ) if opp_active else 0
        active_speed = round(
            (active_stats.get('spe', 0) or 0) * sim.boost_multiplier('spe', active_boosts.get('spe', 0))
        )

        # ===== SYSTEM PROMPT =====
        gen = getattr(sim.gen, 'gen', 9)
        if active.fainted or len(battle.available_moves) == 0:
            system_prompt = (
                f"You are a pokemon battler in generation {gen} OU format Pokemon "
                f"Showdown that targets to win the pokemon battle. Your "
                f"{active.species} just fainted. Choose a suitable pokemon to "
                f"continue the battle."
            )
        else:
            system_prompt = (
                f"You are a pokemon battler in generation {gen} OU format Pokemon "
                f"Showdown that targets to win the pokemon battle. You can choose "
                f"to use a move or switch to another pokemon.\n"
                f"Tips: Use stat-boosting moves strategically (boosts reset on switch). "
                f"Set hazards (stealthrock, spikes, toxicspikes, stickyweb) when possible. "
                f"When the opponent has boosted, KO it ASAP even at a sacrifice. "
                f"Switching forfeits your move — the opponent acts first, so consider "
                f"the switch-in's speed, typing, and defenses."
            )

        # ===== SECTIONS =====
        sections = []

        # --- 1. HISTORY ---
        history_text = ""
        msg_history = getattr(battle, 'battle_msg_history', '')
        if msg_history:
            turns = msg_history.split("[sep]")[-(5 + 1):]
            raw = "\n".join(turns)
            if is_p1:
                raw = raw.replace("p1a: ", "").replace("p2a:", "opposing")
                raw = raw.replace("Player1", "You").replace("Player2", "Opponent")
            else:
                raw = raw.replace("p2a: ", "").replace("p1a:", "opposing")
                raw = raw.replace("Player2", "You").replace("Player1", "Opponent")
            history_text = raw.strip()
        sections.append(f"<history>\n{history_text if history_text else 'Battle just started.'}\n</history>")

        # --- 2. FIELD CONDITIONS ---
        fc_lines = []
        # Side conditions
        your_sc = []
        for sc in battle.side_conditions:
            name = " ".join(sc.name.lower().split("_"))
            if sc == SideCondition.SPIKES:
                name += " (damages on switch except flying)"
            elif sc == SideCondition.STEALTH_ROCK:
                name += " (rock-type damage on switch)"
            elif sc == SideCondition.STICKY_WEB:
                name += " (lowers speed on switch)"
            elif sc == SideCondition.TOXIC_SPIKES:
                name += " (poisons on switch)"
            your_sc.append(name)
        fc_lines.append(f"Your side: {', '.join(your_sc) if your_sc else 'none'}")

        opp_sc = []
        for sc in battle.opponent_side_conditions:
            opp_sc.append(" ".join(sc.name.lower().split("_")))
        fc_lines.append(f"Opponent side: {', '.join(opp_sc) if opp_sc else 'none'}")

        # Weather
        weather = getattr(battle, 'weather', {})
        if weather:
            w_name = " ".join(list(weather.keys())[0].name.lower().split("_")).replace("_", " ")
            fc_lines.append(f"Weather: {w_name.capitalize()}")

        # Terrain / field
        fields = getattr(battle, 'fields', {})
        if fields:
            for f in fields:
                fc_lines.append(f"Field: {' '.join(f.name.lower().split('_')).capitalize()}")

        # Tera status
        can_tera = getattr(battle, 'can_tera', None)
        opp_can_tera = getattr(battle, 'opponent_can_tera', False)
        if can_tera:
            tera_type = can_tera.name.capitalize() if hasattr(can_tera, 'name') else str(can_tera)
            fc_lines.append(f"You can terastallize: yes (your tera type: {tera_type})")
        elif hasattr(battle, 'can_tera'):
            fc_lines.append("You can terastallize: no (already used)")
        if hasattr(battle, 'opponent_can_tera'):
            fc_lines.append(f"Opponent can terastallize: {'yes' if opp_can_tera else 'no'}")

        # Tera explanation (once per game is fine, small cost)
        if can_tera or opp_can_tera:
            fc_lines.append(
                "Terastallize changes a Pokemon's type to its tera type, boosting "
                "moves of that type. Once per battle. Can terastallize and use a "
                "move in the same turn."
            )

        # Dynamax (gen8)
        can_dynamax = getattr(battle, 'can_dynamax', False)
        if can_dynamax:
            fc_lines.append("You can dynamax: yes")
            fc_lines.append(
                "Dynamax increases max HP and move power for 3 turns. "
                "Once per battle. Can dynamax and use a move in the same turn."
            )

        sections.append(f"<field_conditions>\n" + "\n".join(fc_lines) + "\n</field_conditions>")

        # --- 3. OPPONENT REMAINING POKEMON ---
        opp_bench_lines = []
        is_teampreview = getattr(battle, '_teampreview', False)
        for mon in battle.opponent_team.values():
            if mon.active or mon.fainted:
                continue
            hp = "unknown" if is_teampreview else _hp_pct(mon)
            tera_note = ""
            if getattr(mon, 'terastallized', False):
                orig_type = getattr(mon, '_terastallized_type', None)
                tera_note = f" [Terastallized: {orig_type.name.capitalize() if orig_type else '?'}]"

            line = f"Pokemon: {mon.species}, Type: {_type_str(mon)}{tera_note}, HP: {hp}, Status: {_status_str(mon)}"

            # Revealed moves
            if mon.moves:
                move_strs = []
                for m in mon.moves.values():
                    move_strs.append(f"{m.id}({m.type.name.capitalize()})")
                line += f"\n  Revealed moves: [{', '.join(move_strs)}]"
            else:
                line += "\n  Revealed moves: none"

            line += f"\n  Ability: {_ability_str(mon)}"
            line += f"\n  Item: {_item_str(mon)}"
            opp_bench_lines.append(line)

        opp_bench_text = "\n".join(opp_bench_lines) if opp_bench_lines else "No revealed bench pokemon."
        opp_remaining_count = sum(
            1 for m in battle.opponent_team.values() if not m.fainted
        )
        sections.append(
            f"<opponent_remaining_pokemon>\n"
            f"Opponent has {opp_remaining_count} unfainted pokemon total.\n"
            f"{opp_bench_text}\n"
            f"</opponent_remaining_pokemon>"
        )

        # --- 4. YOUR TEAM (bench) ---
        team_lines = []
        for mon in battle.available_switches:
            stats = _safe_stats(mon)
            spd_cmp = (
                f"(faster than {opp_active.species})"
                if (stats.get('spe', 0) or 0) > opp_speed
                else f"(slower than {opp_active.species})"
            ) if opp_active else ""

            move_strs = []
            for m in mon.moves.values():
                if m.base_power > 0:
                    eff = move_type_damage_wrapper(opp_active, sim.gen.type_chart, [m.type.name]) if opp_active else ""
                    # Extract multiplier
                    mult = "1x"
                    for tag in ["4x", "2x", "0.5x", "0.25x", "0x"]:
                        if tag in eff:
                            mult = tag
                            break
                    move_strs.append(f"{m.id}({m.type.name.capitalize()},{mult})")
                else:
                    move_strs.append(f"{m.id}({m.type.name.capitalize()})")

            line = (
                f"Pokemon: {mon.species}, Type: {_type_str(mon)}, HP: {_hp_pct(mon)}, Status: {_status_str(mon)}\n"
                f"  Stats: Atk:{stats.get('atk','?')}, Def:{stats.get('def','?')}, "
                f"SpA:{stats.get('spa','?')}, SpD:{stats.get('spd','?')}, Spe:{stats.get('spe','?')}\n"
                f"  Moves: [{', '.join(move_strs)}]\n"
                f"  Ability: {_ability_str(mon)}, Item: {_item_str(mon)}\n"
                f"  {spd_cmp}"
            )
            team_lines.append(line)

        team_text = "\n".join(team_lines) if team_lines else "No available switches."
        sections.append(f"<your_team>\n{team_text}\n</your_team>")

        # --- 5. OPPONENT ACTIVE POKEMON ---
        if opp_active and not opp_active.fainted:
            tera_note = ""
            if getattr(opp_active, 'terastallized', False):
                t = getattr(opp_active, '_terastallized_type', None)
                tera_note = f" [Terastallized: {t.name.capitalize() if t else '?'}]"

            # Stats with boost stages
            stat_parts = []
            for stat_key, stat_label in [('atk','Atk'),('def','Def'),('spa','SpA'),('spd','SpD'),('spe','Spe')]:
                base_val = opp_stats.get(stat_key, 0)
                boost_lvl = opp_boosts.get(stat_key, 0)
                boosted_val = round(base_val * sim.boost_multiplier(stat_key, boost_lvl))
                stat_parts.append(f"{stat_label}:{boosted_val}{_boost_str(opp_boosts, stat_key)}")
            stats_line = ", ".join(stat_parts)

            # Revealed moves with Est.dmg vs our active
            revealed_move_strs = []
            for move in opp_active.moves.values():
                dmg = _est_dmg(opp_active, active, move, opp_stats, active_stats, opp_boosts, active_boosts)
                type_name = move.type.name.capitalize()
                if move.base_power > 0:
                    revealed_move_strs.append(f"{move.id}({type_name},Est.dmg:{dmg})")
                else:
                    revealed_move_strs.append(f"{move.id}({type_name},Status)")

            # Top possible moves with Est.dmg vs our active
            possible_move_strs = []
            species_key = opp_active.species
            if species_key == 'polteageistantique':
                species_key = 'polteageist'
            try:
                if species_key in sim.pokemon_move_dict:
                    possible_moves = list(sim.pokemon_move_dict[species_key].values())
                    # Each entry is [name, type, base_power, ...]
                    # Sort by base_power descending, take top 10
                    possible_moves.sort(key=lambda x: x[2] if len(x) > 2 else 0, reverse=True)
                    from poke_env.environment.move import Move as _Move
                    for pm in possible_moves[:10]:
                        pm_name, pm_type, pm_bp = pm[0], pm[1], pm[2]
                        if pm_bp > 0:
                            # Create a temporary move to compute est.dmg
                            try:
                                tmp_move = _Move(pm_name, gen=gen)
                                dmg = _est_dmg(opp_active, active, tmp_move, opp_stats, active_stats, opp_boosts, active_boosts)
                                possible_move_strs.append(f"{pm_name}({pm_type.capitalize()},Est.dmg:{dmg})")
                            except Exception:
                                possible_move_strs.append(f"{pm_name}({pm_type.capitalize()},BP:{pm_bp})")
                        else:
                            possible_move_strs.append(f"{pm_name}({pm_type.capitalize()},Status)")
            except Exception:
                pass

            # Type weaknesses from our team's moves
            team_move_types = []
            for m in battle.available_moves:
                if m.base_power > 0:
                    team_move_types.append(m.type.name)
            for mon in battle.available_switches:
                for m in mon.moves.values():
                    if m.base_power > 0:
                        team_move_types.append(m.type.name)
            type_weak_str = move_type_damage_wrapper(opp_active, sim.gen.type_chart, team_move_types)

            opp_lines = [
                f"Pokemon: {opp_active.species}, Type: {_type_str(opp_active)}{tera_note}, HP: {_hp_pct(opp_active)}, Status: {_status_str(opp_active)}",
                f"  Stats: {stats_line}",
                f"  Ability: {_ability_str(opp_active)}",
                f"  Item: {_item_str(opp_active)}",
            ]
            if revealed_move_strs:
                opp_lines.append(f"  Revealed moves vs your {active.species}: [{', '.join(revealed_move_strs)}]")
            else:
                opp_lines.append(f"  Revealed moves: none")
            if possible_move_strs:
                opp_lines.append(f"  Top possible moves vs your {active.species}: [{', '.join(possible_move_strs)}]")
            if type_weak_str:
                opp_lines.append(f"  {type_weak_str.strip()}")

            sections.append(f"<opponent_active_pokemon>\n" + "\n".join(opp_lines) + "\n</opponent_active_pokemon>")

        # --- 6. YOUR ACTIVE POKEMON ---
        if active:
            a_stat_parts = []
            for stat_key, stat_label in [('atk','Atk'),('def','Def'),('spa','SpA'),('spd','SpD'),('spe','Spe')]:
                base_val = active_stats.get(stat_key, 0) or 0
                boost_lvl = active_boosts.get(stat_key, 0)
                boosted_val = round(base_val * sim.boost_multiplier(stat_key, boost_lvl))
                a_stat_parts.append(f"{stat_label}:{boosted_val}{_boost_str(active_boosts, stat_key)}")

            speed_cmp = (
                f"(slower than {opp_active.species})"
                if active_speed < opp_speed
                else f"(faster than {opp_active.species})"
            ) if opp_active else ""

            active_lines = [
                f"Pokemon: {active.species}, Type: {_type_str(active)}, HP: {_hp_pct(active)}, Status: {_status_str(active)}",
                f"  Stats: {', '.join(a_stat_parts)}",
                f"  Ability: {_ability_str(active)}, Item: {_item_str(active)}",
                f"  {speed_cmp}",
            ]
            sections.append(f"<your_active_pokemon>\n" + "\n".join(active_lines) + "\n</your_active_pokemon>")

        # --- 7. AVAILABLE ACTIONS ---
        action_lines = []
        if battle.available_moves:
            opp_name = opp_active.species if opp_active else "opponent"
            action_lines.append(f"Moves vs opponent's {opp_name}:")
            for move in battle.available_moves:
                dmg = _est_dmg(active, opp_active, move, active_stats, opp_stats, active_boosts, opp_boosts) if opp_active else 0
                type_name = move.type.name.capitalize()

                # Move effect
                try:
                    effect = sim.move_effect.get(move.id, "")
                except Exception:
                    effect = ""

                # Type effectiveness
                eff_str = ""
                if opp_active and move.base_power > 0:
                    eff = move_type_damage_wrapper(opp_active, sim.gen.type_chart, [move.type.name])
                    if eff:
                        # Extract just the multiplier part
                        for tag in ["4x super effective", "2x super effective", "0.5x not very effective", "0.25x not very effective", "0x immune"]:
                            if tag.split()[0] in eff:
                                eff_str = f" ({tag})"
                                break

                acc = round(move.accuracy * sim.boost_multiplier('accuracy', active_boosts.get('accuracy', 0)) * 100)

                if move.base_power > 0:
                    action_lines.append(
                        f"  - {move.id}: {type_name}, Est.dmg:{dmg}, Acc:{acc}%"
                        + (f", Effect: {effect}" if effect else "")
                        + eff_str
                    )
                else:
                    cat = move.category.name.capitalize() if hasattr(move.category, 'name') else "Status"
                    action_lines.append(
                        f"  - {move.id}: {type_name}, {cat}"
                        + (f", Acc:{acc}%" if move.accuracy < 1.0 else ", Acc:100%")
                        + (f", Effect: {effect}" if effect else "")
                    )

        if battle.available_switches:
            action_lines.append("Switches:")
            for mon in battle.available_switches:
                stats = _safe_stats(mon)
                spd = stats.get('spe', 0) or 0
                spd_cmp = (
                    f"(faster)" if spd > opp_speed else f"(slower)"
                ) if opp_active else ""

                # Key moves with effectiveness
                key_moves = []
                for m in mon.moves.values():
                    if m.base_power > 0 and opp_active:
                        eff = move_type_damage_wrapper(opp_active, sim.gen.type_chart, [m.type.name])
                        mult = "1x"
                        for tag in ["4x", "2x", "0.5x", "0.25x", "0x"]:
                            if tag in eff:
                                mult = tag
                                break
                        key_moves.append(f"{m.id}({m.type.name.capitalize()},{mult})")
                moves_str = f", Moves: [{', '.join(key_moves)}]" if key_moves else ""

                action_lines.append(
                    f"  - {mon.species}: {_type_str(mon)}, HP:{_hp_pct(mon)}, "
                    f"Spe:{spd} {spd_cmp}{moves_str}"
                )

        # Action choice lists
        move_choices = [m.id for m in battle.available_moves]
        switch_choices = [p.species for p in battle.available_switches]
        if move_choices:
            action_lines.append(f"[<move_name>] = {move_choices}")
        if switch_choices:
            action_lines.append(f"[<switch_pokemon_name>] = {switch_choices}")

        sections.append(f"<available_actions>\n" + "\n".join(action_lines) + "\n</available_actions>")

        # --- 8. CONSTRAINT ---
        # Build gimmick output format options
        gimmick_opts = ""
        if can_tera:
            gimmick_opts += ' or {"thought":"<your reasoning>", "terastallize":"<move_name>"}'
        if can_dynamax:
            gimmick_opts += ' or {"thought":"<your reasoning>", "dynamax":"<move_name>"}'

        if active.fainted or len(battle.available_moves) == 0:
            constraint = (
                'Choose the most suitable pokemon to switch by reasoning step by step. '
                'Your output MUST be a JSON like: '
                '{"thought":"<your reasoning>", "switch":"<switch_pokemon_name>"}\n'
            )
        elif len(battle.available_switches) == 0:
            constraint = (
                'Choose the best action by reasoning step by step. '
                'Your output MUST be a JSON like: '
                '{"thought":"<your reasoning>", "move":"<move_name>"}'
                + gimmick_opts + '\n'
            )
        else:
            constraint = (
                'Choose the best action by reasoning step by step. '
                'Your output MUST be a JSON like: '
                '{"thought":"<your reasoning>", "move":"<move_name>"}'
                + gimmick_opts
                + ' or {"thought":"<your reasoning>", "switch":"<switch_pokemon_name>"}\n'
            )

        sections.append(f"<constraint>\n{constraint}</constraint>")

        # ===== ASSEMBLE =====
        user_prompt = "\n\n".join(sections)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # ------------------------------------------------------------------
    # Interleaved trajectory prompt builders
    # ------------------------------------------------------------------

    def battle_to_prompt_interleaved_first(self, battle: Any) -> list[dict[str, str]]:
        """First-turn prompt for interleaved trajectory mode.

        Calls _full_obs_cot_prompt to get the base messages, then modifies:
        1. System prompt: appends multi-turn instruction
        2. Constraint section: replaces JSON output instruction with
           reasoning-only instruction (extraction happens separately).

        Returns:
            List of message dicts: [{"role": "system", ...}, {"role": "user", ...}]
        """
        messages = self._full_obs_cot_prompt(battle)

        # 1. Append multi-turn instruction to system prompt
        if messages and messages[0].get("role") == "system":
            messages[0] = {
                "role": "system",
                "content": messages[0]["content"]
                + "\nYou will play multiple turns. Each turn I will describe "
                "the situation and you will reason about it, then I will ask "
                "you to choose an action.",
            }

        # 2. Replace the <constraint> section in the user message
        for msg in messages:
            if msg.get("role") == "user":
                import re as _re

                new_constraint = (
                    "<constraint>\n"
                    "Analyze the current situation carefully. Consider type "
                    "matchups, speed tiers, HP thresholds, and strategic "
                    "positioning. Reason about which action is best and why. "
                    "Do not output JSON — I will ask for your choice separately.\n"
                    "</constraint>"
                )
                msg["content"] = _re.sub(
                    r"<constraint>\n.*?</constraint>",
                    new_constraint,
                    msg["content"],
                    flags=_re.DOTALL,
                )
                break

        return messages

    def battle_to_prompt_light(self, battle: Any) -> dict[str, str]:
        """Compact observation prompt for subsequent turns in interleaved mode.

        Returns a single user message with compact sections:
        <situation>, <hp_summary>, <field>, <available_actions>.

        Uses LocalSim for damage estimates on available moves.
        Estimated ~400-500 tokens.

        Returns:
            Single message dict: {"role": "user", "content": ...}
        """
        try:
            import poke_env  # noqa: F401
            from poke_env.player.local_simulation import LocalSim
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
                "battle_to_prompt_light requires pokechamp installed. "
                "Run: uv pip install -e vendor/pokechamp"
            ) from e

        battle_format = getattr(battle, '_format', None) or 'gen9ou'
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
            _copy_battle=False,
        )

        # --- helpers (duplicated simple ones for locality) ---
        def _hp_pct(mon: Any) -> str:
            max_hp = mon.max_hp if mon.max_hp != 0 else 1
            return f"{round(mon.current_hp / max_hp * 100)}%"

        def _type_str(mon: Any) -> str:
            parts = []
            if mon.type_1:
                parts.append(mon.type_1.name.capitalize())
            if mon.type_2:
                parts.append(mon.type_2.name.capitalize())
            return "/".join(parts) if parts else "Unknown"

        def _status_str(mon: Any) -> str:
            s = sim.check_status(mon.status)
            return s if s else ""

        def _safe_stats(mon: Any) -> dict:
            own_stats = getattr(mon, 'stats', None)
            if own_stats and own_stats.get('atk') is not None:
                return own_stats
            try:
                return mon.calculate_stats(battle_format=battle_format)
            except Exception:
                return mon.base_stats

        def _est_dmg(atk_mon: Any, def_mon: Any, move: Any,
                     atk_stats: dict, def_stats: dict,
                     atk_boosts: dict, def_boosts: dict) -> int:
            if move.base_power == 0:
                return 0
            cat = getattr(move, 'category', None)
            cat_name = cat.name if cat else ""
            if cat_name == "SPECIAL":
                a = atk_stats.get('spa', 100) * sim.boost_multiplier('spa', atk_boosts.get('spa', 0))
                d = def_stats.get('spd', 100) * sim.boost_multiplier('spd', def_boosts.get('spd', 0))
            elif cat_name == "PHYSICAL":
                a = atk_stats.get('atk', 100) * sim.boost_multiplier('atk', atk_boosts.get('atk', 0))
                d = def_stats.get('def', 100) * sim.boost_multiplier('def', def_boosts.get('def', 0))
            else:
                return 0
            d = d if d != 0 else 1
            return round(a / d * move.base_power)

        # --- data collection ---
        active = battle.active_pokemon
        opp_active = battle.opponent_active_pokemon
        active_stats = _safe_stats(active) if active else {}
        active_boosts = getattr(active, '_boosts', {}) if active else {}
        opp_stats = _safe_stats(opp_active) if opp_active else {}
        opp_boosts = getattr(opp_active, '_boosts', {}) if opp_active else {}
        opp_speed = round(
            opp_stats.get('spe', 0) * sim.boost_multiplier('spe', opp_boosts.get('spe', 0))
        ) if opp_active else 0
        active_speed = round(
            (active_stats.get('spe', 0) or 0) * sim.boost_multiplier('spe', active_boosts.get('spe', 0))
        ) if active else 0

        sections = []

        # --- 1. SITUATION ---
        sit_lines = []
        if active and not active.fainted:
            boost_parts = []
            for stat_key, stat_label in [('atk','Atk'),('def','Def'),('spa','SpA'),('spd','SpD'),('spe','Spe')]:
                lvl = active_boosts.get(stat_key, 0)
                if lvl != 0:
                    boost_parts.append(f"{stat_label}{'+' if lvl > 0 else ''}{lvl}")
            boost_str = ", " + ", ".join(boost_parts) if boost_parts else ""
            ability = active.ability or "unknown"
            item = active.item if active.item and active.item != "unknown_item" else "unknown"
            speed_cmp = ""
            if opp_active:
                speed_cmp = "faster than opponent" if active_speed > opp_speed else "slower than opponent"
            sit_lines.append(
                f"Your active: {active.species} ({_type_str(active)}), "
                f"HP: {_hp_pct(active)}{boost_str}, "
                f"Ability: {ability}, Item: {item}"
            )
            if speed_cmp:
                sit_lines.append(f"  Speed: {active_speed} ({speed_cmp})")

        if opp_active and not opp_active.fainted:
            opp_boost_parts = []
            for stat_key, stat_label in [('atk','Atk'),('def','Def'),('spa','SpA'),('spd','SpD'),('spe','Spe')]:
                lvl = opp_boosts.get(stat_key, 0)
                if lvl != 0:
                    opp_boost_parts.append(f"{stat_label}{'+' if lvl > 0 else ''}{lvl}")
            opp_boost_str = ", " + ", ".join(opp_boost_parts) if opp_boost_parts else ""
            opp_ability = opp_active.ability or "unknown"
            # Revealed moves only
            revealed = []
            for m in opp_active.moves.values():
                revealed.append(m.id)
            sit_lines.append(
                f"Opponent active: {opp_active.species} ({_type_str(opp_active)}), "
                f"HP: {_hp_pct(opp_active)}{opp_boost_str}, "
                f"Ability: {opp_ability}"
            )
            if revealed:
                sit_lines.append(f"  Revealed moves: {', '.join(revealed)}")

        sections.append("<situation>\n" + "\n".join(sit_lines) + "\n</situation>")

        # --- 2. HP SUMMARY ---
        hp_lines = []
        # Your team
        your_mons = []
        for mon in battle.team.values():
            if mon.fainted:
                your_mons.append(f"{mon.species} FAINTED")
            else:
                status = _status_str(mon)
                status_tag = f" ({status})" if status else ""
                your_mons.append(f"{mon.species} {_hp_pct(mon)}{status_tag}")
        hp_lines.append(f"Your team: {', '.join(your_mons)}")

        # Opponent team
        opp_mons = []
        for mon in battle.opponent_team.values():
            if mon.fainted:
                opp_mons.append(f"{mon.species} FAINTED")
            else:
                # Opponent HP may be unknown for unrevealed pokemon
                if mon.current_hp is None:
                    opp_mons.append(f"{mon.species} unknown")
                else:
                    status = _status_str(mon)
                    status_tag = f" ({status})" if status else ""
                    opp_mons.append(f"{mon.species} {_hp_pct(mon)}{status_tag}")
        # Count unknown opponents
        revealed_count = len(battle.opponent_team)
        total_opp = 6  # standard team size
        unknown_count = total_opp - revealed_count
        if unknown_count > 0:
            opp_mons.append(f"{unknown_count} unknown")
        hp_lines.append(f"Opponent: {', '.join(opp_mons)}")

        sections.append("<hp_summary>\n" + "\n".join(hp_lines) + "\n</hp_summary>")

        # --- 3. FIELD ---
        from poke_env.environment.side_condition import SideCondition

        field_parts = []
        # Weather
        weather = getattr(battle, 'weather', {})
        if weather:
            w_name = list(weather.keys())[0].name.lower().replace("_", " ")
            field_parts.append(f"Weather: {w_name}")
        else:
            field_parts.append("Weather: none")

        # Terrain
        fields = getattr(battle, 'fields', {})
        if fields:
            terrain_names = [f.name.lower().replace("_", " ") for f in fields]
            field_parts.append(f"Terrain: {', '.join(terrain_names)}")
        else:
            field_parts.append("Terrain: none")

        field_line = " | ".join(field_parts)

        # Side conditions
        your_sc = []
        for sc in battle.side_conditions:
            name = sc.name.lower().replace("_", " ")
            if sc == SideCondition.SPIKES:
                layers = battle.side_conditions.get(sc, 0)
                name += f" ({layers})" if layers > 1 else ""
            your_sc.append(name)
        opp_sc = []
        for sc in battle.opponent_side_conditions:
            name = sc.name.lower().replace("_", " ")
            if sc == SideCondition.SPIKES:
                layers = battle.opponent_side_conditions.get(sc, 0)
                name += f" ({layers})" if layers > 1 else ""
            opp_sc.append(name)

        sc_line = f"Your side: {', '.join(your_sc) if your_sc else 'none'} | Opponent: {', '.join(opp_sc) if opp_sc else 'none'}"

        sections.append(f"<field>\n{field_line}\n{sc_line}\n</field>")

        # --- 4. AVAILABLE ACTIONS ---
        act_lines = []
        if battle.available_moves:
            move_parts = []
            for move in battle.available_moves:
                type_name = move.type.name.capitalize()
                if move.base_power > 0 and opp_active:
                    dmg = _est_dmg(active, opp_active, move, active_stats, opp_stats, active_boosts, opp_boosts)
                    acc = round(move.accuracy * sim.boost_multiplier('accuracy', active_boosts.get('accuracy', 0)) * 100)
                    move_parts.append(f"{move.id} ({type_name}, ~{dmg}dmg, {acc}%)")
                elif move.base_power > 0:
                    move_parts.append(f"{move.id} ({type_name}, BP:{move.base_power})")
                else:
                    move_parts.append(f"{move.id} ({type_name}, status)")
            act_lines.append(f"Moves: {', '.join(move_parts)}")

        if battle.available_switches:
            switch_parts = []
            for mon in battle.available_switches:
                switch_parts.append(f"{mon.species} ({_type_str(mon)}, {_hp_pct(mon)})")
            act_lines.append(f"Switches: {', '.join(switch_parts)}")

        sections.append("<available_actions>\n" + "\n".join(act_lines) + "\n</available_actions>")

        # --- Trailing instruction ---
        content = "\n\n".join(sections) + "\n\nAnalyze the situation and reason about the best action."

        return {"role": "user", "content": content}

    def extraction_prompt(self, battle: Any) -> dict[str, str]:
        """Extraction prompt for interleaved trajectory mode.

        Returns a short prompt asking the model to output its action as JSON.
        Adapts format based on whether a force-switch is needed.

        Returns:
            Single message dict: {"role": "user", "content": ...}
        """
        if battle.active_pokemon.fainted or len(battle.available_moves) == 0:
            content = 'Now output your switch choice as JSON: {"switch": "<name>"}. No other text.'
        else:
            content = 'Now output your chosen action as JSON. Use {"move": "<name>"} or {"switch": "<name>"}. No other text.'

        return {"role": "user", "content": content}

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
