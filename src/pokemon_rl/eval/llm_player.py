"""LLMPlayer — poke-env Player that uses a vLLM OpenAI-compatible API.

choose_move returns an Awaitable[BattleOrder] that:
1. Translates battle state → prompt via StateTranslator
2. Calls vLLM API (async HTTP, awaited on POKE_LOOP — non-blocking)
3. Parses response → BattleOrder via StateTranslator
4. Falls back to random legal action on parse failure
5. Forfeits after 3 consecutive failures (zombie prevention)

The AsyncOpenAI client is created lazily on first choose_move call to
ensure it binds to POKE_LOOP's event loop (not the caller's).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Union

logger = logging.getLogger(__name__)

# Max consecutive failures before forfeiting (mirrors ControllablePlayer)
_MAX_CONSECUTIVE_FAILURES = 3


class LLMPlayer:
    """Factory for creating LLM-backed poke-env Players."""

    @staticmethod
    def create(
        base_url: str,
        model_name: str,
        battle_format: str,
        server_config: Any,
        observation_format: str = "pokechamp_io",
        max_tokens: int = 800,
        temperature: float = 1.0,
        timeout: float = 60.0,
        account_name: str | None = None,
        team: str | None = None,
    ) -> Any:
        """Create a poke-env Player that calls vLLM for move selection.

        Args:
            base_url: vLLM OpenAI-compatible endpoint (e.g., "http://localhost:8002/v1")
            model_name: Model served by vLLM (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
            battle_format: Pokemon Showdown format string
            server_config: poke-env ServerConfiguration
            observation_format: "pokechamp_io" or "simple"
            max_tokens: Max tokens for API call
            temperature: Sampling temperature
            timeout: Per-move API call timeout in seconds
            account_name: Showdown account name (auto-generated if None)
            team: Team string (None for random formats)

        Returns:
            A poke-env Player instance.
        """
        from poke_env import AccountConfiguration
        from poke_env.player import Player

        from pokemon_rl.translator import StateTranslator

        if account_name is None:
            import itertools
            import time

            _counter = getattr(LLMPlayer, "_counter", None)
            if _counter is None:
                _counter = itertools.count()
                LLMPlayer._counter = _counter
            account_name = f"llm-{next(_counter)}-{int(time.time()) % 10000}"

        translator = StateTranslator(format_style=observation_format)

        class _LLMPlayerImpl(Player):

            def __init__(self, **kwargs: Any):
                super().__init__(**kwargs)
                self._llm_base_url = base_url
                self._llm_model = model_name
                self._llm_max_tokens = max_tokens
                self._llm_temperature = temperature
                self._llm_timeout = timeout
                self._translator = translator
                self._client = None  # Lazy — created on POKE_LOOP
                self._consecutive_failures = 0

            def choose_move(
                self, battle: Any
            ) -> Union[Any, Any]:  # BattleOrder | Awaitable[BattleOrder]
                return self._async_choose_move(battle)

            async def _async_choose_move(self, battle: Any) -> Any:
                from poke_env.player.battle_order import DefaultBattleOrder

                # Forfeit after too many consecutive failures
                if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    logger.warning(
                        f"LLMPlayer: {self._consecutive_failures} consecutive failures, "
                        f"forfeiting battle {getattr(battle, 'battle_tag', '?')}"
                    )
                    return DefaultBattleOrder()

                # Lazy client creation (binds to POKE_LOOP's event loop)
                if self._client is None:
                    from openai import AsyncOpenAI

                    self._client = AsyncOpenAI(
                        base_url=self._llm_base_url,
                        api_key="EMPTY",
                    )

                # 1. Translate battle state to prompt
                try:
                    messages = await asyncio.to_thread(
                        self._translator.battle_to_prompt, battle
                    )
                except Exception:
                    logger.warning(
                        "LLMPlayer: prompt building failed, using default move",
                        exc_info=True,
                    )
                    self._consecutive_failures += 1
                    return self.choose_default_move()

                # 2. Call vLLM API with timeout
                try:
                    response = await asyncio.wait_for(
                        self._client.chat.completions.create(
                            model=self._llm_model,
                            messages=messages,
                            max_tokens=self._llm_max_tokens,
                            temperature=self._llm_temperature,
                        ),
                        timeout=self._llm_timeout,
                    )
                    text = response.choices[0].message.content or ""
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(
                        f"LLMPlayer: API call failed ({type(e).__name__}), "
                        f"using default move"
                    )
                    self._consecutive_failures += 1
                    return self.choose_default_move()

                # 3. Parse action from response
                action = self._translator.parse_action(text, battle)
                if action is None:
                    action = self._translator.get_fallback_action(battle)
                    if action is None:
                        self._consecutive_failures += 1
                        return self.choose_default_move()
                    logger.debug("LLMPlayer: parse failed, using fallback action")
                    self._consecutive_failures += 1
                    return action

                # Success — reset failure counter
                self._consecutive_failures = 0
                return action

        kwargs: dict[str, Any] = dict(
            battle_format=battle_format,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(account_name, None),
            max_concurrent_battles=1,
        )
        if team is not None:
            kwargs["team"] = team

        return _LLMPlayerImpl(**kwargs)
