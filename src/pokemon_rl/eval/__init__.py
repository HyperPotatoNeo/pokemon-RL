"""Pokemon-RL evaluation package.

Provides standalone evaluation of LLM agents against diverse opponents
(heuristic, metamon RL, other LLMs) using the existing PokemonBattleEnv
+ verifiers interface.

Usage:
    python -m pokemon_rl.eval.runner configs/pokemon/eval_example.toml
"""

from pokemon_rl.eval.config import OpponentConfig, PokemonEvalConfig
from pokemon_rl.eval.llm_player import LLMPlayer

__all__ = ["LLMPlayer", "OpponentConfig", "PokemonEvalConfig"]
