"""Pokemon Showdown multi-agent RL environment for prime-rl.

Architecture (4 Layers):
    Layer 1: ShowdownEngine    — Manages Node.js Showdown process
    Layer 2: BattleAdapter     — Full-battle mode (callback-driven)
             BattleManager     — Turn-by-turn mode (imperative control)
             ControllablePlayer — Queue-based external control
    Layer 3: StateTranslator   — Battle state <-> LLM text
    Layer 4: PokemonBattleEnv  — MultiTurnEnv hooks for RL integration
"""

__version__ = "0.1.0"
