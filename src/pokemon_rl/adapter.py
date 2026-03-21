"""Layer 2: Battle adapter — full-battle mode via poke-env callbacks.

Runs a complete battle via poke-env's battle_against(), capturing the
trajectory through a callback player. For turn-by-turn control, see
BattleManager in battle.py.

Dependencies: poke-env (via pokechamp's PYTHONPATH)

Usage:
    adapter = BattleAdapter(port=8000, battle_format="gen1randombattle")
    result = await adapter.run_battle()
    print(f"Won: {result['won']}, Turns: {result['turns']}")
    print(f"Trajectory: {len(result['trajectory'])} steps")
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable


class CallbackPlayer:
    """Player that delegates move selection to an external callback.

    Wraps a poke-env Player, routing choose_move() calls to a callback
    function while recording the full trajectory.

    This is a factory — call create() to get an actual Player instance,
    because Player requires poke-env imports at construction time.
    """

    @staticmethod
    def create(
        callback: Callable,
        battle_format: str,
        server_config: Any,
        account_name: str | None = None,
        team: str | None = None,
    ):
        """Create a poke-env Player with callback-based move selection.

        Args:
            callback: fn(battle) -> BattleOrder
            battle_format: e.g. "gen1randombattle"
            server_config: poke-env ServerConfiguration
            account_name: Player username (auto-generated if None)
            team: Team string (None for random battle formats)

        Returns:
            Player instance with .trajectory list attribute
        """
        from poke_env import AccountConfiguration
        from poke_env.player.player import Player

        if account_name is None:
            from pokemon_rl.players import _next_username
            account_name = _next_username("cbp")

        class _CallbackPlayerImpl(Player):
            def __init__(self, cb, **kwargs):
                super().__init__(**kwargs)
                self._callback = cb
                self.trajectory = []

            def choose_move(self, battle):
                order = self._callback(battle)
                self.trajectory.append({
                    "turn": battle.turn,
                    "available_moves": [m.id for m in battle.available_moves],
                    "available_switches": [
                        p.species for p in battle.available_switches
                    ],
                    "action": order.message if order else "/choose default",
                    "active_pokemon": (
                        battle.active_pokemon.species
                        if battle.active_pokemon
                        else None
                    ),
                    "opponent_pokemon": (
                        battle.opponent_active_pokemon.species
                        if battle.opponent_active_pokemon
                        else None
                    ),
                    "hp_fraction": (
                        battle.active_pokemon.current_hp_fraction
                        if battle.active_pokemon
                        else 0
                    ),
                    "opponent_hp_fraction": (
                        battle.opponent_active_pokemon.current_hp_fraction
                        if battle.opponent_active_pokemon
                        else 0
                    ),
                    "battle_tag": battle.battle_tag,
                })
                return order

        kwargs = dict(
            battle_format=battle_format,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(account_name, None),
            max_concurrent_battles=1,
        )
        if team is not None:
            kwargs["team"] = team

        return _CallbackPlayerImpl(callback, **kwargs)


def default_action(battle) -> Any:
    """Default action: first available move, else first switch."""
    from poke_env.player.battle_order import BattleOrder

    if battle.available_moves:
        return BattleOrder(battle.available_moves[0])
    elif battle.available_switches:
        return BattleOrder(battle.available_switches[0])
    return BattleOrder(None)


def random_action(battle) -> Any:
    """Random legal action."""
    from poke_env.player.battle_order import BattleOrder

    actions = []
    for m in battle.available_moves:
        actions.append(BattleOrder(m))
    for p in battle.available_switches:
        actions.append(BattleOrder(p))
    if actions:
        return random.choice(actions)
    return BattleOrder(None)


class BattleAdapter:
    """Manages Pokemon battles via poke-env.

    Args:
        port: Showdown server port (default 8000)
        battle_format: Pokemon Showdown format string
        server_host: Showdown server hostname (default "localhost",
            use node hostname for cross-node play)
    """

    def __init__(
        self,
        port: int = 8000,
        battle_format: str = "gen1randombattle",
        server_host: str = "localhost",
    ):
        self.port = port
        self.battle_format = battle_format
        self.server_host = server_host
        self._server_config = None

    def _get_server_config(self):
        """Lazy-load server configuration."""
        if self._server_config is None:
            from poke_env.ps_client.server_configuration import ServerConfiguration

            self._server_config = ServerConfiguration(
                f"{self.server_host}:{self.port}",
                "https://play.pokemonshowdown.com/action.php?",
            )
        return self._server_config

    async def run_battle(
        self,
        action_fn: Callable | None = None,
        opponent_type: str = "random",
        player_team: str | None = None,
        opponent_team: str | None = None,
    ) -> dict:
        """Run a complete battle and return the trajectory.

        Args:
            action_fn: fn(battle) -> BattleOrder. Default: first legal move.
            opponent_type: "random" for RandomPlayer.
            player_team: Team string (None for random battle formats).
            opponent_team: Team string (None for random battle formats).

        Returns:
            dict with keys:
                trajectory: list of turn records
                won: bool or None
                turns: int
                format: str
                battle_tag: str
        """
        from poke_env.player.random_player import RandomPlayer
        from poke_env import AccountConfiguration

        if action_fn is None:
            action_fn = default_action

        server_config = self._get_server_config()

        # Create our player (captures trajectory)
        player = CallbackPlayer.create(
            callback=action_fn,
            battle_format=self.battle_format,
            server_config=server_config,
            team=player_team,
        )

        # Create opponent
        from pokemon_rl.players import _next_username
        opp_name = _next_username("opp")
        opp_kwargs = dict(
            battle_format=self.battle_format,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(opp_name, None),
            max_concurrent_battles=1,
        )
        if opponent_team is not None:
            opp_kwargs["team"] = opponent_team

        opponent = RandomPlayer(**opp_kwargs)

        # Run the battle
        await player.battle_against(opponent, n_battles=1)

        # Extract results
        battle = list(player.battles.values())[0]

        return {
            "trajectory": player.trajectory,
            "won": battle.won,
            "turns": battle.turn,
            "format": self.battle_format,
            "battle_tag": battle.battle_tag,
        }

    # Turn-by-turn control is implemented in BattleManager (battle.py).
    # BattleAdapter handles full-battle (callback-driven) mode only.
