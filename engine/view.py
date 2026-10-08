"""
What one player is allowed to see. Bots (and later the neural network) only ever get
a PlayerView, never the Game itself, so they can't peek at other hands or the wall.

Every per-player list is RELATIVE to the viewer:
    index 0 = me, 1 = next player (acts after me), 2 = opposite, 3 = previous player
"""

from dataclasses import dataclass

from engine.tiles import wind_tile


@dataclass
class PlayerView:
    seat: int               # my absolute seat (0-3); for bookkeeping only
    hand: list              # my concealed tiles, sorted
    my_melds: list          # my Meld objects (pongs, chows, kongs on the table)
    discards: list          # 4 lists: tiles still lying in each player's discard pile
                            #   (a tile that was claimed moves into the claimer's meld instead)
    melds: list             # 4 lists: the tiles in each player's melds
    bonus: list             # 4 lists: each player's bonus tiles (flowers, animals)
    last_discard: int       # the most recent tile anyone threw, or None
    seat_wind: int          # tile number of my seat wind (27 = East ... 30 = North)
    prevailing_wind: int    # tile number of the round wind
    wall_left: int          # tiles left in the wall
    dealer: int             # who deals, relative to me (0 = I'm the dealer)
    phase: str              # "turn", "claim" or "rob" (see engine/game.py)
    offer_tile: int         # "claim"/"rob": the tile on offer, else None
    offer_from: int         # "claim"/"rob": who offers it, relative to me, else None
    drawn: int              # "turn": the tile I just drew, else None
    legal: list             # the Actions I may take right now


def view_for(game, seat):
    """Build the PlayerView for `seat` from a Game."""
    order = [(seat + k) % 4 for k in range(4)]            # me, next, opposite, previous
    players = [game.players[s] for s in order]

    # The most recent discard by anyone (discards remember the turn they were made on).
    last, last_turn = None, -1
    for p in game.players:
        if p.discards and p.discards[-1].turn > last_turn:
            last, last_turn = p.discards[-1].tile, p.discards[-1].turn

    me = game.players[seat]
    return PlayerView(
        seat=seat,
        hand=me.hand_tiles(),
        my_melds=list(me.melds),
        discards=[[d.tile for d in p.discards if not d.claimed] for p in players],
        melds=[[t for m in p.melds for t in m.tiles()] for p in players],
        bonus=[list(p.bonus) for p in players],
        last_discard=last,
        seat_wind=wind_tile(game.seat_wind(seat)),
        prevailing_wind=wind_tile(game.prevailing),
        wall_left=game.live_tiles(),
        dealer=(game.dealer - seat) % 4,
        phase=game.phase,
        offer_tile=game.offer_tile if game.phase in ("claim", "rob") else None,
        offer_from=(game.offer_from - seat) % 4 if game.phase in ("claim", "rob") else None,
        drawn=game.drawn if game.phase == "turn" else None,
        legal=game.legal_actions(seat),
    )
