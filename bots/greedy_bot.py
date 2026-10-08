"""
GreedyBot: the teacher for behavioural cloning and the main benchmark.

"Greedy" means it only looks at its own hand and always takes the move that gets it
closest to winning right now. No looking ahead, no defence (it never worries about
what other players are waiting for), and it ignores tai when discarding.

The exact rules (decided by the team, see CLAUDE.md):

  On my turn
    1. Win if the rules allow it (at least 1 tai).
    2. Declare any kong I can (four in hand, or a fourth tile for my own pong).
    3. Otherwise discard. For every tile I could throw, look at the hand that's left:
         - lowest shanten first (closest to ready)
         - then the most useful tiles (tiles that would lower the shanten if drawn,
           counting 4 copies minus the copies I hold myself)
         - then the lowest tile number
       This makes the bot fully predictable: same hand, same discard.

  When someone discards
    1. Win if allowed.
    2. Kong, else pong, but only if the tile scores tai for me (any dragon, my seat
       wind, the prevailing wind) or taking it lowers my shanten.
    3. Never chow. Otherwise pass.

  When someone adds to a pong (a kong I could rob)
    Win if allowed, otherwise pass.
"""

from engine.game import PASS, WIN, Action
from engine.shanten import shanten, useful_tiles
from engine.tiles import NUM_TILE_TYPES, counts_of, is_dragon


class GreedyBot:
    name = "greedy"

    def choose_action(self, view):
        if WIN in view.legal:
            return WIN
        if view.phase == "turn":
            return self._own_turn(view)
        if view.phase == "claim":
            return self._claim(view)
        return PASS                                   # "rob" phase, and nothing to win

    # ------------------------------------------------------------------ my turn
    def _own_turn(self, view):
        kongs = [a for a in view.legal if a.kind == "kong"]
        if kongs:
            return kongs[0]                           # always declare a kong (lowest tile first)
        return Action("discard", self.choose_discard(view))

    def choose_discard(self, view):
        """The tile to throw, by the rules in the module docstring."""
        counts = counts_of(view.hand)
        n_melds = len(view.my_melds)
        best_key, best_tile = None, None
        for tile in range(NUM_TILE_TYPES):
            if counts[tile] == 0:
                continue
            counts[tile] -= 1                          # pretend to throw it
            after = shanten(counts, n_melds)
            useful, _ = useful_tiles(counts, n_melds, self._available(counts, view))
            counts[tile] += 1
            key = (after, -useful, tile)               # smaller is better
            if best_key is None or key < best_key:
                best_key, best_tile = key, tile
        return best_tile

    @staticmethod
    def _available(counts, view):
        """Copies of each tile that could still be drawn, as far as a greedy bot cares:
        4 minus the copies it holds itself (in hand and in its own melds)."""
        held = list(counts)
        for meld in view.my_melds:
            for t in meld.tiles():
                held[t] += 1
        return [4 - held[t] for t in range(NUM_TILE_TYPES)]

    # ------------------------------------------------------------------ someone discarded
    def _claim(self, view):
        tile = view.offer_tile
        counts = counts_of(view.hand)
        n_melds = len(view.my_melds)
        now = shanten(counts, n_melds)
        valued = is_dragon(tile) or tile in (view.seat_wind, view.prevailing_wind)

        if Action("exposed_kong") in view.legal:      # kong beats pong
            counts[tile] -= 3
            better = shanten(counts, n_melds + 1) < now
            counts[tile] += 3
            if valued or better:
                return Action("exposed_kong")
        if Action("pong") in view.legal:
            counts[tile] -= 2
            better = shanten(counts, n_melds + 1) < now
            counts[tile] += 2
            if valued or better:
                return Action("pong")
        return PASS                                   # never chow
