"""
Scoring: is this a winning hand, how many tai is it worth, and who pays what.

The rules come from RULES.md. In short:
  - A win is 4 sets (pong or chow, including declared melds) + 1 pair,
    or Thirteen Orphans. Seven pairs is NOT a winning hand.
  - Tai from every pattern is added up, then capped at 5 (MAX_TAI).
  - A hand needs at least 1 tai (MIN_TAI) to be allowed to win.
  - When a hand can be read more than one way, the best reading counts.
"""

from dataclasses import dataclass, field
from functools import lru_cache

from engine.tiles import (
    ANIMALS, DRAGONS, NUM_TILE_TYPES, ORPHANS, PLANTS, SEASONS, WINDS,
    is_suited, number, suit, wind_tile,
)

MIN_TAI = 1
MAX_TAI = 5


# --------------------------------------------------------------------------
# Melds (sets placed face up on the table) and the situation of a win
# --------------------------------------------------------------------------
@dataclass
class Meld:
    kind: str               # "chow", "pong", "exposed_kong", "added_kong", "concealed_kong"
    tile: int               # for a chow: the LOWEST tile of the run; otherwise the tile itself
    from_seat: int = None   # who discarded the claimed tile (None for a concealed kong)

    def tiles(self):
        if self.kind == "chow":
            return [self.tile, self.tile + 1, self.tile + 2]
        if self.kind == "pong":
            return [self.tile] * 3
        return [self.tile] * 4                          # any kong

    def is_kong(self):
        return self.kind.endswith("kong")

    def is_open(self):
        """Made with someone else's tile. A concealed kong does not open the hand."""
        return self.kind != "concealed_kong"


@dataclass
class WinContext:
    seat_wind: int                  # 0 = East ... 3 = North (the winner's seat wind)
    prevailing: int                 # round wind, same numbering
    bonus: list                     # the winner's bonus tiles
    self_draw: bool                 # won on own draw (True) or on a discard / robbed kong (False)
    win_tile: int                   # the tile that completed the hand
    kong_replacement: bool = False  # won on the replacement tile after a kong
    robbing_kong: bool = False      # won on a tile someone added to their pong
    last_tile: bool = False         # won on the last drawable tile, or the discard after it
    heavenly: bool = False          # dealer wins with the dealt hand
    earthly: bool = False           # non-dealer wins on their own first draw, before any claim


@dataclass
class Score:
    tai: int                        # after the 5-tai cap
    raw_tai: int                    # before the cap
    patterns: list = field(default_factory=list)   # [(pattern name, tai), ...]


# --------------------------------------------------------------------------
# Hand shapes
# --------------------------------------------------------------------------
@lru_cache(maxsize=200_000)
def _all_sets(counts):
    """Can these tiles (a tuple of 34 counts) be split completely into pongs and chows?"""
    first = next((t for t in range(NUM_TILE_TYPES) if counts[t] > 0), None)
    if first is None:
        return True
    c = list(counts)
    # Option 1: the lowest tile starts a pong.
    if c[first] >= 3:
        c[first] -= 3
        if _all_sets(tuple(c)):
            return True
        c[first] += 3
    # Option 2: the lowest tile starts a chow (only suited tiles, and not 8 or 9).
    if is_suited(first) and number(first) <= 7 and c[first + 1] > 0 and c[first + 2] > 0:
        c[first] -= 1
        c[first + 1] -= 1
        c[first + 2] -= 1
        if _all_sets(tuple(c)):
            return True
    return False


def is_thirteen_orphans(counts, n_melds):
    return (n_melds == 0
            and all(counts[t] >= 1 for t in ORPHANS)
            and sum(counts[t] for t in ORPHANS) == 14)


def is_winning_shape(counts, n_melds):
    """
    counts: the concealed tiles (34 numbers), INCLUDING the winning tile.
    n_melds: how many sets are already declared on the table.
    True if the tiles make (4 - n_melds) sets + 1 pair, or Thirteen Orphans.
    """
    if sum(counts) != 14 - 3 * n_melds:
        return False
    if is_thirteen_orphans(counts, n_melds):
        return True
    for pair in range(NUM_TILE_TYPES):
        if counts[pair] >= 2:
            rest = list(counts)
            rest[pair] -= 2
            if _all_sets(tuple(rest)):
                return True
    return False


def readings(counts):
    """
    Every way to read the concealed tiles as one pair + sets.
    Returns a list of (pair_tile, [("pong" or "chow", tile), ...]).
    For a chow, the tile is the lowest of the run.
    """
    results = []

    def split(c, start, sets):
        t = start
        while t < NUM_TILE_TYPES and c[t] == 0:
            t += 1
        if t == NUM_TILE_TYPES:                      # everything used up: a full reading
            results.append((pair, list(sets)))
            return
        if c[t] >= 3:                                # read the lowest tile as a pong
            c[t] -= 3
            sets.append(("pong", t))
            split(c, t, sets)
            sets.pop()
            c[t] += 3
        if is_suited(t) and number(t) <= 7 and c[t + 1] > 0 and c[t + 2] > 0:
            for k in range(3):                       # ... or as the start of a chow
                c[t + k] -= 1
            sets.append(("chow", t))
            split(c, t, sets)
            sets.pop()
            for k in range(3):
                c[t + k] += 1

    for pair in range(NUM_TILE_TYPES):
        if counts[pair] >= 2:
            c = list(counts)
            c[pair] -= 2
            split(c, 0, [])
    return results


def is_nine_gates(counts, n_melds):
    """Fully concealed 1112345678999 in one suit, plus any one more tile of that suit."""
    if n_melds > 0:
        return False
    suits = {suit(t) for t in range(NUM_TILE_TYPES) if counts[t] > 0}
    if len(suits) != 1 or None in suits:
        return False
    base = 9 * suits.pop()
    c = counts[base:base + 9]
    return c[0] >= 3 and c[8] >= 3 and all(x >= 1 for x in c[1:8])


# --------------------------------------------------------------------------
# Tai
# --------------------------------------------------------------------------
def bonus_patterns(bonus, seat_wind):
    """
    Tai from bonus tiles. Returns (patterns, has_scoring_bonus, only_plain_flowers).
      - Flower matching your seat: 1 tai per set (seasons, plants). East = f1/g1, South = f2/g2 ...
      - All 4 of one set: 2 tai, replacing that set's matching-flower tai.
      - Each animal: 1 tai.
    "Plain flowers" are flowers that don't match your seat; they matter for ping hu.
    """
    patterns = []
    for group in (SEASONS, PLANTS):
        held = [b for b in bonus if b in group]
        if len(held) == 4:
            patterns.append(("Complete flower set", 2))
        elif group[seat_wind] in held:
            patterns.append(("Flower matching seat", 1))
    for b in bonus:
        if b in ANIMALS:
            patterns.append(("Animal", 1))
    has_scoring = len(patterns) > 0
    only_plain_flowers = len(bonus) > 0 and not has_scoring
    return patterns, has_scoring, only_plain_flowers


def _context_patterns(ctx):
    """Tai that depend on how the hand was won rather than what it contains."""
    patterns = []
    if ctx.kong_replacement:
        patterns.append(("Win on kong replacement", 1))
    if ctx.robbing_kong:
        patterns.append(("Robbing a kong", 1))
    if ctx.last_tile:
        patterns.append(("Win on last tile", 1))
    if ctx.heavenly:
        patterns.append(("Heavenly hand", 5))
    if ctx.earthly:
        patterns.append(("Earthly hand", 5))
    return patterns


def _score_reading(pair, sets, win_set, ctx, nine_gates):
    """
    Tai patterns for ONE reading of the hand.
      pair:     the pair's tile
      sets:     all 4 sets, each a dict with keys
                kind ("pong"/"chow"), tile, kong, concealed, declared
      win_set:  index in `sets` of the set the winning tile completed, or None if it
                completed the pair
    """
    seat_wind_tile = wind_tile(ctx.seat_wind)
    prevailing_tile = wind_tile(ctx.prevailing)
    pongs = [s for s in sets if s["kind"] == "pong"]          # kongs count as pongs
    chows = [s for s in sets if s["kind"] == "chow"]
    hand_is_open = any(s["declared"] and not s["concealed"] for s in sets)

    patterns, bonus_scores, plain_flowers_only = bonus_patterns(ctx.bonus, ctx.seat_wind)
    patterns = list(patterns)

    # Dragons
    dragon_pongs = [s for s in pongs if s["tile"] in DRAGONS]
    if len(dragon_pongs) == 3:
        patterns.append(("Big three dragons", 5))
    elif len(dragon_pongs) == 2 and pair in DRAGONS:
        patterns.append(("Small three dragons", 4))           # replaces the dragon pong tai
    else:
        patterns += [("Dragon pong", 1)] * len(dragon_pongs)

    # Winds
    wind_pongs = [s for s in pongs if s["tile"] in WINDS]
    if len(wind_pongs) == 4:
        patterns.append(("Big four winds", 5))
    elif len(wind_pongs) == 3 and pair in WINDS:
        patterns.append(("Small four winds", 4))              # replaces the wind pong tai
    else:
        for s in wind_pongs:
            if s["tile"] == seat_wind_tile:
                patterns.append(("Seat wind pong", 1))
            if s["tile"] == prevailing_tile:                  # stacks if both are the same wind
                patterns.append(("Prevailing wind pong", 1))

    # Concealed hand
    if not hand_is_open:
        patterns.append(("Men qing (fully concealed)", 1))

    # All pongs / four concealed pongs / four kongs
    if len(pongs) == 4:
        patterns.append(("All pongs (pong pong hu)", 2))
        if all(s["concealed"] for s in pongs):
            patterns.append(("Four concealed pongs", 5))
    if sum(1 for s in sets if s["kong"]) == 4:
        patterns.append(("Four kongs", 5))

    # Flushes and all honours
    tiles = [pair] + [s["tile"] for s in sets]
    suits_used = {suit(t) for t in tiles if is_suited(t)}
    has_honours = any(not is_suited(t) for t in tiles)
    if not suits_used:
        patterns.append(("All honours", 5))
    elif len(suits_used) == 1:
        if has_honours:
            patterns.append(("Half flush", 2))
        else:
            patterns.append(("Full flush", 5))

    if nine_gates:
        patterns.append(("Nine gates", 5))

    # Ping hu: 4 chows, a plain pair, won on a two-sided wait.
    plain_pair = pair not in DRAGONS and pair != seat_wind_tile and pair != prevailing_tile
    if len(chows) == 4 and plain_pair and win_set is not None:
        ws = sets[win_set]
        if ws["kind"] == "chow" and ws["concealed"]:
            low, win = ws["tile"], ctx.win_tile
            # Two-sided: won on the low end of a run that isn't 7-8-9, or the high end of a
            # run that isn't 1-2-3. (Edge, middle and pair waits don't count.)
            two_sided = (win == low and number(low) <= 6) or (win == low + 2 and number(low) >= 2)
            if two_sided and not bonus_scores:
                if not hand_is_open:
                    patterns.append(("Ping hu", 4))           # + men qing above = 5
                elif not ctx.bonus:
                    patterns.append(("Ping hu", 4))
                elif plain_flowers_only:
                    patterns.append(("Ping hu (open, non-matching flowers)", 1))

    patterns += _context_patterns(ctx)
    return patterns


def score_hand(counts, melds, ctx):
    """
    Score a winning hand.
      counts: concealed tiles (34 numbers) INCLUDING the winning tile
      melds:  list of Meld already on the table
      ctx:    WinContext
    Returns a Score, or None if the tiles are not a winning shape.
    The caller still has to check score.tai >= MIN_TAI before allowing the win.
    """
    if not is_winning_shape(counts, len(melds)):
        return None

    candidates = []                                            # one list of patterns per reading

    if is_thirteen_orphans(counts, len(melds)):
        patterns, _, _ = bonus_patterns(ctx.bonus, ctx.seat_wind)
        candidates.append(list(patterns) + [("Thirteen orphans", 5), ("Men qing (fully concealed)", 1)]
                          + _context_patterns(ctx))

    declared = [{"kind": "chow" if m.kind == "chow" else "pong", "tile": m.tile, "kong": m.is_kong(),
                 "concealed": not m.is_open(), "declared": True} for m in melds]
    nine_gates = is_nine_gates(counts, len(melds))

    for pair, hidden_sets in readings(counts):
        hidden = [{"kind": k, "tile": t, "kong": False, "concealed": True, "declared": False}
                  for k, t in hidden_sets]
        # Which part did the winning tile complete? Try every possibility.
        places = []
        if pair == ctx.win_tile:
            places.append(None)
        for i, s in enumerate(hidden):
            if (s["kind"] == "pong" and s["tile"] == ctx.win_tile) or \
               (s["kind"] == "chow" and s["tile"] <= ctx.win_tile <= s["tile"] + 2):
                places.append(len(declared) + i)
        for place in places:
            sets = [dict(s) for s in declared + hidden]
            if place is not None and not ctx.self_draw and sets[place]["kind"] == "pong":
                sets[place]["concealed"] = False    # a pong finished with a discard counts as exposed
            candidates.append(_score_reading(pair, sets, place, ctx, nine_gates))

    if not candidates:
        return None
    best = None
    for patterns in candidates:
        raw = sum(t for _, t in patterns)
        score = Score(tai=min(raw, MAX_TAI), raw_tai=raw, patterns=patterns)
        if best is None or (score.tai, score.raw_tai) > (best.tai, best.raw_tai):
            best = score
    return best


# --------------------------------------------------------------------------
# Payments (points only, no money)
# --------------------------------------------------------------------------
def payments(tai, winner, discarder):
    """
    Point changes for the 4 seats.
      Win on a discard: the discarder pays 4, 8, 16, 32, 64 for 1-5 tai (for everyone).
      Self-draw: each other player pays 2, 4, 8, 16, 32.
    """
    unit = 2 ** (tai - 1)                     # 1, 2, 4, 8, 16
    deltas = [0, 0, 0, 0]
    if discarder is None:
        for seat in range(4):
            if seat != winner:
                deltas[seat] -= 2 * unit
                deltas[winner] += 2 * unit
    else:
        deltas[discarder] -= 4 * unit
        deltas[winner] += 4 * unit
    return deltas
