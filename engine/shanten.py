"""
Shanten: how many tiles a hand is away from being ready to win.

    shanten = -1   the hand is already a winning shape
    shanten =  0   "ready": one more tile wins
    shanten =  1   two tiles away, and so on

How it's worked out (the standard method):
  Split the concealed tiles into
    - sets      (pong or chow)                         call the number m
    - partials  (two tiles that need one more: a pair,
                 4-5 waiting on 3/6, 4-6 waiting on 5)   call the number t
    - one pair kept as the hand's pair ("head")          p = 1 if there is one
  If the hand still needs n sets (4 minus melds already on the table):
      shanten = 2*n - 2*m - t - p,   counting at most n - m partials.
  Try every possible split and keep the smallest answer.

Thirteen Orphans is a special hand, so it gets its own small formula.
Seven pairs is not a winning hand in our rules, so it is not counted.

Each suit is split separately (results are cached, so this is fast), then the
suits are combined.
"""

from functools import lru_cache

from engine.tiles import NUM_TILE_TYPES, ORPHANS, is_suited


@lru_cache(maxsize=None)
def _group_options(counts, suited):
    """
    All ways to split one suit (9 counts) or the honours (7 counts).
    Returns a dict {(p, m): best t}: for each number of heads p (0 or 1) and sets m,
    the most partial sets t that can be made alongside.
    """
    first = next((i for i, c in enumerate(counts) if c > 0), None)
    if first is None:
        return {(0, 0): 0}

    results = {}

    def keep(sub_counts, dp, dm, dt):
        """Remove some tiles, split the rest, and add (dp, dm, dt) to every result."""
        for (p, m), t in _group_options(tuple(sub_counts), suited).items():
            key = (p + dp, m + dm)
            if key[0] > 1 or key[1] > 4:
                continue
            new_t = min(t + dt, 4)
            if results.get(key, -1) < new_t:
                results[key] = new_t

    c = list(counts)
    i = first

    # Leave one copy of this tile unused.
    c[i] -= 1
    keep(c, 0, 0, 0)
    c[i] += 1
    if c[i] >= 3:                                    # pong
        c[i] -= 3
        keep(c, 0, 1, 0)
        c[i] += 3
    if c[i] >= 2:                                    # pair, used as the head ...
        c[i] -= 2
        keep(c, 1, 0, 0)
        keep(c, 0, 0, 1)                             # ... or as a partial set
        c[i] += 2
    if suited:
        pos = i                                      # position 0-8 within the suit
        if pos <= 6 and c[i + 1] > 0 and c[i + 2] > 0:   # chow
            c[i] -= 1; c[i + 1] -= 1; c[i + 2] -= 1
            keep(c, 0, 1, 0)
            c[i] += 1; c[i + 1] += 1; c[i + 2] += 1
        if pos <= 7 and c[i + 1] > 0:                # two in a row, e.g. 4-5
            c[i] -= 1; c[i + 1] -= 1
            keep(c, 0, 0, 1)
            c[i] += 1; c[i + 1] += 1
        if pos <= 6 and c[i + 2] > 0:                # with a gap, e.g. 4-6
            c[i] -= 1; c[i + 2] -= 1
            keep(c, 0, 0, 1)
            c[i] += 1; c[i + 2] += 1
    return results


def _combine(a, b):
    """Merge the options of two groups of tiles (at most one head in total)."""
    out = {}
    for (p1, m1), t1 in a.items():
        for (p2, m2), t2 in b.items():
            key = (p1 + p2, m1 + m2)
            if key[0] > 1 or key[1] > 4:
                continue
            t = min(t1 + t2, 4)
            if out.get(key, -1) < t:
                out[key] = t
    return out


def standard_shanten(counts, sets_needed):
    """Shanten for the normal shape: `sets_needed` sets + 1 pair."""
    options = _group_options(tuple(counts[0:9]), True)
    options = _combine(options, _group_options(tuple(counts[9:18]), True))
    options = _combine(options, _group_options(tuple(counts[18:27]), True))
    options = _combine(options, _group_options(tuple(counts[27:34]), False))
    n = sets_needed
    best = 8
    for (p, m), t in options.items():
        if m > n:
            continue
        t = min(t, n - m)
        best = min(best, 2 * n - 2 * m - t - p)
    return best


def orphans_shanten(counts):
    """Thirteen Orphans: one of each terminal and honour, plus a pair of any of them."""
    kinds = sum(1 for t in ORPHANS if counts[t] > 0)
    has_pair = any(counts[t] >= 2 for t in ORPHANS)
    return 13 - kinds - (1 if has_pair else 0)


# Remembers recent answers. Kept fairly small: most hands are never seen twice, and a
# million entries would use about 1 GB of memory.
@lru_cache(maxsize=100_000)
def _shanten_cached(counts, n_melds):
    s = standard_shanten(counts, 4 - n_melds)
    if n_melds == 0:
        s = min(s, orphans_shanten(counts))
    return s


def shanten(counts, n_melds=0):
    """
    counts:  concealed tiles (34 numbers)
    n_melds: sets already on the table (pongs, chows, kongs)
    """
    return _shanten_cached(tuple(counts), n_melds)


def useful_tiles(counts, n_melds, available):
    """
    For a hand waiting to draw (13, 10, 7 ... concealed tiles): which tile types would
    lower the shanten if drawn, and how many copies of them could still come.
      available[t]: copies of tile t that might still be drawn (the caller decides
                    what it counts, e.g. 4 minus the copies the player holds itself)
    Returns (total copies, list of useful tile types).
    """
    now = shanten(counts, n_melds)
    total, useful = 0, []
    c = list(counts)
    for t in range(NUM_TILE_TYPES):
        if available[t] <= 0 or not _could_help(counts, t):
            continue
        c[t] += 1
        if shanten(c, n_melds) < now:
            total += available[t]
            useful.append(t)
        c[t] -= 1
    return total, useful


def _could_help(counts, t):
    """A drawn tile can only help if the hand holds it, or a neighbour within 2 in the same suit,
    or it is a terminal/honour (for Thirteen Orphans)."""
    if counts[t] > 0 or t in ORPHANS:
        return True
    if is_suited(t):
        for d in (-2, -1, 1, 2):
            u = t + d
            if 0 <= u < 27 and u // 9 == t // 9 and counts[u] > 0:
                return True
    return False
