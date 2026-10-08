"""
Turn what one player can see (a PlayerView) into the input grid for the network.

The grid has 49 lines, each 34 numbers long (one number per tile type). The line order
is fixed (see CLAUDE.md, "Input grid"); never change it once data has been collected.

    0-3    my hand: have >=1, >=2, >=3, 4 copies
    4-19   discards, 4 lines per player (me, next, opposite, previous)
    20-35  exposed melds, 4 lines per player
    36-39  copies visible anywhere (my hand + all discards + all melds)
    40     last tile discarded by anyone (one-hot)
    41     my seat wind (one-hot)
    42     prevailing wind (one-hot)
    43     tiles left in the wall / 148 (the same number all along the line)
    44-47  bonus tiles shown, one line per player: columns 0-11 = f1-f4, g1-g4, a1-a4
    48     dealer: columns 0-3 = me, next, opposite, previous

"One-hot" means a line of zeros with a single 1 marking one tile.
Everything in the grid is between 0 and 1.
"""

import torch

from engine.tiles import FIRST_BONUS, NUM_TILE_TYPES, WALL_SIZE

NUM_PLANES = 49


def counts(tiles):
    c = [0] * NUM_TILE_TYPES
    for t in tiles:
        c[t] += 1
    return c


def at_least(c, k):
    """1 for every tile type with at least k copies, else 0."""
    return [1.0 if c[t] >= k else 0.0 for t in range(NUM_TILE_TYPES)]


def one_hot(position):
    line = [0.0] * NUM_TILE_TYPES
    if position is not None:
        line[position] = 1.0
    return line


def encode(view):
    """
    Returns (state, mask):
      state: tensor [49, 34], float16
      mask:  tensor [34], bool, True where discarding that tile is legal
    """
    lines = []

    hand_c = counts(view.hand)
    lines += [at_least(hand_c, k) for k in (1, 2, 3, 4)]                    # 0-3

    for pile in view.discards:                                              # 4-19
        lines += [at_least(counts(pile), k) for k in (1, 2, 3, 4)]

    for meld_tiles in view.melds:                                           # 20-35
        lines += [at_least(counts(meld_tiles), k) for k in (1, 2, 3, 4)]

    seen = list(view.hand)                                                  # 36-39
    for pile in view.discards:
        seen += pile
    for meld_tiles in view.melds:
        seen += meld_tiles
    lines += [at_least(counts(seen), k) for k in (1, 2, 3, 4)]

    lines.append(one_hot(view.last_discard))                                # 40
    lines.append(one_hot(view.seat_wind))                                   # 41
    lines.append(one_hot(view.prevailing_wind))                             # 42
    lines.append([view.wall_left / WALL_SIZE] * NUM_TILE_TYPES)             # 43

    for bonus in view.bonus:                                                # 44-47
        line = [0.0] * NUM_TILE_TYPES
        for b in bonus:
            line[b - FIRST_BONUS] = 1.0                                     # f1 -> column 0 ... a4 -> 11
        lines.append(line)

    lines.append(one_hot(view.dealer))                                      # 48

    assert len(lines) == NUM_PLANES

    # Legal discards: every tile type in my hand (when it's my turn to discard).
    mask = [hand_c[t] > 0 for t in range(NUM_TILE_TYPES)]
    return torch.tensor(lines, dtype=torch.float16), torch.tensor(mask)
