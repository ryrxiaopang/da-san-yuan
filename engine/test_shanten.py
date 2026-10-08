"""
Shanten tests.

Run:  python -m pytest engine -q
"""

import random

from engine.scoring import is_winning_shape
from engine.shanten import shanten, useful_tiles
from engine.tiles import counts_of, full_set, parse_tiles


def sh(text, n_melds=0):
    return shanten(counts_of(parse_tiles(text)), n_melds)


def test_known_hands():
    assert sh("123m 456p 789s 123s 99m") == -1          # complete
    assert sh("123m 456p 789s 123s 9m") == 0            # ready, waiting on 9m
    assert sh("123m 456p 789s 12s 99m") == 0            # ready, waiting on 3s
    assert sh("123m 456p 789s 1s 3s 9m") == 1           # 3 sets, a 1-3 gap and a single
    assert sh("19m 19p 19s E S W N Rd Gr Wh") == 0      # thirteen orphans, 13-sided wait
    assert sh("135m 246p 789s E S W N") == 4
    assert sh("11m 22m 33p 44p 55s 66s N") == 3         # seven pairs doesn't count in our rules


def test_melds_reduce_what_is_needed():
    # One meld on the table: the concealed part needs 3 sets + a pair.
    assert sh("123m 456p 789s 9m", n_melds=1) == 0
    assert sh("123m 456p 78s 99m", n_melds=1) == 0
    assert sh("5p", n_melds=4) == 0                    # four melds: waiting on the pair


def test_agrees_with_win_detection_on_random_hands():
    """Shanten -1 must mean a winning shape, and shanten 0 must mean some tile completes it."""
    rng = random.Random(0)
    playing = [t for t in full_set() if t < 34]
    for _ in range(3000):
        tiles = rng.sample(playing, 14)
        c = counts_of(tiles)
        assert (shanten(c) == -1) == is_winning_shape(c, 0)
        c[tiles[0]] -= 1                                 # a 13-tile hand
        completes = any(c[t] < 4 and is_winning_shape([x + (i == t) for i, x in enumerate(c)], 0)
                        for t in range(34))
        assert (shanten(c) == 0) == completes


def test_one_draw_changes_shanten_by_at_most_one():
    rng = random.Random(1)
    playing = [t for t in full_set() if t < 34]
    for _ in range(1000):
        c = counts_of(rng.sample(playing, 13))
        before = shanten(c)
        for t in range(34):
            if c[t] < 4:
                c[t] += 1
                assert before - 1 <= shanten(c) <= before
                c[t] -= 1


def test_useful_tiles():
    c = counts_of(parse_tiles("123m 456p 789s 12s 99m"))        # ready on 3s only
    total, tiles = useful_tiles(c, 0, [4] * 34)
    assert tiles == parse_tiles("3s") and total == 4
    total, _ = useful_tiles(c, 0, [4 - x for x in c])           # counting my own copies
    assert total == 4
