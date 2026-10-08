"""
Scoring tests: one or more per rule in RULES.md.

Run:  python -m pytest engine -q

Each expected number was worked out by hand from RULES.md, not copied from the code.
"""

import pytest

from engine.scoring import Meld, WinContext, payments, score_hand, is_winning_shape
from engine.tiles import counts_of, parse_tiles


def tile(name):
    return parse_tiles(name)[0]


def score(hand, win, melds=(), seat=0, prevailing=0, self_draw=False, **flags):
    """Score a hand written as text. `hand` includes the winning tile and any bonus tiles."""
    tiles = parse_tiles(hand)
    ctx = WinContext(seat_wind=seat, prevailing=prevailing, bonus=[t for t in tiles if t >= 34],
                     self_draw=self_draw, win_tile=tile(win), **flags)
    return score_hand(counts_of(tiles), list(melds), ctx)


def pong(name):
    return Meld("pong", tile(name), from_seat=1)


def chow(low):
    return Meld("chow", tile(low), from_seat=3)


def names(s):
    return [name for name, _ in s.patterns]


# ------------------------------------------------------------ shapes
def test_winning_shapes():
    assert is_winning_shape(counts_of(parse_tiles("123m 456p 789s 123s 99m")), 0)
    assert not is_winning_shape(counts_of(parse_tiles("123m 456p 789s 123s 9m 8m")), 0)
    # Seven pairs is not a winning hand.
    assert score("11m 22m 33p 44p 55s 66s E E", "E") is None


# ------------------------------------------------------------ simple patterns
def test_men_qing_is_one_tai():
    # Concealed, no other source of tai, closed wait on 3s.
    s = score("123m 456p 789s 234s 99m", "3s", seat=1)
    assert names(s) == ["Men qing (fully concealed)"] and s.tai == 1


def test_open_hand_without_tai_is_zero():
    s = score("456p 789s 234s 99m", "3s", [chow("1m")], seat=1)
    assert s.tai == 0


def test_dragon_and_wind_pongs():
    # Seat East, round East: an East pong counts twice; a dragon pong once.
    s = score("123m 456p 22p E E E", "2p", [pong("Rd")], seat=0, prevailing=0)
    assert sorted(names(s)) == ["Dragon pong", "Prevailing wind pong", "Seat wind pong"]
    assert s.tai == 3


def test_all_pongs_two_tai():
    s = score("222m 555p 888s 99s", "9s", [pong("4p")], seat=1)
    assert "All pongs (pong pong hu)" in names(s) and s.tai == 2


def test_half_flush_two_tai():
    # Seat West, round East: the East pong is the prevailing wind.
    s = score("123m 456m 789m 22m E E E", "2m", seat=2)
    assert "Half flush" in names(s) and "Prevailing wind pong" in names(s)


def test_full_flush_is_five_tai():
    s = score("111m 234m 567m 888m 99m", "9m", [], seat=1)
    assert "Full flush" in names(s) and s.tai == 5


# ------------------------------------------------------------ ping hu (see RULES.md table)
def test_ping_hu_concealed_is_five():
    s = score("123m 456p 789s 123s 99m", "1s", seat=1)   # won 1s on 2s-3s: two-sided
    assert sorted(names(s)) == ["Men qing (fully concealed)", "Ping hu"] and s.raw_tai == 5


def test_ping_hu_needs_two_sided_wait():
    s = score("123m 456p 789s 123s 99m", "3s", seat=1)   # 3s on 1s-2s: edge wait
    assert "Ping hu" not in names(s)


def test_ping_hu_concealed_with_plain_flower_keeps_five():
    s = score("123m 456p 789s 123s 99m f1", "1s", seat=1)  # seat South: f1 doesn't match
    assert "Ping hu" in names(s) and s.raw_tai == 5


def test_open_ping_hu():
    s = score("456p 789s 123s 99m", "1s", [chow("1m")], seat=1)
    assert names(s) == ["Ping hu"] and s.raw_tai == 4
    s = score("456p 789s 123s 99m f1", "1s", [chow("1m")], seat=1)
    assert names(s) == ["Ping hu (open, non-matching flowers)"] and s.raw_tai == 1


def test_ping_hu_void_with_scoring_bonus():
    s = score("123m 456p 789s 123s 99m f2", "1s", seat=1)  # f2 matches South: flower scores instead
    assert "Ping hu" not in names(s) and "Flower matching seat" in names(s)
    s = score("123m 456p 789s 123s 99m a3", "1s", seat=1)  # animals void it too
    assert "Ping hu" not in names(s)


def test_ping_hu_pair_cannot_be_valued_honour():
    s = score("123m 456p 789s 123s Rd Rd", "1s", seat=1)
    assert "Ping hu" not in names(s)


# ------------------------------------------------------------ dragons, winds, limits
def test_small_three_dragons_replaces_dragon_pongs():
    s = score("Rd Rd Rd Gr Gr Gr Wh Wh 123m", "1m", [pong("9s")], seat=0, prevailing=1)
    assert names(s) == ["Small three dragons"] and s.tai == 4


def test_small_four_winds():
    # 4 for small four winds + 2 for half flush = 6, capped at 5.
    s = score("W W W N N 456p", "6p", [pong("E"), pong("S")], seat=3, prevailing=0)
    assert "Small four winds" in names(s) and s.tai == 5


@pytest.mark.parametrize("hand,win,melds,self_draw,pattern", [
    ("19m 19p 19s E S W N Rd Gr Wh Wh", "Wh", [], False, "Thirteen orphans"),
    ("Rd Rd Rd Gr Gr Gr Wh Wh Wh 123m 99p", "9p", [], False, "Big three dragons"),
    ("E E E S S S W W W N N N 99p", "9p", [], True, "Big four winds"),
    ("E E E S S S W W W Rd Rd Rd Gr Gr", "Gr", [], True, "All honours"),
    ("55p", "5p", [Meld("exposed_kong", 0, 1), Meld("exposed_kong", 19, 2),
                   Meld("exposed_kong", 17, 3), Meld("concealed_kong", 3)], False, "Four kongs"),
    ("111m 2345m 5678m 999m", "5m", [], False, "Nine gates"),
])
def test_limit_hands_are_five(hand, win, melds, self_draw, pattern):
    s = score(hand, win, melds, seat=1, self_draw=self_draw)
    assert pattern in names(s) and s.tai == 5


def test_four_concealed_pongs_needs_self_draw_or_pair_wait():
    s = score("222m 555p 888s 999s 11p", "9s", seat=1)                  # pong finished on a discard
    assert "Four concealed pongs" not in names(s) and s.tai == 3        # all pongs 2 + men qing 1
    s = score("222m 555p 888s 999s 11p", "9s", seat=1, self_draw=True)
    assert "Four concealed pongs" in names(s)
    s = score("222m 555p 888s 999s 11p", "1p", seat=1)                  # pair wait on a discard is fine
    assert "Four concealed pongs" in names(s)


# ------------------------------------------------------------ bonus tiles and context
def test_bonus_tiles():
    # Seat East: f1 and g1 match (1 each), a1 is an animal (1), f3 doesn't match.
    s = score("123m 456p 789s 234s 99m f1 g1 a1 f3", "3s", seat=0)
    assert s.raw_tai == 1 + 1 + 1 + 1                                   # + men qing
    # A complete set of seasons is 2, not 2 + 1.
    s = score("123m 456p 789s 234s 99m f1 f2 f3 f4", "3s", seat=2)
    assert "Complete flower set" in names(s) and "Flower matching seat" not in names(s)


def test_context_patterns():
    s = score("123m 456p 789s 234s 99m", "3s", seat=1, self_draw=True, kong_replacement=True, last_tile=True)
    assert {"Win on kong replacement", "Win on last tile"} <= set(names(s))
    s = score("123m 456p 789s 234s 99m", "3s", seat=1, robbing_kong=True)
    assert "Robbing a kong" in names(s)


def test_tai_cap_is_five():
    s = score("111m 234m 567m 888m 99m f1 a1 a2", "9m", seat=0, self_draw=True)
    assert s.tai == 5 and s.raw_tai > 5


# ------------------------------------------------------------ the 19 hand-worked hands from the old project
@pytest.mark.parametrize("hand,win,melds,seat,prev,self_draw,expected", [
    ("W W W N N 456p", "6p", ["pong E", "pong S"], 3, 0, False, 5),        # small four winds + half flush
    ("N N N 55m", "5m", ["pong E", "pong S", "pong W"], 0, 0, False, 5),   # big four winds
    ("E E E S S S W W W Rd Rd Rd Gr Gr", "Gr", [], 2, 1, True, 5),         # all honours
    ("55p", "5p", ["kong 1m", "kong 2s", "kong 9p", "ckong 4m"], 1, 0, False, 5),
    ("111m 234m 556m 789m 99m", "5m", [], 1, 0, False, 5),                 # full flush (nine-gates-like)
    ("456p 789s 234s 55m", "2s", ["chow 1m"], 0, 0, False, 4),             # open ping hu
    ("456p 789s 234s 55m f3", "2s", ["chow 1m"], 0, 0, False, 1),          # open ping hu, plain flower
    ("123m 456p 789s 234s 55m f3", "2s", [], 0, 0, False, 5),              # concealed ping hu, plain flower
    ("123m 456p 789s 234s 55m", "3m", [], 0, 0, False, 1),                 # edge wait: men qing only
    ("123m 456p 789s 234s 55m f1", "2s", [], 0, 0, False, 2),              # matching flower replaces ping hu
    ("123m 456p 789s 234s S S", "2s", [], 1, 0, False, 1),                 # seat-wind pair: no ping hu
    ("Rd Rd Rd Gr Gr Gr Wh Wh 123m", "1m", ["pong 9s"], 0, 1, False, 4),   # small three dragons
    ("123m 456p 789s 11p", "1p", ["pong E"], 0, 0, False, 2),              # seat + prevailing wind
    ("123m 456p 789s 234s 99m f1 f2 f3 f4", "9m", [], 0, 0, True, 3),      # flower set 2 + men qing
    ("123m 456m 789m E E", "E", ["pong Rd"], 1, 0, False, 3),              # half flush 2 + dragon 1
    ("222p 333s 99m", "9m", ["pong 1m", "pong 4p"], 1, 0, False, 2),       # all pongs
    ("111m 222p 333s 444m S S", "4m", [], 0, 0, False, 3),                 # pongs on a discard: not concealed
    ("111m 222p 333s 444m S S", "4m", [], 0, 0, True, 5),                  # four concealed pongs
    ("11m 22m 33p 44p 55s 66s N N", "N", [], 0, 0, False, None),           # seven pairs: not a win
])
def test_hand_worked_examples(hand, win, melds, seat, prev, self_draw, expected):
    kinds = {"pong": "pong", "chow": "chow", "kong": "exposed_kong", "ckong": "concealed_kong"}
    meld_list = [Meld(kinds[m.split()[0]], tile(m.split()[1]), None if m.startswith("ckong") else 1)
                 for m in melds]
    s = score(hand, win, meld_list, seat=seat, prevailing=prev, self_draw=self_draw)
    assert (None if s is None else s.tai) == expected, s


# ------------------------------------------------------------ payments
def test_payments_match_team_table():
    for tai, amount in [(1, 4), (2, 8), (3, 16), (4, 32), (5, 64)]:
        assert payments(tai, winner=0, discarder=2) == [amount, 0, -amount, 0]
    for tai, each in [(1, 2), (2, 4), (3, 8), (4, 16), (5, 32)]:
        assert payments(tai, winner=1, discarder=None) == [-each, 3 * each, -each, -each]
