"""
Game-engine tests.

Run:  python -m pytest engine -q
      python -m pytest engine -q -k random --hands 10000   (longer random run)
"""

import random

from engine.game import PASS, WIN, Action, Game, HandResult, TableState, RESERVE
from engine.scoring import MIN_TAI
from engine.tiles import full_set


def check_census(game):
    """Every playing tile exactly 4 times, every bonus tile once, wherever it is."""
    seen = game.tile_census()
    expected = {}
    for t in full_set():
        expected[t] = expected.get(t, 0) + 1
    assert seen == expected, "a tile appeared or disappeared"


def play_random_hand(seed, take_wins=True):
    """Play one hand with every seat choosing random legal moves. Checks invariants at every step."""
    rng = random.Random(seed)
    game = Game(dealer=seed % 4, prevailing=(seed // 4) % 4, seed=seed)
    steps = 0
    while not game.is_over():
        check_census(game)
        seat = game.to_act()
        options = game.legal_actions(seat)
        assert options, "the seat to act must have at least one legal move"
        if take_wins and WIN in options:
            action = WIN
        else:
            action = rng.choice(options)
        game.apply(seat, action)
        steps += 1
        assert steps < 1000, "hand never ended"
    check_census(game)
    return game


# ------------------------------------------------------------ dealing
def test_deal():
    game = Game(dealer=2, seed=42)
    sizes = [p.hand_size() for p in game.players]
    assert sizes == [13, 13, 14, 13]                       # the dealer (seat 2) has 14
    for p in game.players:
        assert all(t >= 34 for t in p.bonus)               # bonus tiles are set aside
    check_census(game)
    assert game.to_act() == 2 and game.phase == "turn"


def test_same_seed_same_hand():
    a = play_random_hand(7)
    b = play_random_hand(7)
    assert a.result == b.result


def test_illegal_move_is_refused():
    game = Game(seed=1)
    missing = next(t for t in range(34) if game.players[0].hand[t] == 0)
    try:
        game.apply(0, Action("discard", missing))
        assert False, "discarding a tile you don't have must fail"
    except ValueError:
        pass


# ------------------------------------------------------------ random play
def test_random_hands_keep_every_rule(hands=2000):
    """Thousands of random hands: tiles never appear or vanish, wins have >= 1 tai,
    points always add to zero, and draws happen only when the wall reaches 15 tiles."""
    wins = draws = 0
    for seed in range(hands):
        game = play_random_hand(seed)
        r = game.result
        assert sum(r.deltas) == 0
        if r.winner is None:
            draws += 1
            assert game.live_tiles() <= RESERVE + 1
        else:
            wins += 1
            assert r.score.tai >= MIN_TAI
    assert wins + draws == hands
    assert wins > 0 and draws > 0


# ------------------------------------------------------------ claims
def test_only_next_player_may_chow_and_win_beats_pong():
    """Search random hands for a discard that one player can win on while another can pong,
    answer win + pong, and check the win is the one that happens."""
    found = False
    for seed in range(3000):
        rng = random.Random(seed)
        game = Game(seed=seed)
        while not game.is_over():
            if game.phase == "claim":
                for s in range(4):
                    if any(a.kind == "chow" for a in game.legal_actions(s)):
                        assert s == (game.offer_from + 1) % 4
                winners = [s for s in game.waiting if WIN in game.legal_actions(s)]
                pongers = [s for s in game.waiting if Action("pong") in game.legal_actions(s)
                           and s not in winners]
                if winners and pongers:
                    while not game.is_over() and game.phase == "claim":
                        s = game.to_act()
                        a = WIN if s in winners else Action("pong") if s in pongers else PASS
                        game.apply(s, a)
                    assert game.result.winner in winners
                    found = True
                    break
            seat = game.to_act()
            options = game.legal_actions(seat)
            game.apply(seat, WIN if WIN in options else rng.choice(options))
        if found:
            break
    assert found, "no win-versus-pong situation found"


# ------------------------------------------------------------ dealer rule
def test_dealer_rule():
    table = TableState()
    win_by = lambda s: HandResult(winner=s)
    drawn = HandResult(winner=None)

    table.next_hand(win_by(0), any_kong=False)       # dealer wins: stays
    assert (table.dealer, table.repeat) == (0, 1)
    table.next_hand(drawn, any_kong=False)           # draw, no kong: stays
    assert (table.dealer, table.repeat) == (0, 2)
    table.next_hand(drawn, any_kong=True)            # draw with a kong: passes
    assert (table.dealer, table.repeat, table.dealer_number) == (1, 0, 2)
    table.next_hand(win_by(3), any_kong=False)       # someone else wins: passes
    assert (table.dealer, table.dealer_number) == (2, 3)
    table.next_hand(win_by(0), any_kong=False)
    table.next_hand(win_by(0), any_kong=False)       # 4th dealer passes: South round
    assert (table.dealer, table.prevailing, table.dealer_number) == (0, 1, 1)
    for _ in range(12):                              # through West and North rounds
        table.next_hand(win_by((table.dealer + 1) % 4), any_kong=False)
    assert (table.prevailing, table.game_number) == (0, 2)
