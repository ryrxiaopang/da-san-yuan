"""
Bot tests.

Run:  python -m pytest bots -q
"""

from engine.game import PASS, WIN, Action, Game, TableState
from engine.scoring import Meld
from engine.tiles import EAST, SOUTH, RED, parse_tiles
from engine.view import PlayerView, view_for
from bots.greedy_bot import GreedyBot
from bots.random_bot import RandomBot


def make_view(hand, legal, phase="turn", melds=(), offer=None, seat_wind=EAST, prevailing=EAST):
    """A hand-made PlayerView for testing one decision (opponent fields left empty)."""
    return PlayerView(seat=0, hand=sorted(parse_tiles(hand)), my_melds=list(melds),
                      discards=[[], [], [], []], melds=[[], [], [], []], bonus=[[], [], [], []],
                      last_discard=offer, seat_wind=seat_wind, prevailing_wind=prevailing, wall_left=80,
                      dealer=0, phase=phase, offer_tile=offer, offer_from=1 if offer is not None else None,
                      drawn=None, legal=list(legal))


def discards_for(hand):
    return [Action("discard", t) for t in sorted(set(parse_tiles(hand)))]


def tile(name):
    return parse_tiles(name)[0]


# ------------------------------------------------------------ GreedyBot on its own turn
def test_greedy_throws_the_isolated_tile():
    hand = "123m 456p 789s 23s 99m N"                    # North is the only tile doing nothing
    assert GreedyBot().choose_action(make_view(hand, discards_for(hand))) == Action("discard", tile("N"))


def test_greedy_breaks_ties_by_useful_tiles_then_lowest_tile():
    # East and West are equally useless (same shanten and same useful tiles after throwing
    # either), so the tie goes to the lower tile number: East (27) before West (29).
    bot = GreedyBot()
    hand = "123m 456p 789s 11s E W"
    choice = bot.choose_action(make_view(hand, discards_for(hand)))
    assert choice == Action("discard", tile("E"))        # same shanten and useful tiles: lowest number


def test_greedy_wins_and_declares_kongs():
    bot = GreedyBot()
    assert bot.choose_action(make_view("123m 456p 789s 123s 99m", [WIN] + discards_for("1m"))) == WIN
    kong = Action("kong", tile("5p"))
    assert bot.choose_action(make_view("5555p 123m 789s 99m E", [kong] + discards_for("5p 1m E"))) == kong


# ------------------------------------------------------------ GreedyBot on someone's discard
def test_greedy_pongs_valuable_tiles_only_or_when_it_helps():
    bot = GreedyBot()
    options = [Action("pong"), PASS]
    # A pair of Red dragons: a Red pong scores tai, so take it.
    v = make_view("Rd Rd 123m 456p 789s 2s 5s", options, phase="claim", offer=RED)
    assert bot.choose_action(v) == Action("pong")
    # A pair of 9m in an otherwise good hand: ponging doesn't lower shanten, so pass.
    v = make_view("99m 123m 456p 78s 24s", options, phase="claim", offer=tile("9m"))
    assert bot.choose_action(v) == PASS


def test_greedy_never_chows():
    v = make_view("123m 456p 789s 45s E", [Action("chow", tile("3s")), Action("chow", tile("4s")), PASS],
                  phase="claim", offer=tile("6s"))
    assert GreedyBot().choose_action(v) == PASS


# ------------------------------------------------------------ whole hands
def play(bots, hands=100, seed=0):
    table = TableState()
    results = []
    for h in range(hands):
        game = Game(dealer=table.dealer, prevailing=table.prevailing, seed=seed + h)
        while not game.is_over():
            seat = game.to_act()
            view = view_for(game, seat)
            action = bots[seat].choose_action(view)
            assert action in view.legal                  # bots only ever pick legal moves
            game.apply(seat, action)
        results.append(game.result)
        table.next_hand(game.result, game.any_kong())
    return results


def test_greedy_is_predictable():
    a = play([GreedyBot() for _ in range(4)], hands=30)
    b = play([GreedyBot() for _ in range(4)], hands=30)
    assert a == b


def test_greedy_beats_random():
    results = play([GreedyBot(), RandomBot(1), GreedyBot(), RandomBot(2)], hands=150)
    greedy = sum(r.deltas[0] + r.deltas[2] for r in results)
    random_ = sum(r.deltas[1] + r.deltas[3] for r in results)
    assert greedy > 0 > random_


def test_views_hide_other_hands():
    """Swapping hidden tiles between two opponents must not change what seat 0 sees,
    or what GreedyBot decides."""
    game = Game(seed=5)
    other = Game(seed=5)
    p1, p2 = other.players[1], other.players[2]
    a = next(t for t in range(34) if p1.hand[t] > 0 and p2.hand[t] == 0)
    b = next(t for t in range(34) if p2.hand[t] > 0 and p1.hand[t] == 0)
    p1.hand[a] -= 1; p1.hand[b] += 1
    p2.hand[b] -= 1; p2.hand[a] += 1
    assert view_for(game, 0) == view_for(other, 0)
    assert GreedyBot().choose_action(view_for(game, 0)) == GreedyBot().choose_action(view_for(other, 0))


def test_greedy_prefers_more_useful_tiles_over_lower_number():
    # Throwing 1s, 4s, 9s or East all leave shanten 1. Afterwards East leaves 20 useful
    # copies to draw, 1s and 9s leave 17, 4s leaves 13. So East wins the tie-break,
    # even though 1s has the lower tile number.
    hand = "123m 456p 789s 1s 3s 4s 9s E"
    assert GreedyBot().choose_action(make_view(hand, discards_for(hand))) == Action("discard", tile("E"))
