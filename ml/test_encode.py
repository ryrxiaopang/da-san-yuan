"""
Tests for the input grid (ml/encode.py) and the collected data file.

Run:  python -m pytest ml -q
"""

import os
import sys

import torch

from bots.greedy_bot import GreedyBot
from engine.game import Game
from engine.tiles import EAST, SOUTH, parse_tiles
from engine.view import view_for
from ml.encode import NUM_PLANES, encode


def views_from_a_hand(seed=3, stop_after=40):
    """Play a hand with GreedyBots and return the views at each discard decision."""
    game = Game(dealer=1, prevailing=2, seed=seed)
    bots = [GreedyBot() for _ in range(4)]
    views = []
    while not game.is_over() and len(views) < stop_after:
        seat = game.to_act()
        view = view_for(game, seat)
        action = bots[seat].choose_action(view)
        if action.kind == "discard":
            views.append((game, view))
        game.apply(seat, action)
    return views


def line_counts(state, first):
    """Turn 4 'at least k copies' lines back into counts per tile."""
    return [int(state[first:first + 4, t].sum().item()) for t in range(34)]


def test_shape_and_range():
    for _, view in views_from_a_hand():
        state, mask = encode(view)
        assert state.shape == (NUM_PLANES, 34) and state.dtype == torch.float16
        assert mask.shape == (34,) and mask.dtype == torch.bool
        assert state.min() >= 0 and state.max() <= 1


def test_every_line_says_what_claude_md_says():
    for _, view in views_from_a_hand():
        state, mask = encode(view)
        hand = [view.hand.count(t) for t in range(34)]
        assert line_counts(state, 0) == hand                                   # 0-3 hand
        for p in range(4):                                                     # 4-19 discards
            assert line_counts(state, 4 + 4 * p) == [view.discards[p].count(t) for t in range(34)]
        for p in range(4):                                                     # 20-35 melds
            assert line_counts(state, 20 + 4 * p) == [view.melds[p].count(t) for t in range(34)]
        seen = view.hand + sum(view.discards, []) + sum(view.melds, [])       # 36-39 visible
        assert line_counts(state, 36) == [seen.count(t) for t in range(34)]
        assert max(seen.count(t) for t in range(34)) <= 4                      # no tile counted twice
        assert state[40].sum() == (view.last_discard is not None)              # 40 last discard
        if view.last_discard is not None:
            assert state[40, view.last_discard] == 1
        assert state[41, view.seat_wind] == 1 and state[41].sum() == 1         # 41 seat wind
        assert state[42, view.prevailing_wind] == 1 and state[42].sum() == 1   # 42 prevailing wind
        assert abs(state[43, 0].item() - view.wall_left / 148) < 1e-3          # 43 wall
        for p in range(4):                                                     # 44-47 bonus tiles
            cols = sorted(b - 34 for b in view.bonus[p])
            assert torch.nonzero(state[44 + p]).flatten().tolist() == cols
        assert torch.nonzero(state[48]).flatten().tolist() == [view.dealer]    # 48 dealer
        assert mask.tolist() == [h > 0 for h in hand]                          # mask = tiles held


def test_seat_wind_follows_the_dealer():
    # Dealer is seat 1, so seat 1 is East and seat 2 is South.
    game = Game(dealer=1, prevailing=0, seed=1)
    assert view_for(game, 1).seat_wind == EAST
    assert view_for(game, 2).seat_wind == SOUTH
    state, _ = encode(view_for(game, 2))
    assert state[41, SOUTH] == 1 and state[48, 3] == 1        # the dealer is my "previous" player


def test_no_hidden_information():
    """Swapping tiles between two opponents' hidden hands must not change my grid at all."""
    for game, view in views_from_a_hand(stop_after=10):
        me = view.seat
        o1, o2 = game.players[(me + 1) % 4], game.players[(me + 2) % 4]
        a = next((t for t in range(34) if o1.hand[t] > 0 and o2.hand[t] == 0), None)
        b = next((t for t in range(34) if o2.hand[t] > 0 and o1.hand[t] == 0), None)
        if a is None or b is None:
            continue
        before, _ = encode(view_for(game, me))
        o1.hand[a] -= 1; o1.hand[b] += 1
        o2.hand[b] -= 1; o2.hand[a] += 1
        after, _ = encode(view_for(game, me))
        o1.hand[a] += 1; o1.hand[b] -= 1                     # put them back
        o2.hand[b] += 1; o2.hand[a] -= 1
        assert torch.equal(before, after)


# ------------------------------------------------------------ the collected data file
def test_collected_data(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    from collect_data import collect

    data, _ = collect(num_games=20, seed=7, progress_every=10**9)
    n = len(data["labels"])
    assert data["states"].shape == (n, NUM_PLANES, 34)
    for key in ("game_id", "masks", "seat", "turn", "won", "points"):
        assert len(data[key]) == n
    assert data["masks"][torch.arange(n), data["labels"]].all()          # every label was legal

    again, _ = collect(num_games=20, seed=7, progress_every=10**9)       # same seed, same data
    assert torch.equal(data["states"], again["states"]) and torch.equal(data["labels"], again["labels"])

    # The file must load in the behavioural-cloning script as it is.
    path = tmp_path / "small.pt"
    torch.save(data, path)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from train_bc import load_data
    states, masks, labels, game_id = load_data(str(path))
    assert states.shape[1] == NUM_PLANES


def test_packing_in_blocks_changes_nothing():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
    import collect_data

    normal, _ = collect_data.collect(num_games=15, seed=3, progress_every=10**9)
    saved = collect_data.CHUNK
    collect_data.CHUNK = 100                                   # force many small blocks
    try:
        packed, _ = collect_data.collect(num_games=15, seed=3, progress_every=10**9)
    finally:
        collect_data.CHUNK = saved
    for key in normal:
        assert torch.equal(normal[key], packed[key]), key
