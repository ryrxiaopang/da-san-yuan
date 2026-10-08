"""
Collect training data: 4 GreedyBots play hands of Singapore mahjong, and every
discard is saved as one row in the format ml/train_bc.py expects.

Run it with:   python data/collect_data.py --games 5000 --out greedy_games.pt
Then:          python ml/train_bc.py greedy_games.pt
Needs:         pip install -r requirements.txt

"Games" here means hands: each hand is one game_id. Hands are played one after
another with the dealer rule (engine.game.TableState), so every seat wind and
every round wind appears in the data.

What is saved (one row per discard, from the discarding bot's own point of view):
    game_id  (N,)          which hand the row came from (train/test are split by this)
    states   (N, 49, 34)   the input grid, float16 (see ml/encode.py)
    masks    (N, 34)       True where that tile was legal to discard
    labels   (N,)          the tile GreedyBot discarded, 0-33
    seat     (N,)          extras for debugging and RL; train_bc.py ignores them
    turn     (N,)
    won      (N,)          True if this seat won the hand
    points   (N,)          points this seat won or lost in the hand
"""

import argparse
import os
import sys
import time

import torch

# Let this script find the engine/, bots/ and ml/ folders when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.greedy_bot import GreedyBot          # noqa: E402
from engine.game import Game, TableState       # noqa: E402
from engine.view import view_for               # noqa: E402
from ml.encode import encode                   # noqa: E402


CHUNK = 20_000   # pack grids into blocks this size as we go, to keep memory use down


def collect(num_games, seed=0, progress_every=500):
    """Play `num_games` hands and return the rows as a dict of tensors."""
    bots = [GreedyBot() for _ in range(4)]
    rows = {"game_id": [], "seat": [], "turn": [], "states": [], "masks": [], "labels": []}
    packed_states, packed_masks = [], []     # finished blocks of CHUNK rows
    won, points = [], []
    table = TableState()
    hands_won = 0
    start = time.time()

    for game_id in range(num_games):
        game = Game(dealer=table.dealer, prevailing=table.prevailing, seed=seed * 10_000_000 + game_id)
        first_row = len(rows["labels"])

        while not game.is_over():
            seat = game.to_act()
            view = view_for(game, seat)
            action = bots[seat].choose_action(view)

            if action.kind == "discard":
                state, mask = encode(view)
                # Sanity check: the mask must match the engine's list of legal discards.
                legal_discards = {a.tile for a in view.legal if a.kind == "discard"}
                assert legal_discards == {t for t in range(34) if mask[t]}, "mask doesn't match the engine"
                rows["game_id"].append(game_id)
                rows["seat"].append(seat)
                rows["turn"].append(game.turns)
                rows["states"].append(state)
                rows["masks"].append(mask)
                rows["labels"].append(action.tile)

            game.apply(seat, action)

        # Thousands of separate small tensors use much more memory than one block,
        # so every CHUNK rows the grids are packed into a single tensor.
        if len(rows["states"]) >= CHUNK:
            packed_states.append(torch.stack(rows["states"]))
            packed_masks.append(torch.stack(rows["masks"]))
            rows["states"], rows["masks"] = [], []

        # The hand is over: now we know who won and the points, so tag this hand's rows.
        result = game.result
        for seat in rows["seat"][first_row:]:
            won.append(result.winner == seat)
            points.append(result.deltas[seat])
        hands_won += result.winner is not None
        table.next_hand(result, game.any_kong())

        if (game_id + 1) % progress_every == 0:
            rate = (game_id + 1) / (time.time() - start)
            print(f"{game_id + 1:,} hands, {len(rows['labels']):,} discards so far "
                  f"({rate:.1f} hands/s)", flush=True)

    if rows["states"]:
        packed_states.append(torch.stack(rows["states"]))
        packed_masks.append(torch.stack(rows["masks"]))
    data = {
        "game_id": torch.tensor(rows["game_id"]),
        "states": torch.cat(packed_states),
        "masks": torch.cat(packed_masks),
        "labels": torch.tensor(rows["labels"]),
        "seat": torch.tensor(rows["seat"]),
        "turn": torch.tensor(rows["turn"]),
        "won": torch.tensor(won),
        "points": torch.tensor(points),
    }
    return data, hands_won


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5000, help="number of hands to play")
    parser.add_argument("--out", default="greedy_games.pt")
    parser.add_argument("--seed", type=int, default=0, help="same seed = exactly the same data")
    args = parser.parse_args()

    data, hands_won = collect(args.games, args.seed)
    torch.save(data, args.out)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Saved {len(data['labels']):,} discards from {args.games:,} hands to {args.out} "
          f"({size_mb:.0f} MB). {hands_won / args.games:.0%} of hands ended in a win.")


if __name__ == "__main__":
    main()
