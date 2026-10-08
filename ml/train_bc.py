"""
Behavioral cloning for Singapore mahjong discards, trained on games recorded
from your 4 greedy bots.

Run it with:   python ml/train_bc.py greedy_games.pt
Options:       python ml/train_bc.py greedy_games.pt --epochs 20 --test-fraction 0.1
Needs:         pip install torch

---------------------------------------------------------------------------
EXPECTED DATA FILE (what your collection script should save with torch.save)
---------------------------------------------------------------------------
A dict with four entries, one row per discard made by any of the 4 bots:

    "states"   shape (N, P, 34)  the input grid for that decision, from the
                                 discarding bot's own point of view.
                                 P = number of feature lines (e.g. 48).
                                 Lines 0-3 must be the hand (have >=1..>=4);
                                 the rest can be anything, in a fixed order.
                                 Saving as torch.float16 halves the file size.
    "masks"    shape (N, 34)     True/1 where that tile was legal to discard
    "labels"   shape (N,)        the tile the bot actually discarded, 0..33
    "game_id"  shape (N,)        which game the row came from (0, 1, 2, ...)

Tile numbering: 0-8 = 1m..9m, 9-17 = 1p..9p, 18-26 = 1s..9s,
                27-33 = E, S, W, N, Red, Green, White.
---------------------------------------------------------------------------
"""

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.model import DiscardNet  # noqa: E402

NUM_TILE_TYPES = 34
TILE_NAMES = (
    [f"{n}m" for n in range(1, 10)]
    + [f"{n}p" for n in range(1, 10)]
    + [f"{n}s" for n in range(1, 10)]
    + ["E", "S", "W", "N", "Rd", "Gr", "Wh"]
)


# ---------------------------------------------------------------------------
# Step 1: load the recorded games and check they make sense
# ---------------------------------------------------------------------------
def load_data(path):
    data = torch.load(path)
    states = torch.as_tensor(data["states"])
    masks = torch.as_tensor(data["masks"]).bool()
    labels = torch.as_tensor(data["labels"]).long()
    game_id = torch.as_tensor(data["game_id"]).long()

    n = len(labels)
    assert states.shape[0] == masks.shape[0] == game_id.shape[0] == n, \
        "states, masks, labels and game_id must all have the same number of rows"
    assert states.shape[2] == NUM_TILE_TYPES, "states must be (N, P, 34)"
    assert masks.shape[1] == NUM_TILE_TYPES, "masks must be (N, 34)"

    # Sanity check: the bot must only ever discard a tile that was legal.
    # If this fails, there's a bug in your engine or collection script.
    illegal = (~masks[torch.arange(n), labels]).sum().item()
    if illegal:
        raise ValueError(f"{illegal} rows discard a tile the mask says is illegal")

    print(f"Loaded {n:,} decisions from {game_id.unique().numel():,} games, "
          f"{states.shape[1]} feature lines each")
    return states, masks, labels, game_id


def split_by_game(states, masks, labels, game_id, test_fraction, seed=0):
    """Put whole games into either training or testing, never both."""
    games = game_id.unique().tolist()
    random.Random(seed).shuffle(games)
    n_test = max(1, int(len(games) * test_fraction))
    test_games = torch.tensor(games[:n_test])
    is_test = torch.isin(game_id, test_games)

    train = TensorDataset(states[~is_test], masks[~is_test], labels[~is_test])
    test = TensorDataset(states[is_test], masks[is_test], labels[is_test])
    print(f"Training on {len(train):,} decisions, testing on {len(test):,} "
          f"({n_test} games held out)")
    return train, test


# ---------------------------------------------------------------------------
# Step 2: the neural network
# ---------------------------------------------------------------------------
# DiscardNet now lives in ml/model.py, so the bots and the RL stage use the same network.


# ---------------------------------------------------------------------------
# Step 3: training
# ---------------------------------------------------------------------------
def evaluate(model, loader, device):
    """Top-1: network's favourite tile matches the bot.
       Top-3: the bot's tile is among the network's three favourites."""
    model.eval()
    top1 = top3 = total = 0
    with torch.no_grad():
        for state, mask, label in loader:
            state, mask, label = state.to(device).float(), mask.to(device), label.to(device)
            logits = model(state, mask)
            top1 += (logits.argmax(dim=1) == label).sum().item()
            top3 += (logits.topk(3, dim=1).indices == label[:, None]).any(dim=1).sum().item()
            total += len(label)
    model.train()
    return top1 / total, top3 / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="file saved by your collection script")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--out", default="bc_model.pt")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    states, masks, labels, game_id = load_data(args.data)
    num_planes = states.shape[1]
    train_data, test_data = split_by_game(states, masks, labels, game_id,
                                          args.test_fraction)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1024)

    model = DiscardNet(num_planes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_top1 = 0.0
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for state, mask, label in train_loader:
            state, mask, label = state.to(device).float(), mask.to(device), label.to(device)
            logits = model(state, mask)          # 1. predict
            loss = loss_fn(logits, label)        # 2. how wrong vs the greedy bot?
            optimizer.zero_grad()
            loss.backward()                      # 3. work out the nudge
            optimizer.step()                     # 4. apply it
            total_loss += loss.item()

        top1, top3 = evaluate(model, test_loader, device)
        note = ""
        if top1 > best_top1:
            # Keep the best version seen so far, judged on the held-out games.
            best_top1 = top1
            torch.save({"state_dict": model.state_dict(), "num_planes": num_planes},
                       args.out)
            note = "  <- saved"
        print(f"epoch {epoch:2d}  loss {total_loss / len(train_loader):.3f}  "
              f"matches bot {top1:.1%}  (top-3 {top3:.1%}){note}")

    print(f"\nBest test accuracy {best_top1:.1%}, model saved to {args.out}")

    # Show one decision from a held-out game.
    checkpoint = torch.load(args.out)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    state, mask, label = test_data[0]
    hand_counts = state[:4].float().sum(dim=0).long()   # lines 0-3 = hand
    hand = [TILE_NAMES[t] for t in range(NUM_TILE_TYPES) for _ in range(hand_counts[t])]
    with torch.no_grad():
        probs = torch.softmax(model(state[None].to(device).float(), mask[None].to(device)), dim=1)[0]
    top = probs.topk(3)
    print("\nExample hand:", " ".join(hand))
    print("Greedy bot discarded:", TILE_NAMES[label])
    print("Network's top 3:", ", ".join(
        f"{TILE_NAMES[i]} {p:.0%}" for p, i in zip(top.values.tolist(), top.indices.tolist())))


if __name__ == "__main__":
    main()
