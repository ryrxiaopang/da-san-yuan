"""
One-off check: replay hands recorded by the OLD Rust engine (branch "old", dataset games1000)
through the NEW Python engine, and confirm they agree on everything:
  - the same players are asked to decide, in the same order
  - the legal moves at every decision are exactly the same
  - the hand ends the same way: winner, thrower, tai, points, number of turns

The old engine numbered dragons White=31, Green=32, Red=33; the new one uses Red=31,
Green=32, White=33, so tiles are translated on the way in.

Run:  python tools/crosscheck_old_data.py data/games1000 --shards 0
Needs: numpy, and an old dataset folder (made by `dsy selfplay` on the old branch).
"""

import argparse
import csv
import gzip
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.game import PASS, WIN, Action, Game          # noqa: E402

OLD_TO_NEW = {t: t for t in range(46)}
OLD_TO_NEW[31], OLD_TO_NEW[33] = 33, 31                  # swap White and Red


# ---- the old engine's random-number generator, to rebuild each wall from its seed ----
M64 = (1 << 64) - 1


def _splitmix(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return x, z ^ (z >> 31)


def _rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & M64


def old_wall(seed):
    x, s = seed & M64, []
    for _ in range(4):
        x, z = _splitmix(x)
        s.append(z)
    wall = [k for k in range(34) for _ in range(4)] + list(range(34, 46))
    for i in range(len(wall) - 1, 0, -1):
        result = (_rotl((s[0] + s[3]) & M64, 23) + s[0]) & M64
        t = (s[1] << 17) & M64
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = _rotl(s[3], 45)
        j = (result * (i + 1)) >> 64
        wall[i], wall[j] = wall[j], wall[i]
    return [OLD_TO_NEW[t] for t in wall]


# ---- translating moves between the old 76-number action list and the new Action ----
def to_old_index(game, action):
    if action.kind == "discard":
        return OLD_TO_NEW[action.tile]
    if action.kind == "kong":
        return 34 + OLD_TO_NEW[action.tile]
    if action.kind == "win":
        return 68 if game.phase == "turn" else 69
    if action.kind == "pong":
        return 70
    if action.kind == "exposed_kong":
        return 71
    if action.kind == "chow":
        return 72 + (game.offer_tile - action.tile)      # where the claimed tile sits in the run
    return 75                                            # pass


def from_old_index(game, i):
    if i < 34:
        return Action("discard", OLD_TO_NEW[i])
    if i < 68:
        return Action("kong", OLD_TO_NEW[i - 34])
    if i in (68, 69):
        return WIN
    if i == 70:
        return Action("pong")
    if i == 71:
        return Action("exposed_kong")
    if i in (72, 73, 74):
        return Action("chow", game.offer_tile - (i - 72))
    return PASS


def load(path):
    gz = Path(str(path) + ".gz")
    if gz.exists():
        with gzip.open(gz, "rb") as f:
            return np.load(io.BytesIO(f.read()))
    return np.load(path)


def check_shard(shard):
    action, mask, meta = load(shard / "action.npy"), load(shard / "mask.npy"), load(shard / "meta.npy")
    hands = list(csv.DictReader(open(shard / "hands.csv", newline="")))
    row, problems, decisions = 0, [], 0
    for h in hands:
        hid = int(h["hand_id"])
        game = Game(dealer=int(h["dealer"]), prevailing=int(h["prevailing"]), wall=old_wall(int(h["seed"])))
        try:
            while not game.is_over():
                seat = game.to_act()
                if row >= len(action) or meta[row, 0] != hid or meta[row, 1] != seat:
                    raise AssertionError(f"decision {row}: expected seat {seat} to decide")
                new_legal = {to_old_index(game, a) for a in game.legal_actions(seat)}
                old_legal = set(np.flatnonzero(mask[row]).tolist())
                if new_legal != old_legal:
                    raise AssertionError(f"decision {row}: legal moves differ, old only {sorted(old_legal - new_legal)}, "
                                         f"new only {sorted(new_legal - old_legal)}")
                game.apply(seat, from_old_index(game, int(action[row])))
                row += 1
                decisions += 1
            r = game.result
            old = (int(h["winner"]), int(h["discarder"]), int(h["tai"]), [int(h[f"d{i}"]) for i in range(4)], int(h["turns"]))
            new = (-1 if r.winner is None else r.winner, -1 if r.discarder is None else r.discarder,
                   0 if r.score is None else r.score.tai, r.deltas, r.turns)
            if old != new:
                raise AssertionError(f"result differs: old {old}, new {new}")
        except AssertionError as e:
            problems.append(f"hand {hid}: {e}")
            while row < len(meta) and meta[row, 0] == hid:
                row += 1
    return len(hands), decisions, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--shards", default="0", help="e.g. 0 or 0-9")
    args = ap.parse_args()
    a, _, b = args.shards.partition("-")
    picks = range(int(a), int(b or a) + 1)
    shards = sorted(p for p in Path(args.run_dir).glob("shard_*") if p.is_dir())
    total_hands = total_dec = 0
    all_problems = []
    for i in picks:
        n, d, probs = check_shard(shards[i])
        total_hands += n
        total_dec += d
        all_problems += probs
        print(f"{shards[i].name}: {n} hands, {d:,} decisions, {len(probs)} problem(s)", flush=True)
    print(f"\n{total_hands:,} hands, {total_dec:,} decisions replayed.")
    for p in all_problems[:20]:
        print("  " + p)
    print("The new engine agrees with the old one on every hand." if not all_problems
          else f"{len(all_problems)} hands differ.")
    sys.exit(1 if all_problems else 0)


if __name__ == "__main__":
    main()
