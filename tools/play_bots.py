"""
Let bots play each other and print how each one did.

    python tools/play_bots.py --hands 1000 --bots greedy,greedy,random,random

Seats keep their bot for the whole run; hands follow the dealer rule (TableState),
so every seat gets to be dealer and every round wind comes up.

For each bot it prints: points per hand (with its margin of error), how often it won,
how often it threw the winning tile to someone else ("dealt in"), its average tai when
winning, and how often it had a complete hand but could not win because it was worth 0 tai.
"""

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bots.greedy_bot import GreedyBot                      # noqa: E402
from bots.network_bot import NetworkBot                    # noqa: E402
from bots.random_bot import RandomBot                      # noqa: E402
from engine.game import WIN, Game, TableState               # noqa: E402
from engine.scoring import is_winning_shape                # noqa: E402
from engine.view import view_for                           # noqa: E402


def make_bot(name, seed, model_path):
    if name == "greedy":
        return GreedyBot()
    if name == "random":
        return RandomBot(seed)
    if name == "net":
        return NetworkBot(model_path)
    raise ValueError(f"unknown bot {name!r} (use greedy, random or net)")


def complete_but_no_tai(view):
    """True if the tiles form a winning shape right now but the rules don't allow the win
    (which can only be because the hand is worth 0 tai)."""
    if WIN in view.legal:
        return False
    counts = [0] * 34
    for t in view.hand:
        counts[t] += 1
    if view.phase in ("claim", "rob"):
        counts[view.offer_tile] += 1
    elif view.phase != "turn":
        return False
    return is_winning_shape(counts, len(view.my_melds))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=1000)
    ap.add_argument("--bots", default="greedy,greedy,greedy,greedy", help="4 names: greedy, random or net")
    ap.add_argument("--model", default="bc_model.pt", help="trained network for 'net' seats")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = args.bots.split(",")
    assert len(names) == 4, "give exactly 4 bots"
    bots = [make_bot(n, args.seed + i, args.model) for i, n in enumerate(names)]

    points = [[] for _ in range(4)]
    wins, deal_ins, tai_sum, blocked = [0] * 4, [0] * 4, [0] * 4, [0] * 4
    draws = 0
    table = TableState()
    start = time.time()

    for h in range(args.hands):
        game = Game(dealer=table.dealer, prevailing=table.prevailing, seed=args.seed * 1_000_000 + h)
        blocked_this_hand = [False] * 4
        while not game.is_over():
            seat = game.to_act()
            view = view_for(game, seat)
            if complete_but_no_tai(view):
                blocked_this_hand[seat] = True
            game.apply(seat, bots[seat].choose_action(view))
        r = game.result
        for s in range(4):
            points[s].append(r.deltas[s])
            blocked[s] += blocked_this_hand[s]
        if r.winner is None:
            draws += 1
        else:
            wins[r.winner] += 1
            tai_sum[r.winner] += r.score.tai
            if r.discarder is not None:
                deal_ins[r.discarder] += 1
        table.next_hand(r, game.any_kong())

    n = args.hands
    print(f"{n} hands in {time.time() - start:.0f}s; {draws / n:.1%} ended with no winner\n")
    print(f"{'seat':<5}{'bot':<8}{'points/hand':>16}{'won':>8}{'dealt in':>10}{'avg tai':>9}"
          f"{'had a full hand worth 0 tai':>30}")
    for s in range(4):
        mean = sum(points[s]) / n
        sd = math.sqrt(sum((x - mean) ** 2 for x in points[s]) / (n - 1))
        avg_tai = tai_sum[s] / wins[s] if wins[s] else 0
        print(f"{s:<5}{names[s]:<8}{mean:>+9.2f} ± {sd / math.sqrt(n):<4.2f}{wins[s] / n:>8.1%}"
              f"{deal_ins[s] / n:>10.1%}{avg_tai:>9.2f}{blocked[s] / n:>29.1%} of hands")
    for s in range(4):
        if isinstance(bots[s], NetworkBot):
            print(f"\nseat {s} (net): threw the same tile as GreedyBot would have in "
                  f"{bots[s].agreement():.1%} of {bots[s].discards:,} discards")


if __name__ == "__main__":
    main()
