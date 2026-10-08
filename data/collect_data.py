"""
Collect training data: 4 greedy bots play mahjong, and every discard is saved
as one row in the format train_bc.py expects.

Run it with:   python collect_data.py --games 5000 --out greedy_games.pt
Then:          python train_bc.py greedy_games.pt
Needs:         pip install torch

What's real and what's a placeholder:
  - encode()            REAL. The 48-line grid. Keep it when you swap engines.
  - save format         REAL. Exactly what train_bc.py reads.
  - SimpleGame          PLACEHOLDER engine: draw and discard only, self-drawn
                        wins only, no pong/chow/kong, no bonus tiles, no tai.
                        Replace with your real Singapore mahjong engine.
  - GreedyBot           PLACEHOLDER teacher: "throw the tile with the fewest
                        copies/neighbours". Replace with your shanten-based bot.
"""

import argparse
import random
from dataclasses import dataclass, field

import torch

NUM_TILE_TYPES = 34      # 0-8 1m..9m, 9-17 1p..9p, 18-26 1s..9s, 27-33 E S W N Rd Gr Wh
EAST, SOUTH, WEST, NORTH = 27, 28, 29, 30
SEAT_WINDS = [EAST, SOUTH, WEST, NORTH]       # seat 0 is East (dealer), and so on
NUM_PLANES = 48
DEAD_WALL = 15           # placeholder: the hand is a draw when this many tiles are left


# ===========================================================================
# What one player is allowed to see. Your real engine should produce this too.
# Everything is RELATIVE to the player: index 0 = me, 1 = next, 2 = opposite,
# 3 = previous. No opponent hands, no wall order.
# ===========================================================================
@dataclass
class PlayerView:
    hand: list                      # my tiles (14 when it's my turn to discard)
    discards: list                  # 4 lists: discards by me, next, opposite, previous
    melds: list                     # 4 lists: exposed meld tiles, same order
    bonus_counts: list              # 4 numbers: flowers/animals revealed, same order
    last_discard: int | None        # most recent tile thrown by anyone, or None
    seat_wind: int                  # tile number of my seat wind
    prevailing_wind: int            # tile number of the round wind
    wall_left: int                  # tiles left to draw


# ===========================================================================
# REAL: turn a PlayerView into the 48 x 34 grid. The line ORDER must never
# change once you start collecting, or old and new data won't match.
# ===========================================================================
def counts(tiles):
    c = [0] * NUM_TILE_TYPES
    for t in tiles:
        c[t] += 1
    return c


def at_least(c, k):
    return [1.0 if c[t] >= k else 0.0 for t in range(NUM_TILE_TYPES)]


def one_hot(tile):
    line = [0.0] * NUM_TILE_TYPES
    if tile is not None:
        line[tile] = 1.0
    return line


def encode(view: PlayerView):
    lines = []
    hand_c = counts(view.hand)
    lines += [at_least(hand_c, k) for k in (1, 2, 3, 4)]                 # 0-3   hand
    for d in view.discards:                                              # 4-19  discards x4 players
        lines += [at_least(counts(d), k) for k in (1, 2, 3, 4)]
    for m in view.melds:                                                 # 20-35 melds x4 players
        lines += [at_least(counts(m), k) for k in (1, 2, 3, 4)]
    seen = view.hand + sum(view.discards, []) + sum(view.melds, [])
    lines += [at_least(counts(seen), k) for k in (1, 2, 3, 4)]           # 36-39 visible anywhere
    lines.append(one_hot(view.last_discard))                             # 40    last discard
    lines.append(one_hot(view.seat_wind))                                # 41    my seat wind
    lines.append(one_hot(view.prevailing_wind))                          # 42    prevailing wind
    lines.append([view.wall_left / 136] * NUM_TILE_TYPES)                # 43    wall left
    for b in view.bonus_counts:                                          # 44-47 bonus tiles x4
        lines.append([b / 8] * NUM_TILE_TYPES)
    assert len(lines) == NUM_PLANES

    mask = [hand_c[t] > 0 for t in range(NUM_TILE_TYPES)]                # legal discards
    return torch.tensor(lines, dtype=torch.float16), torch.tensor(mask)


# ===========================================================================
# PLACEHOLDER teacher. Swap in your shanten-based greedy bot.
# ===========================================================================
class GreedyBot:
    def choose_discard(self, view: PlayerView) -> int:
        c = counts(view.hand)

        def usefulness(t):
            score = c[t] - 1                       # other copies
            if t < 27:                             # suited: neighbours in same suit
                pos = t % 9
                if pos > 0:
                    score += c[t - 1]
                if pos < 8:
                    score += c[t + 1]
            return score

        return min(set(view.hand), key=lambda t: (usefulness(t), t))


# ===========================================================================
# PLACEHOLDER engine. Replace with your real one; keep view_for() returning
# a PlayerView so encode() keeps working.
# ===========================================================================
def is_winning_hand(hand):
    """Standard win: 4 sets (triplet or run) + 1 pair, 14 tiles."""
    c = counts(hand)

    def sets_only(c):
        i = next((t for t in range(NUM_TILE_TYPES) if c[t]), None)
        if i is None:
            return True
        if c[i] >= 3:                                   # try a triplet
            c[i] -= 3
            ok = sets_only(c)
            c[i] += 3
            if ok:
                return True
        if i < 27 and i % 9 <= 6 and c[i + 1] and c[i + 2]:   # try a run
            for t in (i, i + 1, i + 2):
                c[t] -= 1
            ok = sets_only(c)
            for t in (i, i + 1, i + 2):
                c[t] += 1
            if ok:
                return True
        return False

    for p in range(NUM_TILE_TYPES):
        if c[p] >= 2:
            c[p] -= 2
            ok = sets_only(c)
            c[p] += 2
            if ok:
                return True
    return False


@dataclass
class SimpleGame:
    rng: random.Random
    hands: list = field(default_factory=list)
    discards: list = field(default_factory=lambda: [[], [], [], []])
    wall: list = field(default_factory=list)
    current: int = 0
    last_discard: int | None = None
    winner: int | None = None
    over: bool = False

    def __post_init__(self):
        self.wall = [t for t in range(NUM_TILE_TYPES) for _ in range(4)]
        self.rng.shuffle(self.wall)
        self.hands = [[self.wall.pop() for _ in range(13)] for _ in range(4)]

    def draw(self):
        """Current player draws. Returns True if the hand ended (win or draw)."""
        if len(self.wall) <= DEAD_WALL:
            self.over = True                         # exhaustive draw, nobody wins
            return True
        self.hands[self.current].append(self.wall.pop())
        if is_winning_hand(self.hands[self.current]):
            self.winner, self.over = self.current, True
            return True
        return False

    def view_for(self, seat):
        rel = [(seat + k) % 4 for k in range(4)]     # me, next, opposite, previous
        return PlayerView(
            hand=list(self.hands[seat]),
            discards=[list(self.discards[p]) for p in rel],
            melds=[[] for _ in rel],                 # no claims in the placeholder
            bonus_counts=[0, 0, 0, 0],               # no bonus tiles in the placeholder
            last_discard=self.last_discard,
            seat_wind=SEAT_WINDS[seat],
            prevailing_wind=EAST,
            wall_left=len(self.wall),
        )

    def discard(self, tile):
        self.hands[self.current].remove(tile)
        self.discards[self.current].append(tile)
        self.last_discard = tile
        self.current = (self.current + 1) % 4       # no claims: always next player


# ===========================================================================
# REAL: the collection loop and save format.
# ===========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--out", default="greedy_games.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    bots = [GreedyBot() for _ in range(4)]
    rows = {"game_id": [], "seat": [], "turn": [], "states": [], "masks": [], "labels": []}
    won = []                                         # filled in once each game ends
    wins = 0

    for game_id in range(args.games):
        game = SimpleGame(rng)
        first_row = len(rows["labels"])
        turn = 0
        while not game.draw():
            seat = game.current
            view = game.view_for(seat)
            state, mask = encode(view)
            tile = bots[seat].choose_discard(view)

            rows["game_id"].append(game_id)
            rows["seat"].append(seat)
            rows["turn"].append(turn)
            rows["states"].append(state)
            rows["masks"].append(mask)
            rows["labels"].append(tile)

            game.discard(tile)
            turn += 1

        # Now we know who won; tag this game's rows (useful later for RL/eval).
        won += [seat == game.winner for seat in rows["seat"][first_row:]]
        wins += game.winner is not None
        if (game_id + 1) % 500 == 0:
            print(f"{game_id + 1:,} games, {len(rows['labels']):,} decisions so far")

    torch.save({
        "game_id": torch.tensor(rows["game_id"]),
        "states": torch.stack(rows["states"]),
        "masks": torch.stack(rows["masks"]),
        "labels": torch.tensor(rows["labels"]),
        # extras: train_bc.py ignores these, but they're handy later
        "seat": torch.tensor(rows["seat"]),
        "turn": torch.tensor(rows["turn"]),
        "won": torch.tensor(won),
    }, args.out)
    print(f"Saved {len(rows['labels']):,} decisions from {args.games:,} games to {args.out} "
          f"({wins / args.games:.0%} of hands ended in a win)")


if __name__ == "__main__":
    main()
