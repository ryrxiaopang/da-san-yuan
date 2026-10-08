"""Per-move expected value by Monte Carlo rollouts.

For one decision, every legal move is tried in the same set of sampled "worlds", and each
world is played to the end of the hand by the bots. A move's expected value (EV) is the
average final points of the player who made it. Regret = best move's EV - chosen move's EV.

Fair mode (default) samples worlds the player could not tell apart from the real one: the
tiles they cannot see (opponents' concealed tiles and the wall) are reshuffled, keeping every
opponent's tile count. Oracle mode keeps the real hidden tiles and wall order, so it answers
"what would actually have happened", using information the player did not have.

Every move is evaluated in the same worlds with the same bot seeds (common random numbers),
so differences between moves are measured far more precisely than each EV on its own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from .bots import HeuristicBot, Style
from .game import Game
from .rng import Rng
from .selfplay import parallel_map, play_hand
from .tile import NUM_KINDS, is_bonus


@dataclass
class MoveValue:
    action: int
    ev: float            # mean final points of the acting seat over all worlds
    se: float            # standard error of `ev`
    se_vs_chosen: float  # standard error of (this move's EV - the chosen move's EV), using the paired worlds


def sample_world(g: Game, seat: int, rng: Rng) -> Game:
    """Re-deal everything `seat` cannot see, keeping what it can (its own hand, all melds,
    discards, bonus tiles, every player's tile count and the number of tiles in the wall)."""
    w = g.clone()
    pool: List[int] = []
    need = [0] * 4
    for o in range(4):
        if o == seat:
            continue
        p = w.players[o]
        for k in range(NUM_KINDS):
            pool.extend([k] * p.hand[k])
            p.hand[k] = 0
        need[o] = len(pool) - sum(need)
    pool.extend(w.wall[w.front:w.back])
    rng.shuffle(pool)
    # Opponents' concealed hands never hold bonus tiles (they are set aside at once),
    # so hands take the first playing tiles; everything left becomes the unseen wall.
    rest: List[int] = []
    it = iter(pool)
    for o in range(4):
        n = need[o]
        while n > 0:
            t = next(it)
            if is_bonus(t):
                rest.append(t)
            else:
                w.players[o].hand[t] += 1
                n -= 1
    rest.extend(it)
    w.wall[w.front:w.back] = rest
    w.reopen_window(seat)
    return w


def _rollouts(job) -> List[float]:
    g, seat, styles, a, worlds, seed, oracle = job
    out = []
    for k in range(worlds):
        rng = Rng.derive(seed, k)
        world = g.clone() if oracle else sample_world(g, seat, rng)
        bots = [HeuristicBot(styles[i].clone(), rng.next_u64() ^ i) for i in range(4)]
        world.apply(seat, a)
        _, r = play_hand(world, bots)
        out.append(float(r.deltas[seat]))
    return out


def evaluate(g: Game, seat: int, styles: Sequence[Style], chosen: int, worlds: int, seed: int,
             oracle: bool) -> List[MoveValue]:
    """Expected final points of `seat` for every legal move at this decision."""
    acts = g.legal_actions(seat)
    # results[a][k] = points for move a in world k
    results = parallel_map(_rollouts, [(g, seat, list(styles), a, worlds, seed, oracle) for a in acts])
    ci = acts.index(chosen)
    n = float(worlds)
    out = []
    for a, xs in zip(acts, results):
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / max(n - 1.0, 1.0)
        diffs = [x - c for x, c in zip(xs, results[ci])]
        dm = sum(diffs) / n
        dvar = sum((d - dm) ** 2 for d in diffs) / max(n - 1.0, 1.0)
        out.append(MoveValue(a, mean, math.sqrt(var / n), math.sqrt(dvar / n)))
    return out
