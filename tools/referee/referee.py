"""Independent referee: replays a self-play dataset and checks every hand is a legal game of mahjong.

    python tools/referee/referee.py data/games1000                 # every hand, every decision
    python tools/referee/referee.py data/games1000 --shards 0-2    # quicker spot check
    python tools/referee/referee.py data/games1000 --no-waits      # skip the (slow) danger-label check

Why this exists: the Rust engine generated the data, so the engine cannot be the one to vouch for it.
This file is a second, separate implementation written from RULES.md. It imports nothing from the
engine. It reads only the stored dataset (hands.csv, action, mask, meta, obs, oracle, waits) and:

  1. Rebuilds every wall from the stored seed with its own copy of the shuffle and checks it holds
     exactly the 148 tiles of a Singapore set (4 of each playing tile, 1 of each bonus tile).
  2. Deals and plays the hand itself: bonus replacement, draws from the front, kong replacements from
     the back, claim windows, claim priority, the draw at 15 live tiles.
  3. At every recorded decision checks that the right player was asked, that the set of legal moves
     the engine offered is exactly the set the rules allow, and that the move taken is one of them.
  4. Checks the stored snapshot (own hand, everyone's melds, discards, bonus tiles, hidden hands,
     wall count, phase) matches the referee's own table at that moment: no tile appears or vanishes.
  5. Checks the opponent "waits" labels: for each ready opponent and each tile, the tai they would
     win with if it were discarded now.
  6. Scores every win from scratch (all readings of the hand, highest wins, cap 5) and checks winner,
     thrower, tai and payments, and that every hand's points add up to zero.
  7. Checks the dealer rule across each full game: who deals next, repeats, round wind, game end.
  8. Checks the shuffle is fair: every tile kind turns up equally often at every wall position.

Any disagreement is printed with the hand and decision so it can be replayed (`dsy trace`).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------- tiles
# 0-8 characters, 9-17 dots, 18-26 bamboo, 27-30 winds E S W N, 31-33 dragons,
# 34-37 seasons, 38-41 plants, 42-45 animals (RULES.md "Tiles").
KINDS = 34
WALL_SIZE = 148
RESERVE = 15
WINDS = range(27, 31)
DRAGONS = range(31, 34)
ORPHANS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


def is_bonus(t):
    return t >= 34


def is_suited(t):
    return t < 27


def suit(t):
    return t // 9 if t < 27 else 3


def rank(t):  # 0-based rank within a suit
    return t % 9


# ----------------------------------------------------------------------------- RNG
# A re-implementation of the engine's documented RNG (xoshiro256++ seeded through splitmix64)
# and Fisher-Yates shuffle, so walls can be rebuilt from the stored seed alone.
M64 = (1 << 64) - 1


def _splitmix(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return x, z ^ (z >> 31)


def _rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & M64


class Rng:
    def __init__(self, seed):
        x = seed & M64
        s = []
        for _ in range(4):
            x, z = _splitmix(x)
            s.append(z)
        self.s = s

    def next(self):
        s = self.s
        result = (_rotl((s[0] + s[3]) & M64, 23) + s[0]) & M64
        t = (s[1] << 17) & M64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 45)
        return result

    def below(self, n):
        return (self.next() * n) >> 64

    def shuffle(self, v):
        for i in range(len(v) - 1, 0, -1):
            j = self.below(i + 1)
            v[i], v[j] = v[j], v[i]

    @staticmethod
    def derive(seed, stream):
        x = (seed ^ ((stream * 0xD1B54A32D192ED03) & M64)) & M64
        x, _ = _splitmix(x)
        x, z = _splitmix(x)
        return Rng(z)


def hand_seed(master, hand_no):
    return Rng.derive(master, hand_no).next()


def build_wall(seed):
    wall = [k for k in range(KINDS) for _ in range(4)] + list(range(34, 46))
    Rng(seed).shuffle(wall)
    return wall


# ----------------------------------------------------------------------------- hand shapes
@lru_cache(maxsize=1 << 20)
def _can_sets(counts: tuple, start: int) -> bool:
    """Can `counts` be split entirely into pongs and chows?"""
    i = start
    while i < KINDS and counts[i] == 0:
        i += 1
    if i == KINDS:
        return True
    c = list(counts)
    if c[i] >= 3:
        c[i] -= 3
        if _can_sets(tuple(c), i):
            return True
        c[i] += 3
    if is_suited(i) and rank(i) <= 6 and c[i + 1] and c[i + 2]:
        c[i] -= 1
        c[i + 1] -= 1
        c[i + 2] -= 1
        if _can_sets(tuple(c), i):
            return True
    return False


@lru_cache(maxsize=1 << 20)
def is_complete(counts: tuple, n_melds: int) -> bool:
    """4 sets and a pair (counting declared melds), or thirteen orphans. Seven pairs is not a hand."""
    if sum(counts) != 14 - 3 * n_melds:
        return False
    if n_melds == 0 and all(counts[t] >= 1 for t in ORPHANS) and sum(counts[t] for t in ORPHANS) == 14:
        return True
    for p in range(KINDS):
        if counts[p] >= 2:
            c = list(counts)
            c[p] -= 2
            if _can_sets(tuple(c), 0):
                return True
    return False


def decompositions(counts):
    """Every way to read concealed tiles as (pair, [sets]); a set is ('pong'|'chow', tile)."""
    out = []

    def rec(c, i, sets):
        while i < KINDS and c[i] == 0:
            i += 1
        if i == KINDS:
            out.append(list(sets))
            return
        if c[i] >= 3:
            c[i] -= 3
            sets.append(("pong", i))
            rec(c, i, sets)
            sets.pop()
            c[i] += 3
        if is_suited(i) and rank(i) <= 6 and c[i + 1] and c[i + 2]:
            for d in range(3):
                c[i + d] -= 1
            sets.append(("chow", i))
            rec(c, i, sets)
            sets.pop()
            for d in range(3):
                c[i + d] += 1

    res = []
    for p in range(KINDS):
        if counts[p] >= 2:
            c = list(counts)
            c[p] -= 2
            out.clear()
            rec(c, 0, [])
            res.extend((p, s) for s in out)
    return res


# ----------------------------------------------------------------------------- scoring (RULES.md "Tai")
def bonus_tai(bonus, seat_wind):
    """(tai from bonus tiles, has a scoring bonus tile, has only non-matching flowers)."""
    tai = 0
    scoring = False
    for first in (34, 38):  # seasons, plants
        held = [b for b in bonus if first <= b < first + 4]
        if len(held) == 4:
            tai += 2
            scoring = True
        elif first + seat_wind in held:
            tai += 1
            scoring = True
    animals = sum(1 for b in bonus if b >= 42)
    tai += animals
    scoring = scoring or animals > 0
    only_plain_flowers = bool(bonus) and not scoring
    return tai, scoring, only_plain_flowers


def score_reading(pair, sets, win_set, ctx):
    """Tai for one reading. sets: list of dicts {kind, tile, concealed, kong}; win_set: index of the
    set the winning tile completed, or None if it completed the pair. Returns raw tai."""
    sw, pw = 27 + ctx["seat_wind"], 27 + ctx["prevailing"]
    tai = 0
    limit = False
    pongs = [s for s in sets if s["kind"] == "pong"]
    chows = [s for s in sets if s["kind"] == "chow"]
    open_melds = any(not s["concealed"] and s["declared"] for s in sets)

    b, b_scoring, b_plain = bonus_tai(ctx["bonus"], ctx["seat_wind"])
    tai += b

    dragon_pongs = [s for s in pongs if s["tile"] in DRAGONS]
    wind_pongs = [s for s in pongs if s["tile"] in WINDS]
    if len(dragon_pongs) == 3:
        limit = True  # big three dragons
    elif len(dragon_pongs) == 2 and pair in DRAGONS:
        tai += 4  # small three dragons, replaces the dragon pong tai
    else:
        tai += len(dragon_pongs)
    if len(wind_pongs) == 4:
        limit = True  # big four winds
    elif len(wind_pongs) == 3 and pair in WINDS:
        tai += 4  # small four winds, replaces the wind pong tai
    else:
        tai += sum(1 for s in wind_pongs if s["tile"] == sw) + sum(1 for s in wind_pongs if s["tile"] == pw)

    if not open_melds:
        tai += 1  # men qing
    if len(pongs) == 4:
        tai += 2  # all pongs
        if all(s["concealed"] for s in pongs):
            limit = True  # four concealed pongs
    if sum(1 for s in sets if s["kong"]) == 4:
        limit = True

    tiles = [pair] + [s["tile"] for s in sets]
    suits = {suit(t) for t in tiles if is_suited(t)}
    honours = any(not is_suited(t) for t in tiles)
    if not suits:
        limit = True  # all honours
    elif len(suits) == 1:
        if honours:
            tai += 2  # half flush
        else:
            tai += 5  # full flush (limit)

    if len(chows) == 4 and pair not in DRAGONS and pair not in (sw, pw) and win_set is not None:
        ws = sets[win_set]
        if ws["kind"] == "chow" and ws["concealed"]:
            lo, wt = ws["tile"], ctx["win_tile"]
            two_sided = (wt == lo and rank(lo) <= 5) or (wt == lo + 2 and rank(lo) >= 1)
            if two_sided:
                if not b_scoring:
                    if not open_melds:
                        tai += 4
                    elif not ctx["bonus"]:
                        tai += 4
                    elif b_plain:
                        tai += 1

    if ctx["kong_replacement"]:
        tai += 1
    if ctx["robbing"]:
        tai += 1
    if ctx["last_tile"]:
        tai += 1
    if ctx["heavenly"] or ctx["earthly"]:
        limit = True
    if limit:
        tai += 5
    return tai


def score_win(counts, melds, ctx):
    """Best (tai, raw) over every reading and every set the winning tile could have completed,
    or None if the tiles are not a winning hand."""
    n_melds = len(melds)
    if not is_complete(tuple(counts), n_melds):
        return None
    declared = [
        {"kind": "pong" if m["kind"] != "chow" else "chow", "tile": m["tile"],
         "concealed": m["kind"] == "concealed_kong", "kong": "kong" in m["kind"], "declared": True}
        for m in melds
    ]
    best = None

    def consider(raw):
        nonlocal best
        key = (min(raw, 5), raw)
        if best is None or key > best:
            best = key

    if n_melds == 0 and all(counts[t] >= 1 for t in ORPHANS) and sum(counts[t] for t in ORPHANS) == 14:
        consider(5 + bonus_tai(ctx["bonus"], ctx["seat_wind"])[0] + 1)  # thirteen orphans + men qing
    # Nine gates: concealed 1112345678999 + one more of the same suit.
    if n_melds == 0:
        s = {suit(t) for t in range(KINDS) if counts[t]}
        if len(s) == 1 and next(iter(s)) < 3:
            base = 9 * next(iter(s))
            c = counts[base:base + 9]
            if c[0] >= 3 and c[8] >= 3 and all(x >= 1 for x in c[1:8]):
                consider(5 + 5 + 1)  # nine gates (also a full flush, men qing)

    wt = ctx["win_tile"]
    for pair, sets in decompositions(counts):
        hidden = [{"kind": k, "tile": t, "concealed": True, "kong": False, "declared": False} for k, t in sets]
        all_sets = declared + hidden
        # Which concealed set (or the pair) did the winning tile complete? Try every possibility.
        options = []
        if pair == wt:
            options.append(None)
        for i, s in enumerate(hidden):
            if (s["kind"] == "pong" and s["tile"] == wt) or (s["kind"] == "chow" and s["tile"] <= wt <= s["tile"] + 2):
                options.append(n_melds + i)
        for opt in options:
            sets_here = [dict(s) for s in all_sets]
            if opt is not None and not ctx["self_draw"] and sets_here[opt]["kind"] == "pong":
                sets_here[opt]["concealed"] = False  # a pong finished on a discard counts as exposed
            consider(score_reading(pair, sets_here, opt, ctx))
    return best


def payments(tai, winner, discarder):
    unit = 1 << (tai - 1)
    d = [0, 0, 0, 0]
    if discarder is None:
        for s in range(4):
            if s != winner:
                d[s] = -2 * unit
                d[winner] += 2 * unit
    else:
        d[discarder] = -4 * unit
        d[winner] = 4 * unit
    return d


# ----------------------------------------------------------------------------- table state
class Mismatch(Exception):
    pass


class Player:
    def __init__(self):
        self.hand = [0] * KINDS
        self.bonus = []
        self.melds = []  # {kind, tile, src}
        self.discards = []  # [tile, claimed, tsumogiri, turn]
        self.draws = 0

    def size(self):
        return sum(self.hand)


class Table:
    """The referee's own copy of the table, driven only by the wall and the recorded choices."""

    def __init__(self, wall, dealer, prevailing):
        self.wall, self.front, self.back = wall, 0, WALL_SIZE
        self.dealer, self.prevailing = dealer, prevailing
        self.p = [Player() for _ in range(4)]
        self.turns = 0
        self.last_draw = False
        self.any_claim = False
        self.result = None
        self._pending = [0, 0, 0, 0]
        # deal: dealer 14, others 13, from the front; then bonus tiles replaced from the back, dealer first
        for i in range(4):
            s = (dealer + i) % 4
            for _ in range(14 if i == 0 else 13):
                self._give(s, self.wall[self.front])
                self.front += 1
        for i in range(4):
            s = (dealer + i) % 4
            while self._pending[s] > 0:
                self._pending[s] -= 1
                self.back -= 1
                self._give(s, self.wall[self.back])
        self.p[dealer].draws = 1
        self.phase = ("self", dealer, None, False, False)  # seat, drawn, after_kong, after_claim

    def _give(self, s, t):
        if is_bonus(t):
            self.p[s].bonus.append(t)
            self._pending[s] += 1
        else:
            self.p[s].hand[t] += 1

    def live(self):
        return self.back - self.front

    def draws_left(self):
        return max(0, self.live() - RESERVE)

    def seat_wind(self, s):
        return (s - self.dealer) % 4

    def draw(self, s, from_back):
        """Draw for seat s, replacing bonus tiles from the back. False if the wall is exhausted."""
        if self.live() <= RESERVE:
            return None
        if from_back:
            self.back -= 1
            t = self.wall[self.back]
        else:
            t = self.wall[self.front]
            self.front += 1
        while True:
            if self.live() <= RESERVE:
                self.last_draw = True
            if not is_bonus(t):
                break
            self.p[s].bonus.append(t)
            if self.live() <= RESERVE:
                return None
            self.back -= 1
            t = self.wall[self.back]
        self.p[s].hand[t] += 1
        self.p[s].draws += 1
        return t

    # ---- wins
    def ctx(self, s, win_tile, self_draw, after_kong=False, robbing=False):
        pl = self.p[s]
        first = pl.draws == 1 and not self.any_claim and not pl.discards
        return {
            "seat_wind": self.seat_wind(s), "prevailing": self.prevailing, "bonus": pl.bonus,
            "self_draw": self_draw, "win_tile": win_tile, "kong_replacement": self_draw and after_kong,
            "robbing": robbing, "last_tile": self.last_draw,
            "heavenly": self_draw and first and s == self.dealer,
            "earthly": self_draw and first and s != self.dealer,
        }

    def win_value(self, s, extra_tile=None, robbing=False):
        """(tai, raw) if seat s can legally win now, else None."""
        pl = self.p[s]
        if self.phase[0] == "self":
            _, seat, drawn, after_kong, after_claim = self.phase
            if seat != s or after_claim:
                return None
            counts = list(pl.hand)
            if drawn is None:  # dealer's opening 14: any tile may be read as the last one
                best = None
                for k in range(KINDS):
                    if counts[k]:
                        v = score_win(counts, pl.melds, self.ctx(s, k, True, after_kong))
                        if v and (best is None or v > best):
                            best = v
                return best if best and best[0] >= 1 else None
            v = score_win(counts, pl.melds, self.ctx(s, drawn, True, after_kong))
        else:
            counts = list(pl.hand)
            counts[extra_tile] += 1
            v = score_win(counts, pl.melds, self.ctx(s, extra_tile, False, robbing=robbing))
        return v if v and v[0] >= 1 else None

    def waits(self, s):
        pl = self.p[s]
        if pl.size() % 3 != 1:
            return [0] * KINDS
        key = (tuple(pl.hand), tuple((m["kind"], m["tile"]) for m in pl.melds), tuple(pl.bonus),
               self.seat_wind(s), self.prevailing, self.last_draw)
        return list(_waits_cached(key))

    def ron_tai_if_discarded(self, s, k):
        pl = self.p[s]
        if pl.hand[k] >= 4 or pl.size() % 3 != 1:
            return 0
        counts = list(pl.hand)
        counts[k] += 1
        if not is_complete(tuple(counts), len(pl.melds)):
            return 0
        ctx = {"seat_wind": self.seat_wind(s), "prevailing": self.prevailing, "bonus": pl.bonus,
               "self_draw": False, "win_tile": k, "kong_replacement": False, "robbing": False,
               "last_tile": self.last_draw, "heavenly": False, "earthly": False}
        v = score_win(counts, pl.melds, ctx)
        return v[0] if v and v[0] >= 1 else 0

    # ---- legal moves (action ids: 0-33 discard, 34-67 kong, 68 tsumo, 69 ron, 70 pong,
    #      71 exposed kong, 72-74 chow with the claimed tile low/middle/high, 75 pass)
    def legal(self, s):
        ph = self.phase
        pl = self.p[s]
        acts = set()
        if ph[0] == "self":
            _, seat, drawn, after_kong, after_claim = ph
            if seat != s:
                return acts
            if self.win_value(s):
                acts.add(68)
            if not after_claim and self.draws_left() > 0:  # a kong needs a replacement tile
                for k in range(KINDS):
                    if pl.hand[k] == 4:
                        acts.add(34 + k)
                    elif pl.hand[k] >= 1 and any(m["kind"] == "pong" and m["tile"] == k for m in pl.melds):
                        acts.add(34 + k)
            for k in range(KINDS):
                if pl.hand[k]:
                    acts.add(k)
        elif ph[0] == "claim":
            _, frm, t = ph
            if s == frm:
                return acts
            if self.win_value(s, t):
                acts.add(69)
            n = pl.hand[t]
            if n >= 2:
                acts.add(70)
            if n == 3 and self.draws_left() > 0:
                acts.add(71)
            if s == (frm + 1) % 4 and is_suited(t):
                r, h = rank(t), pl.hand
                if r <= 6 and h[t + 1] and h[t + 2]:
                    acts.add(72)
                if 1 <= r <= 7 and h[t - 1] and h[t + 1]:
                    acts.add(73)
                if r >= 2 and h[t - 1] and h[t - 2]:
                    acts.add(74)
            acts.add(75)
        elif ph[0] == "rob":
            _, konger, t = ph
            if s == konger:
                return acts
            if self.win_value(s, t, robbing=True):
                acts.add(69)
            acts.add(75)
        return acts

    def responders(self, actor):
        """Seats that have a real choice in a claim window, in turn order after the actor."""
        return [s for s in ((actor + i) % 4 for i in (1, 2, 3)) if self.legal(s) != {75}]

    # ---- applying moves
    def apply_self(self, s, a):
        self.turns += 1
        pl = self.p[s]
        _, _, drawn, after_kong, _ = self.phase
        if a == 68:
            tai, raw = self.win_value(s)
            self.finish(s, None, tai, raw)
        elif a < 34:
            pl.hand[a] -= 1
            pl.discards.append([a, False, drawn == a, self.turns])
            self.phase = ("claim", s, a)
        else:
            k = a - 34
            if pl.hand[k] == 4:
                pl.hand[k] = 0
                pl.melds.append({"kind": "concealed_kong", "tile": k, "src": None})
                self.replacement(s)
            else:
                pl.hand[k] -= 1
                self.phase = ("rob", s, k)

    def replacement(self, s):
        t = self.draw(s, True)
        if t is None:
            self.finish(None, None, 0, 0)
        else:
            self.phase = ("self", s, t, True, False)

    def resolve_claim(self, answers):
        _, frm, t = self.phase
        order = [(frm + i) % 4 for i in (1, 2, 3)]
        for s in order:  # win beats everything; nearest seat after the thrower first
            if answers.get(s) == 69:
                tai, raw = self.win_value(s, t)
                self.p[s].hand[t] += 1
                self.p[frm].discards[-1][1] = True
                self.finish(s, frm, tai, raw)
                return
        for s in order:  # then pong / kong
            if answers.get(s) in (70, 71):
                kong = answers[s] == 71
                self.p[s].hand[t] -= 3 if kong else 2
                self.p[s].melds.append({"kind": "exposed_kong" if kong else "pong", "tile": t, "src": frm})
                self.p[frm].discards[-1][1] = True
                self.any_claim = True
                if kong:
                    self.replacement(s)
                else:
                    self.phase = ("self", s, None, False, True)
                return
        nxt = (frm + 1) % 4
        if answers.get(nxt) in (72, 73, 74):  # then chow, next player only
            low = t - (answers[nxt] - 72)
            for x in range(low, low + 3):
                if x != t:
                    self.p[nxt].hand[x] -= 1
            self.p[nxt].melds.append({"kind": "chow", "tile": low, "src": frm})
            self.p[frm].discards[-1][1] = True
            self.any_claim = True
            self.phase = ("self", nxt, None, False, True)
            return
        t2 = self.draw(nxt, False)
        if t2 is None:
            self.finish(None, None, 0, 0)
        else:
            self.phase = ("self", nxt, t2, False, False)

    def resolve_rob(self, answers):
        _, konger, t = self.phase
        for s in ((konger + i) % 4 for i in (1, 2, 3)):
            if answers.get(s) == 69:
                tai, raw = self.win_value(s, t, robbing=True)
                self.p[s].hand[t] += 1
                self.finish(s, konger, tai, raw)
                return
        m = next(m for m in self.p[konger].melds if m["kind"] == "pong" and m["tile"] == t)
        m["kind"] = "added_kong"
        self.replacement(konger)

    def finish(self, winner, discarder, tai, raw):
        deltas = payments(tai, winner, discarder) if winner is not None else [0, 0, 0, 0]
        self.result = {"winner": winner, "discarder": discarder, "tai": tai, "raw": raw,
                       "deltas": deltas, "turns": self.turns}
        self.phase = ("over",)

    def census(self):
        """Every tile is either still in the wall or somewhere on the table, exactly once."""
        seen = Counter()
        for pl in self.p:
            for k in range(KINDS):
                seen[k] += pl.hand[k]
            for b in pl.bonus:
                seen[b] += 1
            for m in pl.melds:
                for x in meld_tiles(m):
                    seen[x] += 1
            for d in pl.discards:
                if not d[1]:
                    seen[d[0]] += 1
        for t in self.wall[self.front:self.back]:
            seen[t] += 1
        if self.phase[0] == "rob":  # the tile being added to a pong is on its way, not yet in a meld
            seen[self.phase[2]] += 1
        return all(seen[k] == 4 for k in range(KINDS)) and all(seen[b] == 1 for b in range(34, 46))


@lru_cache(maxsize=1 << 18)
def _waits_cached(key):
    hand, melds, bonus, sw, pw, last = key
    melds = [{"kind": k, "tile": t} for k, t in melds]
    out = [0] * KINDS
    near = set()
    for k in range(KINDS):
        if hand[k]:
            near.add(k)
            if is_suited(k):
                for d in (-2, -1, 1, 2):
                    j = k + d
                    if 0 <= j < 27 and suit(j) == suit(k):
                        near.add(j)
    if len(melds) == 0:
        near.update(ORPHANS)
    for k in near:
        if hand[k] >= 4:
            continue
        c = list(hand)
        c[k] += 1
        if not is_complete(tuple(c), len(melds)):
            continue
        ctx = {"seat_wind": sw, "prevailing": pw, "bonus": list(bonus), "self_draw": False, "win_tile": k,
               "kong_replacement": False, "robbing": False, "last_tile": last, "heavenly": False, "earthly": False}
        v = score_win(c, melds, ctx)
        out[k] = v[0] if v and v[0] >= 1 else 0
    return tuple(out)


def meld_tiles(m):
    if m["kind"] == "chow":
        return [m["tile"], m["tile"] + 1, m["tile"] + 2]
    return [m["tile"]] * (3 if m["kind"] == "pong" else 4)


MELD_CODE = {"chow": 1, "pong": 2, "exposed_kong": 3, "added_kong": 4, "concealed_kong": 5}


def expected_obs(tb, s):
    """The referee's version of what the logger should have stored for seat s right now."""
    o = np.zeros(672, dtype=np.int32)
    o[0:34] = tb.p[s].hand
    for a in range(4):
        r = (a - s) % 4
        pl = tb.p[a]
        for m in pl.melds:
            for x in meld_tiles(m):
                o[34 + r * 34 + x] += 1
        for i, (t, claimed, tsumogiri, turn) in enumerate(pl.discards):
            o[170 + r * 34 + t] += 1
            if i < 32:
                o[306 + r * 32 + i] = t + (64 if claimed else 0) + (128 if tsumogiri else 0)
                o[434 + r * 32 + i] = min(turn, 254)
        for i in range(len(pl.discards), 32):
            o[306 + r * 32 + i] = 255
            o[434 + r * 32 + i] = 255
        for b in pl.bonus:
            o[562 + r * 12 + b - 34] = 1
        for i in range(4):
            base = 610 + (r * 4 + i) * 3
            if i < len(pl.melds):
                m = pl.melds[i]
                o[base] = MELD_CODE[m["kind"]]
                o[base + 1] = m["tile"]
                o[base + 2] = 255 if m["src"] is None else (m["src"] - s) % 4
            else:
                o[base + 1] = o[base + 2] = 255
    mo = 658
    ph = tb.phase
    o[mo] = tb.seat_wind(s)
    o[mo + 1] = tb.prevailing
    o[mo + 2] = (tb.dealer - s) % 4
    o[mo + 3] = min(tb.draws_left(), 255)
    o[mo + 4] = min(tb.turns, 255)
    if ph[0] == "self":
        o[mo + 5], o[mo + 6], o[mo + 7] = 0, 255 if ph[2] is None else ph[2], 255
        o[mo + 8], o[mo + 9] = int(ph[3]), int(ph[4])
    else:
        o[mo + 5], o[mo + 6], o[mo + 7] = (1 if ph[0] == "claim" else 2), ph[2], (ph[1] - s) % 4
    o[mo + 10] = int(tb.last_draw)
    for r in (1, 2, 3):
        o[mo + 10 + r] = tb.p[(s + r) % 4].size()
    return o


# ----------------------------------------------------------------------------- replay one shard
def load(path):
    p = Path(str(path) + ".gz")
    if p.exists():
        with gzip.open(p, "rb") as f:
            return np.load(io.BytesIO(f.read()))
    return np.load(path)


def check_shard(args):
    shard_dir, master_seed, check_waits = args
    shard_dir = Path(shard_dir)
    errors = []
    stats = Counter()
    first_tiles = np.zeros((WALL_SIZE, 46), dtype=np.int64)
    action = load(shard_dir / "action.npy")
    mask = load(shard_dir / "mask.npy")
    meta = load(shard_dir / "meta.npy")
    obs = load(shard_dir / "obs.npy")
    oracle = load(shard_dir / "oracle.npy")
    waits = load(shard_dir / "waits.npy") if check_waits else None
    with open(shard_dir / "hands.csv", newline="") as f:
        hands = list(csv.DictReader(f))

    row = 0
    n_rows = len(action)
    for h in hands:
        hid = int(h["hand_id"])
        where = f"{shard_dir.name} hand {hid}"

        def bad(msg):
            errors.append(f"{where}: {msg}")

        seed = int(h["seed"])
        if seed != hand_seed(master_seed, hid):
            bad("stored seed is not the one derived from the run seed")
        wall = build_wall(seed)
        if sorted(Counter(wall).items()) != sorted({**{k: 4 for k in range(KINDS)}, **{b: 1 for b in range(34, 46)}}.items()):
            bad("wall is not a complete 148-tile set")
        for i, t in enumerate(wall):
            first_tiles[i, t] += 1

        tb = Table(wall, int(h["dealer"]), int(h["prevailing"]))
        stats["hands"] += 1

        def check_decision(tb, s, row):
            if row >= n_rows or meta[row, 0] != hid:
                raise Mismatch(f"expected a decision by seat {s}, the stored hand has none")
            if meta[row, 1] != s:
                raise Mismatch(f"decision {row}: stored seat {meta[row, 1]}, referee expected seat {s}")
            legal = tb.legal(s)
            stored_legal = set(np.flatnonzero(mask[row]).tolist())
            if stored_legal != legal:
                raise Mismatch(f"decision {row}: legal moves differ: engine-only {sorted(stored_legal - legal)}, "
                               f"rules-only {sorted(legal - stored_legal)}")
            a = int(action[row])
            if a not in legal:
                raise Mismatch(f"decision {row}: move {a} is not legal")
            exp = expected_obs(tb, s)
            diff = np.flatnonzero(exp != obs[row].astype(np.int32))
            if diff.size:
                raise Mismatch(f"decision {row}: stored table snapshot differs at obs offsets {diff[:8].tolist()}")
            for r in (1, 2, 3):
                if list(oracle[row, (r - 1) * 34:r * 34]) != tb.p[(s + r) % 4].hand:
                    raise Mismatch(f"decision {row}: stored hidden hand of seat {(s + r) % 4} differs")
            if check_waits:
                for r in (1, 2, 3):
                    o = (s + r) % 4
                    if list(waits[row, r - 1]) != tb.waits(o):
                        raise Mismatch(f"decision {row}: waits label for seat {o} differs: "
                                       f"stored {waits[row, r - 1].tolist()}, referee {tb.waits(o)}")
                stats["waits_checked"] += 1
            if not tb.census():
                raise Mismatch(f"decision {row}: tile census failed")
            stats["decisions"] += 1
            return a
        try:
            while tb.phase[0] != "over":
                if tb.phase[0] == "self":
                    s = tb.phase[1]
                    a = check_decision(tb, s, row)
                    row += 1
                    tb.apply_self(s, a)
                else:
                    answers = {}
                    for s in tb.responders(tb.phase[1]):
                        answers[s] = check_decision(tb, s, row)
                        row += 1
                    if tb.phase[0] == "claim":
                        tb.resolve_claim(answers)
                    else:
                        tb.resolve_rob(answers)
            h["_kong"] = any("kong" in m["kind"] for pl in tb.p for m in pl.melds)
        except Mismatch as e:
            bad(str(e))
            while row < n_rows and meta[row, 0] == hid:
                row += 1
            continue
        except Exception as e:  # a bug in the referee or an impossible state
            bad(f"referee error: {e!r}")
            while row < n_rows and meta[row, 0] == hid:
                row += 1
            continue

        # result
        r = tb.result
        w = int(h["winner"])
        exp_w = -1 if r["winner"] is None else r["winner"]
        exp_d = -1 if r["discarder"] is None else r["discarder"]
        if w != exp_w or int(h["discarder"]) != exp_d:
            bad(f"winner/thrower {w}/{h['discarder']}, referee {exp_w}/{exp_d}")
        if w >= 0 and int(h["tai"]) != r["tai"]:
            bad(f"tai {h['tai']} ({h['patterns']}), referee {r['tai']} (raw {r['raw']})")
        elif w >= 0 and int(h["raw_tai"]) != r["raw"]:
            stats["raw_tai_differs"] += 1
        deltas = [int(h[f"d{i}"]) for i in range(4)]
        if sum(deltas) != 0:
            bad("points do not add up to zero")
        if deltas != r["deltas"]:
            bad(f"payments {deltas}, referee {r['deltas']}")
        if int(h["turns"]) != r["turns"]:
            bad(f"turns {h['turns']}, referee {r['turns']}")
        if not tb.census():
            bad("tile census failed at the end of the hand")
        stats["wins" if w >= 0 else "draws"] += 1
    if row != n_rows:
        errors.append(f"{shard_dir.name}: {n_rows - row} stored decisions were never reached")
    return errors, stats, first_tiles, hands


# ----------------------------------------------------------------------------- games and dealer rule
def check_games(all_hands):
    errors = []
    games = defaultdict(list)
    for h in all_hands:
        if h.get("game"):
            games[int(h["game"])].append(h)
    for g, hs in sorted(games.items()):
        hs.sort(key=lambda h: int(h["hand_in_game"]))
        lineup = tuple(h[f"bot{i}"] for h in hs[:1] for i in range(4))
        prevailing, dealer_no, repeat, dealer = 0, 1, 0, int(hs[0]["dealer"])
        ended = False
        for i, h in enumerate(hs):
            where = f"game {g} hand {h['hand_in_game']}"
            if ended:
                errors.append(f"{where}: hand played after the game ended")
                break
            if tuple(h[f"bot{j}"] for j in range(4)) != lineup:
                errors.append(f"{where}: players changed during the game")
            got = (int(h["prevailing"]), int(h["dealer_no"]), int(h["repeat"]), int(h["dealer"]))
            if got != (prevailing, dealer_no, repeat, dealer):
                errors.append(f"{where}: round/dealer/repeat/seat {got}, rules give {(prevailing, dealer_no, repeat, dealer)}")
                break
            winner = int(h["winner"])
            dealer_won = winner == dealer
            if dealer_won:
                repeat += 1
                continue
            if winner < 0 and "_kong" in h and not h["_kong"]:
                repeat += 1  # draw and nobody holds a kong: dealer stays
                continue
            # deal passes
            repeat = 0
            dealer = (dealer + 1) % 4
            dealer_no += 1
            if dealer_no > 4:
                dealer_no = 1
                prevailing += 1
                if prevailing > 3:
                    ended = True
        if not ended:
            errors.append(f"game {g}: ended before the North round finished")
    return errors, len(games)


def chi_square_uniform(first_tiles):
    """For every wall position, compare how often each tile kind appears with the fair-shuffle
    expectation (4/148 per playing kind, 1/148 per bonus tile). Returns the worst p-value-ish z."""
    n = first_tiles[0].sum()
    exp = np.array([4] * 34 + [1] * 12, dtype=float) / WALL_SIZE * n
    chi = ((first_tiles - exp) ** 2 / exp).sum(axis=1)  # one statistic per position, 45 degrees of freedom
    z = (chi - 45) / math.sqrt(2 * 45)
    return float(chi.mean()), float(np.abs(z).max())


def parse_range(s, n):
    if not s:
        return list(range(n))
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--shards", help="e.g. 0-2,5 (default: all)")
    ap.add_argument("--no-waits", action="store_true", help="skip checking the opponent-waits labels (faster)")
    ap.add_argument("--jobs", type=int, default=None, help="parallel processes (default: all cores)")
    ap.add_argument("--show", type=int, default=20, help="print at most this many problems")
    args = ap.parse_args()

    run = Path(args.run_dir)
    fmt = dict(l.split("=", 1) for l in (run / "format.txt").read_text().split() if "=" in l)
    master = int(fmt["seed"])
    shards = sorted(p for p in run.glob("shard_*") if p.is_dir())
    pick = parse_range(args.shards, len(shards))
    shards = [shards[i] for i in pick]
    print(f"Referee: {run.name}, {len(shards)} shard(s), run seed {master}, "
          f"waits labels {'off' if args.no_waits else 'on'}")

    errors, stats, hands = [], Counter(), []
    first_tiles = np.zeros((WALL_SIZE, 46), dtype=np.int64)
    with Pool(args.jobs) as pool:
        for i, (e, s, ft, hs) in enumerate(pool.imap(check_shard, [(str(p), master, not args.no_waits) for p in shards])):
            errors += e
            stats += s
            first_tiles += ft
            hands += hs
            print(f"  {shards[i].name}: {s['hands']} hands, {s['decisions']:,} decisions, {len(e)} problem(s)", flush=True)

    game_errors, n_games = check_games(hands) if fmt.get("mode") == "full_games" else ([], 0)
    errors += game_errors
    chi_mean, worst_z = chi_square_uniform(first_tiles)

    print()
    print(f"Hands replayed:        {stats['hands']:,} ({stats['wins']:,} wins, {stats['draws']:,} draws)")
    print(f"Decisions checked:     {stats['decisions']:,} (legal moves, move taken, table snapshot, hidden hands)")
    if not args.no_waits:
        print(f"Waits labels checked:  {stats['waits_checked']:,} decisions x 3 opponents x 34 tiles")
    if n_games:
        print(f"Full games checked:    {n_games:,} (dealer rule, round wind, labels, same players)")
    print(f"Shuffle fairness:      mean chi-square {chi_mean:.1f} per wall position (fair = 45), "
          f"worst position z = {worst_z:.1f} ({'ok' if worst_z < 5 else 'SUSPICIOUS'})")
    if stats["raw_tai_differs"]:
        print(f"Note: {stats['raw_tai_differs']} wins have the same capped tai but a different uncapped total.")
    print()
    if errors:
        print(f"PROBLEMS: {len(errors)}")
        for e in errors[: args.show]:
            print("  " + e)
        sys.exit(1)
    print("No problems: every hand is a legal game under RULES.md.")


if __name__ == "__main__":
    main()
