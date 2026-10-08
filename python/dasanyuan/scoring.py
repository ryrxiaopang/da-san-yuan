"""Melds, win decomposition and tai scoring for the team's Singapore ruleset.

RULES.md in the repo root describes every rule here in plain language.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Sequence, Tuple

from .shanten import shanten_standard
from .tile import (NUM_KINDS, ORPHANS, flower_number, flower_set, is_animal, is_dragon, is_flower, is_wind,
                   rank, wind_tile)


class MeldKind(IntEnum):
    CHOW = 0           # sequence starting at `tile`
    PONG = 1
    EXPOSED_KONG = 2   # claimed from a discard
    ADDED_KONG = 3     # pong upgraded with a self-drawn fourth tile
    CONCEALED_KONG = 4


_KONGS = (MeldKind.EXPOSED_KONG, MeldKind.ADDED_KONG, MeldKind.CONCEALED_KONG)


class Meld:
    """An exposed (or declared) meld. Treated as immutable: replace it rather than mutate."""

    __slots__ = ("kind", "tile", "claimed", "from_")

    def __init__(self, kind: MeldKind, tile: int, claimed: Optional[int] = None, from_: Optional[int] = None):
        self.kind = kind
        self.tile = tile            # lowest tile for a chow, the tile itself otherwise
        self.claimed = claimed      # which tile was claimed, if claimed
        self.from_ = from_          # seat the claimed tile came from

    def is_kong(self) -> bool:
        return self.kind in _KONGS

    def is_triplet_like(self) -> bool:
        return self.kind != MeldKind.CHOW

    def is_open(self) -> bool:
        """True when the meld was built from another player's discard (breaks men qing)."""
        return self.kind != MeldKind.CONCEALED_KONG

    def tiles(self) -> List[int]:
        if self.kind == MeldKind.CHOW:
            return [self.tile, self.tile + 1, self.tile + 2]
        if self.kind == MeldKind.PONG:
            return [self.tile] * 3
        return [self.tile] * 4

    def replace(self, **kw) -> "Meld":
        d = {"kind": self.kind, "tile": self.tile, "claimed": self.claimed, "from_": self.from_}
        d.update(kw)
        return Meld(**d)

    def __eq__(self, o):
        return isinstance(o, Meld) and (self.kind, self.tile, self.claimed, self.from_) == (o.kind, o.tile, o.claimed, o.from_)

    def __repr__(self):
        return f"Meld({self.kind.name}, {self.tile}, claimed={self.claimed}, from_={self.from_})"


# A set inside a winning decomposition of the concealed tiles: (is_chow, tile).
Set = Tuple[bool, int]


@dataclass
class Decomposition:
    pair: int
    sets: List[Set]


def decompositions(c: Sequence[int], n: int) -> List[Decomposition]:
    """All ways to split `c` into `n` sets plus one pair (standard form)."""
    if sum(c) != 3 * n + 2:
        return []
    out: List[Decomposition] = []
    cc = list(c)
    for p in range(NUM_KINDS):
        if cc[p] >= 2:
            cc[p] -= 2
            sets: List[Set] = []
            _split_sets(cc, 0, sets, lambda s, p=p: out.append(Decomposition(p, list(s))))
            cc[p] += 2
    return out


def _split_sets(c: List[int], start: int, sets: List[Set], emit) -> None:
    i = start
    while i < NUM_KINDS and c[i] == 0:
        i += 1
    if i == NUM_KINDS:
        emit(sets)
        return
    if c[i] >= 3:
        c[i] -= 3
        sets.append((False, i))
        _split_sets(c, i, sets, emit)
        sets.pop()
        c[i] += 3
    if i < 27 and i % 9 <= 6 and c[i + 1] > 0 and c[i + 2] > 0:
        c[i] -= 1
        c[i + 1] -= 1
        c[i + 2] -= 1
        sets.append((True, i))
        _split_sets(c, i, sets, emit)
        sets.pop()
        c[i] += 1
        c[i + 1] += 1
        c[i + 2] += 1


def is_thirteen_orphans(c: Sequence[int]) -> bool:
    pair = False
    for t in ORPHANS:
        n = c[t]
        if n == 0 or n > 2:
            return False
        if n == 2:
            pair = True
    return pair and sum(c) == 14


def is_complete_shape(c: Sequence[int], n_melds: int) -> bool:
    """Is the concealed part plus exposed melds a complete hand shape?"""
    if n_melds == 0 and is_thirteen_orphans(c):
        return True
    return shanten_standard(c, 4 - n_melds) == -1


# ---------------------------------------------------------------- scoring

class Pattern(IntEnum):
    SEAT_FLOWER = 0
    FLOWER_SET = 1
    ANIMAL = 2
    DRAGON_PONG = 3
    SEAT_WIND_PONG = 4
    PREVAILING_WIND_PONG = 5
    ALL_PONGS = 6
    HALF_FLUSH = 7
    PING_HU = 8
    PING_HU_WITH_FLOWERS = 9
    MEN_QING = 10
    SMALL_THREE_DRAGONS = 11
    SMALL_FOUR_WINDS = 12
    KONG_REPLACEMENT_WIN = 13
    ROBBING_KONG = 14
    LAST_TILE = 15
    # limit hands
    FULL_FLUSH = 16
    THIRTEEN_ORPHANS = 17
    BIG_THREE_DRAGONS = 18
    BIG_FOUR_WINDS = 19
    ALL_HONOURS = 20
    FOUR_CONCEALED_PONGS = 21
    FOUR_KONGS = 22
    NINE_GATES = 23
    HEAVENLY_HAND = 24
    EARTHLY_HAND = 25

    def label(self) -> str:
        return _PATTERN_NAMES[self]


_PATTERN_NAMES = {
    Pattern.SEAT_FLOWER: "Flower matching seat",
    Pattern.FLOWER_SET: "Complete flower set",
    Pattern.ANIMAL: "Animal",
    Pattern.DRAGON_PONG: "Dragon pong",
    Pattern.SEAT_WIND_PONG: "Seat wind pong",
    Pattern.PREVAILING_WIND_PONG: "Prevailing wind pong",
    Pattern.ALL_PONGS: "All pongs (pong pong hu)",
    Pattern.HALF_FLUSH: "Half flush",
    Pattern.PING_HU: "Ping hu",
    Pattern.PING_HU_WITH_FLOWERS: "Open ping hu with non-scoring flowers",
    Pattern.MEN_QING: "Men qing (fully concealed)",
    Pattern.SMALL_THREE_DRAGONS: "Small three dragons",
    Pattern.SMALL_FOUR_WINDS: "Small four winds",
    Pattern.KONG_REPLACEMENT_WIN: "Win on kong replacement",
    Pattern.ROBBING_KONG: "Robbing a kong",
    Pattern.LAST_TILE: "Win on last tile",
    Pattern.FULL_FLUSH: "Full flush",
    Pattern.THIRTEEN_ORPHANS: "Thirteen orphans",
    Pattern.BIG_THREE_DRAGONS: "Big three dragons",
    Pattern.BIG_FOUR_WINDS: "Big four winds",
    Pattern.ALL_HONOURS: "All honours",
    Pattern.FOUR_CONCEALED_PONGS: "Four concealed pongs",
    Pattern.FOUR_KONGS: "Four kongs",
    Pattern.NINE_GATES: "Nine gates",
    Pattern.HEAVENLY_HAND: "Heavenly hand",
    Pattern.EARTHLY_HAND: "Earthly hand",
}

MIN_TAI = 1
MAX_TAI = 5


@dataclass
class WinContext:
    seat_wind: int = 0          # 0 = East .. 3 = North, relative to the dealer
    prevailing_wind: int = 0
    self_draw: bool = False
    win_tile: int = 0
    kong_replacement: bool = False
    robbing_kong: bool = False
    last_tile: bool = False
    heavenly: bool = False
    earthly: bool = False

    def copy(self, **kw) -> "WinContext":
        d = dict(self.__dict__)
        d.update(kw)
        return WinContext(**d)


@dataclass
class Score:
    tai: int                                   # after the cap
    raw_tai: int                               # before the cap
    items: List[Tuple[Pattern, int]] = field(default_factory=list)


def bonus_tai(bonus: Sequence[int], seat_wind: int) -> Tuple[int, List[Tuple[Pattern, int]]]:
    """Tai from bonus tiles alone (flowers and animals)."""
    items: List[Tuple[Pattern, int]] = []
    tai = 0
    for st in range(2):
        held = [b for b in bonus if is_flower(b) and flower_set(b) == st]
        if len(held) == 4:
            items.append((Pattern.FLOWER_SET, 2))
            tai += 2
        elif any(flower_number(b) == seat_wind + 1 for b in held):
            items.append((Pattern.SEAT_FLOWER, 1))
            tai += 1
    for b in bonus:
        if is_animal(b):
            items.append((Pattern.ANIMAL, 1))
            tai += 1
    return tai, items


def score_hand(concealed: Sequence[int], melds: Sequence[Meld], bonus: Sequence[int],
               ctx: WinContext) -> Optional[Score]:
    """Score a complete hand. `concealed` includes the winning tile.

    Returns None if the tiles do not form a winning shape.
    """
    n_melds = len(melds)
    n_sets = 4 - n_melds
    men_qing = all(not m.is_open() for m in melds)
    btai, bitems = bonus_tai(bonus, ctx.seat_wind)
    has_bonus = len(bonus) > 0

    situational: List[Tuple[Pattern, int]] = []
    if ctx.kong_replacement:
        situational.append((Pattern.KONG_REPLACEMENT_WIN, 1))
    if ctx.robbing_kong:
        situational.append((Pattern.ROBBING_KONG, 1))
    if ctx.last_tile:
        situational.append((Pattern.LAST_TILE, 1))

    # All playing tiles in the hand (for flush/honour checks).
    all_ = list(concealed)
    for m in melds:
        for t in m.tiles():
            all_[t] += 1

    best: List[Optional[Score]] = [None]

    def consider(items: List[Tuple[Pattern, int]]) -> None:
        raw = min(sum(t for _, t in items), 255)
        s = Score(min(raw, MAX_TAI), raw, items)
        b = best[0]
        if b is None or (s.tai, s.raw_tai) > (b.tai, b.raw_tai):
            best[0] = s

    limit_common: List[Tuple[Pattern, int]] = []
    if ctx.heavenly:
        limit_common.append((Pattern.HEAVENLY_HAND, MAX_TAI))
    if ctx.earthly:
        limit_common.append((Pattern.EARTHLY_HAND, MAX_TAI))

    if n_melds == 0 and is_thirteen_orphans(concealed):
        consider([(Pattern.THIRTEEN_ORPHANS, MAX_TAI)] + limit_common)

    suits_used = [s for s in range(3) if any(all_[s * 9 + r] > 0 for r in range(9))]
    has_honor = any(all_[t] > 0 for t in range(27, 34))
    full_flush = len(suits_used) == 1 and not has_honor
    half_flush = len(suits_used) == 1 and has_honor
    all_honours = len(suits_used) == 0

    seat_w = wind_tile(ctx.seat_wind)
    prev_w = wind_tile(ctx.prevailing_wind)
    win = ctx.win_tile

    for d in decompositions(concealed, n_sets):
        # Each placement of the winning tile can change wait-dependent patterns.
        placements: List[Optional[int]] = []  # None = in pair
        if d.pair == win:
            placements.append(None)
        for i, (chow, st) in enumerate(d.sets):
            covers = (st <= win <= st + 2) if chow else st == win
            if covers:
                placements.append(i)
        for placement in placements:
            items: List[Tuple[Pattern, int]] = list(limit_common) + list(situational)

            # triplet-like sets: (tile, concealed?)
            trips: List[Tuple[int, bool]] = []
            chows = 0
            for i, (chow, st) in enumerate(d.sets):
                if chow:
                    chows += 1
                else:
                    completed_by_discard = not ctx.self_draw and placement == i
                    trips.append((st, not completed_by_discard))
            for m in melds:
                if m.is_triplet_like():
                    trips.append((m.tile, m.kind == MeldKind.CONCEALED_KONG))
                else:
                    chows += 1
            n_kongs = sum(1 for m in melds if m.is_kong())

            dragon_trips = sum(1 for t, _ in trips if is_dragon(t))
            wind_trips = sum(1 for t, _ in trips if is_wind(t))

            # limit hands
            if dragon_trips == 3:
                items.append((Pattern.BIG_THREE_DRAGONS, MAX_TAI))
            if wind_trips == 4:
                items.append((Pattern.BIG_FOUR_WINDS, MAX_TAI))
            if all_honours:
                items.append((Pattern.ALL_HONOURS, MAX_TAI))
            if full_flush:
                items.append((Pattern.FULL_FLUSH, MAX_TAI))
            if len(trips) == 4 and all(conc for _, conc in trips):
                items.append((Pattern.FOUR_CONCEALED_PONGS, MAX_TAI))
            if n_kongs == 4:
                items.append((Pattern.FOUR_KONGS, MAX_TAI))
            if n_melds == 0 and full_flush and _is_nine_gates(concealed):
                items.append((Pattern.NINE_GATES, MAX_TAI))

            # dragons
            if dragon_trips == 2 and is_dragon(d.pair):
                items.append((Pattern.SMALL_THREE_DRAGONS, 4))
            elif dragon_trips < 3:
                for _ in range(dragon_trips):
                    items.append((Pattern.DRAGON_PONG, 1))
            # winds
            if wind_trips == 3 and is_wind(d.pair):
                items.append((Pattern.SMALL_FOUR_WINDS, 4))
            elif wind_trips < 4:
                if any(t == seat_w for t, _ in trips):
                    items.append((Pattern.SEAT_WIND_PONG, 1))
                if any(t == prev_w for t, _ in trips):
                    items.append((Pattern.PREVAILING_WIND_PONG, 1))

            if len(trips) == 4:
                items.append((Pattern.ALL_PONGS, 2))
            if half_flush:
                items.append((Pattern.HALF_FLUSH, 2))

            # ping hu: four chows, plain pair, two-sided wait
            if chows == 4 and not is_dragon(d.pair) and d.pair != seat_w and d.pair != prev_w:
                two_sided = False
                if placement is not None:
                    chow, st = d.sets[placement]
                    if chow:
                        r = rank(st)
                        two_sided = (win == st and r <= 5) or (win == st + 2 and r >= 1)
                if two_sided:
                    # Scoring bonus tiles (seat flower, animal, flower set) void ping hu.
                    # Non-scoring flowers: a fully concealed ping hu keeps its 4 tai,
                    # an open one (exposed chows) drops to 1 tai.
                    if btai == 0:
                        if not has_bonus or n_melds == 0:
                            items.append((Pattern.PING_HU, 4))
                        else:
                            items.append((Pattern.PING_HU_WITH_FLOWERS, 1))

            if men_qing:
                items.append((Pattern.MEN_QING, 1))
            items.extend(bitems)
            consider(items)
    return best[0]


def _is_nine_gates(c: Sequence[int]) -> bool:
    s = next((s for s in range(3) if any(c[s * 9 + r] > 0 for r in range(9))), None)
    if s is None:
        return False
    base = (3, 1, 1, 1, 1, 1, 1, 1, 3)
    extra = 0
    for r in range(9):
        n = c[s * 9 + r]
        if n < base[r]:
            return False
        extra += n - base[r]
    return extra == 1


def unit_points(tai: int) -> int:
    """Unit price of a hand: 1 tai = 1, doubling per tai."""
    return 1 << (max(tai, 1) - 1)


def payments(tai: int, winner: int, discarder: Optional[int]) -> List[int]:
    """Point transfers for a win. Index = seat. Sum is always zero.

    Discard win: the discarder pays for everyone (4 / 8 / 16 / 32 / 64 for 1-5 tai).
    Self-draw: each other player pays double (2 / 4 / 8 / 16 / 32).
    """
    d = [0, 0, 0, 0]
    unit = unit_points(tai)
    if discarder is not None:
        amt = 4 * unit
        d[discarder] -= amt
        d[winner] += amt
    else:
        for s in range(4):
            if s != winner:
                d[s] -= 2 * unit
        d[winner] += 6 * unit
    return d
