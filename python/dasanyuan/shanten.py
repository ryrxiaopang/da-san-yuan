"""Shanten (tiles-away-from-ready) calculation.

Shanten -1 means the hand is complete, 0 means ready (one tile away), etc.
Standard form: 4 sets + 1 pair. Thirteen Orphans is handled separately.

Each suit's count vector maps to an entry: for (has_pair, melds) the maximum number
of partial sets ("taatsu"), or -1 if unreachable. Entries follow the same recurrence
as a full table over all 5^9 vectors, but are computed lazily and cached, since a
game only ever meets a small fraction of them. Suits are merged with a tiny
max-plus convolution.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .tile import NUM_KINDS, ORPHANS

POW5 = (1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125)

# An entry is a tuple of (p, m, t) triples for the reachable slots, where t is the
# max taatsu (capped at 4) with p pairs (0/1) and m melds (0..4).
Entry = Tuple[Tuple[int, int, int], ...]

_EMPTY: Entry = ((0, 0, 0),)


def _merge(dst: List[int], src: Entry, dm: int, dt: int, dp: int) -> None:
    """dst is a flat [2][5] array of max taatsu (-1 = unreachable)."""
    for p, m, t in src:
        np_ = p + dp
        nm = m + dm
        if np_ > 1 or nm > 4:
            continue
        nt = t + dt
        if nt > 4:
            nt = 4
        i = np_ * 5 + nm
        if nt > dst[i]:
            dst[i] = nt


def _pack(e: List[int]) -> Entry:
    return tuple((i // 5, i % 5, t) for i, t in enumerate(e) if t >= 0)


_suit_cache: Dict[int, Entry] = {0: _EMPTY}
_honor_cache: Dict[int, Entry] = {0: _EMPTY}


def _suit_entry(idx: int) -> Entry:
    e = _suit_cache.get(idx)
    if e is not None:
        return e
    c = [(idx // POW5[k]) % 5 for k in range(9)]
    i = next(k for k in range(9) if c[k] > 0)
    d = [-1] * 10
    # leave one tile isolated
    _merge(d, _suit_entry(idx - POW5[i]), 0, 0, 0)
    if c[i] >= 3:
        _merge(d, _suit_entry(idx - 3 * POW5[i]), 1, 0, 0)
    if c[i] >= 2:
        s = _suit_entry(idx - 2 * POW5[i])
        _merge(d, s, 0, 0, 1)  # pair as head
        _merge(d, s, 0, 1, 0)  # pair as partial set
    if i <= 6 and c[i + 1] > 0 and c[i + 2] > 0:
        _merge(d, _suit_entry(idx - POW5[i] - POW5[i + 1] - POW5[i + 2]), 1, 0, 0)
    if i <= 7 and c[i + 1] > 0:
        _merge(d, _suit_entry(idx - POW5[i] - POW5[i + 1]), 0, 1, 0)
    if i <= 6 and c[i + 2] > 0:
        _merge(d, _suit_entry(idx - POW5[i] - POW5[i + 2]), 0, 1, 0)
    e = _pack(d)
    _suit_cache[idx] = e
    return e


def _honor_entry(idx: int) -> Entry:
    # Honours can't form sequences, so each kind is independent.
    e = _honor_cache.get(idx)
    if e is not None:
        return e
    c = [(idx // POW5[k]) % 5 for k in range(7)]
    i = next(k for k in range(7) if c[k] > 0)
    d = [-1] * 10
    _merge(d, _honor_entry(idx - POW5[i]), 0, 0, 0)
    if c[i] >= 3:
        _merge(d, _honor_entry(idx - 3 * POW5[i]), 1, 0, 0)
    if c[i] >= 2:
        s = _honor_entry(idx - 2 * POW5[i])
        _merge(d, s, 0, 0, 1)
        _merge(d, s, 0, 1, 0)
    e = _pack(d)
    _honor_cache[idx] = e
    return e


def init_tables() -> None:
    """Kept for API compatibility: entries are now built on demand."""


_combine_cache: Dict[Tuple[Entry, Entry], Entry] = {}


def combine(a: Entry, b: Entry) -> Entry:
    key = (a, b)
    r = _combine_cache.get(key)
    if r is not None:
        return r
    out = [-1] * 10
    for pa, ma, ta in a:
        for pb, mb, tb in b:
            p = pa + pb
            m = ma + mb
            if p > 1 or m > 4:
                continue
            t = ta + tb
            if t > 4:
                t = 4
            i = p * 5 + m
            if t > out[i]:
                out[i] = t
    r = _pack(out)
    if len(_combine_cache) > 200_000:
        _combine_cache.clear()
    _combine_cache[key] = r
    return r


def _suit_index(c: Sequence[int], base: int) -> int:
    idx = 0
    for i in range(8, -1, -1):
        idx = idx * 5 + c[base + i]
    return idx


def _honor_index(c: Sequence[int]) -> int:
    idx = 0
    for i in range(6, -1, -1):
        idx = idx * 5 + c[27 + i]
    return idx


def _best(acc: Entry, n: int) -> int:
    best = 8
    for p, m, t in acc:
        if m > n:
            continue
        if t > n - m:
            t = n - m
        s = 2 * n - 2 * m - t - p
        if s < best:
            best = s
    return best


def _entries(c: Sequence[int]):
    return (_suit_entry(_suit_index(c, 0)), _suit_entry(_suit_index(c, 9)),
            _suit_entry(_suit_index(c, 18)), _honor_entry(_honor_index(c)))


def shanten_standard(c: Sequence[int], melds_needed: int) -> int:
    """Shanten of the standard 4-sets-plus-pair form.

    `melds_needed` = 4 minus the number of melds already exposed (or declared kongs).
    """
    e0, e1, e2, e3 = _entries(c)
    return _best(combine(combine(combine(e0, e1), e2), e3), melds_needed)


def shanten_orphans(c: Sequence[int]) -> int:
    """Thirteen Orphans shanten (only possible with no exposed melds)."""
    kinds = 0
    pair = 0
    for t in ORPHANS:
        n = c[t]
        if n > 0:
            kinds += 1
            if n >= 2:
                pair = 1
    return 13 - kinds - pair


def shanten(c: Sequence[int], melds_needed: int) -> int:
    """Overall shanten: min over standard form and (when fully concealed) Thirteen Orphans."""
    s = shanten_standard(c, melds_needed)
    if melds_needed == 4:
        return min(s, shanten_orphans(c))
    return s


def ukeire(c: Sequence[int], melds_needed: int, unseen: Sequence[int]) -> Tuple[int, int]:
    """Tiles that would lower shanten if drawn, weighted by how many copies remain unseen.

    `unseen[k]` is the number of copies of kind k this player cannot see.
    Returns (total useful copies, bitmask of useful kinds).
    """
    n = melds_needed
    es = list(_entries(c))
    base = _best(combine(combine(combine(es[0], es[1]), es[2]), es[3]), n)
    orph = n == 4
    if orph:
        base = min(base, shanten_orphans(c))
    # For each suit, the merge of the other three suits, so adding one tile needs one combine.
    rest = [
        combine(combine(es[1], es[2]), es[3]),
        combine(combine(es[0], es[2]), es[3]),
        combine(combine(es[0], es[1]), es[3]),
        combine(combine(es[0], es[1]), es[2]),
    ]
    idx = [_suit_index(c, 0), _suit_index(c, 9), _suit_index(c, 18), _honor_index(c)]
    cc = list(c)
    total = 0
    mask = 0
    for k in range(NUM_KINDS):
        if unseen[k] == 0 or cc[k] >= 4:
            continue
        if k < 27:
            s_i = k // 9
            e = _suit_entry(idx[s_i] + POW5[k % 9])
        else:
            s_i = 3
            e = _honor_entry(idx[3] + POW5[k - 27])
        sh = _best(combine(rest[s_i], e), n)
        if orph:
            cc[k] += 1
            o = shanten_orphans(cc)
            cc[k] -= 1
            if o < sh:
                sh = o
        if sh < base:
            total += unseen[k]
            mask |= 1 << k
    return total, mask
