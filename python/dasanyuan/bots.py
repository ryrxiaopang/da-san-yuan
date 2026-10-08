"""Heuristic bots. They see only a `View` (public information + own hand).

Each bot is the same algorithm driven by a `Style` of weights, so the four
archetypes and any random variations of them live in one parameter space.
That is what the evolution / league scripts mutate.

All scores are computed in single precision (see `f32.py`) so that play matches
the recorded datasets move for move.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from .f32 import exp32, f32, fmt32
from .game import CLAIM, CHOW, EXPOSED_KONG, KONG, PASS, PONG, ROB_KONG, RON, SELF_TURN, TSUMO
from .obs import View
from .rng import Rng
from .scoring import Meld, MeldKind, bonus_tai
from .shanten import shanten, ukeire
from .tile import is_dragon, is_honor, is_suited, rank, suit, wind_tile

NEG_INF = float("-inf")

# f32 constants used by the heuristics
_C0_3 = f32(0.3)
_C0_35 = f32(0.35)
_C0_4 = f32(0.4)
_C0_45 = f32(0.45)
_C0_6 = f32(0.6)
_C0_7 = f32(0.7)
_C0_15 = f32(0.15)


class Bot:
    def act(self, view: View) -> int:
        raise NotImplementedError

    def name(self) -> str:
        raise NotImplementedError


@dataclass
class Style:
    """Weights for the heuristic. All non-negative; scale is roughly "tiles of ukeire"."""
    name: str
    speed: float     # weight on number of useful tiles (speed to ready)
    value: float     # weight on keeping a hand that scores tai (honour pongs, flush, all pongs)
    defence: float   # weight on avoiding dangerous discards when opponents look close
    claim: float     # willingness to claim pong/chow (0 = almost never, 1 = whenever it speeds up)
    flush: float     # preference for flush building
    pongs: float     # preference for all-pongs shapes

    def __post_init__(self):
        for k in ("speed", "value", "defence", "claim", "flush", "pongs"):
            setattr(self, k, f32(getattr(self, k)))

    def clone(self) -> "Style":
        return replace(self)

    @staticmethod
    def fast() -> "Style":
        return Style("fast", 1.0, 0.6, 0.2, 0.9, 0.2, 0.2)

    @staticmethod
    def high_tai() -> "Style":
        return Style("high_tai", 0.6, 2.0, 0.4, 0.4, 1.5, 1.0)

    @staticmethod
    def defensive() -> "Style":
        return Style("defensive", 0.8, 0.8, 2.5, 0.3, 0.4, 0.3)

    @staticmethod
    def balanced() -> "Style":
        return Style("balanced", 0.9, 1.0, 1.0, 0.6, 0.6, 0.5)

    @staticmethod
    def by_name(name: str) -> Optional["Style"]:
        return {
            "fast": Style.fast, "high_tai": Style.high_tai, "value": Style.high_tai,
            "defensive": Style.defensive, "balanced": Style.balanced,
        }.get(name, lambda: None)()

    @staticmethod
    def archetypes() -> List["Style"]:
        return [Style.fast(), Style.high_tai(), Style.defensive(), Style.balanced()]

    def mutate(self, rng: Rng, sigma: float, name: str) -> "Style":
        """Random variation for evolutionary search: each weight scaled by exp(N(0, sigma))."""
        sigma = f32(sigma)

        def g() -> float:
            # Box-Muller
            u1 = max(rng.unit(), 1e-12)
            u2 = rng.unit()
            return f32(f32(math.sqrt(-2.0 * math.log(u1)) * math.cos(math.tau * u2)) * sigma)

        speed = f32(self.speed * exp32(g()))
        value = f32(self.value * exp32(g()))
        defence = f32(self.defence * exp32(g()))
        claim = min(f32(self.claim * exp32(g())), 1.5)
        flush = f32(self.flush * exp32(g()))
        pongs = f32(self.pongs * exp32(g()))
        return Style(name, speed, value, defence, claim, flush, pongs)

    def to_csv(self) -> str:
        return ",".join([self.name] + [fmt32(x) for x in (self.speed, self.value, self.defence, self.claim,
                                                          self.flush, self.pongs)])

    @staticmethod
    def from_csv(s: str) -> Optional["Style"]:
        f = s.strip().split(",")
        if len(f) != 7:
            return None
        try:
            vals = [float(x) for x in f[1:]]
        except ValueError:
            return None
        if any(x.strip() != x or not x for x in f[1:]):
            return None
        return Style(f[0], *vals)


def valued_honor(t: int, seat_wind: int, prevailing: int) -> bool:
    """Is `t` a tile whose pong scores tai for this seat?"""
    return is_dragon(t) or t == wind_tile(seat_wind) or t == wind_tile(prevailing)


class _Ctx:
    __slots__ = ("melds", "bonus_tai", "seat_wind", "prevailing", "unseen", "threat", "opp_discards",
                 "threat_by_seat", "suit_focus", "me")

    def open(self) -> bool:
        return any(m.is_open() for m in self.melds)


def tai_outlook(hand: Sequence[int], melds: Sequence[Meld], ctx: _Ctx, style: Style) -> Tuple[bool, float]:
    """How likely the hand can reach the 1-tai minimum, plus how much tai it is building.

    Returns (secured, potential).
    """
    open_ = any(m.is_open() for m in melds)
    secured = ctx.bonus_tai >= 1 or not open_
    potential = 0.0
    sw, pw = ctx.seat_wind, ctx.prevailing
    for m in melds:
        if m.is_triplet_like() and valued_honor(m.tile, sw, pw):
            secured = True
            potential = f32(potential + 1.0)
    for t in range(27, 34):
        if not valued_honor(t, sw, pw):
            continue
        n = hand[t]
        if n >= 3:
            secured = True
            potential = f32(potential + 1.0)
        elif n == 2 and ctx.unseen[t] > 0:
            potential = f32(potential + _C0_45)
    # Flush: share of playing tiles in the dominant suit (+ honours for half flush)
    by_suit = [sum(hand[0:9]), sum(hand[9:18]), sum(hand[18:27]), sum(hand[27:34])]
    for m in melds:
        by_suit[suit(m.tile)] += 3
    total = by_suit[0] + by_suit[1] + by_suit[2] + by_suit[3]
    if total > 0:
        main = max(by_suit[0], by_suit[1], by_suit[2])
        others = total - main - by_suit[3]
        frac = f32((main + by_suit[3]) / total)
        if others == 0:
            secured = True  # a half flush is already guaranteed
            potential = f32(potential + (2.0 + (2.0 if by_suit[3] == 0 else 0.0)))
        elif frac >= _C0_7:
            x = f32(f32(f32(style.flush * f32(frac - _C0_6)) * 5.0) / others)
            potential = f32(potential + x)
    # All pongs shape
    pairs_trips = float(sum(1 for k in range(34) if hand[k] >= 2) + sum(1 for m in melds if m.is_triplet_like()))
    chow_melds = sum(1 for m in melds if m.kind == MeldKind.CHOW)
    if chow_melds == 0 and pairs_trips >= 4.0:
        potential = f32(potential + f32(f32(style.pongs * (pairs_trips - 3.0)) * _C0_6))
    return secured, potential


def danger(t: int, ctx: _Ctx) -> float:
    """How risky is discarding `t` right now (0 = safe, ~1 = dangerous middle tile)."""
    if ctx.threat <= 0.0:
        return 0.0
    if is_honor(t):
        u = ctx.unseen[t]
        base = 0.0 if u == 0 else (_C0_15 if u == 1 else _C0_45)
    else:
        rk = rank(t)
        r = 0.5 if rk in (0, 8) else (0.75 if rk in (1, 7) else 1.0)
        # No unseen copies: nobody can be waiting on it as a pair or pong, only in a sequence.
        base = f32(r * _C0_6) if ctx.unseen[t] == 0 else r
    d = 0.0
    bit = 1 << t
    for s in range(4):
        if s == ctx.me:
            continue
        w = ctx.threat_by_seat[s]
        if w <= 0.0:
            continue
        # Tiles an opponent threw recently are less likely to be what they wait on.
        safe = ctx.opp_discards[s] & bit != 0
        # Flush read: an opponent whose melds are all one suit wants that suit (and honours).
        f = ctx.suit_focus[s]
        if f is not None and is_suited(t):
            focus = 2.0 if suit(t) == f else _C0_3
        else:
            focus = 1.0
        d = f32(d + f32(f32(f32(w * base) * focus) * (0.25 if safe else 1.0)))
    return d


def eval_hand(hand: Sequence[int], need: int, ctx: _Ctx, style: Style, melds: Sequence[Meld]) -> Tuple[int, float]:
    sh = shanten(hand, need)
    uk, _ = ukeire(hand, need, ctx.unseen)
    secured, potential = tai_outlook(hand, melds, ctx, style)
    score = f32(f32(f32(-sh * 40.0) + f32(style.speed * uk)) + f32(f32(style.value * potential) * 3.0))
    if not secured:
        # No route to the 1-tai minimum yet: heavily discourage unless potential exists.
        score = f32(score - f32(max(f32(2.0 - potential), 0.0) * 6.0))
    return sh, score


class HeuristicBot(Bot):
    def __init__(self, style: Style, seed: int):
        self.style = style
        self.rng = Rng(seed)
        # Small random noise added to scores so identical styles don't play identically.
        self.noise = f32(0.05)

    def name(self) -> str:
        return self.style.name

    def _make_ctx(self, v: View) -> _Ctx:
        me = v.seat
        bt, _ = bonus_tai(v.my_bonus(), v.seat_wind())
        threat_by_seat = [0.0] * 4
        opp_discards = [0] * 4
        suit_focus: List[Optional[int]] = [None] * 4
        late = f32(1.0 - min(f32(v.draws_left() / 80.0), 1.0))
        late_term = f32(late * _C0_6)
        for s in range(4):
            if s == me:
                continue
            melds = v.melds_of(s)
            open_ = float(len(melds))
            small_hand = v.hand_size_of(s) <= 7
            x = f32(f32(open_ * _C0_3) + late_term)
            x = f32(x + (_C0_4 if small_hand else 0.0))
            threat_by_seat[s] = max(f32(x - _C0_35), 0.0)
            d = v.discards_of(s)
            for disc in d[-8:]:
                opp_discards[s] |= 1 << disc.tile
            suits = [suit(m.tile) for m in melds if is_suited(m.tile)]
            if len(melds) >= 2 and suits and all(x == suits[0] for x in suits):
                # and they have not been throwing that suit away
                thrown = sum(1 for x in d[-10:] if is_suited(x.tile) and suit(x.tile) == suits[0])
                if thrown <= 1:
                    suit_focus[s] = suits[0]
                    threat_by_seat[s] = f32(threat_by_seat[s] + _C0_3)
        c = _Ctx()
        c.melds = v.my_melds()
        c.bonus_tai = bt
        c.seat_wind = v.seat_wind()
        c.prevailing = v.prevailing_wind()
        c.unseen = v.unseen()
        threat = 0.0
        for t in threat_by_seat:
            threat = f32(threat + t)
        c.threat = threat
        c.opp_discards = opp_discards
        c.threat_by_seat = threat_by_seat
        c.suit_focus = suit_focus
        c.me = me
        return c

    def _jitter(self) -> float:
        return f32(f32(f32(self.rng.unit()) - 0.5) * self.noise)

    def _best_discard(self, hand: Sequence[int], melds: Sequence[Meld], ctx: _Ctx) -> Tuple[int, float, int]:
        """Best discard for a hand with 3n+2 tiles: (tile, score, shanten after discard)."""
        need = 4 - len(melds)
        best = (0, NEG_INF, 99)
        h = list(hand)
        style = self.style
        for k in range(34):
            if h[k] == 0:
                continue
            h[k] -= 1
            sh, s = eval_hand(h, need, ctx, style, melds)
            s = f32(s - f32(f32(style.defence * danger(k, ctx)) * 25.0))
            s = f32(s + self._jitter())
            if s > best[1]:
                best = (k, s, sh)
            h[k] += 1
        return best

    def _own_turn(self, v: View, acts: List[int]) -> int:
        if TSUMO in acts:
            return TSUMO
        ctx = self._make_ctx(v)
        hand = v.hand()
        tile, _score, _ = self._best_discard(hand, ctx.melds, ctx)
        # Consider kongs: take one if the hand after it is at least as good.
        for a in acts:
            if not KONG <= a < KONG + 34:
                continue
            t = a - KONG
            h = list(hand)
            melds = list(ctx.melds)
            if h[t] == 4:
                h[t] = 0
                melds.append(Meld(MeldKind.CONCEALED_KONG, t, None, None))
            else:
                h[t] -= 1
                # Added kongs can be robbed; defensive styles shy away when someone is close.
                if f32(self.style.defence * ctx.threat) > 1.5:
                    continue
                for i, m in enumerate(melds):
                    if m.tile == t and m.kind == MeldKind.PONG:
                        melds[i] = m.replace(kind=MeldKind.ADDED_KONG)
                        break
            need = 4 - len(melds)
            x = list(hand)
            x[tile] -= 1
            before = shanten(x, 4 - len(ctx.melds))
            # After a kong we draw a replacement, so compare 3n+1-tile shapes.
            after = shanten(h, need)
            if after <= before:
                return a
        return tile

    def _claim_turn(self, v: View, acts: List[int], tile: int) -> int:
        if RON in acts:
            return RON
        ctx = self._make_ctx(v)
        hand = v.hand()
        need = 4 - len(ctx.melds)
        cur_sh, cur_score = eval_hand(hand, need, ctx, self.style, ctx.melds)
        best_a, best_s = PASS, cur_score
        for a in acts:
            if a == PONG or a == EXPOSED_KONG:
                h = list(hand)
                h[tile] -= 2 if a == PONG else 3
                meld = Meld(MeldKind.PONG if a == PONG else MeldKind.EXPOSED_KONG, tile, tile, None)
            elif CHOW <= a <= CHOW + 2:
                low = tile - (a - CHOW)
                h = list(hand)
                for t in range(low, low + 3):
                    if t != tile:
                        h[t] -= 1
                meld = Meld(MeldKind.CHOW, low, tile, None)
            else:
                continue
            melds = list(ctx.melds) + [meld]
            if a == EXPOSED_KONG:
                # replacement draw follows: evaluate the 3n+1 hand directly
                score = eval_hand(h, 4 - len(melds), ctx, self.style, melds)[1]
            else:
                _, score, sh = self._best_discard(h, melds, ctx)
                if sh >= cur_sh:
                    continue  # claim must move us closer to ready
            # Opening the hand forfeits men qing; demand more gain the less the style likes claiming.
            breaks_men_qing = not ctx.open()
            valued = a in (PONG, EXPOSED_KONG) and valued_honor(tile, ctx.seat_wind, ctx.prevailing)
            threshold = f32(f32(1.0 - self.style.claim) * 30.0)
            if breaks_men_qing and not valued:
                threshold = f32(threshold + 10.0)
            secured, _ = tai_outlook(h, melds, ctx, self.style)
            if not secured and not valued:
                continue
            gain = f32(f32(f32(score - cur_score) - threshold) + self._jitter())
            net = f32(score - threshold)
            if gain > 0.0 and net > best_s:
                best_a, best_s = a, net
        return best_a

    def act(self, v: View) -> int:
        acts = v.legal_actions()
        if len(acts) == 1:
            return acts[0]
        ph = v.phase()
        if ph.kind == SELF_TURN:
            return self._own_turn(v, acts)
        if ph.kind == CLAIM:
            return self._claim_turn(v, acts, ph.tile)
        if ph.kind == ROB_KONG:
            return RON if RON in acts else PASS
        raise RuntimeError("bot asked to act after the hand ended")


class RandomBot(Bot):
    """Uniformly random legal discards, but always takes a win. Useful as a floor baseline."""

    def __init__(self, seed: int):
        self.rng = Rng(seed)

    def act(self, v: View) -> int:
        acts = v.legal_actions()
        for w in (TSUMO, RON):
            if w in acts:
                return w
        discards = [a for a in acts if a < 34]
        if discards:
            return discards[self.rng.below(len(discards))]
        return PASS

    def name(self) -> str:
        return "random"


class EfficiencyBot(Bot):
    """Pure tile efficiency: always wins when it can, never claims, and discards the tile
    that keeps the lowest shanten with the most useful tiles. No defence, no tai planning.
    The baseline every learned model must beat on the scenario bank."""

    def act(self, v: View) -> int:
        acts = v.legal_actions()
        for w in (TSUMO, RON):
            if w in acts:
                return w
        if v.phase().kind == SELF_TURN:
            from .scenario import efficiency_ranking
            return efficiency_ranking(v)[0][0]
        return PASS

    def name(self) -> str:
        return "efficiency"


def make_bot(name: str, seed: int) -> Optional[Bot]:
    if name == "random":
        return RandomBot(seed)
    if name == "efficiency":
        return EfficiencyBot()
    s = Style.by_name(name)
    return HeuristicBot(s, seed) if s is not None else None
