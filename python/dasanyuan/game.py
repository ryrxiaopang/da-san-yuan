"""Game state machine for one hand of Singapore 4-player mahjong.

Drive it with `to_act()` -> `legal_actions(seat)` -> `apply(seat, action)` until
`is_over()`. Every decision point is explicit, which is what both the heuristic
bots and the RL environment need.

Actions are plain ints laid out exactly like the RL action space (see `obs.py`):
  0..33 discard tile k, 34..67 kong tile k, 68 tsumo, 69 ron, 70 pong,
  71 exposed kong, 72..74 chow with the claimed tile lowest/middle/highest, 75 pass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import scoring
from .rng import Rng
from .scoring import Meld, MeldKind, Score, WinContext
from .tile import FIRST_BONUS, NUM_KINDS, NUM_TILE_IDS, WALL_SIZE, is_bonus, is_suited, rank

# ------------------------------------------------------------------ actions

DISCARD = 0        # + tile kind
KONG = 34          # + tile kind: concealed kong or adding a fourth tile to an exposed pong
TSUMO = 68         # win on a self-drawn tile
RON = 69           # win on a discard (or by robbing a kong)
PONG = 70
EXPOSED_KONG = 71  # exposed kong from a discard
CHOW = 72          # + position 0 (claimed tile lowest), 1 (middle), 2 (highest)
PASS = 75


def Discard_(t: int) -> int:
    return DISCARD + t


def Kong(t: int) -> int:
    return KONG + t


def Chow(p: int) -> int:
    return CHOW + p


def is_discard(a: int) -> bool:
    return a < 34


def is_kong_action(a: int) -> bool:
    return 34 <= a < 68


def is_chow(a: int) -> bool:
    return 72 <= a <= 74


def action_debug(a: int) -> str:
    """Name like the original engine's Debug output: Discard(5), Kong(31), Tsumo, Chow(1)..."""
    if a < 34:
        return f"Discard({a})"
    if a < 68:
        return f"Kong({a - 34})"
    if 72 <= a <= 74:
        return f"Chow({a - 72})"
    return {TSUMO: "Tsumo", RON: "Ron", PONG: "Pong", EXPOSED_KONG: "ExposedKong", PASS: "Pass"}[a]


# ------------------------------------------------------------------- state

@dataclass
class Config:
    reserve: int = 15          # live tiles left undrawn when the hand is declared a draw
    prevailing_wind: int = 0   # 0 = East round
    dealer: int = 0            # seat that deals (sits East)

    def clone(self) -> "Config":
        return Config(self.reserve, self.prevailing_wind, self.dealer)


class Discard:
    """One discarded tile as everyone at the table saw it. Treated as immutable."""

    __slots__ = ("tile", "claimed", "tsumogiri", "turn")

    def __init__(self, tile: int, claimed: bool = False, tsumogiri: bool = False, turn: int = 0):
        self.tile = tile
        self.claimed = claimed      # later claimed by another player (pong/chow/kong/win)
        self.tsumogiri = tsumogiri  # the tile thrown was the one just drawn
        self.turn = turn            # global action counter when it was discarded

    def __eq__(self, o):
        return isinstance(o, Discard) and (self.tile, self.claimed, self.tsumogiri, self.turn) == (
            o.tile, o.claimed, o.tsumogiri, o.turn)

    def __repr__(self):
        return f"Discard({self.tile}, claimed={self.claimed}, tsumogiri={self.tsumogiri}, turn={self.turn})"


class Player:
    __slots__ = ("hand", "bonus", "melds", "discards", "draws", "pending_bonus")

    def __init__(self):
        self.hand: List[int] = [0] * NUM_KINDS
        self.bonus: List[int] = []
        self.melds: List[Meld] = []
        self.discards: List[Discard] = []  # in order
        self.draws = 0
        self.pending_bonus = 0

    def clone(self) -> "Player":
        p = Player.__new__(Player)
        p.hand = list(self.hand)
        p.bonus = list(self.bonus)
        p.melds = list(self.melds)
        p.discards = list(self.discards)
        p.draws = self.draws
        p.pending_bonus = self.pending_bonus
        return p

    def hand_size(self) -> int:
        return sum(self.hand)

    def melds_needed(self) -> int:
        return 4 - len(self.melds)


# Phase kinds
SELF_TURN = 0  # `seat` holds 3n+2 tiles and must discard, kong or declare tsumo
CLAIM = 1      # others may claim `tile` discarded by `seat`
ROB_KONG = 2   # others may rob the kong `seat` is adding with `tile`
OVER = 3


class Phase:
    """Immutable phase. For CLAIM `seat` is the discarder; for ROB_KONG it is the konger."""

    __slots__ = ("kind", "seat", "tile", "drawn", "after_kong", "after_claim")

    def __init__(self, kind: int, seat: int = 0, tile: int = 0, drawn: Optional[int] = None,
                 after_kong: bool = False, after_claim: bool = False):
        self.kind = kind
        self.seat = seat
        self.tile = tile
        self.drawn = drawn
        self.after_kong = after_kong
        self.after_claim = after_claim

    @staticmethod
    def self_turn(seat: int, drawn: Optional[int] = None, after_kong: bool = False, after_claim: bool = False):
        return Phase(SELF_TURN, seat, 0, drawn, after_kong, after_claim)

    @staticmethod
    def claim(from_: int, tile: int):
        return Phase(CLAIM, from_, tile)

    @staticmethod
    def rob_kong(seat: int, tile: int):
        return Phase(ROB_KONG, seat, tile)

    def __eq__(self, o):
        return isinstance(o, Phase) and (self.kind, self.seat, self.tile, self.drawn, self.after_kong,
                                         self.after_claim) == (o.kind, o.seat, o.tile, o.drawn, o.after_kong,
                                                               o.after_claim)

    def __repr__(self):
        if self.kind == SELF_TURN:
            return (f"SelfTurn(seat={self.seat}, drawn={self.drawn}, after_kong={self.after_kong}, "
                    f"after_claim={self.after_claim})")
        if self.kind == CLAIM:
            return f"Claim(from={self.seat}, tile={self.tile})"
        if self.kind == ROB_KONG:
            return f"RobKong(seat={self.seat}, tile={self.tile})"
        return "Over"


_OVER = Phase(OVER)


@dataclass
class HandResult:
    winner: Optional[int]
    discarder: Optional[int]
    self_draw: bool
    score: Optional[Score]
    deltas: List[int]
    turns: int


class Game:
    __slots__ = ("cfg", "wall", "front", "back", "players", "phase", "responses", "result", "any_claim",
                 "turns", "last_draw")

    def __init__(self, cfg: Config, seed: int):
        """Deal a new hand from a seeded shuffle."""
        wall = [k for k in range(NUM_KINDS) for _ in range(4)] + list(range(FIRST_BONUS, NUM_TILE_IDS))
        Rng(seed).shuffle(wall)
        self._init_from_wall(cfg, wall)

    @classmethod
    def from_wall(cls, cfg: Config, wall: List[int]) -> "Game":
        """Deal from an explicit wall order (used by duplicate tournaments and tests)."""
        g = cls.__new__(cls)
        g._init_from_wall(cfg, list(wall))
        return g

    def _init_from_wall(self, cfg: Config, wall: List[int]) -> None:
        assert len(wall) == WALL_SIZE
        self.cfg = cfg
        self.wall = wall
        self.front = 0
        self.back = len(wall)                 # one past the next replacement tile (taken from the back)
        self.players = [Player() for _ in range(4)]
        self.phase = _OVER
        self.responses: List[Optional[int]] = [None] * 4  # answers collected in a claim / rob-kong window
        self.result: Optional[HandResult] = None
        self.any_claim = False
        self.turns = 0
        self.last_draw = False                # set once the last live tile has been drawn
        self._deal()

    @classmethod
    def from_parts(cls, cfg: Config, wall: List[int], front: int, players: List[Player], phase: Phase,
                   decider: int, turns: int) -> "Game":
        """Build a mid-hand position directly (used by scenarios).

        `wall` holds every tile; the first `front` are treated as already drawn.
        For a Claim or RobKong phase only `decider` is asked; the other seats pass.
        """
        assert len(wall) == WALL_SIZE
        g = cls.__new__(cls)
        g.cfg = cfg
        g.wall = wall
        g.front = front
        g.back = len(wall)
        g.players = players
        g.phase = phase
        g.responses = [None] * 4
        if phase.kind in (CLAIM, ROB_KONG):
            for s in range(4):
                if s != decider:
                    g.responses[s] = PASS
        g.result = None
        g.any_claim = any(m.is_open() for p in players for m in p.melds)
        g.turns = turns
        g.last_draw = g.live_remaining() <= cfg.reserve
        return g

    def clone(self) -> "Game":
        g = Game.__new__(Game)
        g.cfg = self.cfg
        g.wall = list(self.wall)
        g.front = self.front
        g.back = self.back
        g.players = [p.clone() for p in self.players]
        g.phase = self.phase
        g.responses = list(self.responses)
        g.result = self.result
        g.any_claim = self.any_claim
        g.turns = self.turns
        g.last_draw = self.last_draw
        return g

    def dealer(self) -> int:
        return self.cfg.dealer

    def seat_wind(self, seat: int) -> int:
        """Seat wind 0..=3 (East..North) for a seat."""
        return (seat + 4 - self.cfg.dealer) % 4

    def live_remaining(self) -> int:
        return self.back - self.front

    def draws_left(self) -> int:
        return max(self.live_remaining() - self.cfg.reserve, 0)

    def is_over(self) -> bool:
        return self.phase.kind == OVER

    def _deal(self) -> None:
        d = self.cfg.dealer
        for i in range(4):
            seat = (d + i) % 4
            for _ in range(14 if i == 0 else 13):
                t = self.wall[self.front]
                self.front += 1
                self._give(seat, t)
        # Replace bonus tiles in seat order starting from the dealer.
        for i in range(4):
            seat = (d + i) % 4
            p = self.players[seat]
            while p.pending_bonus > 0:
                p.pending_bonus -= 1
                self._give(seat, self._draw_back())
        self.players[d].draws = 1
        self.phase = Phase.self_turn(d)

    def _give(self, seat: int, t: int) -> None:
        p = self.players[seat]
        if is_bonus(t):
            p.bonus.append(t)
            p.pending_bonus += 1
        else:
            p.hand[t] += 1

    def _draw_back(self) -> int:
        self.back -= 1
        return self.wall[self.back]

    def _draw_for(self, seat: int, from_back: bool) -> Optional[int]:
        """Draw for `seat`: front tile, replacing any bonus tiles from the back.

        Returns the playing tile finally drawn, or None if the wall ran out.
        """
        reserve = self.cfg.reserve
        if self.live_remaining() <= reserve:
            return None
        if from_back:
            t = self._draw_back()
        else:
            t = self.wall[self.front]
            self.front += 1
        p = self.players[seat]
        while True:
            if self.live_remaining() <= reserve:
                self.last_draw = True
            if not is_bonus(t):
                break
            p.bonus.append(t)
            if self.live_remaining() <= reserve:
                return None
            t = self._draw_back()
        p.hand[t] += 1
        p.draws += 1
        return t

    def to_act(self) -> Optional[int]:
        """Seat that must decide next, if any."""
        ph = self.phase
        if ph.kind == SELF_TURN:
            return ph.seat
        if ph.kind == OVER:
            return None
        frm = ph.seat
        for i in range(1, 4):
            s = (frm + i) % 4
            if self.responses[s] is None:
                return s
        return None

    def _win_context(self, seat: int, win_tile: int, self_draw: bool) -> WinContext:
        ph = self.phase
        after_kong = ph.kind == SELF_TURN and ph.after_kong
        robbing = ph.kind == ROB_KONG
        p = self.players[seat]
        first_turn = p.draws == 1 and not self.any_claim and not p.discards
        return WinContext(
            seat_wind=self.seat_wind(seat),
            prevailing_wind=self.cfg.prevailing_wind,
            self_draw=self_draw,
            win_tile=win_tile,
            kong_replacement=self_draw and after_kong,
            robbing_kong=robbing,
            last_tile=self.last_draw,
            heavenly=self_draw and first_turn and seat == self.cfg.dealer,
            earthly=self_draw and first_turn and seat != self.cfg.dealer,
        )

    def win_score(self, seat: int) -> Optional[Score]:
        """Score `seat` winning now (tsumo in SelfTurn, ron in Claim/RobKong). None if not a legal win."""
        p = self.players[seat]
        ph = self.phase
        if ph.kind == SELF_TURN and ph.seat == seat and not ph.after_claim:
            # Dealer's first turn has no drawn tile; any tile can serve as the "winning" one.
            wt = ph.drawn if ph.drawn is not None else next(k for k in range(NUM_KINDS) if p.hand[k] > 0)
            hand, self_draw = p.hand, True
        elif ph.kind in (CLAIM, ROB_KONG) and ph.seat != seat:
            hand = list(p.hand)
            hand[ph.tile] += 1
            wt, self_draw = ph.tile, False
        else:
            return None
        if not scoring.is_complete_shape(hand, len(p.melds)):
            return None
        ctx = self._win_context(seat, wt, self_draw)
        best = scoring.score_hand(hand, p.melds, p.bonus, ctx)
        if best is None:
            return None
        # On the dealer's opening hand, try every tile as the winning tile.
        if self_draw and ph.drawn is None:
            for k in range(NUM_KINDS):
                if hand[k] > 0:
                    s = scoring.score_hand(hand, p.melds, p.bonus, ctx.copy(win_tile=k))
                    if s is not None and s.tai > best.tai:
                        best = s
        if best.tai < scoring.MIN_TAI:
            return None
        return best

    def ron_tai_if_discarded(self, seat: int, tile: int) -> int:
        """Tai `seat` would score by winning on `tile` if it were discarded right now.

        0 if that tile does not complete a hand worth at least the minimum. This is exact
        ground truth that uses hidden information: use it for labels and evaluation,
        never as an input to a policy.
        """
        p = self.players[seat]
        if p.hand[tile] >= 4 or p.hand_size() % 3 != 1:
            return 0
        h = list(p.hand)
        h[tile] += 1
        if not scoring.is_complete_shape(h, len(p.melds)):
            return 0
        ctx = WinContext(seat_wind=self.seat_wind(seat), prevailing_wind=self.cfg.prevailing_wind,
                         self_draw=False, win_tile=tile, last_tile=self.last_draw)
        s = scoring.score_hand(h, p.melds, p.bonus, ctx)
        return s.tai if s is not None and s.tai >= scoring.MIN_TAI else 0

    def legal_actions(self, seat: int) -> List[int]:
        acts: List[int] = []
        p = self.players[seat]
        ph = self.phase
        if ph.kind == SELF_TURN and ph.seat == seat:
            if self.win_score(seat) is not None:
                acts.append(TSUMO)
            hand = p.hand
            if not ph.after_claim and self.draws_left() > 0:
                pongs = {m.tile for m in p.melds if m.kind == MeldKind.PONG}
                for k in range(NUM_KINDS):
                    if hand[k] == 4 or (hand[k] >= 1 and k in pongs):
                        acts.append(KONG + k)
            for k in range(NUM_KINDS):
                if hand[k] > 0:
                    acts.append(DISCARD + k)
        elif ph.kind == CLAIM and ph.seat != seat and self.responses[seat] is None:
            t = ph.tile
            frm = ph.seat
            if self.win_score(seat) is not None:
                acts.append(RON)
            h = p.hand
            n = h[t]
            if n >= 2:
                acts.append(PONG)
            if n == 3 and self.draws_left() > 0:
                acts.append(EXPOSED_KONG)
            if seat == (frm + 1) % 4 and is_suited(t):
                r = rank(t)
                if r <= 6 and h[t + 1] > 0 and h[t + 2] > 0:
                    acts.append(CHOW + 0)
                if 1 <= r <= 7 and h[t - 1] > 0 and h[t + 1] > 0:
                    acts.append(CHOW + 1)
                if r >= 2 and h[t - 1] > 0 and h[t - 2] > 0:
                    acts.append(CHOW + 2)
            acts.append(PASS)
        elif ph.kind == ROB_KONG and ph.seat != seat and self.responses[seat] is None:
            if self.win_score(seat) is not None:
                acts.append(RON)
            acts.append(PASS)
        return acts

    def apply(self, seat: int, a: int) -> None:
        """Apply an action. Callers should pick from `legal_actions`."""
        ph = self.phase
        if ph.kind == SELF_TURN:
            self._apply_self(seat, a)
        elif ph.kind == CLAIM:
            self.responses[seat] = a
            self._skip_trivial_responders(ph.seat)
            if self.to_act() is None:
                self._resolve_claims(ph.seat, ph.tile)
        elif ph.kind == ROB_KONG:
            self.responses[seat] = a
            self._skip_trivial_responders(ph.seat)
            if self.to_act() is None:
                self._resolve_rob(ph.seat, ph.tile)
        else:
            raise RuntimeError("game is over")

    def _apply_self(self, seat: int, a: int) -> None:
        self.turns += 1
        if a == TSUMO:
            score = self.win_score(seat)
            if score is None:
                raise RuntimeError("tsumo not legal")
            self._finish(seat, None, score)
        elif a < 34:
            t = a
            tsumogiri = self.phase.drawn == t
            p = self.players[seat]
            p.hand[t] -= 1
            p.discards.append(Discard(t, False, tsumogiri, self.turns))
            self._open_window(Phase.claim(seat, t), seat)
        elif a < 68:
            t = a - KONG
            p = self.players[seat]
            if p.hand[t] == 4:
                p.hand[t] = 0
                p.melds.append(Meld(MeldKind.CONCEALED_KONG, t, None, None))
                self._replacement_draw(seat)
            else:
                p.hand[t] -= 1
                self._open_window(Phase.rob_kong(seat, t), seat)
        else:
            raise RuntimeError(f"action {action_debug(a)} not valid on own turn")

    def _replacement_draw(self, seat: int) -> None:
        t = self._draw_for(seat, True)
        if t is not None:
            self.phase = Phase.self_turn(seat, t, after_kong=True)
        else:
            self._finish(None, None, None)

    def _open_window(self, phase: Phase, actor: int) -> None:
        self.responses = [None] * 4
        self.responses[actor] = PASS
        self.phase = phase
        self._skip_trivial_responders(actor)
        if self.to_act() is None:
            if phase.kind == CLAIM:
                self._resolve_claims(phase.seat, phase.tile)
            else:
                self._resolve_rob(phase.seat, phase.tile)

    def reopen_window(self, seat: int) -> None:
        """In a claim or rob-kong window, forget the answers other seats already gave, so they
        decide again with the tiles they now hold (used when re-dealing hidden tiles for rollouts)."""
        ph = self.phase
        if ph.kind not in (CLAIM, ROB_KONG):
            return
        actor = ph.seat
        for s in range(4):
            if s != actor and s != seat:
                self.responses[s] = None
        self._skip_trivial_responders(actor)

    def _skip_trivial_responders(self, actor: int) -> None:
        """Auto-pass seats whose only option is Pass so callers only see real decisions."""
        for i in range(1, 4):
            s = (actor + i) % 4
            if self.responses[s] is None:
                acts = self.legal_actions(s)
                if len(acts) == 1 and acts[0] == PASS:
                    self.responses[s] = PASS

    def _resolve_claims(self, frm: int, tile: int) -> None:
        # Priority: win (nearest seat after the discarder), then pong/kong, then chow.
        order = [(frm + i) % 4 for i in range(1, 4)]
        w = next((s for s in order if self.responses[s] == RON), None)
        if w is not None:
            score = self.win_score(w)
            if score is None:
                raise RuntimeError("ron not legal")
            self.players[w].hand[tile] += 1
            self._mark_claimed(frm)
            self._finish(w, frm, score)
            return
        for s in order:
            r = self.responses[s]
            if r == PONG or r == EXPOSED_KONG:
                kong = r == EXPOSED_KONG
                p = self.players[s]
                p.hand[tile] -= 3 if kong else 2
                p.melds.append(Meld(MeldKind.EXPOSED_KONG if kong else MeldKind.PONG, tile, tile, frm))
                self._mark_claimed(frm)
                self.any_claim = True
                if kong:
                    self._replacement_draw(s)
                else:
                    self.phase = Phase.self_turn(s, None, after_claim=True)
                return
        nxt = (frm + 1) % 4
        r = self.responses[nxt]
        if r is not None and CHOW <= r <= CHOW + 2:
            low = tile - (r - CHOW)
            p = self.players[nxt]
            for t in range(low, low + 3):
                if t != tile:
                    p.hand[t] -= 1
            p.melds.append(Meld(MeldKind.CHOW, low, tile, frm))
            self._mark_claimed(frm)
            self.any_claim = True
            self.phase = Phase.self_turn(nxt, None, after_claim=True)
            return
        # Nobody claimed: next player draws.
        t = self._draw_for(nxt, False)
        if t is not None:
            self.phase = Phase.self_turn(nxt, t)
        else:
            self._finish(None, None, None)

    def _resolve_rob(self, konger: int, tile: int) -> None:
        order = [(konger + i) % 4 for i in range(1, 4)]
        w = next((s for s in order if self.responses[s] == RON), None)
        if w is not None:
            score = self.win_score(w)
            if score is None:
                raise RuntimeError("rob not legal")
            self.players[w].hand[tile] += 1
            # The robbed tile leaves the konger's pong as is.
            self._finish(w, konger, score)
            return
        p = self.players[konger]
        i = next(i for i, m in enumerate(p.melds) if m.kind == MeldKind.PONG and m.tile == tile)
        p.melds[i] = p.melds[i].replace(kind=MeldKind.ADDED_KONG)
        self.phase = Phase.self_turn(konger)
        self._replacement_draw(konger)

    def _mark_claimed(self, frm: int) -> None:
        d = self.players[frm].discards
        if d:
            last = d[-1]
            d[-1] = Discard(last.tile, True, last.tsumogiri, last.turn)

    def _finish(self, winner: Optional[int], discarder: Optional[int], score: Optional[Score]) -> None:
        if winner is not None and score is not None:
            deltas = scoring.payments(score.tai, winner, discarder)
        else:
            deltas = [0, 0, 0, 0]
        self.result = HandResult(winner, discarder, winner is not None and discarder is None, score, deltas,
                                 self.turns)
        self.phase = _OVER

    def unseen_for(self, seat: int) -> List[int]:
        """Copies of each kind `seat` cannot see (wall + other players' concealed hands)."""
        seen = list(self.players[seat].hand)
        for p in self.players:
            for m in p.melds:
                for t in m.tiles():
                    seen[t] += 1
            for d in p.discards:
                if not d.claimed:
                    seen[d.tile] += 1
        return [max(4 - x, 0) for x in seen]

    def tile_census(self) -> Tuple[int, int]:
        """(tiles drawn from the wall, tiles accounted for in hands/melds/discards).

        The two must always match; tests use this as a conservation check.
        """
        drawn = self.front + (len(self.wall) - self.back)
        held = 0
        for p in self.players:
            held += p.hand_size() + len(p.bonus)
            for m in p.melds:
                held += len(m.tiles())
            held += sum(1 for d in p.discards if not d.claimed)
        return drawn, held
