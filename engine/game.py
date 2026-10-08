"""
One hand of Singapore mahjong, played step by step.

How to drive a hand (this is all a bot loop needs):

    game = Game(dealer=0, prevailing=0, seed=1)
    while not game.is_over():
        seat = game.to_act()                    # whose decision is it?
        options = game.legal_actions(seat)      # everything that seat may do now
        game.apply(seat, options[0])            # do one of them
    print(game.result)

There are three kinds of decision ("phases"):
  "turn"   It's your turn: win (self-draw), declare a kong, or discard.
  "claim"  Someone discarded: win on it, pong, kong, chow, or pass.
           Only players with a real choice are asked; the others pass automatically.
           After everyone has answered, the claim priority decides:
           win > pong/kong > chow (and only the next player may chow).
  "rob"    Someone added a 4th tile to their pong: anyone who can win on that tile
           may "rob the kong"; otherwise pass.

Rules follow RULES.md. Tiles are numbered as in engine/tiles.py.
"""

import random
from dataclasses import dataclass, field

from engine.scoring import MIN_TAI, Meld, Score, WinContext, payments, score_hand
from engine.tiles import (
    NUM_TILE_TYPES, WALL_SIZE, full_set, is_bonus, is_suited, number, tile_name,
)

RESERVE = 15          # the hand is a draw when this many live tiles are left


# --------------------------------------------------------------------------
# Small data types
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    """
    kind is one of:
      "discard"       tile = the tile thrown
      "kong"          tile = the tile (own turn: concealed kong, or adding to your own pong)
      "win"           self-draw on your turn, or winning on a discard / robbed kong
      "pong"          claim the discard as a pong
      "exposed_kong"  claim the discard as a kong (you hold the other 3)
      "chow"          tile = LOWEST tile of the run you make with the discard
      "pass"
    """
    kind: str
    tile: int = None

    def __str__(self):
        return self.kind if self.tile is None else f"{self.kind} {tile_name(self.tile)}"


PASS = Action("pass")
WIN = Action("win")


@dataclass
class Discard:
    tile: int
    claimed: bool = False        # later taken by someone (pong, chow, kong or win)
    just_drawn: bool = False     # the player threw the tile they had just drawn
    turn: int = 0                # turn counter when it was thrown (order across players)


@dataclass
class PlayerState:
    hand: list = field(default_factory=lambda: [0] * NUM_TILE_TYPES)   # concealed tile counts
    bonus: list = field(default_factory=list)                          # flowers / animals shown
    melds: list = field(default_factory=list)                          # Meld objects on the table
    discards: list = field(default_factory=list)                       # Discard objects, in order
    draws: int = 0                                                     # tiles drawn so far

    def hand_size(self):
        return sum(self.hand)

    def hand_tiles(self):
        return [t for t in range(NUM_TILE_TYPES) for _ in range(self.hand[t])]


@dataclass
class HandResult:
    winner: int = None           # seat, or None for a drawn hand
    discarder: int = None        # who threw the winning tile (None for self-draw or a draw)
    self_draw: bool = False
    score: Score = None
    deltas: list = field(default_factory=lambda: [0, 0, 0, 0])   # points won/lost per seat
    turns: int = 0


# --------------------------------------------------------------------------
# The hand
# --------------------------------------------------------------------------
class Game:
    def __init__(self, dealer=0, prevailing=0, seed=None, wall=None):
        """
        dealer:     seat (0-3) that deals and sits East
        prevailing: round wind, 0 = East ... 3 = North
        seed:       shuffle the wall with this seed (same seed = same hand)
        wall:       or give the exact wall order (148 tiles), e.g. for tests
        """
        if wall is None:
            wall = full_set()
            random.Random(seed).shuffle(wall)
        assert len(wall) == WALL_SIZE and sorted(wall) == sorted(full_set()), "not a full 148-tile wall"

        self.dealer = dealer
        self.prevailing = prevailing
        self.wall = list(wall)
        self.front = 0                  # next tile drawn normally
        self.back = WALL_SIZE           # replacement tiles are taken from the back (index back-1)
        self.players = [PlayerState() for _ in range(4)]
        self.turns = 0                  # counts own-turn actions (discards, kongs, self-draw wins)
        self.any_claim = False          # has anyone claimed a discard yet this hand?
        self.last_draw = False          # has the last live tile been drawn?
        self.result = None

        # Phase details (see the module docstring)
        self.phase = "turn"
        self.current = dealer           # "turn": whose turn it is
        self.drawn = None               # "turn": tile just drawn (None after a claim, or the dealer's first turn)
        self.after_kong = False         # "turn": the drawn tile is a kong replacement
        self.after_claim = False        # "turn": just ponged/chowed, so must discard
        self.offer_tile = None          # "claim"/"rob": the tile on offer
        self.offer_from = None          # "claim"/"rob": who threw / added it
        self.waiting = []               # "claim"/"rob": seats still to answer, in order
        self.answers = {}               # "claim"/"rob": seat -> Action

        self._deal()

    # ------------------------------------------------------------------ dealing and drawing
    def _deal(self):
        # Dealer takes 14 tiles, the others 13, from the front of the wall.
        pending_bonus = [0, 0, 0, 0]
        for i in range(4):
            seat = (self.dealer + i) % 4
            for _ in range(14 if i == 0 else 13):
                tile = self.wall[self.front]
                self.front += 1
                pending_bonus[seat] += self._give(seat, tile)
        # Bonus tiles are set aside and replaced from the back, dealer first.
        for i in range(4):
            seat = (self.dealer + i) % 4
            while pending_bonus[seat] > 0:
                pending_bonus[seat] -= 1
                self.back -= 1
                pending_bonus[seat] += self._give(seat, self.wall[self.back])
        self.players[self.dealer].draws = 1

    def _give(self, seat, tile):
        """Put a dealt tile in a hand. Returns 1 if it was a bonus tile (needs replacing)."""
        if is_bonus(tile):
            self.players[seat].bonus.append(tile)
            return 1
        self.players[seat].hand[tile] += 1
        return 0

    def live_tiles(self):
        """Tiles left in the wall (front and back together)."""
        return self.back - self.front

    def draws_left(self):
        """How many more tiles can be drawn before the hand is a draw."""
        return max(0, self.live_tiles() - RESERVE)

    def _draw(self, seat, from_back):
        """
        Draw a tile for `seat`. Bonus tiles are set aside and replaced from the back.
        Returns the playing tile drawn, or None if the wall ran out (the hand is a draw).
        """
        if self.live_tiles() <= RESERVE:
            return None
        if from_back:
            self.back -= 1
            tile = self.wall[self.back]
        else:
            tile = self.wall[self.front]
            self.front += 1
        while True:
            if self.live_tiles() <= RESERVE:
                self.last_draw = True
            if not is_bonus(tile):
                break
            self.players[seat].bonus.append(tile)
            if self.live_tiles() <= RESERVE:
                return None
            self.back -= 1
            tile = self.wall[self.back]
        self.players[seat].hand[tile] += 1
        self.players[seat].draws += 1
        return tile

    # ------------------------------------------------------------------ who acts
    def is_over(self):
        return self.phase == "over"

    def to_act(self):
        """The seat that must decide now, or None if the hand is over."""
        if self.phase == "turn":
            return self.current
        if self.phase in ("claim", "rob"):
            return self.waiting[0]
        return None

    def seat_wind(self, seat):
        """0 = East ... 3 = North. The dealer is always East."""
        return (seat - self.dealer) % 4

    # ------------------------------------------------------------------ winning
    def _win_context(self, seat, win_tile, self_draw):
        p = self.players[seat]
        first_turn = p.draws == 1 and not self.any_claim and len(p.discards) == 0
        return WinContext(
            seat_wind=self.seat_wind(seat),
            prevailing=self.prevailing,
            bonus=list(p.bonus),
            self_draw=self_draw,
            win_tile=win_tile,
            kong_replacement=self_draw and self.phase == "turn" and self.after_kong,
            robbing_kong=self.phase == "rob",
            last_tile=self.last_draw,
            heavenly=self_draw and first_turn and seat == self.dealer,
            earthly=self_draw and first_turn and seat != self.dealer,
        )

    def win_score(self, seat):
        """The Score if `seat` may win right now (at least MIN_TAI), else None."""
        p = self.players[seat]
        if self.phase == "turn":
            if seat != self.current or self.after_claim:
                return None
            counts = list(p.hand)
            if self.drawn is not None:
                win_tiles = [self.drawn]
            else:
                # Dealer's dealt hand: any tile in it can count as the winning tile.
                win_tiles = [t for t in range(NUM_TILE_TYPES) if counts[t] > 0]
            best = None
            for w in win_tiles:
                s = score_hand(counts, p.melds, self._win_context(seat, w, True))
                if s is not None and (best is None or (s.tai, s.raw_tai) > (best.tai, best.raw_tai)):
                    best = s
        elif self.phase in ("claim", "rob") and seat != self.offer_from:
            counts = list(p.hand)
            counts[self.offer_tile] += 1
            best = score_hand(counts, p.melds, self._win_context(seat, self.offer_tile, False))
        else:
            return None
        if best is None or best.tai < MIN_TAI:
            return None
        return best

    # ------------------------------------------------------------------ legal moves
    def legal_actions(self, seat):
        """Everything `seat` may do right now (empty if it's not their decision)."""
        p = self.players[seat]
        actions = []

        if self.phase == "turn" and seat == self.current:
            if self.win_score(seat):
                actions.append(WIN)
            # A kong needs a replacement tile, so it's only allowed while tiles remain.
            if not self.after_claim and self.draws_left() > 0:
                for t in range(NUM_TILE_TYPES):
                    four_in_hand = p.hand[t] == 4
                    adds_to_pong = p.hand[t] >= 1 and any(m.kind == "pong" and m.tile == t for m in p.melds)
                    if four_in_hand or adds_to_pong:
                        actions.append(Action("kong", t))
            for t in range(NUM_TILE_TYPES):
                if p.hand[t] > 0:
                    actions.append(Action("discard", t))

        elif self.phase == "claim" and seat != self.offer_from and seat not in self.answers:
            t = self.offer_tile
            if self.win_score(seat):
                actions.append(WIN)
            if p.hand[t] >= 2:
                actions.append(Action("pong"))
            if p.hand[t] == 3 and self.draws_left() > 0:
                actions.append(Action("exposed_kong"))
            if seat == (self.offer_from + 1) % 4 and is_suited(t):   # only the next player may chow
                for low in (t - 2, t - 1, t):
                    run = [low, low + 1, low + 2]
                    if number(low) <= 7 and low >= 0 and all(x // 9 == t // 9 for x in run):
                        others = [x for x in run if x != t]
                        if all(p.hand[x] > 0 for x in others):
                            actions.append(Action("chow", low))
            actions.append(PASS)

        elif self.phase == "rob" and seat != self.offer_from and seat not in self.answers:
            if self.win_score(seat):
                actions.append(WIN)
            actions.append(PASS)

        return actions

    # ------------------------------------------------------------------ making moves
    def apply(self, seat, action):
        """Carry out `action` for `seat`. Raises ValueError if it isn't legal."""
        if self.is_over():
            raise ValueError("the hand is over")
        if seat != self.to_act() or action not in self.legal_actions(seat):
            raise ValueError(f"seat {seat} can't {action} now")

        if self.phase == "turn":
            self._apply_turn(seat, action)
        else:
            self.answers[seat] = action
            self.waiting.pop(0)
            if not self.waiting:
                if self.phase == "claim":
                    self._resolve_claims()
                else:
                    self._resolve_rob()

    def _apply_turn(self, seat, action):
        p = self.players[seat]
        self.turns += 1
        if action.kind == "win":
            self._finish(seat, None, self.win_score(seat))
        elif action.kind == "discard":
            t = action.tile
            p.hand[t] -= 1
            p.discards.append(Discard(t, just_drawn=(t == self.drawn), turn=self.turns))
            self._open_window("claim", t, seat)
        elif action.kind == "kong":
            t = action.tile
            if p.hand[t] == 4:                                    # concealed kong
                p.hand[t] = 0
                p.melds.append(Meld("concealed_kong", t))
                self._replacement_draw(seat)
            else:                                                 # adding to own pong: can be robbed
                p.hand[t] -= 1
                self._open_window("rob", t, seat)

    def _open_window(self, phase, tile, from_seat):
        """Offer a tile to the other players. Only those with a real choice are asked."""
        self.phase = phase
        self.offer_tile = tile
        self.offer_from = from_seat
        self.answers = {}
        self.waiting = []
        for i in (1, 2, 3):
            s = (from_seat + i) % 4
            if self.legal_actions(s) != [PASS]:
                self.waiting.append(s)
        if not self.waiting:
            if phase == "claim":
                self._resolve_claims()
            else:
                self._resolve_rob()

    def _resolve_claims(self):
        t, frm = self.offer_tile, self.offer_from
        order = [(frm + i) % 4 for i in (1, 2, 3)]           # nearest after the discarder first
        # 1. A win beats everything.
        for s in order:
            if self.answers.get(s) == WIN:
                score = self.win_score(s)
                self.players[s].hand[t] += 1
                self.players[frm].discards[-1].claimed = True
                self._finish(s, frm, score)
                return
        # 2. Then pong or kong.
        for s in order:
            a = self.answers.get(s)
            if a is not None and a.kind in ("pong", "exposed_kong"):
                p = self.players[s]
                kong = a.kind == "exposed_kong"
                p.hand[t] -= 3 if kong else 2
                p.melds.append(Meld("exposed_kong" if kong else "pong", t, from_seat=frm))
                self.players[frm].discards[-1].claimed = True
                self.any_claim = True
                if kong:
                    self._replacement_draw(s)
                else:
                    self._start_turn(s, drawn=None, after_claim=True)
                return
        # 3. Then chow, by the next player only.
        nxt = (frm + 1) % 4
        a = self.answers.get(nxt)
        if a is not None and a.kind == "chow":
            p = self.players[nxt]
            for x in (a.tile, a.tile + 1, a.tile + 2):
                if x != t:
                    p.hand[x] -= 1
            p.melds.append(Meld("chow", a.tile, from_seat=frm))
            self.players[frm].discards[-1].claimed = True
            self.any_claim = True
            self._start_turn(nxt, drawn=None, after_claim=True)
            return
        # 4. Nobody claimed: the next player draws.
        drawn = self._draw(nxt, from_back=False)
        if drawn is None:
            self._finish(None, None, None)
        else:
            self._start_turn(nxt, drawn=drawn)

    def _resolve_rob(self):
        t, konger = self.offer_tile, self.offer_from
        for i in (1, 2, 3):
            s = (konger + i) % 4
            if self.answers.get(s) == WIN:
                score = self.win_score(s)
                self.players[s].hand[t] += 1
                self._finish(s, konger, score)               # the konger pays, like a discarder
                return
        # Nobody robbed: the pong becomes a kong and the konger draws a replacement.
        for m in self.players[konger].melds:
            if m.kind == "pong" and m.tile == t:
                m.kind = "added_kong"
                break
        self._replacement_draw(konger)

    def _replacement_draw(self, seat):
        drawn = self._draw(seat, from_back=True)
        if drawn is None:
            self._finish(None, None, None)
        else:
            self._start_turn(seat, drawn=drawn, after_kong=True)

    def _start_turn(self, seat, drawn, after_kong=False, after_claim=False):
        self.phase = "turn"
        self.current = seat
        self.drawn = drawn
        self.after_kong = after_kong
        self.after_claim = after_claim
        self.offer_tile = self.offer_from = None
        self.waiting, self.answers = [], {}

    def _finish(self, winner, discarder, score):
        deltas = payments(score.tai, winner, discarder) if winner is not None else [0, 0, 0, 0]
        self.result = HandResult(winner=winner, discarder=discarder,
                                 self_draw=winner is not None and discarder is None,
                                 score=score, deltas=deltas, turns=self.turns)
        self.phase = "over"

    # ------------------------------------------------------------------ checks
    def any_kong(self):
        """Does anyone hold a kong? (Decides whether the dealer stays after a drawn hand.)"""
        return any(m.is_kong() for p in self.players for m in p.melds)

    def tile_census(self):
        """
        Count every tile: in the wall, in hands, in melds, shown as bonus, or in discard piles.
        Returns a dict tile -> count. In a correct game every playing tile appears 4 times and
        every bonus tile once. (A tile being added to a pong is "in the air" during a rob.)
        """
        seen = {}

        def add(t):
            seen[t] = seen.get(t, 0) + 1

        for t in self.wall[self.front:self.back]:
            add(t)
        for p in self.players:
            for t in p.hand_tiles() + p.bonus:
                add(t)
            for m in p.melds:
                for t in m.tiles():
                    add(t)
            for d in p.discards:
                if not d.claimed:
                    add(d.tile)
        if self.phase == "rob":
            add(self.offer_tile)
        return seen


# --------------------------------------------------------------------------
# Dealer rule between hands (RULES.md "A full game")
# --------------------------------------------------------------------------
@dataclass
class TableState:
    """
    Who deals the next hand, and in which round.
      - The dealer stays (a "repeat") after winning, or after a drawn hand where nobody holds a kong.
      - Otherwise the deal passes to the next seat.
      - After the round's 4th dealer passes the deal, the round wind moves on (E -> S -> W -> N).
      - After the North round, a new game starts from the East round.
    """
    dealer: int = 0
    prevailing: int = 0
    dealer_number: int = 1        # 1-4: the round's first to fourth dealer
    repeat: int = 0               # 0 = this dealer's first hand, 1 = first repeat ...
    game_number: int = 1

    def next_hand(self, result, any_kong):
        dealer_stays = (result.winner == self.dealer) or (result.winner is None and not any_kong)
        if dealer_stays:
            self.repeat += 1
            return
        self.repeat = 0
        self.dealer = (self.dealer + 1) % 4
        self.dealer_number += 1
        if self.dealer_number > 4:
            self.dealer_number = 1
            self.prevailing += 1
            if self.prevailing > 3:
                self.prevailing = 0
                self.game_number += 1
