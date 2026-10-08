"""Fixed action space and observation encoding shared by bots, the data logger and
the RL environment. Everything here only uses information the acting player can
legally see, except `encode_oracle` and `encode_labels`, which are stored
separately as training targets.
"""
from __future__ import annotations

from typing import List, Optional

from .game import CLAIM, ROB_KONG, SELF_TURN, Discard, Game, Phase, action_debug
from .scoring import Meld, MeldKind
from .shanten import shanten
from .tile import FIRST_BONUS, NUM_BONUS, tile_name

# ------------------------------------------------------------------ actions

A_DISCARD = 0       # + tile kind (34)
A_KONG = 34         # + tile kind (34)
A_TSUMO = 68
A_RON = 69
A_PONG = 70
A_EXPOSED_KONG = 71
A_CHOW = 72         # + position 0..=2
A_PASS = 75
N_ACTIONS = 76


def action_index(a: int) -> int:
    """Actions are already their index; kept so callers read like the action-space docs."""
    return a


def action_from_index(i: int) -> Optional[int]:
    return i if 0 <= i < N_ACTIONS else None


def action_name(a: int) -> str:
    if a < 34:
        return f"discard {tile_name(a)}"
    if a < 68:
        return f"kong {tile_name(a - 34)}"
    if 72 <= a <= 74:
        return f"chow(pos {a - 72})"
    return action_debug(a).lower()


def legal_mask(g: Game, seat: int) -> bytearray:
    m = bytearray(N_ACTIONS)
    for a in g.legal_actions(seat):
        m[a] = 1
    return m


# ------------------------------------------------------------- observation

# Bump when the observation layout changes; written to every dataset's format.txt.
OBS_VERSION = 2

MAX_DISCARD_SEQ = 32

OFF_HAND = 0
OFF_MELD_COUNTS = OFF_HAND + 34                         # 4 x 34
OFF_DISCARD_COUNTS = OFF_MELD_COUNTS + 4 * 34            # 4 x 34
OFF_DISCARD_SEQ = OFF_DISCARD_COUNTS + 4 * 34            # 4 x 32 tile codes
OFF_DISCARD_TURN = OFF_DISCARD_SEQ + 4 * MAX_DISCARD_SEQ  # 4 x 32 turn indices
OFF_BONUS = OFF_DISCARD_TURN + 4 * MAX_DISCARD_SEQ       # 4 x 12
OFF_MELDS = OFF_BONUS + 4 * NUM_BONUS                    # 4 x 4 x 3
MELD_BYTES = 3
OFF_META = OFF_MELDS + 4 * 4 * MELD_BYTES
META_LEN = 14
OBS_LEN = OFF_META + META_LEN

ORACLE_LEN = 3 * 34

EMPTY = 255
CLAIMED_FLAG = 64     # added to a tile id in the discard sequence when that discard was claimed
TSUMOGIRI_FLAG = 128  # added when the player threw the tile they just drew


def rel(me: int, other: int) -> int:
    """Relative seat: 0 = self, 1 = next (right, plays after you), 2 = opposite, 3 = previous (left)."""
    return (other + 4 - me) % 4


def abs_seat(me: int, r: int) -> int:
    return (me + r) % 4


_MELD_CODE = {MeldKind.CHOW: 1, MeldKind.PONG: 2, MeldKind.EXPOSED_KONG: 3, MeldKind.ADDED_KONG: 4,
              MeldKind.CONCEALED_KONG: 5}


def encode_obs(g: Game, seat: int) -> bytearray:
    """Encode what `seat` can see into a flat byte vector of length OBS_LEN (version 2).

    All seats are relative: 0 = self, 1 = next, 2 = opposite, 3 = previous.

    | offset             | size      | content |
    |--------------------|-----------|---------|
    | OFF_HAND           | 34        | own concealed tile counts |
    | OFF_MELD_COUNTS    | 4 x 34    | tiles in each player's melds |
    | OFF_DISCARD_COUNTS | 4 x 34    | tiles each player has discarded (claimed ones included) |
    | OFF_DISCARD_SEQ    | 4 x 32    | discards in order: tile id, +64 if claimed, +128 if tsumogiri, 255 empty |
    | OFF_DISCARD_TURN   | 4 x 32    | global turn of each discard (capped 254, 255 empty) |
    | OFF_BONUS          | 4 x 12    | bonus tiles each player has shown (0/1) |
    | OFF_MELDS          | 4 x 4 x 3 | melds: kind (1 chow, 2 pong, 3 exposed kong, 4 added kong, 5 concealed kong), tile, source seat (rel, 255 none) |
    | OFF_META           | 14        | seat wind, prevailing wind, dealer (rel), draws left, turn (cap 255), phase (0 own turn, 1 claim, 2 rob kong), target tile, target seat (rel), after kong, after claim, last draw, concealed tile counts of seats 1..3 |
    """
    out = bytearray(OBS_LEN)
    me = g.players[seat]
    out[OFF_HAND:OFF_HAND + 34] = bytes(me.hand)
    for ab, p in enumerate(g.players):
        r = rel(seat, ab)
        for m in p.melds:
            # Concealed kongs are declared and shown, so their tile is public.
            for t in m.tiles():
                out[OFF_MELD_COUNTS + r * 34 + t] += 1
        for i, d in enumerate(p.discards):
            out[OFF_DISCARD_COUNTS + r * 34 + d.tile] += 1
            if i < MAX_DISCARD_SEQ:
                code = d.tile
                if d.claimed:
                    code += CLAIMED_FLAG
                if d.tsumogiri:
                    code += TSUMOGIRI_FLAG
                out[OFF_DISCARD_SEQ + r * MAX_DISCARD_SEQ + i] = code
                out[OFF_DISCARD_TURN + r * MAX_DISCARD_SEQ + i] = min(d.turn, 254)
        for i in range(min(len(p.discards), MAX_DISCARD_SEQ), MAX_DISCARD_SEQ):
            out[OFF_DISCARD_SEQ + r * MAX_DISCARD_SEQ + i] = EMPTY
            out[OFF_DISCARD_TURN + r * MAX_DISCARD_SEQ + i] = EMPTY
        for b in p.bonus:
            out[OFF_BONUS + r * NUM_BONUS + (b - FIRST_BONUS)] = 1
        for i in range(4):
            o = OFF_MELDS + (r * 4 + i) * MELD_BYTES
            if i < len(p.melds):
                m = p.melds[i]
                out[o] = _MELD_CODE[m.kind]
                out[o + 1] = m.tile
                out[o + 2] = EMPTY if m.from_ is None else rel(seat, m.from_)
            else:
                out[o + 1] = EMPTY
                out[o + 2] = EMPTY
    mo = OFF_META
    out[mo] = g.seat_wind(seat)
    out[mo + 1] = g.cfg.prevailing_wind
    out[mo + 2] = rel(seat, g.cfg.dealer)
    out[mo + 3] = min(g.draws_left(), 255)
    out[mo + 4] = min(g.turns, 255)
    ph = g.phase
    if ph.kind == SELF_TURN:
        phase, tile, frm, ak, ac = 0, EMPTY if ph.drawn is None else ph.drawn, EMPTY, ph.after_kong, ph.after_claim
    elif ph.kind == CLAIM:
        phase, tile, frm, ak, ac = 1, ph.tile, rel(seat, ph.seat), False, False
    elif ph.kind == ROB_KONG:
        phase, tile, frm, ak, ac = 2, ph.tile, rel(seat, ph.seat), False, False
    else:
        phase, tile, frm, ak, ac = 3, EMPTY, EMPTY, False, False
    out[mo + 5] = phase
    out[mo + 6] = tile
    out[mo + 7] = frm
    out[mo + 8] = int(ak)
    out[mo + 9] = int(ac)
    out[mo + 10] = int(g.last_draw)
    for r in range(1, 4):
        out[mo + 10 + r] = g.players[abs_seat(seat, r)].hand_size()
    return out


# ------------------------------------------------------------------ labels

# Per-opponent winning tiles: [3][34], value = tai that opponent would score by
# winning on that tile if it were discarded now (0 = cannot win on it).
WAITS_LEN = 3 * 34
# Shanten of every seat, relative order (self first). -1 = complete, 0 = ready.
SHANTEN_LEN = 4


def encode_labels(g: Game, seat: int):
    """Exact labels for the opponent-reading heads: (waits bytearray[102], shanten list[4]).

    Uses hidden information, so it is a training target and an evaluation tool, never a policy input.
    """
    waits = bytearray(WAITS_LEN)
    sh_out = [0] * SHANTEN_LEN
    for r in range(4):
        s = abs_seat(seat, r)
        p = g.players[s]
        sh = shanten(p.hand, p.melds_needed())
        sh_out[r] = max(-1, min(8, sh))
        # A player with 3n+1 tiles at shanten 0 is ready; only then can a discard complete them.
        if r > 0 and sh == 0 and p.hand_size() % 3 == 1:
            for k in range(34):
                waits[(r - 1) * 34 + k] = g.ron_tai_if_discarded(s, k)
    return waits, sh_out


def encode_oracle(g: Game, seat: int) -> bytearray:
    """Hidden information: concealed hands of seats 1..3 relative to `seat`.

    Never feed this to a policy at play time; it is a training target.
    """
    out = bytearray(ORACLE_LEN)
    for r in range(1, 4):
        out[(r - 1) * 34:r * 34] = bytes(g.players[abs_seat(seat, r)].hand)
    return out


# --------------------------------------------------------------------- view

class View:
    """The public view one seat has of the table. Bots only receive this, so they
    cannot peek at other hands or the wall."""

    __slots__ = ("_g", "seat")

    def __init__(self, g: Game, seat: int):
        self._g = g
        self.seat = seat

    def hand(self) -> List[int]:
        return self._g.players[self.seat].hand

    def my_melds(self) -> List[Meld]:
        return self._g.players[self.seat].melds

    def melds_of(self, seat: int) -> List[Meld]:
        return self._g.players[seat].melds

    def my_bonus(self) -> List[int]:
        return self._g.players[self.seat].bonus

    def discards_of(self, seat: int) -> List[Discard]:
        return self._g.players[seat].discards

    def hand_size_of(self, seat: int) -> int:
        return self._g.players[seat].hand_size()

    def unseen(self) -> List[int]:
        return self._g.unseen_for(self.seat)

    def seat_wind(self) -> int:
        return self._g.seat_wind(self.seat)

    def seat_wind_of(self, seat: int) -> int:
        return self._g.seat_wind(seat)

    def prevailing_wind(self) -> int:
        return self._g.cfg.prevailing_wind

    def draws_left(self) -> int:
        return self._g.draws_left()

    def phase(self) -> Phase:
        return self._g.phase

    def legal_actions(self) -> List[int]:
        return self._g.legal_actions(self.seat)

    def win_tai(self) -> Optional[int]:
        """Tai this seat would score by winning right now, if winning is legal."""
        s = self._g.win_score(self.seat)
        return None if s is None else s.tai
