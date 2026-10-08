"""Hand-built test positions ("scenarios") with checkable expectations.

Used two ways:
 * as rule tests (the engine must offer / refuse a win), and
 * as a skill benchmark for any policy, heuristic or learned: does it fold
   against an obvious threat, read a flush, keep efficient shapes, protect tai?

File format (`scenarios/*.txt`): blocks separated by a line `---`.
Each block is `key: value` lines; `#` starts a comment. Seats are absolute
(0..3) and the dealer is East.

    name: fold-vs-bamboo-flush
    desc: Right player has two bamboo melds and has thrown no bamboo.
    tags: defence, flush-read
    hero: 0              # seat being tested (default 0)
    dealer: 0            # default 0
    prevailing: 0        # default 0
    draws_left: 20       # live draws remaining (default 30)
    seed: 1              # for filling unspecified hidden hands
    phase: turn          # turn | claim 5z from 3 | rob 4p from 2
    drawn: 9m            # optional, the tile hero just drew (turn phase)
    hand.0: 123m 456p ...   # concealed tiles (no bonus tiles here)
    bonus.0: f1 a2
    melds.1: pong:2s from 0, chow:678s
    discards.1: 1m 9p 4p 3z
    hand.1: 34s 11z      # optional hidden hand; random filler if omitted
    assert_ready: 1      # validation: these seats must be ready (one tile from winning)
    expect: safe         # see Expectation below; several expect lines allowed
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .game import (CHOW, EXPOSED_KONG, KONG, PASS, PONG, RON, ROB_KONG, SELF_TURN, TSUMO, Config, Discard,
                   Game, Phase, Player)
from .obs import View, action_name
from .rng import Rng
from .scoring import Meld, MeldKind
from .shanten import shanten, ukeire
from .tile import (FIRST_BONUS, NUM_KINDS, NUM_TILE_IDS, WALL_SIZE, is_bonus, is_suited, parse_tiles, suit,
                   tile_name)


@dataclass(frozen=True)
class Expectation:
    """kind is one of:
      safe            the chosen discard must not be a winning tile for any opponent (exact, uses hidden hands)
      win             must declare a win (tsumo or ron)
      pass            must pass (claim / rob phases)
      no_win_offered  the engine must not offer a win (rule check, independent of the policy)
      win_offered     the engine must offer a win (rule check)
      max_ukeire      chosen discard keeps the lowest shanten and the most useful tiles (public information only)
      any_of          chosen action must be one of `actions`
      none_of         chosen action must not be any of `actions`
    """
    kind: str
    actions: Tuple[int, ...] = ()

    def __str__(self):
        names = {"safe": "Safe", "win": "Win", "pass": "Pass", "no_win_offered": "NoWinOffered",
                 "win_offered": "WinOffered", "max_ukeire": "MaxUkeire", "any_of": "AnyOf", "none_of": "NoneOf"}
        if self.actions:
            from .game import action_debug
            return f"{names[self.kind]}([{', '.join(action_debug(a) for a in self.actions)}])"
        return names[self.kind]


class ScenarioError(ValueError):
    pass


def _parse_action(s: str) -> int:
    s = s.strip()
    verb, _, rest = s.partition(" ")

    def tile() -> int:
        try:
            v = parse_tiles(rest.strip())
        except ValueError:
            v = []
        if not v:
            raise ScenarioError(f"bad tile in '{s}'")
        return v[0]

    if verb == "discard":
        return tile()
    if verb == "kong":
        return KONG + tile()
    if verb == "chow":
        try:
            p = int(rest.strip())
            if not 0 <= p <= 255:
                raise ValueError
        except ValueError:
            raise ScenarioError(f"chow needs position 0-2: '{s}'")
        return CHOW + p
    simple = {"tsumo": TSUMO, "ron": RON, "pong": PONG, "exposed_kong": EXPOSED_KONG, "pass": PASS}
    if verb in simple:
        return simple[verb]
    raise ScenarioError(f"unknown action '{s}'")


def _parse_meld(s: str, owner: int) -> Meld:
    s = s.strip()
    if " from " in s:
        body, _, f = s.partition(" from ")
        body = body.strip()
        try:
            frm: Optional[int] = int(f.strip())
        except ValueError:
            raise ScenarioError(f"bad seat in '{s}'")
    else:
        body, frm = s, None
    if ":" not in body:
        raise ScenarioError(f"meld must be kind:tiles, got '{s}'")
    kind_s, _, tiles = body.partition(":")
    t = _tiles(tiles)
    if kind_s == "chow":
        v = sorted(t)
        if len(v) != 3 or not is_suited(v[0]) or v[1] != v[0] + 1 or v[2] != v[0] + 2 or suit(v[0]) != suit(v[2]):
            raise ScenarioError(f"bad chow '{s}'")
        kind, tile = MeldKind.CHOW, v[0]
    elif kind_s in ("pong", "kong", "ckong"):
        kind = {"pong": MeldKind.PONG, "kong": MeldKind.EXPOSED_KONG, "ckong": MeldKind.CONCEALED_KONG}[kind_s]
        tile = t[0]
    else:
        raise ScenarioError(f"unknown meld kind in '{s}'")
    frm = None if kind == MeldKind.CONCEALED_KONG else (frm if frm is not None else (owner + 3) % 4)
    return Meld(kind, tile, tile, frm)


def _tiles(s: str) -> List[int]:
    try:
        return parse_tiles(s)
    except ValueError as e:
        raise ScenarioError(str(e))


@dataclass
class Scenario:
    name: str
    desc: str
    tags: List[str]
    hero: int
    expectations: List[Expectation]
    assert_ready: List[int]
    fields: Dict[str, str] = field(default_factory=dict)

    def get(self, k: str) -> Optional[str]:
        return self.fields.get(k)

    def _num(self, k: str, default: int) -> int:
        v = self.get(k)
        if v is None:
            return default
        try:
            n = int(v)
            if n < 0:
                raise ValueError
            return n
        except ValueError:
            raise ScenarioError(f"{k}: '{v}' is not a number")

    def build(self) -> Game:
        """Build the position. Hidden hands not given are filled randomly from unused tiles."""
        dealer = self._num("dealer", 0)
        prevailing = self._num("prevailing", 0)
        draws_left = self._num("draws_left", 30)
        seed = self._num("seed", 1)
        players = [Player() for _ in range(4)]
        used = [0] * NUM_TILE_IDS

        def add_used(t: int) -> None:
            used[t] += 1
            if used[t] > (1 if is_bonus(t) else 4):
                raise ScenarioError(f"too many copies of {tile_name(t)}")

        turns = 0
        phase_spec = self.get("phase") or "turn"
        words = phase_spec.split()
        for s in range(4):
            p = players[s]
            m = self.get(f"melds.{s}")
            if m is not None:
                for part in m.split(","):
                    if not part.strip():
                        continue
                    meld = _parse_meld(part, s)
                    for t in meld.tiles():
                        add_used(t)
                    p.melds.append(meld)
            b = self.get(f"bonus.{s}")
            if b is not None:
                for t in _tiles(b):
                    if not is_bonus(t):
                        raise ScenarioError(f"bonus.{s} has non-bonus tile {tile_name(t)}")
                    add_used(t)
                    p.bonus.append(t)
            d = self.get(f"discards.{s}")
            if d is not None:
                for i, t in enumerate(_tiles(d)):
                    add_used(t)
                    p.discards.append(Discard(t, False, False, i * 4 + s))
                    turns += 1
            h = self.get(f"hand.{s}")
            if h is not None:
                for t in _tiles(h):
                    if is_bonus(t):
                        raise ScenarioError(f"put bonus tiles in bonus.{s}, not hand.{s}")
                    add_used(t)
                    p.hand[t] += 1
            p.draws = len(p.discards) + 1
        if words == ["turn"]:
            dr = self.get("drawn")
            drawn = _tiles(dr)[0] if dr is not None else None
            phase = Phase.self_turn(self.hero, drawn)
        elif len(words) == 4 and words[0] in ("claim", "rob") and words[2] == "from":
            tile = _tiles(words[1])[0]
            try:
                frm = int(words[3])
                if not 0 <= frm <= 255:
                    raise ValueError
            except ValueError:
                raise ScenarioError("bad seat in phase")
            add_used(tile)
            if words[0] == "claim":
                players[frm].discards.append(Discard(tile, False, False, turns))
                turns += 1
                phase = Phase.claim(frm, tile)
            else:
                # The robbed tile is the 4th copy being added to `from`'s pong.
                if not any(m.kind == MeldKind.PONG and m.tile == tile for m in players[frm].melds):
                    raise ScenarioError(f"rob phase needs seat {frm} to hold a pong of {tile_name(tile)}")
                phase = Phase.rob_kong(frm, tile)
        else:
            raise ScenarioError(f"bad phase '{phase_spec}'")
        decider = self.hero
        # Fill hidden hands that were not specified.
        pool: List[int] = []
        for k in range(NUM_KINDS):
            pool.extend([k] * max(4 - used[k], 0))
        for b in range(FIRST_BONUS, NUM_TILE_IDS):
            if used[b] == 0:
                pool.append(b)
        rng = Rng(seed)
        rng.shuffle(pool)
        # Bonus tiles stay in the wall; filler hands only take playing tiles.
        playing = [t for t in pool if not is_bonus(t)]
        bonus_left = [t for t in pool if is_bonus(t)]
        for s in range(4):
            p = players[s]
            want = 13 - 3 * len(p.melds) + (1 if phase.kind == SELF_TURN and phase.seat == s else 0)
            have = p.hand_size()
            if self.get(f"hand.{s}") is None:
                for _ in range(want):
                    if not playing:
                        raise ScenarioError("ran out of tiles while filling hands")
                    p.hand[playing.pop()] += 1
            elif have != want:
                raise ScenarioError(
                    f"hand.{s} has {have} tiles, expected {want} (13 - 3 per meld, +1 on own turn)")
        # Wall: everything already in play first ("drawn"), then the live tiles.
        wall: List[int] = []
        for p in players:
            for k in range(NUM_KINDS):
                wall.extend([k] * p.hand[k])
            for m in p.melds:
                wall.extend(m.tiles())
            wall.extend(p.bonus)
            wall.extend(d.tile for d in p.discards)
        if phase.kind == ROB_KONG:
            wall.append(phase.tile)  # the tile being added to the kong is in play but in nobody's hand
        front = len(wall)
        live = playing + bonus_left
        rng.shuffle(live)
        wall.extend(live)
        if len(wall) != WALL_SIZE:
            raise ScenarioError(f"tile accounting error: {len(wall)} tiles")
        if len(live) < draws_left:
            raise ScenarioError(f"only {len(live)} live tiles but draws_left = {draws_left}")
        cfg = Config(reserve=len(live) - draws_left, prevailing_wind=prevailing, dealer=dealer)
        return Game.from_parts(cfg, wall, front, players, phase, decider, turns)

    def validate(self) -> None:
        """Structural checks: position builds, ready seats really are ready, and each
        expectation is satisfiable and non-trivial. Raises ScenarioError."""
        g = self.build()
        if g.to_act() != self.hero:
            raise ScenarioError(f"hero {self.hero} is not the seat to act")
        for s in self.assert_ready:
            p = g.players[s]
            if shanten(p.hand, p.melds_needed()) != 0:
                raise ScenarioError(f"seat {s} is not ready")
            if all(g.ron_tai_if_discarded(s, k) == 0 for k in range(34)):
                raise ScenarioError(f"seat {s} is ready but cannot win any tile with 1+ tai")
        legal = g.legal_actions(self.hero)
        if self.get("trap") == "yes":
            # The purely efficient discard(s) must deal in, so passing needs real reading.
            rnk = efficiency_ranking(View(g, self.hero))
            top = rnk[0][1]
            for t, sc in rnk:
                if sc == top and deal_in_tai(g, self.hero, t) == 0:
                    raise ScenarioError(f"trap: most efficient discard {tile_name(t)} (score {sc}) is already safe")
        for e in self.expectations:
            if e.kind == "safe":
                discards = [a for a in legal if a < 34]
                safe = sum(1 for t in discards if deal_in_tai(g, self.hero, t) == 0)
                if safe == 0:
                    raise ScenarioError("expect safe, but no discard is safe")
                if safe == len(discards):
                    raise ScenarioError("expect safe, but every discard is safe (trivial)")
            elif e.kind in ("any_of", "none_of"):
                for a in e.actions:
                    if a not in legal:
                        raise ScenarioError(f"{action_name(a)} is not legal here")
            elif e.kind == "win":
                if TSUMO not in legal and RON not in legal:
                    raise ScenarioError("expect win, but no win is legal")

    def check(self, g: Game, a: int) -> Optional[str]:
        """Check a policy's chosen action. Returns None if passed, else the reason it failed."""
        legal = g.legal_actions(self.hero)
        if a not in legal:
            return f"illegal action {action_name(a)}"
        for e in self.expectations:
            k = e.kind
            if k == "safe":
                ok = True
                if a < 34:
                    tai = deal_in_tai(g, self.hero, a)
                    if tai > 0:
                        return f"{action_name(a)} deals in for {tai} tai"
            elif k == "win":
                ok = a in (TSUMO, RON)
            elif k == "pass":
                ok = a == PASS
            elif k == "no_win_offered":
                ok = TSUMO not in legal and RON not in legal
            elif k == "win_offered":
                ok = TSUMO in legal or RON in legal
            elif k == "max_ukeire":
                if a < 34:
                    best = efficiency_ranking(View(g, self.hero))
                    top = best[0][1]
                    ok = any(bt == a and score == top for bt, score in best)
                else:
                    ok = False
            elif k == "any_of":
                ok = a in e.actions
            else:
                ok = a not in e.actions
            if not ok:
                return f"{action_name(a)} fails {e}"
        return None


def deal_in_tai(g: Game, hero: int, t: int) -> int:
    """Exact: the most tai any opponent would win if `hero` discarded `t` now (0 = safe).

    Respects claim priority only loosely: any opponent able to win counts.
    """
    return max(g.ron_tai_if_discarded((hero + i) % 4, t) for i in range(1, 4))


def efficiency_ranking(v: View) -> List[Tuple[int, int]]:
    """Discards ranked by (lower shanten, more useful tiles), public information only.

    Returns (tile, combined score) best first; equal scores are equally good.
    """
    hand = v.hand()
    need = 4 - len(v.my_melds())
    unseen = v.unseen()
    out: List[Tuple[int, int]] = []
    h = list(hand)
    for k in range(34):
        if h[k] == 0:
            continue
        h[k] -= 1
        sh = shanten(h, need)
        uk, _ = ukeire(h, need, unseen)
        out.append((k, -sh * 1000 + uk))
        h[k] += 1
    out.sort(key=lambda x: -x[1])  # stable, like the original
    return out


def parse_scenarios(text: str) -> List[Scenario]:
    out: List[Scenario] = []
    for bi, block in enumerate(text.split("\n---")):
        fields: Dict[str, str] = {}
        expectations: List[Expectation] = []
        assert_ready: List[int] = []
        for line in block.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or line == "---":
                continue
            if ":" not in line:
                raise ScenarioError(f"block {bi}: expected 'key: value', got '{line}'")
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k == "expect":
                kind, _, rest = v.partition(" ")
                if kind in ("safe", "win", "pass", "no_win_offered", "win_offered", "max_ukeire"):
                    expectations.append(Expectation(kind))
                elif kind in ("any_of", "none_of"):
                    expectations.append(Expectation(kind, tuple(_parse_action(x) for x in rest.split(","))))
                else:
                    raise ScenarioError(f"unknown expectation '{v}'")
            elif k == "assert_ready":
                for s in v.split(","):
                    try:
                        assert_ready.append(int(s.strip()))
                    except ValueError:
                        raise ScenarioError(f"bad seat '{s}'")
            else:
                fields[k] = v
        if not fields and not expectations:
            continue
        if "name" not in fields:
            raise ScenarioError(f"block {bi} has no name")
        try:
            hero = int(fields["hero"]) if "hero" in fields else 0
        except ValueError:
            hero = 0
        tags = [x.strip() for x in fields["tags"].split(",")] if "tags" in fields else []
        out.append(Scenario(fields["name"], fields.get("desc", ""), tags, hero, expectations, assert_ready, fields))
    return out


def load_dir(directory: str) -> List[Tuple[str, Scenario]]:
    """Every scenario in the .txt files of `directory`, as (file name, scenario), files sorted."""
    files = sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".txt"))
    out: List[Tuple[str, Scenario]] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            text = fh.read()
        try:
            items = parse_scenarios(text)
        except ScenarioError as e:
            raise ScenarioError(f"{f}: {e}")
        for s in items:
            out.append((os.path.basename(f), s))
    return out
