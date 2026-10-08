"""da-san-yuan: Singapore 4-player mahjong engine, heuristic bots and self-play data, in pure Python.

Quick start::

    import dasanyuan as dsy
    env = dsy.Env(seed=1)
    while not env.is_over():
        seat = env.to_act()
        obs, mask = env.observe(seat)          # numpy uint8 arrays
        action = env.bot_action(seat, "balanced")
        env.step(seat, action)
    print(env.result())

Load self-play data::

    data = dsy.load_shards("data/selfplay")   # dict of concatenated numpy arrays

The engine modules (tile, game, scoring, shanten, bots, obs, scenario, selfplay,
evaluate, trace) can be used directly; `python -m dasanyuan --help` is the CLI.
"""
from __future__ import annotations

import glob
import gzip
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import obs as _obs
from . import scenario as _scenario
from . import selfplay as _selfplay
from .bots import HeuristicBot, Style
from .game import Config, Game
from .obs import N_ACTIONS, OBS_LEN, ORACLE_LEN, View
from .tile import hand_string

# Action index layout (see obs.py)
A_DISCARD = 0      # + tile kind 0..33
A_KONG = 34        # + tile kind 0..33
A_TSUMO = 68
A_RON = 69
A_PONG = 70
A_EXPOSED_KONG = 71
A_CHOW = 72        # + 0/1/2: claimed tile is lowest/middle/highest
A_PASS = 75

META_COLUMNS = ["hand_id", "seat", "bot_id", "points", "won", "dealt_in"]

_SUITS = "mps"


def obs_len() -> int:
    return OBS_LEN


def n_actions() -> int:
    return N_ACTIONS


def oracle_len() -> int:
    return ORACLE_LEN


def tile_name(k: int) -> str:
    """0..33 -> '1m'..'9m', '1p'.., '1s'.., '1z'..'7z' (E S W N White Green Red)."""
    if k < 27:
        return f"{k % 9 + 1}{_SUITS[k // 9]}"
    return f"{k - 26}z"


def action_name(i: int) -> str:
    if i < 34:
        return f"discard {tile_name(i)}"
    if i < 68:
        return f"kong {tile_name(i - 34)}"
    return {68: "tsumo", 69: "ron", 70: "pong", 71: "exposed kong",
            72: "chow (low)", 73: "chow (mid)", 74: "chow (high)", 75: "pass"}[i]


def _style(s: str) -> Style:
    st = Style.by_name(s) or Style.from_csv(s)
    if st is None:
        raise ValueError(f"bad style {s}")
    return st


class Env:
    """One hand as a turn-based environment. Seats are 0..3; dealer sits East."""

    def __init__(self, seed: int = 0, dealer: int = 0, prevailing: int = 0):
        self.reset(seed, dealer, prevailing)

    @classmethod
    def from_game(cls, game: Game, seed: int = 0) -> "Env":
        e = cls.__new__(cls)
        e.game = game
        e.seed = seed
        e._bots = {}
        return e

    def reset(self, seed: int, dealer: int = 0, prevailing: int = 0) -> None:
        """Start a new hand."""
        self.game = Game(Config(dealer=dealer, prevailing_wind=prevailing), seed)
        self.seed = seed
        self._bots: Dict[Tuple[int, str], HeuristicBot] = {}

    def to_act(self) -> int:
        """Seat that must act next, or -1 if the hand is over."""
        s = self.game.to_act()
        return -1 if s is None else s

    def is_over(self) -> bool:
        return self.game.is_over()

    def observe(self, seat: int):
        """(obs[OBS_LEN] uint8, legal_mask[N_ACTIONS] uint8) for `seat`."""
        o = np.frombuffer(bytes(_obs.encode_obs(self.game, seat)), dtype=np.uint8)
        mask = np.frombuffer(bytes(_obs.legal_mask(self.game, seat)), dtype=np.uint8)
        return o, mask

    def oracle(self, seat: int) -> np.ndarray:
        """Hidden hands of the other three seats [3, 34]. Training targets only."""
        return np.frombuffer(bytes(_obs.encode_oracle(self.game, seat)), dtype=np.uint8).reshape(3, 34)

    def labels(self, seat: int):
        """Exact training targets (uses hidden information, never a policy input).

        waits [3, 34]: tai each opponent (relative seats 1..3) would win with if that
        tile were discarded now; 0 means the tile cannot deal in to them.
        shanten [4]: tiles-from-ready of every seat, self first (0 = ready).
        """
        w, sh = _obs.encode_labels(self.game, seat)
        return np.frombuffer(bytes(w), dtype=np.uint8).reshape(3, 34), np.array(sh, dtype=np.int8)

    def step(self, seat: int, action: int) -> None:
        """Apply an action index (see the A_* constants)."""
        a = _obs.action_from_index(int(action))
        if a is None:
            raise ValueError("bad action index")
        if self.game.to_act() != seat:
            raise ValueError(f"seat {seat} is not to act")
        if a not in self.game.legal_actions(seat):
            raise ValueError(f"illegal action {_obs.action_name(a)}")
        self.game.apply(seat, a)

    def bot_action(self, seat: int, style: str = "balanced") -> int:
        """Action index a heuristic bot of `style` would choose for `seat` now."""
        key = (seat, style)
        bot = self._bots.get(key)
        if bot is None:
            st = Style.by_name(style)
            if st is None:
                raise ValueError(f"unknown style {style}")
            bot = self._bots[key] = HeuristicBot(st, self.seed ^ (seat + 1) * 7919)
        return _obs.action_index(bot.act(View(self.game, seat)))

    def deltas(self) -> List[int]:
        """Points change for each seat (all zero until the hand ends or on a draw)."""
        r = self.game.result
        return list(r.deltas) if r is not None else [0, 0, 0, 0]

    def result(self) -> dict:
        """Result dict: winner, discarder, self_draw, tai, patterns, deltas."""
        r = self.game.result
        if r is None:
            return {}
        d = {"winner": r.winner, "discarder": r.discarder, "self_draw": r.self_draw, "deltas": list(r.deltas)}
        if r.score is not None:
            d["tai"] = r.score.tai
            d["patterns"] = [(p.label(), t) for p, t in r.score.items]
        return d

    def hand_str(self, seat: int) -> str:
        """Readable hand for debugging, e.g. "123m456p789s11z f1"."""
        p = self.game.players[seat]
        tiles = [k for k in range(34) for _ in range(p.hand[k])] + list(p.bonus)
        return hand_string(tiles)

    def draws_left(self) -> int:
        return self.game.draws_left()


class Scenarios:
    """Scenario bank for evaluating any policy.

        bank = dsy.Scenarios("scenarios")
        for i in range(len(bank)):
            env = bank.env(i)
            seat = env.to_act()
            obs, mask = env.observe(seat)
            ok, msg = bank.check(i, my_policy(obs, mask))
    """

    def __init__(self, directory: str = "scenarios"):
        try:
            self._items = _scenario.load_dir(directory)
            for f, sc in self._items:
                try:
                    sc.validate()
                except ValueError as e:
                    raise ValueError(f"{f}:{sc.name}: {e}")
        except _scenario.ScenarioError as e:
            raise ValueError(str(e))

    def __len__(self) -> int:
        return len(self._items)

    def _get(self, i: int):
        if not 0 <= i < len(self._items):
            raise ValueError("index out of range")
        return self._items[i]

    def info(self, i: int) -> dict:
        f, s = self._get(i)
        return {"file": f, "name": s.name, "desc": s.desc, "tags": list(s.tags), "hero": s.hero}

    def env(self, i: int) -> Env:
        """A fresh Env positioned at scenario `i`; the hero is the seat to act."""
        _, s = self._get(i)
        return Env.from_game(s.build(), seed=i)

    def check(self, i: int, action: int) -> Tuple[bool, str]:
        """Check an action index for scenario `i`: (passed, message)."""
        _, s = self._get(i)
        a = _obs.action_from_index(int(action))
        if a is None:
            raise ValueError("bad action index")
        err = s.check(s.build(), a)
        return (True, "") if err is None else (False, err)


_DEFAULT_STYLES = ("fast", "high_tai", "defensive", "balanced")


def run_selfplay(out_dir: str, hands: int = 10000, seed: int = 1, styles: Sequence[str] = _DEFAULT_STYLES,
                 random_lineup: bool = True, shard_size: int = 2000, compress: bool = True,
                 games: int = 0) -> Tuple[int, int]:
    """Generate self-play shards (shards run in parallel processes). Returns (hands, decisions)."""
    cfg = _selfplay.SelfPlayConfig(hands=hands, seed=seed, styles=[_style(s) for s in styles],
                                   random_lineup=random_lineup, shard_size=shard_size, out_dir=str(out_dir),
                                   compress=compress, games=games)
    s = _selfplay.run_selfplay(cfg)
    return s.hands, s.decisions


def tournament(styles: Sequence[str], walls: int = 2000, seed: int = 7) -> List[dict]:
    """Duplicate tournament between 4 styles (names or CSV lines). Returns a list of dicts."""
    if len(styles) != 4:
        raise ValueError("need exactly 4 styles")
    stats = _selfplay.duplicate_tournament([_style(s) for s in styles], walls, seed)
    return [{"name": s.name, "hands": s.hands, "points_per_hand": s.mean(), "stderr": s.stderr(),
             "wins": s.wins, "self_draws": s.self_draws, "deal_ins": s.deal_ins} for s in stats]


def load_npy(path_without_ext: str, mmap: bool = True) -> np.ndarray:
    """Load `<path>.npy`, or `<path>.npy.gz` if only the compressed file exists."""
    if os.path.exists(path_without_ext + ".npy"):
        return np.load(path_without_ext + ".npy", mmap_mode="r" if mmap else None)
    with gzip.open(path_without_ext + ".npy.gz", "rb") as f:
        return np.load(f)


def load_shards(root: str, limit: Optional[int] = None, mmap: bool = True) -> Dict[str, np.ndarray]:
    """Concatenate self-play shards under `root` into one dict of arrays.

    Keys: obs [N, OBS_LEN], mask [N, 76], action [N], oracle [N, 102],
    waits [N, 3, 34] (tai each opponent would win with on that tile, 0 = safe),
    shanten [N, 4] (self first; 0 = ready), meta [N, 6].
    Handles both plain .npy and compressed .npy.gz shards. Plain shards are
    memory-mapped when mmap=True. For datasets larger than RAM, iterate over
    shards with `limit` or load shard directories one at a time.
    """
    shards = sorted(glob.glob(os.path.join(root, "shard_*")))
    if limit is not None:
        shards = shards[:limit]
    if not shards:
        raise FileNotFoundError(f"no shard_* directories in {root}")
    out: Dict[str, List[np.ndarray]] = {k: [] for k in ("obs", "mask", "action", "oracle", "waits", "shanten", "meta")}
    for s in shards:
        for k in out:
            out[k].append(load_npy(os.path.join(s, k), mmap))
    return {k: np.concatenate(v) for k, v in out.items()}


__all__ = [
    "Env", "Scenarios", "load_shards", "load_npy", "run_selfplay", "tournament", "tile_name", "action_name",
    "OBS_LEN", "N_ACTIONS", "ORACLE_LEN", "META_COLUMNS",
]
