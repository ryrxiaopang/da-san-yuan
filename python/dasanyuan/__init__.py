"""Python bindings for the da-san-yuan Singapore mahjong engine.

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
"""
from __future__ import annotations

import glob
import gzip
import os
from typing import Dict, List

import numpy as np

from . import _core
from ._core import n_actions, obs_len, oracle_len, run_selfplay, tournament

OBS_LEN = obs_len()
N_ACTIONS = n_actions()
ORACLE_LEN = oracle_len()

# Action index layout (must match engine/src/obs.rs)
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


class Env:
    """One hand as a turn-based environment. Seats are 0..3; dealer sits East."""

    def __init__(self, seed: int = 0, dealer: int = 0, prevailing: int = 0):
        self._e = _core.Env(seed, dealer, prevailing)

    def reset(self, seed: int, dealer: int = 0, prevailing: int = 0) -> None:
        self._e.reset(seed, dealer, prevailing)

    def to_act(self) -> int:
        return self._e.to_act()

    def is_over(self) -> bool:
        return self._e.is_over()

    def observe(self, seat: int):
        """(obs[OBS_LEN] uint8, legal_mask[N_ACTIONS] uint8) for `seat`."""
        obs = np.frombuffer(self._e.obs_bytes(seat), dtype=np.uint8)
        mask = np.frombuffer(self._e.mask_bytes(seat), dtype=np.uint8)
        return obs, mask

    def oracle(self, seat: int) -> np.ndarray:
        """Hidden hands of the other three seats [3, 34]. Training targets only."""
        return np.frombuffer(self._e.oracle_bytes(seat), dtype=np.uint8).reshape(3, 34)

    def labels(self, seat: int):
        """Exact training targets (uses hidden information, never a policy input).

        waits [3, 34]: tai each opponent (relative seats 1..3) would win with if that
        tile were discarded now; 0 means the tile cannot deal in to them.
        shanten [4]: tiles-from-ready of every seat, self first (0 = ready).
        """
        w, sh = self._e.labels_bytes(seat)
        return np.frombuffer(w, dtype=np.uint8).reshape(3, 34), np.frombuffer(sh, dtype=np.int8)

    def step(self, seat: int, action: int) -> None:
        self._e.step(seat, int(action))

    def bot_action(self, seat: int, style: str = "balanced") -> int:
        return self._e.bot_action(seat, style)

    def deltas(self) -> List[int]:
        return list(self._e.deltas())

    def result(self) -> dict:
        return self._e.result()

    def hand_str(self, seat: int) -> str:
        return self._e.hand_str(seat)

    def draws_left(self) -> int:
        return self._e.draws_left()


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
        self._b = _core.ScenarioBank(directory)

    def __len__(self) -> int:
        return len(self._b)

    def info(self, i: int) -> dict:
        f, name, desc, tags, hero = self._b.info(i)
        return {"file": f, "name": name, "desc": desc, "tags": tags, "hero": hero}

    def env(self, i: int) -> "Env":
        e = Env.__new__(Env)
        e._e = self._b.env(i)
        return e

    def check(self, i: int, action: int):
        return self._b.check(i, int(action))


def load_npy(path_without_ext: str, mmap: bool = True) -> np.ndarray:
    """Load `<path>.npy`, or `<path>.npy.gz` if only the compressed file exists."""
    if os.path.exists(path_without_ext + ".npy"):
        return np.load(path_without_ext + ".npy", mmap_mode="r" if mmap else None)
    with gzip.open(path_without_ext + ".npy.gz", "rb") as f:
        return np.load(f)


def load_shards(root: str, limit: int | None = None, mmap: bool = True) -> Dict[str, np.ndarray]:
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
