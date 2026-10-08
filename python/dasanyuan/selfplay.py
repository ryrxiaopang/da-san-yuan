"""Self-play: run hands between bots, record every decision, and run duplicate-format
tournaments to compare styles fairly.

Work is spread over processes (one shard or one batch of walls per task). Set
`selfplay.WORKERS` (or pass `--threads` to the CLI) to limit how many run at once.
"""
from __future__ import annotations

import atexit
import gzip
import math
import os
import struct
import sys
from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from . import obs
from .bots import Bot, HeuristicBot, Style
from .game import Config, Game, HandResult
from .obs import N_ACTIONS, OBS_LEN, OBS_VERSION, ORACLE_LEN, SHANTEN_LEN, View
from .rng import MASK, Rng
from .tile import NUM_KINDS

# Worker processes for parallel work; None = one per CPU core.
WORKERS: Optional[int] = None


def workers() -> int:
    return WORKERS or os.cpu_count() or 1


_pool: Optional[ProcessPoolExecutor] = None
_pool_size = 0


def _executor(n: int) -> ProcessPoolExecutor:
    """One pool kept alive for the whole run: starting worker processes is slow, and rollout
    evaluation submits work at every decision."""
    global _pool, _pool_size
    if _pool is None or _pool_size != n:
        if _pool is not None:
            _pool.shutdown()
        else:
            atexit.register(lambda: _pool is not None and _pool.shutdown())
        _pool, _pool_size = ProcessPoolExecutor(max_workers=n), n
    return _pool


def parallel_map(fn, items: Sequence) -> list:
    """Map `fn` over `items` in worker processes, keeping order. Runs inline for one worker or one item."""
    n = workers()
    if n <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    return list(_executor(n).map(fn, items))


def hand_config(hand_no: int) -> Config:
    """Config for a hand number within a session: dealer rotates every hand,
    prevailing wind advances every 4 hands."""
    return Config(dealer=hand_no % 4, prevailing_wind=(hand_no // 4) % 4)


def hand_seed(master: int, hand_no: int) -> int:
    return Rng.derive(master, hand_no).next_u64()


def play_hand(game: Game, bots: Sequence[Bot], on_decision: Optional[Callable] = None) -> Tuple[Game, HandResult]:
    """Play one hand to completion. `on_decision(game, seat, action)` is called before each
    action is applied, which is where the logger snapshots the state."""
    guard = 0
    while True:
        seat = game.to_act()
        if seat is None:
            break
        action = bots[seat].act(View(game, seat))
        if on_decision is not None:
            on_decision(game, seat, action)
        game.apply(seat, action)
        guard += 1
        assert guard < 2000, "hand did not terminate"
    assert game.result is not None, "finished game has a result"
    return game, game.result


# ----------------------------------------------------------------- logging

META_COLS = 6


class Buffers:
    """Rows recorded at each decision. meta per row: hand_id, seat, bot_id, points (filled
    after the hand), won, dealt_in."""

    def __init__(self):
        self.obs = bytearray()
        self.mask = bytearray()
        self.action = bytearray()
        self.oracle = bytearray()
        self.waits = bytearray()        # exact per-opponent winning tiles (tai) [3 x 34]
        self.shanten: List[int] = []    # shanten of every seat, relative [4]
        self.meta: List[int] = []
        self.rows = 0

    def record(self, g: Game, seat: int, a: int, hand_id: int, bot_id: int) -> None:
        self.obs += obs.encode_obs(g, seat)
        self.mask += obs.legal_mask(g, seat)
        self.action.append(a)
        self.oracle += obs.encode_oracle(g, seat)
        w, sh = obs.encode_labels(g, seat)
        self.waits += w
        self.shanten.extend(sh)
        self.meta.extend((hand_id, seat, bot_id, 0, 0, 0))
        self.rows += 1

    def fill_outcome(self, from_row: int, r: HandResult) -> None:
        m = self.meta
        for row in range(from_row, self.rows):
            o = row * META_COLS
            seat = m[o + 1]
            m[o + 3] = r.deltas[seat]
            m[o + 4] = int(r.winner == seat)
            m[o + 5] = int(r.discarder == seat)


def _npy_header(descr: str, shape: Sequence[int]) -> bytes:
    shape_s = f"({shape[0]},)" if len(shape) == 1 else "(" + ", ".join(str(s) for s in shape) + ")"
    d = f"{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_s}, }}"
    total = 10 + len(d) + 1
    d += " " * ((64 - total % 64) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(d)) + d.encode("latin1")


def _write_npy(path: str, descr: str, shape: Sequence[int], body: bytes, compress: bool) -> None:
    """Write a .npy file, or a gzip-compressed .npy.gz when `compress` is set
    (observations are mostly zeros and shrink roughly 15x)."""
    header = _npy_header(descr, shape)
    if compress:
        with open(path + ".gz", "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=1,
                                                             mtime=0) as f:
            f.write(header)
            f.write(body)
    else:
        with open(path, "wb") as f:
            f.write(header)
            f.write(body)


def write_npy_u8(path: str, data: bytes, shape: Sequence[int], compress: bool) -> None:
    _write_npy(path, "|u1", shape, bytes(data), compress)


def write_npy_i8(path: str, data: Sequence[int], shape: Sequence[int], compress: bool) -> None:
    _write_npy(path, "|i1", shape, bytes(v & 0xFF for v in data), compress)


def write_npy_i32(path: str, data: Sequence[int], shape: Sequence[int], compress: bool) -> None:
    a = array("i", data)
    if sys.byteorder != "little":
        a.byteswap()
    _write_npy(path, "<i4", shape, a.tobytes(), compress)


@dataclass
class SelfPlayConfig:
    hands: int = 10_000
    seed: int = 1
    styles: List[Style] = field(default_factory=Style.archetypes)
    # True: each hand (or game) samples 4 styles at random from `styles`; False: styles[0..4] fixed by seat.
    random_lineup: bool = True
    shard_size: int = 2_000
    out_dir: str = "data/selfplay"
    compress: bool = True               # write gzip-compressed .npy.gz files
    # Full-game mode when > 0: play this many complete games (East round to North round,
    # dealer stays on a dealer win or a kong-free draw) instead of `hands` independent
    # hands. `shard_size` then counts games per shard.
    games: int = 0


@dataclass
class ShardSummary:
    hands: int = 0
    decisions: int = 0
    wins: int = 0
    draws: int = 0


def hand_csv_header() -> str:
    return ("hand_id,seed,dealer,prevailing,bot0,bot1,bot2,bot3,winner,discarder,self_draw,tai,raw_tai,patterns,"
            "d0,d1,d2,d3,turns,game,dealer_no,repeat,hand_in_game")


@dataclass(frozen=True)
class HandLabel:
    """Where a hand sits in a full game, all 1-based except `repeat`:
    game 1, East round (prevailing), dealer 1-4 within the round, repeat 0, 1, 2..."""
    game: int
    dealer_no: int
    repeat: int
    hand_in_game: int

    @staticmethod
    def rotating(hand_id: int) -> "HandLabel":
        """Label for independent-hand mode, where the dealer passes every hand."""
        return HandLabel(hand_id // 16 + 1, hand_id % 4 + 1, 0, hand_id % 16 + 1)


@dataclass(frozen=True)
class GameProgress:
    """Position within a full game: round wind, which of the round's four dealers, and repeats."""
    prevailing: int = 0   # 0 = East round .. 3 = North round
    dealer_no: int = 0    # 0..=3: the round's first to fourth dealer, which is also the dealer's seat
    repeat: int = 0       # how many times the current dealer has stayed on

    @staticmethod
    def start() -> "GameProgress":
        return GameProgress(0, 0, 0)

    def next(self, dealer_won: bool, draw: bool, any_kong: bool) -> Optional["GameProgress"]:
        """The next hand's position, or None when the game is over.

        The dealer stays after a dealer win, or after a draw in which nobody holds a kong;
        otherwise the deal passes to the next seat. After the fourth dealer of a round the
        round wind moves on, and the game ends after the North round.
        """
        if dealer_won or (draw and not any_kong):
            return GameProgress(self.prevailing, self.dealer_no, self.repeat + 1)
        if self.dealer_no < 3:
            return GameProgress(self.prevailing, self.dealer_no + 1, 0)
        if self.prevailing < 3:
            return GameProgress(self.prevailing + 1, 0, 0)
        return None


def game_lineup(cfg: SelfPlayConfig, game: int) -> List[int]:
    """Style ids for the four players of full game `game` (0-based); they keep their seats all game."""
    n = len(cfg.styles)
    if not cfg.random_lineup:
        return [i % n for i in range(4)]
    rng = Rng.derive(cfg.seed ^ 0x6A3E5, game)
    return [rng.below(n) for _ in range(4)]


def full_game_hand_id(game: int, i: int) -> int:
    """hand_id of the i-th hand (0-based) of full game `game` (0-based): unique and sortable."""
    return game * 1000 + i


def _bot_seed(seed: int, i: int) -> int:
    return (seed ^ ((i + 1) * 0x9E37)) & MASK


def play_full_game(cfg: SelfPlayConfig, game_no: int, on_decision: Optional[Callable] = None,
                   on_hand: Optional[Callable] = None) -> List[int]:
    """Play one full game. `on_decision(game, seat, action, hand_id)` fires before every action;
    `on_hand(hand_id, seed, config, label, result, final_state)` after every hand.
    Returns the style ids of the four seats."""
    ids = game_lineup(cfg, game_no)
    gseed = hand_seed(cfg.seed ^ 0x6A3E5, game_no)
    bots = [HeuristicBot(cfg.styles[ids[i]].clone(), _bot_seed(gseed, i)) for i in range(4)]
    prog = GameProgress.start()
    i = 0
    while True:
        hand_id = full_game_hand_id(game_no, i)
        seed = hand_seed(cfg.seed, hand_id)
        hcfg = Config(dealer=prog.dealer_no, prevailing_wind=prog.prevailing)
        cb = None if on_decision is None else (lambda g, s, a: on_decision(g, s, a, hand_id))
        end, r = play_hand(Game(hcfg.clone(), seed), bots, cb)
        label = HandLabel(game_no + 1, prog.dealer_no + 1, prog.repeat, i + 1)
        if on_hand is not None:
            on_hand(hand_id, seed, hcfg, label, r, end)
        any_kong = any(m.is_kong() for p in end.players for m in p.melds)
        nxt = prog.next(r.winner == prog.dealer_no, r.winner is None, any_kong)
        if nxt is None:
            break
        prog = nxt
        i += 1
        assert i < 1000, f"game {game_no} did not finish"
    return ids


def hand_csv_row(hand_id: int, seed: int, cfg: Config, names: Sequence[str], r: HandResult, label: HandLabel) -> str:
    if r.score is not None:
        tai, raw = r.score.tai, r.score.raw_tai
        pats = "|".join(f"{p.label()}:{t}" for p, t in r.score.items)
    else:
        tai, raw, pats = 0, 0, ""
    return ",".join(str(x) for x in (
        hand_id, seed, cfg.dealer, cfg.prevailing_wind, names[0], names[1], names[2], names[3],
        -1 if r.winner is None else r.winner, -1 if r.discarder is None else r.discarder, int(r.self_draw),
        tai, raw, f'"{pats}"', r.deltas[0], r.deltas[1], r.deltas[2], r.deltas[3], r.turns,
        label.game, label.dealer_no, label.repeat, label.hand_in_game))


def run_selfplay(cfg: SelfPlayConfig) -> ShardSummary:
    """Run self-play and write shards to `out_dir/shard_XXXXX/`:
    obs.npy [N, OBS_LEN] u8, mask.npy [N, 76] u8, action.npy [N] u8,
    oracle.npy [N, 102] u8, waits.npy [N, 3, 34] u8, shanten.npy [N, 4] i8,
    meta.npy [N, 6] i32 (each .npy.gz when compressed), hands.csv."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "bots.csv"), "w", newline="\n") as f:
        f.write("bot_id,name,speed,value,defence,claim,flush,pongs\n")
        for i, s in enumerate(cfg.styles):
            f.write(f"{i},{s.to_csv()}\n")
    total_items = cfg.games if cfg.games > 0 else cfg.hands
    n_shards = -(-total_items // cfg.shard_size)
    jobs = [(cfg, shard, shard * cfg.shard_size, min((shard + 1) * cfg.shard_size, total_items))
            for shard in range(n_shards)]
    summaries = parallel_map(_run_game_shard_job if cfg.games > 0 else _run_shard_job, jobs)
    total = ShardSummary()
    for s in summaries:
        total.hands += s.hands
        total.decisions += s.decisions
        total.wins += s.wins
        total.draws += s.draws
    # Written last so it records the real hand count (full games vary in length).
    with open(os.path.join(cfg.out_dir, "format.txt"), "w", newline="\n") as f:
        f.write(f"obs_version={OBS_VERSION}\nobs_len={OBS_LEN}\nn_actions={N_ACTIONS}\noracle_len={ORACLE_LEN}\n"
                f"waits_shape=3,34\nshanten_len={SHANTEN_LEN}\nmeta_cols=hand_id,seat,bot_id,points,won,dealt_in\n")
        f.write(f"seed={cfg.seed}\nhands={total.hands}\nrandom_lineup={'true' if cfg.random_lineup else 'false'}\n")
        f.write(f"mode={'full_games' if cfg.games > 0 else 'hands'}\ngames={cfg.games}\n")
    return total


def _run_shard_job(job) -> ShardSummary:
    return run_shard(*job)


def _run_game_shard_job(job) -> ShardSummary:
    return run_game_shard(*job)


def run_shard(cfg: SelfPlayConfig, shard: int, first: int, last: int) -> ShardSummary:
    d = os.path.join(cfg.out_dir, f"shard_{shard:05d}")
    os.makedirs(d, exist_ok=True)
    buf = Buffers()
    lines = [hand_csv_header()]
    summary = ShardSummary()
    lineup_rng = Rng.derive(cfg.seed ^ 0x5EED, shard)
    n = len(cfg.styles)
    for hand_id in range(first, last):
        seed = hand_seed(cfg.seed, hand_id)
        gcfg = hand_config(hand_id)
        ids = [lineup_rng.below(n) for _ in range(4)] if cfg.random_lineup else [i % n for i in range(4)]
        bots = [HeuristicBot(cfg.styles[ids[i]].clone(), _bot_seed(seed, i)) for i in range(4)]
        names = [cfg.styles[ids[i]].name for i in range(4)]
        start_row = buf.rows
        _, r = play_hand(Game(gcfg.clone(), seed), bots,
                         lambda g, seat, a: buf.record(g, seat, a, hand_id, ids[seat]))
        buf.fill_outcome(start_row, r)
        lines.append(hand_csv_row(hand_id, seed, gcfg, names, r, HandLabel.rotating(hand_id)))
        summary.hands += 1
        if r.winner is not None:
            summary.wins += 1
        else:
            summary.draws += 1
    with open(os.path.join(d, "hands.csv"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    write_shard_arrays(d, buf, cfg.compress)
    summary.decisions = buf.rows
    return summary


def run_game_shard(cfg: SelfPlayConfig, shard: int, first: int, last: int) -> ShardSummary:
    d = os.path.join(cfg.out_dir, f"shard_{shard:05d}")
    os.makedirs(d, exist_ok=True)
    buf = Buffers()
    lines = [hand_csv_header()]
    summary = ShardSummary()
    for game_no in range(first, last):
        ids = game_lineup(cfg, game_no)
        names = [cfg.styles[ids[i]].name for i in range(4)]
        start_row = [buf.rows]

        def on_hand(hand_id, seed, hcfg, label, r, _end):
            buf.fill_outcome(start_row[0], r)
            start_row[0] = buf.rows
            lines.append(hand_csv_row(hand_id, seed, hcfg, names, r, label))
            summary.hands += 1
            if r.winner is not None:
                summary.wins += 1
            else:
                summary.draws += 1

        play_full_game(cfg, game_no, lambda g, seat, a, hand_id: buf.record(g, seat, a, hand_id, ids[seat]), on_hand)
    with open(os.path.join(d, "hands.csv"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    write_shard_arrays(d, buf, cfg.compress)
    summary.decisions = buf.rows
    return summary


def write_shard_arrays(d: str, buf: Buffers, z: bool) -> None:
    n = buf.rows
    write_npy_u8(os.path.join(d, "obs.npy"), buf.obs, [n, OBS_LEN], z)
    write_npy_u8(os.path.join(d, "mask.npy"), buf.mask, [n, N_ACTIONS], z)
    write_npy_u8(os.path.join(d, "action.npy"), buf.action, [n], z)
    write_npy_u8(os.path.join(d, "oracle.npy"), buf.oracle, [n, ORACLE_LEN], z)
    write_npy_u8(os.path.join(d, "waits.npy"), buf.waits, [n, 3, 34], z)
    write_npy_i8(os.path.join(d, "shanten.npy"), buf.shanten, [n, SHANTEN_LEN], z)
    write_npy_i32(os.path.join(d, "meta.npy"), buf.meta, [n, META_COLS], z)


# ------------------------------------------------------------- tournament

@dataclass
class EntrantStats:
    name: str = ""
    hands: int = 0
    points: int = 0
    sum_sq: float = 0.0
    wins: int = 0
    self_draws: int = 0
    deal_ins: int = 0
    tai_sum: int = 0

    def mean(self) -> float:
        return self.points / max(self.hands, 1)

    def stderr(self) -> float:
        """Standard error of the mean points per hand, estimated across wall-sets
        (each wall-set = the 4 seat rotations of one wall)."""
        w = float(max(self.hands // 4, 2))
        m = self.mean()
        return math.sqrt(max(self.sum_sq / w - m * m, 0.0) / (w - 1.0))


def _tournament_wall(styles: Sequence[Style], w: int, seed: int):
    acc = [[0, 0, 0, 0, 0] for _ in range(4)]  # points, wins, tsumo, deal-ins, tai
    gseed = hand_seed(seed, w)
    cfg = hand_config(w)
    base = Game(cfg.clone(), gseed).wall
    for rot in range(4):
        # entrant e sits at seat (e + rot) % 4
        bots = [HeuristicBot(styles[(seat + 4 - rot) % 4].clone(), (gseed ^ (rot * 4 + seat + 7)) & MASK)
                for seat in range(4)]
        _, r = play_hand(Game.from_wall(cfg.clone(), base), bots)
        for seat in range(4):
            e = (seat + 4 - rot) % 4
            acc[e][0] += r.deltas[seat]
            if r.winner == seat:
                acc[e][1] += 1
                if r.self_draw:
                    acc[e][2] += 1
                acc[e][4] += r.score.tai if r.score is not None else 0
            if r.discarder == seat:
                acc[e][3] += 1
    return acc


def _tournament_job(job):
    styles, walls, seed = job
    return [_tournament_wall(styles, w, seed) for w in walls]


def duplicate_tournament(styles: Sequence[Style], walls: int, seed: int) -> List[EntrantStats]:
    """Duplicate format: every wall is played 4 times with the entrants rotated through all
    seats, so the luck of the deal cancels out. Points are accumulated per wall-set (the 4
    rotations) for the variance estimate."""
    n = workers()
    chunk = max(1, min(64, -(-walls // (4 * n))))
    jobs = [(list(styles), range(a, min(a + chunk, walls)), seed) for a in range(0, walls, chunk)]
    per_wall = [acc for part in parallel_map(_tournament_job, jobs) for acc in part]
    stats = [EntrantStats(name=s.name) for s in styles]
    for acc in per_wall:
        for e in range(4):
            st = stats[e]
            st.hands += 4
            st.points += acc[e][0]
            # variance over wall-sets, scaled to per-hand
            per_hand = acc[e][0] / 4.0
            st.sum_sq += per_hand * per_hand
            st.wins += acc[e][1]
            st.self_draws += acc[e][2]
            st.deal_ins += acc[e][3]
            st.tai_sum += acc[e][4]
    return stats


def verify_hand_invariants(g: Game) -> None:
    """Quick sanity helper used by tests and the CLI. Raises ValueError on a broken invariant."""
    drawn, held = g.tile_census()
    if drawn != held:
        raise ValueError(f"tile census mismatch: drawn {drawn} held {held}")
    r = g.result
    if r is not None:
        if sum(r.deltas) != 0:
            raise ValueError(f"points not zero-sum: {r.deltas}")
        if r.score is not None and not 1 <= r.score.tai <= 5:
            raise ValueError(f"tai out of range: {r.score.tai}")
    for p in g.players:
        if any(p.hand[k] > 4 for k in range(NUM_KINDS)):
            raise ValueError("more than 4 copies in a hand")


# ------------------------------------------------------------------ replay

def lineup_for(cfg: SelfPlayConfig, hand_id: int) -> List[int]:
    """Style ids seated at each seat for `hand_id` of a self-play run, exactly as `run_selfplay`
    drew them (same shard RNG stream), so any hand in a dataset can be replayed move for move."""
    n = len(cfg.styles)
    if not cfg.random_lineup:
        return [i % n for i in range(4)]
    shard = hand_id // cfg.shard_size
    rng = Rng.derive(cfg.seed ^ 0x5EED, shard)
    for _ in range((hand_id - shard * cfg.shard_size) * 4):
        rng.below(n)
    return [rng.below(n) for _ in range(4)]


def bots_for(cfg: SelfPlayConfig, hand_id: int) -> Tuple[List[int], List[Bot]]:
    """The four bots for a dataset hand, seeded exactly as in `run_selfplay`."""
    ids = lineup_for(cfg, hand_id)
    seed = hand_seed(cfg.seed, hand_id)
    return ids, [HeuristicBot(cfg.styles[ids[i]].clone(), _bot_seed(seed, i)) for i in range(4)]
