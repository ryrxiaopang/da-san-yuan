"""Load a self-play run (output of `dsy selfplay`) into PostgreSQL.

    python db/load_selfplay.py data/selfplay_100k
    python db/load_selfplay.py data/selfplay_100k --name baseline-v1 --replace
    python db/load_selfplay.py data/selfplay_100k --no-decisions     # hands only, much smaller

Connection: --db or the DATABASE_URL environment variable
(default postgresql://dsy:dsy@localhost:5432/dasanyuan, matching db/docker-compose.yml).

Needs only numpy, pandas and psycopg (pip install numpy pandas "psycopg[binary]").
The schema is applied automatically. Each run loads in a single transaction, so a
failed load leaves nothing behind.
"""
from __future__ import annotations

import argparse
import csv
import glob
import gzip
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psycopg

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = "postgresql://dsy:dsy@localhost:5432/dasanyuan"
SUPPORTED_OBS_VERSIONS = {2}
META_LEN = 14  # trailing meta block of each observation (engine/src/obs.rs)

ACTION_TYPES = np.array(
    ["discard"] * 34 + ["kong"] * 34
    + ["tsumo", "ron", "pong", "exposed_kong", "chow", "chow", "chow", "pass"]
)
PHASES = np.array(["own_turn", "claim", "rob_kong", "over"])
TILE_NAMES = np.array(
    [f"{k % 9 + 1}{'mps'[k // 9]}" for k in range(27)] + [f"{k - 26}z" for k in range(27, 34)]
)


def read_format(run_dir: str) -> dict:
    path = os.path.join(run_dir, "format.txt")
    if not os.path.exists(path):
        sys.exit(f"{path} not found: is this a `dsy selfplay` output folder?")
    out = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                out[k] = v
    return out


def read_bots(run_dir: str) -> list[dict]:
    with open(os.path.join(run_dir, "bots.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["bot_id"] = int(r["bot_id"])
        for k in ("speed", "value", "defence", "claim", "flush", "pongs"):
            r[k] = float(r[k])
    return rows


def load_array(shard: str, name: str) -> np.ndarray:
    plain = os.path.join(shard, name + ".npy")
    if os.path.exists(plain):
        return np.load(plain)
    with gzip.open(plain + ".gz", "rb") as f:
        return np.load(f)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", HERE, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def copy_df(cur, table: str, df: pd.DataFrame) -> None:
    """Bulk insert through COPY (much faster than INSERT)."""
    if df.empty:
        return
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N", lineterminator="\n")  # \n on Windows too
    cols = ", ".join(df.columns)
    with cur.copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as cp:
        cp.write(buf.getvalue())


def hand_tables(run_id: int, shard: str):
    h = pd.read_csv(os.path.join(shard, "hands.csv"), keep_default_na=False, dtype={"seed": str})
    hands = pd.DataFrame({
        "run_id": run_id,
        "hand_id": h.hand_id,
        "seed": h.seed,
        "dealer": h.dealer,
        "prevailing": h.prevailing,
        "winner": h.winner.where(h.winner >= 0).astype("Int16"),
        "discarder": h.discarder.where(h.discarder >= 0).astype("Int16"),
        "self_draw": h.self_draw.astype(bool),
        "tai": h.tai,
        "raw_tai": h.raw_tai,
        "turns": h.turns,
    })
    # Full-game position (only in runs made with `dsy selfplay --games`).
    for src, dst in (("game", "game"), ("dealer_no", "dealer_no"), ("repeat", "repeat_no"), ("hand_in_game", "hand_in_game")):
        hands[dst] = h[src].astype("Int64") if src in h.columns else pd.array([pd.NA] * len(h), dtype="Int64")
    seats = []
    for s in range(4):
        seats.append(pd.DataFrame({
            "run_id": run_id,
            "hand_id": h.hand_id,
            "seat": s,
            "seat_wind": (s - h.dealer) % 4,
            "bot": h[f"bot{s}"],
            "points": h[f"d{s}"],
            "won": h.winner == s,
            "dealt_in": h.discarder == s,
        }))
    seats = pd.concat(seats, ignore_index=True)
    pats = []
    for hid, p in zip(h.hand_id, h.patterns):
        for item in filter(None, str(p).split("|")):
            name, tai = item.rsplit(":", 1)
            pats.append((run_id, hid, name, int(tai)))
    patterns = pd.DataFrame(pats, columns=["run_id", "hand_id", "pattern", "tai"])
    return hands, seats, patterns


def only_discards(values: np.ndarray, is_discard: np.ndarray) -> pd.Series:
    """Integer column that is NULL for non-discard decisions."""
    return pd.Series(values.astype(np.int16), dtype="Int16").where(is_discard)


def decision_table(run_id: int, shard: str, bot_names: dict, obs_len: int) -> pd.DataFrame:
    meta = load_array(shard, "meta")          # hand_id, seat, bot_id, points, won, dealt_in
    action = load_array(shard, "action").astype(np.int16)
    mask = load_array(shard, "mask")
    waits = load_array(shard, "waits")        # [n, 3, 34] tai per opponent per tile
    shanten = load_array(shard, "shanten")    # [n, 4], self first
    obs = load_array(shard, "obs")
    assert obs.shape[1] == obs_len, f"obs length {obs.shape[1]} != {obs_len}"
    m = obs[:, obs_len - META_LEN:]
    n = len(action)

    hand_id = meta[:, 0]
    starts = np.r_[True, hand_id[1:] != hand_id[:-1]]
    group_start = np.maximum.accumulate(np.where(starts, np.arange(n), 0))
    idx = np.arange(n) - group_start

    is_discard = action < 34
    danger = waits.max(axis=1)                # [n, 34] worst opponent per tile
    legal_disc = mask[:, :34] == 1
    unsafe = ((danger > 0) & legal_disc).sum(1)
    safe = legal_disc.sum(1) - unsafe
    chosen = danger[np.arange(n), np.where(is_discard, action, 0)]

    tile = np.full(n, None, dtype=object)
    tile[is_discard] = TILE_NAMES[action[is_discard]]
    kong = (action >= 34) & (action < 68)
    tile[kong] = TILE_NAMES[action[kong] - 34]

    claim_tile, claim_from, meld, options, move = describe_moves(action, mask, obs[:, :34], m, meta[:, 1])

    return pd.DataFrame({
        "run_id": run_id,
        "hand_id": hand_id,
        "idx": idx,
        "seat": meta[:, 1],
        "bot": pd.Series(meta[:, 2]).map(bot_names),
        "phase": PHASES[np.minimum(m[:, 5], 3)],
        "action": action,
        "action_type": ACTION_TYPES[action],
        "tile": tile,
        "n_legal": mask.sum(1),
        "draws_left": m[:, 3],
        "own_shanten": shanten[:, 0],
        "opp_ready": (shanten[:, 1:] == 0).sum(1),
        "unsafe_options": only_discards(unsafe, is_discard),
        "safe_options": only_discards(safe, is_discard),
        "deal_in_tai": only_discards(chosen, is_discard),
        "claim_tile": claim_tile,
        "claim_from": pd.Series(claim_from, dtype="Int16"),
        "meld": meld,
        "options": options,
        "move": move,
    })


SEAT_NAMES = ["Seat 0", "Seat 1", "Seat 2", "Seat 3"]


def describe_moves(action, mask, hand, m, seat):
    """Plain-language description of every move, spelling out pongs, chows and kongs taken from a
    discard: which tile, from whom, and the set it made. Also lists what else the player could do."""
    n = len(action)
    phase = m[:, 5]
    target = m[:, 6]                      # tile on offer (claim / rob kong) or tile just drawn
    src_rel = m[:, 7]                     # seat that threw it, relative to the player
    claim_tile = np.full(n, None, dtype=object)
    claim_from = np.full(n, None, dtype=object)
    meld = np.full(n, None, dtype=object)
    options = np.full(n, None, dtype=object)
    move = np.empty(n, dtype=object)
    for i in range(n):
        a = int(action[i])
        if phase[i] in (1, 2):            # reacting to a discard / an added kong
            t = int(target[i])
            frm = (int(seat[i]) + int(src_rel[i])) % 4
            tn = TILE_NAMES[t]
            claim_tile[i], claim_from[i] = tn, frm
            legal = np.flatnonzero(mask[i])
            opts = []
            if 69 in legal: opts.append("win")
            if 70 in legal: opts.append("pong")
            if 71 in legal: opts.append("kong")
            if any(x in legal for x in (72, 73, 74)): opts.append("chow")
            options[i] = ", ".join(opts)
            who = SEAT_NAMES[frm]
            if a == 69:
                move[i] = (f"Rob the kong: win on {tn} from {who}" if phase[i] == 2 else f"Win on {tn} thrown by {who}")
            elif a == 70:
                meld[i] = tn[0] * 3 + tn[1]
                move[i] = f"Pong {tn} thrown by {who}"
            elif a == 71:
                meld[i] = tn[0] * 4 + tn[1]
                move[i] = f"Kong {tn} thrown by {who}"
            elif a in (72, 73, 74):
                low = t - (a - 72)
                run = [TILE_NAMES[low + k] for k in range(3)]
                meld[i] = "".join(x[0] for x in run) + tn[1]
                move[i] = f"Chow {meld[i]} using {tn} thrown by {who}"
            else:
                move[i] = f"Pass on {tn} from {who} (could {options[i].replace(', ', ' / ')})"
        else:
            if a < 34:
                move[i] = f"Discard {TILE_NAMES[a]}"
            elif a < 68:
                t = a - 34
                kind = "Concealed kong" if hand[i, t] == 4 else "Added kong (onto own pong)"
                meld[i] = TILE_NAMES[t][0] * 4 + TILE_NAMES[t][1]
                move[i] = f"{kind} {TILE_NAMES[t]}"
            elif a == 68:
                move[i] = "Self-drawn win"
            else:
                move[i] = f"Action {a}"
    return claim_tile, claim_from, meld, options, move


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="folder written by `dsy selfplay --out ...`")
    ap.add_argument("--db", default=os.environ.get("DATABASE_URL", DEFAULT_DB))
    ap.add_argument("--name", help="run name (default: folder name)")
    ap.add_argument("--replace", action="store_true", help="delete an existing run with this name first")
    ap.add_argument("--no-decisions", action="store_true", help="skip the per-decision table")
    ap.add_argument("--notes", default=None)
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    fmt = read_format(run_dir)
    obs_version = int(fmt.get("obs_version", 1))
    if obs_version not in SUPPORTED_OBS_VERSIONS:
        sys.exit(f"obs_version {obs_version} is not supported by this loader; regenerate the data")
    obs_len = int(fmt["obs_len"])
    bots = read_bots(run_dir)
    bot_names = {b["bot_id"]: b["name"] for b in bots}
    shards = sorted(glob.glob(os.path.join(run_dir, "shard_*")))
    if not shards:
        sys.exit(f"no shard_* folders in {run_dir}")
    name = args.name or os.path.basename(run_dir.rstrip("/"))

    t0 = time.time()
    with psycopg.connect(args.db) as conn:
        with open(os.path.join(HERE, "schema.sql")) as f:
            conn.execute(f.read())
        cur = conn.cursor()
        old = cur.execute("SELECT run_id FROM runs WHERE name = %s", (name,)).fetchone()
        if old:
            if not args.replace:
                sys.exit(f"run '{name}' is already loaded (run_id {old[0]}); use --replace or --name")
            print(f"replacing run '{name}' (run_id {old[0]})")
            cur.execute("DELETE FROM runs WHERE run_id = %s", (old[0],))
        run_id = cur.execute(
            """INSERT INTO runs (name, seed, hands, random_lineup, obs_version, engine_commit,
                                 shard_path, bots, notes)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING run_id""",
            (name, int(fmt["seed"]), int(fmt["hands"]), fmt["random_lineup"] == "true",
             obs_version, git_commit(), run_dir, json.dumps(bots), args.notes),
        ).fetchone()[0]

        totals = {"hands": 0, "decisions": 0}
        for i, shard in enumerate(shards, 1):
            hands, seats, patterns = hand_tables(run_id, shard)
            copy_df(cur, "hands", hands)
            copy_df(cur, "hand_seats", seats)
            copy_df(cur, "hand_patterns", patterns)
            totals["hands"] += len(hands)
            if not args.no_decisions:
                dec = decision_table(run_id, shard, bot_names, obs_len)
                copy_df(cur, "decisions", dec)
                totals["decisions"] += len(dec)
            print(f"  shard {i}/{len(shards)}: {totals['hands']:,} hands, "
                  f"{totals['decisions']:,} decisions ({time.time() - t0:.0f}s)", flush=True)
        if totals["hands"] != int(fmt["hands"]):
            sys.exit(f"expected {fmt['hands']} hands but found {totals['hands']}; nothing was saved")
        conn.commit()
        cur.execute("ANALYZE hands; ANALYZE hand_seats; ANALYZE hand_patterns; ANALYZE decisions;")
        conn.commit()

    print(f"loaded run '{name}' as run_id {run_id}: {totals['hands']:,} hands, "
          f"{totals['decisions']:,} decisions in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
