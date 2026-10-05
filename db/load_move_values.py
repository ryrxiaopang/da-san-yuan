"""Load per-move expected values written by `dsy evaluate` into the move_values table.

    dsy evaluate --games 1-100 --seed 1 --out data/games1000/move_values.csv
    python db/load_move_values.py data/games1000/move_values.csv --run games1000

The run must already be loaded with its decisions (db/load_selfplay.py without --no-decisions),
since each value is matched to its move by (hand_id, move number). Re-loading replaces the rows.
"""

import argparse
import io
import os

import pandas as pd
import psycopg

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB = "postgresql://dsy:dsy@localhost:5432/dasanyuan"
COLS = ["run_id", "hand_id", "idx", "worlds", "oracle", "chosen_ev", "best_action", "best_move", "best_ev",
        "regret", "regret_se", "hand_points", "all_moves"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--run", required=True, help="run name in the runs table, e.g. games1000")
    ap.add_argument("--worlds", type=int, default=64, help="worlds used by dsy evaluate (for the record)")
    ap.add_argument("--oracle", action="store_true", help="the file was made with dsy evaluate --oracle")
    ap.add_argument("--db", default=os.environ.get("DATABASE_URL", DEFAULT_DB))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    with psycopg.connect(args.db) as conn:
        with open(os.path.join(HERE, "schema.sql")) as f:
            conn.execute(f.read())
        row = conn.execute("select run_id from runs where name = %s", (args.run,)).fetchone()
        if row is None:
            raise SystemExit(f"run {args.run!r} is not loaded; run db/load_selfplay.py first")
        run_id = row[0]
        df["run_id"] = run_id
        df["worlds"] = args.worlds
        df["oracle"] = args.oracle
        n_missing = conn.execute(
            "select count(*) from unnest(%s::int[], %s::smallint[]) as x(h, i) "
            "where not exists (select 1 from decisions d where d.run_id = %s and d.hand_id = x.h and d.idx = x.i)",
            (df.hand_id.tolist(), df.idx.tolist(), run_id),
        ).fetchone()[0]
        if n_missing:
            raise SystemExit(f"{n_missing} rows do not match a stored move: wrong --run, or decisions not loaded")
        conn.execute(
            "delete from move_values where run_id = %s and oracle = %s and hand_id = any(%s)",
            (run_id, args.oracle, sorted(set(df.hand_id.tolist()))),
        )
        buf = io.StringIO()
        df[COLS].to_csv(buf, index=False, header=False, lineterminator="\n")
        buf.seek(0)
        with conn.cursor() as cur:
            with cur.copy(f"COPY move_values ({', '.join(COLS)}) FROM STDIN WITH (FORMAT csv)") as cp:
                cp.write(buf.getvalue())
    print(f"loaded {len(df):,} move values into run {args.run}")


if __name__ == "__main__":
    main()
