"""Loader round-trip against a real PostgreSQL. Skipped unless DATABASE_URL is set."""
import os
import subprocess
import sys
import uuid

import pytest

import dasanyuan as dsy

DB = os.environ.get("DATABASE_URL")
psycopg = pytest.importorskip("psycopg")
LOADER = os.path.join(os.path.dirname(__file__), "..", "..", "db", "load_selfplay.py")


@pytest.mark.skipif(not DB, reason="set DATABASE_URL to run database tests")
def test_load_selfplay_roundtrip(tmp_path):
    hands, decisions = dsy.run_selfplay(str(tmp_path), hands=60, seed=11, shard_size=25)
    name = f"pytest-{uuid.uuid4().hex[:8]}"
    subprocess.run([sys.executable, LOADER, str(tmp_path), "--name", name, "--db", DB], check=True)
    try:
        with psycopg.connect(DB) as conn:
            run_id = conn.execute("SELECT run_id FROM runs WHERE name = %s", (name,)).fetchone()[0]
            n_hands = conn.execute("SELECT count(*) FROM hands WHERE run_id = %s", (run_id,)).fetchone()[0]
            n_seats = conn.execute("SELECT count(*) FROM hand_seats WHERE run_id = %s", (run_id,)).fetchone()[0]
            n_dec = conn.execute("SELECT count(*) FROM decisions WHERE run_id = %s", (run_id,)).fetchone()[0]
            zero_sum = conn.execute(
                "SELECT bool_and(s = 0) FROM (SELECT sum(points) s FROM hand_seats WHERE run_id = %s GROUP BY hand_id) x",
                (run_id,)).fetchone()[0]
            wins_match = conn.execute(
                """SELECT count(*) FROM hands h JOIN hand_seats s USING (run_id, hand_id)
                   WHERE h.run_id = %s AND s.won AND s.seat <> h.winner""", (run_id,)).fetchone()[0]
        assert n_hands == hands == 60
        assert n_seats == 4 * hands
        assert n_dec == decisions
        assert zero_sum
        assert wins_match == 0
    finally:
        with psycopg.connect(DB) as conn:
            conn.execute("DELETE FROM runs WHERE name = %s", (name,))
