import numpy as np
import dasanyuan as dsy


def play(seed, style="balanced"):
    env = dsy.Env(seed=seed)
    steps = 0
    while not env.is_over():
        seat = env.to_act()
        obs, mask = env.observe(seat)
        assert obs.shape == (dsy.OBS_LEN,) and mask.shape == (dsy.N_ACTIONS,)
        a = env.bot_action(seat, style)
        assert mask[a] == 1
        env.step(seat, a)
        steps += 1
    return env, steps


def test_hands_finish_and_are_zero_sum():
    for seed in range(50):
        env, steps = play(seed)
        assert steps > 0
        assert sum(env.deltas()) == 0


def test_illegal_action_rejected():
    env = dsy.Env(seed=1)
    seat = env.to_act()
    _, mask = env.observe(seat)
    illegal = int(np.flatnonzero(mask == 0)[0])
    try:
        env.step(seat, illegal)
    except ValueError:
        return
    raise AssertionError("illegal action accepted")


def test_selfplay_roundtrip(tmp_path):
    hands, decisions = dsy.run_selfplay(str(tmp_path), hands=40, seed=3, shard_size=20)
    assert hands == 40
    data = dsy.load_shards(str(tmp_path))
    n = data["action"].shape[0]
    assert n == decisions
    assert data["obs"].shape == (n, dsy.OBS_LEN)
    assert data["oracle"].shape == (n, dsy.ORACLE_LEN)
    # every recorded action was legal
    assert np.all(data["mask"][np.arange(n), data["action"]] == 1)
    # outcomes are zero-sum per hand when summed over one row per seat
    meta = data["meta"]
    for hid in np.unique(meta[:, 0])[:10]:
        rows = meta[meta[:, 0] == hid]
        per_seat = {int(r[1]): int(r[3]) for r in rows}
        if len(per_seat) == 4:
            assert sum(per_seat.values()) == 0


def test_tournament_runs():
    res = dsy.tournament(["fast", "high_tai", "defensive", "balanced"], walls=20)
    assert len(res) == 4
    assert abs(sum(r["points_per_hand"] for r in res)) < 1e-9
