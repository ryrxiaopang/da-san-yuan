# da-san-yuan 大三元

Singapore 4-player mahjong engine, heuristic bots and self-play data generator. It is the foundation for a learning platform that advises players on the best move and explains why.

This repo currently contains **stage 1 of the training pipeline**: a fast, tested rules engine, four heuristic bots, a self-play logger that writes training data with exact opponent-reading labels, a duplicate-format tournament for comparing bots fairly, an evolution loop, a scenario bank for testing specific skills, and Python bindings for the RL work.

- **[RULES.md](RULES.md)**: the exact ruleset, confirmed with the team.
- **[docs/DESIGN.md](docs/DESIGN.md)**: the reinforcement-learning plan, how explanations stay faithful, how to evaluate, and open issues.

## Layout

```
engine/         Rust library: rules, scoring, shanten, bots, self-play
  src/tile.rs       tile ids and notation (1m..9m, 1p.., 1s.., 1z..7z, f1-f4, g1-g4, a1-a4)
  src/game.rs       state machine for one hand (deal, draws, claims, kongs, robbing, draw game)
  src/scoring.rs    win decomposition, tai patterns, payments
  src/shanten.rs    table-based shanten and useful-tile counts
  src/obs.rs        action space (76) and observation encoding (672 bytes)
  src/bots.rs       heuristic bots: fast, high_tai, defensive, balanced
  src/selfplay.rs   data logger, duplicate tournament
  src/scenario.rs   scenario format, position builder, exact checks
  tests/rules.rs    rule tests (one per scoring rule), engine invariants, label checks
scenarios/      hand-built positions: defence, flush/honour reading, pushing, tai planning, rules
db/             PostgreSQL schema, loader and docker-compose for analysis
notebooks/      EDA notebook and exported figures
docs/           design and RL plan
cli/            `dsy` command-line tool
python/         PyO3 bindings: `import dasanyuan`
```

## Setup

You need Rust (https://rustup.rs) and Python 3.9+.

```bash
cargo build --release          # builds target/release/dsy
cargo test --release           # rule tests + 3,000 full simulated hands
cargo install --path cli       # optional: puts `dsy` on your PATH

cd python
pip install maturin numpy pytest
maturin develop --release      # installs `dasanyuan` into your active environment
pytest tests
```

Without `cargo install`, run it as `target/release/dsy` (`target\release\dsy.exe` on Windows).

## Commands

```bash
# Check a hand against the rules (seat 0 = East). Prints every tai pattern and the payment.
dsy score "123m456p789s123s99m" --win 1s --seat 1

# Generate training data (all cores). 100k hands is about 700 MB compressed.
dsy selfplay --hands 100000 --out data/selfplay

# Fair comparison: every wall is played 4 times with the bots rotated through all seats.
dsy tournament --styles fast,high_tai,defensive,balanced --walls 5000

# Evolution: champion vs 3 mutated copies of itself; a challenger is promoted only if it
# beats the champion by more than 2 standard errors.
dsy evolve --generations 30 --walls 3000 --start balanced --log data/evolve.csv

# Score bots on the scenario bank (pass rate per scenario and per skill tag).
dsy scenarios --verbose

dsy bench                       # throughput on this machine
```

Styles can also be given as CSV lines (`name,speed,value,defence,claim,flush,pongs`), so a champion from `evolve.csv` can be fed straight back into `selfplay` or `tournament`.

## Python

```python
import dasanyuan as dsy

env = dsy.Env(seed=1)
while not env.is_over():
    seat = env.to_act()
    obs, mask = env.observe(seat)            # uint8 [672], uint8 [76]
    action = env.bot_action(seat, "balanced")  # or your policy's choice
    env.step(seat, action)
print(env.result())   # winner, tai, patterns, point changes

data = dsy.load_shards("data/selfplay")      # obs, mask, action, oracle, meta
```

`python examples/inspect_data.py data/selfplay` prints the action mix, results per bot, tai distribution, and how often each bot threw a tile that could deal in compared with a random choice.

```python
w, sh = env.labels(seat)     # exact: [3, 34] tai each opponent wins with per tile, [4] shanten
bank = dsy.Scenarios("../scenarios")
env = bank.env(0); seat = env.to_act()
ok, msg = bank.check(0, my_policy(*env.observe(seat)))
```

## Data format

Each `shard_XXXXX/` directory holds gzip-compressed numpy arrays (`--no-compress` for plain `.npy`), about 7 KB per hand:

| File | Shape | Meaning |
|---|---|---|
| `obs` | [N, 672] u8 | What the acting player can see. Layout documented in `engine/src/obs.rs` |
| `mask` | [N, 76] u8 | Legal actions |
| `action` | [N] u8 | Action taken |
| `oracle` | [N, 102] u8 | The other three players' hidden hands. **Training target only**, never a policy input |
| `waits` | [N, 3, 34] u8 | For each opponent and tile: tai they would win with if you discarded it now (0 = cannot deal in). Exact, from the engine |
| `shanten` | [N, 4] i8 | Tiles from ready for every seat, you first (0 = ready, -1 = complete) |
| `meta` | [N, 6] i32 | hand_id, seat, bot_id, final points for this seat, won, dealt in |
| `hands.csv` | per hand | seed, dealer, bots, winner, tai, patterns, point changes |

Seats in `obs` and `oracle` are relative: 0 = you, 1 = next player, 2 = opposite, 3 = previous. Actions: 0–33 discard, 34–67 kong, 68 tsumo, 69 ron, 70 pong, 71 exposed kong, 72–74 chow (claimed tile low/middle/high), 75 pass.

**Line-ups and why styles play different numbers of hands.** By default each seat's style is drawn at random every hand (`random_lineup=true` in `format.txt`). Over 20,000 hands (80,000 seats) the styles therefore get roughly, not exactly, 20,000 seats each; in `run20k` the counts were 19,838 to 20,109, within about 1%, which is ordinary random variation. This is deliberate: varied line-ups (for example three aggressive players against one cautious one) teach the model to handle any mix of opponents. When comparing styles from this data, use average points per hand, never totals, since the average adjusts for the count. For a fair head-to-head comparison use `dsy tournament`, which gives every style the same deals from every seat; for a dataset with exactly equal counts, generate with `--fixed-lineup`.

`oracle`, `waits` and `shanten` are the targets for the opponent-reading heads ("is the left player ready, and on what?"). `format.txt` in each dataset records `obs_version` (currently 2); bump it whenever the layout changes.

## Analysis database (PostgreSQL)

Training reads the `.npy` shards directly. For analysis and reporting, hand-level and decision-level summaries go into PostgreSQL:

| Table / view | One row per | Contents |
|---|---|---|
| `runs` | data-generation run | seed, bots and their weights, observation version, engine commit, shard path |
| `hands` | hand | deal seed (replays the hand exactly), dealer, wind, winner, discarder, tai, turns |
| `hand_seats` | player per hand | bot style, seat wind, points, won, dealt in |
| `hand_patterns` | tai pattern per win | pattern name, tai |
| `decisions` | decision | phase, action, tile, legal options, wall count, own shanten, opponents ready, safe / unsafe options, tai the chosen discard gives away |
| `v_style_summary`, `v_pattern_frequency`, `v_defence` | | ready-made summaries |

```bash
docker compose -f db/docker-compose.yml up -d        # or any PostgreSQL 13+; set DATABASE_URL
pip install numpy pandas "psycopg[binary]" matplotlib jupyter

dsy selfplay --hands 20000 --out data/run20k
python db/load_selfplay.py data/run20k               # --replace to reload, --no-decisions for hands only
psql postgresql://dsy:dsy@localhost:5432/dasanyuan -c "select * from v_style_summary"
```

Size guide: 20,000 hands is about 1.2 million decision rows and roughly 300 MB in PostgreSQL. Free hosted tiers (about 0.5 GB) fit hand-level tables comfortably; use `--no-decisions` there.

`notebooks/01_selfplay_eda.ipynb` turns a loaded run into the baseline charts and key findings (set `DSY_RUN` to pick a run). Exported figures are in `notebooks/figures/`.

## Easy-to-read hand table

For browsing results in Excel or pgAdmin, there is a plain-language version of the hands:

```bash
python tools/readable_hands.py data/run20k     # writes data/run20k/hands_readable.csv
```

In PostgreSQL the same table is the view `v_hands_readable` (`SELECT * FROM v_hands_readable WHERE run_id = 1 LIMIT 20;`).

| Column | Meaning |
|---|---|
| Hand | Hand number, starting at 1 |
| Round wind | The prevailing wind of the round: East, South, West, then North (changes every 4 hands) |
| East player (dealer) … North player | The play style sitting at each wind: Fast, High-tai, Defensive or Balanced. The dealer always sits East, and seats move every hand |
| Result | Won on a discard, Self-drawn win, or Draw (no winner) |
| Winner | Who won, as wind and style, e.g. "South (Balanced)". The style alone is not enough because two players can share a style |
| Threw the winning tile | Who discarded the tile the winner took, and so pays for everyone. "Nobody (self-drawn)" when the winner drew it themselves |
| Tai / Tai before 5-tai cap | Hand value as paid, and before the 5-tai limit |
| Where the tai came from | Each scoring pattern and its tai, e.g. "Men qing (fully concealed) +1, Animal x2 +2" |
| East points … North points | Points each player gained (+) or paid (−) this hand; they always add up to 0 |
| Turns | How many turns the hand lasted |

The original `shard_*/hands.csv` files stay as they are: the loader and training read them, and they keep the exact seed for replaying each hand. Don't save `hands.csv` from Excel, which rounds the seeds.

## Replay viewer

`web/replay/index.html` plays one full game (hands 17–32 of the dataset, East round to North round) on its own, with an action log and running points by play style. It replays real hands move by move and shows, for every decision, the training row being recorded: the 672-number snapshot, the allowed moves and the one chosen, the hidden answers used to train the opponent reader, and the points added as the reward when the hand ends. Open it in any browser.

The hands are reproduced exactly (same line-up, same bot seeds), so each one matches its rows in `hands.csv` and the database. To show other hands:

```bash
dsy trace --hands 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 --out web/replay/replay.json   # 0-based hand ids
python web/replay/build.py
```

## Scenario bank

`scenarios/*.txt` holds hand-built positions with checkable expectations. The format is documented at the top of `engine/src/scenario.rs`. Expectations include `safe` (the discard must not deal in, checked against the real hidden hands), `max_ukeire`, `win`, `pass`, `any_of`, `none_of` and rule checks. `trap: yes` makes the validator confirm that the most efficient discard deals in, so a bot can't pass a defence scenario by playing normally.

Current baselines (pass rate over 20 runs):

| Skill | random | efficiency only | fast | high_tai | defensive | balanced |
|---|---|---|---|---|---|---|
| defence | 89% | 20% | 60% | 80% | 80% | 80% |
| push (don't over-fold) | 10% | 100% | 93% | 100% | 100% | 100% |
| tai planning | 65% | 67% | 67% | 100% | 67% | 67% |
| rules | 100% | 100% | 100% | 100% | 100% | 100% |

Read defence and push together: random discards look safe because only one tile is dangerous, but random play fails everything else. Add scenarios freely; `cargo test` validates them all.

## How this fits the training plan

1. **Heuristic archetypes** (this repo): four styles that only use visible information.
2. **Evolution** (`dsy evolve`): tune style weights by duplicate-format tournaments.
3. **Behaviour cloning**: train a network on `obs` → `action` from the best bots' games, plus opponent-reading heads on `waits` / `shanten` / `oracle`.
4. **PPO self-play league** (the reinforcement learning) using `dasanyuan.Env`: start from the cloned network; play against past versions and the heuristic bots; reward = points.
5. **Explanations**: afterstate values per discard + opponent-reading predictions + exact tile odds.

Details in [docs/DESIGN.md](docs/DESIGN.md).

## Performance

About 850 hands/s on 2 cloud CPU threads with heuristic bots and full label logging. The Python environment alone runs about 178,000 decisions/s on one thread. Scaling is roughly linear with cores, so a 16-thread desktop should do several thousand hands per second. Run `dsy bench` to measure. In the duplicate format, 5,000 walls give a standard error of about 0.08 points per hand.

## Sample results (12,000 duplicate hands)

| Style | Points/hand (± se) | Win % | Deal-in % |
|---|---|---|---|
| high_tai | +1.72 ± 0.13 | 32.6 | 15.7 |
| balanced | −0.38 ± 0.12 | 20.6 | 15.8 |
| fast | −0.54 ± 0.12 | 21.1 | 18.6 |
| defensive | −0.80 ± 0.11 | 16.2 | 13.1 |

These are hand-tuned starting points, not strong play. Beating them, and then beating the behaviour-cloned model, is how the RL stage proves itself.
