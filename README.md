# da-san-yuan 大三元

Singapore 4-player mahjong engine, heuristic bots and self-play data generator. It is the foundation for a learning platform that advises players on the best move and explains why.

This repo currently contains **stage 1 of the training pipeline**: a fast, tested rules engine, four heuristic bots, a self-play logger that writes training data, a duplicate-format tournament for comparing bots fairly, an evolution loop, and Python bindings for the RL work.

The exact ruleset is in **[RULES.md](RULES.md)**. Please check the "confirm with your table" section there before generating large datasets.

## Layout

```
engine/         Rust library: rules, scoring, shanten, bots, self-play
  src/tile.rs       tile ids and notation (1m..9m, 1p.., 1s.., 1z..7z, f1-f4, g1-g4, a1-a4)
  src/game.rs       state machine for one hand (deal, draws, claims, kongs, robbing, draw game)
  src/scoring.rs    win decomposition, tai patterns, payments
  src/shanten.rs    table-based shanten and useful-tile counts
  src/obs.rs        action space (76) and observation encoding (528 bytes)
  src/bots.rs       heuristic bots: fast, high_tai, defensive, balanced
  src/selfplay.rs   data logger, duplicate tournament
  tests/rules.rs    rule tests (one per scoring rule) + engine invariants
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

# Generate training data (all cores). 100k hands is about 600 MB compressed.
dsy selfplay --hands 100000 --out data/selfplay

# Fair comparison: every wall is played 4 times with the bots rotated through all seats.
dsy tournament --styles fast,high_tai,defensive,balanced --walls 5000

# Evolution: champion vs 3 mutated copies of itself; a challenger is promoted only if it
# beats the champion by more than 2 standard errors.
dsy evolve --generations 30 --walls 3000 --start balanced --log data/evolve.csv

dsy bench                       # throughput on this machine
```

Styles can also be given as CSV lines (`name,speed,value,defence,claim,flush,pongs`), so a champion from `evolve.csv` can be fed straight back into `selfplay` or `tournament`.

## Python

```python
import dasanyuan as dsy

env = dsy.Env(seed=1)
while not env.is_over():
    seat = env.to_act()
    obs, mask = env.observe(seat)            # uint8 [528], uint8 [76]
    action = env.bot_action(seat, "balanced")  # or your policy's choice
    env.step(seat, action)
print(env.result())   # winner, tai, patterns, point changes

data = dsy.load_shards("data/selfplay")      # obs, mask, action, oracle, meta
```

`python examples/inspect_data.py data/selfplay` prints the action mix, results per bot, and tai distribution.

## Data format

Each `shard_XXXXX/` directory holds gzip-compressed numpy arrays (`--no-compress` for plain `.npy`):

| File | Shape | Meaning |
|---|---|---|
| `obs` | [N, 528] u8 | What the acting player can see. Layout documented in `engine/src/obs.rs` |
| `mask` | [N, 76] u8 | Legal actions |
| `action` | [N] u8 | Action taken |
| `oracle` | [N, 102] u8 | The other three players' hidden hands. **Training target only**, never a policy input |
| `meta` | [N, 6] i32 | hand_id, seat, bot_id, final points for this seat, won, dealt in |
| `hands.csv` | per hand | seed, dealer, bots, winner, tai, patterns, point changes |

Seats in `obs` and `oracle` are relative: 0 = you, 1 = next player, 2 = opposite, 3 = previous. Actions: 0–33 discard, 34–67 kong, 68 tsumo, 69 ron, 70 pong, 71 exposed kong, 72–74 chow (claimed tile low/middle/high), 75 pass.

`oracle` is there so you can train the opponent-reading head ("what is the left player waiting on?") and try oracle guiding later.

## How this fits the training plan

1. **Heuristic archetypes** (this repo): four styles that only use visible information.
2. **Evolution** (`dsy evolve`): tune style weights by duplicate-format tournaments.
3. **Behaviour cloning**: train a network on `obs` → `action` from the best bots' games, plus an opponent-hand head on `oracle`.
4. **PPO self-play league** using `dasanyuan.Env`: start from the cloned network; play against past versions and the heuristic bots; reward = points.
5. **Explanations**: value estimates per legal action + opponent-wait predictions + exact tile odds.

## Performance

About 950 hands/s on 2 cloud CPU threads with heuristic bots. Scaling is roughly linear with cores, so a 16-thread desktop should do several thousand hands per second. Run `dsy bench` to measure. In the duplicate format, 5,000 walls give a standard error of about 0.08 points per hand.

## Sample results (12,000 duplicate hands)

| Style | Points/hand | Win % | Deal-in % |
|---|---|---|---|
| high_tai | +1.58 | 32.7 | 15.7 |
| balanced | −0.28 | 20.8 | 15.8 |
| fast | −0.56 | 21.1 | 18.7 |
| defensive | −0.73 | 16.3 | 13.3 |

These are hand-tuned starting points, not strong play. Beating them is the first milestone for the learned models.
