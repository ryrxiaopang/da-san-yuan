# Singapore Mahjong RL

A bot that learns Singapore mahjong: first by copying a rule-based greedy bot (behavioural cloning), then by improving through self-play reinforcement learning.

- **The plan, decisions and conventions:** [CLAUDE.md](CLAUDE.md)
- **The rules the engine follows:** [RULES.md](RULES.md)
- **The previous Rust-based version** is on the `old` branch.

## Setup

```bash
pip install -r requirements.txt
python -m pytest engine bots ml -q   # should print "70 passed"
```

## Try the engine

```python
from engine.game import Game, WIN
import random

game = Game(dealer=0, prevailing=0, seed=1)
rng = random.Random(1)
while not game.is_over():
    seat = game.to_act()
    options = game.legal_actions(seat)
    game.apply(seat, WIN if WIN in options else rng.choice(options))
print(game.result)
```

## Progress

| Stage | Status |
|---|---|
| 1. Game engine | Built and tested |
| 2. Rule-based bots | Built and tested |
| 3. Data collection | Built and tested |
| 4. Behavioural cloning | Next |
| 5. Reinforcement learning | |
| 6. Evaluation | |
