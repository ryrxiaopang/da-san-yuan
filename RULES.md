# Rules encoded in the engine

This is the exact ruleset the engine, bots and training data follow. If your table plays differently, change the rule here **and** the matching test in `engine/tests/rules.rs`, then regenerate any data.

## Tiles (148)

| Group | Tiles | Copies | Notation |
|---|---|---|---|
| Characters 1–9 | 万 | 4 each | `1m`–`9m` |
| Dots 1–9 | 筒 | 4 each | `1p`–`9p` |
| Bamboo 1–9 | 条 | 4 each | `1s`–`9s` |
| Winds E S W N | 东南西北 | 4 each | `1z`–`4z` |
| Dragons White Green Red | 白发中 | 4 each | `5z`–`7z` |
| Seasons 1–4 | 春夏秋冬 | 1 each | `f1`–`f4` |
| Plants 1–4 | 梅兰菊竹 | 1 each | `g1`–`g4` |
| Animals | cat, rat, rooster, centipede | 1 each | `a1`–`a4` |

## Play

- Dealer (East) gets 14 tiles, others 13. Bonus tiles are set aside and replaced from the back of the wall, dealer first.
- A drawn bonus tile is set aside and replaced from the back immediately.
- Claims on a discard: **win > pong / kong > chow**. Only the next player may chow. If several players can win on the same discard, the one nearest after the discarder wins.
- Kongs: concealed kong (4 in hand), exposed kong (from a discard, holding 3), added kong (4th tile onto your own pong). Every kong draws a replacement from the back. An added kong can be **robbed** by a player who wins on that tile.
- After a pong or chow you must discard straight away.
- The hand is a **draw** when 15 live tiles remain (`Config::reserve`). No points change.
- Minimum **1 tai** to win, maximum **5 tai**.

## Tai

| Pattern | Tai | Notes |
|---|---|---|
| Flower matching your seat | 1 | Per set (seasons, plants). Seat East = 1, South = 2, West = 3, North = 4 |
| Complete flower set (all 4 of one set) | 2 | Replaces that set's matching-flower tai |
| Animal | 1 each | |
| Dragon pong / kong | 1 each | |
| Seat wind pong / kong | 1 | |
| Prevailing wind pong / kong | 1 | Stacks with seat wind when they are the same wind |
| Men qing (no pong / chow / exposed kong) | 1 | Concealed kongs allowed. Applies to discard and self-draw wins |
| All pongs (pong pong hu) | 2 | |
| Half flush (one suit + honours) | 2 | |
| Ping hu, no bonus tiles | 4 | See definition below |
| Ping hu with only non-scoring flowers | 1 | Flowers that don't match your seat; no animals |
| Small three dragons | 4 | Two dragon pongs + dragon pair; replaces the dragon pong tai |
| Small four winds | 4 | Three wind pongs + wind pair; replaces the wind pong tai |
| Win on kong replacement tile | 1 | |
| Robbing a kong | 1 | |
| Win on the last tile | 1 | Last live draw, or the discard after it |
| **Full flush** (one suit, no honours) | **5** | Limit |
| Thirteen orphans, big three dragons, big four winds, all honours, four concealed pongs, four kongs, nine gates, heavenly hand, earthly hand | 5 | Limit |

Tai from every pattern is added up, then capped at 5. When a hand can be read more than one way, the highest-scoring reading counts.

**Ping hu** means all four sets are chows, the pair is not a dragon, your seat wind or the prevailing wind, and the winning tile completed a chow from a two-sided wait (not an edge, middle or pair wait). Exposed chows are allowed. If you hold any scoring bonus tile (a matching flower or an animal), ping hu is not counted and the bonus tiles score instead.

**Four concealed pongs**: a pong completed by someone else's discard counts as exposed, so it needs a self-draw or a pair wait.

## Points (no money: points only)

Unit price doubles per tai: 1, 2, 4, 8, 16 for 1–5 tai.

| Tai | Discard win: discarder pays for everyone | Self-draw: each other player pays |
|---|---|---|
| 1 | 4 | 2 |
| 2 | 8 | 4 |
| 3 | 16 | 8 |
| 4 | 32 | 16 |
| 5 | 64 | 32 |

## Not implemented (can be switched on later)

- Instant animal payments ("bites": cat + rat, rooster + centipede) and other instant flower payments.
- Bao / penalty rules for feeding a big hand.
- Seven pairs is not a winning hand.
- Self-draw itself adds no tai.

## Please confirm with your table

These were judgement calls. Each one changes what the bots learn:

1. **Men qing stacks with ping hu**, so a fully concealed, clean ping hu is 4 + 1 = 5 tai. Some tables treat ping hu as already including concealment.
2. Ping hu is allowed with **exposed chows**.
3. The draw happens with **15 tiles** left.
4. In generated data the **dealer rotates every hand** and the prevailing wind changes every 4 hands, so all seats and winds appear evenly. Real games keep the dealer after a dealer win; that only matters for match-level play, not for single-hand decisions.
