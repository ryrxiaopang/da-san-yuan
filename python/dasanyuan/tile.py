"""Tile encoding.

Playing tiles (34 kinds, 4 copies each = 136):
  0..=8   characters 1-9  (m, 万)
  9..=17  dots 1-9        (p, 筒)
  18..=26 bamboo 1-9      (s, 条)
  27..=30 winds E S W N   (1z-4z)
  31..=33 dragons White Green Red (5z 白, 6z 发, 7z 中)

Bonus tiles (one copy each = 12):
  34..=37 seasons 1-4  (春夏秋冬), notation f1-f4
  38..=41 plants 1-4   (梅兰菊竹), notation g1-g4
  42..=45 animals      cat, rat, rooster, centipede, notation a1-a4

Total 148 tiles. A hand's concealed tiles are kept as `Counts`: a list of 34 ints.
"""
from __future__ import annotations

from typing import Iterable, List

NUM_KINDS = 34
NUM_TILE_IDS = 46
NUM_BONUS = 12
WALL_SIZE = 148

EAST, SOUTH, WEST, NORTH = 27, 28, 29, 30
WHITE, GREEN, RED = 31, 32, 33

FIRST_BONUS = 34
FIRST_PLANT = 38
FIRST_ANIMAL = 42

# The 13 terminal and honour kinds used by Thirteen Orphans.
ORPHANS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


def is_bonus(t: int) -> bool:
    return t >= FIRST_BONUS


def is_flower(t: int) -> bool:
    return FIRST_BONUS <= t < FIRST_ANIMAL


def is_animal(t: int) -> bool:
    return FIRST_ANIMAL <= t < NUM_TILE_IDS


def flower_number(t: int) -> int:
    """Flower number 1-4 for a season/plant tile."""
    return (t - FIRST_BONUS) % 4 + 1


def flower_set(t: int) -> int:
    """0 = seasons, 1 = plants."""
    return (t - FIRST_BONUS) // 4


def is_honor(t: int) -> bool:
    return 27 <= t < 34


def is_wind(t: int) -> bool:
    return 27 <= t < 31


def is_dragon(t: int) -> bool:
    return 31 <= t < 34


def is_suited(t: int) -> bool:
    return t < 27


def suit(t: int) -> int:
    """0 = characters, 1 = dots, 2 = bamboo, 3 = honours."""
    return t // 9 if t < 27 else 3


def rank(t: int) -> int:
    """Rank 0..=8 within a suit."""
    return t % 9


def is_terminal(t: int) -> bool:
    return t < 27 and (t % 9 == 0 or t % 9 == 8)


def is_terminal_or_honor(t: int) -> bool:
    return is_terminal(t) or is_honor(t)


def wind_tile(w: int) -> int:
    """Wind tile for a wind index 0..=3 (E, S, W, N)."""
    return EAST + w


def tile_name(t: int) -> str:
    if 0 <= t <= 26:
        return f"{t % 9 + 1}{'mps'[t // 9]}"
    if 27 <= t <= 33:
        return f"{t - 26}z"
    if 34 <= t <= 37:
        return f"f{t - 33}"
    if 38 <= t <= 41:
        return f"g{t - 37}"
    if 42 <= t <= 45:
        return f"a{t - 41}"
    return "??"


def parse_tiles(s: str) -> List[int]:
    """Parse a hand string like "123m456p789s1155z f1 a2" into tile ids.

    Bonus tiles are written as f1-f4, g1-g4, a1-a4 separated by spaces.
    Raises ValueError on bad input.
    """
    out: List[int] = []
    for word in s.split():
        if len(word) == 2 and word[0] in "fga" and "1" <= word[1] <= "4":
            n = ord(word[1]) - ord("1")
            out.append({"f": FIRST_BONUS, "g": FIRST_PLANT}.get(word[0], FIRST_ANIMAL) + n)
            continue
        pending: List[int] = []
        for c in word:
            if "0" <= c <= "9":
                pending.append(ord(c) - ord("0"))
            elif c in "mpsz":
                if not pending:
                    raise ValueError(f"suit '{c}' with no digits in '{word}'")
                for d in pending:
                    if c == "z":
                        if not 1 <= d <= 7:
                            raise ValueError(f"bad honour {d}z")
                        out.append(26 + d)
                    else:
                        if not 1 <= d <= 9:
                            raise ValueError(f"bad rank {d}")
                        out.append({"m": 0, "p": 9}.get(c, 18) + d - 1)
                pending.clear()
            else:
                raise ValueError(f"unexpected character '{c}'")
        if pending:
            raise ValueError(f"digits without suit in '{word}'")
    return out


def hand_string(tiles: Iterable[int]) -> str:
    v = sorted(tiles)
    s = ""
    for suit_c in range(4):
        if suit_c == 3:
            digits = "".join(str(t - 26) for t in v if t < 34 and suit(t) == 3)
        else:
            digits = "".join(str(rank(t) + 1) for t in v if t < 34 and suit(t) == suit_c)
        if digits:
            s += digits + "mpsz"[suit_c]
    for t in v:
        if t >= 34:
            s += " " + tile_name(t)
    return s


def counts_of(tiles: Iterable[int]) -> List[int]:
    """Counts of the 34 playing kinds (bonus tiles are ignored)."""
    c = [0] * NUM_KINDS
    for t in tiles:
        if t < NUM_KINDS:
            c[t] += 1
    return c
