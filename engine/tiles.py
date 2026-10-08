"""
Tiles: how every tile is numbered, named and counted.

Playing tiles (34 types, 4 copies each = 136 tiles):
    0-8    1m-9m   characters (万)
    9-17   1p-9p   dots (筒)
    18-26  1s-9s   bamboo (条)
    27-30  E S W N winds
    31-33  Rd Gr Wh dragons (Red, Green, White)

Bonus tiles (1 copy each = 12 tiles). They are never part of a hand: a player who
draws one puts it face up and draws a replacement.
    34-37  f1-f4   seasons
    38-41  g1-g4   plants (flowers)
    42-45  a1-a4   animals: cat, rat, rooster, centipede

Total: 148 tiles in the wall.
"""

NUM_TILE_TYPES = 34          # playing tile types
NUM_BONUS = 12               # bonus tiles
WALL_SIZE = 148              # 136 playing tiles + 12 bonus tiles

EAST, SOUTH, WEST, NORTH = 27, 28, 29, 30
RED, GREEN, WHITE = 31, 32, 33
WINDS = (EAST, SOUTH, WEST, NORTH)
DRAGONS = (RED, GREEN, WHITE)

FIRST_BONUS = 34
SEASONS = range(34, 38)      # f1-f4
PLANTS = range(38, 42)       # g1-g4
ANIMALS = range(42, 46)      # a1-a4

# The 13 tiles of Thirteen Orphans: every 1 and 9, every wind, every dragon.
ORPHANS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)

HONOUR_NAMES = ["E", "S", "W", "N", "Rd", "Gr", "Wh"]
ANIMAL_NAMES = ["cat", "rat", "rooster", "centipede"]


# --------------------------------------------------------------------------
# Simple questions about one tile
# --------------------------------------------------------------------------
def is_bonus(tile):
    return tile >= FIRST_BONUS


def is_suited(tile):
    """Characters, dots or bamboo (the tiles that can form runs)."""
    return tile < 27


def is_honour(tile):
    return 27 <= tile <= 33


def is_wind(tile):
    return tile in WINDS


def is_dragon(tile):
    return tile in DRAGONS


def suit(tile):
    """0 = characters, 1 = dots, 2 = bamboo, None for honours and bonus tiles."""
    return tile // 9 if tile < 27 else None


def number(tile):
    """The number printed on a suited tile, 1 to 9."""
    return tile % 9 + 1


def wind_tile(wind):
    """Wind index (0 = East, 1 = South, 2 = West, 3 = North) -> tile number."""
    return EAST + wind


# --------------------------------------------------------------------------
# The full set and counting
# --------------------------------------------------------------------------
def full_set():
    """All 148 tiles: 4 copies of each playing tile, then the 12 bonus tiles."""
    tiles = []
    for t in range(NUM_TILE_TYPES):
        tiles += [t] * 4
    tiles += list(range(FIRST_BONUS, FIRST_BONUS + NUM_BONUS))
    return tiles


def counts_of(tiles):
    """List of tiles -> how many of each of the 34 playing tile types."""
    c = [0] * NUM_TILE_TYPES
    for t in tiles:
        if not is_bonus(t):
            c[t] += 1
    return c


# --------------------------------------------------------------------------
# Names, for printing and for writing tests
# --------------------------------------------------------------------------
def tile_name(tile):
    if tile < 27:
        return f"{number(tile)}{'mps'[suit(tile)]}"
    if tile <= 33:
        return HONOUR_NAMES[tile - 27]
    if tile in SEASONS:
        return f"f{tile - 33}"
    if tile in PLANTS:
        return f"g{tile - 37}"
    return f"a{tile - 41}"


def hand_name(tiles):
    """Sorted, readable list of tiles, e.g. '1m 2m 3m E E'."""
    return " ".join(tile_name(t) for t in sorted(tiles))


def parse_tiles(text):
    """
    Turn text into tile numbers. Examples:
        "123m 456p 99s"   -> nine suited tiles
        "E E E Rd Rd"     -> honours by name
        "f1 g2 a3"        -> bonus tiles
    Groups and names can be mixed, separated by spaces.
    """
    tiles = []
    for word in text.split():
        if word in HONOUR_NAMES:
            tiles.append(27 + HONOUR_NAMES.index(word))
        elif word[0] in "fga" and len(word) == 2 and word[1].isdigit():
            first = {"f": 34, "g": 38, "a": 42}[word[0]]
            tiles.append(first + int(word[1]) - 1)
        elif word[-1] in "mps" and word[:-1].isdigit():
            base = {"m": 0, "p": 9, "s": 18}[word[-1]]
            tiles += [base + int(d) - 1 for d in word[:-1]]
        else:
            raise ValueError(f"can't read tile(s) {word!r}")
    return tiles
