"""Single-precision helpers.

The bot heuristics were tuned (and their recorded games generated) with 32-bit floats.
Rounding every intermediate result to f32 keeps their decisions identical, including
which of two near-equal moves wins.
"""
from __future__ import annotations

import math
import struct
from decimal import Decimal

_F = struct.Struct("<f")
_pack = _F.pack
_unpack = _F.unpack


def f32(x: float) -> float:
    """Round a Python float to the nearest single-precision value."""
    return _unpack(_pack(x))[0]


def exp32(x: float) -> float:
    return f32(math.exp(x))


def fmt32(x: float) -> str:
    """Shortest decimal that reads back as the same f32, without exponent ("0.6", "1", "12.5")."""
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if x == 0:
        return "-0" if math.copysign(1.0, x) < 0 else "0"
    for p in range(1, 10):
        s = "%.*e" % (p - 1, x)
        if f32(float(s)) == x:
            break
    return format(Decimal(s), "f")
