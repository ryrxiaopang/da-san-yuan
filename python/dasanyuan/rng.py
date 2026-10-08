"""Small, fast, seedable RNG (xoshiro256++ seeded through splitmix64).

Kept dependency-free so a seed reproduces the exact same deal on every machine.
All arithmetic is masked to 64 bits so the stream matches the original engine bit for bit.
"""
from __future__ import annotations

from typing import List

MASK = (1 << 64) - 1


def _splitmix(x: int):
    """One splitmix64 step: returns (new state, output)."""
    x = (x + 0x9E3779B97F4A7C15) & MASK
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return x, z ^ (z >> 31)


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & MASK


class Rng:
    __slots__ = ("s",)

    def __init__(self, seed: int):
        x = seed & MASK
        s = []
        for _ in range(4):
            x, z = _splitmix(x)
            s.append(z)
        self.s = s

    def clone(self) -> "Rng":
        r = Rng.__new__(Rng)
        r.s = list(self.s)
        return r

    def next_u64(self) -> int:
        s0, s1, s2, s3 = self.s
        result = (_rotl((s0 + s3) & MASK, 23) + s0) & MASK
        t = (s1 << 17) & MASK
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        s3 = _rotl(s3, 45)
        self.s = [s0, s1, s2, s3]
        return result

    def below(self, n: int) -> int:
        """Uniform integer in 0..n (n > 0). Lemire's multiply-shift."""
        return (self.next_u64() * n) >> 64

    def unit(self) -> float:
        """Uniform float in [0, 1)."""
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def shuffle(self, v: List) -> None:
        for i in range(len(v) - 1, 0, -1):
            j = self.below(i + 1)
            v[i], v[j] = v[j], v[i]

    @staticmethod
    def derive(seed: int, stream: int) -> "Rng":
        """Derive an independent stream, e.g. one per hand in a batch."""
        x = (seed ^ ((stream * 0xD1B54A32D192ED03) & MASK)) & MASK
        x, _ = _splitmix(x)
        _, z = _splitmix(x)
        return Rng(z)
