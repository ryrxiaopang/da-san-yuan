"""
RandomBot: the lowest benchmark, and a good way to stress-test the engine.

It wins whenever the rules allow it (otherwise it would almost never win at all),
and otherwise picks any legal move at random: random discards, random claims,
random kongs.
"""

import random

from engine.game import WIN


class RandomBot:
    name = "random"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def choose_action(self, view):
        if WIN in view.legal:
            return WIN
        return self.rng.choice(view.legal)
