"""
NetworkBot: a trained network decides the discards; everything else follows
GreedyBot's fixed rules (win when allowed, always declare own kongs, pong/kong
only valuable tiles or when it lowers shanten, never chow). See CLAUDE.md:
"Network decides discards only at first".

    bot = NetworkBot("bc_model.pt")
    action = bot.choose_action(view)

It also counts how often its discard matches what GreedyBot would have thrown
in the same spot (bot.agreement()), which is a direct check of behavioural cloning.
"""

import torch

from bots.greedy_bot import GreedyBot
from engine.game import Action
from ml.encode import encode
from ml.model import DiscardNet


class NetworkBot:
    name = "net"

    def __init__(self, model_path, device="cpu"):
        checkpoint = torch.load(model_path, map_location=device)
        self.model = DiscardNet(checkpoint["num_planes"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()                      # switch off training-only behaviour
        self.device = device
        self.greedy = GreedyBot()              # used for wins, kongs and claims
        self.discards = 0
        self.same_as_greedy = 0

    def choose_action(self, view):
        action = self.greedy.choose_action(view)
        if action.kind != "discard":
            return action                      # a win, a kong, a claim or a pass: fixed rules
        state, mask = encode(view)
        with torch.no_grad():                  # we're only using the network, not training it
            logits = self.model(state[None].float(), mask[None])
        tile = int(logits.argmax(dim=1))       # the network's favourite legal tile
        self.discards += 1
        self.same_as_greedy += tile == action.tile
        return Action("discard", tile)

    def agreement(self):
        """Share of discards where the network threw the same tile GreedyBot would have."""
        return self.same_as_greedy / self.discards if self.discards else 0.0
