"""
The neural networks. Kept in one place so training, the bots and (later) RL all
build exactly the same network.
"""

import torch.nn as nn

NUM_TILE_TYPES = 34


class DiscardNet(nn.Module):
    """
    A simple MLP ("multi-layer perceptron": layers of numbers connected to each other).

    Input:  game state [batch, P, 34]   (P = 49 lines, see ml/encode.py)
    Output: one score ("logit") per tile type [batch, 34]; higher = more likely to discard
    """

    def __init__(self, num_planes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),                                   # 49 x 34 grid -> one row of 1,666 numbers
            nn.Linear(num_planes * NUM_TILE_TYPES, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_TILE_TYPES),
        )

    def forward(self, state, legal_mask):
        logits = self.layers(state)
        # Action masking: tiles you can't discard get probability ~0.
        return logits.masked_fill(~legal_mask, -1e9)
