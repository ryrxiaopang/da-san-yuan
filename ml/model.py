"""
The neural networks. Kept in one place so training, the bots and (later) RL all
build exactly the same network.
"""

import torch
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


# --------------------------------------------------------------------------
# Convolutional model
# --------------------------------------------------------------------------
class _ResBlock(nn.Module):
    """
    One residual block. Suited tiles: a window of 3 neighbouring tiles slides along each
    suit (shared by all three suits). Honours: each tile on its own (they can't form runs).
    A summary of the whole hand is added to every tile first, so suits can "talk".
    The block's result is ADDED to its input (that's what makes it "residual").
    """

    def __init__(self, channels):
        super().__init__()
        self.summary = nn.Linear(channels, channels)
        self.suit_conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.suit_conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.honour_conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.honour_conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, suits, honours):
        # suits: [batch * 3, channels, 9]    honours: [batch, channels, 7]
        batch, channels = honours.shape[0], honours.shape[1]
        everything = torch.cat([suits.reshape(batch, 3, channels, 9).permute(0, 2, 1, 3).reshape(batch, channels, 27),
                                honours], dim=2)
        summary = self.relu(self.summary(everything.mean(dim=2)))          # [batch, channels]

        s = suits + summary.repeat_interleave(3, dim=0)[:, :, None]
        s = self.suit_conv2(self.relu(self.suit_conv1(s)))
        h = honours + summary[:, :, None]
        h = self.honour_conv2(self.relu(self.honour_conv1(h)))
        return self.relu(suits + s), self.relu(honours + h)


class DiscardResNet(nn.Module):
    """
    Convolutional ResNet over the suit layout.

    Input:  game state [batch, P, 34]   (same grid as DiscardNet)
    Output: one score per tile type [batch, 34], illegal tiles masked
    """

    def __init__(self, num_planes, channels=64, num_blocks=6):
        super().__init__()
        self.channels = channels
        self.suit_in = nn.Conv1d(num_planes, channels, kernel_size=3, padding=1)
        self.honour_in = nn.Conv1d(num_planes, channels, kernel_size=1)
        # The whole grid, flattened, gives every tile the table-wide information
        # (winds, wall, bonus tiles, dealer), which isn't laid out tile by tile.
        self.table_in = nn.Linear(num_planes * NUM_TILE_TYPES, channels)
        self.blocks = nn.ModuleList([_ResBlock(channels) for _ in range(num_blocks)])
        self.suit_out = nn.Conv1d(channels, 1, kernel_size=1)
        self.honour_out = nn.Conv1d(channels, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, state, legal_mask):
        batch, planes = state.shape[0], state.shape[1]
        # [batch, P, 27] -> three suits of 9, stacked as separate rows: [batch * 3, P, 9]
        suits = state[:, :, :27].reshape(batch, planes, 3, 9).permute(0, 2, 1, 3).reshape(batch * 3, planes, 9)
        honours = state[:, :, 27:34]
        table = self.relu(self.table_in(state.flatten(1)))                  # [batch, channels]

        s = self.relu(self.suit_in(suits) + table.repeat_interleave(3, dim=0)[:, :, None])
        h = self.relu(self.honour_in(honours) + table[:, :, None])
        for block in self.blocks:
            s, h = block(s, h)

        suit_logits = self.suit_out(s).reshape(batch, 27)                   # back to 1m..9s order
        honour_logits = self.honour_out(h).reshape(batch, 7)
        logits = torch.cat([suit_logits, honour_logits], dim=1)
        return logits.masked_fill(~legal_mask, -1e9)


MODELS = {"mlp": DiscardNet, "resnet": DiscardResNet}


def build_model(model_type, num_planes):
    """Make a network by name: "mlp" (DiscardNet) or "resnet" (DiscardResNet)."""
    return MODELS[model_type](num_planes)
