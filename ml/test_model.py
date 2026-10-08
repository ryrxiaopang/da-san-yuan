"""
Tests for the network, NetworkBot and the training script.

Run:  python -m pytest ml -q
"""

import os
import subprocess
import sys

import torch

from bots.network_bot import NetworkBot
from engine.game import Game, TableState
from engine.view import view_for
from ml.encode import NUM_PLANES
from ml.model import DiscardNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_network_never_prefers_an_illegal_tile():
    torch.manual_seed(0)
    net = DiscardNet(NUM_PLANES)
    states = torch.rand(8, NUM_PLANES, 34)
    masks = torch.zeros(8, 34, dtype=torch.bool)
    masks[:, [3, 10, 30]] = True                           # only three legal tiles
    logits = net(states, masks)
    assert logits.shape == (8, 34)
    assert set(logits.argmax(dim=1).tolist()) <= {3, 10, 30}
    probs = torch.softmax(logits, dim=1)
    assert probs[:, ~masks[0]].max() < 1e-6                 # illegal tiles get ~0 probability


def test_network_bot_plays_legal_moves(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "untrained.pt"
    torch.save({"state_dict": DiscardNet(NUM_PLANES).state_dict(), "num_planes": NUM_PLANES}, path)
    bots = [NetworkBot(str(path)) for _ in range(4)]       # an untrained network: random-ish discards
    table = TableState()
    for h in range(3):
        game = Game(dealer=table.dealer, prevailing=table.prevailing, seed=h)
        while not game.is_over():
            seat = game.to_act()
            view = view_for(game, seat)
            action = bots[seat].choose_action(view)
            assert action in view.legal
            game.apply(seat, action)
        table.next_hand(game.result, game.any_kong())
    assert bots[0].discards > 0


def test_training_end_to_end(tmp_path):
    """Collect a few hands, train for one epoch, and load the result into NetworkBot."""
    data = tmp_path / "tiny.pt"
    model = tmp_path / "tiny_model.pt"
    subprocess.run([sys.executable, os.path.join(ROOT, "data", "collect_data.py"),
                    "--games", "20", "--out", str(data)], check=True, capture_output=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "ml", "train_bc.py"), str(data),
                    "--epochs", "1", "--out", str(model)], check=True, capture_output=True, cwd=tmp_path)
    bot = NetworkBot(str(model))
    assert bot.model is not None
