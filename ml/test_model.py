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


def _without_cross_suit_paths(model):
    """Switch off the two parts that share information across the whole hand on purpose
    (the table input and the per-block hand summary), leaving only the convolutions."""
    with torch.no_grad():
        model.table_in.weight.zero_(); model.table_in.bias.zero_()
        for block in model.blocks:
            block.summary.weight.zero_(); block.summary.bias.zero_()
    return model


def test_resnet_window_never_crosses_suits_or_honours():
    from ml.model import DiscardResNet
    torch.manual_seed(0)
    model = _without_cross_suit_paths(DiscardResNet(NUM_PLANES))
    mask = torch.ones(1, 34, dtype=torch.bool)
    x = torch.rand(1, NUM_PLANES, 34)
    y = x.clone()
    y[:, :, 8] += 1.0                                     # change only 9m
    a, b = model(x, mask), model(y, mask)
    changed = (a - b).abs()[0] > 1e-6
    assert changed[0:9].any()                             # characters may change ...
    assert not changed[9:].any()                          # ... but no dots, bamboo or honour score may
    z = x.clone()
    z[:, :, 27] += 1.0                                    # change only East
    changed = (model(x, mask) - model(z, mask)).abs()[0] > 1e-6
    assert changed.tolist() == [t == 27 for t in range(34)]   # only East's own score moves


def test_resnet_shares_its_window_across_suits():
    """The same 1x3 window is used for every suit: swapping two suits swaps their scores."""
    from ml.model import DiscardResNet
    torch.manual_seed(0)
    model = DiscardResNet(NUM_PLANES)
    with torch.no_grad():
        model.table_in.weight.zero_(); model.table_in.bias.zero_()
    x = torch.rand(2, NUM_PLANES, 34)
    y = x.clone()
    y[:, :, 0:9], y[:, :, 9:18] = x[:, :, 9:18], x[:, :, 0:9]
    mask = torch.ones(2, 34, dtype=torch.bool)
    a, b = model(x, mask), model(y, mask)
    assert torch.allclose(a[:, 0:9], b[:, 9:18], atol=1e-5) and torch.allclose(a[:, 9:18], b[:, 0:9], atol=1e-5)


def test_models_stay_small_enough_for_the_website():
    from ml.model import build_model
    for name in ("mlp", "resnet"):
        size = sum(p.numel() for p in build_model(name, NUM_PLANES).parameters())
        assert size < 2_000_000, f"{name} has {size:,} weights"
