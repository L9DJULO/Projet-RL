import numpy as np
import torch

from mcts_rl.games import make_game
from mcts_rl.nn import NetConfig, PolicyValueNet
from mcts_rl.selfplay import TrainConfig, self_play_games, train_on_buffer


def _tiny(game):
    return PolicyValueNet(NetConfig.for_game(make_game(game), channels=8, blocks=1)).eval()


def test_self_play_samples_are_consistent():
    cfg = TrainConfig(game="connect4", n_simulations=8)
    samples, outcomes, lengths = self_play_games(_tiny("connect4"), cfg, n_games=3, seed=0)
    assert len(outcomes) == 3
    assert len(samples) == 2 * sum(lengths)  # symétrie gauche/droite
    for planes, idx, probs, z in samples:
        assert planes.shape == (3, 6, 7)
        assert abs(probs.sum() - 1) < 1e-5 and (idx < 7).all()
        assert z in (-1.0, 0.0, 1.0)


def test_value_target_sign():
    # le dernier coup d'une partie gagnée est joué par le vainqueur -> z = +1
    cfg = TrainConfig(game="connect4", n_simulations=4, temp_moves=100)
    samples, outcomes, lengths = self_play_games(_tiny("connect4"), cfg, n_games=1, seed=3)
    if outcomes[0] != 0:
        last = samples[2 * lengths[0] - 2]  # dernier exemple (avant sa copie symétrique)
        assert last[3] == 1.0


def test_training_reduces_loss_on_fixed_buffer():
    torch.manual_seed(0)
    cfg = TrainConfig(game="connect4", n_simulations=8)
    net = _tiny("connect4")
    samples, _, _ = self_play_games(net, cfg, n_games=4, seed=1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    rng = np.random.default_rng(0)
    first = train_on_buffer(net, opt, samples, 1, 64, rng)["loss"]
    last = train_on_buffer(net, opt, samples, 60, 64, rng)["loss"]
    assert last < first


def test_chess_self_play_runs():
    cfg = TrainConfig(game="chess-micro", n_simulations=2)
    samples, outcomes, _ = self_play_games(_tiny("chess-micro"), cfg, n_games=1, seed=0)
    assert len(outcomes) == 1 and samples[0][0].shape == (17, 8, 8)
