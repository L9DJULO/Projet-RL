import numpy as np
import torch

from mcts_rl.agents import AlphaZeroAgent, MCTSAgent, RandomAgent
from mcts_rl.arena import play_game, play_match
from mcts_rl.games import ChessState, Connect4State
from mcts_rl.nn import NetConfig, NetEvaluator, PolicyValueNet


def uniform_evaluator(state):
    legal = state.legal_actions()
    return legal, np.full(len(legal), 1 / len(legal)), 0.0


def test_mcts_takes_immediate_win():
    s = Connect4State.from_moves("010101")  # X joue 0 et gagne
    agent = MCTSAgent(n_simulations=300, seed=0)
    assert agent.select_action(s) == 0


def test_mcts_blocks_immediate_threat():
    s = Connect4State.from_moves("01020")  # O doit bloquer en 0 (X a 3 pierres en colonne 0)
    agent = MCTSAgent(n_simulations=1000, seed=0)
    assert agent.select_action(s) == 0


def test_mcts_beats_random():
    res = play_match({"type": "mcts", "n_simulations": 200}, {"type": "random"}, n_games=10, workers=1)
    assert res.wins >= 9


def test_alphazero_mcts_finds_win_with_uniform_prior():
    s = Connect4State.from_moves("010101")
    agent = AlphaZeroAgent(uniform_evaluator, n_simulations=200, seed=0)
    assert agent.select_action(s) == 0


def test_alphazero_mcts_blocks_with_uniform_prior():
    s = Connect4State.from_moves("01020")
    agent = AlphaZeroAgent(uniform_evaluator, n_simulations=400, seed=0)
    assert agent.select_action(s) == 0


def test_chess_mate_in_one_found():
    import chess
    # Blancs : Dh5xf7# possible (mat du berger)
    b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
    s = ChessState("standard", b)
    agent = MCTSAgent(n_simulations=400, max_rollout_depth=10, seed=0)
    assert s.action_to_str(agent.select_action(s)) == "h5f7"
    az = AlphaZeroAgent(uniform_evaluator, n_simulations=400, seed=0)
    assert s.action_to_str(az.select_action(s)) == "h5f7"


def test_network_shapes():
    for state in (Connect4State(), ChessState("mini")):
        net = PolicyValueNet(NetConfig.for_game(state, channels=16, blocks=1))
        x = torch.from_numpy(np.stack([state.encode()] * 3))
        p, v = net(x)
        assert p.shape == (3, state.ACTION_SIZE) and v.shape == (3,)
        legal, priors, value = NetEvaluator(net)(state)
        assert len(legal) == len(priors) and abs(priors.sum() - 1) < 1e-5 and -1 <= value <= 1


def test_search_stats_display():
    from mcts_rl.arena import format_search_stats
    s = Connect4State.from_moves("010101")
    mcts = MCTSAgent(n_simulations=200, seed=0)
    mcts.select_action(s)
    text = format_search_stats(mcts, s)
    assert "200 visites" in text and "victoire estimée" in text
    az = AlphaZeroAgent(uniform_evaluator, n_simulations=50, seed=0)
    az.select_action(s)
    assert "réseau" in format_search_stats(az, s)


def test_no_stale_stats_on_forced_move():
    s = Connect4State.from_moves("000000111111222222444444555555")  # colonnes 3 et 6 libres
    agent = MCTSAgent(n_simulations=50, seed=0)
    agent.select_action(s)
    assert agent.root_stats()
    s = Connect4State.from_moves("000000111111222222444444555555666666")  # seule la colonne 3 reste
    assert s.legal_actions() == [3]
    agent.select_action(s)
    assert agent.root_stats() == []


def test_play_game_chess_random():
    r = play_game(RandomAgent(0), RandomAgent(1), game="chess-micro")
    assert r["winner"] in (-1, 0, 1)
