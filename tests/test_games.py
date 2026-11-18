import random

import chess
import numpy as np
import pytest

from mcts_rl.games import ChessState, Connect4State, make_game


# ---------------------------------------------------------------- Puissance 4
def test_connect4_vertical_win():
    s = Connect4State.from_moves("0101010")
    assert s.winner() == 1 and s.is_terminal()


def test_connect4_horizontal_win():
    # X : 0,1,2,3 en bas ; O : 3 pierres en colonne 6 -> X gagne au 7e coup
    s = Connect4State.from_moves("0616263")
    assert s.winner() == 1


def test_connect4_second_player_win():
    s = Connect4State()
    for col in [0, 6, 0, 6, 1, 6, 2, 6]:  # X : 0,0,1,2 ; O : 6 x4 -> O gagne
        s = s.apply(col)
    assert s.winner() == -1


def test_connect4_diagonal_wins():
    # diagonale / pour X : (0,0) (1,1) (2,2) (3,3)
    s = Connect4State.from_moves("01122323363")
    assert s.winner() == 1
    # diagonale \ pour X : (3,0) (2,1) (1,2) (0,3)
    s = Connect4State.from_moves("32211010060")
    assert s.winner() == 1


def test_connect4_matches_naive_checker():
    # parties aléatoires : le vainqueur détecté par bitboard doit être confirmé par une vérification naïve
    rng = random.Random(0)
    for _ in range(300):
        s = Connect4State()
        while not s.is_terminal():
            s = s.apply(rng.choice(s.legal_actions()))
        assert _naive_winner(s.to_array()) == s.winner()


def _naive_winner(board):
    rows, cols = board.shape
    for r in range(rows):
        for c in range(cols):
            p = board[r, c]
            if p == 0:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                if all(0 <= r + k * dr < rows and 0 <= c + k * dc < cols and board[r + k * dr, c + k * dc] == p
                       for k in range(4)):
                    return int(p)
    return 0 if (board != 0).all() else None


def test_connect4_full_column_illegal():
    s = Connect4State.from_moves("000000")
    assert 0 not in s.legal_actions()
    with pytest.raises(ValueError):
        s.apply(0)


def test_connect4_encode_perspective():
    s = Connect4State.from_moves("3")
    planes = s.encode()  # au tour de O : la pierre de X est "adverse"
    assert planes[1, 5, 3] == 1 and planes[0].sum() == 0


# ---------------------------------------------------------------- Échecs
def test_chess_standard_has_20_moves():
    s = make_game("chess-standard")
    assert len(s.legal_actions()) == 20


def test_chess_action_roundtrip_both_colors():
    s = make_game("chess-standard")
    rng = random.Random(1)
    for _ in range(40):
        if s.is_terminal():
            break
        for a in s.legal_actions():
            assert s.str_to_action(s.action_to_str(a)) == a
        s = s.apply(rng.choice(s.legal_actions()))


def test_chess_black_moves_are_mirrored():
    s = make_game("chess-standard").apply(make_game("chess-standard").str_to_action("e2e4"))
    # e7e5 pour les Noirs s'encode comme e2e4 pour les Blancs
    assert s.str_to_action("e7e5") == make_game("chess-standard").str_to_action("e2e4")


def test_chess_checkmate():
    s = ChessState("standard")
    for mv in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        s = s.apply(s.str_to_action(mv))
    assert s.is_terminal() and s.winner() == -1


def test_chess_adjudication():
    from mcts_rl.games.chess_game import ChessVariant
    v = ChessVariant("t", "4k3/8/8/8/8/8/8/R3K3 w - - 0 1", max_plies=1, adjudication=3)
    s = ChessState(v).apply(ChessState(v).str_to_action("a1a2"))
    assert s.winner() == 1


def test_chess_only_queen_promotions():
    s = ChessState("standard", chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1"))
    promos = [a for a in s.legal_actions() if s.action_to_str(a).startswith("a7a8")]
    assert len(promos) == 1 and s.action_to_str(promos[0]) == "a7a8q"


def test_chess_encode_shape():
    for name in ("chess-mini", "chess-micro", "chess-standard"):
        s = make_game(name)
        p = s.encode()
        assert p.shape == ChessState.ENCODED_SHAPE
        assert p[:6].sum() == p[6:12].sum()  # position symétrique
