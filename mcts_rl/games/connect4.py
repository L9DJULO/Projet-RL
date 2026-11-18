"""Puissance 4 (6 lignes x 7 colonnes) en bitboard.

Moteur minimal et rapide (représentation bitboard classique, cf. John Tromp /
Pascal Pons "Connect 4 solver"). Le projet ne porte pas sur le moteur : nous
reprenons la représentation standard car les rollouts du MCTS pur demandent
des millions de coups simulés et une implémentation à base de listes Python
serait ~20x plus lente.

Disposition des bits : chaque colonne occupe 7 bits (6 cases + 1 bit sentinelle)
    colonne c, ligne r (0 = bas) -> bit c * 7 + r
"""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from .base import GameState

ROWS, COLS = 6, 7
H1 = ROWS + 1
BOTTOM_MASK = sum(1 << (c * H1) for c in range(COLS))
BOARD_MASK = BOTTOM_MASK * ((1 << ROWS) - 1)
TOP_MASK = sum(1 << (c * H1 + ROWS - 1) for c in range(COLS))
COLUMN_ORDER = [3, 2, 4, 1, 5, 0, 6]  # centre d'abord (utile pour l'heuristique)


def _has_four(bb: int) -> bool:
    # horizontal
    m = bb & (bb >> H1)
    if m & (m >> (2 * H1)):
        return True
    # diagonale /
    m = bb & (bb >> (H1 + 1))
    if m & (m >> (2 * (H1 + 1))):
        return True
    # diagonale \
    m = bb & (bb >> (H1 - 1))
    if m & (m >> (2 * (H1 - 1))):
        return True
    # vertical
    m = bb & (bb >> 1)
    if m & (m >> 2):
        return True
    return False


def _column_mask(col: int) -> int:
    return ((1 << ROWS) - 1) << (col * H1)


class Connect4State(GameState):
    ACTION_SIZE = COLS
    ENCODED_SHAPE = (3, ROWS, COLS)

    __slots__ = ("cur", "mask", "moves", "_winner", "_player")

    def __init__(self, cur: int = 0, mask: int = 0, moves: int = 0,
                 winner: Optional[int] = None):
        # `cur` = pierres du joueur qui doit jouer, `mask` = toutes les pierres
        self.cur = cur
        self.mask = mask
        self.moves = moves
        self._winner = winner
        self._player = 1 if moves % 2 == 0 else -1

    # ------------------------------------------------------------------
    @property
    def current_player(self) -> int:
        return self._player

    def can_play(self, col: int) -> bool:
        return (self.mask & (1 << (col * H1 + ROWS - 1))) == 0

    def legal_actions(self) -> List[int]:
        if self._winner is not None:
            return []
        return [c for c in range(COLS) if self.can_play(c)]

    def apply(self, action: int) -> "Connect4State":
        if self._winner is not None or not self.can_play(action):
            raise ValueError(f"coup illégal : {action}")
        new_mask = self.mask | (self.mask + (1 << (action * H1)))
        mover = self.cur | (new_mask ^ self.mask)  # pierres du joueur qui vient de jouer
        moves = self.moves + 1
        if _has_four(mover):
            winner = self._player
        elif moves == ROWS * COLS:
            winner = 0
        else:
            winner = None
        # on échange les rôles : `cur` devient l'adversaire
        return Connect4State(self.cur ^ self.mask, new_mask, moves, winner)

    def is_terminal(self) -> bool:
        return self._winner is not None

    def winner(self) -> Optional[int]:
        return self._winner

    # ------------------------------------------------------------------
    # Outils pour l'heuristique de rollout
    # ------------------------------------------------------------------
    def winning_move(self, for_current: bool = True) -> Optional[int]:
        """Colonne qui gagne immédiatement pour le joueur courant (ou l'adversaire)."""
        stones = self.cur if for_current else self.cur ^ self.mask
        for c in COLUMN_ORDER:
            if self.can_play(c):
                pos = (self.mask + (1 << (c * H1))) & _column_mask(c)
                if _has_four(stones | pos):
                    return c
        return None

    def heuristic_rollout_action(self, rng: random.Random) -> int:
        """Politique de rollout "intelligente" : gagner si possible, sinon bloquer, sinon aléatoire."""
        c = self.winning_move(True)
        if c is not None:
            return c
        c = self.winning_move(False)
        if c is not None:
            return c
        return rng.choice(self.legal_actions())

    # ------------------------------------------------------------------
    # Encodage pour le réseau
    # ------------------------------------------------------------------
    def to_array(self) -> np.ndarray:
        """Plateau (ROWS, COLS) avec +1 / -1 / 0 en coordonnées absolues (ligne 0 = haut)."""
        board = np.zeros((ROWS, COLS), dtype=np.int8)
        opp = self.cur ^ self.mask
        for c in range(COLS):
            for r in range(ROWS):
                bit = 1 << (c * H1 + r)
                if self.cur & bit:
                    board[ROWS - 1 - r, c] = self._player
                elif opp & bit:
                    board[ROWS - 1 - r, c] = -self._player
        return board

    def encode(self) -> np.ndarray:
        board = self.to_array() * self._player  # point de vue du joueur courant
        planes = np.zeros(self.ENCODED_SHAPE, dtype=np.float32)
        planes[0] = board == 1
        planes[1] = board == -1
        planes[2] = 1.0 if self._player == 1 else 0.0
        return planes

    def symmetries(self, planes, pi):
        # symétrie gauche/droite du plateau
        return [(planes, pi), (planes[:, :, ::-1].copy(), pi[::-1].copy())]

    def heuristic_value(self) -> float:
        return 0.0

    # ------------------------------------------------------------------
    def render(self) -> str:
        sym = {1: "X", -1: "O", 0: "."}
        board = self.to_array()
        lines = [" ".join(sym[int(v)] for v in row) for row in board]
        lines.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(lines)

    def __repr__(self):
        return f"Connect4State(moves={self.moves}, winner={self._winner})\n" + self.render()

    @staticmethod
    def from_moves(moves: str) -> "Connect4State":
        """Construit une position depuis une suite de colonnes, ex. '3344'."""
        s = Connect4State()
        for ch in moves:
            s = s.apply(int(ch))
        return s
