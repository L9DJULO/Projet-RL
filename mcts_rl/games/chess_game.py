"""Échecs via la bibliothèque `python-chess` (moteur de règles existant).

Nous ajoutons seulement la couche nécessaire aux agents :
  * encodage des actions en entiers  (case_départ * 64 + case_arrivée, 4096 actions),
  * encodage du plateau en plans (12 pièces + 4 roques + 1 compteur de demi-coups),
  * toujours du point de vue du joueur courant (le plateau est "retourné"
    quand c'est aux Noirs de jouer, comme dans AlphaZero),
  * une variante allégée "mini" pour pouvoir entraîner un réseau sur CPU.

Simplifications (documentées dans le rapport) :
  * les promotions se font uniquement en dame (les sous-promotions sont retirées
    des coups légaux) — cela permet un encodage de 4096 actions ;
  * la partie est limitée à `max_plies` demi-coups ; à la limite elle est
    adjugée à l'avantage matériel (>= `adjudication` points), sinon nulle ;
  * la répétition de position n'est pas détectée (la limite de coups la remplace),
    la règle des 50 coups l'est.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import chess
import numpy as np

from .base import GameState

PIECE_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


@dataclass(frozen=True)
class ChessVariant:
    name: str
    fen: str
    max_plies: int
    adjudication: int  # avantage matériel nécessaire pour gagner à la limite de coups


VARIANTS: Dict[str, ChessVariant] = {
    # échecs classiques
    "standard": ChessVariant("standard", chess.STARTING_FEN, 300, 3),
    # variante allégée : pas de dame ni de fou -> roi, 2 tours, 2 cavaliers, 8 pions
    "mini": ChessVariant("mini", "rn2k1nr/pppppppp/8/8/8/8/PPPPPPPP/RN2K1NR w KQkq - 0 1", 80, 2),
    # variante très allégée : roi + tour + 4 pions de chaque côté
    "micro": ChessVariant("micro", "4k2r/2pppp2/8/8/8/8/2PPPP2/4K2R w Kk - 0 1", 60, 2),
}


def _material(board: chess.Board) -> int:
    """Matériel des Blancs moins matériel des Noirs."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        if value:
            total += value * (len(board.pieces(piece_type, chess.WHITE))
                              - len(board.pieces(piece_type, chess.BLACK)))
    return total


class ChessState(GameState):
    ACTION_SIZE = 64 * 64
    ENCODED_SHAPE = (17, 8, 8)

    __slots__ = ("board", "variant", "ply", "_legal", "_winner", "_done")

    def __init__(self, variant: str | ChessVariant = "mini",
                 board: Optional[chess.Board] = None, ply: int = 0):
        self.variant = VARIANTS[variant] if isinstance(variant, str) else variant
        self.board = board if board is not None else chess.Board(self.variant.fen)
        self.ply = ply
        self._legal: Optional[Dict[int, chess.Move]] = None
        self._winner: Optional[int] = None
        self._done = False

    # ------------------------------------------------------------------
    # encodage des coups
    # ------------------------------------------------------------------
    def _sq(self, sq: int) -> int:
        """Case vue par le joueur courant (miroir vertical pour les Noirs)."""
        return sq if self.board.turn == chess.WHITE else chess.square_mirror(sq)

    def encode_move(self, move: chess.Move) -> int:
        return self._sq(move.from_square) * 64 + self._sq(move.to_square)

    def _legal_map(self) -> Dict[int, chess.Move]:
        if self._legal is None:
            legal = {}
            for mv in self.board.legal_moves:
                if mv.promotion is not None and mv.promotion != chess.QUEEN:
                    continue
                legal[self.encode_move(mv)] = mv
            self._legal = legal
        return self._legal

    # ------------------------------------------------------------------
    @property
    def current_player(self) -> int:
        return 1 if self.board.turn == chess.WHITE else -1

    def legal_actions(self) -> List[int]:
        if self.is_terminal():
            return []
        return list(self._legal_map().keys())

    def apply(self, action: int) -> "ChessState":
        move = self._legal_map().get(action)
        if move is None or self.is_terminal():
            raise ValueError(f"coup illégal : {action}")
        board = self.board.copy(stack=False)
        board.push(move)
        return ChessState(self.variant, board, self.ply + 1)

    def _compute_outcome(self):
        if self._done:
            return
        self._done = True
        b = self.board
        if not self._legal_map():
            # mat ou pat
            self._winner = -self.current_player if b.is_check() else 0
        elif b.is_insufficient_material() or b.halfmove_clock >= 100:
            self._winner = 0
        elif self.ply >= self.variant.max_plies:
            diff = _material(b)
            if diff >= self.variant.adjudication:
                self._winner = 1
            elif diff <= -self.variant.adjudication:
                self._winner = -1
            else:
                self._winner = 0
        else:
            self._winner = None

    def is_terminal(self) -> bool:
        self._compute_outcome()
        return self._winner is not None

    def winner(self) -> Optional[int]:
        self._compute_outcome()
        return self._winner

    # ------------------------------------------------------------------
    def heuristic_value(self) -> float:
        """Avantage matériel écrasé dans [-1, 1] (point de vue des Blancs)."""
        return math.tanh(_material(self.board) / 5.0)

    def encode(self) -> np.ndarray:
        planes = np.zeros(self.ENCODED_SHAPE, dtype=np.float32)
        turn = self.board.turn
        for sq, piece in self.board.piece_map().items():
            s = self._sq(sq)
            offset = 0 if piece.color == turn else 6
            planes[offset + piece.piece_type - 1, s // 8, s % 8] = 1.0
        b = self.board
        opp = not turn
        planes[12] = float(b.has_kingside_castling_rights(turn))
        planes[13] = float(b.has_queenside_castling_rights(turn))
        planes[14] = float(b.has_kingside_castling_rights(opp))
        planes[15] = float(b.has_queenside_castling_rights(opp))
        planes[16] = self.ply / self.variant.max_plies
        return planes

    # ------------------------------------------------------------------
    def render(self) -> str:
        return str(self.board) + f"\n({'Blancs' if self.board.turn else 'Noirs'} au trait, demi-coup {self.ply}/{self.variant.max_plies})"

    def action_to_str(self, action: int) -> str:
        mv = self._legal_map().get(action)
        return mv.uci() if mv is not None else f"<{action}>"

    def str_to_action(self, text: str) -> int:
        text = text.strip()
        try:
            move = self.board.parse_uci(text)
        except ValueError:
            move = self.board.parse_san(text)  # lève ValueError si invalide
        if move.promotion is not None and move.promotion != chess.QUEEN:
            raise ValueError("seules les promotions en dame sont autorisées")
        action = self.encode_move(move)
        if action not in self._legal_map():
            raise ValueError("coup illégal")
        return action
