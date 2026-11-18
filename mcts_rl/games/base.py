"""Interface commune à tous les jeux à deux joueurs, à information complète.

Les joueurs sont notés +1 (premier joueur) et -1 (second joueur).
Les états sont *immuables* du point de vue des agents : `apply` renvoie un
nouvel état, ce qui simplifie énormément l'écriture du MCTS.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class GameState(ABC):
    #: nombre total d'actions possibles (taille de la sortie "policy" du réseau)
    ACTION_SIZE: int = 0
    #: forme du tenseur renvoyé par `encode()` (C, H, W)
    ENCODED_SHAPE: tuple = ()

    @property
    @abstractmethod
    def current_player(self) -> int:
        """+1 ou -1 : joueur qui doit jouer."""

    @abstractmethod
    def legal_actions(self) -> List[int]:
        """Liste des actions légales (entiers dans [0, ACTION_SIZE))."""

    @abstractmethod
    def apply(self, action: int) -> "GameState":
        """Renvoie l'état obtenu après avoir joué `action`."""

    @abstractmethod
    def is_terminal(self) -> bool:
        ...

    @abstractmethod
    def winner(self) -> Optional[int]:
        """+1 / -1 si un joueur a gagné, 0 si nul, None si la partie continue."""

    # ------------------------------------------------------------------
    # Pour le réseau de neurones (niveau 2)
    # ------------------------------------------------------------------
    def encode(self) -> np.ndarray:
        """Encodage de l'état du point de vue du joueur courant."""
        raise NotImplementedError

    def symmetries(self, planes: np.ndarray, pi: np.ndarray):
        """Augmentations de données (planes, pi) -> liste de couples équivalents."""
        return [(planes, pi)]

    # ------------------------------------------------------------------
    # Pour les rollouts du MCTS pur
    # ------------------------------------------------------------------
    def heuristic_value(self) -> float:
        """Évaluation heuristique dans [-1, 1] du point de vue du joueur +1.

        Utilisée quand un rollout est tronqué (profondeur max) avant la fin.
        """
        return 0.0

    # ------------------------------------------------------------------
    # Interaction humaine
    # ------------------------------------------------------------------
    @abstractmethod
    def render(self) -> str:
        ...

    def action_to_str(self, action: int) -> str:
        return str(action)

    def str_to_action(self, text: str) -> int:
        return int(text)

    def terminal_value(self, player: int) -> float:
        """Résultat final (+1 victoire, -1 défaite, 0 nul) du point de vue de `player`."""
        w = self.winner()
        if w is None:
            raise ValueError("la partie n'est pas terminée")
        return float(w * player)
