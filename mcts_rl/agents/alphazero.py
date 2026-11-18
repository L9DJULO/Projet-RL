"""MCTS guidé par un réseau policy/value (PUCT, AlphaGo Zero / AlphaZero).

Différences avec le MCTS pur (mcts.py) :
  * pas de rollout : la feuille est évaluée par le réseau (valeur v),
  * l'expansion crée d'un coup tous les enfants avec leur prior P(s,a),
  * sélection PUCT :  Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)),
  * bruit de Dirichlet à la racine pendant le self-play (exploration),
  * le coup est tiré selon les visites ^ (1/τ) (τ=1 en début de partie en self-play).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..games.base import GameState
from .base import Agent

# evaluator(state) -> (actions légales, priors, valeur du point de vue du joueur courant)
Evaluator = Callable[[GameState], Tuple[List[int], np.ndarray, float]]


class PUCTNode:
    __slots__ = ("state", "P", "N", "W", "children")

    def __init__(self, prior: float, state: Optional[GameState] = None):
        self.state = state
        self.P = prior
        self.N = 0
        self.W = 0.0  # du point de vue du joueur qui a joué le coup menant à ce nœud
        self.children: Dict[int, PUCTNode] = {}

    @property
    def expanded(self) -> bool:
        return bool(self.children)


class AlphaZeroMCTS:
    def __init__(self, evaluator: Evaluator, n_simulations: int = 100, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.3, noise_frac: float = 0.25,
                 rng: Optional[np.random.Generator] = None):
        self.evaluator = evaluator
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_frac = noise_frac
        self.rng = rng if rng is not None else np.random.default_rng()

    # Les briques ci-dessous (select_leaf / expand / backup) sont aussi utilisées
    # par le self-play "batché" (selfplay.py) qui évalue les feuilles de plusieurs
    # arbres en un seul appel au réseau.
    @staticmethod
    def expand(node: PUCTNode, actions, priors) -> None:
        for a, p in zip(actions, priors):
            node.children[a] = PUCTNode(float(p))

    def add_root_noise(self, root: PUCTNode) -> None:
        if not root.children:
            return
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(root.children))
        for child, n in zip(root.children.values(), noise):
            child.P = (1 - self.noise_frac) * child.P + self.noise_frac * n

    def select_leaf(self, root: PUCTNode):
        """Descend jusqu'à une feuille. Renvoie (chemin, valeur_terminale ou None)."""
        node, path = root, [root]
        while node.expanded:
            action, child = self._select(node)
            if child.state is None:
                child.state = node.state.apply(action)
            node = child
            path.append(node)
        leaf = node.state
        if leaf.is_terminal():
            return path, leaf.terminal_value(leaf.current_player)
        return path, None

    @staticmethod
    def backup(path, value: float) -> None:
        """`value` est du point de vue du joueur au trait dans la feuille ; chaque
        nœud stocke le point de vue du joueur qui a joué le coup y menant."""
        for n in reversed(path):
            value = -value
            n.N += 1
            n.W += value

    def _expand(self, node: PUCTNode) -> float:
        actions, priors, value = self.evaluator(node.state)
        self.expand(node, actions, priors)
        return value

    def _select(self, node: PUCTNode) -> Tuple[int, PUCTNode]:
        sqrt_n = math.sqrt(node.N)
        c = self.c_puct
        best_a, best_child, best_score = None, None, -math.inf
        for a, child in node.children.items():
            q = child.W / child.N if child.N else 0.0
            score = q + c * child.P * sqrt_n / (1 + child.N)
            if score > best_score:
                best_a, best_child, best_score = a, child, score
        return best_a, best_child

    def search(self, state: GameState, add_noise: bool = False) -> PUCTNode:
        root = PUCTNode(1.0, state)
        self._expand(root)
        root.N = 1
        if add_noise:
            self.add_root_noise(root)

        for _ in range(self.n_simulations):
            # 1. sélection (PUCT)
            path, value = self.select_leaf(root)
            # 2. évaluation + expansion par le réseau (si la feuille n'est pas terminale)
            if value is None:
                value = self._expand(path[-1])
            # 3. rétropropagation
            self.backup(path, value)
        return root

    @staticmethod
    def visit_policy(root: PUCTNode, action_size: int) -> np.ndarray:
        pi = np.zeros(action_size, dtype=np.float32)
        for a, child in root.children.items():
            pi[a] = child.N
        s = pi.sum()
        return pi / s if s > 0 else pi


def sample_action(root: PUCTNode, temperature: float, rng: np.random.Generator) -> int:
    actions = list(root.children.keys())
    visits = np.array([root.children[a].N for a in actions], dtype=np.float64)
    if temperature <= 1e-6:
        return actions[int(np.argmax(visits))]
    probs = visits ** (1.0 / temperature)
    probs /= probs.sum()
    return actions[int(rng.choice(len(actions), p=probs))]


class AlphaZeroAgent(Agent):
    """Agent de match (pas de bruit, coup le plus visité)."""

    def __init__(self, evaluator: Evaluator, n_simulations: int = 100, c_puct: float = 1.5,
                 temperature: float = 0.0, seed: Optional[int] = None, name: str = None):
        self.rng = np.random.default_rng(seed)
        self.mcts = AlphaZeroMCTS(evaluator, n_simulations, c_puct, rng=self.rng)
        self.temperature = temperature
        self.name = name or f"AlphaZero(n={n_simulations})"
        self.last_root: Optional[PUCTNode] = None

    def select_action(self, state: GameState) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        root = self.mcts.search(state, add_noise=False)
        self.last_root = root
        return sample_action(root, self.temperature, self.rng)

    def root_stats(self):
        if self.last_root is None:
            return []
        return sorted(((a, c.N, c.W / c.N if c.N else 0.0) for a, c in self.last_root.children.items()),
                      key=lambda t: -t[1])


class PolicyAgent(Agent):
    """Joue directement l'argmax de la policy du réseau (aucune recherche)."""

    def __init__(self, evaluator: Evaluator, name: str = "Policy-net"):
        self.evaluator = evaluator
        self.name = name

    def select_action(self, state: GameState) -> int:
        actions, priors, _ = self.evaluator(state)
        return actions[int(np.argmax(priors))]
