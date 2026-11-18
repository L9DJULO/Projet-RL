"""MCTS "pur" (UCT, Kocsis & Szepesvári 2006) avec rollouts.

Les quatre phases d'une simulation :
  1. Sélection   : on descend dans l'arbre en maximisant UCB1
                   Q(s,a) + c * sqrt( ln N(s) / N(s,a) )
  2. Expansion   : on ajoute UN nouvel enfant (action non encore essayée)
  3. Simulation  : rollout jusqu'à la fin de la partie (politique aléatoire ou
                   heuristique), éventuellement tronqué -> évaluation heuristique
  4. Rétropropagation : on remonte le résultat jusqu'à la racine.

Convention de signe : `node.W` est la somme des résultats du point de vue du
joueur qui a joué le coup menant à ce nœud (`node.mover`). Ainsi, depuis le
parent, on choisit simplement l'enfant de plus grande valeur moyenne.
"""
from __future__ import annotations

import math
import random
import time
from typing import Callable, List, Optional

from ..games.base import GameState
from .base import Agent


class UCTNode:
    __slots__ = ("state", "parent", "action", "children", "untried", "N", "W", "mover")

    def __init__(self, state: GameState, parent: Optional["UCTNode"], action: Optional[int],
                 rng: random.Random):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[UCTNode] = []
        self.untried = state.legal_actions()
        rng.shuffle(self.untried)
        self.N = 0
        self.W = 0.0
        self.mover = parent.state.current_player if parent is not None else 0


def random_rollout_policy(state: GameState, rng: random.Random) -> int:
    return rng.choice(state.legal_actions())


def heuristic_rollout_policy(state: GameState, rng: random.Random) -> int:
    """Utilise `state.heuristic_rollout_action` si le jeu en propose une (Puissance 4)."""
    fn = getattr(state, "heuristic_rollout_action", None)
    if fn is None:
        return rng.choice(state.legal_actions())
    return fn(rng)


ROLLOUT_POLICIES = {"random": random_rollout_policy, "heuristic": heuristic_rollout_policy}


class MCTSAgent(Agent):
    def __init__(self, n_simulations: int = 1000, c: float = math.sqrt(2),
                 rollout: str | Callable = "random", max_rollout_depth: Optional[int] = None,
                 time_limit: Optional[float] = None, seed: Optional[int] = None):
        """
        n_simulations     : nombre de simulations (itérations) par coup
        c                 : constante d'exploration UCB1
        rollout           : 'random', 'heuristic' ou fonction (state, rng) -> action
        max_rollout_depth : tronque les rollouts (utile aux échecs) -> heuristic_value()
        time_limit        : si fourni, budget en secondes par coup (remplace n_simulations)
        """
        self.n_simulations = n_simulations
        self.c = c
        self.rollout_policy = ROLLOUT_POLICIES[rollout] if isinstance(rollout, str) else rollout
        self.max_rollout_depth = max_rollout_depth
        self.time_limit = time_limit
        self.rng = random.Random(seed)
        rname = rollout if isinstance(rollout, str) else "custom"
        self.name = f"MCTS(n={n_simulations}, c={c:.2f}, {rname})"
        self.last_root: Optional[UCTNode] = None

    # ------------------------------------------------------------------
    def _select_child(self, node: UCTNode) -> UCTNode:
        log_n = math.log(node.N)
        c = self.c
        best, best_score = None, -math.inf
        for child in node.children:
            score = child.W / child.N + c * math.sqrt(log_n / child.N)
            if score > best_score:
                best, best_score = child, score
        return best

    def _rollout(self, state: GameState) -> float:
        """Renvoie le résultat du point de vue du joueur +1."""
        depth = 0
        policy, rng, max_depth = self.rollout_policy, self.rng, self.max_rollout_depth
        while not state.is_terminal():
            if max_depth is not None and depth >= max_depth:
                return state.heuristic_value()
            state = state.apply(policy(state, rng))
            depth += 1
        return float(state.winner())

    def search(self, root_state: GameState) -> UCTNode:
        root = UCTNode(root_state, None, None, self.rng)
        deadline = time.perf_counter() + self.time_limit if self.time_limit else None
        sims = 0
        while True:
            if deadline is not None:
                if time.perf_counter() >= deadline and sims > 0:
                    break
            elif sims >= self.n_simulations:
                break
            sims += 1

            node = root
            # 1. sélection
            while not node.untried and node.children:
                node = self._select_child(node)
            # 2. expansion
            if node.untried:
                action = node.untried.pop()
                child = UCTNode(node.state.apply(action), node, action, self.rng)
                node.children.append(child)
                node = child
            # 3. simulation
            value = self._rollout(node.state)
            # 4. rétropropagation
            while node is not None:
                node.N += 1
                node.W += value * node.mover
                node = node.parent
        self.last_root = root
        return root

    def select_action(self, state: GameState) -> int:
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        root = self.search(state)
        # "robust child" : l'enfant le plus visité
        best = max(root.children, key=lambda ch: (ch.N, ch.W / ch.N))
        return best.action

    def root_stats(self):
        """[(action, visites, valeur moyenne)] de la dernière recherche."""
        if self.last_root is None:
            return []
        return sorted(((ch.action, ch.N, ch.W / ch.N) for ch in self.last_root.children),
                      key=lambda t: -t[1])
