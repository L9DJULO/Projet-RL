"""Construction d'agents à partir d'une spécification (dict sérialisable).

Une spécification permet de recréer l'agent dans un sous-processus
(parties jouées en parallèle). Exemples :
    {"type": "random"}
    {"type": "human"}
    {"type": "mcts", "n_simulations": 500, "c": 1.4, "rollout": "heuristic"}
    {"type": "alphazero", "checkpoint": "checkpoints/connect4/best.pt", "n_simulations": 100}
    {"type": "policy", "checkpoint": "checkpoints/connect4/best.pt"}
"""
from __future__ import annotations

from typing import Optional

from .base import Agent, HumanAgent, RandomAgent
from .mcts import MCTSAgent
from .alphazero import AlphaZeroAgent, AlphaZeroMCTS, PolicyAgent

_NET_CACHE = {}


def _load_evaluator(checkpoint: str):
    from ..nn.model import NetEvaluator, PolicyValueNet
    if checkpoint not in _NET_CACHE:
        _NET_CACHE[checkpoint] = NetEvaluator(PolicyValueNet.load(checkpoint))
    return _NET_CACHE[checkpoint]


def make_agent(spec: dict, seed: Optional[int] = None) -> Agent:
    spec = dict(spec)
    kind = spec.pop("type")
    label = spec.pop("name", None)
    if kind == "random":
        agent = RandomAgent(seed=seed)
    elif kind == "human":
        agent = HumanAgent()
    elif kind == "mcts":
        agent = MCTSAgent(seed=seed, **spec)
    elif kind == "alphazero":
        ev = _load_evaluator(spec.pop("checkpoint"))
        agent = AlphaZeroAgent(ev, seed=seed, **spec)
    elif kind == "policy":
        agent = PolicyAgent(_load_evaluator(spec.pop("checkpoint")))
    else:
        raise ValueError(f"type d'agent inconnu : {kind}")
    if label:
        agent.name = label
    return agent


__all__ = ["Agent", "RandomAgent", "HumanAgent", "MCTSAgent", "AlphaZeroAgent",
           "AlphaZeroMCTS", "PolicyAgent", "make_agent"]
