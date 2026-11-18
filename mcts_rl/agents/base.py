from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

from ..games.base import GameState


class Agent(ABC):
    name: str = "agent"

    @abstractmethod
    def select_action(self, state: GameState) -> int:
        ...

    def reset(self) -> None:
        """Appelé au début de chaque partie."""

    def __repr__(self):
        return self.name


class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.name = "random"

    def select_action(self, state: GameState) -> int:
        return self.rng.choice(state.legal_actions())


class HumanAgent(Agent):
    """Joueur humain en ligne de commande."""

    def __init__(self, prompt: str = "Votre coup"):
        self.prompt = prompt
        self.name = "human"

    def select_action(self, state: GameState) -> int:
        legal = state.legal_actions()
        hint = ", ".join(sorted(state.action_to_str(a) for a in legal))
        while True:
            text = input(f"{self.prompt} (coups légaux : {hint}) > ")
            try:
                action = state.str_to_action(text)
            except (ValueError, KeyError) as exc:
                print(f"  entrée invalide : {exc}")
                continue
            if action in legal:
                return action
            print("  coup illégal, recommencez.")
