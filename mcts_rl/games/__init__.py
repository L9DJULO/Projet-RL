from .base import GameState
from .connect4 import Connect4State
from .chess_game import ChessState, VARIANTS


def make_game(name: str) -> GameState:
    """Fabrique l'état initial d'un jeu : 'connect4', 'chess' (=mini), 'chess-standard', 'chess-micro'..."""
    name = name.lower()
    if name in ("connect4", "c4", "puissance4"):
        return Connect4State()
    if name.startswith("chess"):
        variant = name.split("-", 1)[1] if "-" in name else "mini"
        return ChessState(variant)
    raise ValueError(f"jeu inconnu : {name}")


__all__ = ["GameState", "Connect4State", "ChessState", "VARIANTS", "make_game"]
