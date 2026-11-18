"""Faire jouer deux agents (ou un humain contre un agent).

Syntaxe des joueurs :
    human                       joueur humain (clavier)
    random                      coups aléatoires
    mcts[:N[:c]]                MCTS pur, N simulations (déf. 1000), constante c (déf. 1.41)
    mcts-h[:N[:c]]              MCTS pur avec rollouts heuristiques (Puissance 4 : gagner/bloquer)
    az:CHECKPOINT[:N]           MCTS guidé par réseau (AlphaZero), N simulations (déf. 100)
    policy:CHECKPOINT           réseau seul, sans recherche

Exemples :
    python scripts/play.py --p1 human --p2 mcts:2000
    python scripts/play.py --p1 az:checkpoints/connect4/latest.pt --p2 human
    python scripts/play.py --game chess-mini --p1 human --p2 az:checkpoints/chess-mini/latest.pt:200
    python scripts/play.py --p1 mcts:500 --p2 random --games 50          # match sans affichage
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcts_rl.agents import make_agent  # noqa: E402
from mcts_rl.arena import play_game, play_match  # noqa: E402


def parse_player(text: str, game: str) -> dict:
    parts = text.split(":")
    kind = parts[0].lower()
    if kind in ("human", "random"):
        return {"type": kind}
    if kind in ("mcts", "mcts-h"):
        spec = {"type": "mcts", "n_simulations": int(parts[1]) if len(parts) > 1 else 1000,
                "rollout": "heuristic" if kind == "mcts-h" else "random"}
        if len(parts) > 2:
            spec["c"] = float(parts[2])
        if game.startswith("chess"):
            spec["max_rollout_depth"] = 20
        return spec
    if kind in ("az", "alphazero"):
        # le chemin peut contenir ':' sous Windows (C:\...) -> le dernier champ numérique = N
        n = 100
        if len(parts) > 2 and parts[-1].isdigit():
            n = int(parts.pop())
        return {"type": "alphazero", "checkpoint": ":".join(parts[1:]), "n_simulations": n}
    if kind == "policy":
        return {"type": "policy", "checkpoint": ":".join(parts[1:])}
    raise ValueError(f"joueur inconnu : {text}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="connect4", help="connect4, chess-mini, chess-micro, chess-standard")
    p.add_argument("--p1", default="human", help="premier joueur (X / Blancs)")
    p.add_argument("--p2", default="mcts:1000", help="second joueur (O / Noirs)")
    p.add_argument("--games", type=int, default=1, help="nombre de parties (sans humain)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    s1, s2 = parse_player(args.p1, args.game), parse_player(args.p2, args.game)
    interactive = "human" in (s1["type"], s2["type"])
    if args.games > 1 and not interactive:
        res = play_match(s1, s2, n_games=args.games, game=args.game, seed=args.seed or 0)
        print(res)
        return
    a1, a2 = make_agent(s1, seed=args.seed), make_agent(s2, seed=args.seed)
    for _ in range(args.games):
        play_game(a1, a2, game=args.game, verbose=True)


if __name__ == "__main__":
    main()
