"""Entraînement AlphaZero simplifié par self-play.

Exemples :
    python scripts/train.py --preset connect4
    python scripts/train.py --preset chess-mini
    python scripts/train.py --preset connect4 --iterations 2 --games-per-iter 8 --out checkpoints/test   # essai rapide
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcts_rl.selfplay import TrainConfig, run_training  # noqa: E402

PRESETS = {
    "connect4": dict(
        game="connect4", out_dir="checkpoints/connect4", iterations=30, games_per_iter=160,
        games_in_parallel=40, n_simulations=100, c_puct=1.5, dirichlet_alpha=1.0, temp_moves=8,
        buffer_iters=6, channels=48, blocks=3, eval_every=3, eval_games=30, eval_sims=100,
        eval_opponents=[{"type": "random"},
                        {"type": "mcts", "n_simulations": 200},
                        {"type": "mcts", "n_simulations": 1000}],
    ),
    "chess-mini": dict(
        game="chess-mini", out_dir="checkpoints/chess-mini", iterations=20, games_per_iter=96,
        games_in_parallel=24, n_simulations=64, c_puct=1.5, dirichlet_alpha=0.3, temp_moves=16,
        buffer_iters=5, channels=48, blocks=3, eval_every=4, eval_games=20, eval_sims=64,
        eval_opponents=[{"type": "random"},
                        {"type": "mcts", "n_simulations": 200, "max_rollout_depth": 20}],
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, default="connect4")
    p.add_argument("--iterations", type=int)
    p.add_argument("--games-per-iter", type=int)
    p.add_argument("--games-in-parallel", type=int)
    p.add_argument("--sims", type=int, dest="n_simulations")
    p.add_argument("--workers", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--eval-games", type=int)
    p.add_argument("--out", dest="out_dir")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    params = dict(PRESETS[args.preset])
    for k in ("iterations", "games_per_iter", "games_in_parallel", "n_simulations", "workers",
              "eval_every", "eval_games", "out_dir"):
        v = getattr(args, k)
        if v is not None:
            params[k] = v
    run_training(TrainConfig(**params), resume=not args.no_resume)


if __name__ == "__main__":
    main()
