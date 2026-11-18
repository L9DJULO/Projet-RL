"""Niveau 1 — Puissance 4 : MCTS pur, étude des hyperparamètres.

Expériences :
  E1  MCTS vs aléatoire en fonction du nombre de simulations
  E2  influence de la constante d'exploration c (UCB1) : MCTS(c) vs MCTS(c=√2)
  E3  MCTS(n) vs MCTS(200) : rendement du budget de simulations
  E4  politique de rollout : heuristique (gagner/bloquer) vs aléatoire, à budget égal
      en simulations puis à temps de réflexion égal
  E5  avantage du premier joueur (MCTS vs MCTS identiques)

Usage :
    python scripts/level1_experiments.py               # toutes les expériences (~30 min sur 4 cœurs)
    python scripts/level1_experiments.py --only E2 --games 200
    python scripts/level1_experiments.py --plots-only  # régénère les figures depuis results/level1/*.json
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcts_rl.arena import play_match  # noqa: E402
from mcts_rl.plotting import INK_2, SERIES, errorbar_ci, plt, save, setup  # noqa: E402

OUT = "results/level1"
SQRT2 = math.sqrt(2)


def mcts(n, c=SQRT2, rollout="random", **kw):
    return {"type": "mcts", "n_simulations": n, "c": c, "rollout": rollout, **kw}


def run(name, pairs, games, workers):
    """pairs : liste de (x, spec_a, spec_b). Sauvegarde results/level1/<name>.json"""
    rows = []
    for i, (x, a, b) in enumerate(pairs):
        t0 = time.time()
        r = play_match(a, b, n_games=games, game="connect4", workers=workers, seed=i + 1)
        d = r.as_dict()
        d["x"] = x
        d["wins_as_first"] = r.wins_as_first
        d["games_as_first"] = r.games_as_first
        rows.append(d)
        print(f"  [{name}] x={x}: {r}  ({time.time() - t0:.0f}s, {d['ms_per_move_a']:.1f} ms/coup)")
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    return rows


def exp1(games, workers):
    sims = [1, 5, 10, 25, 50, 100, 200, 500, 1000]
    return run("E1_vs_random", [(n, mcts(n), {"type": "random"}) for n in sims], games, workers)


def exp2(games, workers):
    cs = [0.05, 0.25, 0.5, 1.0, SQRT2, 2.0, 3.0, 5.0]
    return run("E2_exploration", [(c, mcts(300, c), mcts(300)) for c in cs], games, workers)


def exp3(games, workers):
    sims = [25, 50, 100, 200, 400, 800, 1600, 3200]
    return run("E3_budget", [(n, mcts(n), mcts(200)) for n in sims], games, workers)


def exp4(games, workers):
    rows = run("E4_rollout_sims", [(n, mcts(n, rollout="heuristic"), mcts(n)) for n in (50, 200, 800)],
               games, workers)
    rows += run("E4_rollout_time", [(t, mcts(0, rollout="heuristic", time_limit=t), mcts(0, time_limit=t))
                                    for t in (0.05, 0.2)], games, workers)
    return rows


def exp5(games, workers):
    # swap_colors=False : A joue toujours en premier
    rows = []
    for n in (50, 200, 1000):
        r = play_match(mcts(n), mcts(n), n_games=games, game="connect4", workers=workers, swap_colors=False)
        d = r.as_dict()
        d["x"] = n
        rows.append(d)
        print(f"  [E5] n={n}: premier joueur {r}")
    with open(os.path.join(OUT, "E5_first_player.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    return rows


# ---------------------------------------------------------------------------
def _load(name):
    path = os.path.join(OUT, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plots():
    setup()
    e1 = _load("E1_vs_random")
    if e1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
        x = [r["x"] for r in e1]
        errorbar_ci(ax1, x, e1, SERIES[0], "MCTS vs aléatoire", key="score")
        ax1.set(xscale="log", xlabel="simulations par coup", ylabel="score (V=1, N=½)",
                title="E1 · MCTS contre un joueur aléatoire", ylim=(0.4, 1.02))
        ax1.axhline(0.5, color=INK_2, lw=1, ls=":")
        ax2.plot(x, [r["ms_per_move_a"] for r in e1], color=SERIES[0], marker="o")
        ax2.set(xscale="log", yscale="log", xlabel="simulations par coup", ylabel="ms par coup",
                title="Coût : temps de réflexion par coup")
        save(fig, os.path.join(OUT, "E1_vs_random.png"))

    e2 = _load("E2_exploration")
    if e2:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        errorbar_ci(ax, [r["x"] for r in e2], e2, SERIES[0], "MCTS(c, 300) vs MCTS(√2, 300)")
        ax.axhline(0.5, color=INK_2, lw=1, ls=":")
        ax.axvline(SQRT2, color=INK_2, lw=1, ls="--")
        ax.text(SQRT2 * 1.05, 0.06, "c = √2", color=INK_2)
        ax.set(xscale="log", xlabel="constante d'exploration c (UCB1)", ylabel="score contre c = √2",
               title="E2 · Influence de la constante d'exploration", ylim=(0, 1))
        save(fig, os.path.join(OUT, "E2_exploration.png"))

    e3 = _load("E3_budget")
    if e3:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        errorbar_ci(ax, [r["x"] for r in e3], e3, SERIES[0], "MCTS(n) vs MCTS(200)")
        ax.axhline(0.5, color=INK_2, lw=1, ls=":")
        ax.axvline(200, color=INK_2, lw=1, ls="--")
        ax.set(xscale="log", xlabel="simulations n de l'agent testé", ylabel="score contre MCTS(200)",
               title="E3 · Rendement du budget de simulations", ylim=(0, 1))
        save(fig, os.path.join(OUT, "E3_budget.png"))

    e4s, e4t = _load("E4_rollout_sims"), _load("E4_rollout_time")
    if e4s and e4t:
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        labels = [f"{r['x']} sim." for r in e4s] + [f"{int(r['x'] * 1000)} ms" for r in e4t]
        rows = e4s + e4t
        colors = [SERIES[0]] * len(e4s) + [SERIES[1]] * len(e4t)
        ys = [r["score"] for r in rows]
        err = [[max(0.0, r["score"] - r["ci95"][0]) for r in rows],
               [max(0.0, r["ci95"][1] - r["score"]) for r in rows]]
        ax.bar(range(len(rows)), ys, color=colors, width=0.6, yerr=err, capsize=3,
               error_kw={"elinewidth": 1, "ecolor": INK_2})
        ax.set_xticks(range(len(rows)), labels)
        ax.axhline(0.5, color=INK_2, lw=1, ls=":")
        ax.set(ylabel="score rollout heuristique", ylim=(0, 1),
               title="E4 · Rollout heuristique vs aléatoire")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=SERIES[0], label="budget égal en simulations"),
                           Patch(color=SERIES[1], label="temps égal par coup")], loc="upper right")
        save(fig, os.path.join(OUT, "E4_rollout.png"))


EXPERIMENTS = {"E1": exp1, "E2": exp2, "E3": exp3, "E4": exp4, "E5": exp5}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", choices=EXPERIMENTS)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--plots-only", action="store_true")
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not args.plots_only:
        for name in args.only or EXPERIMENTS:
            print(f"== {name} ==")
            EXPERIMENTS[name](args.games, args.workers)
    plots()


if __name__ == "__main__":
    main()
