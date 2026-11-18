"""Niveau 2 — comparaison des agents AVEC et SANS réseau de neurones.

Pour un jeu entraîné par `scripts/train.py`, produit dans results/level2/<jeu>/ :
  * training_curves.png : pertes et score en évaluation au fil des itérations de self-play
  * C1 : AlphaZero(n) vs MCTS pur(n) à nombre de simulations égal
  * C2 : AlphaZero(100) vs MCTS pur(n) pour plusieurs budgets n
  * C3 : ablations — réseau non entraîné (itération 0) + MCTS, réseau seul (sans recherche),
         réseau entraîné vs réseau intermédiaire (apport du self-play)

Usage :
    python scripts/level2_compare.py --game connect4
    python scripts/level2_compare.py --game chess-mini --games 20
    python scripts/level2_compare.py --game connect4 --plots-only
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcts_rl.arena import play_match  # noqa: E402
from mcts_rl.plotting import INK_2, SERIES, errorbar_ci, plt, save, setup  # noqa: E402

CONFIGS = {
    "connect4": dict(az_sims=[25, 50, 100, 200], mcts_big=[1000, 3000, 10000], depth=None),
    "chess-mini": dict(az_sims=[16, 32, 64, 128], mcts_big=[10, 25, 50], depth=20),
}


def mcts(n, depth):
    spec = {"type": "mcts", "n_simulations": n}
    if depth:
        spec["max_rollout_depth"] = depth
    return spec


def az(ckpt, n, label=None):
    spec = {"type": "alphazero", "checkpoint": ckpt, "n_simulations": n}
    if label:
        spec["name"] = label
    return spec


def match(rows, tag, a, b, args):
    t0 = time.time()
    r = play_match(a, b, n_games=args.games, game=args.game, workers=args.workers, seed=len(rows) + 1)
    d = r.as_dict()
    d["tag"] = tag
    rows.append(d)
    print(f"  [{tag}] {r} ({time.time() - t0:.0f}s, A : {d['ms_per_move_a']:.0f} ms/coup)")
    return d


def run(args, out):
    cfg = CONFIGS[args.game]
    ckdir = args.checkpoints or os.path.join("checkpoints", args.game)
    final = os.path.join(ckdir, "latest.pt")
    iters = sorted(glob.glob(os.path.join(ckdir, "iter_*.pt")))
    depth = cfg["depth"]
    results = {"C1": [], "C2": [], "C3": []}

    print("== C1 : même nombre de simulations ==")
    for n in cfg["az_sims"]:
        d = match(results["C1"], f"n={n}", az(final, n), mcts(n, depth), args)
        d["x"] = n

    print("== C2 : AlphaZero(100) contre MCTS pur(n) ==")
    for n in cfg["mcts_big"]:
        d = match(results["C2"], f"AZ(100) vs MCTS({n})", az(final, 100), mcts(n, depth), args)
        d["x"] = n

    print("== C3 : ablations ==")
    match(results["C3"], "réseau non entraîné + MCTS(100) vs MCTS pur(100)",
          az(iters[0], 100, "AZ-iter0(100)"), mcts(100, depth), args)
    match(results["C3"], "réseau entraîné + MCTS(100) vs MCTS pur(100)", az(final, 100), mcts(100, depth), args)
    match(results["C3"], "réseau seul (sans recherche) vs aléatoire",
          {"type": "policy", "checkpoint": final}, {"type": "random"}, args)
    match(results["C3"], "réseau seul (sans recherche) vs MCTS pur(100)",
          {"type": "policy", "checkpoint": final}, mcts(100, depth), args)
    if depth:
        # MCTS sans réseau ET sans connaissance experte : rollouts aléatoires jusqu'à la fin de la partie
        match(results["C3"], "réseau entraîné + MCTS(100) vs MCTS pur(200) sans heuristique",
              az(final, 100), mcts(200, None) | {"name": "MCTS-rollouts-complets(200)"}, args)
        match(results["C3"], "MCTS pur(200) avec heuristique vs sans heuristique",
              mcts(200, depth), mcts(200, None) | {"name": "MCTS-rollouts-complets(200)"}, args)
    if len(iters) > 2:
        mid = iters[len(iters) // 2]
        match(results["C3"], f"final vs {os.path.basename(mid)} (AZ 100 sim.)",
              az(final, 100, "AZ-final"), az(mid, 100, "AZ-" + os.path.basename(mid)[:-3]), args)
        match(results["C3"], "final vs iter_000 (AZ 100 sim.)",
              az(final, 100, "AZ-final"), az(iters[0], 100, "AZ-iter0"), args)

    with open(os.path.join(out, "comparisons.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)


def plots(args, out):
    setup()
    ckdir = args.checkpoints or os.path.join("checkpoints", args.game)
    mpath = os.path.join(ckdir, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath, encoding="utf-8") as f:
            hist = json.load(f)["history"]
        it = [h["iteration"] for h in hist]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 3.8))
        ax1.plot(it, [h["loss_pi"] for h in hist], color=SERIES[0], label="policy (entropie croisée)")
        ax1.plot(it, [h["loss_v"] for h in hist], color=SERIES[1], label="value (MSE)")
        ax1.set(xlabel="itération", ylabel="perte", title="Pertes d'entraînement")
        ax1.legend()
        ev = [h for h in hist if "eval" in h]
        if ev:
            opponents = list(ev[0]["eval"].keys())
            for k, opp in enumerate(opponents):
                errorbar_ci(ax2, [h["iteration"] for h in ev], [h["eval"][opp] for h in ev],
                            SERIES[k], opp.replace(", c=1.41, random", ""))
            ax2.axhline(0.5, color=INK_2, lw=1, ls=":")
            ax2.set(xlabel="itération", ylabel="score d'AlphaZero", ylim=(-0.02, 1.02),
                    title="Évaluation pendant l'entraînement")
            ax2.legend(fontsize=8.5)
        ax3.plot(it, [h["selfplay_draw"] for h in hist], color=SERIES[2], label="nulles")
        ax3.plot(it, [h["selfplay_first_win"] for h in hist], color=SERIES[0], label="victoires 1er joueur")
        ax3.plot(it, [h["selfplay_second_win"] for h in hist], color=SERIES[1], label="victoires 2nd joueur")
        ax3.set(xlabel="itération", ylabel="proportion", ylim=(0, 1), title="Issues des parties de self-play")
        ax3.legend(fontsize=8.5)
        save(fig, os.path.join(out, "training_curves.png"))

    cpath = os.path.join(out, "comparisons.json")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            res = json.load(f)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
        errorbar_ci(ax1, [r["x"] for r in res["C1"]], res["C1"], SERIES[0], "AlphaZero(n) vs MCTS pur(n)")
        ax1.axhline(0.5, color=INK_2, lw=1, ls=":")
        ax1.set(xscale="log", xlabel="simulations par coup (les deux agents)", ylabel="score d'AlphaZero",
                ylim=(-0.02, 1.02), title="C1 · À nombre de simulations égal")
        errorbar_ci(ax2, [r["x"] for r in res["C2"]], res["C2"], SERIES[1], "AlphaZero(100) vs MCTS pur(n)")
        ax2.axhline(0.5, color=INK_2, lw=1, ls=":")
        ax2.set(xscale="log", xlabel="simulations du MCTS pur", ylabel="score d'AlphaZero(100)",
                ylim=(-0.02, 1.02), title="C2 · AlphaZero(100) contre MCTS pur(n)")
        for ax, rows in ((ax1, res["C1"]), (ax2, res["C2"])):
            xs = [r["x"] for r in rows]
            ax.minorticks_off()
            ax.set_xticks(xs, [str(v) for v in xs])
        save(fig, os.path.join(out, "comparisons.png"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--game", default="connect4", choices=CONFIGS)
    p.add_argument("--checkpoints", default=None)
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--plots-only", action="store_true")
    args = p.parse_args()
    out = os.path.join("results", "level2", args.game)
    os.makedirs(out, exist_ok=True)
    if not args.plots_only:
        run(args, out)
    plots(args, out)


if __name__ == "__main__":
    main()
