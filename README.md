# MCTS et Self-Play pour jeux à deux joueurs (Puissance 4 → Échecs)

Agent de jeu à information complète fondé sur **Monte Carlo Tree Search**, d'abord
« pur » (UCT + rollouts) sur Puissance 4, puis guidé par un **réseau policy/value
entraîné par self-play** (AlphaZero simplifié) sur une variante allégée des échecs
(et, en bonus, sur Puissance 4 pour une comparaison nette avec/sans réseau).

Le rapport complet (méthode, expériences, résultats, analyse) est dans **[RAPPORT.md](RAPPORT.md)**.

## Résultats en bref

| | résultat |
|---|---|
| MCTS pur vs aléatoire (Puissance 4) | 99,5 % de score dès 50 simulations, 100 % dès 200 |
| Constante d'exploration c | plateau entre 1 et 3 ; c = 0,05 → 5 % de score contre c = √2 |
| Rollout heuristique vs aléatoire | meilleur à simulations égales (0,78), moins bon à temps égal (0,41) |
| **Puissance 4 : AlphaZero vs MCTS pur, simulations égales** | **0,79 à 0,93** |
| Puissance 4 : AlphaZero(100) vs MCTS pur(10 000) | 0,55, avec environ 15 à 20 fois moins de temps de calcul |
| Puissance 4 : réseau final vs réseau initial | 40/40 |
| Échecs mini : réseau final vs réseau initial | 20/20 ; 0,85 contre l'aléatoire |
| Échecs mini : AlphaZero vs MCTS pur avec heuristique matérielle | 0,27 à 0,50 : le budget CPU (1 920 parties) ne suffit pas pour le dépasser |

## Installation

```bash
pip install -r requirements.txt      # numpy, torch (CPU suffit), chess (python-chess), matplotlib, pytest
python -m pytest -q                  # 26 tests
```

## Jouer

```bash
# Humain (X, commence) contre MCTS pur 2000 simulations — on tape le numéro de colonne 0..6
python scripts/play.py --p1 human --p2 mcts:2000

# Humain contre l'agent AlphaZero entraîné
python scripts/play.py --p1 human --p2 az:checkpoints/connect4/latest.pt:200

# Échecs (variante "mini") : coups en UCI (e2e4) ou SAN (Nf3)
python scripts/play.py --game chess-mini --p1 human --p2 az:checkpoints/chess-mini/latest.pt:100

# Match entre deux agents sur 50 parties (couleurs alternées, en parallèle)
python scripts/play.py --p1 mcts:500 --p2 random --games 50
```

Joueurs disponibles : `human`, `random`, `mcts[:N[:c]]`, `mcts-h[:N[:c]]` (rollouts heuristiques),
`az:CHECKPOINT[:N]` (MCTS + réseau), `policy:CHECKPOINT` (réseau seul, sans recherche).
Jeux : `connect4`, `chess-mini`, `chess-micro`, `chess-standard`.

## Reproduire les expériences

```bash
# Niveau 1 — MCTS pur sur Puissance 4 (E1..E5, figures dans results/level1/)
python scripts/level1_experiments.py --games 100

# Niveau 2 — entraînement par self-play puis comparaison avec / sans réseau
python scripts/train.py --preset connect4          # ~45 min sur un CPU 4 cœurs
python scripts/train.py --preset chess-mini        # ~1 h
python scripts/level2_compare.py --game connect4
python scripts/level2_compare.py --game chess-mini --games 20
```

`train.py` reprend automatiquement depuis `latest.pt` (`--no-resume` pour repartir de zéro).
Toutes les options des préréglages se surchargent en ligne de commande (`--iterations`, `--sims`, ...).

## Organisation du code

```
mcts_rl/
  games/
    base.py          interface GameState commune (joueurs +1 / -1, états immuables)
    connect4.py      Puissance 4 en bitboard (moteur minimal standard)
    chess_game.py    échecs via python-chess + encodage actions (4096) / plans (17x8x8) + variantes
  agents/
    base.py          Agent, RandomAgent, HumanAgent
    mcts.py          MCTS pur : UCT (UCB1), rollouts aléatoires / heuristiques / tronqués
    alphazero.py     MCTS PUCT guidé par réseau, bruit de Dirichlet, température ; agent "réseau seul"
  nn/model.py        ResNet policy + value, évaluation par lots
  selfplay.py        boucle AlphaZero : self-play batché multi-processus -> entraînement -> évaluation
  arena.py           parties et matchs (parallèles), score + intervalle de confiance de Wilson
scripts/             play.py, train.py, level1_experiments.py, level2_compare.py
tests/               tests unitaires (règles, MCTS, réseau, self-play)
results/             figures et JSON des expériences
checkpoints/         réseaux entraînés (latest.pt, iter_000.pt = réseau non entraîné)
```
