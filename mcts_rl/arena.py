"""Faire jouer deux agents l'un contre l'autre, en série ou en parallèle."""
from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional

from .agents import Agent, make_agent
from .games import make_game


def play_game(agent_first: Agent, agent_second: Agent, game: str = "connect4",
              verbose: bool = False, state=None) -> dict:
    """Joue une partie. Renvoie le vainqueur (+1 = premier joueur, -1 = second, 0 = nul)."""
    state = state if state is not None else make_game(game)
    agents = {1: agent_first, -1: agent_second}
    for a in agents.values():
        a.reset()
    times = {1: 0.0, -1: 0.0}
    n_moves = 0
    if verbose:
        print(state.render(), "\n")
    while not state.is_terminal():
        player = state.current_player
        t0 = time.perf_counter()
        action = agents[player].select_action(state)
        times[player] += time.perf_counter() - t0
        if verbose:
            print(f"{agents[player].name} ({'1er' if player == 1 else '2nd'}) joue {state.action_to_str(action)}")
        state = state.apply(action)
        n_moves += 1
        if verbose:
            print(state.render(), "\n")
    w = state.winner()
    if verbose:
        print("Résultat :", {1: f"victoire de {agent_first.name}", -1: f"victoire de {agent_second.name}",
                             0: "match nul"}[w])
    return {"winner": w, "moves": n_moves, "time_first": times[1], "time_second": times[-1]}


@dataclass
class MatchResult:
    name_a: str
    name_b: str
    wins: int = 0
    draws: int = 0
    losses: int = 0
    wins_as_first: int = 0
    games_as_first: int = 0
    total_moves: int = 0
    time_a: float = 0.0
    moves_a: int = 0
    details: List[int] = field(default_factory=list)

    @property
    def n(self):
        return self.wins + self.draws + self.losses

    @property
    def win_rate(self):
        return self.wins / self.n if self.n else 0.0

    @property
    def score(self):
        """Score type échecs : victoire = 1, nul = 0.5."""
        return (self.wins + 0.5 * self.draws) / self.n if self.n else 0.0

    def ci95(self):
        """Intervalle de confiance à 95 % (Wilson) sur le score."""
        n, p, z = self.n, self.score, 1.96
        if n == 0:
            return (0.0, 1.0)
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))

    def as_dict(self):
        lo, hi = self.ci95()
        return {"a": self.name_a, "b": self.name_b, "n": self.n, "wins": self.wins, "draws": self.draws,
                "losses": self.losses, "win_rate": self.win_rate, "score": self.score,
                "ci95": [lo, hi], "avg_moves": self.total_moves / max(1, self.n),
                "ms_per_move_a": 1000 * self.time_a / max(1, self.moves_a)}

    def __str__(self):
        lo, hi = self.ci95()
        return (f"{self.name_a} vs {self.name_b} : {self.wins}V / {self.draws}N / {self.losses}D "
                f"sur {self.n} parties | score = {self.score:.3f} [IC95 {lo:.2f}-{hi:.2f}]")


def _play_one(args):
    spec_a, spec_b, game, a_first, seed = args
    a = make_agent(spec_a, seed=seed)
    b = make_agent(spec_b, seed=seed + 1_000_003)
    if a_first:
        r = play_game(a, b, game)
        result_a, t_a = r["winner"], r["time_first"]
        moves_a = (r["moves"] + 1) // 2
    else:
        r = play_game(b, a, game)
        result_a, t_a = -r["winner"], r["time_second"]
        moves_a = r["moves"] // 2
    return result_a, a_first, r["moves"], t_a, moves_a, a.name, b.name


def _init_worker():
    import torch
    torch.set_num_threads(1)


def play_match(spec_a: dict, spec_b: dict, n_games: int = 100, game: str = "connect4",
               workers: Optional[int] = None, seed: int = 0, swap_colors: bool = True) -> MatchResult:
    """Match de `n_games` parties entre A et B (couleurs alternées). Statistiques du point de vue de A."""
    jobs = [(spec_a, spec_b, game, (i % 2 == 0) or not swap_colors, seed * 100_000 + i)
            for i in range(n_games)]
    workers = workers if workers is not None else max(1, (os.cpu_count() or 2) - 1)
    if workers > 1 and n_games > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            outcomes = list(ex.map(_play_one, jobs, chunksize=max(1, n_games // (4 * workers))))
    else:
        outcomes = [_play_one(j) for j in jobs]

    res = MatchResult(outcomes[0][5], outcomes[0][6])
    for result_a, a_first, moves, t_a, moves_a, _, _ in outcomes:
        res.details.append(result_a)
        res.total_moves += moves
        res.time_a += t_a
        res.moves_a += moves_a
        if a_first:
            res.games_as_first += 1
        if result_a > 0:
            res.wins += 1
            res.wins_as_first += int(a_first)
        elif result_a < 0:
            res.losses += 1
        else:
            res.draws += 1
    return res
