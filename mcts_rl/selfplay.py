"""Boucle AlphaZero simplifiée : self-play -> entraînement -> évaluation.

    pour chaque itération :
      1. self-play : le réseau courant + MCTS (PUCT, bruit de Dirichlet, température)
         joue contre lui-même ; chaque position donne un exemple (s, π, z) où
         π = distribution des visites MCTS à la racine et z = résultat final
         du point de vue du joueur au trait ;
      2. entraînement : minimisation de (z - v)^2 - π·log p + L2 sur un buffer
         des dernières parties ;
      3. évaluation : matchs contre un joueur aléatoire et contre le MCTS pur.

Le self-play est "batché" : `games_in_parallel` parties avancent simultanément
et leurs feuilles sont évaluées en un seul passage dans le réseau (gros gain sur CPU).
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .agents.alphazero import AlphaZeroMCTS, PUCTNode, sample_action
from .arena import play_match
from .games import make_game
from .nn.model import NetConfig, NetEvaluator, PolicyValueNet


@dataclass
class TrainConfig:
    game: str = "connect4"
    out_dir: str = "checkpoints/connect4"
    iterations: int = 20
    games_per_iter: int = 128
    games_in_parallel: int = 32      # parties par processus (taille du batch d'évaluation)
    workers: int = 4                 # processus de self-play
    n_simulations: int = 100         # simulations MCTS par coup en self-play
    c_puct: float = 1.5
    dirichlet_alpha: float = 1.0
    noise_frac: float = 0.25
    temp_moves: int = 10             # τ = 1 pendant les premiers coups, puis τ -> 0
    buffer_iters: int = 6            # taille du replay buffer, en itérations
    batch_size: int = 256
    sample_reuse: float = 4.0        # nb moyen de passages sur chaque nouvel exemple
    lr: float = 1e-3
    weight_decay: float = 1e-4
    channels: int = 64
    blocks: int = 4
    eval_every: int = 2
    eval_games: int = 30
    eval_sims: int = 100             # simulations de l'agent AlphaZero en évaluation
    eval_opponents: List[dict] = field(default_factory=lambda: [
        {"type": "random"},
        {"type": "mcts", "n_simulations": 200},
        {"type": "mcts", "n_simulations": 1000},
    ])
    seed: int = 0


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------
def _sparse(pi: np.ndarray):
    idx = np.nonzero(pi)[0].astype(np.int32)
    return idx, pi[idx].astype(np.float32)


def self_play_games(net: PolicyValueNet, cfg: TrainConfig, n_games: int, seed: int):
    """Joue `n_games` parties de self-play en parallèle (batch). Renvoie les exemples et des stats."""
    rng = np.random.default_rng(seed)
    evaluator = NetEvaluator(net)
    mcts = AlphaZeroMCTS(evaluator, cfg.n_simulations, cfg.c_puct, cfg.dirichlet_alpha,
                         cfg.noise_frac, rng=rng)
    action_size = make_game(cfg.game).ACTION_SIZE

    states = [make_game(cfg.game) for _ in range(n_games)]
    histories = [[] for _ in range(n_games)]   # (planes, pi, joueur)
    move_idx = [0] * n_games
    samples, outcomes, lengths = [], [], []
    active = list(range(n_games))

    while active:
        # --- racines : évaluation batchée + bruit de Dirichlet
        roots = {g: PUCTNode(1.0, states[g]) for g in active}
        for g, (acts, pri, _) in zip(active, evaluator.batch([states[g] for g in active])):
            mcts.expand(roots[g], acts, pri)
            roots[g].N = 1
            mcts.add_root_noise(roots[g])
        # --- simulations : une feuille par arbre, évaluées toutes ensemble
        for _ in range(cfg.n_simulations):
            pending = []
            for g in active:
                path, value = mcts.select_leaf(roots[g])
                if value is None:
                    pending.append(path)
                else:
                    mcts.backup(path, value)
            if pending:
                for path, (acts, pri, v) in zip(pending, evaluator.batch([p[-1].state for p in pending])):
                    mcts.expand(path[-1], acts, pri)
                    mcts.backup(path, v)
        # --- on joue un coup dans chaque partie
        still_active = []
        for g in active:
            root, s = roots[g], states[g]
            pi = AlphaZeroMCTS.visit_policy(root, action_size)
            histories[g].append((s.encode(), pi, s.current_player))
            temp = 1.0 if move_idx[g] < cfg.temp_moves else 0.0
            states[g] = s.apply(sample_action(root, temp, rng))
            move_idx[g] += 1
            if states[g].is_terminal():
                w = states[g].winner()
                outcomes.append(w)
                lengths.append(move_idx[g])
                proto = states[g]
                for planes, pi_, player in histories[g]:
                    z = float(w * player)
                    for planes_s, pi_s in proto.symmetries(planes, pi_):
                        samples.append((planes_s, *_sparse(pi_s), z))
                histories[g] = []
            else:
                still_active.append(g)
        active = still_active
    return samples, outcomes, lengths


def _self_play_worker(args):
    state_dict, net_cfg, cfg_dict, n_games, seed = args
    torch.set_num_threads(1)
    net = PolicyValueNet(NetConfig(**net_cfg))
    net.load_state_dict(state_dict)
    cfg = TrainConfig(**cfg_dict)
    out_samples, out_outcomes, out_lengths = [], [], []
    remaining, k = n_games, 0
    while remaining > 0:
        n = min(cfg.games_in_parallel, remaining)
        s, o, l = self_play_games(net, cfg, n, seed + k)
        out_samples += s
        out_outcomes += o
        out_lengths += l
        remaining -= n
        k += 1
    return out_samples, out_outcomes, out_lengths


def generate_self_play(net: PolicyValueNet, cfg: TrainConfig, seed: int):
    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
    per_worker = [cfg.games_per_iter // cfg.workers + (1 if i < cfg.games_per_iter % cfg.workers else 0)
                  for i in range(cfg.workers)]
    jobs = [(state_dict, asdict(net.cfg), asdict(cfg), n, seed * 1000 + i * 97)
            for i, n in enumerate(per_worker) if n > 0]
    if cfg.workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            results = list(ex.map(_self_play_worker, jobs))
    else:
        results = [_self_play_worker(j) for j in jobs]
    samples, outcomes, lengths = [], [], []
    for s, o, l in results:
        samples += s
        outcomes += o
        lengths += l
    return samples, outcomes, lengths


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------
def _make_batch(buffer, idx, action_size):
    planes = torch.from_numpy(np.stack([buffer[i][0] for i in idx]))
    pi = torch.zeros(len(idx), action_size)
    for row, i in enumerate(idx):
        pi[row, torch.from_numpy(buffer[i][1]).long()] = torch.from_numpy(buffer[i][2])
    z = torch.tensor([buffer[i][3] for i in idx], dtype=torch.float32)
    return planes, pi, z


def train_on_buffer(net, optimizer, buffer, n_steps, batch_size, rng):
    net.train()
    action_size = net.cfg.action_size
    stats = {"loss": 0.0, "loss_pi": 0.0, "loss_v": 0.0}
    for _ in range(n_steps):
        idx = rng.integers(0, len(buffer), size=min(batch_size, len(buffer)))
        planes, pi, z = _make_batch(buffer, idx, action_size)
        logits, v = net(planes)
        loss_pi = -(pi * F.log_softmax(logits, dim=1)).sum(1).mean()
        loss_v = F.mse_loss(v, z)
        loss = loss_pi + loss_v
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stats["loss"] += loss.item()
        stats["loss_pi"] += loss_pi.item()
        stats["loss_v"] += loss_v.item()
    net.eval()
    return {k: v / max(1, n_steps) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------
def evaluate(checkpoint: str, cfg: TrainConfig, seed: int):
    res = {}
    for opp in cfg.eval_opponents:
        spec = {"type": "alphazero", "checkpoint": checkpoint, "n_simulations": cfg.eval_sims}
        m = play_match(spec, opp, n_games=cfg.eval_games, game=cfg.game, workers=cfg.workers, seed=seed)
        res[m.name_b] = m.as_dict()
        print(f"      {m}")
    return res


def run_training(cfg: TrainConfig, resume: bool = True):
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    metrics_path = os.path.join(cfg.out_dir, "metrics.json")
    latest = os.path.join(cfg.out_dir, "latest.pt")

    proto = make_game(cfg.game)
    if resume and os.path.exists(latest):
        net = PolicyValueNet.load(latest)
        with open(metrics_path, encoding="utf-8") as f:
            history = json.load(f)["history"]
        start = history[-1]["iteration"] + 1 if history else 1
        print(f"Reprise à l'itération {start}")
    else:
        net = PolicyValueNet(NetConfig.for_game(proto, cfg.channels, cfg.blocks))
        net.save(os.path.join(cfg.out_dir, "iter_000.pt"), iteration=0)
        net.save(latest, iteration=0)
        history, start = [], 1
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    buffer_chunks = deque(maxlen=cfg.buffer_iters)
    print(f"Réseau : {sum(p.numel() for p in net.parameters()):,} paramètres | config : {asdict(cfg)}")

    for it in range(start, cfg.iterations + 1):
        t0 = time.time()
        samples, outcomes, lengths = generate_self_play(net, cfg, seed=cfg.seed * 7919 + it)
        t_sp = time.time() - t0
        buffer_chunks.append(samples)
        buffer = [s for chunk in buffer_chunks for s in chunk]

        t1 = time.time()
        n_steps = max(1, math.ceil(cfg.sample_reuse * len(samples) / cfg.batch_size))
        losses = train_on_buffer(net, optimizer, buffer, n_steps, cfg.batch_size, rng)
        t_tr = time.time() - t1

        ckpt = os.path.join(cfg.out_dir, f"iter_{it:03d}.pt")
        net.save(ckpt, iteration=it)
        net.save(latest, iteration=it)

        o = np.array(outcomes)
        entry = {"iteration": it, "n_samples": len(samples), "buffer": len(buffer), "steps": n_steps,
                 **losses, "selfplay_first_win": float((o == 1).mean()), "selfplay_draw": float((o == 0).mean()),
                 "selfplay_second_win": float((o == -1).mean()), "avg_length": float(np.mean(lengths)),
                 "time_selfplay": t_sp, "time_train": t_tr}
        print(f"[it {it:3d}] {len(outcomes)} parties ({t_sp:.0f}s) | {len(samples)} ex. | buffer {len(buffer)} | "
              f"loss {losses['loss']:.3f} (π {losses['loss_pi']:.3f}, v {losses['loss_v']:.3f}) | "
              f"1er/nul/2nd {entry['selfplay_first_win']:.2f}/{entry['selfplay_draw']:.2f}/"
              f"{entry['selfplay_second_win']:.2f} | long. moy {entry['avg_length']:.1f}")

        if cfg.eval_every and (it % cfg.eval_every == 0 or it == cfg.iterations):
            t2 = time.time()
            entry["eval"] = evaluate(ckpt, cfg, seed=it)
            entry["time_eval"] = time.time() - t2
        history.append(entry)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"config": asdict(cfg), "history": history}, f, indent=1)
    return net, history
