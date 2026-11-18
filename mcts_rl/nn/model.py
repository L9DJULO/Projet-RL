"""Réseau policy + value façon AlphaZero (ResNet réduit)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..games.base import GameState


@dataclass
class NetConfig:
    in_channels: int
    height: int
    width: int
    action_size: int
    channels: int = 64
    blocks: int = 4
    # 'fc'      : tête policy = conv 1x1 -> couche dense (Puissance 4)
    # 'from_to' : tête policy entièrement convolutive : la sortie (64, 8, 8) donne
    #             logits[case_départ, case_arrivée] (échecs, 4096 actions)
    policy_head: str = "fc"

    @staticmethod
    def for_game(state: GameState, channels: int = 64, blocks: int = 4) -> "NetConfig":
        c, h, w = state.ENCODED_SHAPE
        head = "from_to" if state.ACTION_SIZE == (h * w) ** 2 else "fc"
        return NetConfig(c, h, w, state.ACTION_SIZE, channels, blocks, head)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.cfg = cfg
        ch, hw = cfg.channels, cfg.height * cfg.width
        self.stem = nn.Sequential(nn.Conv2d(cfg.in_channels, ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(ch), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(cfg.blocks)])
        if cfg.policy_head == "from_to":
            assert cfg.action_size == hw * hw
            self.policy = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(),
                                        nn.Conv2d(ch, hw, 1))
        else:
            self.policy = nn.Sequential(nn.Conv2d(ch, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(),
                                        nn.Flatten(), nn.Linear(2 * hw, cfg.action_size))
        self.value = nn.Sequential(nn.Conv2d(ch, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(),
                                   nn.Flatten(), nn.Linear(hw, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.body(self.stem(x))
        p = self.policy(h)
        if self.cfg.policy_head == "from_to":
            # (B, to, H, W) -> (B, from=H*W, to) -> (B, from*64 + to)
            p = p.flatten(2).transpose(1, 2).reshape(x.shape[0], -1)
        v = self.value(h).squeeze(-1)
        return p, v

    # ------------------------------------------------------------------
    def save(self, path: str, **extra):
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict(), **extra}, path)

    @staticmethod
    def load(path: str, map_location="cpu") -> "PolicyValueNet":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        net = PolicyValueNet(NetConfig(**ckpt["config"]))
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        return net


class NetEvaluator:
    """Enveloppe d'inférence : état -> (priors sur les coups légaux, valeur)."""

    def __init__(self, net: PolicyValueNet):
        self.net = net.eval()

    def __call__(self, state: GameState):
        return self.batch([state])[0]

    @torch.inference_mode()
    def batch(self, states):
        """Évalue plusieurs positions en un seul passage dans le réseau."""
        x = torch.from_numpy(np.stack([s.encode() for s in states]))
        logits, values = self.net(x)
        logits, values = logits.numpy(), values.numpy()
        out = []
        for i, s in enumerate(states):
            legal = s.legal_actions()
            lg = logits[i, legal]
            p = np.exp(lg - lg.max())
            out.append((legal, p / p.sum(), float(values[i])))
        return out
