"""
Full runnable example of the trajectory transformer encoder
with a main() that runs random test inputs.

This includes:
- TrajectoryTransformerEncoder
- Utility functions
- Random synthetic batch generation
- Forward pass
- Shape verification
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.append(str(project_root))


@dataclass
class TrajectoryEncoderConfig:
    vlm_embed_dim: int = 3584
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    T_max: int = 64
    use_deltas: bool = True
    pos_scale: float = 10.0
    vel_scale: float = 2.0
    clip: float = 2.0
    learned_pos_emb: bool = True


class TrajectoryTransformerEncoder(nn.Module):
    def __init__(self, cfg: TrajectoryEncoderConfig):
        super().__init__()
        self.cfg = cfg

        d_in = 6 if cfg.use_deltas else 4
        self.in_proj = nn.Linear(d_in, cfg.d_model)
        self.in_proj_layernorm = nn.LayerNorm(cfg.d_model)

        if cfg.learned_pos_emb:
            self.pos_emb = nn.Embedding(cfg.T_max, cfg.d_model)
        else:
            self.pos_emb = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)

        self.out_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.vlm_embed_dim),
            nn.LayerNorm(cfg.vlm_embed_dim),
        )

    def forward(
        self,
        feats: torch.Tensor,
        traj_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        h = self.in_proj(feats)

        if self.pos_emb is not None:
            B, T, _ = h.shape
            pos_ids = torch.arange(T, device=h.device).unsqueeze(0).expand(B, T)
            h = h + self.pos_emb(pos_ids)

        pad_mask = (traj_mask == 0)
        h_ctx = self.encoder(h, src_key_padding_mask=pad_mask)

        traj_tokens = self.out_proj(h_ctx)
        traj_tokens = traj_tokens * traj_mask.unsqueeze(-1)

        return traj_tokens


def main():
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Config
    cfg = TrajectoryEncoderConfig(
        T_max=64,
        d_model=512,
        num_layers=3,
        nhead=8,
        use_deltas=False,
    )

    encoder = TrajectoryTransformerEncoder(cfg).to(device)
    
    # use bf16 for faster inference, but need to convert input to float32
    # encoder = encoder.to(torch.bfloat16)

    # Simulate batch
    B = 4
    T_full = 64  # episode length (can be larger than T_max)

    # input_feats = torch.randn(B, T_full, 4).to(device)  # (x, y, z, delta_x, delta_y, delta_z)
    # traj_mask = torch.ones(B, T_full, device=device)

    # # Forward pass
    # traj_tokens, out_mask = encoder(input_feats, traj_mask)

    # breakpoint()

    input_feats = torch.tensor([[[ 0.0000e+00,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 7.5195e-02,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 1.5039e-01,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 2.2461e-01,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 3.0078e-01,  0.0000e+00,  8.8281e-01, -4.7266e-01],
         [ 3.7305e-01, -1.9409e-02,  8.8281e-01, -4.7266e-01],
         [ 4.2188e-01, -2.5879e-02,  1.0000e+00,  6.1149e-17],
         [ 4.2188e-01, -2.5879e-02,  2.7734e-01,  9.6094e-01],
         [ 4.3359e-01, -4.2419e-03,  2.7734e-01,  9.6094e-01]],

        [[ 0.0000e+00,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 0.0000e+00,  0.0000e+00,  1.3281e-01,  9.9219e-01],
         [-1.7700e-02,  1.7700e-02, -4.4727e-01,  8.9453e-01],
         [-1.0449e-01,  6.7871e-02, -6.5625e-01,  7.5781e-01],
         [-1.5234e-01,  1.2451e-01, -2.7734e-01,  9.6094e-01],
         [-1.5234e-01,  1.9922e-01,  0.0000e+00,  1.0000e+00],
         [-1.5234e-01,  2.9883e-01, -1.3281e-01,  9.9219e-01],
         [-1.9043e-01,  3.9062e-01, -2.7734e-01,  9.6094e-01],
         [-2.0215e-01,  4.1211e-01, -2.7734e-01,  9.6094e-01]],

        [[ 0.0000e+00,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 0.0000e+00,  0.0000e+00,  8.8281e-01, -4.7266e-01],
         [ 2.4170e-02, -6.4697e-03,  8.8281e-01, -4.7266e-01],
         [ 4.8340e-02, -1.2939e-02,  8.8281e-01, -4.7266e-01],
         [ 4.8340e-02, -1.2939e-02,  6.5625e-01, -7.5781e-01],
         [ 6.9824e-02, -2.5391e-02,  6.5625e-01, -7.5781e-01],
         [ 9.1797e-02, -3.7842e-02,  6.5625e-01, -7.5781e-01],
         [ 1.1328e-01, -5.0537e-02,  6.5625e-01, -7.5781e-01],
         [ 1.1328e-01, -5.0537e-02,  8.8281e-01, -4.7266e-01]],

        [[ 0.0000e+00,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 0.0000e+00,  0.0000e+00,  6.5625e-01,  7.5781e-01],
         [ 0.0000e+00,  0.0000e+00,  2.7734e-01,  9.6094e-01],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00],
         [ 0.0000e+00,  0.0000e+00, -2.7734e-01,  9.6094e-01],
         [ 0.0000e+00,  0.0000e+00, -6.5625e-01,  7.5781e-01],
         [-2.4170e-02,  6.4697e-03, -8.8281e-01,  4.7266e-01],
         [-7.3242e-02,  1.2939e-02, -1.0000e+00,  6.1149e-17],
         [-7.3242e-02,  1.2939e-02, -8.8281e-01, -4.7266e-01]],

        [[ 0.0000e+00,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 1.6895e-01,  3.1982e-02,  8.8281e-01,  4.7266e-01],
         [ 3.6719e-01,  4.4922e-02,  8.8281e-01, -4.7266e-01],
         [ 4.3555e-01,  1.3367e-02,  2.7734e-01,  9.6094e-01],
         [ 4.2773e-01,  1.3574e-01,  1.3281e-01,  9.9219e-01],
         [ 4.9805e-01,  1.8359e-01,  8.8281e-01, -4.7266e-01],
         [ 6.2891e-01,  7.8613e-02,  2.7734e-01, -9.6094e-01],
         [ 7.1484e-01,  5.0049e-03,  1.0000e+00,  6.1149e-17],
         [ 7.1484e-01,  5.0049e-03,  8.8281e-01,  4.7266e-01]],

        [[ 0.0000e+00,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 0.0000e+00,  0.0000e+00,  8.8281e-01,  4.7266e-01],
         [ 5.0049e-02,  1.6479e-02,  8.8281e-01,  4.7266e-01],
         [ 5.0049e-02,  1.6479e-02,  6.5625e-01,  7.5781e-01],
         [ 7.1777e-02,  2.8931e-02,  4.4727e-01,  8.9453e-01],
         [ 7.1777e-02,  2.8931e-02,  2.7734e-01,  9.6094e-01],
         [ 7.8125e-02,  5.2979e-02,  1.3281e-01,  9.9219e-01],
         [ 9.0820e-02,  7.4707e-02,  2.7734e-01,  9.6094e-01],
         [ 9.0820e-02,  7.4707e-02,  1.3281e-01,  9.9219e-01]],

        [[ 0.0000e+00,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00],
         [-4.8340e-02,  1.2939e-02, -8.8281e-01,  4.7266e-01],
         [-1.2158e-01, -0.0000e+00, -4.4727e-01, -8.9453e-01],
         [-2.1777e-01, -7.8125e-02, -6.5625e-01, -7.5781e-01],
         [-3.1445e-01, -9.0332e-02, -8.8281e-01, -4.7266e-01],
         [-4.1211e-01, -9.0332e-02, -1.0000e+00,  6.1149e-17],
         [-5.8594e-01, -9.0332e-02, -1.0000e+00,  6.1149e-17],
         [-6.1328e-01, -9.0332e-02, -1.0000e+00,  6.1149e-17]],

        [[ 0.0000e+00,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 2.5024e-02,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 5.0049e-02,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 5.0049e-02,  0.0000e+00,  8.8281e-01, -4.7266e-01],
         [ 5.0049e-02,  0.0000e+00,  6.5625e-01, -7.5781e-01],
         [ 7.1777e-02, -1.2512e-02,  6.5625e-01, -7.5781e-01],
         [ 9.3262e-02, -2.5024e-02,  6.5625e-01, -7.5781e-01],
         [ 1.1475e-01, -3.7598e-02,  6.5625e-01, -7.5781e-01],
         [ 1.1475e-01, -3.7598e-02,  8.8281e-01, -4.7266e-01]],

        [[ 0.0000e+00,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 0.0000e+00,  0.0000e+00,  4.4727e-01, -8.9453e-01],
         [ 6.4697e-03, -2.4170e-02, -1.2230e-16, -1.0000e+00],
         [-6.4697e-03, -9.7656e-02, -1.3281e-01, -9.9219e-01],
         [-3.2471e-02, -1.9434e-01, -1.3281e-01, -9.9219e-01],
         [-3.8818e-02, -2.1777e-01,  2.7734e-01, -9.6094e-01],
         [-3.4637e-03, -2.5391e-01,  6.5625e-01, -7.5781e-01],
         [ 6.6406e-02, -2.7930e-01,  8.8281e-01, -4.7266e-01],
         [ 9.0820e-02, -2.8516e-01,  8.8281e-01, -4.7266e-01]],

        [[ 0.0000e+00,  0.0000e+00,  1.0000e+00,  6.1149e-17],
         [ 0.0000e+00,  0.0000e+00,  8.8281e-01,  4.7266e-01],
         [ 0.0000e+00,  0.0000e+00,  4.4727e-01,  8.9453e-01],
         [ 0.0000e+00,  0.0000e+00,  2.7734e-01,  9.6094e-01],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00],
         [ 0.0000e+00,  0.0000e+00, -1.3281e-01,  9.9219e-01],
         [ 0.0000e+00,  0.0000e+00, -4.4727e-01,  8.9453e-01],
         [-2.1606e-02,  1.2512e-02, -6.5625e-01,  7.5781e-01],
         [-2.1606e-02,  1.2512e-02, -4.4727e-01,  8.9453e-01]]],
       device='cuda:0', dtype=torch.bfloat16)
    traj_mask = torch.tensor([[True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True, True, True]],
       device='cuda:0')

    traj_tokens, out_mask = encoder(input_feats, traj_mask)

    breakpoint()




if __name__ == "__main__":
    main()
