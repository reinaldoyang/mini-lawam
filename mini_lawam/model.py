"""Minimal LaWAM-inspired behavior-cloning policy (vision-only, v0).

Structure (your diagram):
    o_t --DINO(frozen)--> u_t --ConvPrior--> z_hat --LaWM(frozen)--> subgoal u_hat_T
                          u_t ---------------------------------------------\
                                                                           v
                          [pool(u_t), pool(u_hat_T), (state)] --MLP--> action chunk

Losses:
    loss_act     = MSE(pred_actions, actions)                 # behavior cloning
    loss_distill = MSE(z_hat, z_teacher)                      # teacher = frozen LAM IDM(u_t,u_T)
    loss_wm      = MSE(u_hat_T, u_T)                          # subgoal supervision (light)

Everything in the LAM (DINO encoder, inverse-dynamics teacher, LaWM decoder) is
frozen; only ConvPrior + the MLP action head train.

Inputs are expected already preprocessed to the LAM's contract:
    o_t, o_T: float [B, 1, 3, 256, 256], ImageNet-normalized 256x256
    (use latent_action_model.data_loader.video_aug.gpu_two_view_video_aug(..., training=False)).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from latent_action_model.core.lam_model import load_latent_action_model


@dataclass
class MiniLaWAMConfig:
    lam_ckpt: str = "latent_action_model/logs/dino_large_vae/lam_release/checkpoints/pytorch_model.pt"
    lam_yaml: str = "latent_action_model/logs/dino_large_vae/lam_release/dino_large_vae.yaml"
    action_dim: int = 4              # target = absolute [eef_pos(3), gripper_pos(1)]
    action_horizon: int = 32          # 1.6 s @ 20 Hz; keep == frame gap used for pairs
    use_state: bool = False
    state_dim: int = 0
    hidden: int = 512
    lambda_distill: float = 1.0
    lambda_wm: float = 0.1            # subgoal supervision weight (0 to disable)


class ConvPrior(nn.Module):
    """u_t tokens [B, 256, D] -> latent action z_hat [B, 1, code_dim].

    Small conv net over the 16x16 DINO token grid (the ResNet-style box in the
    diagram). Swap for token self-attention later if desired -- same interface.
    """

    def __init__(self, in_dim: int, code_dim: int, grid: int = 16):
        super().__init__()
        self.grid = grid
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 256, 3, padding=1), nn.GroupNorm(8, 256), nn.GELU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.GELU(),  # 16->8
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.GroupNorm(8, 256), nn.GELU(),  # 8->4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, code_dim),
        )

    def forward(self, u_t: torch.Tensor) -> torch.Tensor:  # u_t [B, K, D]
        b, k, d = u_t.shape
        x = u_t.transpose(1, 2).reshape(b, d, self.grid, self.grid)
        return self.net(x).unsqueeze(1)  # [B, 1, code_dim]


class MLPActionHead(nn.Module):
    """[pooled conditioning] -> flat action chunk [B, H, action_dim] (v0: MSE)."""

    def __init__(self, in_dim: int, action_dim: int, horizon: int, hidden: int = 512):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, horizon * action_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        b = cond.shape[0]
        return self.net(cond).view(b, self.horizon, self.action_dim)


class MiniLaWAM(nn.Module):
    def __init__(self, cfg: MiniLaWAMConfig):
        super().__init__()
        self.cfg = cfg
        self.lam = load_latent_action_model(cfg.lam_ckpt, cfg.lam_yaml)  # frozen, eval
        vdim = int(self.lam.input_dim)   # DINOv3 ViT-B -> 768
        cdim = int(self.lam.code_dim)    # LAM latent action dim -> 32
        self.prior = ConvPrior(vdim, cdim)
        cond_dim = 2 * vdim + (cfg.state_dim if cfg.use_state else 0)
        self.action_head = MLPActionHead(cond_dim, cfg.action_dim, cfg.action_horizon, cfg.hidden)

    def _feat(self, imgs: torch.Tensor) -> torch.Tensor:
        # no_grad frozen DINO features, usable as constants in the autograd graph.
        return self.lam.extract_vision_features(imgs)  # [B, T, K, D]

    def _teacher(self, pair: torch.Tensor):
        # Frozen LAM inverse-dynamics teacher on (o_t, o_T); returns z and u_T targets.
        out = self.lam.get_latent_action(
            videos=pair, states=None, dec_videos=pair, predict_future_frame=False,
        )
        # get_latent_action runs under inference_mode -> materialize normal tensors.
        return out["quantized"].detach().clone(), out["tgt"].detach().clone()

    def forward(
        self,
        o_t: torch.Tensor,          # [B, 1, 3, 256, 256]
        o_T: torch.Tensor,          # [B, 1, 3, 256, 256]
        actions: torch.Tensor,      # [B, H, action_dim]
        actions_mask: Optional[torch.Tensor] = None,  # [B, H, action_dim] or None
        state: Optional[torch.Tensor] = None,         # [B, state_dim] or None
    ):
        u_t = self._feat(o_t)[:, :1]                    # [B,1,K,D] (grad-usable constant)
        pair = torch.cat([o_t, o_T], dim=1)             # [B,2,3,256,256]
        z_teacher, u_T_target = self._teacher(pair)     # [B,1,code], [B,1,K,D]

        z_hat = self.prior(u_t[:, 0])                   # [B,1,code]
        loss_distill = F.mse_loss(z_hat, z_teacher)

        u_hat_T = self.lam.decoder(u_t, z_hat)          # [B,1,K,D]
        if isinstance(u_hat_T, tuple):
            u_hat_T = u_hat_T[0]
        loss_wm = F.mse_loss(u_hat_T, u_T_target)

        cond = torch.cat([u_t[:, 0].mean(1), u_hat_T[:, 0].mean(1)], dim=-1)  # [B, 2D]
        if self.cfg.use_state and state is not None:
            cond = torch.cat([cond, state], dim=-1)
        pred = self.action_head(cond)                   # [B,H,action_dim]

        if actions_mask is None:
            loss_act = F.mse_loss(pred, actions)
        else:
            m = actions_mask.to(pred.dtype)
            loss_act = ((pred - actions) ** 2 * m).sum() / m.sum().clamp_min(1.0)

        total = loss_act + self.cfg.lambda_distill * loss_distill + self.cfg.lambda_wm * loss_wm
        return {
            "loss_total": total,
            "loss_act": loss_act,
            "loss_distill": loss_distill,
            "loss_wm": loss_wm,
            "pred": pred,
        }

    @torch.no_grad()
    def predict(self, o_t: torch.Tensor, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Inference: current frame -> action chunk (no future frame needed)."""
        u_t = self._feat(o_t)[:, :1]
        z_hat = self.prior(u_t[:, 0])
        u_hat_T = self.lam.decoder(u_t, z_hat)
        if isinstance(u_hat_T, tuple):
            u_hat_T = u_hat_T[0]
        cond = torch.cat([u_t[:, 0].mean(1), u_hat_T[:, 0].mean(1)], dim=-1)
        if self.cfg.use_state and state is not None:
            cond = torch.cat([cond, state], dim=-1)
        return self.action_head(cond)


if __name__ == "__main__":
    # Shape smoke test on random ImageNet-normalized inputs.
    cfg = MiniLaWAMConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniLaWAM(cfg).to(dev)
    model.prior.train(); model.action_head.train()
    B, H = 2, cfg.action_horizon
    o_t = torch.randn(B, 1, 3, 256, 256, device=dev)
    o_T = torch.randn(B, 1, 3, 256, 256, device=dev)
    acts = torch.randn(B, H, cfg.action_dim, device=dev)
    out = model(o_t, o_T, acts)
    print({k: round(float(v), 4) for k, v in out.items() if k != "pred"})
    print("pred", tuple(out["pred"].shape))
    print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))
