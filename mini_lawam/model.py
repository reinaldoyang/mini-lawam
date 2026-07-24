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
frozen; only ConvPrior + the MLP action head train. Training can be joint or
two-phase (see mini_lawam.train --phase): phase 1 trains ConvPrior alone with
loss_distill (forward(..., prior_only=True)); phase 2 loads that prior
(frozen or finetuned) and trains the action head with the full loss.

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
    # Horizon in FRAMES = 1.2 s @ 20 Hz = 24 (paper §C.5 robot horizon). Same value
    # for the action chunk (MLP head output) and the LaWM future pair (o_{t+H} ->
    # z_teacher, u_T, loss_wm).
    action_horizon: int = 24
    future_horizon: int = 24
    use_state: bool = False          # feed proprioception (current eef_pos) to the head
    state_dim: int = 0               # e.g. 3 for [x,y,z]; set with use_state
    use_wrist: bool = False           # add wrist_cam as aux view to the ACTION HEAD only
                                      # (never the prior/LaWM -- paper §C.2; wrist moves w/ arm)
    head_type: str = "mlp"           # "mlp" = pooled-features MLP (v0);
                                      # "attn" = token-level cross-attention (no pooling)
    target_mode: str = "abs"         # "abs" = absolute eef positions;
                                      # "delta" = pos[t+i]-pos[t] (servo-like at deploy)
    hidden: int = 512                # MLP head width
    attn_hidden: int = 384           # attn head width
    attn_layers: int = 3
    attn_heads: int = 6
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


class _CrossAttnBlock(nn.Module):
    """Pre-norm decoder block: queries self-attend, then cross-attend to context."""

    def __init__(self, hidden: int, n_heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(hidden)
        self.sa = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(hidden)
        self.ca = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.n3 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.GELU(), nn.Linear(hidden * 4, hidden),
        )

    def forward(self, q: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        qn = self.n1(q)
        q = q + self.sa(qn, qn, qn, need_weights=False)[0]
        q = q + self.ca(self.n2(q), ctx, ctx, need_weights=False)[0]
        q = q + self.ffn(self.n3(q))
        return q


class AttnActionHead(nn.Module):
    """Token-level cross-attention head (no mean-pooling).

    One learned query per output timestep cross-attends to the DINO patch tokens
    of every view (u_t, u_hat_T, [wrist]) -- so the head reads *where* things are
    (arm/egg patches) instead of a single averaged vector. Optional proprioception
    (current eef_pos) enters as an extra context token. Deterministic; MSE loss.
    """

    def __init__(self, token_dim: int, action_dim: int, horizon: int,
                 n_views: int, hidden: int = 384, n_layers: int = 3,
                 n_heads: int = 6, state_dim: int = 0):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.in_proj = nn.Linear(token_dim, hidden)        # DINO token 768 -> hidden
        self.view_emb = nn.Parameter(torch.zeros(n_views, hidden))   # per-view tag
        self.queries = nn.Parameter(torch.zeros(horizon, hidden))    # per-step query
        self.state_proj = nn.Linear(state_dim, hidden) if state_dim > 0 else None
        self.layers = nn.ModuleList(
            [_CrossAttnBlock(hidden, n_heads) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, action_dim)
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.view_emb, std=0.02)

    def forward(self, view_tokens, state: Optional[torch.Tensor] = None) -> torch.Tensor:
        # view_tokens: list of [B, K, token_dim] (one per view, in a fixed order)
        b = view_tokens[0].shape[0]
        ctx = [self.in_proj(v) + self.view_emb[i] for i, v in enumerate(view_tokens)]
        ctx = torch.cat(ctx, dim=1)                        # [B, n_views*K, hidden]
        if self.state_proj is not None and state is not None:
            ctx = torch.cat([self.state_proj(state).unsqueeze(1), ctx], dim=1)
        q = self.queries.unsqueeze(0).expand(b, -1, -1)    # [B, horizon, hidden]
        for layer in self.layers:
            q = layer(q, ctx)
        return self.out(self.norm(q))                      # [B, horizon, action_dim]


class MiniLaWAM(nn.Module):
    def __init__(self, cfg: MiniLaWAMConfig):
        super().__init__()
        self.cfg = cfg
        self.lam = load_latent_action_model(cfg.lam_ckpt, cfg.lam_yaml)  # frozen, eval
        vdim = int(self.lam.input_dim)   # DINOv3 ViT-B -> 768
        cdim = int(self.lam.code_dim)    # LAM latent action dim -> 32
        self.prior = ConvPrior(vdim, cdim)
        n_views = 2 + (1 if cfg.use_wrist else 0)   # u_t, u_hat_T, [wrist]
        state_dim = cfg.state_dim if cfg.use_state else 0
        if cfg.head_type == "attn":
            self.action_head = AttnActionHead(
                token_dim=vdim, action_dim=cfg.action_dim, horizon=cfg.action_horizon,
                n_views=n_views, hidden=cfg.attn_hidden, n_layers=cfg.attn_layers,
                n_heads=cfg.attn_heads, state_dim=state_dim,
            )
        elif cfg.head_type == "mlp":
            # cond = [pool(u_t), pool(u_hat_T), (pool(wrist)), (state)]
            cond_dim = n_views * vdim + state_dim
            self.action_head = MLPActionHead(cond_dim, cfg.action_dim,
                                             cfg.action_horizon, cfg.hidden)
        else:
            raise ValueError(f"unknown head_type {cfg.head_type!r} (use 'mlp' or 'attn')")

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

    def _action_pred(self, u_t_tok, u_hat_tok, wrist=None, state=None):
        """(u_t, u_hat_T) patch tokens [B,K,D] -> action chunk [B,H,action_dim].

        Branches on cfg.head_type: 'attn' consumes tokens directly (no pooling);
        'mlp' mean-pools each view first. Wrist/state added if configured.
        """
        wrist_tok = None
        if self.cfg.use_wrist:
            assert wrist is not None, "cfg.use_wrist=True but no wrist image was passed"
            wrist_tok = self._feat(wrist)[:, 0]         # [B,K,D]
        st = state if (self.cfg.use_state and state is not None) else None
        if self.cfg.head_type == "attn":
            views = [u_t_tok, u_hat_tok]
            if wrist_tok is not None:
                views.append(wrist_tok)
            return self.action_head(views, state=st)
        cond = torch.cat([u_t_tok.mean(1), u_hat_tok.mean(1)], dim=-1)
        if wrist_tok is not None:
            cond = torch.cat([cond, wrist_tok.mean(1)], dim=-1)
        if st is not None:
            cond = torch.cat([cond, st], dim=-1)
        return self.action_head(cond)

    def forward(
        self,
        o_t: torch.Tensor,          # [B, 1, 3, 256, 256]
        o_T: torch.Tensor,          # [B, 1, 3, 256, 256]
        actions: Optional[torch.Tensor] = None,       # [B, H, action_dim] (unused if prior_only)
        actions_mask: Optional[torch.Tensor] = None,  # [B, H, action_dim] or None
        state: Optional[torch.Tensor] = None,         # [B, state_dim] or None
        wrist: Optional[torch.Tensor] = None,         # [B, 1, 3, 256, 256] or None (aux view)
        prior_only: bool = False,   # Phase 1: only L_distill; skip decoder + action head
    ):
        u_t = self._feat(o_t)[:, :1]                    # [B,1,K,D] (grad-usable constant)
        pair = torch.cat([o_t, o_T], dim=1)             # [B,2,3,256,256]
        z_teacher, u_T_target = self._teacher(pair)     # [B,1,code], [B,1,K,D]

        z_hat = self.prior(u_t[:, 0])                   # [B,1,code]
        loss_distill = F.mse_loss(z_hat, z_teacher)

        if prior_only:
            zero = z_hat.new_zeros(())
            return {"loss_total": loss_distill, "loss_act": zero,
                    "loss_distill": loss_distill, "loss_wm": zero, "pred": None}

        assert actions is not None, "actions required unless prior_only=True"
        u_hat_T = self.lam.decoder(u_t, z_hat)          # [B,1,K,D]
        if isinstance(u_hat_T, tuple):
            u_hat_T = u_hat_T[0]
        loss_wm = F.mse_loss(u_hat_T, u_T_target)

        pred = self._action_pred(u_t[:, 0], u_hat_T[:, 0], wrist=wrist, state=state)

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
    def predict(self, o_t: torch.Tensor, state: Optional[torch.Tensor] = None,
                wrist: Optional[torch.Tensor] = None,
                return_subgoal: bool = False):
        """Inference: current frame (+ optional wrist view) -> action chunk.

        No future frame needed. Pass `wrist` [B,1,3,256,256] iff cfg.use_wrist.
        With `return_subgoal=True`, also return the current and predicted-subgoal
        DINO patch tokens as `(pred, u_t, u_hat_T)` without recomputing features.
        """
        u_t = self._feat(o_t)[:, :1]
        z_hat = self.prior(u_t[:, 0])
        u_hat_T = self.lam.decoder(u_t, z_hat)
        if isinstance(u_hat_T, tuple):
            u_hat_T = u_hat_T[0]
        pred = self._action_pred(u_t[:, 0], u_hat_T[:, 0], wrist=wrist, state=state)
        if return_subgoal:
            return pred, u_t[:, 0], u_hat_T[:, 0]
        return pred


if __name__ == "__main__":
    # Shape smoke test: both head types x wrist on/off x state on/off.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for head_type in ("mlp", "attn"):
        for use_wrist in (False, True):
            for use_state in (False, True):
                cfg = MiniLaWAMConfig(head_type=head_type, use_wrist=use_wrist,
                                      use_state=use_state, state_dim=3 if use_state else 0)
                model = MiniLaWAM(cfg).to(dev)
                model.prior.train(); model.action_head.train()
                B, H = 2, cfg.action_horizon
                o_t = torch.randn(B, 1, 3, 256, 256, device=dev)
                o_T = torch.randn(B, 1, 3, 256, 256, device=dev)
                wrist = torch.randn(B, 1, 3, 256, 256, device=dev) if use_wrist else None
                state = torch.randn(B, 3, device=dev) if use_state else None
                acts = torch.randn(B, H, cfg.action_dim, device=dev)
                out = model(o_t, o_T, acts, wrist=wrist, state=state)
                out["loss_total"].backward()  # check gradients flow to the head
                gh = sum(p.grad.abs().sum().item() for p in model.action_head.parameters()
                         if p.grad is not None)
                n = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"[{head_type} wrist={use_wrist} state={use_state}] "
                      f"pred={tuple(out['pred'].shape)} act={float(out['loss_act']):.3f} "
                      f"head_grad={gh:.1f} trainable={n:,}")
