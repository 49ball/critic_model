"""Candidate-independent Transformer with explicit CV-agent interaction features."""

import torch
from torch import nn


class TrajectoryCritic(nn.Module):
    def __init__(self, d_model=64, layers=2, dropout=0.1):
        super().__init__()
        if d_model % 4 or layers < 1 or not 0 <= dropout < 1:
            raise ValueError("d_model divisible by 4, layers>=1, 0<=dropout<1 required")
        self.config = dict(d_model=d_model, layers=layers, dropout=dropout)
        self.agent_encoder = nn.Sequential(nn.Linear(8, d_model), nn.GELU())
        self.ego_encoder = nn.Sequential(nn.Linear(7, d_model), nn.GELU())
        self.trajectory_encoder = nn.Linear(8, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, 4, d_model * 2, dropout, batch_first=True, norm_first=True
        )
        self.temporal = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.cross = nn.MultiheadAttention(d_model, 4, dropout=dropout, batch_first=True)
        self.interaction = nn.Sequential(
            nn.Linear(4, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(d_model)
        )
        self.risk_head = nn.Linear(d_model, 3)
        self.quality_head = nn.Linear(d_model, 2)
        self.value_head = nn.Linear(d_model, 1)
        self.register_buffer(
            "agent_scale", torch.tensor([50.0, 50.0, 1.0, 1.0, 20.0, 20.0, 5.0, 3.0])
        )
        self.register_buffer("ego_scale", torch.tensor([20.0, 5.0, 3.0, 20.0, 5.0, 1.0, 1.0]))
        self.register_buffer(
            "traj_scale", torch.tensor([50.0, 50.0, 1.0, 1.0, 20.0, 5.0, 10.0, 6.4])
        )

    def forward(self, batch):
        a, f = batch["agents"], batch["trajectory"]
        b, k, t, _ = f.shape
        ego = self.ego_encoder(batch["ego"] / self.ego_scale)[:, None, :]
        tokens = torch.cat([ego, self.agent_encoder(a / self.agent_scale)], dim=1)
        context_mask = torch.cat(
            [torch.ones(b, 1, dtype=torch.bool, device=f.device), batch["agent_mask"]], dim=1
        )
        tokens = tokens[:, None].expand(-1, k, -1, -1).reshape(b * k, -1, tokens.shape[-1])
        padding = ~batch["time_mask"].reshape(b * k, t)
        # Dummy time token prevents all-masked attention on padded candidates.
        safe_padding = padding.clone()
        safe_padding[:, 0] = False
        traj = self.trajectory_encoder(f / self.traj_scale).reshape(b * k, t, -1)
        traj = self.temporal(traj, src_key_padding_mask=safe_padding)
        memory_mask = ~context_mask[:, None].expand(-1, k, -1).reshape(b * k, -1)
        attended, _ = self.cross(
            traj, tokens, tokens, key_padding_mask=memory_mask, need_weights=False
        )
        traj = traj + attended
        valid = (~padding).to(f.dtype)
        pooled = (traj * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        # [B,K,T,N,2]: each candidate against each actor's CV future.
        agent_future = a[:, None, None, :, 0:2] + (f[:, :, :, None, 7:8] * a[:, None, None, :, 4:6])
        relative = agent_future - f[:, :, :, None, 0:2]
        dist = relative.norm(dim=-1, keepdim=True)
        ego_radius = batch["ego"][:, 1:3].norm(dim=-1)[:, None, None, None, None] * 0.5
        actor_radius = a[..., 6:8].norm(dim=-1)[:, None, None, :, None] * 0.5
        pair = torch.cat(
            [relative / 50.0, dist / 50.0, (dist - ego_radius - actor_radius) / 10.0], dim=-1
        )
        embedded = self.interaction(pair)
        pair_mask = batch["time_mask"][:, :, :, None] & batch["agent_mask"][:, None, None, :]
        embedded = embedded.masked_fill(~pair_mask[..., None], -1e4)
        relation = embedded.amax(dim=(2, 3))
        any_pair = pair_mask.any(dim=(2, 3))
        relation = torch.where(any_pair[..., None], relation, torch.zeros_like(relation))
        fused = self.fusion(torch.cat([pooled.reshape(b, k, -1), relation], dim=-1))
        return {
            "risk_logits": self.risk_head(fused),
            "quality": self.quality_head(fused).sigmoid(),
            "utility": self.value_head(fused).squeeze(-1).sigmoid(),
        }
