# ---- shared model code: identical in training and submission notebooks ----
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

def to_25d(x, slice_idx):
    """x: [B, S, N, H, W] uint8 -> [B, S, n, 3, H, W] float, channels = slices (i-1, i, i+1)."""
    N = x.shape[2]
    idx = torch.as_tensor(slice_idx, device=x.device)
    trip = torch.stack([(idx - 1).clamp(0, N - 1), idx, (idx + 1).clamp(0, N - 1)], 1)  # [n, 3]
    out = x[:, :, trip]  # [B, S, n, 3, H, W]
    return out.float().div_(255.0).sub_(0.5).div_(0.25)

class KneeNet(nn.Module):
    def __init__(self, backbone="efficientnet_b0.ra_in1k", pretrained=True, n_slots=5, n_slices=20,
                 d=256, n_layers=2, n_labels=12, drop=0.2, grad_ckpt=True):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3)
        if grad_ckpt and hasattr(self.enc, "set_grad_checkpointing"):
            self.enc.set_grad_checkpointing(True)
        nf = self.enc.num_features
        self.proj = nn.Sequential(nn.Linear(nf, d), nn.LayerNorm(d))
        self.slot_emb = nn.Parameter(torch.zeros(n_slots, 1, d))
        self.pos_emb = nn.Parameter(torch.zeros(1, n_slices, d))
        nn.init.trunc_normal_(self.slot_emb, std=0.02); nn.init.trunc_normal_(self.pos_emb, std=0.02)
        layer = nn.TransformerEncoderLayer(d, nhead=8, dim_feedforward=2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.mixer = nn.TransformerEncoder(layer, n_layers)
        self.queries = nn.Parameter(torch.randn(n_labels, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 8, dropout=0.1, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.head_w = nn.Parameter(torch.zeros(n_labels, d))
        self.head_b = nn.Parameter(torch.zeros(n_labels))

    def forward(self, x, mask, slice_idx):
        """x: [B, S, N, H, W] uint8 on device; mask: [B, S] (1 = slot present); slice_idx: slices to use."""
        B, S = x.shape[:2]
        n = len(slice_idx)
        imgs = to_25d(x, slice_idx)                                  # [B, S, n, 3, H, W]
        feats = self.enc(imgs.flatten(0, 2))                         # [B*S*n, nf]
        t = self.proj(feats).view(B, S, n, -1)
        pos = self.pos_emb[:, torch.as_tensor(slice_idx, device=x.device)]  # [1, n, d]
        t = t + pos.unsqueeze(0) + self.slot_emb.unsqueeze(0)       # [B, S, n, d]
        t = t.flatten(1, 2)                                          # [B, S*n, d]
        pad = (mask == 0).repeat_interleave(n, dim=1)                # [B, S*n] True = ignore
        pad = pad & ~pad.all(1, keepdim=True)                        # a study with no slots: attend to zeros
        t = self.mixer(t, src_key_padding_mask=pad)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        o, _ = self.attn(q, t, t, key_padding_mask=pad)              # [B, 12, d]
        o = self.drop(o)
        return (o * self.head_w).sum(-1) + self.head_b               # [B, 12] logits

SAG_SLOTS, INPLANE_X_SLOTS = [0, 1], [2, 3, 4]   # sagittal slices run along x; coronal/axial columns run along x

def mirror(x):
    """Mirror whole studies left<->right (patient x axis). x: [B, S, N, H, W].
    Anatomy is preserved (a mirrored right knee is a valid left knee), so labels do not change."""
    x = x.clone()
    x[:, SAG_SLOTS] = x[:, SAG_SLOTS].flip(2)          # reverse sagittal slice order
    x[:, INPLANE_X_SLOTS] = x[:, INPLANE_X_SLOTS].flip(4)  # flip image columns
    return x

def predict_tta(model, x, mask, slice_idx, tta=True):
    p = torch.sigmoid(model(x, mask, slice_idx).float())
    if tta:
        p = (p + torch.sigmoid(model(mirror(x), mask, slice_idx).float())) / 2
    return p

def slice_indices(n_total, n_use):
    return np.round(np.linspace(0, n_total - 1, n_use)).astype(int).tolist()
