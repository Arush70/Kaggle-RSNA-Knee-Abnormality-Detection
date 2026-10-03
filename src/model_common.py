# ---- shared model code: identical in training and submission notebooks ----
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch.utils.checkpoint

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
                 d=256, n_layers=2, n_labels=12, drop=0.2, grad_ckpt=True, freeze_blocks=0, img_size=None):
        super().__init__()
        kw = {"img_size": img_size} if img_size else {}
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0, in_chans=3, **kw)
        self.is_vit = hasattr(self.enc, "blocks") and hasattr(self.enc, "patch_embed")
        self.img_size = img_size
        self.freeze_blocks = freeze_blocks if self.is_vit else 0
        self.grad_ckpt = grad_ckpt
        if self.is_vit:
            # DINOv2-style: frozen patch embedding + first k blocks, trainable rest; feature = [CLS, mean(patches)]
            for prm in list(self.enc.patch_embed.parameters()) + [self.enc.pos_embed] + \
                       ([self.enc.cls_token] if getattr(self.enc, "cls_token", None) is not None else []):
                prm.requires_grad_(self.freeze_blocks == 0)
            for blk in self.enc.blocks[:self.freeze_blocks]:
                for prm in blk.parameters():
                    prm.requires_grad_(False)
            nf = 2 * self.enc.num_features
        else:
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

    def encode(self, imgs):
        if not self.is_vit:
            return self.enc(imgs)
        enc, k = self.enc, self.freeze_blocks
        if self.img_size and imgs.shape[-1] != self.img_size:   # centre-crop (e.g. 256 -> 252, a multiple of 14)
            o = (imgs.shape[-1] - self.img_size) // 2
            imgs = imgs[..., o:o + self.img_size, o:o + self.img_size]
        with torch.set_grad_enabled(torch.is_grad_enabled() and k == 0):
            t = enc.patch_embed(imgs)
            t = enc._pos_embed(t)
            t = enc.norm_pre(t)
            for blk in enc.blocks[:k]:
                t = blk(t)
        for blk in enc.blocks[k:]:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                t = torch.utils.checkpoint.checkpoint(blk, t, use_reentrant=False)
            else:
                t = blk(t)
        t = enc.norm(t)
        npt = getattr(enc, "num_prefix_tokens", 1)
        return torch.cat([t[:, 0], t[:, npt:].mean(1)], 1)

    def forward(self, x, mask, slice_idx):
        """x: [B, S, N, H, W] uint8 on device; mask: [B, S] (1 = slot present); slice_idx: slices to use."""
        B, S = x.shape[:2]
        n = len(slice_idx)
        imgs = to_25d(x, slice_idx)                                  # [B, S, n, 3, H, W]
        feats = self.encode(imgs.flatten(0, 2))                      # [B*S*n, nf]
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
