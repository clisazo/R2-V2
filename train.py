#!/usr/bin/env python3
"""Training script for R2-V2 (RRWNet-based) artery/vein segmentation.

Dataset layout
--------------
Each dataset root must follow the convention (matching the unified output of
standardize_datasets.py after renaming images→cfp, masks→mask, av→label):

    <dataset_root>/
        cfp/    ← original colour fundus images
        label/  ← 3-channel GT  (R=A, G=BV, B=V  in GAVE format
                                   or R=A, G=V,  B=BV in native format)
        mask/   ← ROI masks  (optional; auto-detected if absent)
        pre/    ← preprocessed images  (optional; computed on-the-fly if absent)

Multiple dataset roots can be passed with repeated --data_dir flags. Each root
is inspected at startup: if its label/ directory contains images where both the
artery channel (R) and the vein channel (B in GAVE fmt, G in native fmt) are
all-zero, that dataset is tagged as **BV-only** and its samples only receive
supervision on the blood-vessel head (L_bv + L_bg), skipping L_av / L_mx /
L_cr.  This lets HRF and RITE contribute to vessel localisation without
injecting false negatives into the A/V heads.

Two model types:
    av  -- artery/vein model, 3-channel input  (preprocessed only)
    bv  -- blood vessel model, 6-channel input (preprocessed + original CFP)
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import io
from skimage.transform import resize
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

from model import RRWNet
import preprocessing

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

def dice_score(logits: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> dict:
    """Compute per-channel Dice scores (A, V, BV) inside the ROI mask.
    
    Returns a dict with keys 'a', 'v', 'bv', 'mean'.
    """
    probs = torch.sigmoid(logits)          # [B, 3, H, W]
    preds = (probs > 0.5).float()
    m     = mask.expand_as(preds)          # broadcast [B,1,H,W] → [B,3,H,W]

    names = ['a', 'v', 'bv']
    scores = {}
    for i, name in enumerate(names):
        p = preds[:, i] * m[:, i]
        g = gt[:, i]    * m[:, i]
        inter = (p * g).sum()
        denom = p.sum() + g.sum()
        scores[name] = (2 * inter / (denom + eps)).item()
    scores['mean'] = sum(scores[n] for n in names) / 3
    return scores

def set_seed(seed: int) -> None:
    """Set seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Initialize random seed for DataLoader workers.
    
    Args:
        worker_id: Worker ID from DataLoader
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def read_img(path: Path) -> np.ndarray:
    """Return H×W×3 float32 in [0, 1]."""
    img = io.imread(str(path)).astype(np.float32)
    if img.max() > 1.0:
        img /= 65535.0 if img.max() > 255.0 else 255.0
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] > 3:
        img = img[..., :3]
    return img


def find_file(stem: str, directory: Path) -> Optional[Path]:
    for fn in directory.iterdir():
        if fn.is_file() and fn.suffix.lower() in IMG_EXTS and fn.stem == stem:
            return fn
    return None


# ─────────────────────────────────────────────────────────────────────────────
# BV-only auto-detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_bv_only(label_dir: Path, gave_label_fmt: bool, n_probe: int = 5) -> bool:
    """Return True if this dataset has no artery/vein channel information.

    Probes up to `n_probe` label images and checks whether the artery channel
    (R in GAVE format, R in native format) and the vein channel (B in GAVE
    format, G in native format) are both entirely zero.  If every probed image
    satisfies this condition the dataset is declared BV-only.

    Parameters
    ----------
    label_dir:       Path to the label/ directory.
    gave_label_fmt:  True → GAVE format (R=A, G=BV, B=V).
    n_probe:         Number of files to sample for the check.
    """
    files = sorted(
        f for f in label_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )
    if not files:
        return False

    probe = files[:n_probe]
    for f in probe:
        lab = read_img(f)   # H×W×3, float32 [0,1]
        if gave_label_fmt:
            artery_ch = lab[..., 0]   # R = artery
            vein_ch   = lab[..., 2]   # B = vein
        else:
            artery_ch = lab[..., 0]   # R = artery
            vein_ch   = lab[..., 1]   # G = vein
        # If either A or V channel has any signal, this is a full AV dataset
        if artery_ch.max() > 0.1 or vein_ch.max() > 0.1:
            return False

    return True   # every probed image had silent A/V channels → BV-only


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v    = maxc
    s    = np.where(maxc > 0, (maxc - minc) / np.where(maxc > 0, maxc, 1), 0.0)
    diff = (maxc - minc) + 1e-8
    h    = np.zeros_like(v)
    nz   = maxc > minc
    mr   = nz & (maxc == r)
    mg   = nz & (maxc == g)
    mb   = nz & (maxc == b) & ~mr & ~mg
    h[mr] = ((g[mr] - b[mr]) / diff[mr]) % 6.0
    h[mg] = (b[mg] - r[mg]) / diff[mg] + 2.0
    h[mb] = (r[mb] - g[mb]) / diff[mb] + 4.0
    h /= 6.0
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(img: np.ndarray) -> np.ndarray:
    h, s, v = img[..., 0], img[..., 1], img[..., 2]
    h6 = h * 6.0
    i  = np.floor(h6).astype(int) % 6
    f  = h6 - np.floor(h6)
    p  = v * (1 - s)
    q  = v * (1 - f * s)
    t  = v * (1 - (1 - f) * s)
    out = np.zeros_like(img)
    for idx, (rv, gv, bv) in enumerate(
        [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    ):
        m = i == idx
        out[m, 0] = rv[m]; out[m, 1] = gv[m]; out[m, 2] = bv[m]
    return out


def _hsv_jitter(t: torch.Tensor) -> torch.Tensor:
    dh  = random.uniform(-0.02, 0.02) * random.uniform(0.8, 1.2)
    ds  = random.uniform(-0.2, 0.2)
    dv  = random.uniform(-0.2, 0.2) * random.uniform(0.8, 1.2)
    img = t.numpy().transpose(1, 2, 0)
    hsv = _rgb_to_hsv(img)
    hsv[..., 0] = (hsv[..., 0] + dh) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] + ds, 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] + dv, 0.0, 1.0)
    return torch.from_numpy(_hsv_to_rgb(hsv).astype(np.float32).transpose(2, 0, 1))


def _cutout(t: torch.Tensor) -> torch.Tensor:
    _, h, w = t.shape
    ph = max(1, int(0.04 * h))
    pw = max(1, int(0.04 * w))
    for _ in range(16):
        j = random.randint(0, max(0, h - ph))
        i = random.randint(0, max(0, w - pw))
        t[:, j:j + ph, i:i + pw] = random.uniform(0.4, 0.6)
    return t


def apply_augmentation(
    pre_t:  torch.Tensor,
    cfp_t:  torch.Tensor,
    gt_t:   torch.Tensor,
    mask_t: torch.Tensor,
) -> tuple:
    if random.random() < 0.5:
        pre_t  = TF.hflip(pre_t);  cfp_t  = TF.hflip(cfp_t)
        gt_t   = TF.hflip(gt_t);   mask_t = TF.hflip(mask_t)
    if random.random() < 0.5:
        pre_t  = TF.vflip(pre_t);  cfp_t  = TF.vflip(cfp_t)
        gt_t   = TF.vflip(gt_t);   mask_t = TF.vflip(mask_t)
    angle = random.uniform(-180.0, 180.0)
    scale = random.uniform(0.7, 1.4)
    shear = [random.uniform(-25.0, 25.0)]
    pre_t  = TF.affine(pre_t,  angle, [0, 0], scale, shear,
                        interpolation=TF.InterpolationMode.BILINEAR)
    cfp_t  = TF.affine(cfp_t,  angle, [0, 0], scale, shear,
                        interpolation=TF.InterpolationMode.BILINEAR)
    gt_t   = TF.affine(gt_t,   angle, [0, 0], scale, shear,
                        interpolation=TF.InterpolationMode.NEAREST)
    mask_t = TF.affine(mask_t, angle, [0, 0], scale, shear,
                        interpolation=TF.InterpolationMode.NEAREST)
    pre_t = _hsv_jitter(pre_t)
    cfp_t = _hsv_jitter(cfp_t)
    if random.random() < 0.8:
        pre_t = _cutout(pre_t)
    return pre_t, cfp_t, gt_t, mask_t


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RetinalDataset(Dataset):
    """Loads retinal fundus images with artery/vein ground-truth labels.

    Args:
        cfp_dir:         Directory with original colour fundus images.
        label_dir:       Directory with 3-channel GT label images.
        mask_dir:        Directory with ROI masks (optional).
        pre_dir:         Directory with preprocessed images (optional;
                         computed on-the-fly via CGCELIN+CLAHE if absent).
        use_cfp:         Concatenate original CFP to preprocessed input (BV model).
        gave_label_fmt:  True  → GAVE format   (R=A, G=BV, B=V).
                         False → native format  (R=A, G=V,  B=BV).
        bv_only:         If True, this dataset has no artery/vein channel
                         information; the returned gt tensor will have A and V
                         channels zeroed out, signalling the loss to skip all
                         A/V-specific terms.  Pass None to auto-detect.
        do_augment:      Apply data augmentation.
        resize_width:    Target width after ROI cropping.
        dataset_name:    Human-readable name used in log messages.
    """

    def __init__(
        self,
        cfp_dir:        Path,
        label_dir:      Path,
        mask_dir:       Optional[Path]  = None,
        pre_dir:        Optional[Path]  = None,
        use_cfp:        bool            = False,
        gave_label_fmt: bool            = True,
        bv_only:        Optional[bool]  = None,
        do_augment:     bool            = True,
        resize_width:   int             = 1408,
        dataset_name:   str             = '',
    ):
        self.cfp_dir        = cfp_dir
        self.label_dir      = label_dir
        self.mask_dir       = mask_dir
        self.pre_dir        = pre_dir
        self.use_cfp        = use_cfp
        self.gave_label_fmt = gave_label_fmt
        self.do_augment     = do_augment
        self.resize_width   = resize_width
        self.dataset_name   = dataset_name

        # Auto-detect BV-only mode if not explicitly set
        if bv_only is None:
            self.bv_only = detect_bv_only(label_dir, gave_label_fmt)
        else:
            self.bv_only = bv_only

        self.files = sorted(
            f for f in cfp_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTS
        )

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, cfp_fn: Path):
        stem = cfp_fn.stem
        cfp  = read_img(cfp_fn)

        # ── ROI mask ────────────────────────────────────────────────────────
        mask = None
        if self.mask_dir:
            mfn = find_file(stem, self.mask_dir)
            if mfn:
                m    = read_img(mfn)
                mask = m[..., 0] if m.ndim == 3 else m
        if mask is None:
            mask = (cfp.sum(axis=2) > 0.01).astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)

        # ── Preprocessed image (CGCELIN + CLAHE) ────────────────────────────
        pfn = find_file(stem, self.pre_dir) if self.pre_dir else None
        if pfn:
            pre = read_img(pfn)
        else:
            pre_u8, mask_u8  = (cfp * 255).astype(np.uint8), (mask * 255).astype(np.uint8)
            pre, mask_proc   = preprocessing.preprocess_img(pre_u8, mask_u8)
            pre  = pre.astype(np.float32)
            mask = (mask_proc > 127).astype(np.float32)

        # ── GT label ────────────────────────────────────────────────────────
        lfn = find_file(stem, self.label_dir)
        assert lfn is not None, f"No label found for '{stem}' in {self.label_dir}"
        lab = read_img(lfn)

        if self.gave_label_fmt:
            gt_a, gt_bv, gt_v = lab[..., 0], lab[..., 1], lab[..., 2]
        else:
            gt_a, gt_v, gt_bv = lab[..., 0], lab[..., 1], lab[..., 2]

        gt_a  = (gt_a  > 0.5).astype(np.float32)
        gt_v  = (gt_v  > 0.5).astype(np.float32)
        gt_bv = (gt_bv > 0.5).astype(np.float32)

        if gt_bv.max() < 0.5:
            gt_bv = np.clip(gt_a + gt_v, 0.0, 1.0)

        # For BV-only datasets zero out A and V to signal the loss function.
        # The BV channel is kept so that L_bv still fires.
        if self.bv_only:
            gt_a = np.zeros_like(gt_bv)
            gt_v = np.zeros_like(gt_bv)

        gt = np.stack([gt_a, gt_v, gt_bv], axis=-1)   # H×W×3  [A, V, BV]

        # ── Crop to ROI bounding box ─────────────────────────────────────────
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)
        if rows.any() and cols.any():
            r0, r1 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
            c0, c1 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
            pre  = pre[r0:r1+1, c0:c1+1]
            cfp  = cfp[r0:r1+1, c0:c1+1]
            gt   = gt[r0:r1+1,  c0:c1+1]
            mask = mask[r0:r1+1, c0:c1+1]

        # ── Resize to fixed width, maintaining aspect ratio ──────────────────
        ch, cw = pre.shape[:2]
        if cw != self.resize_width:
            new_h    = int(round(ch * self.resize_width / cw))
            new_size = (new_h, self.resize_width)
            pre  = resize(pre,  new_size, anti_aliasing=True,  preserve_range=True, order=1).astype(np.float32)
            cfp  = resize(cfp,  new_size, anti_aliasing=True,  preserve_range=True, order=1).astype(np.float32)
            gt   = resize(gt,   new_size, anti_aliasing=False, preserve_range=True, order=0).astype(np.float32)
            mask = resize(mask, new_size, anti_aliasing=False, preserve_range=True, order=0).astype(np.float32)

        return pre, cfp, mask, gt

    def __getitem__(self, idx: int):
        pre, cfp, mask, gt = self._load(self.files[idx])

        pre_t  = torch.from_numpy(pre.transpose(2, 0, 1))    # [3, H, W]
        cfp_t  = torch.from_numpy(cfp.transpose(2, 0, 1))    # [3, H, W]
        gt_t   = torch.from_numpy(gt.transpose(2, 0, 1))     # [3, H, W]
        mask_t = torch.from_numpy(mask).unsqueeze(0)          # [1, H, W]

        if self.do_augment:
            pre_t, cfp_t, gt_t, mask_t = apply_augmentation(pre_t, cfp_t, gt_t, mask_t)

        gt_t   = (gt_t   > 0.5).float()
        mask_t = (mask_t > 0.5).float()

        # bv_only flag is passed as a scalar tensor so ConcatDataset / the
        # loss function can distinguish samples without inspecting the dataset.
        bv_only_t = torch.tensor(float(self.bv_only))

        inp = torch.cat([pre_t, cfp_t], dim=0) if self.use_cfp else pre_t
        return inp, gt_t, mask_t, bv_only_t


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class RRLoss(nn.Module):
    """Recursive Refinement loss from the R2-V2 technical report (Eq. 2–11).

    Total loss: L = w_0·L_S(ŷ_0) + Σ_{k=1}^{K} (k/Z)·L_S(ŷ_k)
    where Z = K(K+1)/2 and w_0 = 1.

    Base segmentation loss:
        L_S = λ_bv·L_bv + λ_av·L_av + λ_mx·L_mx + λ_cr·L_cr + λ_bg·L_bg

    For BV-only samples (bv_only==True) A and V channels are zeroed in gt, so
    we suppress all A/V-specific terms and only compute L_bv + L_bg.  This is
    detected per-sample from the bv_only flag passed through the DataLoader,
    so a mixed batch from multiple datasets is handled correctly.
    """

    def __init__(
        self,
        model_type: str   = 'av',
        lambda_bv:  float = 1.0,
        lambda_av:  float = 2.0,
        lambda_bg:  float = 0.5,
        lambda_mx:  float = 0.5,
        lambda_cr:  float = 0.5,
    ):
        super().__init__()
        self.model_type = model_type
        self.lambda_bv  = lambda_bv
        self.lambda_av  = lambda_av
        self.lambda_bg  = lambda_bg
        self.lambda_mx  = lambda_mx
        self.lambda_cr  = lambda_cr

    @staticmethod
    def _masked_bce(
        logits:  torch.Tensor,
        targets: torch.Tensor,
        mask:    torch.Tensor,
    ) -> torch.Tensor:
        n    = mask.sum().clamp(min=1)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (loss * mask).sum() / n

    def _base_loss(
        self,
        pred:    torch.Tensor,   # [B, 3, H, W] logits  (A, V, BV)
        gt:      torch.Tensor,   # [B, 3, H, W] binary  (A, V, BV)
        roi:     torch.Tensor,   # [B, 1, H, W] binary ROI mask
        bv_only: torch.Tensor,   # [B]          1.0 if BV-only sample
        lam_bv:  float,
        lam_av:  float,
        lam_bg:  float,
    ) -> torch.Tensor:
        pred_a  = pred[:, 0:1];  y_a  = gt[:, 0:1]
        pred_v  = pred[:, 1:2];  y_v  = gt[:, 1:2]
        pred_bv = pred[:, 2:3];  y_bv = gt[:, 2:3]

        bg_m = 1.0 - roi                                    # outside ROI
        bv_m = y_bv * roi                                   # vessel pixels inside ROI
        cr_m = y_a * y_v                                    # crossing pixels (A∩V)
        av_m = (y_a + y_v).clamp(0, 1) - cr_m              # non-crossing vessel pixels

        # ── L_bv: vessel presence loss (always computed) ────────────────────
        l_bv = self._masked_bce(pred_bv, y_bv, roi)

        # ── L_bg: background consistency (always computed) ──────────────────
        zeros = torch.zeros_like(pred_a)
        l_bg  = (self._masked_bce(pred_a, zeros, bg_m) +
                 self._masked_bce(pred_v, zeros, bg_m))

        # ── A/V-specific terms – zeroed for BV-only samples ─────────────────
        # bv_only is [B]; expand to [B,1,1,1] so it broadcasts over H×W.
        bv_flag = bv_only.view(-1, 1, 1, 1).to(pred.device)

        # L_av: A/V discrimination loss inside BV region
        l_av = (self._masked_bce(pred_a, y_a, bv_m) +
                self._masked_bce(pred_v, y_v, bv_m))
        l_av = l_av * (1.0 - bv_flag).squeeze()   # scalar; zero if bv_only

        # L_mx: mutual exclusion at non-crossing vessel pixels
        n_av  = av_m.sum().clamp(min=1)
        l_mx  = (torch.sigmoid(pred_a) * torch.sigmoid(pred_v) * av_m).sum() / n_av
        l_mx  = l_mx * (1.0 - bv_flag).squeeze()

        # L_cr: crossing handling
        n_cr  = cr_m.sum()
        if n_cr > 0:
            l_cr = self._masked_bce(
                (pred_a + pred_v) / 2, torch.ones_like(pred_a), cr_m
            )
            l_cr = l_cr * (1.0 - bv_flag).squeeze()
        else:
            l_cr = pred.new_zeros(1).squeeze()

        return (lam_bv * l_bv
                + lam_av  * l_av
                + self.lambda_mx * l_mx
                + self.lambda_cr * l_cr
                + lam_bg  * l_bg)

    def forward(
        self,
        predictions: list,          # K+1 tensors of [B, 3, H, W] logits
        gt:          torch.Tensor,  # [B, 3, H, W] binary GT [A, V, BV]
        mask:        torch.Tensor,  # [B, 1, H, W] binary ROI mask
        bv_only:     torch.Tensor,  # [B]          1.0 if BV-only sample
    ) -> torch.Tensor:
        K = len(predictions) - 1
        Z = K * (K + 1) / 2

        if self.model_type == 'bv':
            y_a, y_v, y_bv = gt[:, 0:1], gt[:, 1:2], gt[:, 2:3]
            cr      = y_a * y_v
            av      = (y_a + y_v).clamp(0, 1) - cr
            n_roi   = mask.sum().clamp(min=1)
            lam_bv  = ((y_bv * mask).sum() / n_roi).item()
            lam_av  = (2.0 * (av * mask).sum() / n_roi).item()
            lam_bg  = (0.5 * (1.0 - mask).sum() / n_roi).item()
        else:
            lam_bv, lam_av, lam_bg = self.lambda_bv, self.lambda_av, self.lambda_bg

        total = sum(
            (1.0 if k == 0 else k / Z)
            * self._base_loss(p, gt, mask, bv_only, lam_bv, lam_av, lam_bg)
            for k, p in enumerate(predictions)
        )
        return total


# ─────────────────────────────────────────────────────────────────────────────
# UNet padding helpers
# ─────────────────────────────────────────────────────────────────────────────

def pad_batch(t: torch.Tensor, multiple: int = 32) -> tuple:
    _, _, h, w = t.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2
    return F.pad(t, (pl, pr, pt, pb)), (pt, pb, pl, pr)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train R2-V2 (RRWNet) for retinal artery/vein segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data – accepts multiple roots
    p.add_argument('-d', '--data_dir', required=True, action='append',
                   dest='data_dirs', metavar='DATA_DIR',
                   help='Dataset root directory (cfp/ + label/ layout). '
                        'Repeat the flag to include multiple datasets, e.g. '
                        '-d /data/GAVE2 -d /data/LES-AV -d /data/HRF')
    p.add_argument('--val_data_dir', default=None,
                   help='Optional dedicated validation dataset root')
    p.add_argument('--val_split', type=float, default=0.0,
                   help='If > 0, randomly hold out this fraction of the '
                        'combined training set for validation')
    # Per-dataset overrides (applied to ALL datasets; useful when they share
    # the same sub-folder names for mask/ and pre/).
    p.add_argument('--mask_subdir', default='mask',
                   help='Name of the mask sub-directory inside each dataset root')
    p.add_argument('--pre_subdir',  default='pre',
                   help='Name of the preprocessed-image sub-directory')
    p.add_argument('--label_subdir', default='label',
                   help='Name of the label sub-directory')
    p.add_argument('--cfp_subdir',   default='cfp',
                   help='Name of the CFP sub-directory')
    # Model
    p.add_argument('-t', '--model_type', choices=['av', 'bv'], default='av')
    p.add_argument('--base_channels',  type=int, default=64)
    p.add_argument('--num_iterations', type=int, default=5)
    # Training
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--max_iterations', type=int,   default=200_000)
    p.add_argument('--resize_width',   type=int,   default=1408)
    p.add_argument('--seed',           type=int,   default=77)
    p.add_argument('--gpu',            type=int,   default=0)
    # Label format
    p.add_argument('--native_label_fmt', action='store_true', default=False,
                   help='GT labels are R=A, G=V, B=BV instead of GAVE R=A, G=BV, B=V')
    # Checkpointing / logging
    p.add_argument('-s', '--save_dir', default='./checkpoints')
    p.add_argument('--save_every',     type=int, default=5_000)
    p.add_argument('--log_every',      type=int, default=100)
    p.add_argument('--weights',        default=None,
                   help='Path to pre-trained weights to resume from')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    root:           Path,
    args:           argparse.Namespace,
    do_augment:     bool,
) -> RetinalDataset:
    """Construct a RetinalDataset from a single root directory."""
    cfp_dir   = root / args.cfp_subdir
    label_dir = root / args.label_subdir
    mask_dir  = root / args.mask_subdir
    pre_dir   = root / args.pre_subdir

    assert cfp_dir.exists(),   f"CFP dir not found: {cfp_dir}"
    assert label_dir.exists(), f"Label dir not found: {label_dir}"

    ds = RetinalDataset(
        cfp_dir        = cfp_dir,
        label_dir      = label_dir,
        mask_dir       = mask_dir   if mask_dir.exists() else None,
        pre_dir        = pre_dir    if pre_dir.exists()  else None,
        use_cfp        = (args.model_type == 'bv'),
        gave_label_fmt = not args.native_label_fmt,
        bv_only        = None,    # auto-detect
        do_augment     = do_augment,
        resize_width   = args.resize_width,
        dataset_name   = root.name,
    )
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = get_args()
    set_seed(args.seed)
    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {device}')
    print(f'Seed: {args.seed}')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Build one dataset per root, then concatenate ─────────────────────────
    all_datasets = []
    for root_str in args.data_dirs:
        root = Path(root_str)
        ds   = build_dataset(root, args, do_augment=True)
        mode = 'BV-only' if ds.bv_only else 'full AV'
        print(f'  {root.name:20s}  {len(ds):4d} samples  [{mode}]')
        all_datasets.append(ds)

    if not all_datasets:
        raise RuntimeError('No datasets found.')

    combined_ds = ConcatDataset(all_datasets) if len(all_datasets) > 1 else all_datasets[0]
    print(f'Total training samples: {len(combined_ds)}')

    # ── Optional train/val split ─────────────────────────────────────────────
    val_ds = None
    if args.val_split > 0:
        total   = len(combined_ds)
        n_val   = max(1, int(total * args.val_split))
        indices = list(range(total))
        random.shuffle(indices)
        train_idx = indices[n_val:]
        val_idx   = indices[:n_val]
        from torch.utils.data import Subset
        train_ds = Subset(combined_ds, train_idx)
        val_ds   = Subset(combined_ds, val_idx)
        print(f'Train/val split: {len(train_ds)} / {len(val_ds)}')
    elif args.val_data_dir:
        train_ds = combined_ds
        val_ds   = build_dataset(Path(args.val_data_dir), args, do_augment=False)
        print(f'Validation samples: {len(val_ds)}')
    else:
        train_ds = combined_ds

    # ── Model ────────────────────────────────────────────────────────────────
    in_ch = 6 if args.model_type == 'bv' else 3
    model = RRWNet(in_ch, 3, args.base_channels, args.num_iterations)
    if args.weights:
        ckpt = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(ckpt)
        print(f'Loaded weights from {args.weights}')
    model.to(device)

    # ── Loss ─────────────────────────────────────────────────────────────────
    if args.model_type == 'av':
        criterion = RRLoss('av', lambda_bv=1.0, lambda_av=2.0,
                           lambda_bg=0.5, lambda_mx=0.5, lambda_cr=0.5)
    else:
        criterion = RRLoss('bv', lambda_mx=0.5, lambda_cr=0.2)

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # ── Save config ──────────────────────────────────────────────────────────
    config = {
        'model':          'RRWNet',
        'model_type':     args.model_type,
        'in_channels':    in_ch,
        'out_channels':   3,
        'base_channels':  args.base_channels,
        'num_iterations': args.num_iterations,
        'learning_rate':  args.lr,
        'seed':           args.seed,
        'use_cfp':        (args.model_type == 'bv'),
        'gave_label_fmt': not args.native_label_fmt,
        'datasets':       args.data_dirs,
    }
    config_fn = save_dir / f'{args.model_type}_config.json'
    with open(config_fn, 'w') as f:
        json.dump(config, f, indent=4)
    print(f'Config saved to {config_fn}')

    # ── Training loop ─────────────────────────────────────────────────────────
    iteration    = 0
    epoch        = 0
    running_loss = 0.0
    t0           = time.time()
    model.train()

    while iteration < args.max_iterations:
        epoch += 1
        loader = DataLoader(
            train_ds, batch_size=1, shuffle=True,
            num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn,
        )

        for inp, gt, mask, bv_only in loader:
            if iteration >= args.max_iterations:
                break

            inp     = inp.to(device)
            gt      = gt.to(device)
            mask    = mask.to(device)
            bv_only = bv_only.to(device)   # [B]

            inp_p,  _ = pad_batch(inp)
            gt_p,   _ = pad_batch(gt)
            mask_p, _ = pad_batch(mask)

            optimizer.zero_grad()
            predictions = model(inp_p)
            loss        = criterion(predictions, gt_p, mask_p, bv_only)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            iteration    += 1

            if iteration % args.log_every == 0:
                avg  = running_loss / args.log_every
                secs = (time.time() - t0) / args.log_every
                print(
                    f'[{iteration:>7d}/{args.max_iterations}]  '
                    f'loss={avg:.4f}  epoch={epoch}  {secs:.2f}s/it'
                )
                running_loss = 0.0
                t0           = time.time()

            if iteration % args.save_every == 0:
                ckpt_fn = save_dir / f'{args.model_type}_{iteration:07d}.pth'
                torch.save(model.state_dict(), ckpt_fn)
                torch.save(model.state_dict(), save_dir / f'{args.model_type}_latest.pth')
                print(f'  → checkpoint saved: {ckpt_fn.name}')

                if val_ds is not None:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for v_inp, v_gt, v_mask, v_bv_only in DataLoader(val_ds, batch_size=1, worker_init_fn=worker_init_fn):
                            v_inp     = v_inp.to(device)
                            v_gt      = v_gt.to(device)
                            v_mask    = v_mask.to(device)
                            v_bv_only = v_bv_only.to(device)
                            v_inp_p,  _ = pad_batch(v_inp)
                            v_gt_p,   _ = pad_batch(v_gt)
                            v_mask_p, _ = pad_batch(v_mask)
                            v_preds     = model(v_inp_p)
                            val_loss   += criterion(v_preds, v_gt_p, v_mask_p, v_bv_only).item()
                    print(f'  → val loss={val_loss / len(val_ds):.4f}')
                    model.train()

    # ── Final save ────────────────────────────────────────────────────────────
    final_fn = save_dir / f'{args.model_type}.pth'
    torch.save(model.state_dict(), final_fn)
    print(f'\nTraining complete. Final model: {final_fn}')
    print(f'Use with infer.py: --weights {final_fn} --model_type {args.model_type}')


if __name__ == '__main__':
    main()