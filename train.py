#!/usr/bin/env python3
"""Training script for R2-V2 (RRWNet-based) artery/vein segmentation.

Expected dataset layout:
    data_dir/
        cfp/    <- original color fundus images
        label/  <- ground-truth labels  (3-channel: R=A, G=BV, B=V in GAVE format
                                          or R=A, G=V, B=BV in native format)
        mask/   <- ROI masks            (optional; auto-detected if absent)
        pre/    <- preprocessed images  (optional; computed on-the-fly if absent)

Two model types match the technical report:
    av  -- artery/vein model, 3-channel input (preprocessed only)
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
from torch.utils.data import Dataset, DataLoader

from model import RRWNet
import preprocessing


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
# Augmentation
# ─────────────────────────────────────────────────────────────────────────────

def _rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """H×W×3 RGB float32 → H×W×3 HSV float32, all in [0, 1]."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.where(maxc > 0, (maxc - minc) / np.where(maxc > 0, maxc, 1), 0.0)
    diff = (maxc - minc) + 1e-8
    h = np.zeros_like(v)
    nz = maxc > minc
    mr = nz & (maxc == r)
    mg = nz & (maxc == g)
    mb = nz & (maxc == b) & ~mr & ~mg
    h[mr] = ((g[mr] - b[mr]) / diff[mr]) % 6.0
    h[mg] = (b[mg] - r[mg]) / diff[mg] + 2.0
    h[mb] = (r[mb] - g[mb]) / diff[mb] + 4.0
    h /= 6.0
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(img: np.ndarray) -> np.ndarray:
    """H×W×3 HSV float32 → H×W×3 RGB float32, all in [0, 1]."""
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
    """Apply random HSV jitter to a [3, H, W] tensor in [0, 1]."""
    dh = random.uniform(-0.02, 0.02) * random.uniform(0.8, 1.2)
    ds = random.uniform(-0.2, 0.2)
    dv = random.uniform(-0.2, 0.2) * random.uniform(0.8, 1.2)
    img = t.numpy().transpose(1, 2, 0)
    hsv = _rgb_to_hsv(img)
    hsv[..., 0] = (hsv[..., 0] + dh) % 1.0
    hsv[..., 1] = np.clip(hsv[..., 1] + ds, 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] + dv, 0.0, 1.0)
    return torch.from_numpy(_hsv_to_rgb(hsv).astype(np.float32).transpose(2, 0, 1))


def _cutout(t: torch.Tensor) -> torch.Tensor:
    """Apply 16 random filled patches (only to image, not label/mask)."""
    _, h, w = t.shape
    ph = max(1, int(0.04 * h))
    pw = max(1, int(0.04 * w))
    for _ in range(16):
        j = random.randint(0, max(0, h - ph))
        i = random.randint(0, max(0, w - pw))
        t[:, j:j + ph, i:i + pw] = random.uniform(0.4, 0.6)
    return t


def apply_augmentation(
    pre_t: torch.Tensor,
    cfp_t: torch.Tensor,
    gt_t: torch.Tensor,
    mask_t: torch.Tensor,
) -> tuple:
    """Apply all augmentations described in the technical report.

    Spatial transforms (flip, affine) are applied identically to all tensors.
    HSV jitter and cutout are applied only to image inputs (pre, cfp).
    """
    # Horizontal flip (p=0.5)
    if random.random() < 0.5:
        pre_t  = TF.hflip(pre_t)
        cfp_t  = TF.hflip(cfp_t)
        gt_t   = TF.hflip(gt_t)
        mask_t = TF.hflip(mask_t)

    # Vertical flip (p=0.5)
    if random.random() < 0.5:
        pre_t  = TF.vflip(pre_t)
        cfp_t  = TF.vflip(cfp_t)
        gt_t   = TF.vflip(gt_t)
        mask_t = TF.vflip(mask_t)

    # Affine: rotation U(-180,180) + scale U(0.7,1.4) + shear U(-25,25) (p=1)
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

    # HSV color jitter (p=1)
    pre_t = _hsv_jitter(pre_t)
    cfp_t = _hsv_jitter(cfp_t)

    # Cutout: 16 patches of 4%×4% of image size (p=0.8)
    if random.random() < 0.8:
        pre_t = _cutout(pre_t)

    return pre_t, cfp_t, gt_t, mask_t


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RetinalDataset(Dataset):
    """Loads retinal fundus images with artery/vein ground-truth labels.

    Args:
        cfp_dir:         Directory with original color fundus images.
        label_dir:       Directory with 3-channel GT label images.
        mask_dir:        Directory with ROI masks (optional).
        pre_dir:         Directory with preprocessed images (optional;
                         computed on-the-fly via CGCELIN+CLAHE if absent).
        use_cfp:         Concatenate original CFP to preprocessed input (BV model).
        gave_label_fmt:  True → GAVE format (R=A, G=BV, B=V);
                         False → native (R=A, G=V, B=BV).
        do_augment:      Apply data augmentation.
        resize_width:    Target width after ROI cropping.
    """

    def __init__(
        self,
        cfp_dir: Path,
        label_dir: Path,
        mask_dir: Optional[Path] = None,
        pre_dir: Optional[Path] = None,
        use_cfp: bool = False,
        gave_label_fmt: bool = True,
        do_augment: bool = True,
        resize_width: int = 1408,
    ):
        self.cfp_dir       = cfp_dir
        self.label_dir     = label_dir
        self.mask_dir      = mask_dir
        self.pre_dir       = pre_dir
        self.use_cfp       = use_cfp
        self.gave_label_fmt = gave_label_fmt
        self.do_augment    = do_augment
        self.resize_width  = resize_width
        self.files = sorted(
            f for f in cfp_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMG_EXTS
        )

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, cfp_fn: Path):
        stem = cfp_fn.stem
        cfp  = read_img(cfp_fn)

        # ROI mask
        mask = None
        if self.mask_dir:
            mfn = find_file(stem, self.mask_dir)
            if mfn:
                m    = read_img(mfn)
                mask = m[..., 0] if m.ndim == 3 else m
        if mask is None:
            mask = (cfp.sum(axis=2) > 0.01).astype(np.float32)
        mask = (mask > 0.5).astype(np.float32)

        # Preprocessed image (CGCELIN + CLAHE)
        pfn = find_file(stem, self.pre_dir) if self.pre_dir else None
        if pfn:
            pre = read_img(pfn)
        else:
            pre_u8, mask_u8 = (cfp * 255).astype(np.uint8), (mask * 255).astype(np.uint8)
            pre, mask_proc  = preprocessing.preprocess_img(pre_u8, mask_u8)
            pre  = pre.astype(np.float32)
            mask = (mask_proc > 127).astype(np.float32)

        # GT label → [A, V, BV] binary
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
        if gt_bv.max() < 0.5:   # derive BV from A∪V if not provided
            gt_bv = np.clip(gt_a + gt_v, 0.0, 1.0)
        gt = np.stack([gt_a, gt_v, gt_bv], axis=-1)   # H×W×3  [A, V, BV]

        # Crop to ROI bounding box
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)
        if rows.any() and cols.any():
            r0, r1 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
            c0, c1 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
            pre  = pre[r0:r1+1, c0:c1+1]
            cfp  = cfp[r0:r1+1, c0:c1+1]
            gt   = gt[r0:r1+1,  c0:c1+1]
            mask = mask[r0:r1+1, c0:c1+1]

        # Resize to fixed width, maintaining aspect ratio
        ch, cw = pre.shape[:2]
        if cw != self.resize_width:
            new_h = int(round(ch * self.resize_width / cw))
            new_size = (new_h, self.resize_width)
            pre  = resize(pre,  new_size, anti_aliasing=True,  preserve_range=True, order=1).astype(np.float32)
            cfp  = resize(cfp,  new_size, anti_aliasing=True,  preserve_range=True, order=1).astype(np.float32)
            gt   = resize(gt,   new_size, anti_aliasing=False, preserve_range=True, order=0).astype(np.float32)
            mask = resize(mask, new_size, anti_aliasing=False, preserve_range=True, order=0).astype(np.float32)

        return pre, cfp, mask, gt

    def __getitem__(self, idx: int):
        pre, cfp, mask, gt = self._load(self.files[idx])

        pre_t  = torch.from_numpy(pre.transpose(2, 0, 1))   # [3, H, W]
        cfp_t  = torch.from_numpy(cfp.transpose(2, 0, 1))   # [3, H, W]
        gt_t   = torch.from_numpy(gt.transpose(2, 0, 1))    # [3, H, W]
        mask_t = torch.from_numpy(mask).unsqueeze(0)         # [1, H, W]

        if self.do_augment:
            pre_t, cfp_t, gt_t, mask_t = apply_augmentation(pre_t, cfp_t, gt_t, mask_t)

        gt_t   = (gt_t   > 0.5).float()
        mask_t = (mask_t > 0.5).float()

        inp = torch.cat([pre_t, cfp_t], dim=0) if self.use_cfp else pre_t
        return inp, gt_t, mask_t


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

class RRLoss(nn.Module):
    """Recursive Refinement loss from the R2-V2 technical report (Eq. 2–11).

    Total loss: L = w_0·L_S(ŷ_0) + Σ_{k=1}^{K} (k/Z)·L_S(ŷ_k)
    where Z = K(K+1)/2 and w_0 = 1.

    Base segmentation loss L_S = λ_bv·L_bv + λ_av·L_av + λ_mx·L_mx
                                + λ_cr·L_cr + λ_bg·L_bg

    For the AV model, λ values are fixed (Eq. 11 of the report).
    For the BV model, λ_bv, λ_av, λ_bg are computed per-image from pixel
    counts (Eq. 10), while λ_mx and λ_cr are fixed.
    """

    def __init__(
        self,
        model_type: str = 'av',
        lambda_bv: float = 1.0,
        lambda_av: float = 2.0,
        lambda_bg: float = 0.5,
        lambda_mx: float = 0.5,
        lambda_cr: float = 0.5,
    ):
        super().__init__()
        self.model_type = model_type
        self.lambda_bv  = lambda_bv
        self.lambda_av  = lambda_av
        self.lambda_bg  = lambda_bg
        self.lambda_mx  = lambda_mx
        self.lambda_cr  = lambda_cr

    @staticmethod
    def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n    = mask.sum().clamp(min=1)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (loss * mask).sum() / n

    def _base_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        roi: torch.Tensor,
        lam_bv: float,
        lam_av: float,
        lam_bg: float,
    ) -> torch.Tensor:
        """
        pred: [B, 3, H, W] logits  (channels: A, V, BV)
        gt:   [B, 3, H, W] binary  (channels: A, V, BV)
        roi:  [B, 1, H, W] binary ROI mask
        """
        pred_a  = pred[:, 0:1]
        pred_v  = pred[:, 1:2]
        pred_bv = pred[:, 2:3]
        y_a     = gt[:, 0:1]
        y_v     = gt[:, 1:2]
        y_bv    = gt[:, 2:3]

        bg_m  = 1.0 - roi                         # outside ROI
        bv_m  = y_bv * roi                        # vessel pixels inside ROI
        cr_m  = y_a * y_v                         # crossing pixels (A∩V)
        av_m  = (y_a + y_v).clamp(0, 1) - cr_m   # non-crossing vessel pixels

        # L_bv: vessel presence loss inside ROI (Eq. 5)
        l_bv = self._masked_bce(pred_bv, y_bv, roi)

        # L_av: A/V discrimination loss inside BV region (Eq. 6)
        l_av = (self._masked_bce(pred_a, y_a, bv_m) +
                self._masked_bce(pred_v, y_v, bv_m))

        # L_mx: mutual exclusion loss at non-crossing vessel pixels (Eq. 7)
        n_av = av_m.sum().clamp(min=1)
        l_mx = (torch.sigmoid(pred_a) * torch.sigmoid(pred_v) * av_m).sum() / n_av

        # L_cr: crossing handling loss (Eq. 8)
        n_cr = cr_m.sum()
        if n_cr > 0:
            l_cr = self._masked_bce((pred_a + pred_v) / 2, torch.ones_like(pred_a), cr_m)
        else:
            l_cr = pred.new_zeros(1).squeeze()

        # L_bg: background consistency loss (Eq. 9)
        zeros = torch.zeros_like(pred_a)
        l_bg = (self._masked_bce(pred_a, zeros, bg_m) +
                self._masked_bce(pred_v, zeros, bg_m))

        return lam_bv * l_bv + lam_av * l_av + self.lambda_mx * l_mx + self.lambda_cr * l_cr + lam_bg * l_bg

    def forward(
        self,
        predictions: list,
        gt: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        predictions: list of K+1 tensors, each [B, 3, H, W] logits
        gt:          [B, 3, H, W] binary ground truth [A, V, BV]
        mask:        [B, 1, H, W] binary ROI mask
        """
        K = len(predictions) - 1
        Z = K * (K + 1) / 2

        # Compute λ weights
        if self.model_type == 'bv':
            # Per-image pixel-count-based weights (Eq. 10)
            y_a, y_v, y_bv = gt[:, 0:1], gt[:, 1:2], gt[:, 2:3]
            cr       = y_a * y_v
            av       = (y_a + y_v).clamp(0, 1) - cr
            n_roi    = mask.sum().clamp(min=1)
            lam_bv   = ((y_bv * mask).sum() / n_roi).item()
            lam_av   = (2.0 * (av * mask).sum() / n_roi).item()
            lam_bg   = (0.5 * (1.0 - mask).sum() / n_roi).item()
        else:
            # Fixed weights (Eq. 11)
            lam_bv, lam_av, lam_bg = self.lambda_bv, self.lambda_av, self.lambda_bg

        total = sum(
            (1.0 if k == 0 else k / Z) * self._base_loss(p, gt, mask, lam_bv, lam_av, lam_bg)
            for k, p in enumerate(predictions)
        )
        return total


# ─────────────────────────────────────────────────────────────────────────────
# UNet padding helpers (matching inference.py / transformations.py)
# ─────────────────────────────────────────────────────────────────────────────

def pad_batch(t: torch.Tensor, multiple: int = 32) -> tuple:
    """Pad [1, C, H, W] to be divisible by `multiple`. Returns (padded, padding)."""
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
    # Data
    p.add_argument('-d', '--data_dir', required=True,
                   help='Root data directory (must contain a cfp/ and label/ subdirectory)')
    p.add_argument('--label_dir', default=None,
                   help='Override path to label directory')
    p.add_argument('--mask_dir',  default=None,
                   help='Override path to mask directory')
    p.add_argument('--pre_dir',   default=None,
                   help='Override path to preprocessed image directory')
    p.add_argument('--val_data_dir', default=None,
                   help='Optional validation data directory (same layout as data_dir)')
    p.add_argument('--val_split', type=float, default=0.0,
                   help='If > 0, randomly hold out this fraction for validation')
    # Model
    p.add_argument('-t', '--model_type', choices=['av', 'bv'], default='av',
                   help='Model type: av (artery/vein) or bv (blood vessel)')
    p.add_argument('--base_channels',  type=int,   default=64)
    p.add_argument('--num_iterations', type=int,   default=5,
                   help='RR module iterations (K = num_iterations + 1)')
    # Training
    p.add_argument('--lr',             type=float, default=1e-4)
    p.add_argument('--max_iterations', type=int,   default=200_000)
    p.add_argument('--resize_width',   type=int,   default=1408)
    p.add_argument('--seed',           type=int,   default=77)
    p.add_argument('--gpu',            type=int,   default=0)
    # Label format
    p.add_argument('--native_label_fmt', action='store_true', default=False,
                   help='GT labels are in native format (R=A, G=V, B=BV) '
                        'instead of GAVE format (R=A, G=BV, B=V)')
    # Checkpointing / logging
    p.add_argument('-s', '--save_dir', default='./checkpoints')
    p.add_argument('--save_every',     type=int,   default=5_000,
                   help='Save checkpoint every N iterations')
    p.add_argument('--log_every',      type=int,   default=100,
                   help='Print loss every N iterations')
    p.add_argument('--weights',        default=None,
                   help='Path to pre-trained weights to resume from')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def resolve_dir(override: Optional[str], default: Path) -> Optional[Path]:
    if override:
        return Path(override)
    return default if default.exists() else None


def main() -> None:
    args = get_args()
    set_seed(args.seed)

    device = torch.device(
        f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device: {device}')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Data directories ────────────────────────────────────────────────────
    data_dir  = Path(args.data_dir)
    cfp_dir   = data_dir / 'cfp'
    label_dir = resolve_dir(args.label_dir, data_dir / 'label')
    mask_dir  = resolve_dir(args.mask_dir,  data_dir / 'mask')
    pre_dir   = resolve_dir(args.pre_dir,   data_dir / 'pre')

    assert cfp_dir.exists(),   f"CFP directory not found: {cfp_dir}"
    assert label_dir is not None and label_dir.exists(), \
        f"Label directory not found. Expected {data_dir / 'label'} or use --label_dir."

    use_cfp = (args.model_type == 'bv')
    gave_fmt = not args.native_label_fmt

    # Collect file list for optional train/val split
    all_files = sorted(
        f for f in cfp_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMG_EXTS
    )
    if args.val_split > 0:
        rng = random.Random(args.seed)
        rng.shuffle(all_files)
        n_val   = max(1, int(len(all_files) * args.val_split))
        val_files   = all_files[:n_val]
        train_files = all_files[n_val:]
        print(f'Train/val split: {len(train_files)} / {len(val_files)}')
    else:
        train_files = all_files
        val_files   = []

    # ── Datasets ────────────────────────────────────────────────────────────
    def make_dataset(files, augment):
        ds = RetinalDataset(
            cfp_dir=cfp_dir, label_dir=label_dir, mask_dir=mask_dir,
            pre_dir=pre_dir, use_cfp=use_cfp, gave_label_fmt=gave_fmt,
            do_augment=augment, resize_width=args.resize_width,
        )
        ds.files = files
        return ds

    train_ds = make_dataset(train_files, augment=True)

    # Optional validation dataset
    val_ds = None
    if args.val_data_dir:
        vd       = Path(args.val_data_dir)
        val_ds   = RetinalDataset(
            cfp_dir=vd / 'cfp',
            label_dir=resolve_dir(None, vd / 'label'),
            mask_dir=resolve_dir(None, vd / 'mask'),
            pre_dir=resolve_dir(None, vd / 'pre'),
            use_cfp=use_cfp, gave_label_fmt=gave_fmt,
            do_augment=False, resize_width=args.resize_width,
        )
    elif val_files:
        val_ds = make_dataset(val_files, augment=False)

    print(f'Training samples : {len(train_ds)}')
    if val_ds:
        print(f'Validation samples: {len(val_ds)}')

    # ── Model ───────────────────────────────────────────────────────────────
    in_ch = 6 if use_cfp else 3
    model = RRWNet(in_ch, 3, args.base_channels, args.num_iterations)
    if args.weights:
        ckpt = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(ckpt)
        print(f'Loaded weights from {args.weights}')
    model.to(device)

    # ── Loss ────────────────────────────────────────────────────────────────
    if args.model_type == 'av':
        criterion = RRLoss('av', lambda_bv=1.0, lambda_av=2.0, lambda_bg=0.5,
                           lambda_mx=0.5, lambda_cr=0.5)
    else:
        criterion = RRLoss('bv', lambda_mx=0.5, lambda_cr=0.2)

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    # ── Save config (compatible with infer.py) ──────────────────────────────
    config = {
        'model':          'RRWNet',
        'model_type':     args.model_type,
        'in_channels':    in_ch,
        'out_channels':   3,
        'base_channels':  args.base_channels,
        'num_iterations': args.num_iterations,
        'learning_rate':  args.lr,
        'seed':           args.seed,
        'use_cfp':        use_cfp,
        'gave_label_fmt': gave_fmt,
    }
    config_fn = save_dir / f'{args.model_type}_config.json'
    with open(config_fn, 'w') as f:
        json.dump(config, f, indent=4)
    print(f'Config saved to {config_fn}')

    # ── Training loop ────────────────────────────────────────────────────────
    iteration    = 0
    epoch        = 0
    running_loss = 0.0
    t0           = time.time()

    model.train()

    while iteration < args.max_iterations:
        epoch += 1
        loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2,
                            pin_memory=True)

        for inp, gt, mask in loader:
            if iteration >= args.max_iterations:
                break

            inp  = inp.to(device)
            gt   = gt.to(device)
            mask = mask.to(device)

            # Pad spatial dims to multiples of 32 (required by UNet)
            inp_p,  (pt, pb, pl, pr) = pad_batch(inp)
            gt_p,   _                = pad_batch(gt)
            mask_p, _                = pad_batch(mask)

            optimizer.zero_grad()
            predictions = model(inp_p)
            loss        = criterion(predictions, gt_p, mask_p)
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
                # Periodic checkpoint
                ckpt_fn = save_dir / f'{args.model_type}_{iteration:07d}.pth'
                torch.save(model.state_dict(), ckpt_fn)
                torch.save(model.state_dict(), save_dir / f'{args.model_type}_latest.pth')
                print(f'  → checkpoint saved: {ckpt_fn.name}')

                # Optional validation loss
                if val_ds:
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for v_inp, v_gt, v_mask in DataLoader(val_ds, batch_size=1):
                            v_inp    = v_inp.to(device)
                            v_gt     = v_gt.to(device)
                            v_mask   = v_mask.to(device)
                            v_inp_p, _  = pad_batch(v_inp)
                            v_gt_p,  _  = pad_batch(v_gt)
                            v_mask_p, _ = pad_batch(v_mask)
                            v_preds  = model(v_inp_p)
                            val_loss += criterion(v_preds, v_gt_p, v_mask_p).item()
                    print(f'  → val loss={val_loss / len(val_ds):.4f}')
                    model.train()

    # ── Final save ───────────────────────────────────────────────────────────
    final_fn = save_dir / f'{args.model_type}.pth'
    torch.save(model.state_dict(), final_fn)
    print(f'\nTraining complete. Final model: {final_fn}')
    print(f'Use with infer.py: --weights {final_fn} --model_type {args.model_type}')


if __name__ == '__main__':
    main()
