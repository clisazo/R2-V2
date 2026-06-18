"""
standardize_datasets.py
=======================
Unifies all retinal vessel segmentation datasets into a common structure
mirroring the GAVE2 reference format:

    <OUTPUT_ROOT>/
        GAVE2/
            training/
                av/        (R=artery, G=all-vessels, B=vein, 3-channel PNG)
                images/    (CFP images, PNG)
                masks/     (FOV mask, PNG)
                masks_OD/  (optic disc mask, PNG)   ← GAVE2 only
                FFA_A/     (arterial FFA phase, PNG) ← GAVE2 only
                FFA_AV/    (AV FFA phase, PNG)       ← GAVE2 only
        Fundus-AVSeg/
            av/
            images/
            masks/
        HRF/
            av/            (binary 3-ch vessel mask – no A/V labels)
            images/
            masks/
        RITE/
            av/            (binary 3-ch vessel mask – no A/V labels)
            images/
            masks/
        LES-AV/
            av/
            images/
            masks/
        AFIO/
            av/
            images/
            masks/

Conventions
-----------
* All outputs are 3-channel PNG (RGB).
* AV masks  →  R=artery (incl. crossings), G=all-vessels (BV),
               B=vein (incl. crossings)  (values 0 or 255).
  This matches what train.py expects for `gave_label_fmt`
  (R=A, G=BV, B=V).  Crossing pixels belong to BOTH the artery and the
  vein tree, so they are set in R and B; the BV channel is the union.
* Binary masks (no A/V split) → R=0, B=0, G=vessel (BV only).  Empty R/B
  is the signal train.py's detect_bv_only() uses to skip A/V supervision.
* FOV / OD masks → single logical plane stored in all three channels.
* Source files are never modified.
"""

import os
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURE PATHS HERE
# ---------------------------------------------------------------------------
GAVE2_SRC        = Path("/home/Data/GAVE2")
FUNDUAVSEG_SRC   = Path("/home/Data/GAVE2/external_data/Fundus-AVSeg/train")
HRF_SRC          = Path("/home/Data/GAVE2/external_data/HRF_AVLabel_191219/train_karlsson_w1024")
RITE_SRC         = Path("/home/Data/GAVE2/external_data/RITE/train")
LESAV_SRC        = Path("/home/Data/LES-AV")
AFIO_SRC         = Path("/home/Data/AFIO/AV-20191104T162310Z-001/AV")

OUTPUT_ROOT      = Path("/home/Data/unified_GAVE_datasets")
# ---------------------------------------------------------------------------


# ───────────────────────────── helpers ──────────────────────────────────────

def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_rgb(path: Path) -> np.ndarray:
    """Load any image as uint8 RGB numpy array."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def load_gray(path: Path) -> np.ndarray:
    """Load any image as uint8 single-channel numpy array."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def save_rgb(arr: np.ndarray, path: Path) -> None:
    """Save uint8 (H,W,3) array as PNG."""
    Image.fromarray(arr, mode="RGB").save(path)


def gray_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Replicate a (H,W) array into (H,W,3)."""
    return np.stack([arr, arr, arr], axis=-1)


def binarize(arr: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Return 0/255 uint8 array."""
    return (arr > threshold).astype(np.uint8) * 255


def build_av_from_artery_vein(
    artery: np.ndarray,
    vein: np.ndarray,
) -> np.ndarray:
    """
    Combine separate binary artery and vein masks into a 3-channel AV mask
    in the unified convention:  R = artery, G = all-vessels (A∪V), B = vein.
    Crossing pixels (A∩V) are present in both the artery and vein source
    masks, so they naturally end up in both R and B.
    All inputs and outputs are 0/255.
    """
    a  = binarize(artery)
    v  = binarize(vein)
    bv = np.clip(a.astype(np.uint16) + v.astype(np.uint16), 0, 255).astype(np.uint8)
    return np.stack([a, bv, v], axis=-1).astype(np.uint8)


def extract_gave2_av(src: np.ndarray) -> np.ndarray:
    """
    GAVE2 / Fundus-AVSeg / LES-AV AV masks come as RGBA or RGB where
    R=arteries, G=common(A∩V crossing), B=veins.  Remap to the unified
    convention  R=artery(incl. crossings), G=all-vessels, B=vein(incl.
    crossings).
    """
    if src.ndim == 2:
        # Unexpected grayscale – replicate
        return gray_to_rgb(src)
    rgb   = src[:, :, :3]
    a_src = binarize(rgb[:, :, 0])   # pure artery
    cross = binarize(rgb[:, :, 1])   # common / crossing (A∩V)
    v_src = binarize(rgb[:, :, 2])   # pure vein

    artery = np.clip(a_src.astype(np.uint16) + cross, 0, 255).astype(np.uint8)
    vein   = np.clip(v_src.astype(np.uint16) + cross, 0, 255).astype(np.uint8)
    bv     = np.clip(a_src.astype(np.uint16) + cross + v_src, 0, 255).astype(np.uint8)
    return np.stack([artery, bv, vein], axis=-1).astype(np.uint8)


# ───────────────────────────── GAVE2 ────────────────────────────────────────

def process_gave2() -> None:
    print("\n── GAVE2 ──────────────────────────────────────────")
    src_train = GAVE2_SRC / "training"
    dst_train = ensure(OUTPUT_ROOT / "GAVE2" / "training")

    folder_map = {
        "av":       ("av",       None),        # handled specially
        "images":   ("images",   None),
        "masks":    ("masks",    None),
        "masks_OD": ("masks_OD", None),
        "FFA_A":    ("FFA_A",    None),
        "FFA_AV":   ("FFA_AV",  None),
    }

    for src_name, (dst_name, _) in folder_map.items():
        src_folder = src_train / src_name
        if not src_folder.exists():
            print(f"  [WARN] {src_folder} not found – skipping")
            continue

        dst_folder = ensure(dst_train / dst_name)
        files = sorted(src_folder.glob("*.png"))
        print(f"  {src_name}/  →  {dst_name}/   ({len(files)} files)")

        for f in tqdm(files, desc=f"  {src_name}", leave=False):
            dst_file = dst_folder / f.name

            if src_name == "av":
                # Load with PIL preserving all channels; convert RGBA→RGB
                raw = np.array(Image.open(f))
                arr = extract_gave2_av(raw)
                save_rgb(arr, dst_file)
            else:
                # Copy all other folders as-is (already PNG RGB/RGBA)
                raw = np.array(Image.open(f))
                if raw.ndim == 2:
                    arr = gray_to_rgb(raw)
                else:
                    arr = raw[:, :, :3]
                save_rgb(arr.astype(np.uint8), dst_file)


# ───────────────────────────── Fundus-AVSeg ─────────────────────────────────

def process_fundus_avseg() -> None:
    print("\n── Fundus-AVSeg ───────────────────────────────────")
    dst = OUTPUT_ROOT / "Fundus-AVSeg"

    folders = {
        "av3":            ensure(dst / "av"),
        "enhanced":       ensure(dst / "images"),
        "enhanced_masks": ensure(dst / "masks"),
    }

    for src_name, dst_folder in folders.items():
        src_folder = FUNDUAVSEG_SRC / src_name
        if not src_folder.exists():
            print(f"  [WARN] {src_folder} not found – skipping")
            continue

        files = sorted(src_folder.glob("*.png"))
        print(f"  {src_name}/  →  {dst_folder.name}/   ({len(files)} files)")

        for f in tqdm(files, desc=f"  {src_name}", leave=False):
            dst_file = dst_folder / f.name
            raw = np.array(Image.open(f))

            if src_name == "av3":
                # RGB, R=artery G=common B=vein – same convention as GAVE2
                arr = extract_gave2_av(raw)
            else:
                if raw.ndim == 2:
                    arr = gray_to_rgb(raw)
                else:
                    arr = raw[:, :, :3]

            save_rgb(arr.astype(np.uint8), dst_file)


# ───────────────────────────── HRF ──────────────────────────────────────────

def process_hrf() -> None:
    """
    HRF masks are single-channel binary (no A/V split).
    We store them as 3-channel binary PNG (all channels equal).
    FOV mask files follow pattern: XX_{suffix}_mask.png
    Image / AV files follow pattern: XX_{suffix}.png
    """
    print("\n── HRF ────────────────────────────────────────────")
    dst_av     = ensure(OUTPUT_ROOT / "HRF" / "av")
    dst_images = ensure(OUTPUT_ROOT / "HRF" / "images")
    dst_masks  = ensure(OUTPUT_ROOT / "HRF" / "masks")

    src_av     = HRF_SRC / "av3"
    src_images = HRF_SRC / "enhanced"
    src_masks  = HRF_SRC / "enhanced_masks"

    for folder, dst_folder, description in [
        (src_av,     dst_av,     "av3 (binary vessel)"),
        (src_images, dst_images, "enhanced (CFP)"),
        (src_masks,  dst_masks,  "enhanced_masks (FOV)"),
    ]:
        if not folder.exists():
            print(f"  [WARN] {folder} not found – skipping")
            continue

        files = sorted(folder.glob("*.png"))
        print(f"  {folder.name}/  →  {dst_folder.name}/   ({len(files)} files)")

        for f in tqdm(files, desc=f"  {folder.name}", leave=False):
            raw = np.array(Image.open(f))

            if description.startswith("av3"):
                # Vessel-only mask (no A/V split) → BV channel (G); R/B empty
                gray  = binarize(raw if raw.ndim == 2 else raw[:, :, 0])
                zeros = np.zeros_like(gray)
                arr   = np.stack([zeros, gray, zeros], axis=-1)
            else:
                if raw.ndim == 2:
                    arr = gray_to_rgb(raw)
                else:
                    arr = raw[:, :, :3]

            save_rgb(arr.astype(np.uint8), dst_folder / f.name)


# ───────────────────────────── RITE ─────────────────────────────────────────

def process_rite() -> None:
    """
    RITE masks are 3-channel but all channels identical (general vessel seg,
    no A/V differentiation).  We store the first channel replicated to 3 ch.
    """
    print("\n── RITE ───────────────────────────────────────────")
    dst_av     = ensure(OUTPUT_ROOT / "RITE" / "av")
    dst_images = ensure(OUTPUT_ROOT / "RITE" / "images")
    dst_masks  = ensure(OUTPUT_ROOT / "RITE" / "masks")

    src_av     = RITE_SRC / "av3"
    src_images = RITE_SRC / "enhanced"
    src_masks  = RITE_SRC / "enhanced_masks"

    for folder, dst_folder, is_av in [
        (src_av,     dst_av,     True),
        (src_images, dst_images, False),
        (src_masks,  dst_masks,  False),
    ]:
        if not folder.exists():
            print(f"  [WARN] {folder} not found – skipping")
            continue

        files = sorted(folder.glob("*.png"))
        print(f"  {folder.name}/  →  {dst_folder.name}/   ({len(files)} files)")

        for f in tqdm(files, desc=f"  {folder.name}", leave=False):
            raw = np.array(Image.open(f))

            if is_av:
                # Vessel-only mask (no A/V split) → BV channel (G); R/B empty
                ch    = raw[:, :, 0] if raw.ndim == 3 else raw
                gray  = binarize(ch)
                zeros = np.zeros_like(gray)
                arr   = np.stack([zeros, gray, zeros], axis=-1)
            else:
                if raw.ndim == 2:
                    arr = gray_to_rgb(raw)
                else:
                    arr = raw[:, :, :3]

            save_rgb(arr.astype(np.uint8), dst_folder / f.name)


# ───────────────────────────── LES-AV ───────────────────────────────────────

def process_lesav() -> None:
    """
    LES-AV provides an 'arteries-and-veins' folder (R=artery, G=common, B=vein)
    which maps directly to our AV convention.
    FOV masks are .gif files – we convert them to PNG.
    """
    print("\n── LES-AV ─────────────────────────────────────────")
    dst_av     = ensure(OUTPUT_ROOT / "LES-AV" / "av")
    dst_images = ensure(OUTPUT_ROOT / "LES-AV" / "images")
    dst_masks  = ensure(OUTPUT_ROOT / "LES-AV" / "masks")

    # ── AV masks ──────────────────────────────────────────
    src_av = LESAV_SRC / "arteries-and-veins"
    if not src_av.exists():
        print(f"  [WARN] {src_av} not found – skipping av")
    else:
        files = sorted(src_av.glob("*.png"))
        print(f"  arteries-and-veins/  →  av/   ({len(files)} files)")
        for f in tqdm(files, desc="  av", leave=False):
            raw = np.array(Image.open(f))
            arr = extract_gave2_av(raw)
            save_rgb(arr, dst_av / f.name)

    # ── CFP images ────────────────────────────────────────
    src_images = LESAV_SRC / "images"
    if not src_images.exists():
        print(f"  [WARN] {src_images} not found – skipping images")
    else:
        files = sorted(src_images.glob("*.png"))
        print(f"  images/  →  images/   ({len(files)} files)")
        for f in tqdm(files, desc="  images", leave=False):
            raw = np.array(Image.open(f).convert("RGB"))
            save_rgb(raw, dst_images / f.name)

    # ── FOV masks (.gif → .png) ───────────────────────────
    src_masks = LESAV_SRC / "masks"
    if not src_masks.exists():
        print(f"  [WARN] {src_masks} not found – skipping masks")
    else:
        files = sorted(src_masks.glob("*.gif"))
        print(f"  masks/  →  masks/   ({len(files)} .gif files)")
        for f in tqdm(files, desc="  masks", leave=False):
            # Strip _mask suffix if present to align filenames with images
            stem = f.stem.replace("_mask", "")
            dst_file = dst_masks / f"{stem}.png"
            raw = np.array(Image.open(f).convert("L"))
            arr = gray_to_rgb(binarize(raw))
            save_rgb(arr, dst_file)


# ───────────────────────────── AFIO ─────────────────────────────────────────

# Keyword sets used to classify the suffix part of AFIO filenames
# (the part after '--', lowercased).  Any suffix whose lowercase form contains
# one of these keywords is assigned to that role.  Order matters: more
# specific keywords should come first within each group.
_AFIO_ARTERY_KEYWORDS = [
    "artery", "artry", "arteries", "arter",        # correct + common typos
]
_AFIO_VEIN_KEYWORDS = [
    "veins", "vein", "veisn", "viens", "vens",     # correct + common typos
]
_AFIO_VESSEL_KEYWORDS = [
    "vessels", "vessel",
]
_AFIO_BOTH_KEYWORDS = [
    "both",
]

# Image extensions accepted for AFIO files (lowercased)
_AFIO_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def _classify_afio_suffix(suffix: str) -> str | None:
    """
    Given the lowercased suffix token (text after '--'), return one of:
      'artery' | 'vein' | 'vessel' | 'both' | None (= CFP image or unknown)
    """
    for kw in _AFIO_ARTERY_KEYWORDS:
        if kw in suffix:
            return "artery"
    for kw in _AFIO_VEIN_KEYWORDS:
        if kw in suffix:
            return "vein"
    for kw in _AFIO_VESSEL_KEYWORDS:
        if kw in suffix:
            return "vessel"
    for kw in _AFIO_BOTH_KEYWORDS:
        if kw in suffix:
            return "both"
    return None  # no '--' suffix → CFP image candidate


def _parse_afio_folder(folder: Path) -> dict:
    """
    Scan all files in an AFIO IMXXX folder and return a dict:
      {
        'id':      str,           # folder name, e.g. 'IM042'
        'image':   Path | None,   # CFP image
        'artery':  Path | None,
        'vein':    Path | None,
        'vessel':  Path | None,
        'both':    Path | None,
        'mismatched': [Path],     # files whose embedded ID ≠ folder ID
      }

    Classification is purely keyword-based so it survives any typo in the
    suffix as long as it is recognisably related to one of the four roles.
    """
    fid = folder.name.upper()
    result = {
        "id": folder.name,
        "image": None, "artery": None, "vein": None,
        "vessel": None, "both": None,
        "mismatched": [],
    }

    for f in folder.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in _AFIO_IMAGE_EXTS:
            continue  # skip non-image files (txt, xml, …)

        # ── ID consistency check ──────────────────────────────────────────
        # Embedded ID = everything before the first '--' (or the whole stem)
        embedded_id = f.stem.split("--")[0].upper()
        if embedded_id != fid:
            result["mismatched"].append(f)
            continue

        # ── Role classification ───────────────────────────────────────────
        parts = f.stem.split("--", 1)
        if len(parts) == 1:
            # No '--' → this is the CFP image (e.g. IM042.JPG)
            if result["image"] is None:
                result["image"] = f
            else:
                # Multiple bare-stem files: keep the first encountered
                pass
        else:
            suffix = parts[1].lower()
            role = _classify_afio_suffix(suffix)
            if role is None:
                # Unrecognised suffix – log but don't block the folder
                print(f"    [NOTE] Unrecognised suffix '{parts[1]}' in {f.name} – ignoring file")
            elif result[role] is None:
                result[role] = f
            else:
                # Duplicate role – keep the first, warn
                print(f"    [NOTE] Duplicate '{role}' file in {folder.name}: "
                      f"{result[role].name} vs {f.name} – keeping first")

    return result


def _load_afio_binary_mask(path: Path, target_color: str) -> np.ndarray:
    """
    AFIO binary masks are stored as colored JPEGs:
      artery  → red pixels   (R >> G, B)
      vein    → blue pixels  (B >> R, G)
      vessel  → dark pixels on white background

    We extract a binary mask by thresholding the dominant channel.
    Returns a (H,W) uint8 array with values 0 or 255.
    """
    img = np.array(Image.open(path).convert("RGB")).astype(np.int16)
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    if target_color == "red":
        mask = (R - np.maximum(G, B) > 40)
    elif target_color == "blue":
        mask = (B - np.maximum(R, G) > 40)
    elif target_color == "dark":
        lum = (R.astype(np.uint32) + G + B) // 3
        mask = (lum < 128)
    else:
        raise ValueError(f"Unknown target_color: {target_color}")

    return mask.astype(np.uint8) * 255


def process_afio() -> None:
    """
    AFIO stores each sample in its own IMXXX/ folder.

    File roles are detected by keyword matching on the suffix after '--',
    so variants like 'artry', 'arteries', 'veisn', 'viens', etc. are all
    handled automatically.  The mapping is:
      (no suffix)              → CFP image
      artery / artry / …      → artery mask  (red on white JPEG)
      veins / vein / veisn /… → vein mask    (blue on white JPEG)
      vessels / vessel         → all-vessel mask (black on white JPEG)
      both                     → ignored (composite overlay, not used)

    Folders where any file's embedded ID doesn't match the folder name are
    skipped to avoid cross-sample contamination.
    """
    print("\n── AFIO ───────────────────────────────────────────")
    dst_av     = ensure(OUTPUT_ROOT / "AFIO" / "av")
    dst_images = ensure(OUTPUT_ROOT / "AFIO" / "images")
    dst_masks  = ensure(OUTPUT_ROOT / "AFIO" / "masks")

    im_folders = sorted(
        f for f in AFIO_SRC.iterdir()
        if f.is_dir() and re.match(r"^IM\d+$", f.name, re.IGNORECASE)
    )
    print(f"  Found {len(im_folders)} IMXXX folders")

    skipped = 0
    processed = 0

    for folder in tqdm(im_folders, desc="  AFIO"):
        info = _parse_afio_folder(folder)
        fid  = info["id"]

        # ── reject folders with ID-mismatched files ─────────────────────────
        if info["mismatched"]:
            names = ", ".join(f.name for f in info["mismatched"])
            print(f"  [WARN] ID mismatch in {fid} ({names}) – skipping folder")
            skipped += 1
            continue

        # ── require at minimum the CFP image + artery + vein masks ──────────
        missing = [role for role in ("image", "artery", "vein")
                   if info[role] is None]
        if missing:
            print(f"  [WARN] {fid}: missing {missing} – skipping")
            skipped += 1
            continue

        # ── load & convert ──────────────────────────────────────────────────
        try:
            img_arr = np.array(Image.open(info["image"]).convert("RGB"))
            artery  = _load_afio_binary_mask(info["artery"], "red")
            vein    = _load_afio_binary_mask(info["vein"],   "blue")
            av_arr  = build_av_from_artery_vein(artery, vein)

            # FOV mask derived from CFP luminance (no explicit mask in AFIO)
            fov     = (img_arr.max(axis=2) > 10).astype(np.uint8) * 255
            fov_arr = gray_to_rgb(fov)

        except Exception as e:
            print(f"  [ERROR] {fid}: {e} – skipping")
            skipped += 1
            continue

        # ── save ────────────────────────────────────────────────────────────
        save_rgb(av_arr,   dst_av     / f"{fid}.png")
        save_rgb(img_arr,  dst_images / f"{fid}.png")
        save_rgb(fov_arr,  dst_masks  / f"{fid}.png")
        processed += 1

    print(f"  Done: {processed} processed, {skipped} skipped")


# ───────────────────────────── main ─────────────────────────────────────────

def main() -> None:
    print("=" * 55)
    print(" Dataset standardization → GAVE2 reference format")
    print("=" * 55)
    print(f"Output root: {OUTPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    process_gave2()
    process_fundus_avseg()
    process_hrf()
    process_rite()
    process_lesav()
    process_afio()

    print("\n" + "=" * 55)
    print(" All done!  Summary of output:")
    print("=" * 55)
    for dataset_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for sub in sorted(dataset_dir.rglob("*")):
            if sub.is_dir():
                n = len(list(sub.glob("*")))
                rel = sub.relative_to(OUTPUT_ROOT)
                print(f"  {rel}/  ({n} files)")


if __name__ == "__main__":
    main()