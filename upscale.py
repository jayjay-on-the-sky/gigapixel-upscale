"""Minimal, fast reconstruction of Topaz Gigapixel's TBlockProc + StandardCCImage pipeline."""
import math
import cv2
import numpy as np


def grid_starts(dim: int, block: int, overlap: int, pad: int) -> np.ndarray:
    """Evenly spaced tile coordinates matching TBlockProc::TBlockProc (0x4db94)."""
    padded, step = dim + 2 * pad, block - overlap
    n = max(1, (padded - overlap + step - 1) // step)
    if n == 1:
        return np.array([-pad], dtype=np.int32)
    return (np.arange(n, dtype=np.float32) * ((padded - block) / (n - 1))).astype(np.int32) - pad


def feather_weights(starts: np.ndarray, out_dim: int, s: int, block: int = 128, pad: int = 4) -> list[tuple[int, int, int, np.ndarray]]:
    """Precompute 1D trim bounds and partition-of-unity linear ramps once per axis (ParallelBlendXY 0x4edd4)."""
    trim, tb, n = pad * s, block * s, len(starts)
    axis = []
    for i, c in enumerate(starts):
        oc = int(c) * s
        t0 = trim if i > 0 else max(0, -oc)
        t1 = tb - (trim if i < n - 1 else max(0, (oc + tb) - out_dim))
        w = np.ones(t1 - t0, dtype=np.float32)
        if i > 0 and (ov := max(0, (int(starts[i - 1]) * s + tb - trim) - (oc + t0))) > 0:
            w[:ov] = (np.arange(ov, dtype=np.float32) + 0.5) / ov
        if i < n - 1 and (ov := max(0, (oc + t1) - (int(starts[i + 1]) * s + trim))) > 0:
            w[-ov:] = 1.0 - (np.arange(ov, dtype=np.float32) + 0.5) / ov
        axis.append((oc + t0, t0, t1, w))
    return axis


def upscale(
    img_bgr: np.ndarray,
    target_scale: float,
    upscale_fn,
    color_fn=None,
    block: int = 128,
    overlap: int = 32,
    discard: int = 4,
    batch_size: int = 16,
    gamma: bool = False,
) -> np.ndarray:
    """
    Fast tiled upscale pipeline:
    1. Pad entire image once with BORDER_REFLECT_101 instead of per-tile copyMakeBorder.
    2. Vectorize tile extraction and run model inference in batches.
    3. Precompute 1D linear feather ramps per row/col instead of per-tile.
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("Expected HxWx3 BGR image")
    h, w = img_bgr.shape[:2]
    if h < 1 or w < 1 or target_scale <= 0:
        raise ValueError("Invalid dimensions or scale")

    # TBlockProc::TBlockProc (0x4dc00): snap model scale to nearest power of 2
    s = int(2 ** round(math.log2(max(target_scale, 1.0))))
    pad = int(round(min(discard / float(overlap), 0.5) * overlap))

    xs = grid_starts(w, block, overlap, pad)
    ys = grid_starts(h, block, overlap, pad)

    # Pad full image once to cover any out-of-bounds tile coordinates
    pad_l, pad_t = max(0, int(-xs[0])), max(0, int(-ys[0]))
    pad_r = max(0, int(xs[-1] + block - w))
    pad_b = max(0, int(ys[-1] + block - h))
    # shortcut: BORDER_REFLECT_101 requires h > max(pad_t, pad_b) and w > max(pad_l, pad_r), fall back to BORDER_REFLECT for tiny <5px inputs
    border = cv2.BORDER_REFLECT_101 if min(h, w) > max(pad_l, pad_t, pad_r, pad_b) else cv2.BORDER_REFLECT
    work = cv2.copyMakeBorder(img_bgr, pad_t, pad_b, pad_l, pad_r, border)

    # Normalize & BGR->RGB once on the full padded image instead of per tile
    work_f = cv2.cvtColor(work, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
    g = 0.45454545 if gamma else 1.0
    net_in_full = (cv2.pow(np.clip(work_f, 0.0, 1.0), g) if g != 1.0 else work_f) * 2.0 - 1.0

    coords = [(int(y + pad_t), int(x + pad_l)) for y in ys for x in xs]
    patches_in = np.stack([net_in_full[y : y + block, x : x + block] for y, x in coords])

    # Run model in batches (StandardCCImage::process 0x473fc)
    out_batches = []
    for i in range(0, len(patches_in), batch_size):
        batch = patches_in[i : i + batch_size]
        up = np.clip(upscale_fn(batch) * 0.5 + 0.5, 0.0, 1.0)
        if g != 1.0:
            up = cv2.pow(up, 1.0 / g)
        if color_fn is not None:
            raw_batch = np.stack([work_f[y : y + block, x : x + block] for y, x in coords[i : i + batch_size]])
            up = np.clip(color_fn(raw_batch, up * 2.0 - 1.0), 0.0, 1.0)
        out_batches.append(up)
    tiles_out = np.concatenate(out_batches, axis=0)

    # Merge tiles with precomputed 1D feather weights (TBlockProc::mergeBlock 0x4edd4)
    # Since linear ramps on overlapping neighbors sum to 1.0, each tile is multiplied by its 2D weight and added directly without a second weights buffer or division pass.
    out_h, out_w = h * s, w * s
    canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    wx_list = feather_weights(xs, out_w, s, block, pad)
    wy_list = feather_weights(ys, out_h, s, block, pad)

    idx = 0
    for dy, ty0, ty1, wy in wy_list:
        wy_col = wy[:, None, None]
        for dx, tx0, tx1, wx in wx_list:
            sub = tiles_out[idx, ty0:ty1, tx0:tx1] * wy_col
            sub *= wx[None, :, None]
            sh, sw = sub.shape[:2]
            canvas[dy : dy + sh, dx : dx + sw] += sub
            idx += 1

    merged_rgb = (canvas * 255.0 + 0.5).astype(np.uint8)
    merged_bgr = cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR)

    # ResizeEnhancement::postProc (0x615c0): Lanczos4 resize for fractional scales
    target_w, target_h = int(round(w * target_scale)), int(round(h * target_scale))
    if (merged_bgr.shape[1], merged_bgr.shape[0]) != (target_w, target_h):
        merged_bgr = cv2.resize(merged_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return merged_bgr
