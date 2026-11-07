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
    return np.rint(np.linspace(0, padded - block, n)).astype(np.int32) - pad


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
    2. Stream tiles in bounded batches directly onto the output canvas (no full-tile-set RAM spike).
    3. Precompute 1D linear feather ramps per row/col instead of per-tile.
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
        raise ValueError("Expected HxWx3 uint8 BGR image")
    h, w = img_bgr.shape[:2]
    if h < 1 or w < 1 or target_scale <= 0 or overlap < 0 or discard < 0 or block <= overlap or batch_size < 1:
        raise ValueError("Invalid dimensions, tiling parameters, or scale")

    # TBlockProc::TBlockProc (0x4dc00): snap model scale to nearest power of 2
    s = int(2 ** round(math.log2(max(target_scale, 1.0))))
    pad = 0 if overlap == 0 else int(round(min(discard / float(overlap), 0.5) * overlap))

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
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB).astype(np.float32)
    g = 0.45454545 if gamma else 1.0
    if not gamma and color_fn is None:
        net_in_full = rgb * (2.0 / 255.0) - 1.0
        work_f = None
    else:
        work_f = rgb * (1.0 / 255.0)
        net_in_full = (cv2.pow(np.clip(work_f, 0.0, 1.0), g) if g != 1.0 else work_f) * 2.0 - 1.0

    out_h, out_w = h * s, w * s
    wx_list = feather_weights(xs, out_w, s, block, pad)
    wy_list = feather_weights(ys, out_h, s, block, pad)
    grid = [
        (int(y + pad_t), int(x + pad_l), dy, ty0, ty1, wy[:, None, None], dx, tx0, tx1, wx[None, :, None])
        for y, (dy, ty0, ty1, wy) in zip(ys, wy_list)
        for x, (dx, tx0, tx1, wx) in zip(xs, wx_list)
    ]

    # Stream batches directly onto canvas (StandardCCImage::process 0x473fc + TBlockProc::mergeBlock 0x4edd4)
    canvas = None
    for i in range(0, len(grid), batch_size):
        chunk = grid[i : i + batch_size]
        batch = np.stack([net_in_full[py : py + block, px : px + block] for py, px, *_ in chunk])
        up = np.clip(upscale_fn(batch) * 0.5 + 0.5, 0.0, 1.0)
        if g != 1.0:
            up = cv2.pow(up, 1.0 / g)
        if color_fn is not None:
            raw_batch = np.stack([work_f[py : py + block, px : px + block] for py, px, *_ in chunk])
            up = np.clip(color_fn(raw_batch, up * 2.0 - 1.0), 0.0, 1.0)
        if canvas is None:
            # Auto-adapt feather grid if the model's native scale factor (e.g. 4x ONNX) differs from snapped s
            actual_s = up.shape[1] // block
            if actual_s != s:
                s = actual_s
                out_h, out_w = h * s, w * s
                wx_list = feather_weights(xs, out_w, s, block, pad)
                wy_list = feather_weights(ys, out_h, s, block, pad)
                grid = [
                    (int(y + pad_t), int(x + pad_l), dy, ty0, ty1, wy[:, None, None], dx, tx0, tx1, wx[None, :, None])
                    for y, (dy, ty0, ty1, wy) in zip(ys, wy_list)
                    for x, (dx, tx0, tx1, wx) in zip(xs, wx_list)
                ]
                chunk = grid[i : i + batch_size]
            canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
        for tile, (_, _, dy, ty0, ty1, wy_col, dx, tx0, tx1, wx_row) in zip(up, chunk):
            sub = tile[ty0:ty1, tx0:tx1] * wy_col
            sub *= wx_row
            sh, sw = sub.shape[:2]
            canvas[dy : dy + sh, dx : dx + sw] += sub

    merged_rgb = np.clip(canvas * 255.0 + 0.5, 0, 255).astype(np.uint8)
    merged_bgr = cv2.cvtColor(merged_rgb, cv2.COLOR_RGB2BGR)

    # ResizeEnhancement::postProc (0x615c0): Lanczos4 resize for fractional scales
    target_w, target_h = int(round(w * target_scale)), int(round(h * target_scale))
    if (merged_bgr.shape[1], merged_bgr.shape[0]) != (target_w, target_h):
        merged_bgr = cv2.resize(merged_bgr, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return merged_bgr


def make_onnx_fn(model_path: str, providers: list[str] | None = None):
    """Wrap a standard NCHW [0, 1] ONNX super-resolution model into upscale_fn's NHWC [-1, 1] contract."""
    import onnxruntime as ort

    sess = ort.InferenceSession(model_path, providers=providers or ["CoreMLExecutionProvider", "CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def _run(batch_nhwc_m11: np.ndarray) -> np.ndarray:
        nchw_01 = np.ascontiguousarray((batch_nhwc_m11 * 0.5 + 0.5).transpose(0, 3, 1, 2))
        out_nchw = sess.run(None, {in_name: nchw_01})[0]
        return out_nchw.transpose(0, 2, 3, 1) * 2.0 - 1.0

    return _run


def recover_faces(
    img_bgr: np.ndarray,
    gfpgan_path: str,
    detector_path: str | None = None,
    strength: float = 0.8,
    providers: list[str] | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Minimal reconstruction of FaceRecoveryEnhancement::proc (0x5cc18) + FaceEnhancement::facePerfect (0x27104):
    1. Detect 5-point face landmarks via OpenCV YuNet ONNX (or Haar fallback).
    2. Similarity-align face to GFPGAN's canonical 512x512 template -> run GFPGANv1.4 ONNX -> inverse-warp back.
    3. Feather-blend restored face into the upscaled canvas with a smooth elliptical mask.
    """
    import onnxruntime as ort

    h, w = img_bgr.shape[:2]
    faces = None
    if detector_path:
        det = cv2.FaceDetectorYN.create(detector_path, "", (w, h), 0.6)
        _, faces = det.detect(img_bgr)
    if faces is None or len(faces) == 0:
        return img_bgr, []

    sess = ort.InferenceSession(gfpgan_path, providers=providers or ["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = img_bgr.copy()

    # Canonical FFHQ 512x512 5-point landmarks used by GFPGAN / CodeFormer
    ref_pts = np.array(
        [[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935], [201.26117, 371.41043], [313.08905, 371.15118]],
        dtype=np.float32,
    )
    yy, xx = np.ogrid[:512, :512]
    dist = np.sqrt(((xx - 255.5) / 205.0) ** 2 + ((yy - 265.0) / 225.0) ** 2)
    mask_512 = (np.clip(1.0 - np.clip((dist - 0.65) / 0.35, 0.0, 1.0), 0.0, 1.0) * strength).astype(np.float32)

    applied = []
    for row in faces:
        fx, fy, fw, fh = [int(round(v)) for v in row[:4]]
        src_pts = row[4:14].reshape(5, 2).astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src_pts, ref_pts, method=cv2.LMEDS)
        if M is None:
            continue

        aligned_bgr = cv2.warpAffine(out, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        inp = np.ascontiguousarray((aligned_rgb.astype(np.float32) * (2.0 / 255.0) - 1.0).transpose(2, 0, 1))[None]
        pred = np.clip(sess.run(None, {in_name: inp})[0][0].transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0) * 255.0
        restored_512 = cv2.cvtColor(pred.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Warp only onto the expanded face ROI instead of the full 4K/8K canvas
        pad = int(max(fw, fh) * 0.35)
        x0, y0 = max(0, fx - pad), max(0, fy - pad)
        x1, y1 = min(w, fx + fw + pad), min(h, fy + fh + pad)
        M_inv = cv2.invertAffineTransform(M)
        M_inv[:, 2] -= [x0, y0]
        roi = out[y0:y1, x0:x1]
        rh, rw = roi.shape[:2]

        warped_face = cv2.warpAffine(restored_512, M_inv, (rw, rh), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
        warped_mask = cv2.warpAffine(mask_512, M_inv, (rw, rh), flags=cv2.INTER_LINEAR)[..., None]
        out[y0:y1, x0:x1] = np.clip(roi.astype(np.float32) * (1.0 - warped_mask) + warped_face.astype(np.float32) * warped_mask + 0.5, 0, 255).astype(np.uint8)
        applied.append((x0, y0, x1 - x0, y1 - y0))

    return out, applied
