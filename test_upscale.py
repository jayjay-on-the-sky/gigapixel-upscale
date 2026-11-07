"""Self-check and benchmark for gigapixel-upscale."""
import time
import cv2
import numpy as np
from upscale import upscale


def identity_2x(batch: np.ndarray) -> np.ndarray:
    """Mock 2x upscale network operating on a batch (B, H, W, 3) in [-1, 1]."""
    return np.stack([cv2.resize(t, (t.shape[1] * 2, t.shape[0] * 2), interpolation=cv2.INTER_LINEAR) for t in batch])


def test_seamless_and_fractional():
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(300, 420, 3), dtype=np.uint8)

    # 1. Exact 2x scale: constant-color image must have zero seam error anywhere
    flat = np.full((250, 350, 3), 128, dtype=np.uint8)
    out_flat = upscale(flat, 2.0, identity_2x)
    assert out_flat.shape == (500, 700, 3), f"Expected (500, 700, 3), got {out_flat.shape}"
    assert np.all(out_flat == 128), f"Seam or brightness drift detected: max diff {np.max(np.abs(out_flat.astype(int) - 128))}"

    # 2. Fractional 1.5x scale (runs 2x model + Lanczos4 postProc) and gamma enabled
    out_frac = upscale(img, 1.5, identity_2x, gamma=True)
    assert out_frac.shape == (450, 630, 3), f"Expected (450, 630, 3), got {out_frac.shape}"

    # 3. Edge case: tiny image smaller than 128x128 tile
    tiny = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    out_tiny = upscale(tiny, 2.0, identity_2x)
    assert out_tiny.shape == (64, 96, 3)

    # 4. Speed check on 1080p image (1920x1080 -> 3840x2160, ~260 tiles)
    hd = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    out_hd = upscale(hd, 2.0, identity_2x, batch_size=32)
    dt = (time.perf_counter() - t0) * 1000
    assert out_hd.shape == (2160, 3840, 3)
    print(f"PASS: all checks ok | 1080p -> 4K tiling + blend in {dt:.1f} ms")


if __name__ == "__main__":
    test_seamless_and_fractional()
