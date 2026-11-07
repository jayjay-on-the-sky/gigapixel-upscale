"""Self-check and benchmark for gigapixel-upscale."""
import time
import cv2
import numpy as np
from upscale import upscale


def identity_2x(batch: np.ndarray) -> np.ndarray:
    """Mock 2x upscale network operating on a batch (B, H, W, 3) in [-1, 1]."""
    return np.stack([cv2.resize(t, (t.shape[1] * 2, t.shape[0] * 2), interpolation=cv2.INTER_LINEAR) for t in batch])


def mock_color_fn(cinput: np.ndarray, output_m11: np.ndarray) -> np.ndarray:
    """Mock gclc-v1 color correction network taking raw [0, 1] and upscaled [-1, 1]."""
    assert cinput.ndim == 4 and output_m11.ndim == 4
    return output_m11 * 0.5 + 0.5


def test_seamless_and_fractional():
    rng = np.random.default_rng(42)
    img = rng.integers(0, 256, size=(300, 420, 3), dtype=np.uint8)

    # 1. Exact 2x scale on 1000px width (triggers float32 step rounding edge case): zero seam or right-edge drop
    flat = np.full((250, 1000, 3), 128, dtype=np.uint8)
    out_flat = upscale(flat, 2.0, identity_2x)
    assert out_flat.shape == (500, 2000, 3), f"Expected (500, 2000, 3), got {out_flat.shape}"
    assert np.all(out_flat == 128), f"Seam or edge drop detected: max diff {np.max(np.abs(out_flat.astype(int) - 128))}"

    # 2. Fractional 1.5x scale, gamma=True, and second-stage color_fn enabled
    out_frac = upscale(img, 1.5, identity_2x, color_fn=mock_color_fn, gamma=True)
    assert out_frac.shape == (450, 630, 3), f"Expected (450, 630, 3), got {out_frac.shape}"

    # 3. Edge cases: tiny image (<128x128) and zero overlap (overlap=0, discard=0)
    tiny = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    assert upscale(tiny, 2.0, identity_2x).shape == (64, 96, 3)
    assert upscale(tiny, 2.0, identity_2x, overlap=0, discard=0).shape == (64, 96, 3)

    # 4. Validation check on non-uint8 input
    try:
        upscale(tiny.astype(np.float32), 2.0, identity_2x)
        raise AssertionError("Should have rejected non-uint8 input")
    except ValueError:
        pass

    # 5. Speed check on 1080p image (1920x1080 -> 3840x2160, ~260 tiles)
    hd = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    out_hd = upscale(hd, 2.0, identity_2x, batch_size=32)
    dt = (time.perf_counter() - t0) * 1000
    assert out_hd.shape == (2160, 3840, 3)
    print(f"PASS: all checks ok | 1080p -> 4K tiling + blend in {dt:.1f} ms")


if __name__ == "__main__":
    test_seamless_and_fractional()
