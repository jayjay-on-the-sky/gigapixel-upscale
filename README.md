# gigapixel-upscale

Minimal, fast Python reconstruction of Topaz Gigapixel AI's tiled upscale pipeline (`TBlockProc` + `StandardCCImage` + `ResizeEnhancement`), built with `ponytail` principles (~90 lines of core code, only `numpy` and `opencv-python`).

## How it works

1. **Power-of-2 Model Snapping (`TBlockProc::TBlockProc` `0x4db94`)**: Rounds the requested scale factor to the nearest power of 2 ($2^{\text{round}(\log_2(\text{scale}))}$) and spreads $128 \times 128$ tiles evenly from edge to edge with a $4\text{px}$ border discard (`pad`) and $32\text{px}$ overlap.
2. **Whole-Image Pre-Padding & Batched Inference (`StandardCCImage::process` `0x473fc`)**: Instead of calling `cv2.copyMakeBorder`, `cvtColor`, and `np.power` on each tile individually, pads and normalizes the full image once, slices all tiles into a contiguous batch `(N, 128, 128, 3)`, and runs the upscale + optional color-correction model in batches.
3. **Separable 1D Feathering (`TBlockProc::mergeBlock` `0x4edd4`)**: Precomputes 1D linear feather ramps along X and Y once (`wx_list`, `wy_list`) and blends overlapping regions via outer product.
4. **Fractional Post-Scaling (`ResizeEnhancement::postProc` `0x615c0`)**: Resizes to the exact target dimensions with `cv2.INTER_LANCZOS4` when the target scale is non-power-of-2 (e.g. `1.5x`, `3.2x`).

## Usage

```python
import cv2
from upscale import upscale

img = cv2.imread("input.jpg")
# Plug in any batched model function: (B, H, W, 3) float32 in [-1, 1] -> (B, H*s, W*s, 3) in [-1, 1]
out = upscale(img, target_scale=2.0, upscale_fn=my_model, batch_size=16)
cv2.imwrite("output.jpg", out)
```

## Run tests & benchmark

```bash
python3 test_upscale.py
```
