# gigapixel-upscale

Minimal, fast Python reconstruction of **Topaz Gigapixel AI's tiled upscale pipeline** (`TBlockProc` + `StandardCCImage` + `ResizeEnhancement`), built with `ponytail` principles (~100 lines, zero bloat, pure OpenCV + NumPy + ONNXRuntime).

---

## Features

- **Power-of-2 Model Snapping (`TBlockProc::TBlockProc` `0x4db94`)**: Snaps scale factors to $2^{\text{round}(\log_2(\text{scale}))}$ with $128 \times 128$ tiles, $4\text{px}$ discard border, and $32\text{px}$ overlap.
- **Batch Streaming & Single-Pass Pad (`StandardCCImage::process` `0x473fc`)**: Pads the whole image once with `cv2.BORDER_REFLECT_101` and streams bounded batches directly onto the output canvas (fixed ~50 MB intermediate RAM even for 4K/8K images).
- **Separable 1D Feather Blending (`TBlockProc::mergeBlock` `0x4edd4`)**: Linear partition-of-unity ramps along X and Y guarantee **zero seam error** (`max diff = 0`).
- **Face Recovery (`recover_faces`)**: Restores low-res faces using YuNet 5-point alignment + GFPGAN v1.4 ONNX inverse-warp feather blending.

---

## Quickstart

### Prerequisites

```bash
pip install numpy opencv-python onnxruntime
```

Download models:
- [realesr-general-x4v3.onnx](https://github.com/xinntao/Real-ESRGAN)
- [face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [gfpgan_1.4.onnx](https://github.com/TencentARC/GFPGAN)

### Usage

```python
import cv2
from upscale import upscale, make_onnx_fn, recover_faces

img = cv2.imread("input.jpg")

# 1. Tiled AI Upscale
onnx_fn = make_onnx_fn("realesr-general-x4v3.onnx")
up = upscale(img, target_scale=2.0, upscale_fn=onnx_fn, batch_size=8)

# 2. Face Recovery (strength 0.2-0.8)
out, faces = recover_faces(
    up, 
    "gfpgan_1.4.onnx", 
    detector_path="face_detection_yunet_2023mar.onnx", 
    strength=0.3
)
cv2.imwrite("output.jpg", out)
```

---

## Run Tests & Benchmark

```bash
python3 test_upscale.py
```
