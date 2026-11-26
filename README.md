# Tiled Upscale

Minimal, fast Python tiled upscale pipeline with power-of-2 scale snapping, bounded batch streaming, separable feather blending, and optional face recovery. Pure OpenCV + NumPy + ONNXRuntime.

---

## Features

- **Power-of-2 Model Snapping**: Snaps scale factors to $2^{\text{round}(\log_2(\text{scale}))}$ with $128 \times 128$ tiles, $4\text{px}$ discard border, and $32\text{px}$ overlap.
- **Batch Streaming & Single-Pass Pad**: Pads the whole image once with `cv2.BORDER_REFLECT_101` and streams bounded batches directly onto the output canvas (fixed ~50 MB intermediate RAM even for 4K/8K images).
- **Separable 1D Feather Blending**: Linear partition-of-unity ramps along X and Y guarantee **zero seam error** (`max diff = 0`).
- **Face Recovery (`recover_faces`)**: Restores low-res faces using YuNet 5-point alignment + GFPGAN v1.4 ONNX inverse-warp feather blending.

---

## Quickstart

### Prerequisites

```bash
pip install numpy opencv-python onnxruntime
```

### Download Pre-trained Models

All ONNX model weights are hosted on Google Drive:
- 📁 **[Google Drive Folder (All Models)](https://drive.google.com/drive/folders/1GQYws9PgH1GQdt8Oa66BqssjgxzsiHb-)**
  - [realesr-general-x4v3.onnx](https://drive.google.com/file/d/1adNYSujjrL6cWyAvczFdrQSSZbzL7Kn1/view?usp=sharing) (4.6 MB)
  - [face_detection_yunet_2023mar.onnx](https://drive.google.com/file/d/1ysDsFFPp7Y2PXOppDK3H5gS2ixRrurUC/view?usp=sharing) (227 KB)
  - [gfpgan_1.4.onnx](https://drive.google.com/file/d/1YaojZgy6lbgrQFUZoSLbQG03lxSz6Gs7/view?usp=sharing) (325 MB)

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
