# gigapixel-upscale

Minimal, fast Python reconstruction of **Topaz Gigapixel AI's tiled upscale pipeline** (`TBlockProc` + `StandardCCImage` + `ResizeEnhancement`) & **Photoshop Face-Aware Liquify** (468-landmark Thin-Plate Spline deformation), built with `ponytail` principles (~100 lines each, zero bloat, pure OpenCV + NumPy + ONNXRuntime).

---

## 1. Tiled Super-Resolution Upscale (`upscale.py`)

- **Power-of-2 Model Snapping (`TBlockProc::TBlockProc` `0x4db94`)**: Snaps scale factors to $2^{\text{round}(\log_2(\text{scale}))}$ with $128 \times 128$ tiles, $4\text{px}$ discard border, and $32\text{px}$ overlap.
- **Batch Streaming & Single-Pass Pad (`StandardCCImage::process` `0x473fc`)**: Pads the whole image once with `cv2.BORDER_REFLECT_101` and streams bounded batches directly onto the output canvas (fixed ~50 MB intermediate RAM even for 4K/8K images).
- **Separable 1D Feather Blending (`TBlockProc::mergeBlock` `0x4edd4`)**: Linear partition-of-unity ramps along X and Y guarantee **zero seam error** (`max diff = 0`).
- **Face Recovery (`recover_faces`)**: Restores low-res faces using YuNet 5-point alignment + GFPGAN v1.4 ONNX inverse-warp feather blending.

```python
import cv2
from upscale import upscale, make_onnx_fn, recover_faces

img = cv2.imread("input.jpg")

# 1. Tiled AI Upscale
onnx_fn = make_onnx_fn("realesr-general-x4v3.onnx")
up = upscale(img, target_scale=2.0, upscale_fn=onnx_fn, batch_size=8)

# 2. Optional Face Recovery (strength 0.2-0.8)
out, faces = recover_faces(up, "gfpgan_1.4.onnx", detector_path="face_detection_yunet_2023mar.onnx", strength=0.3)
cv2.imwrite("output.jpg", out)
```

---

## 2. Auto Douyin Face Liquify (`auto_liquify.py`)

Reconstructs Photoshop's Vector Distortion Mesh + Nozzle profile with **468 3D MediaPipe landmarks** and **Thin Plate Spline (TPS) RBF** deformation.

- **Dynamic Scale Normalization**: Automatically scales jawline/chin/eye offsets proportional to face pixel width (`face_w / 200px`), preventing pinching on distant/group faces.
- **Background & Hair Lock**: A ring of static TPS anchor points locks the background, shoulders, neck, and hair completely in place (`0px` warp).
- **Pure Texture Preservation**: Lanczos4 inverse mapping keeps original skin pores and sharpness intact without plastic AI blurring.

```python
import cv2
from auto_liquify import auto_douyin_liquify

img = cv2.imread("portrait.jpg")
beautified = auto_douyin_liquify(
    img,
    detector_path="face_detection_yunet_2023mar.onnx",
    mesh_path="face_landmarks_468.onnx",
    v_line=0.8,     # Jawline slimming (0.0 to 1.0)
    chin_lift=0.6,  # Chin point sharpening
    big_eye=0.6,    # Eye enlargement
    nose_slim=0.5,  # Nose contour reduction
)
cv2.imwrite("douyin_portrait.jpg", beautified)
```

---

## Run Tests & Benchmark

```bash
python3 test_upscale.py
```
