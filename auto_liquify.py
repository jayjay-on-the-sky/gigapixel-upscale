"""
Auto Douyin Face Liquify module:
Reconstructs Photoshop Liquify vector mesh deformation combined with 
MediaPipe 468 3D landmarks and dynamic Thin Plate Spline (TPS) RBF interpolation.
"""
import cv2
import numpy as np
import onnxruntime as ort


def auto_douyin_liquify(
    img_bgr: np.ndarray,
    detector_path: str,
    mesh_path: str,
    v_line: float = 0.8,
    chin_lift: float = 0.6,
    big_eye: float = 0.6,
    nose_slim: float = 0.5,
    providers: list[str] | None = None,
) -> np.ndarray:
    """
    Highend Auto Face Liquify:
    - Auto-scales deformation proportionally to face width (no pinched/dented cheeks on distant faces).
    - Preserves skin textures, hair, neck, and background with a static TPS anchor ring.
    """
    h, w = img_bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(detector_path, "", (w, h), 0.5)
    _, faces = det.detect(img_bgr)
    if faces is None or len(faces) == 0:
        return img_bgr

    sess = ort.InferenceSession(mesh_path, providers=providers or ["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = img_bgr.copy()

    for face in faces:
        fx, fy, fw, fh = [int(round(v)) for v in face[:4]]
        cx, cy, half = fx + fw // 2, fy + fh // 2, int(max(fw, fh) * 0.75)
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(w, cx + half), min(h, cy + half)
        patch = out[y0:y1, x0:x1]
        ph, pw = patch.shape[:2]
        if ph < 32 or pw < 32:
            continue

        patch_256 = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB), (256, 256)).astype(np.float32) * (1.0 / 255.0)
        res = sess.run(None, {in_name: patch_256[None]})[0]
        pts = res.reshape(-1, 3)[:468, :2].copy()
        pts[:, 0] = pts[:, 0] * (pw / 256.0) + x0
        pts[:, 1] = pts[:, 1] * (ph / 256.0) + y0

        # Dynamic scale ratio based on standard 200px face width
        face_w = float(np.linalg.norm(pts[454] - pts[234]))
        sr = face_w / 200.0

        src_pts, dst_pts = [], []

        # 1. Jawline V-line
        jaw_pull = 5.0 * sr * v_line
        for idx in [234, 93, 132, 58, 172]:
            src_pts.append(pts[idx])
            dst_pts.append([pts[idx][0] + jaw_pull, pts[idx][1] - 0.2 * sr])
        for idx in [454, 323, 361, 288, 397]:
            src_pts.append(pts[idx])
            dst_pts.append([pts[idx][0] - jaw_pull, pts[idx][1] - 0.2 * sr])

        # 2. Chin tip lift
        chin_pt = pts[152]
        src_pts.append(chin_pt)
        dst_pts.append([chin_pt[0], chin_pt[1] - 3.5 * sr * chin_lift])

        # 3. Nose slim
        nose_delta = 2.5 * sr * nose_slim
        src_pts.extend([pts[48], pts[278]])
        dst_pts.extend([[pts[48][0] + nose_delta, pts[48][1]], [pts[278][0] - nose_delta, pts[278][1]]])

        # 4. Big Eye Bloat
        eye_scale = 0.05 * big_eye
        l_eye = (pts[33] + pts[133]) * 0.5
        r_eye = (pts[362] + pts[263]) * 0.5
        for idx in [33, 160, 158, 133, 153, 144]:
            src_pts.append(pts[idx])
            dst_pts.append(pts[idx] + (pts[idx] - l_eye) * eye_scale)
        for idx in [362, 385, 387, 263, 373, 380]:
            src_pts.append(pts[idx])
            dst_pts.append(pts[idx] + (pts[idx] - r_eye) * eye_scale)

        # 5. Lock hair, neck, forehead, and background with static anchors
        for idx in [10, 338, 297, 332, 284, 109, 67, 103, 54]:
            src_pts.append(pts[idx])
            dst_pts.append(pts[idx])

        pad_a = int(35 * sr + 15)
        for ax in np.linspace(x0 - pad_a, x1 + pad_a, 7):
            for ay in [y0 - pad_a, y1 + pad_a]:
                src_pts.append([ax, ay])
                dst_pts.append([ax, ay])
        for ay in np.linspace(y0 - pad_a, y1 + pad_a, 7):
            for ax in [x0 - pad_a, x1 + pad_a]:
                src_pts.append([ax, ay])
                dst_pts.append([ax, ay])

        src_p = np.array(src_pts, dtype=np.float32)
        dst_p = np.array(dst_pts, dtype=np.float32)

        rx0, ry0 = max(0, x0 - pad_a - 5), max(0, y0 - pad_a - 5)
        rx1, ry1 = min(w, x1 + pad_a + 5), min(h, y1 + pad_a + 5)
        roi_h, roi_w = ry1 - ry0, rx1 - rx0

        # Solve Thin Plate Spline (TPS) RBF linear system
        N = len(dst_p)
        diff = dst_p[:, None, :] - dst_p[None, :, :]
        r = np.sqrt(np.sum(diff**2, axis=-1))
        K = (r**2) * np.log(np.maximum(r, 1e-4))
        np.fill_diagonal(K, 30.0)

        P = np.column_stack([np.ones(N, dtype=np.float32), dst_p])
        L = np.block([[K, P], [P.T, np.zeros((3, 3), dtype=np.float32)]])
        Y = np.vstack([src_p, np.zeros((3, 2), dtype=np.float32)])
        weights = np.linalg.solve(L, Y)
        W_mat = weights[:N]
        A_mat = weights[N:]

        grid_y, grid_x = np.indices((roi_h, roi_w), dtype=np.float32)
        pts_eval = np.column_stack([(grid_x + rx0).ravel(), (grid_y + ry0).ravel()])

        mapped = np.zeros_like(pts_eval)
        chunk_size = 50000
        for i in range(0, len(pts_eval), chunk_size):
            chunk = pts_eval[i : i + chunk_size]
            d_chunk = chunk[:, None, :] - dst_p[None, :, :]
            r_chunk = np.sqrt(np.sum(d_chunk**2, axis=-1))
            K_chunk = (r_chunk**2) * np.log(np.maximum(r_chunk, 1e-4))
            P_chunk = np.column_stack([np.ones(len(chunk), dtype=np.float32), chunk])
            mapped[i : i + chunk_size] = K_chunk @ W_mat + P_chunk @ A_mat

        map_x = mapped[:, 0].reshape(roi_h, roi_w).astype(np.float32)
        map_y = mapped[:, 1].reshape(roi_h, roi_w).astype(np.float32)

        out[ry0:ry1, rx0:rx1] = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

    return out
