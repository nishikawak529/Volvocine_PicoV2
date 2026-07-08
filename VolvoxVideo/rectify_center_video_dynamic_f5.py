import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points in order: top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left
    return ordered


def quad_size(quad: np.ndarray) -> tuple[float, float]:
    q = order_points(quad)
    width_top = np.linalg.norm(q[1] - q[0])
    width_bottom = np.linalg.norm(q[2] - q[3])
    height_left = np.linalg.norm(q[3] - q[0])
    height_right = np.linalg.norm(q[2] - q[1])
    return max(width_top, width_bottom), max(height_left, height_right)


def detect_dark_rectangle(frame: np.ndarray) -> np.ndarray | None:
    """Detect the central dark rectangular region as a quadrilateral.

    This detector is tuned for a dark rectangular video panel on a light slide/background.
    It intentionally chooses a large, vertical, central dark contour and rejects text/edges.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # The target panel is much darker than the background.
    # 150 works well for this video; expose it as a constant if needed.
    _, mask = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

    # Remove small text/noise and close holes caused by bright objects inside the panel.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[float, np.ndarray]] = []
    frame_area = h * w
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.08 * frame_area:
            continue

        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / max(bh, 1)
        cx = x + bw / 2
        cy = y + bh / 2

        # Target is a large vertical rectangle near the center.
        if not (0.35 <= aspect <= 0.75):
            continue
        if not (0.25 * w <= cx <= 0.75 * w):
            continue
        if not (0.25 * h <= cy <= 0.85 * h):
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            # Fallback: use the minimum-area rectangle if contour approximation is not 4 points.
            quad = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)

        quad = order_points(quad)
        qw, qh = quad_size(quad)
        score = area - 0.05 * ((cx - w / 2) ** 2 + (cy - h / 2) ** 2)
        score += 0.1 * qw * qh
        candidates.append((score, quad))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return order_points(candidates[0][1])


def detect_all_quads(input_path: Path) -> tuple[list[np.ndarray | None], float, int, int]:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")

    # Some smartphone MOV files have rotation metadata. OpenCV often auto-rotates them.
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    quads: list[np.ndarray | None] = []
    width = height = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width = frame.shape[:2]
        quads.append(detect_dark_rectangle(frame))

    cap.release()
    return quads, fps, width, height


def fill_missing_quads(quads: list[np.ndarray | None]) -> list[np.ndarray]:
    valid = [(i, q) for i, q in enumerate(quads) if q is not None]
    if not valid:
        raise RuntimeError("Could not detect the target rectangle in any frame.")

    n = len(quads)
    out: list[np.ndarray | None] = [None] * n

    # Fill valid positions.
    for i, q in valid:
        out[i] = q.astype(np.float32)

    valid_indices = [i for i, _ in valid]

    # Fill leading missing frames with first valid quad.
    first = valid_indices[0]
    for i in range(0, first):
        out[i] = out[first].copy()

    # Fill trailing missing frames with last valid quad.
    last = valid_indices[-1]
    for i in range(last + 1, n):
        out[i] = out[last].copy()

    # Linear interpolation for gaps between valid frames.
    for a, b in zip(valid_indices[:-1], valid_indices[1:]):
        qa = out[a]
        qb = out[b]
        assert qa is not None and qb is not None
        for i in range(a + 1, b):
            t = (i - a) / (b - a)
            out[i] = (1 - t) * qa + t * qb

    return [q for q in out if q is not None]


def smooth_quads(
    quads: list[np.ndarray],
    method: str = "gaussian",
    alpha: float = 0.35,
    window_size: int = 25,
    sigma: float = 6.0,
) -> list[np.ndarray]:
    """Smooth corner positions to reduce hand jitter and camera shake.

    Supports 'gaussian' (symmetric non-causal zero-phase filter) and 'ema' (causal exponential moving average).
    """
    if method == "ema":
        smoothed: list[np.ndarray] = []
        prev = quads[0].astype(np.float32)
        smoothed.append(prev.copy())
        for q in quads[1:]:
            q = q.astype(np.float32)
            prev = alpha * q + (1.0 - alpha) * prev
            smoothed.append(prev.copy())
        return smoothed

    # Gaussian smoothing (default)
    n = len(quads)
    coords = np.array(quads, dtype=np.float32)  # Shape: (n, 4, 2)
    smoothed = np.zeros_like(coords)

    half_w = window_size // 2
    # Ensure window_size is odd
    window_size = 2 * half_w + 1

    x = np.arange(-half_w, half_w + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    for c_idx in range(4):
        for coord_idx in range(2):
            signal = coords[:, c_idx, coord_idx]
            padded = np.pad(signal, half_w, mode="edge")
            smoothed_signal = np.convolve(padded, kernel, mode="valid")
            smoothed[:, c_idx, coord_idx] = smoothed_signal

    return [q for q in smoothed]


def output_size_from_quads(quads: list[np.ndarray]) -> tuple[int, int]:
    sizes = np.array([quad_size(q) for q in quads], dtype=np.float32)
    out_w, out_h = np.median(sizes, axis=0)
    out_w = int(round(out_w))
    out_h = int(round(out_h))

    # H.264/yuv420p requires even dimensions.
    out_w += out_w % 2
    out_h += out_h % 2
    return out_w, out_h


def write_rectified_video(
    input_path: Path,
    output_path: Path,
    quads: list[np.ndarray],
    fps: float,
    out_w: int,
    out_h: int,
    debug_path: Path | None = None,
    stabilize: bool = True,
) -> Path:
    tmp_path = output_path.with_name(output_path.stem + "_noaudio.mp4")

    # If stabilization is requested, run background Lucas-Kanade feature tracking first
    accumulated_transforms = []
    avg_quad = None
    if stabilize:
        print("[INFO] Running background tracking for video stabilization...")
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video for stabilization: {input_path}")
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        except Exception:
            pass

        ret, frame0 = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("Cannot read first frame for stabilization.")

        num_frames = len(quads)
        prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        
        # Cumulative transform is initially Identity
        A = np.eye(3, dtype=np.float32)
        accumulated_transforms = [A.copy()]
        all_q_ref = []
        
        # Add first quad mapped with Identity
        q_hom = np.hstack([quads[0], np.ones((4, 1), dtype=np.float32)])
        q_ref = (A @ q_hom.T).T[:, :2]
        all_q_ref.append(q_ref)
        
        p0 = None
        frame_idx = 1
        while frame_idx < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Mask out the well region in prev frame
            h, w = curr_gray.shape
            mask = np.full((h, w), 255, dtype=np.uint8)
            q = quads[frame_idx - 1].astype(np.int32)
            cv2.fillPoly(mask, [q], 0)
            
            # Dilate the well mask to exclude boundaries
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
            mask = cv2.erode(mask, kernel, iterations=1)
            
            if p0 is None or len(p0) < 40:
                p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=150, qualityLevel=0.01, minDistance=10, mask=mask)
                
            T = np.eye(3, dtype=np.float32)
            if p0 is not None and len(p0) >= 4:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
                if p1 is not None:
                    good_p0 = p0[st == 1]
                    good_p1 = p1[st == 1]
                    if len(good_p0) >= 4:
                        M, inliers = cv2.estimateAffinePartial2D(good_p1, good_p0, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                        if M is not None:
                            T[:2, :] = M
                            p0 = good_p1[inliers.ravel() == 1].reshape(-1, 1, 2)
                        else:
                            p0 = good_p1.reshape(-1, 1, 2)
                    else:
                        p0 = None
                else:
                    p0 = None
            else:
                p0 = None
                
            A = A @ T
            accumulated_transforms.append(A.copy())
            
            # Map current quad to reference frame coordinates
            q_hom = np.hstack([quads[frame_idx], np.ones((4, 1), dtype=np.float32)])
            q_ref = (A @ q_hom.T).T[:, :2]
            all_q_ref.append(q_ref)
            
            prev_gray = curr_gray
            frame_idx += 1
            
        cap.release()
        
        while len(accumulated_transforms) < num_frames:
            accumulated_transforms.append(accumulated_transforms[-1].copy())
        while len(all_q_ref) < num_frames:
            all_q_ref.append(all_q_ref[-1])
            
        avg_quad = np.mean(all_q_ref, axis=0)
        print("[INFO] Background stabilization tracking complete.")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {input_path}")
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    except Exception:
        pass

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open temporary output video: {tmp_path}")

    debug_writer = None
    if debug_path is not None:
        ret, frame0 = cap.read()
        if ret:
            h, w = frame0.shape[:2]
            debug_writer = cv2.VideoWriter(str(debug_path), fourcc, fps, (w, h))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    dst = np.float32([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ])

    if stabilize:
        H_rect = cv2.getPerspectiveTransform(avg_quad, dst)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_index >= len(quads):
            break

        if stabilize:
            H_i = H_rect @ accumulated_transforms[frame_index]
            rectified = cv2.warpPerspective(frame, H_i, (out_w, out_h))
        else:
            src = order_points(quads[frame_index])
            M = cv2.getPerspectiveTransform(src, dst)
            rectified = cv2.warpPerspective(frame, M, (out_w, out_h))
            
        writer.write(rectified)

        if debug_writer is not None:
            debug = frame.copy()
            if stabilize:
                A_inv = np.linalg.inv(accumulated_transforms[frame_index])
                q_hom = np.hstack([avg_quad, np.ones((4, 1), dtype=np.float32)])
                src = (A_inv @ q_hom.T).T[:, :2]
            else:
                src = order_points(quads[frame_index])
                
            pts = np.round(src).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(debug, [pts], isClosed=True, color=(0, 0, 255), thickness=5)
            for k, p in enumerate(src):
                cv2.circle(debug, tuple(np.round(p).astype(int)), 8, (0, 255, 0), -1)
                cv2.putText(debug, str(k), tuple(np.round(p).astype(int) + np.array([8, -8])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            debug_writer.write(debug)

        frame_index += 1

    cap.release()
    writer.release()
    if debug_writer is not None:
        debug_writer.release()

    # Re-encode to H.264 and copy audio from the original video if present.
    cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_path),
        "-i", str(input_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if tmp_path.exists():
            tmp_path.unlink()
        return output_path
    except Exception:
        print(f"ffmpeg mux/re-encode failed. No-audio video remains at: {tmp_path}")
        return tmp_path


def resolve_default_paths() -> tuple[Path, Path]:
    """Return default input/output paths for F5 execution in VS Code.

    The script first looks for IMG_0859.MOV in the same folder as this script.
    If it does not exist, it uses the first .MOV/.mp4 file found in that folder.
    """
    script_dir = Path(__file__).resolve().parent

    preferred = script_dir / "IMG_0859.MOV"
    if preferred.exists():
        input_path = preferred
    else:
        candidates = []
        for pattern in ("*.MOV", "*.mov", "*.MP4", "*.mp4"):
            candidates.extend(sorted(script_dir.glob(pattern)))
        if not candidates:
            raise FileNotFoundError(
                "No input video found. Put IMG_0859.MOV in the same folder as this script, "
                "or pass input/output paths as command-line arguments."
            )
        input_path = candidates[0]

    output_path = input_path.with_name(input_path.stem + "_rectified_dynamic.mp4")
    return input_path, output_path


def main() -> None:
    default_input, default_output = resolve_default_paths()

    parser = argparse.ArgumentParser(
        description="Dynamically detect a central quadrilateral video panel and rectify it frame by frame."
    )
    parser.add_argument("input", nargs="?", type=Path, default=default_input,
                        help=f"Input video path. Default: {default_input}")
    parser.add_argument("output", nargs="?", type=Path, default=default_output,
                        help=f"Output video path. Default: {default_output}")
    parser.add_argument("--smoothing", choices=["gaussian", "ema"], default="gaussian",
                        help="Smoothing method. 'gaussian' is zero-phase/symmetric; 'ema' is causal. Default: gaussian")
    parser.add_argument("--alpha", type=float, default=0.35,
                        help="EMA smoothing factor. Only used if --smoothing is 'ema'. Default: 0.35")
    parser.add_argument("--window-size", type=int, default=25,
                        help="Gaussian window size in frames. Default: 25")
    parser.add_argument("--sigma", type=float, default=6.0,
                        help="Gaussian sigma in frames. Default: 6.0")
    parser.add_argument("--stabilize", action="store_true", default=True,
                        help="Stabilize video relative to background slide (recommended). Default: True")
    parser.add_argument("--no-stabilize", dest="stabilize", action="store_false",
                        help="Disable background stabilization.")
    parser.add_argument("--debug", nargs="?", const="auto", default=None,
                        help="Optional debug video. Use '--debug' alone to save '<input>_debug_detected_quad.mp4'.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if args.debug == "auto":
        debug_path = input_path.with_name(input_path.stem + "_debug_detected_quad.mp4")
    elif args.debug is None:
        debug_path = None
    else:
        debug_path = Path(args.debug).resolve()

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    if debug_path is not None:
        print(f"Debug : {debug_path}")

    raw_quads, fps, frame_w, frame_h = detect_all_quads(input_path)
    detected = sum(q is not None for q in raw_quads)
    print(f"Frames: {len(raw_quads)}, detected: {detected}, fps: {fps:.3f}, frame: {frame_w}x{frame_h}")

    quads = fill_missing_quads(raw_quads)
    quads = smooth_quads(quads, method=args.smoothing, alpha=args.alpha, window_size=args.window_size, sigma=args.sigma)
    out_w, out_h = output_size_from_quads(quads)
    print(f"Output size: {out_w}x{out_h}")

    result_path = write_rectified_video(
        input_path, output_path, quads, fps, out_w, out_h, debug_path=debug_path, stabilize=args.stabilize
    )
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
