"""
DepthMapper  —  Windows Edition
=================================
Live monocular depth estimation from webcam using MiDaS (via torch.hub).
Displays a colorized depth heatmap in real time.
Press S to export a point cloud CSV + depth histogram.

Prerequisites:
    pip install opencv-python numpy matplotlib torch torchvision

    On first run, MiDaS-small (~80 MB) downloads automatically from torch.hub.
    No GPU required — CPU inference runs at ~2-5 FPS (expected for MiDaS-small).

Usage:
    python depthmapper.py
    python depthmapper.py --model DPT_Hybrid    # larger model, more accurate
    python depthmapper.py --device 1            # different webcam index
    python depthmapper.py --no-hist             # skip histogram window
"""

import argparse
import sys
import os
import time
import csv
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

try:
    import torch
    import torchvision.transforms as T
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: torch not installed. Run: pip install torch torchvision")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL   = "MiDaS_small"   # ~80MB — fastest for CPU
MODEL_OPTIONS   = ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]
COLORMAP        = cv2.COLORMAP_INFERNO   # fire-palette depth map
WIN_NAME        = "DepthMapper"
EXPORT_DIR      = Path(".")
HIST_BINS       = 64

# Display layout constants
SIDEBAR_W       = 320
INFO_H          = 180

# Terminal colours
CYAN   = "\033[96m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
RED    = "\033[91m"; BOLD  = "\033[1m";  DIM    = "\033[2m"
RESET  = "\033[0m"

def tc(text, k): return f"{k}{text}{RESET}"

# ── MiDaS model loading ───────────────────────────────────────────────────────

def load_midas(model_name: str, device: torch.device):
    """
    Download and load MiDaS from torch.hub.
    Returns (model, transform) ready for inference.
    """
    print(f"  {tc('Loading MiDaS model:', DIM)} {tc(model_name, YELLOW)}")
    print(f"  {tc('(first run downloads ~80 MB — please wait)', DIM)}")

    try:
        model = torch.hub.load(
            "intel-isl/MiDaS",
            model_name,
            pretrained=True,
            trust_repo=True,
        )
        model.to(device)
        model.eval()

        # Load the matching transform
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )
        if model_name in ("DPT_Large", "DPT_Hybrid"):
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        print(f"  {tc('Model loaded.', GREEN)}")
        return model, transform

    except Exception as e:
        print(f"  {tc('ERROR loading MiDaS:', RED)} {e}")
        print(f"  Make sure you have internet access for the first download.")
        print(f"  Alternatively try: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        sys.exit(1)

# ── Depth estimation ──────────────────────────────────────────────────────────

def estimate_depth(frame_bgr: np.ndarray, model, transform,
                   device: torch.device) -> np.ndarray:
    """
    Run MiDaS inference on a single BGR frame.

    Pipeline:
      BGR frame → RGB → MiDaS transform (resize + normalise) →
      tensor → model → raw depth → upsample to original size →
      normalise to [0, 1]

    Returns a float32 array in [0, 1] where larger = closer.
    NOTE: MiDaS outputs INVERSE depth (disparity) — closer objects
    have HIGHER values. This is the same convention as stereo disparity maps.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h0, w0 = frame_bgr.shape[:2]

    # Apply MiDaS pre-processing
    input_batch = transform(rgb).to(device)

    with torch.no_grad():
        prediction = model(input_batch)

        # Upsample back to original frame size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h0, w0),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_raw = prediction.cpu().numpy()

    # Normalise to [0, 1]
    d_min = depth_raw.min()
    d_max = depth_raw.max()
    if d_max - d_min > 1e-6:
        depth_norm = (depth_raw - d_min) / (d_max - d_min)
    else:
        depth_norm = np.zeros_like(depth_raw)

    return depth_norm.astype(np.float32)


def depth_to_colormap(depth_norm: np.ndarray) -> np.ndarray:
    """
    Convert normalised depth [0,1] to a BGR colour image.
    Uses INFERNO colormap: bright yellow=close, dark purple=far.
    """
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, COLORMAP)

# ── Point cloud export ────────────────────────────────────────────────────────

def export_point_cloud(frame_bgr: np.ndarray, depth_norm: np.ndarray,
                       out_path: str) -> int:
    """
    Export a point cloud CSV with columns: x, y, depth, r, g, b.

    x, y are pixel coordinates (0-indexed from top-left).
    depth is the normalised MiDaS inverse depth in [0, 1].
    r, g, b are the original frame pixel colours.

    This is the same data structure as a LiDAR point cloud, just
    sampled on a regular 2D grid instead of a sparse scan pattern.

    Returns the number of points written.
    """
    h, w = depth_norm.shape

    # Build coordinate grids
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    # Flatten everything
    x_flat = xv.ravel()
    y_flat = yv.ravel()
    d_flat = depth_norm.ravel()

    # Colour channels (BGR → RGB)
    b_flat = frame_bgr[:, :, 0].ravel().astype(np.float32) / 255.0
    g_flat = frame_bgr[:, :, 1].ravel().astype(np.float32) / 255.0
    r_flat = frame_bgr[:, :, 2].ravel().astype(np.float32) / 255.0

    n_points = len(x_flat)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "depth", "r", "g", "b"])
        # Write in batches for speed
        batch = 10000
        for i in range(0, n_points, batch):
            sl = slice(i, i + batch)
            rows = zip(
                x_flat[sl].round(1),
                y_flat[sl].round(1),
                d_flat[sl].round(4),
                r_flat[sl].round(3),
                g_flat[sl].round(3),
                b_flat[sl].round(3),
            )
            writer.writerows(rows)

    return n_points

# ── Histogram ─────────────────────────────────────────────────────────────────

def show_histogram(depth_norm: np.ndarray, frame_bgr: np.ndarray,
                   timestamp: str):
    """
    Display depth distribution histogram in a matplotlib window.
    Non-blocking — runs in a daemon thread.
    """
    if not HAS_MPL:
        return

    def _plot():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                                 facecolor="#0a0a10")
        fig.canvas.manager.set_window_title(
            f"DepthMapper — Depth Analysis  {timestamp}"
        )

        # ── Depth histogram ──
        ax1 = axes[0]
        ax1.set_facecolor("#0d0d1a")
        vals = depth_norm.ravel()
        counts, edges = np.histogram(vals, bins=HIST_BINS, range=(0, 1))
        centers = (edges[:-1] + edges[1:]) / 2

        # Colour each bar by the INFERNO colormap
        bar_colors = [
            plt.cm.inferno(v) for v in centers
        ]
        ax1.bar(centers, counts, width=(edges[1]-edges[0]),
                color=bar_colors, edgecolor="none", alpha=0.9)
        ax1.set_xlabel("Normalised Depth (0=far, 1=close)",
                       color="#888", fontsize=9)
        ax1.set_ylabel("Pixel count", color="#888", fontsize=9)
        ax1.set_title("Depth Distribution", color="#00e5ff",
                      fontsize=11, fontweight="bold")
        ax1.tick_params(colors="#666", labelsize=8)
        ax1.spines[:].set_color("#1a1a2e")
        ax1.grid(True, color="#1a1a2e", linewidth=0.5)

        # Stats overlay
        stats = (f"mean={vals.mean():.3f}  "
                 f"std={vals.std():.3f}  "
                 f"min={vals.min():.3f}  "
                 f"max={vals.max():.3f}")
        ax1.text(0.5, 0.97, stats, transform=ax1.transAxes,
                 color="#aaa", fontsize=7, ha="center", va="top")

        # ── Captured frame + depth overlay ──
        ax2 = axes[1]
        ax2.set_facecolor("#0d0d1a")
        depth_color = cv2.cvtColor(
            depth_to_colormap(depth_norm), cv2.COLOR_BGR2RGB
        )
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Blend 50/50
        blended = (0.5 * rgb_frame + 0.5 * depth_color).astype(np.uint8)
        ax2.imshow(blended, aspect="auto")
        ax2.set_title("Captured Frame + Depth Overlay",
                      color="#00e5ff", fontsize=11, fontweight="bold")
        ax2.axis("off")

        fig.tight_layout(pad=1.5)
        plt.show()

    t = threading.Thread(target=_plot, daemon=True)
    t.start()

# ── OpenCV overlay drawing ────────────────────────────────────────────────────

def draw_overlay(frame_bgr: np.ndarray, depth_color: np.ndarray,
                 depth_norm: np.ndarray, fps: float,
                 model_name: str, device_name: str) -> np.ndarray:
    """
    Composite the depth heatmap and original frame side by side
    with an info sidebar.
    """
    h, w = frame_bgr.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX

    # Side-by-side: original | depth heatmap
    combined = np.hstack([frame_bgr, depth_color])
    ch, cw   = combined.shape[:2]

    # Dark sidebar
    sidebar  = np.full((ch, SIDEBAR_W, 3), (14, 14, 22), dtype=np.uint8)
    canvas   = np.hstack([combined, sidebar])
    sw_start = cw   # x start of sidebar

    def put(text, x, y, color=(200, 200, 220), scale=0.5, thick=1):
        cv2.putText(canvas, str(text), (sw_start + x, y),
                    font, scale, color, thick, cv2.LINE_AA)

    def hline(y, color=(40, 40, 60)):
        cv2.line(canvas, (sw_start, y), (sw_start + SIDEBAR_W, y), color, 1)

    # ── Sidebar header ──
    cv2.rectangle(canvas, (sw_start, 0), (sw_start + SIDEBAR_W, 44),
                  (20, 20, 35), -1)
    put("DepthMapper", 10, 22, (0, 220, 255), 0.65, 2)
    put("Day 18  BUILDCORED ORCAS", 10, 38, (70, 70, 100), 0.35)

    # ── Live stats ──
    y = 60
    hline(y - 6)
    stats = [
        ("FPS",    f"{fps:.1f}",            (80, 220, 80)),
        ("Model",  model_name,              (180, 180, 200)),
        ("Device", device_name,             (180, 180, 200)),
        ("Res",    f"{w}x{h}",             (180, 180, 200)),
        ("Close",  f"{depth_norm.max():.3f}", (0, 220, 255)),
        ("Far",    f"{depth_norm.min():.3f}", (0, 150, 200)),
        ("Mean",   f"{depth_norm.mean():.3f}",(200, 200, 100)),
    ]
    for label, val, color in stats:
        put(f"{label}:", 10, y, (90, 90, 120), 0.42)
        put(val, 110, y, color, 0.42, 1)
        y += 20
    hline(y + 2)

    # ── Depth scale bar ──
    y += 14
    put("Depth Scale", 10, y, (150, 150, 170), 0.4)
    y += 14
    bar_w  = SIDEBAR_W - 20
    bar_h  = 18
    bx     = sw_start + 10
    # Draw gradient bar using the colormap
    for i in range(bar_w):
        val    = int(i / bar_w * 255)
        color  = tuple(int(c) for c in cv2.applyColorMap(
            np.array([[val]], dtype=np.uint8), COLORMAP
        )[0, 0])
        cv2.line(canvas, (bx + i, y), (bx + i, y + bar_h), color, 1)
    cv2.rectangle(canvas, (bx, y), (bx + bar_w, y + bar_h),
                  (60, 60, 80), 1)
    put("FAR", 10, y + bar_h + 12, (100, 100, 120), 0.35)
    put("CLOSE", bar_w - 30, y + bar_h + 12, (0, 220, 255), 0.35)
    y += bar_h + 22
    hline(y)

    # ── Controls ──
    y += 12
    put("Controls:", 10, y, (150, 150, 170), 0.4)
    y += 18
    controls = [
        ("S", "save frame + CSV"),
        ("H", "show histogram"),
        ("Q / ESC", "quit"),
    ]
    for key, desc in controls:
        put(f"[{key}]", 10, y, (0, 220, 255), 0.38, 1)
        put(desc, 70, y, (140, 140, 160), 0.38)
        y += 18
    hline(y + 4)

    # ── Column labels on main image ──
    label_y = ch - 18
    cv2.putText(canvas, "Original", (10, label_y), font, 0.5,
                (180, 180, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Depth Heatmap", (w + 10, label_y), font, 0.5,
                (0, 220, 255), 1, cv2.LINE_AA)

    return canvas

# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner(model_name, device):
    print("\n" + "─" * 56)
    print("  DepthMapper  ·  Monocular Depth Estimation")
    print("  Day 18 — BUILDCORED ORCAS")
    print("─" * 56)
    print(f"  Model   : {model_name}")
    print(f"  Device  : {device}")
    print(f"  Note    : MiDaS-small runs ~2-5 FPS on CPU — expected")
    print("─" * 56)
    print("  Controls:")
    print("    S      → save current frame + export point cloud CSV")
    print("    H      → show depth histogram in a new window")
    print("    Q/ESC  → quit")
    print("─" * 56 + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DepthMapper — live monocular depth estimation"
    )
    parser.add_argument("--model",  "-m", default=DEFAULT_MODEL,
                        choices=MODEL_OPTIONS,
                        help=f"MiDaS model (default: {DEFAULT_MODEL})")
    parser.add_argument("--device", "-d", type=int, default=0,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--no-hist", action="store_true",
                        help="Disable histogram window on save")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = str(device).upper()
    if device.type == "cuda":
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"

    print_banner(args.model, device_name)

    # Load model
    model, transform = load_midas(args.model, device)

    # Open webcam
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"\n  {tc('ERROR: Cannot open webcam.', RED)}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"  {tc('Webcam open.', GREEN)} Starting depth estimation...\n")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    # State
    last_depth  = None
    last_frame  = None
    frame_count = 0
    prev_time   = time.time()
    fps         = 0.0
    save_count  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)   # mirror

        # Run depth estimation
        depth_norm  = estimate_depth(frame, model, transform, device)
        depth_color = depth_to_colormap(depth_norm)

        last_frame = frame.copy()
        last_depth = depth_norm.copy()

        # FPS
        frame_count += 1
        now = time.time()
        if now - prev_time >= 0.5:
            fps       = frame_count / (now - prev_time)
            frame_count = 0
            prev_time   = now

        # Draw
        canvas = draw_overlay(frame, depth_color, depth_norm,
                               fps, args.model, device_name)
        cv2.imshow(WIN_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        elif key == ord("s") or key == ord("S"):
            # Save frame + export point cloud CSV
            ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = str(EXPORT_DIR / f"depthmapper_frame_{ts}.png")
            csv_path = str(EXPORT_DIR / f"depthmapper_cloud_{ts}.csv")

            cv2.imwrite(img_path, depth_color)
            print(f"\n  Saved depth image: {img_path}")

            print(f"  Exporting point cloud CSV...", end="", flush=True)
            n_pts = export_point_cloud(last_frame, last_depth, csv_path)
            print(f" {n_pts:,} points → {csv_path}")
            save_count += 1

            if not args.no_hist and HAS_MPL:
                show_histogram(last_depth, last_frame, ts)

        elif key == ord("h") or key == ord("H"):
            if last_depth is not None and HAS_MPL:
                ts = datetime.now().strftime("%H:%M:%S")
                show_histogram(last_depth, last_frame, ts)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n  {tc('DepthMapper stopped.', RED)}")
    print(f"  Frames saved: {save_count}\n")


if __name__ == "__main__":
    main()