# DepthMapper 📷

DepthMapper is a **live monocular depth estimator**: it uses your webcam and the MiDaS neural network (via torch.hub) to generate a real-time colorized depth heatmap. Press S to export the current frame as a point cloud CSV and show a depth histogram. It's built for the **BUILDCORED ORCAS — Day 18** challenge.

## How it works

- Uses **OpenCV** to capture live webcam frames and display the composited output.
- Passes each frame through **MiDaS** (Monocular Depth Estimation in the Wild), a neural network that estimates inverse depth (disparity) from a single RGB image — no stereo camera or structured light needed.
- Normalises the raw MiDaS output to `[0, 1]` and applies the **INFERNO colormap**: bright yellow = close, dark purple = far.
- On `S` key press, exports the current frame as a **point cloud CSV** with columns `x, y, depth, r, g, b` — the same data structure used by LiDAR sensors, just sampled on a regular 2D grid.
- Shows a **depth histogram** in matplotlib, plus a 50/50 blend of the original frame and depth heatmap.
- The sidebar displays live FPS, depth statistics (min/mean/max), and a colour scale bar.

## Requirements

- Python 3.10.x
- A working webcam
- Internet connection for first run (MiDaS-small downloads ~80 MB automatically)

## Python packages:

```bash
pip install opencv-python numpy matplotlib torch torchvision
```

## Setup

1. Install the required Python packages (see above or run:
```
pip install -r requirements.txt
```
after downloading `requirements.txt`)
2. On first run, MiDaS-small (~80 MB) downloads automatically from torch.hub — this may take a minute.

## Usage

```bash
python depthmapper.py                        # default MiDaS-small, webcam 0
python depthmapper.py --model DPT_Hybrid    # larger model, more accurate (~400 MB)
python depthmapper.py --device 1            # use a different webcam
python depthmapper.py --no-hist             # skip histogram window on save
```

| Key | Action |
|---|---|
| `S` | Save depth image + export point cloud CSV + show histogram |
| `H` | Show depth histogram without saving |
| `Q` / `ESC` | Quit |

## Output files

When you press `S`, two files are saved to the current directory:

- `depthmapper_frame_YYYYMMDD_HHMMSS.png` — the colorized depth heatmap
- `depthmapper_cloud_YYYYMMDD_HHMMSS.csv` — point cloud with columns: `x, y, depth, r, g, b`

A 640×480 frame produces **307,200 points** in the CSV (~11 MB). Load it in any point cloud viewer, numpy, or pandas.

## Common fixes

**Slow frame rate (2-5 FPS)** — this is expected on CPU with MiDaS-small. Close other heavy applications to free RAM. If you have an NVIDIA GPU, PyTorch will use CUDA automatically.

**Depth map is all one colour** — the output is being normalised per-frame, so if the model output is nearly constant (e.g. pointing at a plain wall), the heatmap will look flat. Move around or point at a scene with depth variation.

**torch.hub download fails** — check your internet connection, then try:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Webcam not found** — try `--device 1` or `--device 2`. Check Windows Device Manager to confirm your webcam is recognised.

**CSV is empty or wrong size** — ensure the frame is captured successfully (the depth heatmap should be visible) before pressing S.

## Hardware concept

DepthMapper is the **software equivalent of a ToF (Time of Flight) depth sensor**, structured light camera (like Intel RealSense), or LiDAR unit. Real depth sensors measure the time for a light pulse to bounce back from a surface — this gives absolute depth in metres. MiDaS estimates relative depth from appearance cues (perspective, texture gradients, occlusion). The output data structure — a 2D array of depth values with x, y coordinates — is identical to what a real depth sensor produces. In v2.0 you'll replace this webcam + neural network pipeline with a real ToF sensor and work with the same CSV point cloud format.

## Credits

- Depth estimation: [MiDaS](https://github.com/isl-org/MiDaS) by Intel ISL
- Model delivery: [torch.hub](https://pytorch.org/docs/stable/hub.html)
- Video capture & display: [OpenCV](https://opencv.org/)
- Visualization: [Matplotlib](https://matplotlib.org/)

Built as part of the **BUILDCORED ORCAS — Day 18: DepthMapper** challenge.
