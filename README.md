# Seamless Loop Tool for MotionBuilder

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MotionBuilder](https://img.shields.io/badge/MotionBuilder-2024%2B-orange.svg)](https://www.autodesk.com/products/motionbuilder/overview)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A professional tool for automating the creation of seamless loop animations (Walk/Run cycles) directly within Autodesk MotionBuilder. It features intelligent cycle detection, hierarchy-aware blending, foot contact correction, and "In-Place" root motion processing for game engine pipelines.

**Version**: 2.1

---



##  Core Design Logic

This tool automates the tedious process of finding potential loop points and seamless blending. Here is the step-by-step logic:

### 1. Loop Analysis (The "Brain")
The tool searches for a **walk/run cycle segment** (start + end) in the current take.
- **Peak-Based Candidate Detection**: Uses hip vertical motion (`Y`) peaks by default to find stable gait landmarks.
- **Period Estimation**: Estimates the gait half-period from the hip signal and prefers full-stride cycles (2x period) for left-right symmetry.
- **Velocity-Weighted Scoring**: Ranks candidate (start, end) pairs by a weighted position+velocity cost (velocity weighted ~5x) to preserve momentum at the loop boundary.
- **Vertical Bounce Veto (`Min Vertical Bounce`)**: Rejects segments that are too flat (often slide/idle).
- **Minimum Motion Filter (`Min Average Velocity`)**: Rejects near-static segments (e.g., T-pose ranges).

### 2. Hierarchy-Aware Processing
Unlike simple tools that only process the root, this tool handles the **entire skeleton hierarchy**.
- **Resampling**: All animation curves are resampled to integer frames to eliminate sub-frame jitter.
- **Trimming**: The timeline is cropped to exactly `[Start Frame ... Loop Frame]`.
- **Safe FK Writeback**: Hips owns translation; child joints receive rotation only, avoiding Character/HumanIK translation jitter. Foot/toe translation is added only by the explicit Foot Contact Fix.

### 3. Linear Offset Compensation
To create a perfect loop, the last frame must mathematically equal the first frame.
- The difference (offset) between the last frame and first frame is calculated.
- This offset is distributed **linearly across the entire segment** (not a short window).
- **Result**: The animation loops seamlessly without "popping" or foot sliding.

### 4. Root Motion to "In-Place"
For Game Engine integration (Unreal/Unity), the tool processes the Root Bone (Hips):
- **X/Z Lock**: The root's X and Z translation are forced to `0.0` (World Origin).
- **Y Preservation**: The vertical height (Y) is preserved, keeping the natural bounce of the walk.
- **Rotation**: All rotations are preserved.
This allows the Game Engine to drive the character's capsule via code while the animation plays in place.

### 5. "Reset & Inject" Workflow
To ensure data integrity:
1. A **New Take** is created (copy of the current one).
2. The new take is **cleared** of all keys for the character.
3. The processed, seamless data is **injected** starting at Frame 0.
4. This ensures a clean, predictable asset every time.

### 6. Foot Contact Correction (FK In-Place)
For in-place FK data, contacts are detected during **Process** on the original take (pre in-place) and stored as absolute frame intervals:
- Contacts are detected when **height ≤ 2.0** and **speed ≤ 0.5** for at least **3 frames**.
- **Apply** maps stored contacts into the current take (FPS-aware when available), locks **world X/Z**, and clamps **world Y** to ground height (default `0.0`) by keying local translation curves.
- If you change foot/toe names or contact thresholds, re-run **Process** to recompute contacts.

### 7. Orientation Alignment (Hips RotY)
To ensure the character faces the correct direction in-game:
- The tool calculates a delta: `Target Angle - Current Angle at Frame 0`.
- This delta is added to **every frame** of the loop.
- Example: If your mocap starts at 45° but you want 180° (facing back), the tool adds 135° to the entire animation.
- This is a continuous offset, preventing any "pops" or wrapping issues.

### 8. FPS Resampling
If you export at a different frame rate (e.g., 60 FPS to 30 FPS):
- The tool uses **Linear Interpolation** to resample position and rotation curves.
- **Duration is Preserved**: The total time (in seconds) remains exactly the same.
- **Density Changes**: Data points are reduced (downsampling) or increased (upsampling) to match the target grid.
- This happens just before writing data to the scene, ensuring the final asset is clean.

---

##  Project Structure

```text
seamless_loop_tool/
├── launcher.py              #  Drag & Drop entry point for MotionBuilder
├── src/
│   ├── core/                #  Pure Logic (DCC-Agnostic)
│   │   ├── loop_analysis.py # Algorithms for finding loop points
│   │   └── root_motion.py   # Math for Root Motion & In-Place processing
│   ├── mobu/                #  MotionBuilder Adapter Layer
│   │   ├── adapter.py       # Wrapper around pyfbsdk API
│   │   └── loop_processor.py# Orchestrator connecting UI to Logic
│   ├── ui/                  #  User Interface
│   │   ├── export_fps.py    # FPS selection utilities
│   │   └── tool_window.py   # PySide tool window definition
│   ├── pipeline_io.py       # I/O utilities
│   └── main.py              # Application bootstrap
└── tests/                   #  Unit Tests (Pytest)
    ├── test_loop_analysis.py
    └── ...
```

---

##  Dependencies

The tool relies on the following Python libraries:

1.  **`pyfbsdk`**: Built-in MotionBuilder SDK.
2.  **`PySide2`** or **`PySide6`**: Built-in UI framework (Qt) in modern MotionBuilder.
3.  **`numpy`**: **[Required]** Used for high-performance matrix and vector math.

The optional model-training environment is separate from MotionBuilder and uses Python 3.11 plus scikit-learn 1.9.0. MotionBuilder inference still requires only NumPy.

---

## Walk / Run / Other Motion Router

The repository includes a versioned geometric descriptor, a trained Random Forest JSON model, and a pure NumPy MotionBuilder adapter. The UI classifies the current characterized Character before loop analysis. Walk and run continue to the existing gait-cycle detector; other results and classification failures require explicit confirmation before analysis continues.

### Train and evaluate externally

```bash
python3.11 -m venv .venv-training
.venv-training/bin/pip install -r training/requirements.txt
.venv-training/bin/python training/train_motion_router.py \
  --dataset-root /path/to/processed_amass_babel_router
```

Training writes `models/motion_router_v1.json`, its SHA-256 file, the feature schema, and audit reports under `reports/`. The default command uses 30 randomized parameter candidates, five source-grouped folds, and the fixed seed `42`.

### Classify a characterized MotionBuilder character

```python
from pathlib import Path

from mobu.motion_classifier import MotionClassifier

classifier = MotionClassifier(Path("models/motion_router_v1.json"))
result = classifier.classify_character(character)
print(result.label, result.confidence, result.reason)
```

The sampler reads the 18 standard `FBBodyNodeId` slots, converts Hips translation from centimeters to meters, resamples to 30 FPS, and classifies 45-frame windows with a 15-frame stride. Missing characterization, missing joints, short clips, invalid data, schema mismatch, checksum failure, or a damaged model safely return `other` with a diagnostic `reason`. Experimental models require `allow_experimental=True` when constructing `MotionClassifier`.

### MotionBuilder UI workflow

Select the active character in Character Controls, then click **Analyze Loop Point**. The tool classifies the full current Take and shows the walk, run, and other probabilities. Walk and run proceed automatically. Other motions or unavailable classifications show a confirmation dialog because the current loop detector is gait-specific. Changing the active Character, Take, or frame range invalidates the analysis and requires Analyze to run again before Process or Apply.

---

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/XIONGTAO-1/seamless-loop-tool.git
cd seamless_loop_tool
```

### 2. Configure Virtual Environment 
If you want to run the unit tests locally (outside MotionBuilder), set up a virtual environment.
Local development uses **Python 3.10+** (see `pyproject.toml` / `.python-version`):
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```


---

## 🎮 Usage Guide

### Launching the Tool
1. Open **Autodesk MotionBuilder**.
2. Locate `launcher.py` in your file explorer.
3. **Drag and Drop** `launcher.py` directly into the MotionBuilder 3D Viewport.
4. Select **"Execute"**.

### Tool Parameters

#### Basic Settings
- **Root Bone**: The name of your character's hip bone (e.g., `Hips`, `Reference`). Click "Get Selected" to auto-fill.
- **Left Foot / Right Foot**: Foot bone names used for contact detection (e.g., `LeftFoot`, `RightFoot`).
- **Left Toe / Right Toe**: Toe bone names used for contact detection (e.g., `LeftToeBase`, `RightToeBase`).
- **Up Axis**: UI selector exists, but current implementation assumes MotionBuilder's Y-up processing (not wired yet).
- **Blend Frames**: Number of frames to blend the end into the start.
  - *Note*: Current implementation uses linear offset compensation across the full segment, so this value does not change results (kept for API/UI compatibility).
- **Create New Take**: Always recommended. Keeps your original take safe.
- **Enable Foot Contact Fix**: When checked, contacts are computed during **Process** from the original take and **Apply** uses stored contacts to clamp foot/toe bones.
  - If you change foot/toe names or contact thresholds, re-run **Process** before Apply.

#### Advanced Settings (The "Secret Sauce")
- **Min Cycle Frames**: 
  - The tool won't accept loops shorter than this.
  - *Default*: `20`. (Prevents detecting a single step as a full cycle).
- **Max Cycle Frames**: 
  - Upper limit for loop search.
  - *Default*: `60`.
- **Min Vertical Bounce**:
  - **Critical for Walks**. Ensures the character is actually bobbing up and down.
  - Checks if `(MaxY - MinY) >= Threshold`.
  - *Default*: `0.0`. Increase this if the tool is picking "sliding" loops.
- **Hips RotY Target**:
  - Sets the starting Y-rotation (heading) of the Hips at frame 0.
  - The entire animation is rotated to match this starting angle.
  - *Default*: `180.0` (typically facing "back" or "forward" depending on convention).
- **Export FPS**:
  - Resamples the output animation to a specific frame rate (e.g., 30, 60).
  - Useful for game engine export requirements.

### Step-by-Step Workflow
1.  **Select your character's hips** in the scene.
2.  Click **"1. Analyze Loop Point"**. The tool detects and displays a **cycle range** (e.g., `Cycle: 120 - 298`).
3.  Click **"2. Process"**. This runs the math in memory for the entire hierarchy (Crop, Seamless Compensation, Root In-Place, RotY Align).
4.  Click **"3. Apply Changes to Scene"**. This writes the data to the new take.

---

## Author

**Name**: niexiongtao  
**Contact**: niexiongtao@gmail.com

---

*Verified on MotionBuilder 2024 (Linux/Windows).*
