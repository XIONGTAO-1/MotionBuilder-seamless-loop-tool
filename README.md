# Seamless Loop Tool for MotionBuilder

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![MotionBuilder](https://img.shields.io/badge/MotionBuilder-2022%2B-orange.svg)](https://www.autodesk.com/products/motionbuilder/overview)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A professional tool for automating the creation of seamless loop animations (Walk/Run cycles) directly within Autodesk MotionBuilder. It features intelligent cycle detection, hierarchy-aware blending, and "In-Place" root motion processing for game engine pipelines.

---



##  Core Design Logic

This tool automates the tedious process of finding potential loop points and seamless blending. Here is the step-by-step logic:

### 1. Loop Analysis (The "Brain")
The tool analyzes a chosen range of the animation to find the best point to loop back to the start.
- **Velocity-Weighted Comparison**: Instead of just comparing position, the algorithm weights **velocity** similarity 5x higher than position. This ensures the character's momentum is preserved at the loop point, preventing "jerky" transitions.
- **Vertical Bounce Veto (`Range Y`)**: For walk/run cycles, a flat trajectory often indicates a slide or idle. The tool calculates the vertical range of motion. If `Range Y < Min Vertical Bounce`, the candidate loop is immediately vetoed.
- **Local Minimum Search**: It scans the difference curve to find the global minimum, identifying the frame that most closely matches the start frame.

### 2. Hierarchy-Aware Processing
Unlike simple tools that only process the root, this tool handles the **entire skeleton hierarchy**.
- **Resampling**: All animation curves are resampled to integer frames to eliminate sub-frame jitter.
- **Trimming**: The timeline is cropped to exactly `[Start Frame ... Loop Frame]`.

### 3. Linear Offset Compensation
To create a perfect loop, the last frame must mathematically equal the first frame.
- The difference (offset) between the last frame and first frame is calculated.
- This offset is distributed **linearly** back through a "Blend Window" (e.g., 5 frames).
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

### 6. Orientation Alignment (Hips RotY)
To ensure the character faces the correct direction in-game:
- The tool calculates a delta: `Target Angle - Current Angle at Frame 0`.
- This delta is added to **every frame** of the loop.
- Example: If your mocap starts at 45° but you want 180° (facing back), the tool adds 135° to the entire animation.
- This is a continuous offset, preventing any "pops" or wrapping issues.

### 7. FPS Resampling
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

---

##  Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/XIONGTAO-1/seamless-loop-tool.git
cd seamless_loop_tool
```

### 2. Configure Virtual Environment 
If you want to run the unit tests locally (outside MotionBuilder), set up a virtual environment:
```bash
# Using uv (recommended) or pip
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt # (if provided) or just: pip install numpy pytest PySide2
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
- **Blend Frames**: Number of frames to blend the end into the start.
  - *Recommendation*: `5-10` frames for typical walk cycles.
- **Create New Take**: Always recommended. Keeps your original take safe.

#### Advanced Settings (The "Secret Sauce")
- **Min Cycle Frames**: 
  - The tool won't accept loops shorter than this.
  - *Default*: `20`. (Prevents detecting a single step as a full cycle).
- **Max Cycle Frames**: 
  - Upper limit for loop search.
  - *Default*: `60`.
- **Min Avg Velocity**: 
  - Ignores static/idle segments.
  - *Default*: `5.0`.
- **Min Vertical Bounce**:
  - **Critical for Walks**. Ensures the character is actually bobbing up and down.
  - Checks if `(MaxY - MinY) >= Threshold`.
  - *Default*: `7.0`. Increase this if the tool is picking "sliding" loops.
- **Hips RotY Target**:
  - Sets the starting Y-rotation (heading) of the Hips at frame 0.
  - The entire animation is rotated to match this starting angle.
  - *Default*: `180.0` (typically facing "back" or "forward" depending on convention).
- **Export FPS**:
  - Resamples the output animation to a specific frame rate (e.g., 30, 60).
  - Useful for game engine export requirements.

### Step-by-Step Workflow
1.  **Select your character's hips** in the scene.
2.  Click **"1. Analyze Loop Point"**. The tool will calculate and display the best loop frame (e.g., "Best Loop Frame: 48").
3.  Click **"2. Process"**. This runs the math in memory (Trim, Blend, In-Place).
4.  Click **"3. Apply Changes"**. This writes the data to the new take.

---

## Author

**Name**: niexiongtao  
**Contact**: niexiongtao@gmail.com

---

*Verified on MotionBuilder 2024 (Linux/Windows).*
