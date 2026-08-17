# Seamless Loop Tool for MotionBuilder

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![MotionBuilder](https://img.shields.io/badge/MotionBuilder-2024%2B-orange.svg)](https://www.autodesk.com/products/motionbuilder/overview)

**Version:** 2.1.0

## Demo Video

[![Watch the Seamless Loop Tool demonstration on YouTube](https://img.youtube.com/vi/U8uT9hT65pY/maxresdefault.jpg)](https://youtu.be/U8uT9hT65pY)



Seamless Loop Tool creates in-place walk and run cycles directly in Autodesk MotionBuilder. It combines motion-type routing, gait-cycle detection, hierarchy-aware loop processing, root-motion removal, orientation alignment, FPS resampling, namespace-safe bone lookup, and optional foot-contact correction in one non-destructive workflow.

The MotionBuilder runtime uses `pyfbsdk`, PySide, and NumPy. The bundled motion router performs inference with pure NumPy; scikit-learn is required only when training a replacement model outside MotionBuilder.

## Requirements

### MotionBuilder runtime

- Autodesk MotionBuilder 2024 or newer.
- MotionBuilder's built-in `pyfbsdk`.
- MotionBuilder's built-in `PySide2` or `PySide6`.
- A NumPy build compatible with MotionBuilder's embedded Python.

The plugin has been verified with MotionBuilder 2024 on Windows and Linux.

### Character prerequisite for motion classification

The walk/run/other classifier operates on MotionBuilder's characterized HumanIK body nodes. Before clicking **Analyze Loop Point**:

1. Create a valid **Character Definition** for the skeleton.
2. Characterize it successfully.
3. Make that Character current in **Character Controls**.

If no current characterized Character is available, characterization state cannot be read, or a required characterized joint is missing, classification returns a safe unavailable/`other` result. The UI explains the problem and requires explicit confirmation before the gait-specific loop analyzer can continue.

Characterization is required for motion classification. The Root, Foot, and Toe text fields are still used by the loop-processing and foot-contact stages and must identify the intended scene models.

## Features and Processing Model

### Motion routing

The bundled production Random Forest routes the current Take to `walk`, `run`, or `other` before loop analysis:

- Samples 18 standard `FBBodyNodeId` joints from the active characterized Character.
- Reads world-space joint rotations and Hips translation.
- Converts Hips translation from centimetres to metres.
- Resamples input to 30 FPS.
- Classifies 45-frame windows with a 15-frame stride.
- Uses a versioned 175-value geometric descriptor and a JSON forest evaluated with pure NumPy.
- Automatically continues for confident walk/run results.
- Requires **Analyze Anyway** confirmation for `other` results or unavailable classification.

Changing the active Character, Take, or frame range invalidates the analysis. Run **Analyze Loop Point** again before Process or Apply.

### Gait-cycle detection

For walk and run motion, the analyzer:

- Finds candidates from Hips vertical motion peaks/valleys.
- Estimates the gait period and prefers a full left-right stride.
- Scores boundary pose and velocity continuity, with stronger weighting on velocity.
- Uses the configured cycle limits to guide peak spacing, period estimation, and candidate scoring; full-stride candidates may extend to `2.5 × Max Cycle Frames`.
- Filters near-static segments and segments below the configured vertical bounce during peak-based candidate scoring.
- Falls back to the first half of the Take when fewer than two peak/valley candidates are available; that fallback does not apply the candidate length, velocity, or bounce filters.

### Hierarchy-aware seamless processing

The tool samples the selected root hierarchy on integer frames and crops it to the detected `[cycle start ... cycle end]` range. Linear end-to-start offset compensation is applied across the complete segment so the final pose converges to the first pose without a short boundary crossfade.

During Apply:

- The root receives translation and rotation.
- Child joints receive rotation only, avoiding Character/HumanIK translation jitter.
- Foot and toe translation is written only by the explicit Foot Contact Fix.
- Output keys begin at frame 0 and the Take time span is updated to the processed range.

### In-place root motion and orientation

For the root trajectory:

- X and Z translation are set to `0.0`.
- Y translation is preserved to retain vertical body motion.
- Rotation is preserved.
- A uniform Y-rotation offset can align frame 0 to the configured **Hips RotY Target**.

The current processing path assumes MotionBuilder's Y-up coordinate system. The **Up Axis** selector is present in the UI, but Z-up selection is not yet connected to processing.

### Foot-contact correction

When **Enable Foot Contact Fix** is checked, Process detects stance intervals from the original moving Take before in-place conversion. Current internal defaults are:

- Height at or below `2.0` scene units relative to ground height `0.0`.
- Contact metric at or below `0.5`.
- At least 3 consecutive frames.

When root motion is available, the contact metric uses relative horizontal acceleration; otherwise it falls back to world-space speed. Apply maps the stored absolute intervals into the output frame range, including FPS conversion, then locks world X/Z and clamps world Y to the ground by keying local foot/toe translations.

Changing foot/toe names or contact parameters requires Process to be run again so contact intervals can be recomputed.

### FPS resampling

Apply can resample output to 30, 60, 90, or 120 FPS. Linear interpolation changes sample density while preserving duration in seconds. The MotionBuilder transport FPS and output Take span are updated accordingly.

## Project Structure

The tree below contains only source, model, report, test, and project files that are not excluded by `.gitignore`.

```text
seamless_loop_tool/
├── .gitignore
├── .python-version
├── README.md
├── data_preprocessing/
│   ├── amass_babel_router/
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── intervals.py
│   │   ├── labels.py
│   │   └── pipeline.py
│   └── prepare_amass_babel_router.py
├── launcher.py
├── pyproject.toml
├── uv.lock
├── models/
│   ├── feature_schema_v1.json
│   ├── motion_router_v1.json
│   └── motion_router_v1.sha256
├── reports/
│   ├── confusion_matrix.csv
│   ├── metrics.json
│   ├── selected_training_windows.csv
│   └── validation_predictions.csv
├── src/
│   ├── core/
│   │   ├── loop_analysis.py
│   │   ├── motion_classifier.py
│   │   ├── motion_features.py
│   │   └── root_motion.py
│   ├── mobu/
│   │   ├── adapter.py
│   │   ├── loop_processor.py
│   │   └── motion_classifier.py
│   ├── ui/
│   │   ├── bone_namespace.py
│   │   ├── export_fps.py
│   │   ├── motion_routing.py
│   │   └── tool_window.py
│   ├── main.py
│   └── pipeline_io.py
├── training/
│   ├── dataset.py
│   ├── model_export.py
│   ├── requirements.txt
│   ├── thresholds.py
│   └── train_motion_router.py
└── tests/
    ├── test_amass_babel_pipeline.py
    ├── test_bone_namespace.py
    ├── test_export_fps.py
    ├── test_loop_analysis.py
    ├── test_loop_processor.py
    ├── test_mobu_adapter.py
    ├── test_mobu_motion_classifier.py
    ├── test_motion_classifier.py
    ├── test_motion_features.py
    ├── test_root_motion.py
    ├── test_training_dataset.py
    ├── test_training_model.py
    ├── test_ui_motion_routing.py
    └── test_ui_tool_window.py
```

## Installation

### 1. Get the project

```bash
git clone https://github.com/XIONGTAO-1/seamless-loop-tool.git
cd seamless-loop-tool
```

### 2. Check MotionBuilder's Python compatibility

Run this in MotionBuilder's Python Editor on the target computer:

```python
import platform
import sys

print(sys.version)
print(platform.system())
print(platform.machine())
```

Record the Python major/minor version, operating system, and CPU architecture. NumPy must match all three.

### 3. Create a plugin-local runtime environment

The launcher can add this project's `.venv` `site-packages` directory to MotionBuilder's module search path. MotionBuilder does **not** activate or switch to the virtual environment; it continues to use its embedded interpreter.

Create `.venv` on the same type of machine that will run MotionBuilder. For example, if MotionBuilder embeds 64-bit CPython 3.10 on Windows:

```powershell
uv venv --python 3.10 .venv
uv pip install --python .venv\Scripts\python.exe "numpy>=1.21"
```

Without uv:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python.exe -m pip install "numpy>=1.21"
```

On Linux, use `.venv/bin/python` in place of `.venv\Scripts\python.exe`.

Do not copy a virtual environment between Windows, Linux, or macOS; between x86-64 and ARM64; or between different Python major/minor versions. Recreate it on the target platform. A macOS ARM64 Python 3.10 environment, for example, cannot provide NumPy to Windows x64 MotionBuilder.

### 4. Launch in MotionBuilder

1. Open MotionBuilder.
2. Drag `launcher.py` into the 3D Viewer.
3. Choose **Execute**.
4. Complete the Character Definition/Characterization prerequisites before using Analyze.

## Namespace and Multiple-Character Scenes

MotionBuilder scenes often contain names such as `CharacterA:Hips` and `CharacterB:Hips`. The UI provides a **Namespace** field and a **Get from Selected** button:

1. Select any bone from the intended character.
2. Click **Get from Selected** beside Namespace.
3. The namespace is extracted from the selected bone's complete name.
4. Root, Left/Right Foot, and Left/Right Toe fields are updated with that namespace.

Entering `CharacterA` or `CharacterA:` produces the normalized prefix `CharacterA:`. Applying a new namespace replaces any existing prefix instead of stacking prefixes.

Unqualified names are accepted only when they resolve to exactly one scene model. If two namespaces both contain `LeftLeg`, the adapter raises an ambiguity error and lists the matches. Use a complete MotionBuilder `LongName`, such as `CharacterA:LeftLeg`; the plugin must not silently choose the first match.

The classifier itself reads standardized body nodes from the active characterized Character, while hierarchy processing resolves the explicit Root/Foot/Toe names. Configure both the active Character and the namespace fields correctly in multi-character scenes.

## UI Reference

### Bone and output controls

| Control | Default | Current behavior |
| --- | ---: | --- |
| Root Bone | `Hips` | Root used for loop analysis and hierarchy traversal. |
| Namespace | empty | Qualifies Root, Foot, and Toe fields; can be captured from a selected bone. |
| Left Foot / Right Foot | `LeftFoot` / `RightFoot` | Bones used for stance detection and correction. |
| Left Toe / Right Toe | `LeftToeBase` / `RightToeBase` | Optional toe samples used with the corresponding foot. |
| Blend Frames | `5` | Retained for API/UI compatibility; current linear full-segment compensation does not use a short blend window. |
| Up Axis | Y-Up | Selector is visible; current processing remains Y-up. |
| Create new Take | enabled | Preserves the original and writes processed data to a clean Take. |
| Enable Foot Contact Fix | disabled | Detects contacts during Process and applies correction during Apply when enabled. |

### Advanced controls

| Control | Default | Current behavior |
| --- | ---: | --- |
| Min Cycle Frames | `20` | Minimum duration used by the peak-based candidate search. |
| Max Cycle Frames | `60` | Period-search limit; full-stride candidate pairs may extend to 2.5 times this value. |
| Min Vertical Bounce | `0.0` | Minimum Hips vertical range used during scored candidate evaluation. |
| Hips RotY Target | `180.0` | Desired root Y rotation at output frame 0. |
| Export FPS | `30` | Output choices: 30, 60, 90, or 120 FPS. |

The **Motion Type** panel shows the routed label, confidence, walk/run/other probabilities, analyzed window count, and any diagnostic message.

## MotionBuilder Workflow

1. Load the animated skeleton and confirm the current Take and frame range.
2. Define and characterize the skeleton, then select its Character in Character Controls.
3. In multi-character scenes, capture or enter the intended namespace.
4. Set Root, Foot, and Toe bone names; use each **Get Selected** button when appropriate.
5. Configure cycle limits, vertical bounce, target orientation, output FPS, Take preservation, and Foot Contact Fix.
6. Click **1. Analyze Loop Point**.
   - The full current Take is classified first.
   - Confident walk/run motion continues automatically.
   - `other` or unavailable classification requires confirmation.
   - The gait analyzer then reports the selected cycle range.
7. Click **2. Process (Trim + Blend + In-Place)**.
   - The hierarchy is sampled and processed in memory.
   - Foot contacts are detected from the original moving Take when enabled.
8. Click **3. Apply Changes to Scene**.
   - A clean Take is created when preservation is enabled.
   - Processed keys, optional foot correction, output FPS, and Take span are applied.
9. Inspect the loop boundary, feet, knee direction, root height, and orientation before export.

## Local Development

Local development and tests use Python 3.10 or newer. The repository pins `3.10` in `.python-version`.

With uv:

```bash
uv sync
uv run pytest -q
```

Without uv:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -e .
python -m pytest -q
```

`pyfbsdk` and MotionBuilder's PySide package are not installed by this local setup. Unit tests use adapters and stubs so core logic can run outside MotionBuilder.

## Training the Motion Router

Training is optional and separate from MotionBuilder inference. The checked-in training requirements use Python 3.11, NumPy 2.x, and scikit-learn 1.9.0.

### Download and prepare AMASS/BABEL data

This repository does not distribute AMASS, BABEL, or generated training windows. Obtain access to and download the required data yourself:

1. Open the repository's `data_preprocessing/` directory.
2. Download the AMASS HDM05 motion archives from [AMASS](https://amass.is.tue.mpg.de/), then extract them into the current `data_preprocessing/` directory as `HDM05/`. The preprocessing pipeline expects the BABEL-referenced paths below it, such as `HDM05/MPI_HDM05/...`.
3. Download BABEL v1.0 annotations from [BABEL](https://babel.is.tue.mpg.de/), then extract them into the same `data_preprocessing/` directory as `babel_v1.0_release/`. This directory must contain `train.json` and `val.json`.
4. Follow the AMASS and BABEL licenses. Do not commit, upload, or redistribute the downloaded archives, annotations, or generated dataset.

The resulting local layout should be:

```text
data_preprocessing/
├── prepare_amass_babel_router.py
├── amass_babel_router/
│   ├── features.py
│   ├── intervals.py
│   ├── labels.py
│   └── pipeline.py
├── HDM05/
│   └── MPI_HDM05/
├── babel_v1.0_release/
│   ├── train.json
│   └── val.json
└── processed_amass_babel_router/  # created by the preprocessing script
```

`HDM05/`, `babel_v1.0_release/`, and `processed_amass_babel_router/` are excluded by `.gitignore` wherever they appear in the repository.

From the repository root, enter `data_preprocessing/` and prepare the classifier dataset:

```bash
cd data_preprocessing

python3 prepare_amass_babel_router.py \
  --amass-root HDM05 \
  --babel-root babel_v1.0_release \
  --output-root processed_amass_babel_router

cd ..
```

The script reads BABEL's `train.json` and `val.json`, resolves the referenced AMASS HDM05 motion archives, creates labeled clips and 45-frame feature windows, and writes the generated dataset to `processed_amass_babel_router/`. Use `--overwrite` only when you intend to replace an existing generated dataset.

### Train a replacement model

```bash
python3.11 -m venv .venv-training
.venv-training/bin/pip install -r training/requirements.txt
.venv-training/bin/python training/train_motion_router.py \
  --dataset-root data_preprocessing/processed_amass_babel_router
```

Training parameters:

| Parameter | Required/default | Meaning |
| --- | --- | --- |
| `--dataset-root` | Required | Path to the preprocessed dataset. It must contain `windows.jsonl` and the window files referenced by that manifest. |
| `--output-root` | Repository root | Directory in which `models/` and `reports/` are written. |
| `--max-windows-per-source-class` | `10` | Maximum number of evenly sampled training windows kept for each source motion and coarse class. Raising it uses more data but increases training time and memory use. |
| `--search-iterations` | `30` | Number of random-forest hyperparameter combinations tested. Raising it broadens the search but takes longer. |
| `--random-state` | `42` | Random seed used for reproducible data splits and parameter search. |
| `--n-jobs` | `-1` | Number of parallel scikit-learn workers; `-1` uses all available CPU cores. |
| `--verbose` | `1` | scikit-learn search log level; use `0` for quiet output and a larger value for more detail. |

#### What the training uses

- **Data and labels:** the processed AMASS motion windows and BABEL annotations. Detailed BABEL categories are reduced to the three router classes `walk`, `run`, and `other`.
- **Motion features:** each 45-frame window is sampled at 30 FPS. Its 22-joint rotations and root velocity are summarized as a 175-dimensional descriptor containing joint angular motion, root speed and acceleration, body-part motion energy, left/right symmetry, and motion periodicity.
- **Model selection:** scikit-learn trains a class-balanced random forest. A randomized hyperparameter search selects the model by macro F1 using five source-grouped folds, so windows from the same source motion do not appear on both sides of a cross-validation fold.
- **Routing thresholds:** out-of-fold probabilities are averaged per clip and used to calibrate conservative `walk` and `run` probability thresholds. Predictions that do not pass those thresholds can fall back to `other`.
- **Evaluation and export:** the separate validation split is reported at both window and clip level. The selected forest is exported to JSON for pure-NumPy inference in MotionBuilder, then checked against scikit-learn predictions for numerical and label parity.

The training command writes or refreshes:

- `models/motion_router_v1.json`
- `models/motion_router_v1.sha256`
- `models/feature_schema_v1.json`
- Evaluation artifacts under `reports/`

The default search uses 30 randomized parameter candidates, five source-grouped folds, and random seed `42`. Runtime loading verifies the model metadata and checksum; experimental models require `allow_experimental=True`.

## Troubleshooting

### `Error importing numpy: you should not try to import numpy from its source directory`

This message commonly appears when NumPy's compiled extension cannot load; it does not necessarily mean the project is inside the NumPy source tree.

1. Check MotionBuilder's Python version, operating system, and architecture with the snippet in Installation.
2. Remove or move the incompatible project `.venv`.
3. Recreate `.venv` on the target computer with the matching Python major/minor version and architecture.
4. Confirm the project does not contain an unrelated `numpy.py` file or `numpy/` source directory.
5. Restart MotionBuilder so failed imports are cleared from `sys.modules`.

Installing uv alone does not solve the problem; uv must install a compatible NumPy build into the environment loaded by the plugin.

### `pyfbsdk` or PySide cannot be imported

Run `launcher.py` inside MotionBuilder. These packages normally come from MotionBuilder and are not supplied by the local development environment. The UI tries PySide2 first and PySide6 second.

### Motion classification is unavailable

Confirm that:

- A Character is current in Character Controls.
- The Character Definition is valid and the Character is characterized.
- All required standard body nodes are mapped.
- The Take contains at least 45 frames after 30 FPS resampling.
- `models/motion_router_v1.json` and its checksum file are present and unmodified.

The dialog can continue with **Analyze Anyway**, but the loop detector is specialized for gait motion.

### Bone name is ambiguous or not found

Capture the intended namespace from a selected bone or enter complete `LongName` values. Unqualified duplicate names intentionally fail instead of selecting an arbitrary character.

### Process or Apply becomes disabled

Changing the Character, Take, or frame range invalidates the previous analysis. Run Analyze again, then Process, before Apply.

## Author

**niexiongtao**

niexiongtao@gmail.com
