# Export FBX at User-Selected FPS (Sandbox Baking) Design

**Goal**
Add a non‑breaking Export FBX flow that converts already‑processed animation to a user‑selected FPS (30/60/90/120) using a sandbox take, without altering existing Analyze/Process/Apply behavior.

**Architecture**
The export path is isolated from the main workflow. A new Export button in the UI reads a chosen FPS and runs a sandbox baking pipeline. The pipeline creates a new take named `<CurrentTake>_inplace` (with numeric suffix if needed), injects processed in‑memory data, snaps the loop end to the target FPS grid, re‑runs linear offset compensation to close the loop, resamples/bakes at the target FPS, and exports only that sandbox take. The original take and existing buttons remain unchanged.

**Components**
- `src/ui/tool_window.py`: add Export FPS selector and Export FBX button; enable only after Process.
- `src/mobu/loop_processor.py`: add `export_fbx(...)` method that requires processed data and delegates to adapter helpers.
- `src/mobu/adapter.py`: add helpers for unique take naming, create/switch sandbox take, set transport FPS, plot/bake on skeleton with `FBPlotOptions`, export a single take with `FBFbxOptions`, and compute source FPS from time mode (fallback to `FBTime(0,0,0,1).GetSecondDouble()` inverse).
- `src/core/root_motion.py`: reuse `blend_loop_ends` to compensate after end‑frame snapping.

**Data Flow**
Process populates `processed_data`/`processed_trajectory`. Export reads UI FPS → creates sandbox take → clears and writes processed data → computes duration in seconds and snaps end frame to target FPS grid → re‑compensates loop → plots/bakes at target FPS → saves FBX with only the sandbox take selected.

**Error Handling**
Export fails early if no processed data. If take creation or export fails, restore the original take and report status. Sandbox take is retained as requested.

**Testing**
Unit tests use `MockMoBuAdapter` to verify unique take naming, target FPS selection plumbing, and non‑breaking behavior (Export requires Process). Integration testing for actual bake/export is manual in MotionBuilder.
