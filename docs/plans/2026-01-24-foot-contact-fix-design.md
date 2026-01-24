# Foot Contact Fix (FK In-Place) Design

## Goal
Reduce foot sliding for FK in-place loops by detecting stance phases from world-space foot/toe motion, then clamping local Y curves during contacts after the take is written.

## Workflow
1. Apply the processed hierarchy to a new take.
2. Sample world-space translations for foot and toe across the take.
3. Compute per-frame height (min of foot/toe) and speed.
4. Detect contact intervals:
   - height ≤ 2.0
   - speed ≤ 0.5
   - min span ≥ 3 frames
5. For each contact interval, clamp local Y of foot/toe to ground height (default 0.0).
6. Repeat per left/right foot.

## Parameters
- `ground_height`: 0.0
- `height_threshold`: 2.0
- `speed_threshold`: 0.5
- `min_span`: 3

## Rationale
- FK local translations can be static or noisy; world-space sampling is more reliable for contact detection.
- Running the fix after hierarchy write ensures results reflect resampling and in-place processing.
