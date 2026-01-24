# Foot Contact Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reapply the foot contact detection + stance correction workflow for FK in-place animations, update UI fields, and push the result to the `fix-feet` branch.

**Architecture:** Add world-space foot sampling in the MoBu adapter, detect contact intervals in `LoopProcessor`, apply stance-phase correction after hierarchy write, and expose foot/toe bones in the UI. Keep defaults (height ≤ 2.0, speed ≤ 0.5, min span ≥ 3) and ground height = 0.

**Tech Stack:** Python 3, pytest, MotionBuilder adapter APIs, Qt UI (tool_window.py)

---

### Task 1: Add tests for contact detection + stance correction (@test-driven-development)

**Files:**
- Modify: `seamless_loop_tool/tests/test_loop_processor.py`

**Step 1: Write the failing tests**

```python
def test_detect_contact_intervals():
    processor = LoopProcessor(MockMoBuAdapter())
    heights = [5, 2, 1, 0.5, 2, 5]
    speeds = [2, 0.4, 0.3, 0.2, 0.4, 2]
    contacts = processor.detect_contact_intervals(
        heights,
        speeds,
        height_threshold=2.0,
        speed_threshold=0.5,
        min_span=2,
    )
    assert contacts == [(1, 4)]

def test_apply_stance_correction():
    adapter = MockMoBuAdapter()
    processor = LoopProcessor(adapter)
    take = "Take 001"
    node = "LeftFoot"
    frames = list(range(5))
    translations = [(0, 1.0, 0)] * 5
    adapter.set_mock_trajectory(take, node, frames, translations, [])
    contacts = [(1, 3)]
    processor.apply_stance_correction(take, node, contacts, ground_height=0.0)
    updated = adapter.get_mock_trajectory(take, node)[1]
    assert updated[1:4] == [(0, 0.0, 0)] * 3

def test_apply_foot_contact_fix_world_space():
    adapter = MockMoBuAdapter()
    processor = LoopProcessor(adapter)
    take = "Take 001"
    foot = "LeftFoot"
    frames = [0, 1, 2, 3, 4]
    local_translations = [(0, 2.5, 0)] * 5
    adapter.set_mock_trajectory(take, foot, frames, local_translations, [])
    adapter.set_mock_world_positions(take, foot, [(0, 0.5, 0)] * 5)
    adapter.set_mock_world_positions(take, "LeftToeBase", [(0, 0.6, 0)] * 5)
    processor.apply_foot_contact_fix(
        take,
        foot,
        "LeftToeBase",
        ground_height=0.0,
        height_threshold=2.0,
        speed_threshold=0.5,
        min_span=3,
    )
    updated = adapter.get_mock_trajectory(take, foot)[1]
    assert updated[0:5] == [(0, 0.0, 0)] * 5
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest seamless_loop_tool/tests/test_loop_processor.py -k "contact_intervals or stance_correction or foot_contact_fix"`
Expected: FAIL with `AttributeError` (methods missing)

**Step 3: Write minimal implementation**

Add `detect_contact_intervals`, `apply_stance_correction`, and `apply_foot_contact_fix` to `LoopProcessor`, plus world-space sampling hook in adapter. Keep defaults and clamp to ground height.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest seamless_loop_tool/tests/test_loop_processor.py -k "contact_intervals or stance_correction or foot_contact_fix"`
Expected: PASS

**Step 5: Commit**

```bash
git add seamless_loop_tool/tests/test_loop_processor.py
git commit -m "test: cover foot contact detection and stance correction"
```

---

### Task 2: Implement adapter + processor + UI changes (@clean-code)

**Files:**
- Modify: `seamless_loop_tool/src/mobu/adapter.py`
- Modify: `seamless_loop_tool/src/mobu/loop_processor.py`
- Modify: `seamless_loop_tool/src/ui/tool_window.py`

**Step 1: Write the failing integration test (if needed)**

Extend `test_loop_processor.py` to validate adapter world-space sampling plumbing if failures indicate missing mock hooks.

**Step 2: Implement adapter world-space sampling**

Add `get_world_translations` to `MoBuAdapter` and support in `MockMoBuAdapter` with `set_mock_world_positions`.

**Step 3: Implement LoopProcessor contact correction**

Add contact detection helpers, apply stance correction after hierarchy write, and ensure the new workflow runs for left/right feet with toes.

**Step 4: Update UI fields**

Add left/right foot + toe inputs under Root Bone and remove Min Avg Velocity. Ensure Apply uses defaults.

**Step 5: Run tests**

Run: `python3 -m pytest seamless_loop_tool/tests/test_loop_processor.py -k "contact_intervals or stance_correction or foot_contact_fix"`
Expected: PASS

**Step 6: Commit**

```bash
git add seamless_loop_tool/src/mobu/adapter.py seamless_loop_tool/src/mobu/loop_processor.py seamless_loop_tool/src/ui/tool_window.py
git commit -m "feat: add foot contact fix workflow"
```

---

### Task 3: Update docs and verify (@clean-code)

**Files:**
- Modify: `seamless_loop_tool/README.md`
- Add: `seamless_loop_tool/docs/plans/2026-01-24-foot-contact-fix-design.md`

**Step 1: Update README**

Document foot/toe fields under Basic Settings and add a foot-contact correction step in the Core Design Logic.

**Step 2: Add design note**

Create `2026-01-24-foot-contact-fix-design.md` describing world-space detection + stance correction flow.

**Step 3: Run full tests**

Run: `python3 -m pytest`
Expected: PASS

**Step 4: Commit**

```bash
git add seamless_loop_tool/README.md seamless_loop_tool/docs/plans/2026-01-24-foot-contact-fix-design.md
git commit -m "docs: describe foot contact correction workflow"
```
