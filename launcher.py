"""
Seamless Loop Tool Launcher for MotionBuilder.

Drag and drop this file into the MotionBuilder viewport to launch the tool.
"""

import sys
import os
import glob

# Get the directory where this script lives
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _add_sys_path(path: str, label: str) -> bool:
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
        print(f"[SeamlessLoopTool] Added {label}: {path}")
        return True
    return False

def _read_python_version_hint() -> str:
    version_file = os.path.join(_SCRIPT_DIR, ".python-version")
    if not os.path.isfile(version_file):
        return ""
    try:
        with open(version_file, "r", encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return ""

def _add_venv_site_packages() -> bool:
    venv_dir = os.path.join(_SCRIPT_DIR, ".venv")
    candidates = [
        os.path.join(venv_dir, "Lib", "site-packages"),
        os.path.join(
            venv_dir,
            "lib",
            f"python{sys.version_info.major}.{sys.version_info.minor}",
            "site-packages",
        ),
    ]

    version_hint = _read_python_version_hint()
    if version_hint:
        parts = version_hint.split(".")
        if len(parts) >= 2:
            py_version = f"python{parts[0]}.{parts[1]}"
        else:
            py_version = f"python{parts[0]}"
        candidates.append(os.path.join(venv_dir, "lib", py_version, "site-packages"))

    candidates.extend(glob.glob(os.path.join(venv_dir, "lib", "python*", "site-packages")))

    added_any = False
    for path in candidates:
        if _add_sys_path(path, "venv site-packages"):
            added_any = True
    return added_any

def _ensure_numpy_available() -> None:
    try:
        import numpy  # noqa: F401
        return
    except Exception:
        pass

    _add_venv_site_packages()

    try:
        import numpy  # noqa: F401
    except Exception as e:
        print(f"[SeamlessLoopTool] NumPy import failed: {e}")
        print("[SeamlessLoopTool] Install numpy into .venv or add it to sys.path.")

# Add the virtual environment's site-packages to Python path when present.
_add_venv_site_packages()
_ensure_numpy_available()

# Add the 'src' folder to Python's search path
_SRC_PATH = os.path.join(_SCRIPT_DIR, "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

def _reload_all_modules():
    """
    Force reload all project modules to clear cached state.
    This ensures pyfbsdk import is re-evaluated each time.
    """
    from importlib import reload
    
    # List of modules to reload in dependency order
    modules_to_reload = [
        'core.loop_analysis',
        'core.root_motion',
        'mobu.adapter',
        'mobu.loop_processor',
        'ui.tool_window',
        'main',
    ]
    
    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            try:
                reload(sys.modules[mod_name])
            except Exception as e:
                print(f"[SeamlessLoopTool] Warning: Could not reload {mod_name}: {e}")

# Force reload all modules to pick up latest changes
_reload_all_modules()

# Now import and run
try:
    import main
    main.show_ui()
except ImportError as e:
    print(f"[SeamlessLoopTool] Failed to import main module: {e}")
    print(f"[SeamlessLoopTool] Searched in: {_SRC_PATH}")
except Exception as e:
    print(f"[SeamlessLoopTool] Error launching tool: {e}")
    import traceback
    traceback.print_exc()
