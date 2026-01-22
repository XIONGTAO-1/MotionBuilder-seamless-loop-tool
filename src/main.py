"""
Main entry point for the Seamless Loop Tool.
"""

import logging

# Configure logging for the tool
logging.basicConfig(
    level=logging.INFO,
    format='[SeamlessLoopTool] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def _import_ui():
    """Import UI module. Separated for easier mocking in tests."""
    from ui.tool_window import create_tool
    return create_tool


# Global reference to keep the tool alive
_tool_instance = None


def show_ui():
    """
    Show the Seamless Loop Tool UI.
    
    This function is called by launcher.py when dragged into MotionBuilder.
    """
    global _tool_instance
    
    logger.info("Initializing...")
    
    try:
        create_tool = _import_ui()
        _tool_instance = create_tool()
        logger.info("Tool created successfully!")
        logger.info("Core modules loaded:")
        logger.info("  - FrameAnalyzer: Pose similarity scoring")
        logger.info("  - RootProcessor: In-Place / Root Motion conversion")
        return _tool_instance
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    # For testing outside of MotionBuilder
    show_ui()
