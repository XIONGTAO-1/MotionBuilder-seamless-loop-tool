"""
Pipeline IO utilities for file handling.
"""

import os
from typing import List


def find_files(directory: str, extension: str) -> List[str]:
    """
    Find all files in a directory with the given extension.
    
    Args:
        directory: Path to search
        extension: File extension to match (e.g., ".fbx")
        
    Returns:
        List of absolute file paths
    """
    result = []
    if not os.path.isdir(directory):
        return result
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(extension.lower()):
            result.append(os.path.join(directory, filename))
    return result


def load_scene(path: str) -> bool:
    """
    Load a scene file into MotionBuilder.
    
    Note: This is a stub. Implementation requires pyfbsdk.
    
    Args:
        path: Path to the file to load
        
    Returns:
        True if successful, False otherwise
    """
    # TODO: Implement with pyfbsdk
    # from pyfbsdk import FBApplication
    # return FBApplication().FileOpen(path)
    raise NotImplementedError("Requires MotionBuilder environment")


def export_fbx(path: str, settings: dict = None) -> bool:
    """
    Export the current scene as FBX.
    
    Note: This is a stub. Implementation requires pyfbsdk.
    
    Args:
        path: Output file path
        settings: Export settings dictionary
        
    Returns:
        True if successful, False otherwise
    """
    # TODO: Implement with pyfbsdk
    raise NotImplementedError("Requires MotionBuilder environment")
