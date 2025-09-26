"""
Helper module for importing archived modules in pipeline stages.
"""
import sys
import os

def setup_archive_imports():
    """Add the archive/old_src directory to the Python path for imports."""
    archive_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'archive', 'old_src')
    if archive_path not in sys.path:
        sys.path.append(archive_path)

# Set up the imports when this module is imported
setup_archive_imports()
