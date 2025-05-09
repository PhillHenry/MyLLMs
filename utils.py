import os
from datetime import datetime
import shutil

def ensure_unique_dir(path):
    if os.path.exists(path) and os.path.isdir(path):
        # Format current time
        timestamp = datetime.now().strftime('%y%m%d%H%M')
        # Build new name
        new_path = f"{path}_{timestamp}"
        # Rename the existing directory
        shutil.move(path, new_path)
        print(f"Renamed existing directory to: {new_path}")
    # Create a fresh directory
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")

# Example usage
ensure_unique_dir("my_output")
