import h5py
from pathlib import Path

# Pointing to the new WSL location
file_path = Path("~/nasa_research/data/utdtb_v5.h5").expanduser()

print(f"Checking {file_path}...\n")
try:
    with h5py.File(file_path, 'r') as f:
        for split in ['train', 'val', 'test']:
            if split in f:
                print(f"--- {split.upper()} SPLIT ---")
                for key in f[split].keys():
                    shape = f[split][key].shape
                    print(f"  {key:<15}: {shape}")
                print()
            else:
                print(f"Split '{split}' not found!")
except Exception as e:
    print(f"Error loading HDF5: {e}")