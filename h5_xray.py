import os
import h5py

h5_path = os.path.expanduser("~/nasa_research/data/utdtb_v5.h5")
print(f"\nScanning HDF5 internal structure for: {h5_path}\n" + "="*50)

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name:<30} | Shape: {obj.shape} | Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group:   {name}")

with h5py.File(h5_path, 'r') as f:
    f.visititems(print_structure)
print("="*50 + "\n")