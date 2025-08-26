import json
import numpy as np
from typing import Tuple, Dict

def save_npz(path: str, **arrays):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)

class GraphDataset:
    """Loads a padded graph dataset saved by make_dataset.py."""
    def __init__(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.A = data["A"].astype(np.float32)      # (B, N, N)
        self.X = data["X"].astype(np.float32)      # (B, N, D)
        self.mask = data["mask"].astype(np.float32)  # (B, N)
        self.y = data["y"].astype(np.float32)      # (B,)
        meta = json.loads(str(data["meta"].item()))
        self.max_n = int(meta["max_n"])
        self.feat_dim = int(meta["feat_dim"])

    def __len__(self):
        return self.A.shape[0]

    def get_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.A, self.X, self.mask, self.y
