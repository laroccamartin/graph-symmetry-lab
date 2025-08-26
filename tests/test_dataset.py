import os, tempfile, numpy as np
from src.data.make_dataset import main as make_main
from src.data.dataset import GraphDataset

def test_make_and_load_dataset():
    # create a tiny dataset to keep CI snappy
    import sys
    import argparse
    # simulate CLI args
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "tiny.npz")
        import subprocess, sys
        # call the generator via module interface
        # re-implementing minimal arg call (avoids subprocess in CI)
        import types
        import src.data.make_dataset as mk
        import argparse
        parser = argparse.ArgumentParser()
        # direct call to mk.main with args by patching sys.argv is simpler:
        argv = ["prog", "--out", out, "--n_graphs", "20", "--min_n", "4", "--max_n", "6", "--seed", "123"]
        bak = sys.argv
        sys.argv = argv
        try:
            mk.main()
        finally:
            sys.argv = bak
        ds = GraphDataset(out)
        A, X, mask, y = ds.get_arrays()
        assert A.ndim == 3 and X.ndim == 3 and mask.ndim == 2 and y.ndim == 1
        assert A.shape[0] == X.shape[0] == mask.shape[0] == y.shape[0]
        # check masks in {0,1}
        assert set(np.unique(mask)).issubset({0.0, 1.0})
        # quick forward sanity: adjacency diagonal is zero for simple graphs
        assert np.allclose(np.diagonal(A, axis1=1, axis2=2), 0.0)
