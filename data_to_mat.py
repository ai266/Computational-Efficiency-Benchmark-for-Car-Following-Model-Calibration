"""Convert the four FollowNet splits to the .mat contract this package reads.

Requires Python 3.10 or later with numpy and scipy.

    python data_to_mat.py [source_dir] [dest_dir]

source_dir defaults to data/raw and falls back to data/; dest_dir defaults to data/.

Inputs, as FollowNet publishes them:

    NGSIM_I_80_train_data.npy      NGSIM_I_80_test_data.npy
    HighD_train_data.npy           HighD_test_data.npy

Each holds per-event records whose four channels are spacing, follower speed, relative speed and
leader speed. NGSIM is already at 10 Hz; HighD is recorded at 25 Hz and is interpolated onto 150
frames, the same 15 s at 10 Hz.

Each output holds `events`, an N x T x 4 float32 array, and `len`, the live frames of each event.
"""
import os
import sys

import numpy as np
from scipy.io import savemat

JOBS = [("NGSIM_I_80_train_data.npy", "ngsim_train.mat", False),
        ("NGSIM_I_80_test_data.npy", "ngsim_test.mat", False),
        ("HighD_train_data.npy", "highd_train.mat", True),
        ("HighD_test_data.npy", "highd_test.mat", True)]

WINDOW = 150          # 15 s at 10 Hz, the window the reader keeps


def convert(src, dst, resample):
    a = np.load(src, allow_pickle=True)
    if a.dtype == object:
        N = len(a)
        lens = np.array([np.asarray(e[1]).ravel().size for e in a])
        T = int(lens.max())
        out = np.zeros((N, T, 4), dtype=np.float32)
        for k in range(N):
            n = lens[k]
            for c in range(4):
                out[k, :n, c] = np.asarray(a[k][c]).ravel()[:n]
        lenv = lens.astype(float)
    else:
        N, C, T0 = a.shape
        out = np.transpose(a, (0, 2, 1)).astype(np.float32)       # (N, T, C)
        if resample:
            t1 = np.linspace(0.0, 1.0, WINDOW)
            idx = t1 * (T0 - 1)
            lo = np.floor(idx).astype(int)
            hi = np.minimum(lo + 1, T0 - 1)
            w = (idx - lo)[None, :, None]
            out = (out[:, lo, :] * (1 - w) + out[:, hi, :] * w).astype(np.float32)
        lenv = np.full((N, 1), out.shape[1], dtype=float)

    savemat(dst, {"events": out, "len": lenv}, do_compression=True)
    kept = min(out.shape[1], WINDOW)
    print("%-18s %-20s N=%-5d T=%-4d kept=%d" % (os.path.basename(dst), str(out.shape), N, out.shape[1], kept))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "data", "raw")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "data")
    if not os.path.isdir(src):
        alt = os.path.join(here, "data")
        if all(os.path.isfile(os.path.join(alt, j[0])) for j in JOBS):
            src = alt
        else:
            raise SystemExit("no FollowNet splits in %s; see README, Data" % src)
    os.makedirs(dst, exist_ok=True)
    for name, out, resample in JOBS:
        f = os.path.join(src, name)
        if not os.path.isfile(f):
            raise SystemExit("missing %s" % f)
        convert(f, os.path.join(dst, out), resample)


if __name__ == "__main__":
    main()
