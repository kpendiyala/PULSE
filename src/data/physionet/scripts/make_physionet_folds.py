import os
import glob
import numpy as np

def load_subject(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"].astype(np.float32)   # (N,3840,3): bvp, acc_mag, temp
    Y = z["Y"].astype(np.float32)   # (N,3840): eda
    L = z["L"].astype(np.int32)     # (N,): 0=rest, 1=stress

    # Map to WESAD-style labels: rest->1, stress->2
    L = (L + 1).astype(np.int32)

    # Dummy ECG channel (zeros) so X becomes (N,3840,4)
    N, T, C = X.shape
    assert (T, C) == (3840, 3), f"Unexpected X shape: {X.shape}"
    ecg = np.zeros((N, T, 1), dtype=np.float32)
    X4 = np.concatenate([ecg, X], axis=2)

    sid = os.path.basename(npz_path).replace("_windows.npz", "")
    return sid, X4, Y, L


def save_fold(out_path, X_train, Y_train, L_train, S_train, X_test, Y_test, L_test, S_test):
    feature_names = np.array(["ecg", "bvp", "net_acc_wrist", "temp"], dtype=object)
    np.savez_compressed(
        out_path,
        X_train=X_train, Y_train=Y_train, L_train=L_train, S_train=S_train,
        X_test=X_test,   Y_test=Y_test,   L_test=L_test,   S_test=S_test,
        feature_names=feature_names
    )


def main(windows_dir, folds_out_dir):
    os.makedirs(folds_out_dir, exist_ok=True)

    all_paths = sorted(glob.glob(os.path.join(windows_dir, "*_windows.npz")))

    # --- NEW: keep only S-subjects (stage 1), ignore f-subjects (stage 2) ---
    paths = []
    for p in all_paths:
        sid = os.path.basename(p).replace("_windows.npz", "")
        if sid.startswith("S"):
            paths.append(p)

    print(f"Found {len(all_paths)} subjects total; using {len(paths)} S-subjects only.")
    assert len(paths) >= 2, "Need at least 2 S-subjects to make LOSO folds."
    # ----------------------------------------------------------------------

    subjects = [load_subject(p) for p in paths]  # list of (sid, X4, Y, L)

    subjects = sorted(subjects, key=lambda x: x[0])
    sids = [s[0] for s in subjects]
    sid_to_int = {sid: i for i, sid in enumerate(sids)}
    print("Loaded S-subjects (sorted):", sids)

    map_path = os.path.join(folds_out_dir, "fold_subjects.txt")
    with open(map_path, "w") as f:
        for i, sid in enumerate(sids):
            f.write(f"{i}\t{sid}\n")
    print(f"Wrote mapping to {map_path}")

    for test_i, (test_sid, X_test, Y_test, L_test) in enumerate(subjects):
        train = [subjects[i] for i in range(len(subjects)) if i != test_i]

        X_train = np.concatenate([t[1] for t in train], axis=0)
        Y_train = np.concatenate([t[2] for t in train], axis=0)
        L_train = np.concatenate([t[3] for t in train], axis=0)
        S_train = np.concatenate(
            [np.full((t[1].shape[0],), sid_to_int[t[0]], dtype=np.int32) for t in train],
            axis=0
        )
        S_test = np.full((X_test.shape[0],), sid_to_int[test_sid], dtype=np.int32)

        out_path = os.path.join(folds_out_dir, f"fold_{test_i}.npz")
        save_fold(out_path, X_train, Y_train, L_train, S_train, X_test, Y_test, L_test, S_test)

        print(f"Wrote {out_path} (test subject={test_sid}) | train={X_train.shape[0]} test={X_test.shape[0]}")

    print("Done.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows_dir", required=True, help="Folder with *_windows.npz files")
    ap.add_argument("--folds_out", required=True, help="Output folder for fold_*.npz")
    args = ap.parse_args()
    main(args.windows_dir, args.folds_out)
