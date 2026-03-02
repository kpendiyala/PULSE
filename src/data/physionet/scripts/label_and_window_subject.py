#   0 = non-stress (baseline/rest)
#   1 = stress (tasks)
#  -1 = ignore (self-report / transitions / too-short segments)

from pathlib import Path
import numpy as np
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # because src/data/physionet_e4/...
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))

RAW_ROOT = DATA_ROOT / "raw" / "physionet_e4"
PROCESSED_ROOT = DATA_ROOT / "processed" / "physionet_e4"

# -------------------------
# Windowing configuration
# -------------------------
WIN_LEN = 60 * 64       # 60s @ 64Hz = 3840 samples
STRIDE = int(0.25 * 64) # 0.25s @ 64Hz = 16 samples


# -------------------------
# Protocol definitions
# -------------------------

# V1 (S01–S18): 13 tags -> 14 segments
SEGMENTS_V1 = [
    "Baseline", "SL",
    "Stroop", "SL",
    "First Rest",
    "TMCT", "SL",
    "Second Rest",
    "Real Opinion", "SL",
    "Opposite Opinion", "SL",
    "Subtract", "SL",
]

# V2 (f01–f18): fewer tags in your files (9 tags -> 10 segments).
# We'll use duration-based selection of "major" segments, then assign them in this order.
V2_STAGE_ORDER = [
    "Baseline",
    "TMCT",
    "First Rest",
    "Real Opinion",
    "Opposite Opinion",
    "Second Rest",
    "Subtract",
]

# Stage-type mapping
REST_STAGES = {"Baseline", "First Rest", "Second Rest"}
STRESS_STAGES_V1 = {"Stroop", "TMCT", "Real Opinion", "Opposite Opinion", "Subtract"}
STRESS_STAGES_V2 = {"TMCT", "Real Opinion", "Opposite Opinion", "Subtract"}  # no Stroop in v2


# -------------------------
# Helpers
# -------------------------

def compute_segment_minutes(t64: np.ndarray, tags: np.ndarray) -> np.ndarray:
    bounds = [t64[0], *tags.tolist(), t64[-1]]
    mins = np.array([(bounds[i + 1] - bounds[i]) / 60.0 for i in range(len(bounds) - 1)], dtype=np.float64)
    return mins


def segments_to_label64(t64: np.ndarray, tags: np.ndarray, segment_names: list[str], stress_stages: set[str]) -> np.ndarray:
    """
    Label based on fixed segment order.
    """
    if len(segment_names) != len(tags) + 1:
        raise ValueError(f"Need {len(tags)+1} segment names, got {len(segment_names)}")

    label64 = np.full(len(t64), -1, dtype=np.int32)
    boundaries = [t64[0], *tags.tolist(), t64[-1] + 1e-6]

    for i, name in enumerate(segment_names):
        a, b = boundaries[i], boundaries[i + 1]
        ia = np.searchsorted(t64, a, side="left")
        ib = np.searchsorted(t64, b, side="left")

        if name in REST_STAGES:
            label64[ia:ib] = 0
        elif name in stress_stages:
            label64[ia:ib] = 1
        else:
            # SL or unknown -> ignore
            label64[ia:ib] = -1

    return label64


def label_v2_by_duration(t64: np.ndarray, tags: np.ndarray, min_segment_minutes: float = 1.0):
    """
    For f-subjects: tags yield fewer segments (9 tags -> 10 segments).
    We:
      - split by tags
      - treat segments shorter than min_segment_minutes as ignore
      - assign remaining "major" segments in chronological order to V2_STAGE_ORDER
      - if fewer than 7 major segments exist, force the last assigned stage to be 'Subtract'
    Returns: (label64, mapping, seg_mins_list)
    """
    seg_mins = compute_segment_minutes(t64, tags)
    major_idx = [i for i, m in enumerate(seg_mins) if m >= float(min_segment_minutes)]

    label64 = np.full(len(t64), -1, dtype=np.int32)
    bounds = [t64[0], *tags.tolist(), t64[-1] + 1e-6]

    mapping = {}
    if len(major_idx) == 0:
        return label64, {"error": "no major segments"}, seg_mins.tolist()

    if len(major_idx) >= len(V2_STAGE_ORDER):
        use_idx = major_idx[: len(V2_STAGE_ORDER)]
        use_stages = V2_STAGE_ORDER
    else:
        use_idx = major_idx[:]
        use_stages = V2_STAGE_ORDER[: len(use_idx)]
        # anchor end of protocol
        use_stages[-1] = "Subtract"

    for seg_i, stage_name in zip(use_idx, use_stages):
        a, b = bounds[seg_i], bounds[seg_i + 1]
        ia = np.searchsorted(t64, a, side="left")
        ib = np.searchsorted(t64, b, side="left")

        if stage_name in REST_STAGES:
            lab = 0
        elif stage_name in STRESS_STAGES_V2:
            lab = 1
        else:
            lab = -1

        label64[ia:ib] = lab
        mapping[seg_i] = {"stage": stage_name, "minutes": float(seg_mins[seg_i]), "label": int(lab)}

    return label64, mapping, seg_mins.tolist()


def windowize(bvp: np.ndarray, acc: np.ndarray, temp: np.ndarray, eda: np.ndarray, label64: np.ndarray):
    """
    Create (X,Y,L) windows:
      X: (N, 3840, 3)  [bvp, acc_mag, temp]
      Y: (N, 3840)     eda (teacher)
      L: (N,)          {0,1}
    Only keep windows fully inside a single label.
    """
    X_list, Y_list, L_list = [], [], []
    T = len(label64)

    for s in range(0, T - WIN_LEN + 1, STRIDE):
        lab = int(label64[s])
        if lab < 0:
            continue
        if np.all(label64[s:s + WIN_LEN] == lab):
            X = np.stack([bvp[s:s + WIN_LEN], acc[s:s + WIN_LEN], temp[s:s + WIN_LEN]], axis=-1)
            Y = eda[s:s + WIN_LEN]
            X_list.append(X.astype(np.float32))
            Y_list.append(Y.astype(np.float32))
            L_list.append(lab)

    if not X_list:
        return None
    return np.stack(X_list), np.stack(Y_list), np.array(L_list, dtype=np.int32)

EPS = 1e-6

def baseline_zscore_windows(X: np.ndarray, Y: np.ndarray, L: np.ndarray):
    """
    X: (N, WIN_LEN, 3)  [bvp, acc_mag, temp]
    Y: (N, WIN_LEN)     eda
    L: (N,)             0 baseline/rest, 1 stress
    Returns normalized X,Y and stats.
    """
    base = (L == 0)
    if not np.any(base):
        raise ValueError("No baseline windows found; cannot compute baseline z-score stats.")

    # Per-channel stats for X over (windows,time)
    x_mean = X[base].mean(axis=(0, 1))
    x_std  = X[base].std(axis=(0, 1))
    x_std  = np.where(x_std < EPS, 1.0, x_std).astype(np.float32)
    x_mean = x_mean.astype(np.float32)

    # Scalar stats for Y (EDA)
    y_mean = np.float32(Y[base].mean())
    y_std  = np.float32(Y[base].std())
    if float(y_std) < EPS:
        y_std = np.float32(1.0)

    Xn = ((X - x_mean[None, None, :]) / x_std[None, None, :]).astype(np.float32)
    Yn = ((Y - y_mean) / y_std).astype(np.float32)
    return Xn, Yn, x_mean, x_std, y_mean, y_std

# -------------------------
# Main
# -------------------------

def main(subject: str = "S01"):
    in_path = PROCESSED_ROOT / "intermediate" / "STRESS" / f"{subject}_aligned64.npz"
    out_labeled = PROCESSED_ROOT / "intermediate" / "STRESS" / f"{subject}_labeled64.npz"
    out_windows = PROCESSED_ROOT / "STRESS" / f"{subject}_windows.npz"
    out_windows.parent.mkdir(parents=True, exist_ok=True)
    out_labeled.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(in_path, allow_pickle=True)
    t64 = d["t64"]
    tags = d["tags"]

    # Choose labeling strategy
    if len(tags) == 13 and subject.startswith("S"):
        label64 = segments_to_label64(t64, tags, SEGMENTS_V1, STRESS_STAGES_V1)
    elif subject.startswith("f") or len(tags) == 9:
        label64, mapping, seg_mins = label_v2_by_duration(t64, tags, min_segment_minutes=1.0)
        print("V2 segment minutes:", [round(float(x), 2) for x in seg_mins])
        print("V2 mapping (segment_index -> stage):", {k: v["stage"] for k, v in mapping.items()})
    else:
        raise ValueError(f"Unexpected tag count {len(tags)} for subject {subject}")

    # Save labeled64 (aligned signals + label64)
    np.savez_compressed(out_labeled, **{k: d[k] for k in d.files if k != "meta"}, label64=label64)

    # Window and save final training file
    pack = windowize(d["bvp"], d["acc_mag"], d["temp"], d["eda"], label64)
    if pack is None:
        print("No valid windows found. Check labels or window settings.")
        return

    X, Y, L = pack

    # --- NEW: baseline-per-subject normalization (train/test leakage-safe since per subject) ---
    X, Y, x_mean, x_std, y_mean, y_std = baseline_zscore_windows(X, Y, L)

    np.savez_compressed(
        out_windows,
        X=X, Y=Y, L=L,
        x_mean=x_mean, x_std=x_std,
        y_mean=y_mean, y_std=y_std,
    )

    print("Saved labeled:", out_labeled)
    print("Saved windows:", out_windows)
    print("X:", X.shape, "Y:", Y.shape, "L:", L.shape)
    u, c = np.unique(L, return_counts=True)
    print("Label counts:", dict(zip(u.tolist(), c.tolist())))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S01")
    args = ap.parse_args()
    main(args.subject)