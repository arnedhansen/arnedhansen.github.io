#!/usr/bin/env python3
"""
AOC Export for Website — Reads processed .mat files and writes lightweight JSON
for the Alpha Oculomotor Control interactive visualization dashboard.

Run from website repo root: python export-scripts/export_for_website.py

Output: resources/AOC/data/*.json
"""

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

# Try scipy first, mat73 for v7.3 files
try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import mat73
    HAS_MAT73 = True
except ImportError:
    HAS_MAT73 = False

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — adapt to your setup
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
WEBSITE_ROOT = SCRIPT_DIR.parent

PATHS = {
    # AOC data root (features directory with per-subject folders)
    "feat_path": "/Volumes/g_psyplafor_methlab$/Students/Arne/AOC/data/features/",
    # Layout for topography (ANT Neuro cap)
    "layout_path": "/Volumes/g_psyplafor_methlab$/Students/Arne/toolboxes/headmodel/layANThead.mat",
    # Output: relative to website root
    "out_dir": WEBSITE_ROOT / "resources" / "AOC" / "data",
}

# Viz types with parameterized (SpecParam) data — show "Baselined + SpecParam" only for these
VIZ_WITH_PARAMETERIZED = {"spectrum", "tfr", "topo"}

# Posterior/occipital channels (O, I)
def _is_occipital(label: str) -> bool:
    return "O" in label.upper() or "I" in label.upper()


def _load_mat(path: Path):
    """Load .mat file; try scipy first, then mat73 for v7.3."""
    if not path.exists():
        return None
    try:
        if HAS_SCIPY:
            data = loadmat(str(path), struct_as_record=False, squeeze_me=True)
            # Remove MATLAB metadata keys
            return {k: v for k, v in data.items() if not k.startswith("__")}
        return None
    except Exception as e:
        if HAS_MAT73 and "v7.3" in str(e).lower() or "Cannot read" in str(e):
            try:
                return mat73.loadmat(str(path))
            except Exception:
                return None
        raise
    return None


def _unwrap_mat_obj(x):
    """Unwrap scipy 0-d object arrays so mat_struct fields are accessible."""
    if x is None:
        return None
    if isinstance(x, np.ndarray) and x.dtype == object:
        if x.ndim == 0:
            return x.item()
        if x.size == 1:
            return x.flat[0]
    return x


def _ft_pow_freq_labels(powdat, mat_dict=None):
    """
    Extract powspctrm, freq, label from FieldTrip-like MATLAB struct.
    mat_dict: optional parent .mat dict for top-level 'freq' / 'label' fallback.
    Returns (pwr, fq, labels); any element may be None if missing.
    """
    s = _unwrap_mat_obj(powdat)
    if s is None:
        return None, None, None
    pwr = None
    if hasattr(s, "powspctrm"):
        pwr = np.asarray(s.powspctrm)
    elif isinstance(s, np.ndarray) and s.dtype is not None and s.dtype.names and "powspctrm" in s.dtype.names:
        pwr = np.asarray(s["powspctrm"])

    if hasattr(s, "freq"):
        fq = np.asarray(s.freq).flatten()
    elif isinstance(s, np.ndarray) and s.dtype is not None and s.dtype.names and "freq" in s.dtype.names:
        fq = np.asarray(s["freq"]).flatten()
    elif mat_dict is not None and "freq" in mat_dict:
        fq = np.asarray(mat_dict["freq"]).flatten()
    else:
        fq = None

    if hasattr(s, "label"):
        labels = np.asarray(s.label).flatten()
    elif isinstance(s, np.ndarray) and s.dtype is not None and s.dtype.names and "label" in s.dtype.names:
        labels = np.asarray(s["label"]).flatten()
    elif mat_dict is not None and "label" in mat_dict:
        labels = np.asarray(mat_dict["label"]).flatten()
    else:
        labels = None

    return pwr, fq, labels


def _mat_struct_rows(arr):
    """
    Flatten MATLAB struct arrays to a list of per-row objects (mat_struct, void, etc.).
    SciPy often loads N×1 struct arrays as shape (N, 1) object ndarrays; naive iteration fails.
    """
    if arr is None:
        return []
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim == 0:
        return [a.item()]
    rows = []
    if getattr(a.dtype, "names", None):
        for i in range(len(a)):
            rows.append(a[i])
        return rows
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    for i in range(len(a)):
        rows.append(a[i])
    return rows


def _row_field(row, *names):
    """Read first available field from MATLAB struct / numpy structured row."""
    for name in names:
        if name is None:
            continue
        if hasattr(row, name):
            try:
                return getattr(row, name)
            except AttributeError:
                pass
        try:
            if isinstance(row, np.void) and getattr(row.dtype, "names", None) and name in row.dtype.names:
                return row[name]
        except (TypeError, ValueError, KeyError):
            pass
        try:
            return row[name]
        except (TypeError, KeyError, IndexError):
            pass
    return np.nan


def _pick_gaze_aggregate(d, task: str, trials: bool = False):
    """Resolve variable name inside gaze_matrix_*.mat (MATLAB naming may vary)."""
    if d is None:
        return None
    if trials:
        for k in (f"gaze_data_{task}_trials", f"gaze_data_{task}_trial"):
            if k in d:
                return d[k]
    else:
        for k in (f"gaze_data_{task}",):
            if k in d:
                return d[k]
    for key, val in d.items():
        if key.startswith("_"):
            continue
        lk = key.lower()
        if "gaze" in lk and task in key and isinstance(val, (np.ndarray, list)) and np.asarray(val).size > 0:
            return val
    return None


def _pick_trial_struct(d, task: str, family: str):
    """Resolve trial-level struct variable names (e.g., gaze_data_nback_trials)."""
    if d is None:
        return None
    candidates = (
        f"{family}_data_{task}_trials",
        f"{family}_data_{task}_trial",
        f"{family}_{task}_trials",
    )
    for k in candidates:
        if k in d:
            return d[k]
    for key, val in d.items():
        if key.startswith("_"):
            continue
        lk = key.lower()
        if family in lk and task in lk and "trial" in lk and isinstance(val, (np.ndarray, list)):
            if np.asarray(val).size > 0:
                return val
    return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(out_path: Path, obj: dict) -> int:
    """Write JSON; return size in KB."""
    _ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        json.dump(obj, f, indent=0)
    return out_path.stat().st_size // 1024


def _write_data_manifest(out_dir: Path) -> None:
    """Write list of exported JSON files for frontend availability checks."""
    files = sorted(
        [p.name for p in out_dir.glob("*.json") if p.name not in {"data_manifest.json"}]
    )
    _write_json(out_dir / "data_manifest.json", {"files": files})


def _get_subjects(feat_path: Path) -> list:
    """List subject IDs (folder names) in feat_path."""
    if not feat_path.exists():
        return []
    subs = [
        d.name
        for d in feat_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    return sorted(subs, key=lambda x: (len(x), x))


# ─────────────────────────────────────────────────────────────────────────────
# Power spectrum
# ─────────────────────────────────────────────────────────────────────────────
def _export_spectrum_nback(feat_path: Path, out_dir: Path, summary: list) -> None:
    """Export N-back power spectrum (raw, baselined, parameterized)."""
    subjects = _get_subjects(feat_path)
    if not subjects:
        return
    cond_map = {(1, "1back"), (2, "2back"), (3, "3back")}

    for proc, file_suffix, var_prefix in [
        ("raw", "power_nback.mat", "powload"),
        ("baselined", "power_nback_fooof.mat", "pow_fooof_bl"),
        ("parameterized", "power_nback_fooof.mat", "pow_fooof"),
    ]:
        all_cond_data = {c: [] for _, c in cond_map}
        freqs = None
        iaf = None
        for subj in subjects:
            p = feat_path / subj / "eeg" / file_suffix
            d = _load_mat(p)
            if d is None:
                continue
            if proc == "raw":
                for load_idx, cond_name in [(1, "1back"), (2, "2back"), (3, "3back")]:
                    key = f"powload{load_idx}"
                    if key not in d:
                        continue
                    raw_ent = _unwrap_mat_obj(d[key])
                    pwr, fq, labels = _ft_pow_freq_labels(raw_ent, d)
                    if pwr is None or fq is None or labels is None:
                        continue
                    idx = np.array([j for j, l in enumerate(labels) if _is_occipital(str(l))])
                    if len(idx) > 0:
                        mean_pwr = np.nanmean(pwr[idx, :] if pwr.ndim == 2 else pwr[:, idx, :].mean(axis=1), axis=0)
                        all_cond_data[cond_name].append(mean_pwr)
                    if freqs is None:
                        freqs = fq.tolist()
            else:
                for load_idx, cond_name in [(1, "1back"), (2, "2back"), (3, "3back")]:
                    key = f"pow{load_idx}_fooof_bl" if proc == "baselined" else f"pow{load_idx}_fooof"
                    if key not in d:
                        key = f"powload{load_idx}_fooof_bl" if proc == "baselined" else f"powload{load_idx}_fooof"
                    if key not in d:
                        continue
                    arr = _unwrap_mat_obj(d[key])
                    pwr, fq, labels = _ft_pow_freq_labels(arr, d)
                    if pwr is None or fq is None or labels is None:
                        continue
                    idx = np.array([j for j, l in enumerate(labels) if _is_occipital(str(l))])
                    if len(idx) > 0 and pwr.ndim >= 2:
                        mean_pwr = np.nanmean(pwr[idx, :] if pwr.ndim == 2 else pwr[:, idx, :].mean(axis=1), axis=0)
                        all_cond_data[cond_name].append(mean_pwr)
                    if freqs is None:
                        freqs = fq.tolist()

        if not freqs or all(len(v) == 0 for v in all_cond_data.values()):
            continue
        conditions = {}
        sem = {}
        for c, vals in all_cond_data.items():
            if vals:
                m = np.nanmean(vals, axis=0).tolist()
                s = (np.nanstd(vals, axis=0) / np.sqrt(len(vals))).tolist()
                conditions[c] = m
                sem[c] = s
        if not conditions:
            continue
        out = {"freqs": freqs, "conditions": conditions, "sem": sem}
        if iaf is not None:
            out["iaf"] = float(iaf)
        out_path = out_dir / f"nback_spectrum_{proc}.json"
        kb = _write_json(out_path, out)
        summary.append((out_path.name, str([len(conditions), len(freqs)]), kb))


def _export_spectrum_sternberg(feat_path: Path, out_dir: Path, summary: list) -> None:
    """Export Sternberg power spectrum (raw, baselined, parameterized)."""
    subjects = _get_subjects(feat_path)
    if not subjects:
        return
    cond_map = [(2, "2items"), (4, "4items"), (6, "6items")]

    for proc, file_suffix, var_map in [
        ("raw", "power_stern_raw.mat", {2: "powload2", 4: "powload4", 6: "powload6"}),
        ("baselined", "power_stern_fooof.mat", {2: "pow2_fooof_bl", 4: "pow4_fooof_bl", 6: "pow6_fooof_bl"}),
        ("parameterized", "power_stern_fooof.mat", {2: "pow2_fooof", 4: "pow4_fooof", 6: "pow6_fooof"}),
    ]:
        all_cond_data = {"2items": [], "4items": [], "6items": []}
        freqs = None
        for subj in subjects:
            p = feat_path / subj / "eeg" / file_suffix
            d = _load_mat(p)
            if d is None:
                continue
            for load, cond_name in cond_map:
                key = var_map.get(load)
                if key not in d:
                    continue
                arr = _unwrap_mat_obj(d[key])
                pwr, fq, labels = _ft_pow_freq_labels(arr, d)
                if pwr is None or fq is None or labels is None:
                    continue
                idx = np.array([j for j, l in enumerate(labels) if _is_occipital(str(l))])
                if len(idx) > 0 and pwr.ndim >= 2:
                    mean_pwr = np.nanmean(pwr[idx, :] if pwr.ndim == 2 else pwr[:, idx, :].mean(axis=1), axis=0)
                    all_cond_data[cond_name].append(mean_pwr)
                if freqs is None:
                    freqs = fq.tolist()
        if not freqs or all(len(v) == 0 for v in all_cond_data.values()):
            continue
        conditions = {}
        sem = {}
        for c, vals in all_cond_data.items():
            if vals:
                m = np.nanmean(vals, axis=0).tolist()
                s = (np.nanstd(vals, axis=0) / np.sqrt(len(vals))).tolist()
                conditions[c] = m
                sem[c] = s
        if not conditions:
            continue
        out = {"freqs": freqs, "conditions": conditions, "sem": sem}
        out_path = out_dir / f"sternberg_spectrum_{proc}.json"
        kb = _write_json(out_path, out)
        summary.append((out_path.name, str([len(conditions), len(freqs)]), kb))


# ─────────────────────────────────────────────────────────────────────────────
# Gaze metrics — from trial-level gaze_matrix_*_trials
# ─────────────────────────────────────────────────────────────────────────────
def _export_gaze(feat_path: Path, out_dir: Path, summary: list) -> None:
    """Export gaze metrics from trial-level files with subject-level aggregation."""
    for task in ["nback", "sternberg"]:
        trials_file = feat_path / f"gaze_matrix_{task}_trials.mat"
        if not trials_file.exists():
            trials_file = None
        if not trials_file:
            print(f"  Warning: missing trial-level gaze file for {task}")
            continue
        dt = _load_mat(Path(trials_file))
        gaze_trials = _pick_trial_struct(dt, task, "gaze")
        if gaze_trials is None:
            print(f"  Warning: missing gaze trial struct in {trials_file.name}")
            continue
        rows = _mat_struct_rows(gaze_trials)
        if not rows:
            continue

        cond_names = {"nback": ["1back", "2back", "3back"], "sternberg": ["2items", "4items", "6items"]}[task]
        cond_codes = {"nback": (1, 2, 3), "sternberg": (2, 4, 6)}[task]
        for metric, field_raw, field_bl in [
            ("gaze_deviation", "GazeDeviationFull", "GazeDeviationFullBL"),
            ("spl", "ScanPathLengthFull", "ScanPathLengthFullBL"),
            ("bcea", "BCEAFull", "BCEAFullBL"),
        ]:
            for proc, field in [("raw", field_raw), ("baselined", field_bl)]:
                try:
                    conds = np.asarray([_row_field(r, "Condition") for r in rows], dtype=float)
                    ids = np.asarray([_row_field(r, "ID") for r in rows], dtype=float)
                    vals = np.asarray([_row_field(r, field, field_raw) for r in rows], dtype=float)
                except (AttributeError, KeyError, TypeError, ValueError):
                    vals = None
                if vals is None or vals.size == 0:
                    continue

                # Aggregate by subject first, then group-average subject means.
                means_by_cond = {}
                sds_by_cond = {}
                ind_by_cond = {}
                for cn, code in zip(cond_names, cond_codes):
                    mask = conds == code
                    if mask.sum() > 0:
                        cond_ids = ids[mask]
                        cond_vals = vals[mask]
                        subj_means = []
                        for sid in np.unique(cond_ids[np.isfinite(cond_ids)]):
                            smask = cond_ids == sid
                            v = cond_vals[smask]
                            v = v[np.isfinite(v)]
                            if len(v) > 0:
                                subj_means.append(float(np.nanmean(v)))
                        if len(subj_means) > 0:
                            means_by_cond[cn] = float(np.nanmean(subj_means))
                            sds_by_cond[cn] = float(np.nanstd(subj_means))
                            ind_by_cond[cn] = subj_means
                if not means_by_cond:
                    continue
                order = [c for c in cond_names if c in means_by_cond]
                out = {
                    "metric": metric,
                    "unit": "px" if metric != "spl" else "px",
                    "conditions": order,
                    "means": [means_by_cond[c] for c in order],
                    "sds": [sds_by_cond[c] for c in order],
                    "individual_subject_means": {c: ind_by_cond[c] for c in order},
                }
                out_path = out_dir / f"{task}_{metric}_{proc}.json"
                kb = _write_json(out_path, out)
                summary.append((out_path.name, str(means_by_cond.keys()), kb))

        # Gaze velocity: compute from trial data if available
        for proc in ["raw", "baselined"]:
            vel_means, vel_sds, vel_ind = {}, {}, {}
            for ci, cn in enumerate(cond_names):
                vel_means[cn] = []
            for subj in _get_subjects(feat_path):
                gdir = feat_path / subj / "gaze"
                trials_file = gdir / "gaze_matrix_{}_trial.mat".format(task)
                if not trials_file.exists():
                    trials_file = gdir / f"gaze_series_{task}_trials.mat"
                if not trials_file.exists():
                    continue
                d = _load_mat(Path(trials_file))
                if d is None:
                    continue
                td = d.get("subj_data_gaze_trial", d.get("gaze_x", None))
                if td is None:
                    # Try gaze_x, gaze_y from gaze_series
                    gx = d.get("gaze_x")
                    gy = d.get("gaze_y")
                    ti = d.get("trialinfo", [])
                    if gx is not None and gy is not None:
                        try:
                            from scipy.signal import savgol_filter
                            fs = 500
                            for trl in range(min(len(np.atleast_1d(gx)), len(ti) if hasattr(ti, "__len__") else 1)):
                                x = np.atleast_1d(gx)[trl] if hasattr(gx, "__getitem__") else gx
                                y = np.atleast_1d(gy)[trl] if hasattr(gy, "__getitem__") else gy
                                if np.size(x) < 10:
                                    continue
                                x = np.asarray(x).flatten()
                                y = np.asarray(y).flatten()
                                w = min(11, len(x) if len(x) % 2 else len(x) - 1)
                                if w < 3:
                                    continue
                                vx = savgol_filter(x.astype(float), w, 3, deriv=1) * fs
                                vy = savgol_filter(y.astype(float), w, 3, deriv=1) * fs
                                v = np.sqrt(vx ** 2 + vy ** 2)
                                v = v[np.isfinite(v)]
                                if len(v) > 0:
                                    cond_idx = int(ti[trl]) - 20 if hasattr(ti, "__getitem__") else 1
                                    cond_name = cond_names[min(cond_idx, len(cond_names) - 1)] if cond_idx <= 3 else cond_names[0]
                                    vel_means.setdefault(cond_name, []).append(float(np.nanmean(v)))
                        except ImportError:
                            pass
                else:
                    try:
                        td = np.atleast_1d(td)
                        for r in td:
                            c = getattr(r, "Condition", r.get("Condition", 1))
                            cn = cond_names[min(int(c) - 1, len(cond_names) - 1)] if c <= 3 else cond_names[0]
                            # Velocity not stored; skip for now
                            pass
                    except (AttributeError, TypeError, IndexError):
                        pass
            if vel_means and any(vel_means.values()):
                out_means = {}
                out_sds = {}
                out_ind = {}
                for c in cond_names:
                    v = vel_means.get(c, [])
                    if v:
                        out_means[c] = float(np.nanmean(v))
                        out_sds[c] = float(np.nanstd(v))
                        out_ind[c] = v
                if out_means:
                    out = {
                        "metric": "gaze_velocity",
                        "unit": "px/s",
                        "conditions": list(out_means.keys()),
                        "means": list(out_means.values()),
                        "sds": list(out_sds.values()),
                        "individual_subject_means": out_ind,
                    }
                    out_path = out_dir / f"{task}_gaze_velocity_{proc}.json"
                    kb = _write_json(out_path, out)
                    summary.append((out_path.name, "velocity", kb))


def _export_behavior(feat_path: Path, out_dir: Path, summary: list) -> None:
    """Export behavior metrics (accuracy, reaction time) from trial-level files."""
    for task in ["nback", "sternberg"]:
        trial_candidates = [
            feat_path / f"behav_matrix_{task}_trials.mat",
            feat_path / f"behavior_matrix_{task}_trials.mat",
            feat_path / f"behavioral_matrix_{task}_trials.mat",
            feat_path / f"behav_data_{task}_trials.mat",
            feat_path / f"behavioral_matrix_{task}.mat",
            feat_path / f"behavior_matrix_{task}.mat",
        ]
        trial_file = next((p for p in trial_candidates if p.exists()), None)
        if trial_file is None:
            print(f"  Warning: missing trial-level behavior file for {task}")
            continue
        d = _load_mat(trial_file)
        beh_trials = _pick_trial_struct(d, task, "behav")
        if beh_trials is None:
            beh_trials = _pick_trial_struct(d, task, "behavior")
        if beh_trials is None:
            print(f"  Warning: missing behavior trial struct in {trial_file.name}")
            continue
        rows = _mat_struct_rows(beh_trials)
        if not rows:
            continue
        cond_names = {"nback": ["1back", "2back", "3back"], "sternberg": ["2items", "4items", "6items"]}[task]
        cond_codes = {"nback": (1, 2, 3), "sternberg": (2, 4, 6)}[task]
        try:
            ids = np.asarray([_row_field(r, "ID") for r in rows], dtype=float)
            conds = np.asarray([_row_field(r, "Condition") for r in rows], dtype=float)
            acc = np.asarray([_row_field(r, "Accuracy") for r in rows], dtype=float)
            rt = np.asarray([_row_field(r, "ReactionTime") for r in rows], dtype=float)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue

        by_cond = {}
        for cn, code in zip(cond_names, cond_codes):
            mask = conds == code
            if mask.sum() == 0:
                continue
            cond_ids = ids[mask]
            acc_vals = acc[mask]
            rt_vals = rt[mask]
            acc_subj = []
            rt_subj = []
            for sid in np.unique(cond_ids[np.isfinite(cond_ids)]):
                smask = cond_ids == sid
                a = acc_vals[smask]
                a = a[np.isfinite(a)]
                if len(a) > 0:
                    acc_subj.append(float(np.nanmean(a)))
                r = rt_vals[smask]
                r = r[np.isfinite(r)]
                if len(r) > 0:
                    rt_subj.append(float(np.nanmean(r)))
            if len(acc_subj) > 0 or len(rt_subj) > 0:
                by_cond[cn] = {"accuracy": acc_subj, "reaction_time": rt_subj}
        if not by_cond:
            continue
        order = [c for c in cond_names if c in by_cond]
        out = {
            "conditions": order,
            "accuracy_mean": [float(np.nanmean(by_cond[c]["accuracy"])) if by_cond[c]["accuracy"] else None for c in order],
            "accuracy_sd": [float(np.nanstd(by_cond[c]["accuracy"])) if by_cond[c]["accuracy"] else None for c in order],
            "reaction_time_mean": [float(np.nanmean(by_cond[c]["reaction_time"])) if by_cond[c]["reaction_time"] else None for c in order],
            "reaction_time_sd": [float(np.nanstd(by_cond[c]["reaction_time"])) if by_cond[c]["reaction_time"] else None for c in order],
            "individual_subject_means": {
                "accuracy": {c: by_cond[c]["accuracy"] for c in order},
                "reaction_time": {c: by_cond[c]["reaction_time"] for c in order},
            },
        }
        out_path = out_dir / f"{task}_behavior_raw.json"
        kb = _write_json(out_path, out)
        summary.append((out_path.name, str(order), kb))


# ─────────────────────────────────────────────────────────────────────────────
# TFR — time-frequency representation
# ─────────────────────────────────────────────────────────────────────────────
def _export_tfr(feat_path: Path, out_dir: Path, summary: list) -> None:
    """Export TFR (raw, baselined, parameterized) per task."""
    subjects = _get_subjects(feat_path)
    if not subjects:
        return
    for task, file_name, cond_vars in [
        ("nback", "tfr_nback.mat", ["tfr1", "tfr2", "tfr3"]),
        ("sternberg", "tfr_stern.mat", ["tfr2", "tfr4", "tfr6"]),
    ]:
        cond_names = {"nback": ["1back", "2back", "3back"], "sternberg": ["2items", "4items", "6items"]}[task]
        for proc, suffix in [("raw", ""), ("baselined", "_bl"), ("parameterized", "_fooof_bl")]:
            all_cond = []
            times, freqs = None, None
            for subj in subjects:
                p = feat_path / subj / "eeg" / file_name
                d = _load_mat(p)
                if d is None:
                    print(f"  Warning: missing {p}")
                    continue
                for vi, vn in enumerate(cond_vars):
                    key = vn + suffix if suffix else vn
                    if key not in d:
                        continue
                    tfr = d[key]
                    if hasattr(tfr, "powspctrm"):
                        pwr = np.asarray(tfr.powspctrm)
                    else:
                        pwr = np.asarray(tfr) if not hasattr(tfr, "dtype") or not (hasattr(tfr, "dtype") and tfr.dtype.names) else np.asarray(tfr["powspctrm"])
                    if hasattr(tfr, "label"):
                        labels = np.asarray(tfr.label).flatten()
                    else:
                        labels = []
                    if hasattr(tfr, "freq"):
                        fq = np.asarray(tfr.freq).flatten()
                    else:
                        fq = np.linspace(4, 30, 27)
                    if hasattr(tfr, "time"):
                        tm = np.asarray(tfr.time).flatten()
                    else:
                        tm = np.linspace(-0.5, 2, 50)
                    idx = np.array([i for i, l in enumerate(labels) if _is_occipital(str(l))]) if len(labels) > 0 else np.arange(pwr.shape[1])[:5]
                    if len(idx) == 0:
                        idx = np.arange(min(5, pwr.shape[1]))
                    # pwr: rpt x chan x freq x time
                    if pwr.ndim == 4:
                        m = np.nanmean(pwr[:, idx, :, :], axis=(0, 1))
                    elif pwr.ndim == 3:
                        m = np.nanmean(pwr[:, idx, :], axis=(0, 1))
                    else:
                        m = np.nanmean(pwr, axis=0)
                    all_cond.append((cond_names[vi], m))
                    if times is None:
                        times = tm.tolist()
                    if freqs is None:
                        freqs = fq.tolist()
            if not all_cond:
                continue
            # Average across subjects
            by_cond = {}
            for cn, m in all_cond:
                by_cond.setdefault(cn, []).append(m)
            conditions = {}
            for cn, mats in by_cond.items():
                conditions[cn] = np.nanmean(mats, axis=0).tolist()
            out = {
                "times": times,
                "freqs": freqs,
                "conditions": conditions,
                "analysis_window": [0.0, 2.0],
                "alpha_band": [8, 14],
            }
            out_path = out_dir / f"{task}_tfr_{proc}.json"
            kb = _write_json(out_path, out)
            summary.append((out_path.name, "TFR", kb))


# ─────────────────────────────────────────────────────────────────────────────
# Topography
# ─────────────────────────────────────────────────────────────────────────────
def _export_topo(feat_path: Path, layout_path: Path, out_dir: Path, summary: list) -> None:
    """Export topography (alpha power per channel) per task."""
    layout_data = _load_mat(layout_path) if layout_path and layout_path.exists() else None
    if layout_data is None:
        print("  Warning: layout file not found, skipping topography")
        return
    layout = layout_data.get("layANThead", layout_data.get(list(layout_data.keys())[0] if layout_data else None))
    if layout is None:
        return
    if hasattr(layout, "pos"):
        pos = np.asarray(layout.pos)
    else:
        pos = np.asarray(layout["pos"])
    if hasattr(layout, "label"):
        labels = [str(l) for l in np.asarray(layout.label).flatten()]
    else:
        labels = [str(l) for l in np.asarray(layout["label"]).flatten()]
    channel_positions = {labels[i]: [float(pos[i, 0]), float(pos[i, 1])] for i in range(min(len(labels), len(pos)))}
    channel_names = labels[: len(channel_positions)]

    for task, file_suffix, cond_keys in [
        ("nback", "power_nback.mat", [("powload1", "1back"), ("powload2", "2back"), ("powload3", "3back")]),
        ("sternberg", "power_stern_raw.mat", [("powload2", "2items"), ("powload4", "4items"), ("powload6", "6items")]),
    ]:
        for proc, file_override in [("raw", None), ("baselined", "power_nback_fooof.mat" if task == "nback" else "power_stern_fooof.mat"), ("parameterized", None)]:
            if proc == "baselined":
                fn = file_override
            elif task == "sternberg" and proc == "raw":
                fn = "power_stern_raw.mat"
            else:
                fn = file_suffix
            conditions = {}
            for subj in _get_subjects(feat_path):
                p = feat_path / subj / "eeg" / fn
                d = _load_mat(p)
                if d is None:
                    continue
                for key, cname in cond_keys:
                    if proc != "raw":
                        key = key.replace("powload", "pow") + ("_fooof_bl" if proc == "baselined" else "_fooof")
                        if key not in d:
                            key = key.replace("pow_", "powload") if "powload" not in key else key
                    if key not in d:
                        continue
                    arr = d[key]
                    if hasattr(arr, "powspctrm"):
                        pwr = np.asarray(arr.powspctrm)
                    else:
                        pwr = np.asarray(arr)
                    if hasattr(arr, "freq"):
                        fq = np.asarray(arr.freq).flatten()
                    else:
                        fq = np.linspace(4, 30, 27)
                    alpha_idx = (fq >= 8) & (fq <= 14)
                    if pwr.ndim == 2:
                        alpha_power = np.nanmean(pwr[:, alpha_idx], axis=1)
                    else:
                        alpha_power = np.nanmean(pwr[..., alpha_idx], axis=-1)
                    conditions.setdefault(cname, []).append(alpha_power.tolist())
            if not conditions:
                continue
            # Average across subjects
            out_conds = {}
            for c, vals in conditions.items():
                if vals:
                    out_conds[c] = np.nanmean(vals, axis=0).tolist()
            out = {
                "channel_names": channel_names,
                "channel_positions": channel_positions,
                "conditions": out_conds,
                "alpha_band": [8, 14],
            }
            out_path = out_dir / f"{task}_topo_{proc}.json"
            kb = _write_json(out_path, out)
            summary.append((out_path.name, "topo", kb))


# ─────────────────────────────────────────────────────────────────────────────
# ERP
# ─────────────────────────────────────────────────────────────────────────────
def _export_erp(feat_path: Path, out_dir: Path, summary: list) -> None:
    """Export ERP (raw, baselined) per task."""
    for task, cond_keys in [
        ("nback", [("erp1", "1back"), ("erp2", "2back"), ("erp3", "3back")]),
        ("sternberg", [("erp2", "2items"), ("erp4", "4items"), ("erp6", "6items")]),
    ]:
        for proc in ["raw", "baselined"]:
            erp_file = "ERPs_nback.mat" if task == "nback" else "ERPs_sternberg.mat"
            all_times = None
            cond_means = {}
            cond_sem = {}
            for subj in _get_subjects(feat_path):
                p = feat_path / subj / "eeg" / erp_file
                d = _load_mat(p)
                if d is None:
                    continue
                for key, cname in cond_keys:
                    if key not in d:
                        continue
                    erp = d[key]
                    if isinstance(erp, (list, np.ndarray)) and len(erp) > 0:
                        erp = erp[0] if isinstance(erp[0], (dict, type(np.ndarray))) else erp
                    if hasattr(erp, "avg"):
                        avg = np.asarray(erp.avg)
                    elif hasattr(erp, "trial"):
                        avg = np.nanmean(erp.trial, axis=0)
                    else:
                        avg = np.asarray(erp)
                    if hasattr(erp, "time"):
                        tm = np.asarray(erp.time).flatten()
                    else:
                        tm = np.arange(avg.shape[-1]) / 500 - 0.5
                    if hasattr(erp, "label"):
                        labels = np.asarray(erp.label).flatten()
                    else:
                        labels = []
                    idx = np.array([i for i, l in enumerate(labels) if _is_occipital(str(l))]) if len(labels) > 0 else np.arange(avg.shape[0])[:5]
                    if len(idx) == 0 and avg.ndim > 1:
                        idx = np.arange(min(5, avg.shape[0]))
                    if avg.ndim == 2:
                        mean_erp = np.nanmean(avg[idx, :], axis=0)
                    else:
                        mean_erp = avg
                    cond_means.setdefault(cname, []).append(mean_erp)
                    if all_times is None:
                        all_times = tm.tolist()
            if not cond_means:
                continue
            out_conds = {}
            out_sem = {}
            for c, vals in cond_means.items():
                if vals:
                    v = np.array(vals)
                    out_conds[c] = np.nanmean(v, axis=0).tolist()
                    out_sem[c] = (np.nanstd(v, axis=0) / np.sqrt(len(v))).tolist()
            out = {"times": all_times, "conditions": out_conds, "sem": out_sem}
            out_path = out_dir / f"{task}_erp_{proc}.json"
            kb = _write_json(out_path, out)
            summary.append((out_path.name, "ERP", kb))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not HAS_SCIPY and not HAS_MAT73:
        print("Error: Need scipy or mat73 to load .mat files. pip install scipy [mat73]")
        sys.exit(1)

    feat_path = Path(PATHS["feat_path"])
    out_dir = Path(PATHS["out_dir"])
    _ensure_dir(out_dir)

    summary = []
    print("AOC Export for Website")
    print("-" * 50)

    # Spectrum
    print("Exporting power spectra...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _export_spectrum_nback(feat_path, out_dir, summary)
        _export_spectrum_sternberg(feat_path, out_dir, summary)

    # Gaze
    print("Exporting gaze metrics...")
    _export_gaze(feat_path, out_dir, summary)

    # Behavior
    print("Exporting behavior metrics...")
    _export_behavior(feat_path, out_dir, summary)
    _write_data_manifest(out_dir)

    # Summary
    print("-" * 50)
    print("Summary:")
    for name, shape, kb in summary:
        print(f"  {name}: shape ~{shape}, {kb} KB")
    if not summary:
        print("  No files exported. Check feat_path and that .mat files exist.")
    else:
        print(f"\nWrote {len(summary)} files to {out_dir}")


if __name__ == "__main__":
    main()
