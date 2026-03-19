#!/usr/bin/env python3
"""
Validate AOC exported JSON files for expected presence and minimal schema.

Run from repo root:
  python export-scripts/validate_aoc_exports.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "resources" / "AOC" / "data"

TASKS = ["nback", "sternberg"]
PROCS = ["raw", "baselined", "parameterized"]
SPECTRUM = "spectrum"
GAZE_METRICS = ["gaze_deviation", "spl", "bcea"]
BEHAVIOR = "behavior"


def _load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require(path: Path, missing: list[str]) -> bool:
    if not path.exists():
        missing.append(path.name)
        return False
    return True


def main() -> int:
    if not DATA_DIR.exists():
        print(f"ERROR: Missing data folder: {DATA_DIR}")
        return 1

    missing: list[str] = []
    invalid: list[str] = []

    # Spectrum
    for task in TASKS:
        for proc in PROCS:
            p = DATA_DIR / f"{task}_{SPECTRUM}_{proc}.json"
            if not _require(p, missing):
                continue
            try:
                d = _load(p)
                if not isinstance(d.get("freqs"), list) or not isinstance(d.get("conditions"), dict):
                    invalid.append(p.name)
            except Exception:
                invalid.append(p.name)

    # Gaze
    for task in TASKS:
        for metric in GAZE_METRICS:
            for proc in ["raw", "baselined"]:
                p = DATA_DIR / f"{task}_{metric}_{proc}.json"
                if not _require(p, missing):
                    continue
                try:
                    d = _load(p)
                    if not isinstance(d.get("conditions"), list) or not isinstance(d.get("means"), list):
                        invalid.append(p.name)
                except Exception:
                    invalid.append(p.name)

    # Behavior
    for task in TASKS:
        p = DATA_DIR / f"{task}_{BEHAVIOR}_raw.json"
        if not _require(p, missing):
            continue
        try:
            d = _load(p)
            need = ["conditions", "accuracy_mean", "reaction_time_mean"]
            if any(k not in d for k in need):
                invalid.append(p.name)
        except Exception:
            invalid.append(p.name)

    # Data manifest
    manifest = DATA_DIR / "data_manifest.json"
    if not _require(manifest, missing):
        pass
    else:
        try:
            d = _load(manifest)
            if not isinstance(d.get("files"), list):
                invalid.append(manifest.name)
        except Exception:
            invalid.append(manifest.name)

    if missing:
        print("Missing files:")
        for f in sorted(set(missing)):
            print(f"  - {f}")
    if invalid:
        print("Invalid schema/files:")
        for f in sorted(set(invalid)):
            print(f"  - {f}")

    if missing or invalid:
        print("\nValidation: FAILED")
        return 1

    print("Validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
