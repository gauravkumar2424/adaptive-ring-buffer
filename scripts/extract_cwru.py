#!/usr/bin/env python3
"""
extract_cwru.py -- turn downloaded CWRU .mat files into the plain-text
signals the C++ harness reads, plus a metadata manifest.

Outputs
  data/cwru-bearing/vib_<NAME>.txt   one float per line, drive-end channel
  data/cwru-bearing/manifest.csv     condition, load, rpm, fs, n, file

WHY A MANIFEST
  The downstream fault-frequency metric needs shaft rate and sampling
  rate per recording, and the classification CV needs the load label so
  splits can be made at RECORDING level (train on 3 loads, test on the
  held-out load) rather than at window level. Window-level random CV on
  CWRU leaks -- windows from one recording land in both train and test
  -- which is a documented pathology and a standing reviewer objection.

SAMPLING RATE CAVEAT  (read this)
  Fault files here are the 12 kHz drive-end set: fs = 12000, confirmed
  by the source page title.
  The Normal Baseline files (97-100) are NOT labelled with a rate on the
  site. They are commonly cited as 48 kHz, and their sample counts are
  roughly 2x the fault files, which is consistent with that. This script
  reports the counts and flags the assumption rather than burying it,
  because fs enters the fault-frequency calculation directly. VERIFY
  before using normal-baseline recordings in any spectral claim.

BEARING GEOMETRY (drive end: SKF 6205-2RS JEM)
  Characteristic-defect multipliers, in multiples of shaft rate:
      BPFI 5.4152   BPFO 3.5848   BSF 2.3568   FTF 0.3983
  These are the values published with the dataset. Re-derive from the
  bearing spec sheet before putting them in the paper.
"""

import os
import re
import csv
import glob
import sys

try:
    import scipy.io as sio
except ImportError:
    sys.exit("scipy required:  pip3 install scipy --break-system-packages")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "cwru-bearing", "raw")
OUT = os.path.join(ROOT, "data", "cwru-bearing")

RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
MULT = dict(BPFI=5.4152, BPFO=3.5848, BSF=2.3568, FTF=0.3983)


def parse_name(stem):
    """'105_IR007_0' -> ('IR007_0', 'inner_race', 0.007, 0)"""
    m = re.match(r"^(\d+)_(.+)$", stem)
    if not m:
        return None
    num, name = m.group(1), m.group(2)

    load = int(name.rsplit("_", 1)[1]) if re.search(r"_\d$", name) else -1

    if name.startswith("Normal"):
        cond, diam = "normal", 0.0
    elif name.startswith("IR"):
        cond, diam = "inner_race", int(re.search(r"IR(\d+)", name).group(1)) / 1000
    elif name.startswith("B"):
        cond, diam = "ball", int(re.search(r"B(\d+)", name).group(1)) / 1000
    elif name.startswith("OR"):
        cond, diam = "outer_race", int(re.search(r"OR(\d+)", name).group(1)) / 1000
    else:
        cond, diam = "unknown", 0.0
    return name, cond, diam, load, num


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*.mat")))
    if not files:
        sys.exit(f"No .mat files in {RAW}\nRun scripts/download_cwru.sh first.")

    print(f"Found {len(files)} .mat files\n")
    rows, warnings = [], []

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        parsed = parse_name(stem)
        if not parsed:
            warnings.append(f"cannot parse filename: {stem}")
            continue
        name, cond, diam, load, num = parsed

        mat = sio.loadmat(path)
        keys = [k for k in mat if not k.startswith("__")]

        de = [k for k in keys if re.search(r"DE_time$", k)]
        if not de:
            warnings.append(f"{stem}: no *_DE_time variable (has: {keys})")
            continue
        sig = mat[de[0]].ravel()

        rpm_keys = [k for k in keys if k.endswith("RPM")]
        rpm = float(mat[rpm_keys[0]].ravel()[0]) if rpm_keys else RPM_BY_LOAD.get(load, 0)

        # See SAMPLING RATE CAVEAT in the docstring.
        fs = 48000 if cond == "normal" else 12000
        assumed = "ASSUMED-VERIFY" if cond == "normal" else "from-source-page"

        txt = os.path.join(OUT, f"vib_{name}.txt")
        with open(txt, "w") as f:
            f.write("\n".join(f"{v:.6f}" for v in sig))
            f.write("\n")

        shaft = rpm / 60.0
        rows.append(dict(
            name=name, file=os.path.basename(path), condition=cond,
            diameter_in=diam, load_hp=load, rpm=rpm, shaft_hz=round(shaft, 4),
            fs_hz=fs, fs_provenance=assumed, n_samples=len(sig),
            duration_s=round(len(sig) / fs, 3),
            BPFI_hz=round(shaft * MULT["BPFI"], 3),
            BPFO_hz=round(shaft * MULT["BPFO"], 3),
            BSF_hz=round(shaft * MULT["BSF"], 3),
            FTF_hz=round(shaft * MULT["FTF"], 3),
            txt=os.path.relpath(txt, ROOT),
        ))
        print(f"  {name:<14} {cond:<11} load={load} rpm={rpm:>6.0f} "
              f"n={len(sig):>7d}  var={de[0]}")

    if not rows:
        sys.exit("\nNothing extracted.")

    man = os.path.join(OUT, "manifest.csv")
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} signals + {os.path.relpath(man, ROOT)}")

    by_cond, by_load = {}, {}
    for r in rows:
        by_cond[r["condition"]] = by_cond.get(r["condition"], 0) + 1
        by_load[r["load_hp"]] = by_load.get(r["load_hp"], 0) + 1
    print("\n  by condition:", dict(sorted(by_cond.items())))
    print("  by load:     ", dict(sorted(by_load.items())))

    lens = sorted({r["n_samples"] for r in rows})
    print(f"\n  sample counts: min={lens[0]} max={lens[-1]} distinct={len(lens)}")
    print("  -> if normal-baseline counts are ~2x the fault files, the 48 kHz")
    print("     assumption is consistent. Confirm before any spectral claim.")

    if warnings:
        print("\nWARNINGS:")
        for w_ in warnings:
            print("  -", w_)


if __name__ == "__main__":
    main()
