#!/usr/bin/env python3
"""
Adds the Visvalingam-Whyatt triangle-area baseline mode to ring_buffer.h.

Run from ~/DSA/adaptive-ring-buffer:
    python3 apply_vw_patch.py

Creates a backup at src/ring_buffer.h.bak before editing.
Safe to re-run: detects if the patch is already applied and stops.
"""

import os
import shutil
import sys
from pathlib import Path

HEADER = Path(os.path.expanduser("~/DSA/adaptive-ring-buffer/src/ring_buffer.h"))
BACKUP = HEADER.with_suffix(".h.bak")

# ── Patch 1: enum ────────────────────────────────────────────────────
ENUM_OLD = "    IMPORTANCE_INTERP_LOOKAHEAD   // O(N) — error with cascade penalty"
ENUM_NEW = """    IMPORTANCE_INTERP_LOOKAHEAD,  // O(N) — error with cascade penalty

    // === V-W BASELINE ADDITION ===
    // Classical Visvalingam-Whyatt effective-area importance.
    // Included as a baseline: V-W (1993) is the standard heap-based
    // line-simplification method; we compare its triangle-area score
    // against direct interpolation error on 1-D sensor streams.
    IMPORTANCE_VW_AREA            // O(N) — V-W triangle area"""

# ── Patch 2: scorer method ───────────────────────────────────────────
# Anchor: the closing of compute_hypothetical_ie, followed by the
# separator comment before compute_local_freq_weight.
SCORER_ANCHOR = """    // ========================================================
    // Local frequency content estimator — O(W) per sample
    // ========================================================"""

SCORER_NEW = """    // ========================================================
    // === V-W BASELINE ADDITION ===
    // Visvalingam-Whyatt effective area — O(1)
    //
    // The classical V-W importance measure: area of the triangle
    // formed by a point and its two neighbours. For a 1-D time
    // series we treat (original_index, value) as 2-D coordinates,
    // matching Visvalingam & Whyatt (Cartographic Journal, 1993).
    //
    //   Area = 0.5 * | t_p*(x_i - x_s) + t_i*(x_s - x_p)
    //                                  + t_s*(x_p - x_i) |
    //
    // Scale note: area grows with the time span between neighbours,
    // so V-W implicitly favours evicting points in temporally dense
    // regions, whereas interpolation error is span-normalised.
    // ========================================================
    double compute_vw_area(int slot) const {
        int p = nodes_[slot].prev;
        int s = nodes_[slot].next;
        if (p < 0 || s < 0)
            return std::numeric_limits<double>::max();
        double x_p = static_cast<double>(nodes_[p].value);
        double x_i = static_cast<double>(nodes_[slot].value);
        double x_s = static_cast<double>(nodes_[s].value);
        double t_p = static_cast<double>(nodes_[p].original_index);
        double t_i = static_cast<double>(nodes_[slot].original_index);
        double t_s = static_cast<double>(nodes_[s].original_index);
        return 0.5 * std::abs(t_p * (x_i - x_s)
                            + t_i * (x_s - x_p)
                            + t_s * (x_p - x_i));
    }
    // ========================================================
    // Local frequency content estimator — O(W) per sample
    // ========================================================"""

# ── Patch 3a: isInterpMode ───────────────────────────────────────────
MODE_OLD = "               mode_ == BufferMode::IMPORTANCE_INTERP_LOOKAHEAD;  // === LOOKAHEAD ADDITION ==="
MODE_NEW = """               mode_ == BufferMode::IMPORTANCE_VW_AREA ||        // === V-W ADDITION ===
               mode_ == BufferMode::IMPORTANCE_INTERP_LOOKAHEAD;  // === LOOKAHEAD ADDITION ==="""

# ── Patch 3b: eviction branch ────────────────────────────────────────
BRANCH_ANCHOR = """            else if (mode_ == BufferMode::IMPORTANCE_INTERP_SPECTRAL) {"""

BRANCH_NEW = """            else if (mode_ == BufferMode::IMPORTANCE_VW_AREA) {
                // === V-W BASELINE ADDITION ===
                // Classical Visvalingam-Whyatt: evict minimum triangle area
                while (cur >= 0 && cur != search_end) {
                    double area = compute_vw_area(cur);
                    if (area < min_score) {
                        min_score = area;
                        evict_slot = cur;
                    }
                    cur = nodes_[cur].next;
                }
            }
            // === END V-W ADDITION ===
            else if (mode_ == BufferMode::IMPORTANCE_INTERP_SPECTRAL) {"""


def fail(msg):
    print(f"  FAILED: {msg}")
    print("\n  No changes written. Paste this output to mentor.")
    sys.exit(1)


def main():
    print("=" * 68)
    print("V-W BASELINE PATCH — ring_buffer.h")
    print("=" * 68)

    if not HEADER.exists():
        fail(f"{HEADER} not found")

    text = HEADER.read_text()

    if "IMPORTANCE_VW_AREA" in text:
        print("\n  Patch already applied (IMPORTANCE_VW_AREA found). Nothing to do.")
        return

    # Verify every anchor exists before touching anything
    checks = [
        (ENUM_OLD, "enum entry IMPORTANCE_INTERP_LOOKAHEAD"),
        (SCORER_ANCHOR, "compute_local_freq_weight separator comment"),
        (MODE_OLD, "isInterpMode LOOKAHEAD line"),
        (BRANCH_ANCHOR, "INTERP_SPECTRAL eviction branch"),
    ]
    for needle, label in checks:
        if needle not in text:
            fail(f"anchor not found — {label}")
        if text.count(needle) != 1:
            fail(f"anchor appears {text.count(needle)} times (need exactly 1) — {label}")

    print("\n  All 4 anchors located and unique.")

    shutil.copy2(HEADER, BACKUP)
    print(f"  Backup written: {BACKUP}")

    text = text.replace(ENUM_OLD, ENUM_NEW)
    print("  [1/4] enum: added IMPORTANCE_VW_AREA")

    text = text.replace(SCORER_ANCHOR, SCORER_NEW)
    print("  [2/4] added compute_vw_area()")

    text = text.replace(MODE_OLD, MODE_NEW)
    print("  [3/4] isInterpMode(): registered V-W mode")

    text = text.replace(BRANCH_ANCHOR, BRANCH_NEW)
    print("  [4/4] eviction: added V-W scan branch")

    HEADER.write_text(text)

    print(f"\n  Wrote {HEADER}")
    print("\n" + "=" * 68)
    print("NEXT: register the mode name in your experiment driver.")
    print("Run this to find where modes are listed:")
    print()
    print('  grep -rn "IMP_INTERP_SPECTRAL" ~/DSA/adaptive-ring-buffer/src/*.cpp')
    print()
    print("Paste that output to mentor.")
    print("=" * 68)


if __name__ == "__main__":
    main()
