#!/usr/bin/env python3
"""
apply_span_patch.py  --  S5: add the span-weighted criterion family.

Adds BufferMode::IMPORTANCE_INTERP_SPAN, scoring

    score(i) = |x_i - xhat_i| * (t_s - t_p)^beta

ANALYTIC BASIS
--------------
Expanding the Visvalingam-Whyatt triangle area for a 1-D series with
(index, value) coordinates gives EXACTLY

    A_i = 0.5 * |t_p(x_i-x_s) + t_i(x_s-x_p) + t_s(x_p-x_i)|
        = 0.5 * (t_s - t_p) * |x_i - xhat_i|

so V-W area is interpolation error weighted by neighbour span. Because
eviction is an argmin, the constant 0.5 cancels. Therefore:

    beta = 0  ==  IMPORTANCE_INTERP_ERROR   (bit-identical decisions)
    beta = 1  ==  IMPORTANCE_VW_AREA        (bit-identical decisions)

The two "competing criteria" in Progress Reports v2/v3 are one criterion
at two points of a continuum. span_sweep.cpp verifies both endpoints
reproduce their reference mode exactly; if they do not, this patch is
wrong and the sweep aborts.

Idempotent: safe to run more than once. Writes .bak_span backups.
"""

import re, sys, shutil, os

SRC = os.path.dirname(os.path.abspath(__file__))
RB  = os.path.join(SRC, "ring_buffer.h")
IM  = os.path.join(SRC, "importance_metrics.h")

# ---------------------------------------------------------------- 1. enum
ENUM_RE = re.compile(
    r"(    IMPORTANCE_VW_AREA[^\n]*\n)(\};)", re.M)
ENUM_NEW = (
    r"\1"
    "\n"
    "    // === SPAN-WEIGHTED FAMILY (S5) ===\n"
    "    // score(i) = |x_i - xhat_i| * (t_s - t_p)^beta\n"
    "    // Since V-W area == 0.5 * span * interpolation error exactly,\n"
    "    // beta=0 reduces to IMPORTANCE_INTERP_ERROR and beta=1 to\n"
    "    // IMPORTANCE_VW_AREA. Interpolates and extrapolates between them.\n"
    "    IMPORTANCE_INTERP_SPAN        // O(N) - ie * span^beta\n"
    r"\2")

# ------------------------------------------------------- 2. isInterpMode()
INTERP_ANCHOR = ("               mode_ == BufferMode::IMPORTANCE_INTERP_LOOKAHEAD;"
                 "  // === LOOKAHEAD ADDITION ===")
INTERP_NEW = ("               mode_ == BufferMode::IMPORTANCE_INTERP_SPAN ||"
              "        // === SPAN ADDITION ===\n"
              "               mode_ == BufferMode::IMPORTANCE_INTERP_LOOKAHEAD;"
              "  // === LOOKAHEAD ADDITION ===")

# --------------------------------------------------- 3. eviction branch
BRANCH_ANCHOR = "            else if (mode_ == BufferMode::IMPORTANCE_VW_AREA) {"
BRANCH_NEW = """            // === SPAN-WEIGHTED FAMILY ADDITION (S5) ===
            else if (mode_ == BufferMode::IMPORTANCE_INTERP_SPAN) {
                // score = ie * span^beta.
                // beta=0 -> INTERP_ERROR, beta=1 -> VW_AREA (exact).
                // Boundary nodes score +max, matching every other
                // interp mode, so the fallback to head_ is unchanged.
                double beta = imp_config_.span_beta;
                while (cur >= 0 && cur != search_end) {
                    int p = nodes_[cur].prev;
                    int s = nodes_[cur].next;
                    double score;
                    if (p < 0 || s < 0) {
                        score = std::numeric_limits<double>::max();
                    } else {
                        double ie = compute_interp_error(cur);
                        double span = static_cast<double>(
                            nodes_[s].original_index - nodes_[p].original_index);
                        if (span <= 0.0) {
                            score = ie;
                        } else if (beta == 0.0) {
                            score = ie;                    // exact, no pow()
                        } else if (beta == 1.0) {
                            score = ie * span;             // exact, no pow()
                        } else {
                            score = ie * std::pow(span, beta);
                        }
                    }
                    if (score < min_score) {
                        min_score = score;
                        evict_slot = cur;
                    }
                    cur = nodes_[cur].next;
                }
            }
            // === END SPAN ADDITION ===
""" + BRANCH_ANCHOR

# ------------------------------------------------------------ 4. config
CFG_ANCHOR = "    double lookahead_alpha = 0.5;"
CFG_NEW = ("    double span_beta = 0.0;        // === SPAN ADDITION === "
           "0=interp error, 1=V-W area\n"
           "    double lookahead_alpha = 0.5;")


def patch(path, edits):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    orig = txt
    for name, kind, a, b in edits:
        if kind == "re":
            if re.search(r"IMPORTANCE_INTERP_SPAN", txt):
                print(f"  [skip] {name}: already present")
                continue
            new, n = a.subn(b, txt)
            if n != 1:
                print(f"  [FAIL] {name}: anchor matched {n} times (need 1)")
                return False
            txt = new
        else:
            if b.split("\n")[0] in txt and "SPAN ADDITION" in txt and name != "enum":
                print(f"  [skip] {name}: already present")
                continue
            if txt.count(a) != 1:
                print(f"  [FAIL] {name}: anchor found {txt.count(a)} times (need 1)")
                return False
            txt = txt.replace(a, b, 1)
        print(f"  [ok]   {name}")
    if txt != orig:
        shutil.copy(path, path + ".bak_span")
        with open(path, "w", encoding="utf-8") as f:
            f.write(txt)
        print(f"  wrote {path} (backup: {path}.bak_span)")
    return True


def main():
    for p in (RB, IM):
        if not os.path.exists(p):
            print("ERROR: not found:", p)
            print("Run this from the src/ directory.")
            sys.exit(1)

    print("Patching ring_buffer.h ...")
    ok1 = patch(RB, [
        ("enum",         "re",  ENUM_RE,        ENUM_NEW),
        ("isInterpMode", "str", INTERP_ANCHOR,  INTERP_NEW),
        ("evict branch", "str", BRANCH_ANCHOR,  BRANCH_NEW),
    ])
    print("Patching importance_metrics.h ...")
    ok2 = patch(IM, [
        ("span_beta cfg", "str", CFG_ANCHOR, CFG_NEW),
    ])

    if ok1 and ok2:
        print("\nDone. Verify with:")
        print("  grep -n 'IMPORTANCE_INTERP_SPAN\\|span_beta' ring_buffer.h importance_metrics.h")
    else:
        print("\nPATCH FAILED - files unchanged where it failed. "
              "Send me the [FAIL] line.")
        sys.exit(1)


if __name__ == "__main__":
    main()
