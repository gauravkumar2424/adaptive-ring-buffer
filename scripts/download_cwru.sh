#!/usr/bin/env bash
# ============================================================
# download_cwru.sh -- fetch the CWRU 12k Drive End + Normal Baseline set
#
# File numbers verified 2026-08-11 against:
#   https://engineering.case.edu/bearingdatacenter/normal-baseline-data
#   https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data
#
# CORE SET (40 files) = 10 conditions x 4 motor loads (0,1,2,3 hp).
# Balanced factorial design. The four loads are INDEPENDENT recordings
# of the same fault, which is what makes recording-level CV possible
# (train on 3 loads, test on the held-out load) instead of the random
# window-level CV that leaks in most CWRU papers.
#
# EXTENDED SET (+24) adds OR@3:00, OR@12:00 and the 0.028" faults.
# Unbalanced (several cells marked "data not available" on the site),
# so keep it out of the primary analysis.
#
# Usage:
#   ./download_cwru.sh          # core 40
#   ./download_cwru.sh --all    # core + extended (64)
# ============================================================

set -u
BASE="https://engineering.case.edu/sites/default/files"
OUT="$(cd "$(dirname "$0")/.." && pwd)/data/cwru-bearing/raw"
mkdir -p "$OUT"

# name:filenumber
CORE=(
  # --- Normal baseline ---
  "Normal_0:97"   "Normal_1:98"   "Normal_2:99"   "Normal_3:100"
  # --- Inner race ---
  "IR007_0:105"   "IR007_1:106"   "IR007_2:107"   "IR007_3:108"
  "IR014_0:169"   "IR014_1:170"   "IR014_2:171"   "IR014_3:172"
  "IR021_0:209"   "IR021_1:210"   "IR021_2:211"   "IR021_3:212"
  # --- Ball ---
  "B007_0:118"    "B007_1:119"    "B007_2:120"    "B007_3:121"
  "B014_0:185"    "B014_1:186"    "B014_2:187"    "B014_3:188"
  "B021_0:222"    "B021_1:223"    "B021_2:224"    "B021_3:225"
  # --- Outer race, load zone centred @6:00 ---
  "OR007@6_0:130" "OR007@6_1:131" "OR007@6_2:132" "OR007@6_3:133"
  "OR014@6_0:197" "OR014@6_1:198" "OR014@6_2:199" "OR014@6_3:200"
  "OR021@6_0:234" "OR021@6_1:235" "OR021@6_2:236" "OR021@6_3:237"
)

EXTENDED=(
  # NOTE: OR007@12_1 is 158, NOT 157. The site skips 157.
  "OR007@3_0:144"  "OR007@3_1:145"  "OR007@3_2:146"  "OR007@3_3:147"
  "OR007@12_0:156" "OR007@12_1:158" "OR007@12_2:159" "OR007@12_3:160"
  "OR021@3_0:246"  "OR021@3_1:247"  "OR021@3_2:248"  "OR021@3_3:249"
  "OR021@12_0:258" "OR021@12_1:259" "OR021@12_2:260" "OR021@12_3:261"
  "IR028_0:3001"   "IR028_1:3002"   "IR028_2:3003"   "IR028_3:3004"
  "B028_0:3005"    "B028_1:3006"    "B028_2:3007"    "B028_3:3008"
)

LIST=("${CORE[@]}")
if [ "${1:-}" = "--all" ]; then LIST+=("${EXTENDED[@]}"); fi

echo "Destination: $OUT"
echo "Files to fetch: ${#LIST[@]}"
echo

ok=0; fail=0; skip=0
FAILED=""

for entry in "${LIST[@]}"; do
    name="${entry%%:*}"
    num="${entry##*:}"
    dest="$OUT/${num}_${name}.mat"

    if [ -s "$dest" ]; then
        sz=$(stat -c%s "$dest" 2>/dev/null || echo 0)
        if [ "$sz" -gt 100000 ]; then
            printf "  [have] %-14s %s (%s bytes)\n" "$name" "$num.mat" "$sz"
            skip=$((skip+1)); continue
        fi
        rm -f "$dest"   # too small to be real; refetch
    fi

    printf "  [get ] %-14s %s ... " "$name" "$num.mat"
    if wget -q --tries=3 --timeout=45 -O "$dest" "$BASE/${num}.mat"; then
        sz=$(stat -c%s "$dest" 2>/dev/null || echo 0)
        # A real recording is >100 KB. An HTML error page is a few KB.
        if [ "$sz" -gt 100000 ]; then
            echo "ok ($sz bytes)"; ok=$((ok+1))
        else
            echo "FAIL (only $sz bytes -- likely an error page)"
            rm -f "$dest"; fail=$((fail+1)); FAILED="$FAILED $num($name)"
        fi
    else
        echo "FAIL (network/404)"
        rm -f "$dest"; fail=$((fail+1)); FAILED="$FAILED $num($name)"
    fi
    sleep 0.4   # be polite to the server
done

echo
echo "============================================"
echo "  downloaded: $ok    already present: $skip    failed: $fail"
[ -n "$FAILED" ] && echo "  failed files:$FAILED"
echo "  total .mat in raw/: $(ls -1 "$OUT"/*.mat 2>/dev/null | wc -l)"
echo "  disk: $(du -sh "$OUT" 2>/dev/null | cut -f1)"
echo "============================================"
echo
echo "Next:  python3 scripts/extract_cwru.py"
