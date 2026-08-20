#!/usr/bin/env bash
#
# run_gaps.sh -- the follow-up runs Section 6 still needs after the audit.
#
# Same contract as run_all.sh: nothing is overwritten, each run owns
# states/<TAG>/ and logs/<TAG>.log, a failure does not stop the queue, and the
# script is safe to re-run and safe to interrupt.
#
# Usage:
#   bash run_gaps.sh --dry-run     # print the plan, run nothing
#   bash run_gaps.sh required      # E7 repair + the a=0 family      (~25 min)
#   bash run_gaps.sh recommended   # DEIM rank isolation            (~15 min)
#   bash run_gaps.sh               # both
#   FORCE=1 bash run_gaps.sh ...   # re-run even where state dirs exist
#
# NOT included, on purpose:
#   * extending Re_max to reach Re_c(a) for a >= +0.7. Those bifurcations sit
#     near Re 140-155, outside the snapshot range, so the ROM would be
#     extrapolating and a fair run means regenerating the whole offline stage.
#     State the truncation instead.
#   * G2 mesh convergence and G1's full-order eigensweep. Both need code that
#     does not exist yet, not just an env override -- see the notes at the end.
#
set -u -o pipefail

cd "$(dirname "$0")" || exit 1

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
SUMMARY="${LOGDIR}/gaps_summary_${STAMP}.txt"
mkdir -p "$LOGDIR" states

TIER="${1:-all}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=1; TIER="all"; }
FORCE="${FORCE:-0}"

# Must match run_all.sh exactly, or the repaired leg is not comparable.
FULL_AMPS="-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.25,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
NEAR_AMPS="-0.5,-0.3,-0.1,0.1,0.3,0.5"
ZERO_AMP="0.0"


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
preflight() {
    local fail=0

    if ! python -c "import nsrom, nsrom.local_rom, nsrom.bifurcation.detection" 2>/dev/null; then
        echo "PREFLIGHT FAIL: nsrom does not import cleanly."
        fail=1
    fi
    if ! grep -q "NSROM_RUN_TAG" nsrom/workflows/local_pipeline.py; then
        echo "PREFLIGHT FAIL: nsrom/workflows/local_pipeline.py has no env overrides."
        fail=1
    fi

    # The E7 tol1e-4 pathology looks like two writers in one state dir (8398
    # points, 111 spine lines, zero divergences). Refuse to start if another
    # main_local.py is alive, or the repair could reproduce the fault.
    local n
    n=$(pgrep -fc "main_local.py" 2>/dev/null || echo 0)
    if [[ "$n" -gt 0 ]]; then
        echo "PREFLIGHT FAIL: ${n} main_local.py process(es) already running."
        echo "  The doubled E7 leg is consistent with concurrent writers."
        echo "  Wait for them to finish, then re-run this script."
        fail=1
    fi

    return $fail
}


# What is actually at a = 0 right now? This is the question that decides whether
# the a=0 family below is necessary; printed so the answer is on the record.
report_a0() {
    echo ""
    echo "--- what exists at a = 0 in E1 (and what the 22nd amplitude is) ---"
    python - <<'EOF' 2>/dev/null || echo "  (could not read the store)"
import json, collections, os
p = "states/E1_K4_tensor/index.jsonl"
if not os.path.exists(p):
    raise SystemExit("  no E1 index.jsonl")
recs = [json.loads(l) for l in open(p) if l.strip()]
amps = sorted({round(r["amp"], 6) for r in recs})
print(f"  {len(amps)} distinct amplitudes: {amps}")
z = [r for r in recs if abs(r["amp"]) < 1e-12]
print(f"  rows at |a| < 1e-12: {len(z)}")
if z:
    for b, n in collections.Counter(r["branch"] for r in z).most_common():
        conv = sum(1 for r in z if r["branch"] == b and r.get("converged"))
        fom = sum(1 for r in z if r["branch"] == b and r.get("C_L_fom") is not None)
        print(f"    {b:24s} n={n:4d} converged={conv:4d} with_C_L_fom={fom:4d}")
else:
    print("    -> a=0 is ABSENT; the a=0 family in this script is required")
EOF
    echo ""
}


# ---------------------------------------------------------------------------
run_case() {
    local tag="$1"; shift
    local desc="$1"; shift
    local state_dir="states/${tag}"
    local log="${LOGDIR}/${tag}.log"

    if [[ $DRY_RUN -eq 1 ]]; then
        printf '  %-26s %s\n' "$tag" "$desc"
        printf '      env: %s\n' "$*"
        return 0
    fi

    if [[ -d "$state_dir" && "$FORCE" != "1" ]]; then
        echo "[skip] ${tag} -- ${state_dir} exists (FORCE=1 to override)"
        echo "SKIP  ${tag}" >> "$SUMMARY"
        return 0
    fi
    if [[ -d "$state_dir" ]]; then
        mv "$state_dir" "${state_dir}.bak_${STAMP}"
        echo "[force] moved old ${state_dir} aside"
    fi
    [[ -f "$log" ]] && mv "$log" "${log}.bak_${STAMP}"

    echo ""
    echo "=============================================================="
    echo "[run ] ${tag} -- ${desc}"
    echo "       started $(date '+%F %T')"
    echo "=============================================================="

    local t0=$SECONDS
    env "$@" \
        NSROM_RUN_TAG="$tag" \
        NSROM_STATE_DIR="$state_dir" \
        python scripts/main_local.py > "$log" 2>&1
    local rc=$?
    local dt=$(( SECONDS - t0 ))

    if [[ $rc -eq 0 ]]; then
        local npts
        npts=$(wc -l < "${state_dir}/index.jsonl" 2>/dev/null || echo 0)
        echo "[done] ${tag}  ${dt}s  ${npts} points"
        printf 'OK    %-26s %6ss  %6s pts\n' "$tag" "$dt" "$npts" >> "$SUMMARY"
    else
        echo "[FAIL] ${tag}  rc=${rc}  after ${dt}s -- see ${log}"
        printf 'FAIL  %-26s %6ss  rc=%s\n' "$tag" "$dt" "$rc" >> "$SUMMARY"
        tail -n 25 "$log" | sed 's/^/       | /'
    fi
    return 0
}


echo "NS_ROM gap-closing runs   ${STAMP}" | tee -a "$SUMMARY"
echo "tier=${TIER}  force=${FORCE}  dry_run=${DRY_RUN}" | tee -a "$SUMMARY"

if [[ $DRY_RUN -eq 0 ]]; then
    if ! preflight; then
        echo ""; echo "Preflight failed. Nothing was run."; exit 1
    fi
    echo "preflight OK" | tee -a "$SUMMARY"
    report_a0
fi


# ===========================================================================
# REQUIRED
# ===========================================================================
if [[ "$TIER" == "all" || "$TIER" == "required" ]]; then
echo ""; echo "### REQUIRED ###"

# --- G-E7. Repair the doubled tolerance leg.
#     The audit found 8386 rows / 8398 points against ~4100-4295 elsewhere,
#     111 spine lines against 70, and zero divergences at the LOOSEST
#     tolerance -- implausible. fig:err_vs_N cannot use it. The POD/tensor
#     cache at this tolerance (r = 9,12,20,12) is fine and is reused, so this
#     is sweep time only. Set FORCE=1 to overwrite the bad directory.
run_case "E7_K4_tensor_tol1e-4" "K=4 tensor, POD tol=1e-4, FOM off -- REPAIR (~15 min)" \
    NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL=1e-4 \
    NSROM_AMPS="$FULL_AMPS" NSROM_FOM_COMPUTE=0

# --- G1/G2-adjacent. The a = 0 slice, at every POD tolerance.
#     Unblocks fig:fom_diagram (6.1), fig:rec_convergence (6.5) and the
#     tab:critcurve baseline. One amplitude column is ~1/21 of a full sweep,
#     so all four cost less than one E7 leg. FOM is tolerance-independent, so
#     it is computed once (in the tol=1e-8 run) and the rest replay against it.
run_case "E13_K4_a0_tol1e-8" "K=4 tensor, a=0 only, FOM ON  (~10 min)" \
    NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_AMPS="$ZERO_AMP" NSROM_FOM_COMPUTE=1 \
    NSROM_SAVE_FOM_FIELDS=1 NSROM_FOM_FIELD_STRIDE=1

for tol in 1e-4 1e-6 1e-10; do
    run_case "E13_K4_a0_tol${tol}" "K=4 tensor, a=0 only, POD tol=${tol}, FOM off (~2 min)" \
        NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL="$tol" \
        NSROM_AMPS="$ZERO_AMP" NSROM_FOM_COMPUTE=0
done

# Global counterpart at a = 0, for the local-vs-global panel at that slice.
run_case "E14_K1_a0_tol1e-8" "K=1 tensor (global), a=0 only, FOM off (~3 min)" \
    NSROM_K=1 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_AMPS="$ZERO_AMP" NSROM_FOM_COMPUTE=0
fi


# ===========================================================================
# RECOMMENDED
# ===========================================================================
if [[ "$TIER" == "all" || "$TIER" == "recommended" ]]; then
echo ""; echo "### RECOMMENDED ###"

# --- Isolate rank from grid in tab:hyperreduction.
#     E9 (DEIM max rank) used Re_max=105 and 6 amplitudes while E4 used 109 and
#     21, so part of E9's wider coverage is the easier grid. Running E9's exact
#     grid at tol=1e-8 makes the rank the only difference. The R3/R4 replays
#     already carry the main argument, so this closes a referee question rather
#     than establishing a result.
run_case "E15_K4_deim_tol8_near" "K=4 DEIM tol=1e-8 on E9's grid (~15 min)" \
    NSROM_K=4 NSROM_MODE=deim NSROM_POD_TOL=1e-8 \
    NSROM_DEIM_TOL=1e-8 NSROM_RECOMPUTE_DEIM=1 \
    NSROM_RE_MIN=70 NSROM_RE_MAX=105 NSROM_RE_STEP=1 \
    NSROM_AMPS="$NEAR_AMPS" NSROM_FOM_COMPUTE=0
fi


# ---------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "ALL DONE  $(date '+%F %T')"
echo "=============================================================="
cat "$SUMMARY"

if [[ $DRY_RUN -eq 0 ]]; then
cat <<'NOTES'

--- verify, then regenerate ---
  python scripts/audit_section6.py --repo-root .

  Check in the new report:
    * E7_K4_tensor_tol1e-4 back near ~4100-4300 points, NOT ~8400,
      with a nonzero divergence count and 21 amplitudes
    * E13_* each show one amplitude and ~89 symmetric points
    * E13_K4_a0_tol1e-8 shows a nonzero C_Lfom column

  Then:
    python section_6_figures/make_tables.py
    python section_6_figures/fig_err_vs_N.py --width 0.7linewidth
    python section_6_figures/fig_critical_curve.py --width 0.7linewidth

--- still needs code, not an env override ---
  G2  tab:mesh. Needs (a) a refined mesh with halved near-wall spacing, from
      gmsh, and (b) Re_c(0) by full-order bisection on both meshes. The
      bisection driver does not exist; the E13 a=0 runs give you the ROM-side
      value to bracket against.
  G1  fig:mu_overlay. Needs a full-order generalised eigensweep at a=0 over
      the E1 Re grid. nsrom.bifurcation.detection has the reduced machinery;
      the full-order version is a SLEPc solve on the unreduced Jacobian.
      Lower priority now: Re_c(mu) already agrees with the C_L_fom proxy to a
      median of 0.73 against a bracket of 1.0, so fig:rec_convergence and
      tab:critcurve are defensible with one paragraph of justification.
  G3  fig:newton_histories. Consider dropping it. `iterations` is recorded per
      point, so R3 (tensor, FOM-seeded) against R4 (DEIM, FOM-seeded) gives
      Newton cost across all 4285 points instead of three probe points.
NOTES
fi
