#!/usr/bin/env bash
#
# run_all.sh -- overnight experiment matrix for the local-ROM paper (section 6).
#
# Design rules:
#   * NOTHING is overwritten. Each run owns states/<TAG>/ and logs/<TAG>.log.
#     A run whose state dir already exists is SKIPPED, so the script is safe to
#     re-run and safe to interrupt.
#   * A failed run does not stop the queue; it is recorded and the next starts.
#   * Runs are ordered by importance. If the night is cut short you still have
#     the runs section 6 cannot be written without.
#
# Usage:
#   bash run_all.sh              # run everything not already present
#   bash run_all.sh tier1        # only the must-have runs
#   bash run_all.sh --dry-run    # print the plan, run nothing
#   FORCE=1 bash run_all.sh      # re-run even where state dirs exist
#
set -u -o pipefail

cd "$(dirname "$0")" || exit 1

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
SUMMARY="${LOGDIR}/summary_${STAMP}.txt"
mkdir -p "$LOGDIR" states

TIER="${1:-all}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=1; TIER="all"; }
FORCE="${FORCE:-0}"

# Full production grid
FULL_AMPS="-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
# Coarse grid for ablations where the trend, not the resolution, is the point
COARSE_AMPS="-0.8,-0.6,-0.4,-0.2,0.2,0.4,0.6,0.8"
# Narrow grid hugging the bifurcation, for the DEIM failure study
NEAR_AMPS="-0.5,-0.3,-0.1,0.1,0.3,0.5"


# ---------------------------------------------------------------------------
# preflight: two edits must be in place or the whole night is wasted
# ---------------------------------------------------------------------------
preflight() {
    local fail=0

    if ! grep -qE '^\s+if res\[.converged.\]:\s*$' nsrom/bifurcation/diagram.py; then
        echo "PREFLIGHT FAIL: the bare 'if res[\"converged\"]:' guard before"
        echo "  spawns.append(res) is missing in nsrom/bifurcation/diagram.py."
        echo "  Without it no asymmetric branches are ever spawned."
        fail=1
    fi

    if ! grep -q "state_dir=STATE_DIR" nsrom/workflows/local_pipeline.py; then
        echo "PREFLIGHT FAIL: nsrom/workflows/local_pipeline.py does not pass state_dir to"
        echo "  build_full_diagram_bare -- no states would be saved."
        fail=1
    fi

    if ! grep -q "NSROM_RUN_TAG" nsrom/workflows/local_pipeline.py; then
        echo "PREFLIGHT FAIL: nsrom/workflows/local_pipeline.py has no env overrides;"
        echo "  restore NSROM_RUN_TAG handling in nsrom/workflows/local_pipeline.py."
        fail=1
    fi

    if ! python -c "import nsrom, nsrom.local_rom, nsrom.bifurcation.detection" 2>/dev/null; then
        echo "PREFLIGHT FAIL: nsrom does not import cleanly."
        fail=1
    fi

    return $fail
}


# ---------------------------------------------------------------------------
# run one configuration
# ---------------------------------------------------------------------------
run_case() {
    local tag="$1"; shift
    local desc="$1"; shift
    local state_dir="states/${tag}"
    local log="${LOGDIR}/${tag}.log"

    if [[ $DRY_RUN -eq 1 ]]; then
        printf '  %-24s %s\n' "$tag" "$desc"
        printf '      env: %s\n' "$*"
        return 0
    fi

    if [[ -d "$state_dir" && "$FORCE" != "1" ]]; then
        echo "[skip] ${tag} -- ${state_dir} exists (FORCE=1 to override)"
        echo "SKIP  ${tag}" >> "$SUMMARY"
        return 0
    fi
    if [[ -d "$state_dir" && "$FORCE" == "1" ]]; then
        mv "$state_dir" "${state_dir}.bak_${STAMP}"
        echo "[force] moved old ${state_dir} aside"
    fi
    if [[ -f "$log" ]]; then
        mv "$log" "${log}.bak_${STAMP}"
    fi

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
        printf 'OK    %-24s %6ss  %6s pts\n' "$tag" "$dt" "$npts" >> "$SUMMARY"
    else
        echo "[FAIL] ${tag}  rc=${rc}  after ${dt}s -- see ${log}"
        printf 'FAIL  %-24s %6ss  rc=%s\n' "$tag" "$dt" "$rc" >> "$SUMMARY"
        tail -n 25 "$log" | sed 's/^/       | /'
    fi
    return 0
}


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------
echo "NS_ROM experiment matrix   ${STAMP}" | tee -a "$SUMMARY"
echo "tier=${TIER}  force=${FORCE}  dry_run=${DRY_RUN}" | tee -a "$SUMMARY"

if [[ $DRY_RUN -eq 0 ]]; then
    if ! preflight; then
        echo ""
        echo "Preflight failed. Nothing was run."
        exit 1
    fi
    echo "preflight OK" | tee -a "$SUMMARY"
fi


# ===========================================================================
# TIER 1 -- section 6 cannot be written without these   (~5-6 h)
# ===========================================================================
if [[ "$TIER" == "all" || "$TIER" == "tier1" ]]; then
echo ""; echo "### TIER 1 ###"

# E1  master run. Defines the replay point set and every FOM reference field.
run_case "E1_K4_tensor" "K=4 tensor, full grid, FOM on  (~2.5h)" \
    NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_AMPS="$FULL_AMPS" NSROM_FOM_COMPUTE=1

# E2  global baseline at the SAME tolerance. N is expected to be much larger
#     than any local cluster's; that gap is the local-ROM argument.
run_case "E2_K1_tensor" "K=1 tensor (global), full grid, FOM on  (~2.5h)" \
    NSROM_K=1 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_AMPS="$FULL_AMPS" NSROM_FOM_COMPUTE=1

# E3  exact projection = the no-hyper-reduction ceiling. Must match tensor to
#     machine precision; that is what makes the tensor claim airtight.
#     FOM off: errors come from replay against E1's stored fields.
run_case "E3_K4_exact" "K=4 exact, full grid, FOM off  (~10 min)" \
    NSROM_K=4 NSROM_MODE=exact NSROM_POD_TOL=1e-8 \
    NSROM_AMPS="$FULL_AMPS" NSROM_FOM_COMPUTE=0

# E4  DEIM at matched tolerance -- the configuration a practitioner deploys.
#     This is the honest one for the cost table.
run_case "E4_K4_deim_tol8" "K=4 DEIM tol=1e-8, full grid, FOM off  (~40 min)" \
    NSROM_K=4 NSROM_MODE=deim NSROM_POD_TOL=1e-8 \
    NSROM_DEIM_TOL=1e-8 NSROM_RECOMPUTE_DEIM=1 \
    NSROM_AMPS="$FULL_AMPS" NSROM_FOM_COMPUTE=0
fi


# ===========================================================================
# TIER 2 -- the K ablation and the N-convergence study   (~2-3 h)
# ===========================================================================
if [[ "$TIER" == "all" || "$TIER" == "tier2" ]]; then
echo ""; echo "### TIER 2 ###"

# K sensitivity. Coarse amp grid: the trend across K is the claim, not the
# resolution of any single diagram.
run_case "E5_K2_tensor" "K=2 tensor, coarse grid, FOM on  (~50 min)" \
    NSROM_K=2 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_AMPS="$COARSE_AMPS" NSROM_FOM_COMPUTE=1

run_case "E6_K6_tensor" "K=6 tensor, coarse grid, FOM on  (~50 min)" \
    NSROM_K=6 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_AMPS="$COARSE_AMPS" NSROM_FOM_COMPUTE=1

# POD-tolerance sweep -> N per cluster -> Re_c error vs N (figure F9).
# Re_c comes from the mu crossing, so no FOM is needed here.
for tol in 1e-4 1e-6 1e-10; do
    run_case "E7_K4_tensor_tol${tol}" "K=4 tensor, POD tol=${tol}, FOM off  (~10 min)" \
        NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL="$tol" \
        NSROM_RECOMPUTE_POD=1 NSROM_RECOMPUTE_TENSOR=1 \
        NSROM_AMPS="$FULL_AMPS" NSROM_FOM_COMPUTE=0
done

# Same sweep for the global ROM: shows N_global(tol) growing far faster than
# N_local(tol). This is the matched-tolerance half of the ablation.
for tol in 1e-4 1e-6 1e-10; do
    run_case "E8_K1_tensor_tol${tol}" "K=1 tensor, POD tol=${tol}, FOM off  (~15 min)" \
        NSROM_K=1 NSROM_MODE=tensor NSROM_POD_TOL="$tol" \
        NSROM_RECOMPUTE_POD=1 NSROM_RECOMPUTE_TENSOR=1 \
        NSROM_AMPS="$FULL_AMPS" NSROM_FOM_COMPUTE=0
done
fi


# ===========================================================================
# TIER 3 -- the DEIM negative control (section 6.3)   (~1-2 h)
# ===========================================================================
if [[ "$TIER" == "all" || "$TIER" == "tier3" ]]; then
echo ""; echo "### TIER 3 ###"

# Maximal-rank DEIM on a grid hugging the bifurcation. This is the run that
# answers "you just did not take enough modes": at tol=1e-16 every mode is
# retained and the residual still floors.
run_case "E9_K4_deim_tol16" "K=4 DEIM tol=1e-16 (max rank), near-bifurcation grid" \
    NSROM_K=4 NSROM_MODE=deim NSROM_POD_TOL=1e-8 \
    NSROM_DEIM_TOL=1e-16 NSROM_RECOMPUTE_DEIM=1 \
    NSROM_RE_MIN=70 NSROM_RE_MAX=105 NSROM_RE_STEP=1 \
    NSROM_AMPS="$NEAR_AMPS" NSROM_FOM_COMPUTE=0

# Same grid in tensor mode: the matched control the DEIM run is compared to.
run_case "E10_K4_tensor_near" "K=4 tensor, near-bifurcation grid, FOM on" \
    NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_RE_MIN=70 NSROM_RE_MAX=105 NSROM_RE_STEP=1 \
    NSROM_AMPS="$NEAR_AMPS" NSROM_FOM_COMPUTE=1

# Matched online dimension: cap both K=1 and K=4 at cluster 1's size so the
# tolerance never binds and every basis has identical online cost. This is
# where local should win on error -- a single 27-mode global basis must span
# the symmetric sheet and both asymmetric branches at once.
run_case "E11_K4_matchdim" "K=4 tensor, N capped at 27/14, FOM on" \
    NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL=1e-12 \
    NSROM_NVEL_MAX=27 NSROM_NPRES_MAX=14 NSROM_NSUP_MAX=14 \
    NSROM_RECOMPUTE_POD=1 NSROM_RECOMPUTE_TENSOR=1 \
    NSROM_AMPS="$COARSE_AMPS" NSROM_FOM_COMPUTE=1

run_case "E12_K1_matchdim" "K=1 tensor, N capped at 27/14, FOM on" \
    NSROM_K=1 NSROM_MODE=tensor NSROM_POD_TOL=1e-12 \
    NSROM_NVEL_MAX=27 NSROM_NPRES_MAX=14 NSROM_NSUP_MAX=14 \
    NSROM_RECOMPUTE_POD=1 NSROM_RECOMPUTE_TENSOR=1 \
    NSROM_AMPS="$COARSE_AMPS" NSROM_FOM_COMPUTE=1

fi


# ---------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "ALL DONE  $(date '+%F %T')"
echo "=============================================================="
cat "$SUMMARY"
echo ""
echo "state dirs:"
du -sh states/*/ 2>/dev/null | sort -k2
echo ""
echo "summary written to ${SUMMARY}"
