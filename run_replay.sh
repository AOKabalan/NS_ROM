#!/usr/bin/env bash
#
# run_replay.sh -- pointwise replay matrix for section 6.4.
#
# Every point of the reference run is re-solved independently from a stored
# initial guess. No continuation, no spine, no spawns -- so a failure at one
# point cannot suppress another, and the resulting convergence map means
# "DEIM solves here, not there" rather than "the continuation tree collapsed".
#
# This complements, and does not replace, E4/E9: those show DEIM cannot be
# driven through parameter space at all. These show where it fails even when
# handed a good guess.
#
# Same conventions as run_all.sh: nothing is overwritten, a failure does not
# stop the queue, runs are ordered by importance.
#
# Usage:
#   bash run_replay.sh              # everything not already present
#   bash run_replay.sh tier1        # the gate + the main result
#   bash run_replay.sh --dry-run    # print the plan
#   FORCE=1 bash run_replay.sh      # re-run even where state dirs exist
#   STRIDE=3 bash run_replay.sh     # subsample every 3rd point (see note)
#
set -u -o pipefail

cd "$(dirname "$0")" || exit 1

REF="states/E1_K4_tensor"        # reference point set (4258 points, K=4, tol 1e-8)
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
SUMMARY="${LOGDIR}/replay_summary_${STAMP}.txt"
mkdir -p "$LOGDIR" states

TIER="${1:-all}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=1; TIER="all"; }
FORCE="${FORCE:-0}"
STRIDE="${STRIDE:-1}"


# ---------------------------------------------------------------------------
preflight() {
    local fail=0

    if [[ ! -f "${REF}/index.jsonl" ]]; then
        echo "PREFLIGHT FAIL: reference ${REF}/index.jsonl not found."
        fail=1
    fi

    if ! grep -q "NSROM_REPLAY_REF" scripts/main_local.py; then
        echo "PREFLIGHT FAIL: main_local.py has no replay hook."
        echo "  Apply main_local_replay_patch.py first."
        fail=1
    fi

    if [[ ! -f scripts/replay_diagram.py && ! -f replay_diagram.py ]]; then
        echo "PREFLIGHT FAIL: replay_diagram.py not on the path."
        fail=1
    fi

    # the reference must carry u_in everywhere and FOM fields for seed=fom
    python - "$REF" <<'EOF'
import sys, json
recs = [json.loads(l) for l in open(f"{sys.argv[1]}/index.jsonl") if l.strip()]
n = len(recs)
nf = sum(r.get('has_fom_field', False) for r in recs)
meta = json.load(open(f"{sys.argv[1]}/run_meta.json"))
print(f"  reference: {n} points, {nf} FOM fields, "
      f"stride={meta.get('fom_field_stride')}, dtype={meta.get('fom_field_dtype')}")
if nf == 0:
    print("  WARNING: no FOM fields -- tier2 (seed=fom) will skip every point,")
    print("           and replay errors cannot be computed. Tier1 still valid.")
EOF

    return $fail
}


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
    fi
    [[ -f "$log" ]] && mv "$log" "${log}.bak_${STAMP}"

    echo ""
    echo "=============================================================="
    echo "[run ] ${tag} -- ${desc}"
    echo "       started $(date '+%F %T')"
    echo "=============================================================="

    local t0=$SECONDS
    env "$@" \
        MPLBACKEND=Agg \
        NSROM_RUN_TAG="$tag" \
        NSROM_STATE_DIR="$state_dir" \
        NSROM_REPLAY_REF="$REF" \
        NSROM_REPLAY_STRIDE="$STRIDE" \
        NSROM_K=4 NSROM_POD_TOL=1e-8 \
        python scripts/main_local.py > "$log" 2>&1
    local rc=$?
    local dt=$(( SECONDS - t0 ))

    if [[ $rc -eq 0 ]]; then
        local n conv
        n=$(wc -l < "${state_dir}/index.jsonl" 2>/dev/null || echo 0)
        conv=$(grep -c '"converged": true' "${state_dir}/index.jsonl" 2>/dev/null || echo 0)
        echo "[done] ${tag}  ${dt}s  ${conv}/${n} converged"
        printf 'OK    %-24s %6ss  %5s/%-5s converged\n' \
               "$tag" "$dt" "$conv" "$n" >> "$SUMMARY"
    else
        echo "[FAIL] ${tag}  rc=${rc}  after ${dt}s -- see ${log}"
        printf 'FAIL  %-24s %6ss  rc=%s\n' "$tag" "$dt" "$rc" >> "$SUMMARY"
        tail -n 25 "$log" | sed 's/^/       | /'
    fi
    return 0
}


# ---------------------------------------------------------------------------
echo "NS_ROM replay matrix   ${STAMP}" | tee -a "$SUMMARY"
echo "ref=${REF}  tier=${TIER}  force=${FORCE}  stride=${STRIDE}" | tee -a "$SUMMARY"

if [[ $DRY_RUN -eq 0 ]]; then
    if ! preflight; then
        echo ""; echo "Preflight failed. Nothing was run."
        exit 1
    fi
    echo "preflight OK" | tee -a "$SUMMARY"
fi


# ===========================================================================
# TIER 1 -- the gate and the main result
# ===========================================================================
if [[ "$TIER" == "all" || "$TIER" == "tier1" ]]; then
echo ""; echo "### TIER 1 ###"

# R1  HARNESS VALIDATION. Same mode, same guess as the reference: this must
#     reproduce E1 almost exactly. Convergence ~100%, and err_u must agree
#     with ref_err_u to a few ulp. If it does not, the replay is wrong and
#     every run below it is worthless. Run this first and CHECK IT before
#     starting R2.
run_case "R1_tensor_stored" "tensor, seed=stored -- validation vs E1 (~30 min)" \
    NSROM_MODE=tensor NSROM_REPLAY_SEED=stored

# R2  THE MAIN RESULT. DEIM at the tolerance a practitioner would deploy,
#     given exactly the guess the tensor path got. Any difference in outcome
#     is the operator, nothing else.
run_case "R2_deim8_stored" "DEIM tol=1e-8, seed=stored -- negative control (~6-8 h)" \
    NSROM_MODE=deim NSROM_DEIM_TOL=1e-8 NSROM_RECOMPUTE_DEIM=1 \
    NSROM_REPLAY_SEED=stored
fi


# ===========================================================================
# TIER 2 -- the generous-seed controls
# ===========================================================================
if [[ "$TIER" == "all" || "$TIER" == "tier2" ]]; then
echo ""; echo "### TIER 2 ###"

# R3  tensor from the exact solution. Cheap, and the matched control for R4.
run_case "R3_tensor_fom" "tensor, seed=fom -- control (~30 min)" \
    NSROM_MODE=tensor NSROM_REPLAY_SEED=fom

# R4  DEIM handed the exact solution as its initial guess. The strongest form
#     of the argument: zero warm-start error, and it still fails near Re_c.
run_case "R4_deim8_fom" "DEIM tol=1e-8, seed=fom -- most generous (~6-8 h)" \
    NSROM_MODE=deim NSROM_DEIM_TOL=1e-8 NSROM_RECOMPUTE_DEIM=1 \
    NSROM_REPLAY_SEED=fom
fi


# ===========================================================================
# TIER 3 -- the "you just did not take enough modes" rebuttal
# ===========================================================================
if [[ "$TIER" == "all" || "$TIER" == "tier3" ]]; then
echo ""; echo "### TIER 3 ###"

# Max-rank DEIM. At tol=1e-16 every available mode is retained. If the
# convergence map is still holed near Re_c, rank is not the problem.
# NOTE: raise m_F_max / m_J_max to 400 first or cluster 2 truncates at 200
# of 371 available modes and this run does not test what it claims to.
run_case "R5_deim16_stored" "DEIM tol=1e-16 (max rank), seed=stored (~8 h)" \
    NSROM_MODE=deim NSROM_DEIM_TOL=1e-16 NSROM_RECOMPUTE_DEIM=1 \
    NSROM_REPLAY_SEED=stored

run_case "R6_deim16_fom" "DEIM tol=1e-16 (max rank), seed=fom (~8 h)" \
    NSROM_MODE=deim NSROM_DEIM_TOL=1e-16 NSROM_RECOMPUTE_DEIM=1 \
    NSROM_REPLAY_SEED=fom
fi


# ---------------------------------------------------------------------------
echo ""
echo "=============================================================="
echo "ALL DONE  $(date '+%F %T')"
echo "=============================================================="
cat "$SUMMARY"
echo ""
du -sh states/R*/ 2>/dev/null | sort -k2
echo ""
echo "summary written to ${SUMMARY}"
