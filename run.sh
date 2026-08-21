#!/usr/bin/env bash
#
# run.sh -- every experiment the manuscript depends on, in one place.
#
# Each run owns states/<TAG>/ and logs/<TAG>.log. Nothing is overwritten: a run
# whose state directory already exists is SKIPPED, so this is safe to re-run and
# safe to interrupt. A failure does not stop the queue.
#
#   ./run.sh --list              print the matrix and what each run feeds
#   ./run.sh --dry-run           print the plan, run nothing
#   ./run.sh E1_K4_tensor        run one
#   ./run.sh all                 run everything not already present
#   FORCE=1 ./run.sh E7_K4_tensor_tol1e-4    re-run, moving the old dir aside
#
# Adding a run: copy a define() line, give it a tag, and add the Makefile rule
# for whatever figure or table consumes it. Those two edits are the whole job.
#
set -u -o pipefail

cd "$(dirname "$0")" || exit 1

# CLAUDE.md: plain `python` hangs in MPI singleton init on this stack, so every
# Firedrake entry point goes through the launcher. One rank, no numerical
# effect. Override if your environment does not have the bug.
RUNNER="${RUNNER:-mpiexec -n 1 python}"

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
SUMMARY="${LOGDIR}/summary_${STAMP}.txt"
FORCE="${FORCE:-0}"
DRY_RUN=0

# Reference point set for the replay runs.
REF="states/E1_K4_tensor"


# ---------------------------------------------------------------------------
# amplitude grids
#
# These are the grids the STORED results were actually swept on, recovered from
# render/out/critical_curve.csv (21 columns) rather than from the retired
# run_all.sh, which disagreed with the data in two ways:
#
#   * it omitted a = 0.0 from both grids, though every stored run has it;
#   * run_gaps.sh additionally carried a = -0.25, swept by mistake. There is no
#     +0.25, so it made the grid asymmetric, and it was later retracted from the
#     stores by hand. It is not here and must not come back -- fig_err_vs_N.py
#     refuses to plot a tolerance family whose members sit on different grids.
#
# One definition each, shared by every run, so the family stays comparable.
# ---------------------------------------------------------------------------
FULL_AMPS="-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
COARSE_AMPS="-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8"
NEAR_AMPS="-0.5,-0.3,-0.1,0.1,0.3,0.5"
ZERO_AMP="0.0"


# ---------------------------------------------------------------------------
# the matrix
#
#   define <TAG> <feeds> <description> -- <env> ...
#
# <feeds> names the paper artifacts that break if the run is missing. Keep it
# accurate: it is what tells you whether a run is still worth its wall clock.
# ---------------------------------------------------------------------------
declare -A DESC FEEDS ENVV
ORDER=()

define() {
    local tag="$1" feeds="$2" desc="$3"; shift 3
    [[ "${1:-}" == "--" ]] && shift
    DESC[$tag]="$desc"; FEEDS[$tag]="$feeds"; ENVV[$tag]="$*"
    ORDER+=("$tag")
}

# --- the master run. Defines the replay point set and every FOM reference field.
define E1_K4_tensor \
    "fig:branch_errors fig:rom_overlay fig:critical_curve fig:mu_sweeps fig:local_vs_global fig:newton_cost tab:params tab:dim_comparison tab:k_sensitivity tab:critcurve" \
    "K=4 tensor, full grid, FOM on  (~2.5h)" -- \
    NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_AMPS="$FULL_AMPS" NSROM_FOM_COMPUTE=1 NSROM_SYM_START="high"

# --- global baseline at the SAME tolerance. The local-ROM argument is the gap
#     between its N and any local cluster's.
define E2_K1_tensor \
    "fig:local_vs_global tab:dim_comparison tab:k_sensitivity" \
    "K=1 tensor (global), full grid, FOM on  (~2.5h)" -- \
    NSROM_K=1 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_AMPS="$FULL_AMPS" NSROM_FOM_COMPUTE=1 NSROM_SYM_START="high"

# --- DEIM at matched tolerance: the configuration a practitioner deploys.
define E4_K4_deim_tol8 \
    "tab:hyperreduction" \
    "K=4 DEIM tol=1e-8, full grid, FOM off  (~40 min)" -- \
    NSROM_K=4 NSROM_MODE=deim NSROM_POD_TOL=1e-8 \
    NSROM_DEIM_TOL=1e-8 NSROM_RECOMPUTE_DEIM=1 \
    NSROM_AMPS="$FULL_AMPS" NSROM_FOM_COMPUTE=0 NSROM_SYM_START="high"

# --- K sensitivity. Coarse grid: the trend across K is the claim, not the
#     resolution of any single diagram.
define E5_K2_tensor \
    "tab:k_sensitivity" \
    "K=2 tensor, coarse grid, FOM on  (~50 min)" -- \
    NSROM_K=2 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_AMPS="$COARSE_AMPS" NSROM_FOM_COMPUTE=1 NSROM_SYM_START="high"

define E6_K6_tensor \
    "tab:k_sensitivity" \
    "K=6 tensor, coarse grid, FOM on  (~50 min)" -- \
    NSROM_K=6 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_AMPS="$COARSE_AMPS" NSROM_FOM_COMPUTE=1 NSROM_SYM_START="high"

# --- POD-tolerance family. E1 is the tol=1e-8 member; these are the other
#     three. Re_c comes from the mu crossing, so no FOM is needed.
for _tol in 1e-4 1e-6 1e-10; do
    define "E7_K4_tensor_tol${_tol}" \
        "macros_err_vs_N" \
        "K=4 tensor, POD tol=${_tol}, FOM off  (~10 min)" -- \
        NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL="$_tol" \
        NSROM_RECOMPUTE_POD=1 NSROM_RECOMPUTE_TENSOR=1 \
        NSROM_AMPS="$FULL_AMPS" NSROM_FOM_COMPUTE=0 NSROM_SYM_START="high"
done
unset _tol

# --- maximal-rank DEIM near the bifurcation. Answers "you just did not take
#     enough modes": at tol=1e-16 every mode is retained and it still floors.
define E9_K4_deim_tol16 \
    "tab:hyperreduction" \
    "K=4 DEIM tol=1e-16 (max rank), near-bifurcation grid  (~1h)" -- \
    NSROM_K=4 NSROM_MODE=deim NSROM_POD_TOL=1e-8 \
    NSROM_DEIM_TOL=1e-16 NSROM_RECOMPUTE_DEIM=1 \
    NSROM_RE_MIN=70 NSROM_RE_MAX=105 NSROM_RE_STEP=1 \
    NSROM_AMPS="$NEAR_AMPS" NSROM_FOM_COMPUTE=0 NSROM_SYM_START="high"

# --- the matched tensor control on that same grid.
define E10_K4_tensor_near \
    "tab:hyperreduction" \
    "K=4 tensor, near-bifurcation grid, FOM on  (~1h)" -- \
    NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_RE_MIN=70 NSROM_RE_MAX=105 NSROM_RE_STEP=1 \
    NSROM_AMPS="$NEAR_AMPS" NSROM_FOM_COMPUTE=1 NSROM_SYM_START="high"

# --- matched online dimension. Cap both K=1 and K=4 at cluster 1's size so the
#     tolerance never binds and every basis has identical online cost.
define E11_K4_matchdim \
    "fig:local_vs_global tab:dim_comparison" \
    "K=4 tensor, N capped at 27/14, coarse grid, FOM on" -- \
    NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL=1e-12 \
    NSROM_NVEL_MAX=27 NSROM_NPRES_MAX=14 NSROM_NSUP_MAX=14 \
    NSROM_RECOMPUTE_POD=1 NSROM_RECOMPUTE_TENSOR=1 \
    NSROM_AMPS="$COARSE_AMPS" NSROM_FOM_COMPUTE=1 NSROM_SYM_START="high"

define E12_K1_matchdim \
    "fig:local_vs_global tab:dim_comparison" \
    "K=1 tensor, N capped at 27/14, coarse grid, FOM on" -- \
    NSROM_K=1 NSROM_MODE=tensor NSROM_POD_TOL=1e-12 \
    NSROM_NVEL_MAX=27 NSROM_NPRES_MAX=14 NSROM_NSUP_MAX=14 \
    NSROM_RECOMPUTE_POD=1 NSROM_RECOMPUTE_TENSOR=1 \
    NSROM_AMPS="$COARSE_AMPS" NSROM_FOM_COMPUTE=1 NSROM_SYM_START="high"

# --- the a = 0 slice with stored FOM fields. Unique in carrying a principled
#     full-order locator of the pitchfork: with no rotation the symmetric branch
#     carries no lift, so departure from zero IS the symmetry breaking.
define E13_K4_a0_tol1e-8 \
    "macros_fom_diagram" \
    "K=4 tensor, a=0 only, FOM on, fields saved  (~10 min)" -- \
    NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
    NSROM_AMPS="$ZERO_AMP" NSROM_FOM_COMPUTE=1 \
    NSROM_SAVE_FOM_FIELDS=1 NSROM_FOM_FIELD_STRIDE=1 NSROM_SYM_START="high"

# --- isolates rank from grid in tab:hyperreduction. E9 used a narrower grid
#     than E4, so part of its wider coverage was the easier problem; running
#     E9's exact grid at tol=1e-8 makes the rank the only difference.
define E15_K4_deim_tol8_near \
    "tab:hyperreduction" \
    "K=4 DEIM tol=1e-8 on E9's grid  (~15 min)" -- \
    NSROM_K=4 NSROM_MODE=deim NSROM_POD_TOL=1e-8 \
    NSROM_DEIM_TOL=1e-8 NSROM_RECOMPUTE_DEIM=1 \
    NSROM_RE_MIN=70 NSROM_RE_MAX=105 NSROM_RE_STEP=1 \
    NSROM_AMPS="$NEAR_AMPS" NSROM_FOM_COMPUTE=0 NSROM_SYM_START="high"

# --- replays. Every point of E1 is re-solved independently from a stored guess:
#     no continuation, no spine, so a failure at one point cannot suppress
#     another and the convergence map means "DEIM solves here, not there".
#     R1 is the harness gate -- it must reproduce E1. CHECK IT before R2.
define R1_tensor_stored \
    "tab:hyperreduction" \
    "replay: tensor, seed=stored -- validation vs E1  (~30 min)" -- \
    NSROM_MODE=tensor NSROM_REPLAY_SEED=stored

define R2_deim8_stored \
    "tab:hyperreduction" \
    "replay: DEIM tol=1e-8, seed=stored -- negative control  (~6-8h)" -- \
    NSROM_MODE=deim NSROM_DEIM_TOL=1e-8 NSROM_RECOMPUTE_DEIM=1 \
    NSROM_REPLAY_SEED=stored

define R3_tensor_fom \
    "fig:newton_cost tab:hyperreduction" \
    "replay: tensor, seed=fom -- control for R4  (~30 min)" -- \
    NSROM_MODE=tensor NSROM_REPLAY_SEED=fom

define R4_deim8_fom \
    "fig:newton_cost tab:hyperreduction" \
    "replay: DEIM tol=1e-8, seed=fom -- most generous  (~6-8h)" -- \
    NSROM_MODE=deim NSROM_DEIM_TOL=1e-8 NSROM_RECOMPUTE_DEIM=1 \
    NSROM_REPLAY_SEED=fom


# ---------------------------------------------------------------------------
is_replay() { [[ "$1" == R* ]]; }

preflight() {
    local fail=0
    if ! $RUNNER -c "import nsrom, nsrom.rom.local, nsrom.bifurcation.detection" \
            >/dev/null 2>&1; then
        echo "PREFLIGHT FAIL: nsrom does not import cleanly under '$RUNNER'."
        fail=1
    fi
    if ! grep -q "NSROM_RUN_TAG" nsrom/workflows/local_pipeline.py; then
        echo "PREFLIGHT FAIL: local_pipeline.py has no env overrides."
        fail=1
    fi
    return $fail
}

replay_preflight() {
    local fail=0
    if [[ ! -f "${REF}/index.jsonl" ]]; then
        echo "PREFLIGHT FAIL: reference ${REF}/index.jsonl not found."
        echo "  the replays re-solve E1's point set; run E1_K4_tensor first."
        fail=1
    fi
    if ! grep -q "NSROM_REPLAY_REF" nsrom/workflows/local_pipeline.py; then
        echo "PREFLIGHT FAIL: local_pipeline.py has no replay hook."
        fail=1
    fi
    return $fail
}

run_one() {
    local tag="$1"
    local state_dir="states/${tag}" log="${LOGDIR}/${tag}.log"

    if [[ -z "${ENVV[$tag]+x}" ]]; then
        echo "unknown run: ${tag}   (./run.sh --list)"; return 1
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        printf '  %-24s %s\n' "$tag" "${DESC[$tag]}"
        printf '      feeds: %s\n' "${FEEDS[$tag]}"
        printf '      env  : %s\n' "${ENVV[$tag]}"
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

    if is_replay "$tag"; then
        replay_preflight || { echo "SKIP  ${tag} (replay preflight)" >> "$SUMMARY"; return 0; }
    fi

    echo ""
    echo "=============================================================="
    echo "[run ] ${tag} -- ${DESC[$tag]}"
    echo "       feeds ${FEEDS[$tag]}"
    echo "       started $(date '+%F %T')"
    echo "=============================================================="

    local t0=$SECONDS extra=()
    if is_replay "$tag"; then
        # every replay shares E1's reference set, K and tolerance; only the
        # convection mode and the seed differ, which is the whole point
        extra=(NSROM_REPLAY_REF="$REF" NSROM_REPLAY_STRIDE="${STRIDE:-1}"
               NSROM_K=4 NSROM_POD_TOL=1e-8)
    fi

    env ${ENVV[$tag]} "${extra[@]}" \
        MPLBACKEND=Agg \
        NSROM_RUN_TAG="$tag" \
        NSROM_STATE_DIR="$state_dir" \
        $RUNNER scripts/main_local.py > "$log" 2>&1
    local rc=$? dt=$(( SECONDS - t0 ))

    if [[ $rc -eq 0 ]]; then
        local n conv
        n=$(wc -l < "${state_dir}/index.jsonl" 2>/dev/null || echo 0)
        conv=$(grep -c '"converged": true' "${state_dir}/index.jsonl" 2>/dev/null || echo 0)
        echo "[done] ${tag}  ${dt}s  ${conv}/${n} converged"
        printf 'OK    %-24s %6ss  %5s/%-5s converged\n' "$tag" "$dt" "$conv" "$n" >> "$SUMMARY"
    else
        echo "[FAIL] ${tag}  rc=${rc}  after ${dt}s -- see ${log}"
        printf 'FAIL  %-24s %6ss  rc=%s\n' "$tag" "$dt" "$rc" >> "$SUMMARY"
        tail -n 25 "$log" | sed 's/^/       | /'
    fi
    return 0
}


# ---------------------------------------------------------------------------
case "${1:-all}" in
    --list|-l)
        printf '%-24s  %s\n' TAG FEEDS
        printf '%-24s  %s\n' "------------------------" "-----"
        for t in "${ORDER[@]}"; do printf '%-24s  %s\n' "$t" "${FEEDS[$t]}"; done
        echo ""
        echo "${#ORDER[@]} runs. Details: ./run.sh --dry-run"
        exit 0 ;;
    --dry-run|-n)
        DRY_RUN=1; set -- "${2:-all}" ;;
    -h|--help)
        sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

mkdir -p "$LOGDIR" states

TARGET="${1:-all}"

if [[ $DRY_RUN -eq 0 ]]; then
    preflight || { echo ""; echo "Preflight failed. Nothing was run."; exit 1; }
fi

if [[ "$TARGET" == "all" ]]; then
    echo "NS_ROM experiment matrix   ${STAMP}   (${#ORDER[@]} runs)" | tee -a "$SUMMARY"
    for t in "${ORDER[@]}"; do run_one "$t"; done
    if [[ $DRY_RUN -eq 0 ]]; then
        echo ""; echo "ALL DONE  $(date '+%F %T')"; cat "$SUMMARY"
        echo ""; du -sh states/*/ 2>/dev/null | sort -k2
    fi
else
    run_one "$TARGET"
fi
