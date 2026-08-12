#!/usr/bin/env bash
#
# sync_paper.sh -- copy generated Section 6 artefacts into the paper repo.
#
# The code repo generates; the paper repo consumes. Nothing in the paper repo is
# ever edited by hand under generated/, so a stale number cannot survive a
# regeneration, and nothing in the code repo needs to know LaTeX.
#
# Usage, from the NS_ROM repo root:
#   bash sync_paper.sh                       # uses the default paper path below
#   bash sync_paper.sh /path/to/paper        # or point it somewhere else
#   DRY=1 bash sync_paper.sh                 # show what would be copied
#
set -u -o pipefail

SRC="section_6_figures/out"
PAPER="${1:-$HOME/01_research/my_paper/Ali_Paper/paper}"
DRY="${DRY:-0}"

[[ -d "$SRC" ]]   || { echo "no $SRC -- run the figure scripts first"; exit 1; }
[[ -d "$PAPER" ]] || { echo "no such paper directory: $PAPER"; exit 1; }

FIGS="$PAPER/figures"
TABS="$PAPER/tables"
GEN="$PAPER/generated"
mkdir -p "$FIGS" "$TABS" "$GEN"

copy() {   # copy <src> <dstdir>
    if [[ "$DRY" == "1" ]]; then
        printf '  would copy  %-34s -> %s\n' "$(basename "$1")" "${2#$PAPER/}"
    else
        cp -p "$1" "$2/" && printf '  %-34s -> %s\n' "$(basename "$1")" "${2#$PAPER/}"
    fi
}

echo "paper: $PAPER"
echo ""
echo "--- figures ---"
n=0
for f in "$SRC"/fig_*.pdf; do
    [[ -e "$f" ]] || continue
    copy "$f" "$FIGS"; n=$((n+1))
done
[[ $n -eq 0 ]] && echo "  (none -- did the figure scripts write PDFs?)"

echo ""
echo "--- tables ---"
n=0
for f in "$SRC"/tab_*.tex; do
    [[ -e "$f" ]] || continue
    copy "$f" "$TABS"; n=$((n+1))
done
[[ $n -eq 0 ]] && echo "  (none -- run make_tables.py and tab_critcurve.py)"

echo ""
echo "--- generated macros ---"
for f in "$SRC"/macros_*.tex "$SRC"/section6_numbers.tex; do
    [[ -e "$f" ]] || continue
    copy "$f" "$GEN"
done

# ---------------------------------------------------------------------------
# A generated \newcommand that is ALSO defined in macros.tex is a hard LaTeX
# error ("command already defined"), and the message names the macro but not the
# file, which is an annoying afternoon. Check for it here instead.
# ---------------------------------------------------------------------------
echo ""
echo "--- collision check against macros.tex ---"
MACROS="$PAPER/macros.tex"
if [[ -f "$MACROS" ]]; then
    clashes=$(
        cat "$SRC"/macros_*.tex "$SRC"/section6_numbers.tex 2>/dev/null \
            | grep -o '\\newcommand{\\[A-Za-z]*}' \
            | sed 's/.*{\\\(.*\)}/\1/' | sort -u > /tmp/_gen_names.$$
        grep -oE '\\(newcommand|renewcommand|def)\s*\{?\\[A-Za-z]+' "$MACROS" \
            | grep -o '\\[A-Za-z]*$' | sed 's/\\//' | sort -u > /tmp/_paper_names.$$
        comm -12 /tmp/_gen_names.$$ /tmp/_paper_names.$$
        rm -f /tmp/_gen_names.$$ /tmp/_paper_names.$$
    )
    if [[ -n "$clashes" ]]; then
        echo "  !! these names are defined in BOTH macros.tex and the generated"
        echo "     files. LaTeX will stop. Delete the hand-written definition --"
        echo "     the generated one is the measured value:"
        echo "$clashes" | sed 's/^/       \\/'
    else
        echo "  no collisions"
    fi
else
    echo "  (macros.tex not found at $MACROS -- skipped)"
fi

# ---------------------------------------------------------------------------
echo ""
echo "--- unfilled cells ---"
nchk=$(grep -l 'CHECK' "$TABS"/tab_*.tex 2>/dev/null | wc -l)
if [[ "$nchk" -gt 0 ]]; then
    echo "  \\CHECK{} appears in:"
    for f in "$TABS"/tab_*.tex; do
        [[ -e "$f" ]] || continue
        c=$(grep -c 'CHECK' "$f" || true)
        [[ "$c" -gt 0 ]] && printf '       %-24s %s cell(s)\n' "$(basename "$f")" "$c"
    done
    echo "  these render as [CHECK: ...] in the build, which is the point."
else
    echo "  none"
fi

cat <<'NOTES'

--- add to the preamble, once, AFTER \input{macros} ---
  \newcommand{\CHECK}[1]{\textbf{[CHECK: #1]}}
  \input{generated/section6_numbers}
  \input{generated/macros_branch_errors}
  \input{generated/macros_rom_overlay}
  \input{generated/macros_err_vs_N}
  \input{generated/macros_local_vs_global}
  \input{generated/macros_fom_diagram}
  \input{generated/macros_newton_cost}
  \input{generated/macros_rec_convergence}
  \input{generated/macros_critical_curve}
  \input{generated/macros_critcurve}

Re-run this script after ANY figure-script run. The macros are the numbers in
the prose, so regenerating a figure without syncing leaves the text quoting the
previous run.
NOTES
