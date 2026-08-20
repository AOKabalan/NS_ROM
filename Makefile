# =============================================================================
# NS_ROM -- regenerate exactly what changed.
#
#   make paper           every figure, table and macro file the manuscript uses
#   make figures         the 7 figures only
#   make tables          the 5 tables + section6_numbers.tex
#   make sync            copy render/out into the paper repo
#   make list            what each target needs, and whether it can build now
#
#   make run-E5_K2_tensor    one experiment (hours; see ./run.sh --list)
#   make runs                all 18, skipping any already present
#
#   make test-fast       test suite minus the slow marker
#   make test            the whole suite
#
# WHY THE EXPERIMENTS ARE NOT AUTOMATIC. A renderer takes seconds and rebuilds
# whenever its inputs move. An experiment takes hours, so no timestamp may ever
# trigger one: states/ appears below only as a prerequisite, never as a target.
# `make paper` after editing a renderer rebuilds that artifact and nothing else;
# it will never start a sweep.
#
# THE THREE KINDS OF CHANGE
#   presentation (a colour, a cutoff, which amplitudes a table shows)
#       edit the renderer          -> make render/out/tab_k_sensitivity.tex
#   style (font, column width, palette)
#       edit render/style.py       -> make figures
#   numerical (a POD tolerance, a grid, the snapshots)
#       ./run.sh <TAG>             -> make paper  rebuilds only what that run feeds
#
# If you change the SNAPSHOTS every result downstream is stale:
#       make clean-derived && make runs && make paper
#   clean-derived removes caches and rendered output. It never touches
#   states/, snapshots/, logs/ or paper_data/ -- see IRREPLACEABLE in CLAUDE.md.
# =============================================================================

ifeq ($(filter grouped-target,$(.FEATURES)),)
$(error This Makefile needs GNU make 4.3 or newer for grouped targets (&:). \
You have $(MAKE_VERSION).)
endif

PYTHON      ?= python
MPIEXEC     ?= mpiexec
PYTEST      ?= pytest
MPLBACKEND  ?= Agg
PAPER       ?= $(HOME)/01_research/my_paper/Ali_Paper/paper

OUT    := render/out
RENDER := $(MPLBACKEND:%=MPLBACKEND=%) $(PYTHON)

# every renderer depends on these two
COMMON := render/common.py render/style.py

# A run's index is what a renderer actually reads. Wrapped in $(wildcard) on
# purpose: when the run is present it is a real prerequisite, so redoing the run
# rebuilds everything it feeds. When it is absent the prerequisite simply drops
# out, and the artifact still builds from whatever tracked data exists -- which
# is what makes `make tables` work on a fresh checkout with no states/ at all.
# `make list` is where you check which runs are actually present.
idx = $(wildcard states/$(1)/index.jsonl)

# $(call have,TAG TAG ...) -> "yes" when every listed run is present.
#
# A renderer rule is defined only when the runs it reads exist. render/out holds
# committed .csv and .tex -- the reproducibility asset that lets the tables be
# rebuilt without a sweep -- and those files are also renderer OUTPUTS, so make
# cannot tell "tracked input" from "stale intermediate" by itself. Gating the
# rule settles it: with the run present you get real dependency tracking, and
# without it make leaves the committed file alone instead of trying to
# regenerate it from a state directory that is not there.
have = $(if $(strip $(foreach t,$(1),$(if $(wildcard states/$(t)/index.jsonl),,x))),,yes)

.PHONY: paper figures tables extras sync runs list test test-fast \
        clean-derived help
.DEFAULT_GOAL := help

# -----------------------------------------------------------------------------
# the manuscript's artifacts
#
# One rule per \includegraphics and per \input{tables/...} in the paper, plus
# the macro files main.tex reads. Each is commented with the label it carries.
# Adding a figure: add its rule here and to FIGS, and add whatever run it needs
# to run.sh. Nothing else knows about it.
# -----------------------------------------------------------------------------

FIGS := $(OUT)/fig_branch_errors.pdf \
        $(OUT)/fig_rom_overlay.pdf \
        $(OUT)/fig_local_vs_global.pdf \
        $(OUT)/fig_newton_cost.pdf \
        $(OUT)/fig_critical_curve.pdf \
        $(OUT)/fig_mu_sweeps.pdf \
        $(OUT)/hyperreduction_pinball.pdf

TABS := $(OUT)/tab_params.tex \
        $(OUT)/tab_dim_comparison.tex \
        $(OUT)/tab_k_sensitivity.tex \
        $(OUT)/tab_hyperreduction.tex \
        $(OUT)/tab_critcurve.tex

# macro files main.tex \inputs whose figure is not itself included
MACROS := $(OUT)/macros_err_vs_N.tex $(OUT)/macros_fom_diagram.tex

paper:   figures tables $(MACROS)
figures: $(FIGS)
tables:  $(TABS) $(OUT)/section6_numbers.tex

# --- fig:branch_errors  §6.2 -- errors over the (Re, a) plane
ifeq ($(call have,E1_K4_tensor),yes)
$(OUT)/fig_branch_errors.pdf $(OUT)/macros_branch_errors.tex &: \
		render/fig_branch_errors.py $(COMMON) \
		$(call idx,E1_K4_tensor) paper_data/point_errors_E1.csv
	$(RENDER) render/fig_branch_errors.py
endif

# --- fig:rom_overlay  §6.2 -- reduced against full-order lift
ifeq ($(call have,E1_K4_tensor),yes)
$(OUT)/fig_rom_overlay.pdf $(OUT)/macros_rom_overlay.tex &: \
		render/fig_rom_overlay.py $(COMMON) $(call idx,E1_K4_tensor)
	$(RENDER) render/fig_rom_overlay.py
endif

# --- fig:local_vs_global  §6.3 -- (a) matched dimension, (b) matched tolerance
ifeq ($(call have,E1_K4_tensor E2_K1_tensor E11_K4_matchdim E12_K1_matchdim),yes)
$(OUT)/fig_local_vs_global.pdf $(OUT)/macros_local_vs_global.tex &: \
		render/fig_local_vs_global.py $(COMMON) \
		$(call idx,E1_K4_tensor) $(call idx,E2_K1_tensor) \
		$(call idx,E11_K4_matchdim) $(call idx,E12_K1_matchdim)
	$(RENDER) render/fig_local_vs_global.py
endif

# --- fig:newton_cost  §6.4 -- tensor against DEIM over the paired replay
ifeq ($(call have,E1_K4_tensor R3_tensor_fom R4_deim8_fom),yes)
$(OUT)/fig_newton_cost.pdf $(OUT)/macros_newton_cost.tex $(OUT)/newton_cost_paired.csv &: \
		render/fig_newton_cost.py $(COMMON) \
		$(call idx,E1_K4_tensor) $(call idx,R3_tensor_fom) $(call idx,R4_deim8_fom)
	$(RENDER) render/fig_newton_cost.py
endif

# --- fig:critical_curve + fig:mu_sweeps  §6.5 -- one mu extraction, two figures.
#     critical_curve.csv is the input to tab:critcurve, so this rule is upstream
#     of that table and the two can never quote different values.
ifeq ($(call have,E1_K4_tensor),yes)
$(OUT)/fig_critical_curve.pdf $(OUT)/fig_mu_sweeps.pdf \
$(OUT)/critical_curve.csv $(OUT)/macros_critical_curve.tex &: \
		render/fig_critical_curve.py $(COMMON) $(call idx,E1_K4_tensor)
	$(RENDER) render/fig_critical_curve.py --which both
endif

# --- fig:hyperreduction  §4 -- the actual reduced integration cells.
#     Needs Firedrake to rebuild the submesh, hence the launcher.
$(OUT)/hyperreduction_pinball.pdf: render/hyperreduction.py render/style.py mesh/mid_pinball.msh
	MPLBACKEND=$(MPLBACKEND) $(MPIEXEC) -n 1 $(PYTHON) render/hyperreduction.py

# --- tab:params, tab:dim_comparison, tab:k_sensitivity, tab:hyperreduction
#     and section6_numbers.tex all come from one pass over the runs and caches.
ifeq ($(call have,E1_K4_tensor E2_K1_tensor E5_K2_tensor E6_K6_tensor E11_K4_matchdim E12_K1_matchdim E4_K4_deim_tol8 E9_K4_deim_tol16 E10_K4_tensor_near E15_K4_deim_tol8_near R1_tensor_stored R2_deim8_stored R3_tensor_fom R4_deim8_fom),yes)
$(OUT)/tab_params.tex $(OUT)/tab_dim_comparison.tex $(OUT)/tab_k_sensitivity.tex \
$(OUT)/tab_hyperreduction.tex $(OUT)/tab_cost.tex $(OUT)/section6_numbers.tex &: \
		render/make_tables.py $(COMMON) \
		$(call idx,E1_K4_tensor) $(call idx,E2_K1_tensor) \
		$(call idx,E5_K2_tensor) $(call idx,E6_K6_tensor) \
		$(call idx,E11_K4_matchdim) $(call idx,E12_K1_matchdim) \
		$(call idx,E4_K4_deim_tol8) $(call idx,E9_K4_deim_tol16) \
		$(call idx,E10_K4_tensor_near) $(call idx,E15_K4_deim_tol8_near) \
		$(call idx,R1_tensor_stored) $(call idx,R2_deim8_stored) \
		$(call idx,R3_tensor_fom) $(call idx,R4_deim8_fom)
	$(RENDER) render/make_tables.py
endif

# --- tab:critcurve  §6.5 -- reads fig_critical_curve.py's output, computes
#     nothing, so the table and the figure cannot disagree.
$(OUT)/tab_critcurve.tex $(OUT)/macros_critcurve.tex &: \
		render/tab_critcurve.py $(COMMON) $(OUT)/critical_curve.csv
	$(RENDER) render/tab_critcurve.py

# --- macros only: fig:err_vs_N is not \includegraphics'd, but four of its
#     numbers are quoted in the §6.3 prose.
ifeq ($(call have,E1_K4_tensor E7_K4_tensor_tol1e-4 E7_K4_tensor_tol1e-6 E7_K4_tensor_tol1e-10),yes)
$(OUT)/macros_err_vs_N.tex $(OUT)/fig_err_vs_N.pdf &: \
		render/fig_err_vs_N.py $(COMMON) \
		$(call idx,E1_K4_tensor) $(call idx,E7_K4_tensor_tol1e-4) \
		$(call idx,E7_K4_tensor_tol1e-6) $(call idx,E7_K4_tensor_tol1e-10)
	$(RENDER) render/fig_err_vs_N.py
endif

# --- macros only: \RecZero and friends, from the a = 0 slice.
ifeq ($(call have,E13_K4_a0_tol1e-8),yes)
$(OUT)/macros_fom_diagram.tex $(OUT)/fig_fom_diagram.pdf &: \
		render/fig_fom_diagram.py $(COMMON) $(call idx,E13_K4_a0_tol1e-8)
	$(RENDER) render/fig_fom_diagram.py
endif

# -----------------------------------------------------------------------------
# parked renderers
#
# Working, but the current draft includes neither their figure nor any of their
# macros. Kept because a review may ask for them; excluded from `make paper` so
# they cost nothing until then. Both need runs that are no longer in run.sh --
# re-add the tags named below before building these.
#
#   fig_k_sweep         needs E1, E2, E5, E6         (all still present)
#   fig_rec_convergence needs E13_K4_a0_tol{1e-4,1e-6,1e-10}, E14_K1_a0_tol1e-8
# -----------------------------------------------------------------------------
extras: $(OUT)/fig_k_sweep.pdf $(OUT)/fig_rec_convergence.pdf

ifeq ($(call have,E1_K4_tensor E2_K1_tensor E5_K2_tensor E6_K6_tensor),yes)
$(OUT)/fig_k_sweep.pdf $(OUT)/macros_k_sweep.tex &: \
		render/fig_k_sweep.py $(COMMON) \
		$(call idx,E1_K4_tensor) $(call idx,E2_K1_tensor) \
		$(call idx,E5_K2_tensor) $(call idx,E6_K6_tensor)
	$(RENDER) render/fig_k_sweep.py
endif

ifeq ($(call have,E13_K4_a0_tol1e-8),yes)
$(OUT)/fig_rec_convergence.pdf $(OUT)/macros_rec_convergence.tex $(OUT)/rec_convergence.csv &: \
		render/fig_rec_convergence.py $(COMMON) $(call idx,E13_K4_a0_tol1e-8)
	$(RENDER) render/fig_rec_convergence.py
endif

# -----------------------------------------------------------------------------
# derived data (between the runs and the renderers)
# -----------------------------------------------------------------------------

# per-point field errors. Reads stored fields, so it needs the mass matrices
# main_local.py used -- rebuilding them would change the norm the paper quotes.
paper_data/point_errors_E1.csv: $(wildcard scripts/compute_point_errors.py) \
		$(call idx,E1_K4_tensor) $(wildcard mass/M_u.npz)
	$(MPIEXEC) -n 1 $(PYTHON) scripts/compute_point_errors.py \
	    --state-dir states/E1_K4_tensor \
	    --local-rom local_rom/K4_H1_tol1e-08 \
	    --mass-u mass/M_u.npz --mass-p mass/M_p.npz \
	    --out $@

# -----------------------------------------------------------------------------
# experiments -- explicit only. See the note at the top.
# -----------------------------------------------------------------------------
RUN_TAGS := $(shell ./run.sh --list 2>/dev/null | awk 'NR>2 && NF {print $$1}')

runs: ; ./run.sh all
$(RUN_TAGS:%=run-%): run-%: ; ./run.sh $*
.PHONY: $(RUN_TAGS:%=run-%)

# When a guard above suppressed a rule, the artifact has no way to be built.
# An explicit rule always beats a pattern rule, so this only ever fires for the
# suppressed ones -- and says which run is missing instead of leaving make to
# report "No rule to make target".
$(OUT)/%.pdf $(OUT)/%.tex:
	@printf '%s\n' \
	  "cannot build $@ -- the runs it reads are not in this checkout." \
	  "  make list          which runs are present" \
	  "  ./run.sh --list    what each one costs and feeds" ; \
	 exit 1

# -----------------------------------------------------------------------------
sync: ; $(PYTHON) render/sync_paper.py $(PAPER)

list:
	@printf '%-42s %s\n' ARTIFACT STATUS
	@for f in $(FIGS) $(TABS) $(MACROS); do \
	    if [ -f "$$f" ]; then s="built"; else s="not built"; fi; \
	    printf '%-42s %s\n' "$$f" "$$s"; \
	done
	@echo ""
	@echo "runs present:"
	@for t in $(RUN_TAGS); do \
	    if [ -f "states/$$t/index.jsonl" ]; then \
	        printf '  %-24s %s points\n' "$$t" "$$(wc -l < states/$$t/index.jsonl)"; \
	    else printf '  %-24s MISSING\n' "$$t"; fi; \
	done

test-fast: ; $(MPIEXEC) -n 1 $(PYTEST) -m "not slow"
test:      ; $(MPIEXEC) -n 1 $(PYTEST)

# Caches and rendered output only. states/, snapshots/, logs/ and paper_data/
# are IRREPLACEABLE (CLAUDE.md) and are never touched here.
clean-derived:
	rm -rf local_rom
	rm -f $(OUT)/*.pdf $(OUT)/*.png
	@echo "removed local_rom/ and the rendered PDFs."
	@echo "kept: the tracked .csv and .tex in $(OUT), states/, snapshots/, paper_data/"

help:
	@sed -n '2,34p' $(MAKEFILE_LIST) | sed 's/^# \{0,1\}//'
