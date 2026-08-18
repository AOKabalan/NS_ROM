PYTHON ?= python
MPIEXEC ?= mpiexec
PYTEST ?= pytest
MPLBACKEND ?= Agg

.PHONY: help test-fast test figures figures-section6 figures-paper verify-data verify-data-full

help:
	@printf '%s\n' \
	  'make test-fast        Run tests excluding the slow marker (one MPI rank).' \
	  'make test             Run the complete test suite (one MPI rank).' \
	  'make figures          Run the proven repository-contained renderer subset.' \
	  'make figures-section6 Regenerate the tracked-input Section 6 table/macros.' \
	  'make figures-paper    Regenerate the tracked-mesh pinball geometry figure.' \
	  'make verify-data      Verify the external figures data bundle.' \
	  'make verify-data-full Verify the complete external data bundle.'

test-fast:
	$(MPIEXEC) -n 1 $(PYTEST) -m "not slow"

test:
	$(MPIEXEC) -n 1 $(PYTEST)

# Intentionally partial: most Section 6 renderers require untracked states/,
# local_rom/, paper_data/, or mass/ products from expensive numerical runs.
figures-section6:
	@test -f section_6_figures/out/critical_curve.csv || { printf '%s\n' 'missing tracked input: section_6_figures/out/critical_curve.csv'; exit 1; }
	MPLBACKEND=$(MPLBACKEND) $(PYTHON) section_6_figures/tab_critcurve.py
	@printf '%s\n' 'Section 6 note: other figures require local numerical-run artifacts and are not rendered.'

# Intentionally partial: other paper renderers require local state/POD/sweep
# data, while compute_bifurcation_diagram.py and related scripts are heavy.
figures-paper:
	@test -f mesh/mid_pinball.msh || { printf '%s\n' 'missing tracked input: mesh/mid_pinball.msh'; exit 1; }
	MPLBACKEND=$(MPLBACKEND) $(PYTHON) paper_figures/pinball_geometry_figure.py mesh/mid_pinball.msh --outdir figures --width linewidth
	@printf '%s\n' 'Paper note: local-data and heavy-computation figures are not rendered.'

figures: figures-section6 figures-paper

verify-data:
	$(PYTHON) scripts/verify_external_data.py --bundle figures

verify-data-full:
	$(PYTHON) scripts/verify_external_data.py --bundle full
