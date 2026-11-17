# Portable numerical workflow; independent of manuscript sources and LaTeX.
.DEFAULT_GOAL := figures
.PHONY: check-data-src check-directors experiment simulate analyze figures figures-raw figures-publish release-archives export-code verify-results

FORCE_FLAG := $(if $(filter 1 yes true,$(FORCE)),--force,)

check-data-src:
	./scripts/check_data_src

check-directors:
	./scripts/check_directors

experiment:
	uv run python code/scripts/reproduce_experimental_data.py $(FORCE_FLAG)

simulate: check-directors
	uv run --extra simulation python code/scripts/reproduce_simulated_optics.py $(FORCE_FLAG)

analyze:
	uv run python code/scripts/reproduce_derived_analyses.py $(FORCE_FLAG)

figures:
	./scripts/build_figures --output-dir build/figures/raw $(FORCE_FLAG)
	./scripts/publish_figures build/figures/raw figs $(FORCE_FLAG)

figures-raw:
	./scripts/build_figures --output-dir build/figures/raw $(FORCE_FLAG)

figures-publish:
	./scripts/publish_figures build/figures/raw figs $(FORCE_FLAG)

release-archives:
	./scripts/pack_release $(FORCE_FLAG)

export-code:
	python3 scripts/export_code $(DEST)

verify-results:
	uv run python code/scripts/verify_results.py
