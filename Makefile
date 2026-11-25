PYTHON ?= .venv/bin/python
UV ?= uv

# ----- Common experiment inputs ------------------------------------------------
SYSTEM ?= ours
REPO ?=
EXPERIMENT ?=
RUN_EXPERIMENT = $(if $(EXPERIMENT),$(EXPERIMENT),$(SYSTEM)-$(shell echo "$(REPO)" | tr '[:upper:]' '[:lower:]')-v1)
SUITE ?= core
SPLIT ?= core

# ----- Direct ours generation -------------------------------------------------
REPO_PATH ?= $(if $(wildcard $(REPO)),$(REPO),$(if $(wildcard data/repos/$(REPO)),data/repos/$(REPO),$(REPO)))
REPO_NAME ?= $(notdir $(REPO_PATH))
STATE ?= runs/ours/$(REPO_NAME)/state
CLUSTER_MODE ?= auto
WORKERS ?= 4
BATCH_SIZE ?= 8

# ----- Web UI -----------------------------------------------------------------
PORT ?= 8765

.PHONY: help help-all setup setup-harness setup-bench setup-codewiki setup-ours setup-data \
        doctor list prepare prepare-core generate update run evaluate summary batch \
        web web-setup test test-harness test-ours lint check demo clean \
        smoke-codewiki smoke-codewiki-core smoke-codewiki-harness smoke-ours \
        llm-check llm-check-config architecture architecture-rubrics demo-report

help:
	@echo "RepoDoc — common commands"
	@echo ""
	@echo "  make setup                         install Python deps + CodeWiki + RepoDoc"
	@echo "  make doctor                        check environments and data"
	@echo "  make list                          list systems, repos and suites"
	@echo "  make web                           build if needed and serve http://127.0.0.1:$(PORT)"
	@echo "  make generate REPO=<path|name>     generate docs with RepoDoc"
	@echo "  make run REPO=<name>               generate + parse + judge (SYSTEM=ours by default)"
	@echo "  make evaluate REPO=<name>          judge docs that already exist"
	@echo "  make update REPO=<path> [STATE=<state>] incremental RepoDoc update"
	@echo "  make summary EXPERIMENT=<name>     summarize an experiment"
	@echo "  make test                          run harness + RepoDoc tests"
	@echo "  make lint                          run ruff"
	@echo "  make demo                          run offline smoke tests"
	@echo ""
	@echo "Examples:"
	@echo "  make generate REPO=data/repos/graphrag"
	@echo "  make generate REPO=graphrag STATE=runs/ours/graphrag/state"
	@echo "  make run REPO=graphrag"
	@echo "  make run REPO=graphrag SYSTEM=codewiki"
	@echo ""
	@echo "Advanced targets: make help-all"

help-all:
	@echo "Advanced / experiment targets"
	@echo "  make prepare REPO=<name>           download benchmark refs/rubrics and checkout"
	@echo "  make prepare-core                  prepare the 7-repo core split"
	@echo "  make batch SUITE=core              run a suite"
	@echo "  make smoke-codewiki                CodeWiki offline smoke"
	@echo "  make smoke-ours                    RepoDoc toy pipeline smoke"
	@echo "  make llm-check-config              inspect OpenCode Go config"
	@echo "  make llm-check                     send a minimal LLM request"
	@echo "  make architecture RESULTS=<json>   architecture view over judge results"
	@echo "  make architecture-rubrics RUBRICS=<json> OUT=<json>"
	@echo "  make clean                         remove caches and smoke artifacts"
	@echo "  make setup-harness                 install only harness/benchmark"
	@echo "  make setup-codewiki                install only CodeWiki"
	@echo "  make setup-ours                    install only RepoDoc"

# ----- Setup ------------------------------------------------------------------
setup: setup-harness setup-bench setup-codewiki setup-ours

setup-harness:
	$(UV) venv --python 3.12 .venv
	$(UV) pip install --python $(PYTHON) -e ".[dev]"

setup-bench: setup-harness
	$(UV) pip install --python $(PYTHON) -e benchmarks/codewikibench

setup-bench-caw: setup-bench
	$(UV) pip install --python $(PYTHON) -e "benchmarks/codewikibench[caw]"

setup-codewiki:
	$(UV) venv --python 3.12 .venv-codewiki
	$(UV) pip install --python .venv-codewiki/bin/python -e baselines/codewiki

setup-ours:
	$(UV) venv --python 3.12 .venv-ours
	$(UV) pip install --python .venv-ours/bin/python -e "methods/ours[web,dense]"

setup-data: setup-harness
	$(UV) pip install --python $(PYTHON) -e ".[data]"

doctor:
	$(PYTHON) -m harness doctor

list:
	$(PYTHON) -m harness list

# ----- Data -------------------------------------------------------------------
prepare: setup-data
	@test -n "$(REPO)" || (echo "REPO is required, e.g. make prepare REPO=graphrag" >&2; exit 2)
	$(PYTHON) -m harness prepare --repo $(REPO) --from-hf

prepare-core: setup-data
	$(PYTHON) -m harness prepare --split $(SPLIT) --from-hf

# ----- Direct RepoDoc commands ------------------------------------------------
generate:
	@test -n "$(REPO)" || (echo "REPO is required, e.g. make generate REPO=graphrag" >&2; exit 2)
	@test -d "$(REPO_PATH)" || (echo "repository not found: $(REPO_PATH)" >&2; exit 2)
	@mkdir -p "$(dir $(STATE))"
	.venv-ours/bin/repodoc generate \
		--repo "$(REPO_PATH)" \
		--state-dir "$(STATE)" \
		--repo-name "$(REPO_NAME)" \
		--mode full \
		--cluster-mode "$(CLUSTER_MODE)" \
		--workers "$(WORKERS)" \
		--batch-size "$(BATCH_SIZE)"

update:
	@test -n "$(REPO)" || (echo "REPO is required" >&2; exit 2)
	@test -n "$(STATE)" || (echo "STATE is required" >&2; exit 2)
	@test -d "$(REPO_PATH)" || (echo "repository not found: $(REPO_PATH)" >&2; exit 2)
	.venv-ours/bin/repodoc generate \
		--repo "$(REPO_PATH)" \
		--state-dir "$(STATE)" \
		--repo-name "$(REPO_NAME)" \
		--mode update \
		--workers "$(WORKERS)" \
		--batch-size "$(BATCH_SIZE)"

# ----- Benchmark workflow -----------------------------------------------------
run:
	@test -n "$(REPO)" || (echo "REPO is required, e.g. make run REPO=graphrag" >&2; exit 2)
	$(PYTHON) -m harness run \
		--system "$(SYSTEM)" \
		--repo "$(REPO)" \
		--experiment "$(RUN_EXPERIMENT)"

evaluate:
	@test -n "$(REPO)" || (echo "REPO is required" >&2; exit 2)
	$(PYTHON) -m harness evaluate \
		--system "$(SYSTEM)" \
		--repo "$(REPO)" \
		--experiment "$(RUN_EXPERIMENT)"

summary:
	@test -n "$(EXPERIMENT)" || (echo "EXPERIMENT is required" >&2; exit 2)
	$(PYTHON) -m harness summary --experiment "$(EXPERIMENT)"

batch:
	$(PYTHON) -m harness batch --suite "$(SUITE)" --experiment "$(if $(EXPERIMENT),$(EXPERIMENT),$(SUITE))"

# ----- Web UI -----------------------------------------------------------------
web:
	@test -f methods/ours/web/dist/index.html || $(MAKE) web-setup
	.venv-ours/bin/repodoc web --host 127.0.0.1 --port $(PORT)

web-setup:
	cd methods/ours/web && pnpm install --frozen-lockfile && pnpm build

# ----- Quality ----------------------------------------------------------------
test: test-harness test-ours

test-harness:
	$(PYTHON) -m unittest discover -s tests -v

test-ours:
	.venv-ours/bin/python -m unittest discover -s methods/ours/tests -v

lint:
	$(PYTHON) -m ruff check harness tests scripts methods/ours/src methods/ours/tests
	$(PYTHON) -m ruff format --check harness tests scripts methods/ours/src methods/ours/tests

check: test lint doctor

demo: smoke-ours smoke-codewiki

clean:
	rm -rf runs/smoke-codewiki runs/smoke-codewiki-harness runs/smoke-ours
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	rm -rf methods/ours/web/dist
	find . -path './.venv*' -prune -o -path './baselines' -prune -o -path './benchmarks' -prune -o -type d -name '__pycache__' -prune -exec rm -rf {} +

# ----- Advanced / low-level ---------------------------------------------------
smoke-codewiki: smoke-codewiki-core smoke-codewiki-harness

smoke-codewiki-core:
	.venv-codewiki/bin/python scripts/smoke_codewiki_fake.py

smoke-codewiki-harness:
	bash scripts/smoke_harness_codewiki.sh

smoke-ours:
	bash scripts/smoke_ours_toy.sh

llm-check-config:
	.venv-ours/bin/repodoc llm-check --no-call

llm-check:
	.venv-ours/bin/repodoc llm-check

architecture:
	@test -n "$(RESULTS)" || (echo "set RESULTS=<combined_results.json>" >&2; exit 2)
	$(PYTHON) -m harness architecture --results $(RESULTS) $(if $(DOCS),--docs $(DOCS))

architecture-rubrics:
	@test -n "$(RUBRICS)" || (echo "set RUBRICS=<combined_rubrics.json>" >&2; exit 2)
	@test -n "$(OUT)" || (echo "set OUT=<filtered_rubrics.json>" >&2; exit 2)
	$(PYTHON) -m harness architecture-rubrics --rubrics $(RUBRICS) --out $(OUT)

demo-report:
	$(PYTHON) -c 'import sys; from codewikibench.cli import main; sys.exit(main())' report \
		--results benchmarks/codewikibench/examples/OpenHands/codewiki/evaluation_results/combined_evaluation_results.json \
		--format summary
