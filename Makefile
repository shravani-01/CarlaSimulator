.PHONY: help setup setup-ml setup-3d lint type test cpp clean record demo reproduce

help:
	@echo "Targets:"
	@echo "  setup       Install base + dev deps (editable)"
	@echo "  setup-ml    Install deep-learning extras (torch, ultralytics, ...)"
	@echo "  setup-3d    Install 3D/rendering extras (open3d, rerun, gsplat)"
	@echo "  lint        ruff check"
	@echo "  type        mypy"
	@echo "  test        pytest"
	@echo "  cpp         configure + build the C++ core"
	@echo "  record      record a CARLA dataset (needs CARLA running)"
	@echo "  demo        launch the web demo"
	@echo "  reproduce   re-run the headline experiment end-to-end"

setup:
	pip install -e ".[dev]"

setup-ml:
	pip install -e ".[ml]"

setup-3d:
	pip install -e ".[threed]"

lint:
	ruff check .

type:
	mypy perception_py || true

test:
	pytest

cpp:
	cmake -S . -B build_cpp && cmake --build build_cpp

record:
	python -m carla_io.record_dataset

demo:
	streamlit run frontend/app.py

reproduce:
	@echo "TODO: wired up in Phase 4 (classical-vs-splatting relocalization)."
