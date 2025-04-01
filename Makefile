SHELL := /bin/bash

init:
	python3 -m venv .venv
	poetry install
	poetry env info
	@echo "Created virtual environment"

test:
	poetry run pytest --cov=imu_fusion_py/ --cov-report=term-missing --no-cov-on-fail

format:
	ruff format
	ruff check --fix
	poetry run mypy imu_fusion_py/ tests/ --ignore-missing-imports

clean:
	rm -rf .venv
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
	rm -rf juninit-pytest.xml
	find . -name ".coverage*" -delete
	find . -name --pycache__ -exec rm -r {} +
