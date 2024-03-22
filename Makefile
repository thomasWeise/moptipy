# the default goal is build
.DEFAULT_GOAL := build

# Set the shell to bash
SHELL := /bin/bash

# Get the location of the Python package binaries.
PYTHON_PACKAGE_BINARIES := $(shell python3 -m site --user-base)/bin

# Get the current working directory
CWD := $(shell pwd)

# Get the moptipy version.
VERSION := $(shell (less '$(CWD)/moptipy/version.py' | sed -n 's/__version__.*=\s*"\(.*\)"/\1/p'))

# Get the current date and time
NOW = $(shell date +'%0Y-%0m-%0d %0R:%0S')

# Print the status information.
status:
	echo "$(NOW): working directory: '$(CWD)'." &&\
	echo "$(NOW): moptipy version to build: '$(VERSION)'." &&\
	echo "$(NOW): python package binaries: '$(PYTHON_PACKAGE_BINARIES)'." &&\
	echo "$(NOW): shell: '$(SHELL)'"

# Cleaning means that the package is uninstalled if it is installed.
# Also, all build artifacts are deleted (as they will be later re-created).
clean: status
	echo "$(NOW): Cleaning up by first uninstalling moptipy (if installed) and then deleting all auto-generated stuff." && \
	pip uninstall -y moptipy || true && \
	echo "$(NOW): Moptipy is no longer installed; now deleting auto-generated stuff." && \
	rm -rf *.whl && \
	find -type d -name "__pycache__" -prune -exec rm -rf {} \; &&\
	rm -rf .mypy_cache &&\
	rm -rf .ruff_cache &&\
	rm -rf .pytest_cache && \
	rm -rf build && \
	rm -rf dist && \
	rm -rf docs/build && \
	rm -rf docs/source/*.rst && \
	rm -rf moptipy.egg-info && \
	echo "$(NOW): Done cleaning up, moptipy is uninstalled and auto-generated stuff is deleted."

# Initialization: Install all requirements, both for executing the library and for the tests.
init: clean
	echo "$(NOW): Initialization: first install required packages from requirements.txt." && \
	pip install --no-input --timeout 360 --retries 100 -r requirements.txt && ## nosem \
	echo "$(NOW): Finished installing required packages from requirements.txt, now installing packages required for development from requirements-dev.txt." && \
	pip install --no-input --timeout 360 --retries 100 -r requirements-dev.txt && ## nosem \
	echo "$(NOW): Finished installing requirements from requirements-dev.txt, now printing all installed packages." &&\
	pip freeze &&\
	echo "$(NOW): Finished printing all installed packages."

test: init
	echo "$(NOW): Now performing unit tests." &&\
	python3 -m pycommons.dev.building.run_tests --package moptipy &&\
	echo "$(NOW): Finished running unit tests."

static_analysis: init
	echo "$(NOW): Now performing static analysis." &&\
	python3 -m pycommons.dev.building.static_analysis --package moptipy &&\
	echo "$(NOW): Done: All static checks passed."

create_documentation: static_analysis test
	echo "$(NOW): Now building documentation." &&\
	python3 -m pycommons.dev.building.make_documentation --root . --package moptipy &&\
	echo "$(NOW): Done building documentation."

create_distribution: static_analysis test create_documentation
	echo "$(NOW): Now building source distribution file." &&\
	export PYTHONPATH=".:${PYTHONPATH}" &&\
	python3 -m pycommons.dev.building.make_dist --root . --package moptipy &&\
	echo "$(NOW): Successfully finished building source distribution."

# We install the package and see if that works out.
install: create_distribution
	echo "$(NOW): Now installing moptipy." && \
	pip --no-input --timeout 360 --retries 100 -v install . && \
	echo "$(NOW): Successfully installed moptipy."

# now test applying all the tools to an example dataset
test_tools: install
	echo "$(NOW): Testing all tools." &&\
	export tempDir=`mktemp -d` &&\
	echo "$(NOW): Using temporary directory '$$tempDir'." &&\
	cd "$$tempDir" &&\
	echo "$(NOW): Downloading dataset from https://thomasweise.github.io/oa_data/jssp/jssp_hc_swap2.tar.xz." &&\
	(curl -s -o "jssp_hc_swap2.tar.xz" "https://thomasweise.github.io/oa_data/jssp/jssp_hc_swap2.tar.xz" || wget "https://thomasweise.github.io/oa_data/jssp/jssp_hc_swap2.tar.xz")&&\
	echo "$(NOW): Successfully downloaded dataset, now unpacking." &&\
	tar -xf jssp_hc_swap2.tar.xz &&\
	echo "$(NOW): Successfully unpacked, now CDing into directory and applying tools." &&\
	cd "$$tempDir/jssp" &&\
	python3 -m moptipy.evaluation.end_results "$$tempDir/jssp/results" "$$tempDir/jssp/evaluation/end_results.txt" &&\
	ls "$$tempDir/jssp/evaluation/end_results.txt" &&\
	python3 -m moptipy.evaluation.end_statistics "$$tempDir/jssp/evaluation/end_results.txt" "$$tempDir/jssp/evaluation/end_statistics_1.txt" &&\
	ls "$$tempDir/jssp/evaluation/end_statistics_1.txt" &&\
	python3 -m moptipy.evaluation.end_statistics "$$tempDir/jssp/results" "$$tempDir/jssp/evaluation/end_statistics_2.txt" &&\
	ls "$$tempDir/jssp/evaluation/end_statistics_2.txt" &&\
	python3 -m moptipy.evaluation.ioh_analyzer "$$tempDir/jssp/results" "$$tempDir/jssp/ioh" &&\
	ls "$$tempDir/jssp/ioh" &&\
	cd "$(CWD)" &&\
	echo "$(NOW): Now deleting directory $$tempDir." &&\
	rm -rf "$$tempDir" &&\
	echo "$(NOW): Done checking all tools."

# The meta-goal for a full build
build: status clean init test static_analysis create_documentation create_distribution install test_tools
	echo "$(NOW): The build has completed."

# .PHONY means that the targets init and test are not associated with files.
# see https://stackoverflow.com/questions/2145590
.PHONY: build clean create_distribution create_documentation init install static_analysis status test test_tools
