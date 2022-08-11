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

# Print the status information.
status:
	echo "working directory: '$(CWD)'." &&\
	echo "moptipy version to build: '$(VERSION)'." &&\
	echo "python package binaries: '$(PYTHON_PACKAGE_BINARIES)'." &&\
	echo "shell: '$(SHELL)'"

# Cleaning means that the package is uninstalled if it is installed.
# Also, all build artifacts are deleted (as they will be later re-created).
clean: status
	echo "Cleaning up by first uninstalling moptipy (if installed) and then deleting all auto-generated stuff." && \
	pip uninstall -y moptipy || true && \
	echo "Moptipy is no longer installed; now deleting auto-generated stuff." && \
	rm -rf moptipy.egg-info && \
	rm -rf .pytest_cache && \
	rm -rf build && \
	rm -rf dist && \
	rm -rf *.whl && \
	rm -rf docs/build && \
	mv docs/source/index.rst docs/source/index.x && \
	rm -rf docs/source/*.rst && \
	mv docs/source/index.x docs/source/index.rst && \
	echo "Done cleaning up, moptipy is uninstalled and auto-generated stuff is deleted."

# Initialization: Install all requirements, both for executing the library and for the tests.
init: clean
	echo "Initialization: first install required packages from requirements.txt." && \
	pip install --no-input --timeout 360 --retries 100 -r requirements.txt && ## nosem \
	echo "Finished installing required packages from requirements.txt, now installing packages required for development from requirements-dev.txt." && \
	pip install --no-input --timeout 360 --retries 100 -r requirements-dev.txt && ## nosem \
	echo "Finished installing requirements from requirements-dev.txt."

# Run the unit tests.
test: init
	echo "Erasing old coverage data." &&\
	coverage erase &&\
	echo "The original value of PATH is '${PATH}'." &&\
	export PATH="${PATH}:${PYTHON_PACKAGE_BINARIES}" &&\
	echo "PATH is now '${PATH}'." &&\
	echo "Running py.test tests." && \
	coverage run --include="moptipy*" -m py.test --strict-config tests -o faulthandler_timeout=360 && \
	echo "Running py.test with doctests." && \
	coverage run --include="moptipy*" -a -m py.test --strict-config --doctest-modules -o faulthandler_timeout=360 --ignore=tests && \
	echo "Finished running py.test tests."

# Perform static code analysis.
static_analysis: init
	echo "The original value of PATH is '${PATH}'." &&\
	export PATH="${PATH}:${PYTHON_PACKAGE_BINARIES}" &&\
	echo "PATH is now '${PATH}'." &&\
	echo "Running static code analysis, starting with flake8." && \
	flake8 . --ignore=W503,TC003,TC101 && \
	echo "Finished running flake8, now applying pylint to package." &&\
	pylint moptipy --disable=C0103,C0302,C0325,R0801,R0901,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R1702,R1728,W0212,W0238,W0703 &&\
	echo "Done with pylint, now trying mypy." &&\
	mypy moptipy --no-strict-optional &&\
	echo "Done with mypy, now doing pyflakes." &&\
	python3 -m pyflakes . &&\
	echo "Done with pyflakes, now applying bandit to find security issues." &&\
	bandit -r moptipy -s B311 &&\
	bandit -r examples -s B311 &&\
	echo "Done with bandit, now using pyroma to check setup.py." &&\
	pyroma . &&\
	echo "Done with pyroma, now applying semgrep." &&\
	(semgrep --error --strict --use-git-ignore --skip-unknown-extensions --optimizations all --config=auto || semgrep --error --strict --use-git-ignore --skip-unknown-extensions --optimizations all --config=auto --verbose) &&\
	echo "Done with semgrep, now applying pydocstyle." &&\
	pydocstyle --convention=pep257 &&\
	echo "Done with pydocstype, now applying tryceratops." &&\
	tryceratops -i TC003 -i TC101 moptipy &&\
	tryceratops -i TC003 -i TC101 examples &&\
	tryceratops -i TC003 -i TC101 tests &&\
	echo "Done with tryceratops, now applying unimport." &&\
	unimport moptipy &&\
	unimport examples &&\
	unimport tests &&\
	echo "Done with unimport, now applying vulture." &&\
	vulture . --min-confidence 61 &&\
	echo "Done with vulture, now applying dodgy." &&\
	dodgy &&\
	echo "Done: All static checks passed."

# We use sphinx to generate the documentation.
# This automatically checks the docstrings and such and such.
create_documentation: static_analysis test
	echo "The original value of PATH is '${PATH}'." &&\
	export PATH="${PATH}:${PYTHON_PACKAGE_BINARIES}" &&\
	echo "PATH is now '${PATH}'." &&\
	echo "First creating the .rst files from the source code." && \
	sphinx-apidoc -M --ext-autodoc -o docs/source ./moptipy && \
	echo "Now creating the documentation build folder and building the documentation." && \
	sphinx-build -W -a -E -b html docs/source docs/build && \
	echo "Done creating HTML documentation, cleaning up documentation temp files." && \
	mv docs/source/index.rst docs/source/index.tmp && \
	rm -rf docs/source/*.rst && \
	rm -rf docs/source/*.md && \
	mv docs/source/index.tmp docs/source/index.rst && \
	echo "Now we pygmentize all the examples in 'examples' to 'build/examples'." &&\
	mkdir -p docs/build/examples &&\
	for f in examples/*.py; do \
		if [ -z "$$f" ]; then \
			echo "Empty module '$$f'?"; \
		else \
			echo "Now pygmentizing example '$$f'." &&\
			{ pygmentize -f html -l python -O full -O style=default -o docs/build/"$${f%.py}.html" "$$f" || exit 1; };\
		fi \
	done &&\
	echo "Finished pygmentizing all examples, now copying LICENSE." &&\
	pygmentize -f html -l text -O full -O style=default -o docs/build/LICENSE.html LICENSE &&\
	echo "Finished copying LICENSE, now creating coverage report." &&\
	mkdir -p docs/build/tc &&\
	coverage html -d docs/build/tc --include="moptipy*" &&\
	echo "Now creating coverage badge." &&\
	coverage-badge -o docs/build/tc/badge.svg &&\
	if [[ -f docs/build/tc/badge.svg ]];then \
		echo "docs/build/tc/badge.svg exists."; \
	else \
		echo "docs/build/tc/badge.svg does not exist!"; exit 1; \
	fi &&\
	echo "Deleting .gitignore file." &&\
	rm -f docs/build/tc/.gitignore &&\
	echo "Done creating coverage data. Now creating .nojekyll files." &&\
	touch "docs/build/.nojekyll" &&\
	touch "docs/build/.doctrees/.nojekyll" &&\
	touch "docs/build/_modules/.nojekyll" &&\
	touch "docs/build/_sources/.nojekyll" &&\
	touch "docs/build/_static/.nojekyll" &&\
	touch "docs/build/tc/.nojekyll" &&\
	touch "docs/build/examples/.nojekyll" &&\
	echo "Done creating the documentation."

# Create different distribution formats, also to check if there is any error.
create_distribution: static_analysis test create_documentation
	echo "Now building source distribution file." &&\
	python3 setup.py check &&\
	python3 -m build &&\
	echo "Done with the build process, now checking result." &&\
	python3 -m twine check dist/* &&\
	echo "Now testing the tar.gz." &&\
	export tempDir=`mktemp -d` &&\
	echo "Created temp directory '$$tempDir'. Creating virtual environment." &&\
	python3 -m venv "$$tempDir" &&\
	echo "Created virtual environment, now activating it." &&\
	source "$$tempDir/bin/activate" &&\
	echo "Now installing tar.gz." &&\
	python3 -m pip --no-input --timeout 360 --retries 100 --require-virtualenv install "$(CWD)/dist/moptipy-$(VERSION).tar.gz" && ## nosem \
	echo "Finished, cleaning up." &&\
	deactivate &&\
	rm -rf "$$tempDir" &&\
	echo "Now testing the wheel." &&\
	export tempDir=`mktemp -d` &&\
	echo "Created temp directory '$$tempDir'. Creating virtual environment." &&\
	python3 -m venv "$$tempDir" &&\
	echo "Created virtual environment, now activating it." &&\
	source "$$tempDir/bin/activate" &&\
	echo "Now installing wheel." &&\
	python3 -m pip --no-input --timeout 360 --retries 100 --require-virtualenv install "$(CWD)/dist/moptipy-$(VERSION)-py3-none-any.whl" && ## nosem \
	echo "Finished, cleaning up." &&\
	deactivate &&\
	rm -rf "$$tempDir" &&\
	echo "Successfully finished building source distribution."

# We install the package and see if that works out.
install: create_distribution
	echo "Now installing moptipy." && \
	pip --no-input --timeout 360 --retries 100 -v install . && \
	echo "Successfully installed moptipy."

# now test applying all the tools to an example dataset
test_tools: install
	echo "Testing all tools." &&\
	export tempDir=`mktemp -d` &&\
	echo "Using temp dir '$$tempDir'." &&\
	cd "$$tempDir" &&\
	echo "Downloading dataset from https://thomasweise.github.io/oa_data/jssp/jssp_hc_swap2.tar.xz." &&\
	(curl -s -o "jssp_hc_swap2.tar.xz" "https://thomasweise.github.io/oa_data/jssp/jssp_hc_swap2.tar.xz" || wget "https://thomasweise.github.io/oa_data/jssp/jssp_hc_swap2.tar.xz")&&\
	echo "Successfully downloaded dataset, now unpacking." &&\
	tar -xf jssp_hc_swap2.tar.xz &&\
	echo "Successfully unpacked, now CDing into directory and applying tools." &&\
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
	echo "Now deleting directory $$tempDir." &&\
	rm -rf "$$tempDir" &&\
	echo "Done checking all tools."

# The meta-goal for a full build
build: status clean init test static_analysis create_documentation create_distribution install test_tools
	echo "The build has completed."

# .PHONY means that the targets init and test are not associated with files.
# see https://stackoverflow.com/questions/2145590
.PHONY: build clean create_distribution create_documentation init install static_analysis status test test_tools
