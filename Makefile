# the default goal is build
.DEFAULT_GOAL := build

# Get the location of the Python package binaries.
PYTHON_PACKAGE_BINARIES := $(shell python3 -m site --user-base)/bin

# Get the current working directory
CWD := $(shell pwd)

# Cleaning means that the package is uninstalled if it is installed.
# Also, all build artifacts are deleted (as they will be later re-created).
clean:
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
	pip install -r requirements.txt && ## nosem \
	echo "Finished installing required packages from requirements.txt, now installing packages required for development from requirements-dev.txt." && \
	pip install -r requirements-dev.txt && ## nosem \
	echo "Finished installing requirements from requirements-dev.txt."

# Run the unit tests.
test: init
	echo "The original value of PATH is '${PATH}'." &&\
	export PATH="${PATH}:${PYTHON_PACKAGE_BINARIES}" &&\
	echo "PATH is now '${PATH}'." &&\
	echo "Running py.test tests." && \
	py.test --strict-config tests && \
	echo "Running py.test with doctests." && \
	py.test --strict-config --doctest-modules && \
    echo "Finished running py.test tests."

# Perform static code analysis.
static_analysis: init
	echo "The original value of PATH is '${PATH}'." &&\
	export PATH="${PATH}:${PYTHON_PACKAGE_BINARIES}" &&\
	echo "PATH is now '${PATH}'." &&\
	echo "Running static code analysis, starting with flake8." && \
    flake8 . --ignore=W503 && \
    echo "Finished running flake8, now applying pylint to package." &&\
    pylint moptipy --disable=C0103,C0302,C0325,R0201,R0801,R0901,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R1702,R1728,W0212,W0238,W0703 &&\
    echo "Done with pylint, now trying mypy." &&\
    mypy moptipy --no-strict-optional &&\
    echo "Done with mypy, now doing pyflakes." &&\
    python3 -m pyflakes . &&\
    echo "Done with pyflakes, now applying bandit to find security issues." &&\
    bandit -r moptipy -s B311 &&\
    echo "Done with bandit, now using pyroma to check setup.py." &&\
    pyroma . &&\
    echo "Done with pyroma, now applying semgrep." &&\
    (semgrep --error --strict --use-git-ignore --skip-unknown-extensions --optimizations all --config=auto || semgrep --error --strict --use-git-ignore --skip-unknown-extensions --optimizations all --config=auto --verbose) &&\
    echo "Done with semgrep, now applying pydocstyle." &&\
    pydocstyle --convention=pep257 &&\
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
    pygmentize -f html -l python -O full -O style=default -o docs/build/LICENSE.html LICENSE &&\
    echo "Done creating the documentation."

# Create different distribution formats, also to check if there is any error.
create_distribution: static_analysis test create_documentation
	echo "Now building distribution files and folders." && \
	python3 setup.py check && \
	python3 setup.py sdist && \
	python3 setup.py bdist_wheel && \
	echo "Successfully finished building distribution files and folders."

# We install the package and see if that works out.
install: create_distribution
	echo "Now installing moptipy." && \
	pip -v install . && \
	echo "Successfully installed moptipy."

# Now we run all examples
run_examples: install
	echo "Now we execute all the examples in 'examples'." &&\
    for f in examples/*.py; do \
    	if [ -z "$$f" ]; then \
  			echo "Empty module '$$f'?"; \
	  	else \
			echo "Now running example '$$f'." &&\
			{ python3 "$$f" || exit 1; };\
		fi \
    done &&\
    echo "Finished executing all examples."

# now test applying all the tools to an example dataset
test_tools: install
	echo "Testing all tools." &&\
    export tempDir=`mktemp -d` &&\
    echo "Using temp dir '$$tempDir'." &&\
    cd "$$tempDir" &&\
    echo "Downloading dataset from https://thomasweise.github.io/oa_data/jssp/jssp_hc_swap2.zip." &&\
    (curl -s -o "jssp_hc_swap2.zip" "https://thomasweise.github.io/oa_data/jssp/jssp_hc_swap2.zip" || wget "https://thomasweise.github.io/oa_data/jssp/jssp_hc_swap2.zip")&&\
    echo "Successfully downloaded dataset, now unpacking." &&\
    unzip jssp_hc_swap2.zip &&\
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
build: clean init test static_analysis create_documentation create_distribution install run_examples test_tools
	echo "The build has completed."

# .PHONY means that the targets init and test are not associated with files.
# see https://stackoverflow.com/questions/2145590
.PHONY: build clean create_distribution create_documentation init install run_examples static_analysis test test_tools
