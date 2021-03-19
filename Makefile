# the default goal is build
.DEFAULT_GOAL := build

# Cleaning means that the package is uninstalled if it is installed.
# Also, all build artifacts are deleted (as they will be later re-created).
clean:
	echo "Cleaning up by first uninstalling moptipy (if installed) and then deleting all auto-generated stuff." && \
	pip uninstall -y moptipy || true && \
	echo "Moptipy is no longer installed; now deleting auto-generated stuff." && \
    rm -rf moptity.egg-info && \
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
	pip install -r requirements.txt && \
	echo "Finished installing required packages from requirements.txt, now installing packages required for development from requirements-dev.txt." && \
	pip install -r requirements-dev.txt && \
	echo "Finished installing requirements from requirements-dev.txt."

# Run the unit tests.
test: init
	echo "Running py.test tests." && \
	py.test tests && \
	echo "Running py.test with doctests." && \
	py.test --doctest-modules && \
    echo "Finished running py.test tests."

# Perform static code analysis.
static_analysis: init
	echo "Runnning static code analysis, starting with flake8." && \
    flake8 . --ignore=W503 && \
    echo "Finished running flake8, now applying pylint to package." &&\
    pylint moptipy --disable=C0103,C0302,C0325,R0201,R0801,R0901,R0902,R0903,R0912,R0913,R0914,R0915,R1728,W0212,W0703 &&\
    echo "Done with pylint, now trying mypy." &&\
    mypy moptipy --no-strict-optional &&\
    echo "Done with mypy, now doing pyflakes." &&\
    python3 -m pyflakes . &&\
    echo "Done with pyflakes, now applying bandit to find security issues." &&\
    bandit -r moptipy &&\
    echo "Done with bandit, now using pep257 to check comments and documentation." &&\
    pep257 . &&\
    echo "Done with pep257, now using pyroma to check setup.py." &&\
    pyroma . &&\
    echo "Done: All static checks passed."

# We use sphinx to generate the documentation.
# This automatically checks the docstrings and such and such.
create_documentation: static_analysis test
	echo "First creating the .rst files from the source code." && \
	sphinx-apidoc -M --ext-autodoc -o docs/source ./moptipy && \
	echo "Now creating the documentation build folder and building the documentation." && \
    sphinx-build -W -a -E -b html docs/source docs/build && \
    echo "Done creating HTML documentation, cleaning up documentation temp files." && \
    mv docs/source/index.rst docs/source/index.tmp && \
    rm -rf docs/source/*.rst && \
    mv docs/source/index.tmp docs/source/index.rst && \
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

# The meta-goal for a full build
build: clean init test static_analysis create_documentation create_distribution install
	echo "The build has completed."

# .PHONY means that the targets init and test are not associated with files.
# see https://stackoverflow.com/questions/2145590
.PHONY: build clean create_distribution create_documentation init install static_analysis test