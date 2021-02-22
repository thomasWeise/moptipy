# the default goal is build
.DEFAULT_GOAL := build

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

# Initialization: Install all requirements
init: clean
	echo "Initialization: first install required packages from requirements.txt." && \
	pip install -r requirements.txt && \
	echo "Finished installing required packages from requirements.txt, now installing packages required for development from requirements-dev.txt." && \
	pip install -r requirements-dev.txt && \
	echo "Finished installing requirements from requirements-dev.txt."

# Test: Run py.test
test: init
	echo "Running py.test tests." && \
	py.test tests && \
	echo "Running py.test with doctests." && \
	py.test --doctest-modules && \
    echo "Finished running py.test tests, now applying flake8." && \
    flake8 . --ignore=E501,F401,W503 && \
    echo "Finished running flake8."

create_documentation: test
	echo "First creating the .rst files from the source code." && \
	sphinx-apidoc -M --ext-autodoc -o docs/source ./moptipy && \
	echo "Now creating the documentation build folder and building the documentation." && \
    sphinx-build -W -a -E -b html docs/source docs/build && \
    echo "Done creating HTML documentation, cleaning up documentation temp files." && \
    mv docs/source/index.rst docs/source/index.tmp && \
    rm -rf docs/source/*.rst && \
    mv docs/source/index.tmp docs/source/index.rst && \
    echo "Done creating the documentation."

create_distribution: test
	echo "Now building distribution files and folders." && \
	python3 setup.py check && \
	python3 setup.py sdist && \
	python3 setup.py bdist_wheel && \
	echo "Successfully finished building distribution files and folders."

install: create_distribution
	echo "Now installing moptipy." && \
	pip -v install . && \
	echo "Successfully installed moptipy."

# The meta-goal for a full build
build: clean init test create_documentation create_distribution install
	echo "The build has completed."

# .PHONY means that the targets init and test are not associated with files.
# see https://stackoverflow.com/questions/2145590
.PHONY: build clean create_distribution create_documentation init install test