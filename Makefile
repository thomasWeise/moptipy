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
	rm -rf moptipy.egg-info && \
	rm -rf .pytest_cache && \
	rm -rf build && \
	rm -rf dist && \
	rm -rf *.whl && \
	rm -rf docs/build && \
	mv docs/source/index.rst docs/source/index.x && \
	rm -rf docs/source/*.rst && \
	mv docs/source/index.x docs/source/index.rst && \
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


# Run the unit tests.
test: init
	echo "$(NOW): Erasing old coverage data." &&\
	coverage erase &&\
	echo "$(NOW): The original value of PATH is '${PATH}'." &&\
	export PATH="${PATH}:${PYTHON_PACKAGE_BINARIES}" &&\
	echo "$(NOW): PATH is now '${PATH}'." &&\
	echo "$(NOW): Running pytest tests." && \
	coverage run --include="moptipy/*" -m pytest --strict-config tests -o faulthandler_timeout=1800 --ignore=examples && \
	echo "$(NOW): Running pytest with doctests." && \
	coverage run -a --include="moptipy/*" -m pytest --strict-config --doctest-modules -o faulthandler_timeout=720 --ignore=tests --ignore=examples && \
	echo "$(NOW): Finished running pytest tests."

# Perform static code analysis.
static_analysis: init
	echo "$(NOW): The original value of PATH is '${PATH}'." &&\
	export PATH="${PATH}:${PYTHON_PACKAGE_BINARIES}" &&\
	echo "$(NOW): PATH is now '${PATH}'." &&\
	echo "$(NOW): Running static code analysis, starting with flake8." && \
	flake8 . --ignore=,B008,B009,B010,DUO102,TC003,TC101,W503 && \
	echo "$(NOW): Finished running flake8, now applying pylint to package." &&\
	pylint moptipy --disable=C0103,C0302,C0325,R0801,R0901,R0902,R0903,R0911,R0912,R0913,R0914,R0915,R1702,R1728,W0212,W0238,W0703 &&\
	echo "$(NOW): Done with pylint, now trying mypy." &&\
	mypy moptipy --no-strict-optional --check-untyped-defs &&\
	echo "$(NOW): Done with mypy, now doing pyflakes." &&\
	python3 -m pyflakes . &&\
	echo "$(NOW): Done with pyflakes, now applying bandit to find security issues." &&\
	bandit -r moptipy -s B311 &&\
	bandit -r examples -s B311 &&\
	bandit -r tests -s B311,B101 &&\
	echo "$(NOW): Done with bandit, now using pyroma to check setup.py." &&\
	pyroma . &&\
	echo "$(NOW): Done with pyroma, now applying semgrep." &&\
	(semgrep --error --strict --use-git-ignore --skip-unknown-extensions --optimizations all --config=auto || semgrep --error --strict --use-git-ignore --skip-unknown-extensions --optimizations all --config=auto --verbose) &&\
	echo "$(NOW): Done with semgrep, now applying pydocstyle." &&\
	pydocstyle --convention=pep257 &&\
	echo "$(NOW): Done with pydocstype, now applying tryceratops." &&\
	tryceratops -i TC003 -i TC101 moptipy &&\
	tryceratops -i TC003 -i TC101 examples &&\
	tryceratops -i TC003 -i TC101 tests &&\
	echo "$(NOW): Done with tryceratops, now applying unimport." &&\
	unimport moptipy &&\
	unimport examples &&\
	unimport tests &&\
	echo "$(NOW): Done with unimport, now applying vulture." &&\
	vulture . --min-confidence 61 &&\
	echo "$(NOW): Done with vulture, now applying dodgy." &&\
	dodgy &&\
	echo "$(NOW): Done with dodgy, now running pycodestyle." &&\
	pycodestyle moptipy &&\
	pycodestyle --ignore=E731,W503 examples &&\
	pycodestyle tests &&\
	echo "$(NOW): Done with pycodestyle, now running ruff." &&\
	ruff --target-version py310 --select A,ANN,B,C,C4,COM,D,DJ,DTZ,E,ERA,EXE,F,G,I,ICN,INP,ISC,N,NPY,PIE,PLC,PLE,PLR,PLW,PT,PYI,Q,RET,RSE,RUF,S,SIM,T,T10,T20,TID,TRY,UP,W,YTT --ignore=ANN001,ANN002,ANN003,ANN101,ANN204,ANN401,B008,B009,B010,C901,D203,D208,D212,D401,D407,D413,N801,PLR0911,PLR0912,PLR0913,PLR0915,PLR2004,RUF100,TRY003,UP035 --line-length 79 moptipy &&\
	ruff --target-version py310 --select A,ANN,B,C,C4,COM,D,DJ,DTZ,E,ERA,EXE,F,G,I,ICN,ISC,N,NPY,PIE,PLC,PLE,PLR,PLW,PT,PYI,Q,RET,RSE,RUF,S,SIM,T10,TID,TRY,UP,W,YTT --ignore=ANN001,ANN002,ANN003,ANN101,ANN204,ANN401,B008,B009,B010,C901,D203,D208,D212,D401,D407,D413,N801,PLR0911,PLR0912,PLR0913,PLR0915,PLR2004,RUF100,TRY003,UP035 --line-length 79 examples &&\
	ruff --target-version py310 --select A,ANN,B,C,C4,COM,D,DJ,DTZ,E,ERA,EXE,F,G,I,ICN,ISC,N,NPY,PIE,PLC,PLE,PLR,PLW,PYI,Q,RET,RSE,RUF,T,SIM,T10,T20,TID,TRY,UP,W,YTT --ignore=ANN001,ANN002,ANN003,ANN101,ANN204,ANN401,B008,B009,B010,C901,D203,D208,D212,D401,D407,D413,N801,PLR0911,PLR0912,PLR0913,PLR0915,PLR2004,RUF100,TRY003,UP035 --line-length 79 tests &&\
    echo "$(NOW): Done with ruff, now running autoflake." &&\
    autoflake -c -r moptipy &&\
    autoflake -c -r tests &&\
    autoflake -c -r examples &&\
	echo "$(NOW): Done: All static checks passed."

# We use sphinx to generate the documentation.
# This automatically checks the docstrings and such and such.
create_documentation: static_analysis test
	echo "$(NOW): The original value of PATH is '${PATH}'." &&\
	export PATH="${PATH}:${PYTHON_PACKAGE_BINARIES}" &&\
	echo "$(NOW): PATH is now '${PATH}'." &&\
	echo "$(NOW): First creating the .rst files from the source code." && \
	sphinx-apidoc -M --ext-autodoc -o docs/source ./moptipy && \
	echo "$(NOW): Now creating the documentation build folder and building the documentation." && \
	sphinx-build -W -a -E -b html docs/source docs/build && \
	echo "$(NOW): Done creating HTML documentation, cleaning up documentation temp files." && \
	mv docs/source/index.rst docs/source/index.tmp && \
	rm -rf docs/source/*.rst && \
	rm -rf docs/source/*.md && \
	mv docs/source/index.tmp docs/source/index.rst && \
	echo "$(NOW): Now we pygmentize all the examples in 'examples' to 'build/examples'." &&\
	mkdir -p docs/build/examples &&\
	for f in examples/*.py; do \
		if [ -z "$$f" ]; then \
			echo "$(NOW): Empty module '$$f'?"; \
		else \
			echo "$(NOW): Now pygmentizing example '$$f'." &&\
			{ pygmentize -f html -l python3 -O full -O style=default -o docs/build/"$${f%.py}.html" "$$f" || exit 1; };\
		fi \
	done &&\
	echo "$(NOW): Finished pygmentizing all examples, now copying LICENSE and other files." &&\
	pygmentize -f html -l text -O full -O style=default -o docs/build/LICENSE.html LICENSE &&\
	pygmentize -f html -l text -O full -O style=default -o docs/build/requirements.html requirements.txt &&\
	pygmentize -f html -l text -O full -O style=default -o docs/build/requirements-dev.html requirements-dev.txt &&\
	pygmentize -f html -l make -O full -O style=default -o docs/build/Makefile.html Makefile &&\
	echo "$(NOW): Finished copying LICENSE, now creating coverage report." &&\
	mkdir -p docs/build/tc &&\
	coverage html -d docs/build/tc --include="moptipy/*" &&\
	echo "$(NOW): Now creating coverage badge." &&\
	coverage-badge -o docs/build/tc/badge.svg &&\
	if [[ -f docs/build/tc/badge.svg ]];then \
		echo "$(NOW): docs/build/tc/badge.svg exists."; \
	else \
		echo "$(NOW): docs/build/tc/badge.svg does not exist!"; exit 1; \
	fi &&\
	echo "$(NOW): Deleting .gitignore file." &&\
	rm -f docs/build/tc/.gitignore &&\
	echo "$(NOW): Deleting useless _sources." &&\
	rm -rf docs/build/_sources &&\
	echo "$(NOW): Now rendering additional files." &&\
	export PART_A='<!DOCTYPE html><html><title>' &&\
	export PART_B='</title><link href=_static/bizstyle.css rel=stylesheet><body style="background-image:none"><div class=document><div class=documentwrapper><div class=bodywrapper><div class=body role=main><section>' &&\
	export PART_C='</section></div></div></div></div></body></html>' &&\
	export BASE_URL='https\:\/\/thomasweise\.github\.io\/moptipy\/' &&\
	echo "$${PART_A}Contributing to moptipy$${PART_B}$(shell (python3 -m markdown -o html ./CONTRIBUTING.md))$$PART_C" > ./docs/build/CONTRIBUTING.html &&\
	sed -i "s/\"$$BASE_URL/\".\//g" ./docs/build/CONTRIBUTING.html &&\
	sed -i "s/=$$BASE_URL/=.\//g" ./docs/build/CONTRIBUTING.html &&\
	echo "$${PART_A}Security Policy of moptipy$${PART_B}$(shell (python3 -m markdown -o html ./SECURITY.md))$$PART_C" > ./docs/build/SECURITY.html &&\
	sed -i "s/\"$$BASE_URL/\".\//g" ./docs/build/SECURITY.html &&\
	sed -i "s/=$$BASE_URL/=.\//g" ./docs/build/SECURITY.html &&\
	echo "$(NOW): Now minifying all html files." &&\
	cd "docs/build/" &&\
	find -type f -name "*.html" -not -path "./tc/*" -exec python3 -c "print('{}');import minify_html;f=open('{}','r');s=f.read();f.close();s=minify_html.minify(s,do_not_minify_doctype=True,ensure_spec_compliant_unquoted_attribute_values=True,keep_html_and_head_opening_tags=False,minify_css=True,minify_js=True,remove_bangs=True,remove_processing_instructions=True);f=open('{}','w');f.write(s);f.close()" \; &&\
	cd "../../" &&\
	echo "$(NOW): Done creating coverage data. Now creating .nojekyll files." &&\
	cd "docs/build/" &&\
	find -type d -exec touch "{}/.nojekyll" \;
	cd "../../" &&\
	echo "$(NOW): Done creating the documentation."

# Create different distribution formats, also to check if there is any error.
create_distribution: static_analysis test create_documentation
	echo "$(NOW): Now building source distribution file." &&\
	python3 setup.py check &&\
	python3 -m build &&\
	echo "$(NOW): Done with the build process, now checking result." &&\
	python3 -m twine check dist/* &&\
	echo "$(NOW): Now testing the tar.gz." &&\
	export tempDir=`mktemp -d` &&\
	echo "$(NOW): Created temp directory '$$tempDir'. Creating virtual environment." &&\
	python3 -m venv "$$tempDir" &&\
	echo "$(NOW): Created virtual environment, now activating it." &&\
	source "$$tempDir/bin/activate" &&\
	echo "$(NOW): Now installing tar.gz." &&\
	python3 -m pip --no-input --timeout 360 --retries 100 --require-virtualenv install "$(CWD)/dist/moptipy-$(VERSION).tar.gz" && ## nosem \
	echo "$(NOW): Installing tar.gz has worked. We now create the list of packages in this environment via pip freeze." &&\
	pip freeze > "$(CWD)/dist/moptipy-$(VERSION)-requirements_frozen.txt" &&\
	echo "$(NOW): Now fixing moptipy line in requirements file." &&\
	sed -i "s/^moptipy.*/moptipy==$(VERSION)/" "$(CWD)/dist/moptipy-$(VERSION)-requirements_frozen.txt" &&\
	echo "$(NOW): Now we deactivate the environment." &&\
	deactivate &&\
	rm -rf "$$tempDir" &&\
	echo "$(NOW): Now testing the wheel." &&\
	export tempDir=`mktemp -d` &&\
	echo "$(NOW): Created temp directory '$$tempDir'. Creating virtual environment." &&\
	python3 -m venv "$$tempDir" &&\
	echo "$(NOW): Created virtual environment, now activating it." &&\
	source "$$tempDir/bin/activate" &&\
	echo "$(NOW): Now installing wheel." &&\
	python3 -m pip --no-input --timeout 360 --retries 100 --require-virtualenv install "$(CWD)/dist/moptipy-$(VERSION)-py3-none-any.whl" && ## nosem \
	echo "$(NOW): Now we deactivate the environment." &&\
	deactivate &&\
	echo "$(NOW): Finished, cleaning up." &&\
	rm -rf "$$tempDir" &&\
	echo "$(NOW): Now also packaging the documentation." &&\
	cd docs/build &&\
	tar --dereference --exclude=".nojekyll" -c * | xz -v -9e -c > "$(CWD)/dist/moptipy-$(VERSION)-documentation.tar.xz" &&\
	cd $(CWD) &&\
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
