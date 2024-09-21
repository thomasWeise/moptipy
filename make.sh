#!/bin/bash

# Make the moptipy package.

# strict error handling
set -o pipefail  # trace ERR through pipes
set -o errtrace  # trace ERR through 'time command' and other functions
set -o nounset   # set -u : exit the script if you try to use an uninitialized variable
set -o errexit   # set -e : exit the script if any statement returns a non-true return value

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Welcome to the build script."

export BUILD_SCRIPT="${BASH_SOURCE[0]}"
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): We set the environment variable BUILD_SCRIPT='$BUILD_SCRIPT'."

currentDir="$(pwd)"
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): We are working in directory: '$currentDir'."

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Cleaning up old files."
rm -rf *.whl
find -type d -name "__pycache__" -prune -exec rm -rf {} \;
find -type d -name ".ruff_cache" -prune -exec rm -rf {} \;
rm -rf .mypy_cache
rm -rf .ruff_cache
rm -rf .pytest_cache
rm -rf build
rm -rf dist
rm -rf docs/build
rm -rf docs/source/*.rst
rm -rf moptipy.egg-info
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Done cleaning up old files."

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): We setup a virtual environment in a temp directory."
venvDir="$(mktemp -d)"
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Got temp dir '$venvDir', now creating environment in it."
python3 -m venv --upgrade-deps --copies "$venvDir"

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Activating virtual environment in '$venvDir'."
source "$venvDir/bin/activate"

export PYTHON_INTERPRETER="$venvDir/bin/python3"
oldPythonPath="${PYTHONPATH:-}"
if [ -n "$oldPythonPath" ]; then
  export PYTHONPATH="$currentDir:$oldPythonPath"
else
  export PYTHONPATH="$currentDir"
fi
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): PYTHONPATH='$PYTHONPATH', PYTHON_INTERPRETER='$PYTHON_INTERPRETER'."

cycle=1
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Installing requirements."
while ! ("$PYTHON_INTERPRETER" -m pip install --no-input --default-timeout=300 --timeout=300 --retries=100 -r requirements.txt && "$PYTHON_INTERPRETER" -m pip install --no-input --default-timeout=300 --timeout=300 --retries=100 -r requirements-dev.txt) ; do
    cycle=$((cycle+1))
    if (("$cycle" > 100)) ; then
        echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Something odd is happening: We have performed $cycle cycles of pip install and all failed. That's too many. Let's quit."
        exit 2
    fi
    echo "$(date +'%0Y-%0m-%0d %0R:%0S'): pip install failed, we will try again."
done

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Printing the list of installed packages."
"$PYTHON_INTERPRETER" -m pip freeze

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Now performing unit tests."
"$PYTHON_INTERPRETER" -m pycommons.dev.building.run_tests --package moptipy
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Finished running unit tests."

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Now performing static analysis."
"$PYTHON_INTERPRETER" -m pycommons.dev.building.static_analysis --package moptipy
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Done: All static checks passed."

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Now building documentation."
"$PYTHON_INTERPRETER" -m pycommons.dev.building.make_documentation --root "$currentDir" --package moptipy
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Done building documentation."

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Now building source distribution file."
"$PYTHON_INTERPRETER" -m pycommons.dev.building.make_dist --root "$currentDir" --package moptipy
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Successfully finished building source distribution."

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Now trying to install moptipy."
"$PYTHON_INTERPRETER" -m pip install --no-input --timeout 360 --retries 100 -v "$currentDir"
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Successfully installed moptipy."


echo "$(date +'%0Y-%0m-%0d %0R:%0S'): We have finished the build process."
