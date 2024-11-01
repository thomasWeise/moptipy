#!/bin/bash

# Make the moptipy package virtual environment.

# strict error handling
set -o pipefail  # trace ERR through pipes
set -o errtrace  # trace ERR through 'time command' and other functions
set -o nounset   # set -u : exit the script if you try to use an uninitialized variable
set -o errexit   # set -e : exit the script if any statement returns a non-true return value

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Welcome to the virtual environment creation script."

currentDir="$(pwd)"
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): We are working in directory: '$currentDir'."

venvDir="$currentDir/.venv"
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): We setup a virtual environment in directory '$venvDir'."

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): First, we clean up the directory."
rm -rf "$venvDir" || true
mkdir -p "$venvDir"

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Now we create the environment."
python3 -m venv --upgrade-deps --copies "$venvDir"

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Activating virtual environment in '$venvDir'."
source "$venvDir/bin/activate"

export PYTHON_INTERPRETER="$venvDir/bin/python3"
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): PYTHON_INTERPRETER='$PYTHON_INTERPRETER'."

cycle=1
echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Installing requirements."
while ! (timeout --kill-after=60m 58m "$PYTHON_INTERPRETER" -m pip install --no-input --default-timeout=300 --timeout=300 --retries=100 -r requirements.txt && timeout --kill-after=60m 58m "$PYTHON_INTERPRETER" -m pip install --no-input --default-timeout=300 --timeout=300 --retries=100 -r requirements-dev.txt) ; do
    cycle=$((cycle+1))
    if (("$cycle" > 100)) ; then
        echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Something odd is happening: We have performed $cycle cycles of pip install and all failed. That's too many. Let's quit."
        exit 2  # A non-zero exit code indicates failure.
    fi
    echo "$(date +'%0Y-%0m-%0d %0R:%0S'): pip install failed, we will try again."
done

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): Printing the list of installed packages."
"$PYTHON_INTERPRETER" -m pip freeze

echo "$(date +'%0Y-%0m-%0d %0R:%0S'): We have finished the environment making process."
