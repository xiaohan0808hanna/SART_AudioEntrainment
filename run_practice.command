#!/bin/bash
# Double-click to run the PRACTICE version (free block order, see sart_practice.py).
#
# Runs PsychoPy's bundled Python from Terminal instead of from the PsychoPy app,
# because microphone permission on macOS belongs to the launching app: Terminal
# can be granted it, while some PsychoPy versions can never ask for it.

cd "$(dirname "$0")"

APP=/Applications/PsychoPy.app/Contents
PY="$APP/MacOS/python"

if [ ! -x "$PY" ]; then
    echo "PsychoPy not found at /Applications/PsychoPy.app"
    read -p "Press Enter to close."
    exit 1
fi

# Older (py2app) standalone builds need PYTHONHOME to find their own
# standard library; without it their Python hangs on startup
if [ -f "$APP/Resources/__boot__.py" ]; then
    export PYTHONHOME="$APP/Resources"
fi

# Don't write .pyc caches into PsychoPy.app (macOS blocks and warns about it)
export PYTHONDONTWRITEBYTECODE=1

"$PY" sart_practice.py

echo
read -p "Practice finished. Press Enter to close this window."
