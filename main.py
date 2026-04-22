"""
EMS Control GUI - Entry Point
Run this file to start the application



Error usually occurs on Mac:

"
qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in "/Users/gafurmammadov/Documents/Uchicago_classes/practicum/ems_lib/ems-main/ems/app_dev/.practicumvenv/lib/python3.13/site-packages/PyQt6/Qt6/plugins"
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

zsh: abort      python3 main.py
"

May need to run like this:
export QT_QPA_PLATFORM_PLUGIN_PATH="$(python3 -c "import PyQt6, pathlib; print(pathlib.Path(PyQt6.__file__).parent/'Qt6'/'plugins'/'platforms')")"
python3 main.py

Worst case reinstall venv and run again:
Erase .venv folder
run pip install -r requirements 
"""


import sys
import os
from pathlib import Path

# Fix Qt plugin path BEFORE importing any Qt modules
try:
    import PyQt6
    plugin_path = Path(PyQt6.__file__).parent / 'Qt6' / 'plugins'
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(plugin_path)
except ImportError:
    pass

from PyQt6.QtWidgets import QApplication
from ui.main_window import EMSWindow


def main():
    """Launch the EMS Control GUI application"""
    app = QApplication(sys.argv)
    window = EMSWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()