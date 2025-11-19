"""Entry point for the Eclair photogrammetry helper application."""
from __future__ import annotations

import sys

from PyQt5.QtWidgets import QApplication

from chocolate_eclair.src.gui.main_window import MainWindow


def main() -> None:
    """Launch the Qt application and show the main window."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
