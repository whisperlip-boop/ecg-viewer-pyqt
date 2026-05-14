import ctypes
import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication

if __package__:
    from .resources import resource_path
    from .ui.main_window import ECGViewer
else:
    from ecg_app.resources import resource_path
    from ecg_app.ui.main_window import ECGViewer


def run():
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "vuno.ecgviewer.desktop.v1"
        )

    app = QApplication(sys.argv)
    icon = QIcon(resource_path("ecg.ico"))
    app.setWindowIcon(icon)

    viewer = ECGViewer()
    viewer.setWindowIcon(icon)
    viewer.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(run())
