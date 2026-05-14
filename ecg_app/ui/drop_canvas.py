from pathlib import Path

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..waveform_loader import normalize_input_to_base_path


class DropCanvas(FigureCanvas):
    def __init__(self, figure, drop_callback):
        super().__init__(figure)
        self.setAcceptDrops(True)
        self.drop_callback = drop_callback

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if Path(file_path).suffix.lower() in [".dat", ".hea", ".mwf", ".mat", ".csv", ".xml", ".ecg"]:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if Path(file_path).suffix.lower() in [".dat", ".hea", ".mwf", ".mat", ".csv", ".xml", ".ecg"]:
                base_path = normalize_input_to_base_path(file_path)
                self.drop_callback(base_path)
                return
