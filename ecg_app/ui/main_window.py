import os

import numpy as np
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ..constants import (
    GRID_COLOR,
    GRID_MAJOR_LINESTYLE,
    GRID_MAJOR_WIDTH,
    GRID_MINOR_ALPHA,
    GRID_MINOR_LINESTYLE,
    GRID_MINOR_WIDTH,
    MAIN_Y_LIM,
    MV_PER_MM,
    PAPER_FACE_COLOR,
    SEC_PER_MM,
    Y_MAJOR_STEP,
)
from ..resources import resource_path
from ..signal_processing import remove_baseline_wander, safe_bandpass_filter
from ..waveform_loader import load_waveform, normalize_input_to_base_path
from ..loaders.fukuda import load_fukuda_measurements
from ..loaders.xml_mac2000 import load_mac2000_measurements
from ..loaders.xml_muse import load_muse_measurements
from ..loaders.xml_bionet import load_bionet_measurements
from ..loaders.xml_philips import load_philips_measurements
from ..loaders.xml_trismed import load_trismed_measurements
from ..loaders.xml_nihonkohden import load_nihonkohden_measurements
from ..loaders.xml_common import detect_xml_type
from ..exporters import export_csv, export_mat_hea, export_muse_xml, export_wfdb
from .drop_canvas import DropCanvas
from .lead_zoom_dialog import LeadZoomDialog


class ECGViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECG Viewer")
        self.current_path = None
        self.current_signal = None
        self.current_fields = None
        self.ax_lead_map = {}
        self.zoom_dialogs = []

        self.setGeometry(100, 100, 1350, 1380)

        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.load_button = QPushButton("Open ECG File (.dat / .hea / .mat / .mwf / .csv / .xml / .ecg)", self)
        self.load_button.clicked.connect(self.load_ecg)
        self.layout.addWidget(self.load_button)

        metrics_row = QWidget()
        metrics_row.setFixedHeight(52)
        metrics_row_layout = QHBoxLayout(metrics_row)
        metrics_row_layout.setContentsMargins(0, 0, 0, 0)
        metrics_row_layout.setSpacing(0)

        self.metrics_label = QLabel("")
        self.metrics_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.metrics_label.setStyleSheet(
            "background-color: #F0F0F0;"
            "color: #222222;"
            "font-family: monospace;"
            "font-size: 14px;"
            "font-weight: bold;"
            "padding: 3px 8px;"
        )
        metrics_row_layout.addWidget(self.metrics_label, stretch=1)

        converter_widget = QWidget()
        converter_widget.setObjectName("converterWidget")
        converter_widget.setStyleSheet(
            "QWidget#converterWidget { background-color: #F0F0F0; }"
        )
        converter_widget.setFixedWidth(170)
        converter_layout = QVBoxLayout(converter_widget)
        converter_layout.setContentsMargins(4, 2, 4, 2)
        converter_layout.setSpacing(2)

        self.convert_format_combo = QComboBox()
        self.convert_format_combo.addItems([
            "CSV (.csv)",
            "WFDB (.dat + .hea)",
            "MAT (.mat + .hea)",
            "MUSE XML (.xml)",
        ])
        self.convert_format_combo.setEnabled(False)
        converter_layout.addWidget(self.convert_format_combo)

        self.convert_button = QPushButton("Convert")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self._on_convert)
        converter_layout.addWidget(self.convert_button)

        metrics_row_layout.addWidget(converter_widget)
        self.layout.addWidget(metrics_row)

        self.stack_layout = QStackedLayout()
        self.layout.addLayout(self.stack_layout)

        self.figure = Figure(figsize=(20, 16))
        self.canvas = DropCanvas(self.figure, self.load_and_plot)
        self.stack_layout.addWidget(self.canvas)

        self.canvas.mpl_connect("button_press_event", self.on_plot_click)

        self.spinner_widget = QWidget()
        self.spinner_layout = QVBoxLayout(self.spinner_widget)
        self.spinner_layout.setAlignment(Qt.AlignCenter)

        self.spinner_label = QLabel()
        self.spinner_label.setAlignment(Qt.AlignCenter)
        self.spinner_movie = QMovie(resource_path("spinner.gif"))

        if self.spinner_movie.isValid():
            self.spinner_movie.setScaledSize(QSize(64, 64))
            self.spinner_label.setMovie(self.spinner_movie)
        else:
            self.spinner_label.setText("Loading...")
            self.spinner_label.setStyleSheet("font-size: 18px; color: gray;")

        self.spinner_layout.addWidget(self.spinner_label)
        self.stack_layout.addWidget(self.spinner_widget)

        self.canvas_text = self.figure.text(
            0.5,
            0.5,
            "Drag and Drop ECG File Here",
            ha="center",
            va="center",
            fontsize=20,
            alpha=0.3,
        )

    def show_error(self, message):
        QMessageBox.critical(self, "File Load Error", message)

    def _update_metrics(self, fields):
        source = fields.get("source")
        try:
            if source == "fukuda_ecg":
                m = load_fukuda_measurements(fields["ecg_path"])
                line1 = (
                    f"HR: {m['hr']} bpm    "
                    f"R-R: {m['rr']:.3f} s    "
                    f"P-R: {m['pr']:.3f} s    "
                    f"QRS: {m['qrs']:.3f} s    "
                    f"QT: {m['qt']:.3f} s    "
                    f"QTc: {m['qtc']:.3f}    "
                    f"AXIS: {m['axis']} deg"
                )
                line2 = (
                    f"SV1: {m['sv1']:.2f} mV    "
                    f"RV6: {m['rv6']:.2f} mV    "
                    f"R+S: {m['rs']:.2f} mV"
                )
                self.metrics_label.setText(f"{line1}\n{line2}")

            elif source in ("mac2000_xml", "muse_xml", "bionet_xml", "philips_xml"):
                if source == "mac2000_xml":
                    m = load_mac2000_measurements(fields["xml_path"])
                elif source == "muse_xml":
                    m = load_muse_measurements(fields["xml_path"])
                elif source == "bionet_xml":
                    m = load_bionet_measurements(fields["xml_path"])
                else:
                    m = load_philips_measurements(fields["xml_path"])
                hr  = f"{m['hr']} BPM" if m.get('hr')  is not None else "?"
                pr  = f"{m['pr']} ms"  if m.get('pr')  is not None else "?"
                qrs = f"{m['qrs']} ms" if m.get('qrs') is not None else "?"
                qt  = f"{m['qt']} ms"  if m.get('qt')  is not None else "?"
                qtc = f"{m['qtc']} ms" if m.get('qtc') is not None else "?"
                pa  = f"{m['p_axis']} deg" if m.get('p_axis') is not None else "?"
                ra  = f"{m['r_axis']} deg" if m.get('r_axis') is not None else "?"
                ta  = f"{m['t_axis']} deg" if m.get('t_axis') is not None else "?"
                line1 = (
                    f"Vent. rate: {hr}    "
                    f"PR interval: {pr}    "
                    f"QRS duration: {qrs}    "
                    f"QT/QTc: {qt}/{qtc}"
                )
                line2 = f"P-R-T axes: {pa}  {ra}  {ta}"
                self.metrics_label.setText(f"{line1}\n{line2}")

            elif source == "mwf":
                companion = fields.get("companion_xml_path")
                if not companion or detect_xml_type(companion) != "nihonkohden":
                    self.metrics_label.setText("")
                    return
                m = load_nihonkohden_measurements(companion)
                hr  = f"{m['hr']} bpm"  if m.get('hr')    is not None else "?"
                pr  = f"{m['pr']} ms"   if m.get('pr')    is not None else "?"
                qrs = f"{m['qrs']} ms"  if m.get('qrs')   is not None else "?"
                qt  = f"{m['qt']} ms"   if m.get('qt')    is not None else "?"
                qtc = f"{m['qtc']} ms"  if m.get('qtc')   is not None else "?"
                pa  = f"{m['p_axis']}"  if m.get('p_axis') is not None else "?"
                ra  = f"{m['r_axis']}"  if m.get('r_axis') is not None else "?"
                ta  = f"{m['t_axis']}"  if m.get('t_axis') is not None else "?"
                rv5 = f"{m['rv5']:.3f} mV" if m.get('rv5') is not None else "?"
                sv1 = f"{m['sv1']:.3f} mV" if m.get('sv1') is not None else "?"
                rs  = f"{m['rs']:.3f} mV"  if m.get('rs')  is not None else "?"
                line1 = (
                    f"Heart Rate: {hr}    "
                    f"PR Int: {pr}    "
                    f"QRS Int: {qrs}    "
                    f"QT/QTc: {qt}/{qtc}"
                )
                line2 = (
                    f"P/QRS/T Axis: {pa}/{ra}/{ta} deg    "
                    f"RV5/SV1: {rv5}/{sv1}    "
                    f"RV5+SV1: {rs}"
                )
                self.metrics_label.setText(f"{line1}\n{line2}")

            elif source == "trismed_xml":
                m = load_trismed_measurements(fields["xml_path"])
                hr  = f"{m['hr']} BPM" if m.get('hr')  is not None else "?"
                pr  = f"{m['pr']} ms"  if m.get('pr')  is not None else "?"
                qrs = f"{m['qrs']} ms" if m.get('qrs') is not None else "?"
                qt   = f"{m['qt']} ms"   if m.get('qt')    is not None else "?"
                qtc  = f"{m['qtc']} ms"  if m.get('qtc')   is not None else "?"
                qtr  = f"{m['qtr']}"     if m.get('qtr')   is not None else "?"
                pa   = f"{m['p_axis']}"  if m.get('p_axis') is not None else "--"
                ra   = f"{m['r_axis']}"  if m.get('r_axis') is not None else "--"
                ta   = f"{m['t_axis']}"  if m.get('t_axis') is not None else "--"
                qt_t = f"{m['qrs_t']}"   if m.get('qrs_t')  is not None else "--"
                line1 = (
                    f"Vent. rate: {hr}    "
                    f"PR interval: {pr}    "
                    f"QRS duration: {qrs}    "
                    f"QT: {qt}    QTc: {qtc}    QTr: {qtr}"
                )
                line2 = f"P/QRS/T/QRS-T axis: {pa}/{ra}/{ta}/{qt_t} deg"
                self.metrics_label.setText(f"{line1}\n{line2}")

            elif fields.get("measurements"):
                m = fields["measurements"]
                hr  = f"{m['hr']} BPM"    if m.get("hr")     is not None else "?"
                pr  = f"{m['pr']} ms"     if m.get("pr")     is not None else "?"
                qrs = f"{m['qrs']} ms"    if m.get("qrs")    is not None else "?"
                qt  = f"{m['qt']} ms"     if m.get("qt")     is not None else "?"
                qtc = f"{m['qtc']} ms"    if m.get("qtc")    is not None else "?"
                pa  = f"{m['p_axis']} deg" if m.get("p_axis") is not None else "?"
                ra  = f"{m['r_axis']} deg" if m.get("r_axis") is not None else "?"
                ta  = f"{m['t_axis']} deg" if m.get("t_axis") is not None else "?"
                line1 = (
                    f"HR: {hr}    "
                    f"PR: {pr}    "
                    f"QRS: {qrs}    "
                    f"QT/QTc: {qt}/{qtc}"
                )
                line2 = f"P-R-T axes: {pa}  {ra}  {ta}"
                self.metrics_label.setText(f"{line1}\n{line2}")

            else:
                self.metrics_label.setText("")
        except Exception:
            self.metrics_label.setText("")

    def load_ecg(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ECG File",
            "",
            "ECG Files (*.dat *.DAT *.hea *.HEA *.mat *.MAT *.mwf *.MWF *.csv *.CSV *.xml *.XML *.ecg *.ECG);;All Files (*)",
            options=options,
        )

        if file_path:
            base_path = normalize_input_to_base_path(file_path)
            self.load_and_plot(base_path)

    def load_and_plot(self, base_path):
        self.spinner_movie.start()
        self.stack_layout.setCurrentWidget(self.spinner_widget)
        QTimer.singleShot(50, lambda: self._load_and_plot(base_path))

    def _load_and_plot(self, base_path):
        self.current_path = base_path
        filename = os.path.basename(base_path)
        self.setWindowTitle(f"ECG Viewer - {filename}")

        try:
            signal, fields = load_waveform(base_path)
            fs = float(fields["fs"])

            signal = remove_baseline_wander(signal, fs, cutoff=0.5, order=2)
            signal = safe_bandpass_filter(signal, 0.5, 150.0, fs)

            self.current_signal = signal
            self.current_fields = fields

            self.plot_ecg(signal, fields)
            self._update_metrics(fields)
            self.convert_format_combo.setEnabled(True)
            self.convert_button.setEnabled(True)

        except Exception as exc:
            self.show_error(str(exc))
        finally:
            self.spinner_movie.stop()
            self.stack_layout.setCurrentWidget(self.canvas)

    def _collect_measurements(self) -> dict | None:
        """Read clinical measurements from the currently loaded source file.

        Returns a normalized dict with keys hr, pr, qrs, qt, qtc, p_axis,
        r_axis, t_axis (int or None; intervals in ms, axes in degrees),
        or None when the source format carries no measurement data.
        """
        fields = self.current_fields
        if not fields:
            return None
        source = fields.get("source")
        try:
            if source == "fukuda_ecg":
                m = load_fukuda_measurements(fields["ecg_path"])
                return {
                    "hr":     m.get("hr"),
                    "pr":     int(round(m["pr"]  * 1000)) if m.get("pr")  is not None else None,
                    "qrs":    int(round(m["qrs"] * 1000)) if m.get("qrs") is not None else None,
                    "qt":     int(round(m["qt"]  * 1000)) if m.get("qt")  is not None else None,
                    "qtc":    int(round(m["qtc"] * 1000)) if m.get("qtc") is not None else None,
                    "p_axis": None,
                    "r_axis": m.get("axis"),
                    "t_axis": None,
                }
            if source in ("mac2000_xml", "muse_xml", "bionet_xml", "philips_xml", "trismed_xml"):
                if source == "mac2000_xml":
                    m = load_mac2000_measurements(fields["xml_path"])
                elif source == "muse_xml":
                    m = load_muse_measurements(fields["xml_path"])
                elif source == "bionet_xml":
                    m = load_bionet_measurements(fields["xml_path"])
                elif source == "philips_xml":
                    m = load_philips_measurements(fields["xml_path"])
                else:
                    m = load_trismed_measurements(fields["xml_path"])
                return {
                    "hr":     m.get("hr"),
                    "pr":     m.get("pr"),
                    "qrs":    m.get("qrs"),
                    "qt":     m.get("qt"),
                    "qtc":    m.get("qtc"),
                    "p_axis": m.get("p_axis"),
                    "r_axis": m.get("r_axis"),
                    "t_axis": m.get("t_axis"),
                }
            if source == "mwf":
                companion = fields.get("companion_xml_path")
                if not companion or detect_xml_type(companion) != "nihonkohden":
                    return None
                m = load_nihonkohden_measurements(companion)
                return {
                    "hr":     m.get("hr"),
                    "pr":     m.get("pr"),
                    "qrs":    m.get("qrs"),
                    "qt":     m.get("qt"),
                    "qtc":    m.get("qtc"),
                    "p_axis": m.get("p_axis"),
                    "r_axis": m.get("r_axis"),
                    "t_axis": m.get("t_axis"),
                }
        except Exception:
            pass
        return None

    def _on_convert(self) -> None:
        """Open a save dialog and export the current ECG to the selected format."""
        if self.current_signal is None or self.current_fields is None:
            return

        fmt = self.convert_format_combo.currentText()

        if "CSV" in fmt:
            file_filter = "CSV Files (*.csv)"
            default_ext = ".csv"
        elif "WFDB" in fmt:
            file_filter = "WFDB Header (*.hea)"
            default_ext = ".hea"
        elif "MAT" in fmt:
            file_filter = "MATLAB File (*.mat)"
            default_ext = ".mat"
        elif "MUSE" in fmt:
            file_filter = "MUSE XML (*.xml)"
            default_ext = ".xml"
        else:
            return

        suggested = ""
        if self.current_path:
            suggested = os.path.splitext(self.current_path)[0] + default_ext

        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Converted ECG", suggested, file_filter
        )
        if not out_path:
            return

        measurements = self._collect_measurements()

        try:
            if "CSV" in fmt:
                export_csv(self.current_signal, self.current_fields, out_path, measurements)
            elif "WFDB" in fmt:
                export_wfdb(self.current_signal, self.current_fields, out_path, measurements)
            elif "MAT" in fmt:
                export_mat_hea(self.current_signal, self.current_fields, out_path, measurements)
            elif "MUSE" in fmt:
                export_muse_xml(self.current_signal, self.current_fields, out_path, measurements)
            QMessageBox.information(self, "Convert", f"Saved to:\n{out_path}")
        except Exception as exc:
            self.show_error(f"Convert failed: {exc}")

    def on_plot_click(self, event):
        if event.inaxes is None:
            return

        if not event.dblclick:
            return

        lead = self.ax_lead_map.get(event.inaxes)
        if not lead:
            return

        if self.current_signal is None or self.current_fields is None:
            return

        lead_indices = {
            str(name).upper(): idx
            for idx, name in enumerate(self.current_fields["sig_name"])
        }

        lead_index = lead_indices.get(lead.upper())
        if lead_index is None:
            return

        dialog = LeadZoomDialog(
            self.current_signal,
            float(self.current_fields["fs"]),
            lead,
            lead_index,
            self,
        )
        self.zoom_dialogs.append(dialog)
        dialog.show()

    def plot_ecg(self, signal, fields):
        self.figure.clear()
        self.ax_lead_map = {}

        fs = float(fields["fs"])
        total_samples = signal.shape[0]

        short_duration = min(int(fs * 2.5), total_samples)
        long_duration = min(int(fs * 10.0), total_samples)

        time_short = np.arange(short_duration) / fs if short_duration > 0 else np.array([])
        time_long = np.arange(long_duration) / fs if long_duration > 0 else np.array([])

        layout_order = [
            "I", "aVR", "V1", "V4",
            "II", "aVL", "V2", "V5",
            "III", "aVF", "V3", "V6",
        ]

        lead_indices = {str(name).upper(): idx for idx, name in enumerate(fields["sig_name"])}

        grid = GridSpec(
            4,
            4,
            figure=self.figure,
            height_ratios=[1.0, 1.0, 1.0, 1.0],
            hspace=0.30,
            wspace=0.08,
        )

        for i, lead in enumerate(layout_order):
            if lead.upper() not in lead_indices:
                continue

            row = i // 4
            col = i % 4

            idx = lead_indices[lead.upper()]
            ax = self.figure.add_subplot(grid[row, col])
            self.ax_lead_map[ax] = lead
            ax.set_facecolor(PAPER_FACE_COLOR)

            ecg = signal[:short_duration, idx]
            ax.plot(time_short[:len(ecg)], ecg, linewidth=0.8)

            ax.set_title(lead, fontsize=16, pad=2)
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_ylabel("Amplitude (mV)", fontsize=7)
            ax.tick_params(axis="both", which="major", labelsize=6)

            ax.set_ylim([-MAIN_Y_LIM, MAIN_Y_LIM])
            ax.set_xlim([0, 2.5])
            ax.set_box_aspect(0.96)

            ax.set_xticks(np.arange(0, 2.5 + 1e-9, 0.2))
            ax.set_xticks(np.arange(0, 2.5 + 1e-9, SEC_PER_MM), minor=True)
            ax.set_yticks(np.arange(-MAIN_Y_LIM, MAIN_Y_LIM + 1e-9, Y_MAJOR_STEP))
            ax.set_yticks(np.arange(-MAIN_Y_LIM, MAIN_Y_LIM + 1e-9, MV_PER_MM), minor=True)

            ax.grid(
                True,
                which="major",
                linestyle=GRID_MAJOR_LINESTYLE,
                linewidth=GRID_MAJOR_WIDTH,
                color=GRID_COLOR,
            )

            ax.grid(
                True,
                which="minor",
                linestyle=GRID_MINOR_LINESTYLE,
                linewidth=GRID_MINOR_WIDTH,
                color=GRID_COLOR,
                alpha=GRID_MINOR_ALPHA,
            )

        if "II" in lead_indices and long_duration > 0:
            ax = self.figure.add_subplot(grid[3, :])
            self.ax_lead_map[ax] = "II"
            ax.set_facecolor(PAPER_FACE_COLOR)

            idx = lead_indices["II"]
            ecg = signal[:long_duration, idx]
            ax.plot(time_long[:len(ecg)], ecg, linewidth=0.8)

            display_long_sec = 10.0

            ax.set_title("II (repeat)", fontsize=16)
            ax.set_xlabel("Time (s)", fontsize=7)
            ax.set_ylabel("Amplitude (mV)", fontsize=7)
            ax.tick_params(axis="both", which="major", labelsize=6)

            ax.set_ylim([-MAIN_Y_LIM, MAIN_Y_LIM])
            ax.set_xlim([0, display_long_sec])
            ax.set_box_aspect(0.24)

            ax.set_xticks(np.arange(0, display_long_sec + 1e-9, 0.2))
            ax.set_xticks(np.arange(0, display_long_sec + 1e-9, SEC_PER_MM), minor=True)
            ax.set_yticks(np.arange(-MAIN_Y_LIM, MAIN_Y_LIM + 1e-9, Y_MAJOR_STEP))
            ax.set_yticks(np.arange(-MAIN_Y_LIM, MAIN_Y_LIM + 1e-9, MV_PER_MM), minor=True)

            ax.grid(
                True,
                which="major",
                linestyle=GRID_MAJOR_LINESTYLE,
                linewidth=GRID_MAJOR_WIDTH,
                color=GRID_COLOR,
            )

            ax.grid(
                True,
                which="minor",
                linestyle=GRID_MINOR_LINESTYLE,
                linewidth=GRID_MINOR_WIDTH,
                color=GRID_COLOR,
                alpha=GRID_MINOR_ALPHA,
            )

        info_text = (
            f"source: {fields.get('source', '?')}\n"
            f"layout: {fields.get('layout', '?')}\n"
            f"fs: {fields.get('fs', '?')}\n"
            f"channels: {fields.get('channel_count', '?')}\n"
            f"samples: {fields.get('sample_count', '?')}\n"
            f"scale: {fields.get('scale_mode', '-')}"
        )
        self.figure.text(
            0.99,
            0.99,
            info_text,
            ha="right",
            va="top",
            fontsize=8,
            alpha=0.7,
        )

        self.figure.subplots_adjust(
            left=0.025,
            right=0.992,
            top=0.97,
            bottom=0.045,
            wspace=0.08,
            hspace=0.30,
        )
        self.canvas.draw_idle()

    def closeEvent(self, event):
        try:
            self.spinner_movie.stop()
        except Exception:
            pass
        super().closeEvent(event)
