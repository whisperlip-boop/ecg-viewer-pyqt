import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from ..constants import (
    GRID_COLOR,
    GRID_MAJOR_LINESTYLE,
    GRID_MAJOR_WIDTH,
    GRID_MINOR_ALPHA,
    GRID_MINOR_LINESTYLE,
    GRID_MINOR_WIDTH,
    PAPER_FACE_COLOR,
    SEC_PER_MM,
    Y_MAJOR_STEP,
    Y_MINOR_STEP,
    ZOOM_DIALOG_HEIGHT,
    ZOOM_DIALOG_WIDTH,
    ZOOM_Y_LIM,
)


class LeadZoomDialog(QDialog):
    def __init__(self, signal, fs, lead_name, lead_index, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Lead Zoom - {lead_name}")
        self.resize(ZOOM_DIALOG_WIDTH, ZOOM_DIALOG_HEIGHT)

        self.signal = signal
        self.fs = fs
        self.lead_name = lead_name
        self.lead_index = lead_index

        self.current_duration_sec = 10.0
        self.y = None
        self.ax = None
        self.total_duration_sec = 0.0

        self.crosshair_v = None
        self.crosshair_h = None
        self.hover_annot = None
        self.measure_annot = None

        self.measure_point1 = None
        self.measure_point2 = None
        self.measure_marker1 = None
        self.measure_marker2 = None
        self.measure_line = None

        self.initial_xlim = None
        self.initial_ylim = None

        self.is_panning = False
        self.pan_start_mouse = None
        self.pan_start_xlim = None
        self.pan_start_ylim = None

        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Display Duration (s):"))

        self.duration_combo = QComboBox()
        self.duration_combo.addItems(["2.5", "5", "10"])
        self.duration_combo.setCurrentText("10")
        self.duration_combo.currentTextChanged.connect(self.on_duration_changed)
        top_bar.addWidget(self.duration_combo)

        self.save_button = QPushButton("Save PNG")
        self.save_button.clicked.connect(self.save_png)
        top_bar.addWidget(self.save_button)

        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText("hh:mm:ss")
        self.time_input.setFixedWidth(120)
        self.time_input.returnPressed.connect(self.move_to_input_time)
        top_bar.addWidget(self.time_input)

        self.move_button = QPushButton("Move")
        self.move_button.clicked.connect(self.move_to_input_time)
        top_bar.addWidget(self.move_button)

        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.figure = Figure(figsize=(12, 4.8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("axes_leave_event", self.on_axes_leave)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_press)
        self.canvas.mpl_connect("button_release_event", self.on_canvas_release)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)

        self.redraw()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.full_reset()
            event.accept()
            return
        super().keyPressEvent(event)

    def on_duration_changed(self):
        self.full_reset()
        self.redraw()

    def redraw(self):
        self.current_duration_sec = float(self.duration_combo.currentText())
        self.y = self.signal[:, self.lead_index]
        self.total_duration_sec = (len(self.y) / self.fs) if len(self.y) > 0 else 0.0

        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(PAPER_FACE_COLOR)
        if len(self.y) > 0:
            self.ax.plot(np.arange(len(self.y)) / self.fs, self.y, linewidth=1.0)

        self.ax.set_title(self.lead_name, fontsize=16)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude (mV)")

        initial_x1 = min(self.current_duration_sec, self.total_duration_sec)
        if initial_x1 <= 0:
            initial_x1 = self.current_duration_sec
        self.ax.set_xlim(0, initial_x1)
        self.ax.set_ylim(-ZOOM_Y_LIM, ZOOM_Y_LIM)

        self.initial_xlim = self.ax.get_xlim()
        self.initial_ylim = self.ax.get_ylim()

        self.ax.xaxis.set_major_locator(MultipleLocator(0.2))
        self.ax.xaxis.set_minor_locator(MultipleLocator(SEC_PER_MM))
        self.ax.yaxis.set_major_locator(MultipleLocator(Y_MAJOR_STEP))
        self.ax.yaxis.set_minor_locator(MultipleLocator(Y_MINOR_STEP))

        self.ax.grid(
            True,
            which="major",
            linestyle=GRID_MAJOR_LINESTYLE,
            linewidth=GRID_MAJOR_WIDTH,
            color=GRID_COLOR,
        )

        self.ax.grid(
            True,
            which="minor",
            linestyle=GRID_MINOR_LINESTYLE,
            linewidth=GRID_MINOR_WIDTH,
            color=GRID_COLOR,
            alpha=GRID_MINOR_ALPHA,
        )

        self.crosshair_v = self.ax.axvline(0, linewidth=0.8, alpha=0.5, visible=False)
        self.crosshair_h = self.ax.axhline(0, linewidth=0.8, alpha=0.5, visible=False)

        self.hover_annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.90),
            fontsize=9,
        )
        self.hover_annot.set_visible(False)

        self.measure_annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, -40),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="#fff4cc", alpha=0.95),
            fontsize=9,
        )
        self.measure_annot.set_visible(False)

        self.measure_marker1 = None
        self.measure_marker2 = None
        self.measure_line = None

        self.is_panning = False
        self.pan_start_mouse = None
        self.pan_start_xlim = None
        self.pan_start_ylim = None

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def save_png(self):
        default_name = f"{self.lead_name}.png"
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save PNG",
            default_name,
            "PNG Files (*.png);;All Files (*)",
        )
        if file_path:
            self.figure.savefig(file_path, dpi=200, bbox_inches="tight")

    def parse_relative_time_input(self):
        text = self.time_input.text().strip()
        if not text:
            return 0

        parts = text.split(":")
        if len(parts) > 3:
            raise ValueError("Invalid time format")

        values = []
        for part in parts:
            part = part.strip()
            if not part:
                values.append(0)
                continue
            if not part.isdigit():
                raise ValueError("Invalid time format")
            values.append(int(part))

        while len(values) < 3:
            values.insert(0, 0)

        hours, minutes, seconds = values
        return hours * 3600 + minutes * 60 + seconds

    def move_to_input_time(self):
        try:
            target_sec = float(self.parse_relative_time_input())
        except ValueError:
            QMessageBox.warning(self, "Invalid time", "Invalid time format")
            return

        self.move_to_time(target_sec)

    def move_to_time(self, target_sec):
        if self.ax is None or len(self.y) == 0:
            QMessageBox.information(self, "No data exists", "No data exists")
            return

        if target_sec > self.total_duration_sec:
            QMessageBox.information(self, "No data exists", "No data exists")
            return

        if target_sec <= 0:
            if self.initial_xlim is not None:
                self.ax.set_xlim(self.initial_xlim)
                self.canvas.draw_idle()
            return

        cur_xlim = self.ax.get_xlim()
        width = cur_xlim[1] - cur_xlim[0]
        half_width = width / 2.0

        new_x0 = target_sec - half_width
        new_x1 = target_sec + half_width
        new_x0, new_x1 = self.clamp_xlim(new_x0, new_x1)

        self.ax.set_xlim(new_x0, new_x1)
        self.canvas.draw_idle()

    def nearest_sample(self, xdata):
        idx = int(np.clip(round(xdata * self.fs), 0, len(self.y) - 1))
        t = idx / self.fs
        amp = float(self.y[idx])
        return idx, t, amp

    def on_mouse_move(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.hide_hover_overlay()
            return

        if self.is_panning and self.pan_start_mouse is not None:
            dx = event.xdata - self.pan_start_mouse[0]
            dy = event.ydata - self.pan_start_mouse[1]

            x0, x1 = self.pan_start_xlim
            y0, y1 = self.pan_start_ylim

            new_x0 = x0 - dx
            new_x1 = x1 - dx
            new_x0, new_x1 = self.clamp_xlim(new_x0, new_x1)

            self.ax.set_xlim(new_x0, new_x1)
            self.ax.set_ylim(y0 - dy, y1 - dy)
            self.canvas.draw_idle()
            return

        idx, t, amp = self.nearest_sample(event.xdata)

        self.crosshair_v.set_visible(True)
        self.crosshair_h.set_visible(True)
        self.crosshair_v.set_xdata([t, t])
        self.crosshair_h.set_ydata([amp, amp])

        self.hover_annot.xy = (t, amp)
        self.hover_annot.set_text(
            f"idx: {idx}\n"
            f"t: {t:.3f} s\n"
            f"amp: {amp:.3f} mV"
        )
        self.hover_annot.set_visible(True)

        self.canvas.draw_idle()

    def on_axes_leave(self, event):
        if not self.is_panning:
            self.hide_hover_overlay()

    def hide_hover_overlay(self):
        if self.crosshair_v is not None:
            self.crosshair_v.set_visible(False)
        if self.crosshair_h is not None:
            self.crosshair_h.set_visible(False)
        if self.hover_annot is not None:
            self.hover_annot.set_visible(False)
        self.canvas.draw_idle()

    def on_canvas_press(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        if event.button == 3:
            self.is_panning = True
            self.pan_start_mouse = (event.xdata, event.ydata)
            self.pan_start_xlim = self.ax.get_xlim()
            self.pan_start_ylim = self.ax.get_ylim()
            return

        if event.dblclick:
            return

        if event.button != 1:
            return

        idx, t, amp = self.nearest_sample(event.xdata)

        if self.measure_point1 is None or self.measure_point2 is not None:
            self.set_measure_point1(idx, t, amp)
        else:
            self.set_measure_point2(idx, t, amp)

    def on_canvas_release(self, event):
        if event.button == 3:
            self.is_panning = False
            self.pan_start_mouse = None
            self.pan_start_xlim = None
            self.pan_start_ylim = None

    def on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        x_center = event.xdata
        x0, x1 = cur_xlim
        x_range = x1 - x0

        if x_range <= 0:
            return

        scale = 0.8 if event.button == "up" else 1.25

        new_range = x_range * scale
        total_min = 0.0
        total_max = self.total_duration_sec

        left_ratio = (x_center - x0) / x_range
        right_ratio = 1.0 - left_ratio

        new_x0 = x_center - new_range * left_ratio
        new_x1 = x_center + new_range * right_ratio

        min_range = max(5.0 / self.fs, SEC_PER_MM)
        max_range = max(total_max - total_min, self.current_duration_sec)

        if new_range < min_range:
            new_range = min_range
            new_x0 = x_center - new_range * left_ratio
            new_x1 = x_center + new_range * right_ratio

        if new_range > max_range:
            new_range = max_range
            new_x0 = total_min
            new_x1 = total_max

        if new_x0 < total_min:
            shift = total_min - new_x0
            new_x0 += shift
            new_x1 += shift
        if new_x1 > total_max:
            shift = new_x1 - total_max
            new_x0 -= shift
            new_x1 -= shift

        new_x0, new_x1 = self.clamp_xlim(new_x0, new_x1)

        self.ax.set_xlim(new_x0, new_x1)
        self.ax.set_ylim(cur_ylim)
        self.canvas.draw_idle()

    def clamp_xlim(self, x0, x1):
        if len(self.y) == 0:
            return x0, x1

        width = x1 - x0
        total_min = 0.0
        total_max = self.total_duration_sec

        if total_max <= total_min:
            return 0.0, max(self.current_duration_sec, SEC_PER_MM)

        if width >= (total_max - total_min):
            return total_min, total_max

        if x0 < total_min:
            x1 += total_min - x0
            x0 = total_min
        if x1 > total_max:
            x0 -= x1 - total_max
            x1 = total_max

        return x0, x1

    def set_measure_point1(self, idx, t, amp):
        self.clear_measurement_artists_only()

        self.measure_point1 = {"idx": idx, "t": t, "amp": amp}
        self.measure_point2 = None

        (self.measure_marker1,) = self.ax.plot(
            [t],
            [amp],
            marker="o",
            markersize=6,
            linestyle="None",
        )

        self.measure_annot.set_visible(False)
        self.canvas.draw_idle()

    def set_measure_point2(self, idx, t, amp):
        self.measure_point2 = {"idx": idx, "t": t, "amp": amp}

        t1 = self.measure_point1["t"]
        a1 = self.measure_point1["amp"]
        t2 = self.measure_point2["t"]
        a2 = self.measure_point2["amp"]

        if self.measure_marker2 is not None:
            self.measure_marker2.remove()
            self.measure_marker2 = None
        if self.measure_line is not None:
            self.measure_line.remove()
            self.measure_line = None

        (self.measure_marker2,) = self.ax.plot(
            [t2],
            [a2],
            marker="o",
            markersize=6,
            linestyle="None",
        )

        (self.measure_line,) = self.ax.plot(
            [t1, t2],
            [a1, a2],
            linewidth=1.0,
            linestyle="--",
        )

        dt_ms = (t2 - t1) * 1000.0
        da_mv = a2 - a1

        if dt_ms > 0:
            bpm = round(60000.0 / dt_ms)
            bpm_text = f" ({bpm} bpm)"
        else:
            bpm_text = ""

        self.measure_annot.xy = (t2, a2)
        self.measure_annot.set_text(
            f"Δt: {dt_ms:.1f} ms{bpm_text}\n"
            f"Δamp: {da_mv:.3f} mV"
        )
        self.measure_annot.set_visible(True)

        self.canvas.draw_idle()

    def clear_measurement_artists_only(self):
        if self.measure_marker1 is not None:
            self.measure_marker1.remove()
            self.measure_marker1 = None
        if self.measure_marker2 is not None:
            self.measure_marker2.remove()
            self.measure_marker2 = None
        if self.measure_line is not None:
            self.measure_line.remove()
            self.measure_line = None

    def reset_measurement(self):
        self.measure_point1 = None
        self.measure_point2 = None
        self.clear_measurement_artists_only()
        if self.measure_annot is not None:
            self.measure_annot.set_visible(False)
        self.canvas.draw_idle()

    def reset_view(self):
        if self.ax is None:
            return
        if self.initial_xlim is not None:
            self.ax.set_xlim(self.initial_xlim)
        if self.initial_ylim is not None:
            self.ax.set_ylim(self.initial_ylim)

        self.is_panning = False
        self.pan_start_mouse = None
        self.pan_start_xlim = None
        self.pan_start_ylim = None

    def full_reset(self):
        self.reset_measurement()
        self.reset_view()
        self.hide_hover_overlay()
        self.canvas.draw_idle()



