"""Learning / measurement panels: "Unde" (simple waves add up) and "Acustică" (room response)."""

from __future__ import annotations

import csv
import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dsp import THIRD_OCTAVE_CENTRES

from .sky_panel import ObjectInfo
from .views import BG, GRID, TEXT, _band_label

WAVE_COLORS = ("#7fb3ff", "#f2b24a", "#b79cf2", "#4dd0c4")       # a, b, c and their sum d


def noise_buffer(color: str, sample_rate: int = 44100, seconds: float = 8.0, seed: int = 7) -> np.ndarray:
    """A loop of white or pink noise at about -20 dBFS RMS. Built in the frequency domain, so it repeats seamlessly."""
    n = int(sample_rate * seconds)
    spectrum = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    if color == "pink":
        spectrum[1:] /= np.sqrt(np.arange(1, spectrum.size))      # power falls 3 dB per octave
    spectrum[: max(1, int(20.0 * seconds))] = 0.0                  # nothing below 20 Hz
    noise = np.fft.irfft(spectrum, n)
    return (0.1 * noise / np.sqrt(np.mean(noise ** 2))).clip(-1.0, 1.0).astype(np.float32)


def tone_buffer(tones: list[tuple[float, float]], sample_rate: int = 44100) -> np.ndarray:
    """One second of the tones with frequencies rounded to whole hertz, so the loop has no click."""
    t = np.arange(sample_rate) / sample_rate
    out = np.zeros(sample_rate)
    for freq, amp in tones:
        out += amp * np.sin(2.0 * np.pi * round(freq) * t)
    peak = float(np.max(np.abs(out)))
    return (out / max(1.0, peak / 0.9)).astype(np.float32)


class WavesView(QWidget):
    """Three simple waves (a, b, c) drawn above their sum (d)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(300)
        self._tones: list[tuple[float, float]] = []

    def set_tones(self, tones: list[tuple[float, float]]) -> None:
        self._tones = list(tones)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(BG))
        w, h = self.width(), self.height()
        active = [f for f, a in self._tones if a > 0 and f > 0]
        span = 3.0 / min(active) if active else 0.01                # three periods of the slowest wave
        rows = len(self._tones) + 1
        row_h = h / (rows + 0.6)                                    # the sum gets a taller row
        x = np.linspace(0.0, span, max(200, w))
        total = np.zeros(x.size)
        font = QFont(self.font())
        font.setPixelSize(11)
        painter.setFont(font)
        scale_all = max(1e-6, sum(a for _, a in self._tones))

        def trace(values: np.ndarray, top: float, height: float, color: str, label: str, full: float) -> None:
            mid = top + height / 2.0
            painter.setPen(QPen(QColor(GRID), 1.0))
            painter.drawLine(QPointF(0, mid), QPointF(w, mid))
            path = QPainterPath()
            ys = mid - values / full * (height / 2.0 - 6.0)
            path.moveTo(0.0, float(ys[0]))
            for i in range(1, x.size):
                path.lineTo(i * w / (x.size - 1), float(ys[i]))
            painter.setPen(QPen(QColor(color), 1.8))
            painter.drawPath(path)
            painter.drawText(QRectF(8, top + 2, w - 16, 16), Qt.AlignmentFlag.AlignLeft, label)

        for index, (freq, amp) in enumerate(self._tones):
            wave = amp * np.sin(2.0 * np.pi * freq * x)
            total += wave
            trace(wave, index * row_h, row_h, WAVE_COLORS[index % 3], f"unda {'abc'[index % 3]}: {freq:g} Hz", max(scale_all / 1.5, 1e-6))
        trace(total, len(self._tones) * row_h, row_h * 1.6, WAVE_COLORS[3], "unda d = a + b + c (ce ajunge la ureche)", scale_all)
        painter.setPen(QColor(TEXT))
        painter.drawText(QRectF(8, h - 16, w - 16, 14), Qt.AlignmentFlag.AlignRight, f"{span * 1000:.1f} ms")
        painter.end()


class WavesPanel(QWidget):
    """Sliders for three simple waves; shows that a complex wave is only their sum."""

    tones_changed = pyqtSignal(list)                   # [(frequency, amplitude), ...]

    def __init__(self, tones: list[tuple[float, float]], nyquist: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        frame = QFrame()
        frame.setObjectName("pane")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(12, 9, 12, 10)
        self.view = WavesView()
        frame_layout.addWidget(self.view, 1)
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        for column, text in ((1, "Frecvență"), (2, "Tărie")):
            head = QLabel(text)
            head.setObjectName("paneInfo")
            grid.addWidget(head, 0, column)
        self._freq: list[QDoubleSpinBox] = []
        self._amp: list[QDoubleSpinBox] = []
        for index, (freq, amp) in enumerate(tones):
            label = QLabel(f"Unda {'abc'[index % 3]}")
            label.setStyleSheet(f"color: {WAVE_COLORS[index % 3]}; font-weight: 600;")
            f_box = QDoubleSpinBox()
            f_box.setRange(20.0, nyquist - 1.0)
            f_box.setDecimals(0)
            f_box.setSingleStep(10.0)
            f_box.setSuffix(" Hz")
            f_box.setValue(freq)
            a_box = QDoubleSpinBox()
            a_box.setRange(0.0, 1.0)
            a_box.setDecimals(2)
            a_box.setSingleStep(0.05)
            a_box.setValue(amp)
            f_box.valueChanged.connect(self._emit)
            a_box.valueChanged.connect(self._emit)
            grid.addWidget(label, index + 1, 0)
            grid.addWidget(f_box, index + 1, 1)
            grid.addWidget(a_box, index + 1, 2)
            self._freq.append(f_box)
            self._amp.append(a_box)
        frame_layout.addLayout(grid)
        layout.addWidget(frame, 5)

        lower = QFrame()
        lower.setObjectName("pane")
        lower_layout = QVBoxLayout(lower)
        lower_layout.setContentsMargins(12, 10, 12, 10)
        info = ObjectInfo(
            "Unde simple, undă complexă",
            "Transformata Fourier (J. B. J. Fourier, 1822) desface orice sunet în unde simple. "
            "Varianta rapidă, FFT (Cooley și Tukey, 1965), o face în timp real: asta vezi în spectru.",
        )
        info.lbl_story.setText(
            "Trei unde simple, a, b și c, se adună într-una singură, d. Urechea primește doar unda d, "
            "dar analizorul o desface la loc: trei vârfuri, câte unul pentru fiecare undă. "
            "Schimbă frecvențele și privește cum se mută vârfurile. Pune tăria pe 0 ca să scoți o undă."
        )
        lower_layout.addWidget(info)
        layout.addWidget(lower, 3)
        self.view.set_tones(self.tones())

    def tones(self) -> list[tuple[float, float]]:
        return [(f.value(), a.value()) for f, a in zip(self._freq, self._amp)]

    def _emit(self) -> None:
        self.view.set_tones(self.tones())
        self.tones_changed.emit(self.tones())

    def set_tones(self, tones: list[tuple[float, float]]) -> None:
        """Mirror values changed elsewhere (the Controls panel) without emitting."""
        for (freq, amp), f_box, a_box in zip(tones, self._freq, self._amp):
            for box, value in ((f_box, freq), (a_box, amp)):
                box.blockSignals(True)
                box.setValue(value)
                box.blockSignals(False)
        self.view.set_tones(self.tones())


class ResponseView(QWidget):
    """Deviation of each third-octave band from the average level, with the suggested correction."""

    RANGE_DB = 18.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(240)
        self.deviation = np.full(len(THIRD_OCTAVE_CENTRES), np.nan)

    def set_deviation(self, deviation: np.ndarray) -> None:
        self.deviation = deviation
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(BG))
        w, h = self.width(), self.height()
        left, bottom, top = 40.0, 20.0, 8.0
        plot_h = h - bottom - top
        mid = top + plot_h / 2.0
        font = QFont(self.font())
        font.setPixelSize(10)
        painter.setFont(font)
        for db in (-12, -6, 0, 6, 12):
            y = mid - db / self.RANGE_DB * plot_h / 2.0
            painter.setPen(QPen(QColor(GRID if db else TEXT), 1.0))
            painter.drawLine(QPointF(left, y), QPointF(w - 4, y))
            painter.setPen(QColor(TEXT))
            painter.drawText(QRectF(0, y - 7, left - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{db:+d} dB")
        bands = len(THIRD_OCTAVE_CENTRES)
        col_w = (w - left - 4.0) / bands
        painter.setPen(Qt.PenStyle.NoPen)
        for i, value in enumerate(self.deviation):
            if math.isnan(value):
                continue
            clipped = max(-self.RANGE_DB, min(self.RANGE_DB, float(value)))
            height = clipped / self.RANGE_DB * plot_h / 2.0
            painter.setBrush(QColor("#f2b24a" if value > 0 else "#4dd0c4"))
            painter.drawRect(QRectF(left + i * col_w + 2.0, min(mid, mid - height), col_w - 4.0, abs(height)))
        painter.setPen(QColor(TEXT))
        for i, centre in enumerate(THIRD_OCTAVE_CENTRES):
            if i % 3 == 0:
                painter.drawText(QRectF(left + i * col_w - 12, h - bottom + 3, col_w + 24, 14), Qt.AlignmentFlag.AlignCenter, _band_label(centre))
        painter.end()


class AcousticsPanel(QWidget):
    """Room response: test noise from the loudspeakers, measured by the microphone in third-octave bands."""

    start_requested = pyqtSignal(str)                  # noise colour: "pink" or "white"
    stop_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.running = False
        self._power = np.zeros(len(THIRD_OCTAVE_CENTRES))
        self._frames = 0
        self._color = "pink"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        frame = QFrame()
        frame.setObjectName("pane")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(12, 9, 12, 10)
        header = QHBoxLayout()
        title = QLabel("Răspunsul camerei")
        title.setObjectName("paneTitle")
        self.lbl_state = QLabel("oprit")
        self.lbl_state.setObjectName("paneInfo")
        self.lbl_state.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(title)
        header.addWidget(self.lbl_state, 1)
        frame_layout.addLayout(header)
        self.view = ResponseView()
        frame_layout.addWidget(self.view, 1)
        buttons = QHBoxLayout()
        self.cmb_noise = QComboBox()
        self.cmb_noise.addItem("Zgomot roz", "pink")
        self.cmb_noise.addItem("Zgomot alb", "white")
        self.btn_start = QPushButton("Pornește măsurarea")
        self.btn_start.clicked.connect(self._toggle)
        self.btn_reset = QPushButton("Reia media")
        self.btn_reset.clicked.connect(self.reset)
        self.btn_export = QPushButton("Salvează CSV")
        self.btn_export.clicked.connect(self._export)
        for widget in (self.cmb_noise, self.btn_start, self.btn_reset, self.btn_export):
            buttons.addWidget(widget)
        frame_layout.addLayout(buttons)
        layout.addWidget(frame, 5)

        lower = QFrame()
        lower.setObjectName("pane")
        lower_layout = QVBoxLayout(lower)
        lower_layout.setContentsMargins(12, 10, 12, 10)
        self.info = ObjectInfo(
            "Corecție acustică",
            "Boxele redau zgomot cu energie egală în fiecare bandă; microfonul ascultă ce ajunge la el. "
            "Unde camera întărește sau înghite sunetul, banda iese mai sus sau mai jos decât media.",
        )
        self.info.lbl_story.setText(
            "Portocaliu: banda e prea tare. Turcoaz: prea slabă. Corecția de pus într-un egalizator este "
            "valoarea cu semn schimbat (se salvează în CSV). Rezultatul include și microfonul și boxele: "
            "cu microfonul unui laptop este doar orientativ; pentru măsurători serioase trebuie un microfon de măsură."
        )
        lower_layout.addWidget(self.info)
        layout.addWidget(lower, 3)

    def _toggle(self) -> None:
        if self.running:
            self.stop_requested.emit()
        else:
            self._color = self.cmb_noise.currentData()
            self.reset()
            self.start_requested.emit(self._color)

    def set_running(self, running: bool, note: str = "") -> None:
        self.running = running
        self.btn_start.setText("Oprește măsurarea" if running else "Pornește măsurarea")
        self.cmb_noise.setEnabled(not running)
        self.lbl_state.setText(note or ("măsoară..." if running else "oprit"))

    def reset(self) -> None:
        self._power[:] = 0.0
        self._frames = 0
        self.view.set_deviation(np.full(len(THIRD_OCTAVE_CENTRES), np.nan))

    def add_levels(self, levels_db: np.ndarray) -> None:
        """Add one frame of third-octave levels (dBFS) to the running average."""
        self._power += 10.0 ** (np.nan_to_num(levels_db, nan=-200.0) / 10.0)
        self._frames += 1
        if self._frames % 6 == 0:
            self.view.set_deviation(self.deviation())
            self.lbl_state.setText(f"măsoară... medie pe {self._frames} cadre")

    def deviation(self) -> np.ndarray:
        """Each band relative to the mean of the 100 Hz - 10 kHz bands, in dB (NaN where there is no data)."""
        return band_deviation(self._power / max(self._frames, 1), self._color)

    def _export(self) -> None:
        deviation = self.deviation()
        if self._frames == 0 or np.all(np.isnan(deviation)):
            self.lbl_state.setText("nu există încă o măsurătoare")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Salvează răspunsul camerei", "raspuns_camera.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.writer(fh, delimiter=";")
                writer.writerow(["Banda (Hz)", "Abatere (dB)", "Corectie egalizator (dB)"])
                for centre, value in zip(THIRD_OCTAVE_CENTRES, deviation):
                    if not math.isnan(value):
                        writer.writerow([f"{centre:g}", f"{value:+.1f}", f"{max(-12.0, min(12.0, -value)):+.1f}"])
        except OSError as exc:
            self.lbl_state.setText(f"nu s-a putut salva: {exc}")
            return
        self.lbl_state.setText("salvat")


def band_deviation(mean_power: np.ndarray, color: str = "pink") -> np.ndarray:
    """Third-octave band powers -> deviation in dB from the 100 Hz - 10 kHz average.

    Pink noise has the same power in every third-octave band, so a flat system
    gives 0 dB everywhere. White noise rises by 1 dB per band; that slope is
    removed first.
    """
    centres = np.asarray(THIRD_OCTAVE_CENTRES)
    with np.errstate(divide="ignore"):
        levels = 10.0 * np.log10(mean_power)
    if color == "white":
        levels = levels - 10.0 * np.log10(centres / 1000.0)
    valid = np.isfinite(levels) & (levels > -150.0)
    reference = valid & (centres >= 100.0) & (centres <= 10000.0)
    if not reference.any():
        return np.full(centres.size, np.nan)
    out = levels - float(np.mean(levels[reference]))
    out[~valid] = np.nan
    return out
