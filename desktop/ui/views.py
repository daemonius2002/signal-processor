"""Alternative views of the spectrum: LED third-octave analyser and spectrogram."""

from __future__ import annotations

import math
import time

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from dsp import THIRD_OCTAVE_CENTRES

BG = "#161d27"
GRID = "#2a3544"
TEXT = "#8595a8"
LED_OFF = "#1e2733"
LED_COLORS = ("#57d08a", "#f0c04a", "#ff6b6b")          # normal, high, near full scale


def _band_label(hz: float) -> str:
    return f"{hz / 1000:g}k" if hz >= 1000 else f"{hz:g}"


class _TimedWidget(QWidget):
    """Custom-painted widget that reports how long its paint events take (for the latency monitor)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.paint_ms_accum = 0.0
        self.setMinimumHeight(120)

    def take_paint_ms(self) -> float:
        value, self.paint_ms_accum = self.paint_ms_accum, 0.0
        return value

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        t0 = time.perf_counter()
        painter = QPainter(self)
        self._paint(painter)
        painter.end()
        self.paint_ms_accum += (time.perf_counter() - t0) * 1e3

    def _paint(self, painter: QPainter) -> None:
        raise NotImplementedError


class LedBandsView(_TimedWidget):
    """28 third-octave bands as columns of LEDs with peak-hold markers, like a hardware real-time analyser."""

    SEGMENT_DB = 3.0
    FLOOR_DB = -72.0
    TOP_DB = 0.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._levels = np.full(len(THIRD_OCTAVE_CENTRES), np.nan)
        self._peaks = np.full(len(THIRD_OCTAVE_CENTRES), -200.0)
        self._last = time.perf_counter()

    def set_levels(self, levels: np.ndarray) -> None:
        now = time.perf_counter()
        self._peaks -= 12.0 * (now - self._last)          # peak markers fall 12 dB per second
        self._last = now
        self._levels = levels
        self._peaks = np.fmax(self._peaks, np.nan_to_num(levels, nan=-200.0))
        self.update()

    def reset(self) -> None:
        self._peaks[:] = -200.0

    def _paint(self, painter: QPainter) -> None:
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(BG))
        left, bottom, top = 44.0, 20.0, 6.0
        plot_h = h - bottom - top
        segments = int(round((self.TOP_DB - self.FLOOR_DB) / self.SEGMENT_DB))
        seg_h = plot_h / segments
        font = QFont(self.font())
        font.setPixelSize(10)
        painter.setFont(font)
        painter.setPen(QColor(TEXT))
        for db in range(0, int(self.FLOOR_DB) - 1, -12):
            y = top + plot_h * (self.TOP_DB - db) / (self.TOP_DB - self.FLOOR_DB)
            painter.drawText(QRectF(0, y - 7, left - 6, 14), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{db} dB")
        bands = len(THIRD_OCTAVE_CENTRES)
        col_w = (w - left - 4.0) / bands
        painter.setPen(Qt.PenStyle.NoPen)
        off = QColor(LED_OFF)
        colors = [QColor(c) for c in LED_COLORS]
        for i, centre in enumerate(THIRD_OCTAVE_CENTRES):
            x = left + i * col_w
            level = self._levels[i]
            lit = 0 if math.isnan(level) else int(math.ceil((level - self.FLOOR_DB) / self.SEGMENT_DB))
            peak = int(math.ceil((self._peaks[i] - self.FLOOR_DB) / self.SEGMENT_DB))
            for s in range(segments):
                y = top + plot_h - (s + 1) * seg_h
                db_top = self.FLOOR_DB + (s + 1) * self.SEGMENT_DB
                if s < lit or s == peak - 1:
                    painter.setBrush(colors[2] if db_top > -6.0 else colors[1] if db_top > -18.0 else colors[0])
                else:
                    painter.setBrush(off)
                painter.drawRect(QRectF(x + 1.5, y + 1.0, col_w - 3.0, seg_h - 2.0))
        painter.setPen(QColor(TEXT))
        for i, centre in enumerate(THIRD_OCTAVE_CENTRES):
            if i % 3 == 0 or col_w > 34:
                painter.drawText(QRectF(left + i * col_w - 10, h - bottom + 3, col_w + 20, 14), Qt.AlignmentFlag.AlignCenter, _band_label(centre))


def _colour_table() -> list[int]:
    """256 colours from black through violet, orange and yellow to white (quiet -> loud)."""
    stops = ((0.0, (8, 10, 20)), (0.25, (60, 20, 110)), (0.5, (190, 50, 90)), (0.75, (245, 150, 40)), (1.0, (255, 250, 200)))
    table = []
    for index in range(256):
        t = index / 255.0
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t <= t1:
                f = (t - t0) / (t1 - t0)
                r, g, b = (int(round(a + (b_ - a) * f)) for a, b_ in zip(c0, c1))
                table.append((255 << 24) | (r << 16) | (g << 8) | b)
                break
    return table


class SpectrogramView(_TimedWidget):
    """History of the spectrum: frequency (logarithmic) across, time down, level as colour."""

    COLUMNS = 320
    ROWS = 240
    FLOOR_DB = -110.0
    TOP_DB = -10.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._data = np.zeros((self.ROWS, self.COLUMNS), dtype=np.uint8)
        self._table = _colour_table()
        self._f_lo, self._f_hi = 20.0, 20000.0
        self._grid = np.geomspace(self._f_lo, self._f_hi, self.COLUMNS)
        self._seconds_per_row = 1.0 / 60.0
        self._last = time.perf_counter()

    def set_range(self, f_lo: float, f_hi: float) -> None:
        f_hi = max(f_hi, 2.0 * f_lo)
        if (f_lo, f_hi) != (self._f_lo, self._f_hi):
            self._f_lo, self._f_hi = f_lo, f_hi
            self._grid = np.geomspace(f_lo, f_hi, self.COLUMNS)
            self._data[:] = 0

    def add_spectrum(self, frequencies: np.ndarray, magnitude_db: np.ndarray) -> None:
        """Append one spectrum (dBFS) as the newest row."""
        now = time.perf_counter()
        self._seconds_per_row = 0.9 * self._seconds_per_row + 0.1 * min(now - self._last, 0.5)
        self._last = now
        row = np.interp(self._grid, frequencies, magnitude_db)
        scaled = (row - self.FLOOR_DB) * (255.0 / (self.TOP_DB - self.FLOOR_DB))
        self._data[:-1] = self._data[1:]
        self._data[-1] = np.clip(scaled, 0, 255).astype(np.uint8)
        self.update()

    def clear(self) -> None:
        self._data[:] = 0

    def _paint(self, painter: QPainter) -> None:
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(BG))
        left, bottom, top = 44.0, 20.0, 6.0
        area = QRectF(left, top, w - left - 4.0, h - bottom - top)
        image = QImage(self._data.data, self.COLUMNS, self.ROWS, self.COLUMNS, QImage.Format.Format_Indexed8)
        image.setColorTable(self._table)
        painter.drawImage(area, image)
        font = QFont(self.font())
        font.setPixelSize(10)
        painter.setFont(font)
        span = math.log(self._f_hi / self._f_lo)
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1.0))
        ticks = [f for f in (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000) if self._f_lo <= f <= self._f_hi]
        if len(ticks) < 3:
            ticks = [float(f) for f in np.geomspace(self._f_lo, self._f_hi, 6)]
        for f in ticks:
            x = area.left() + area.width() * math.log(f / self._f_lo) / span
            painter.drawLine(int(x), int(area.top()), int(x), int(area.bottom()))
        painter.setPen(QColor(TEXT))
        for f in ticks:
            x = area.left() + area.width() * math.log(f / self._f_lo) / span
            text = f"{f / 1000:g} kHz" if f >= 1000 else f"{f:.3g} Hz"
            painter.drawText(QRectF(x - 30, h - bottom + 3, 60, 14), Qt.AlignmentFlag.AlignCenter, text)
        total = self._seconds_per_row * self.ROWS
        painter.drawText(QRectF(0, area.bottom() - 14, left - 6, 14), Qt.AlignmentFlag.AlignRight, "acum")
        painter.drawText(QRectF(0, area.top(), left - 6, 14), Qt.AlignmentFlag.AlignRight, f"-{total:.0f} s")
