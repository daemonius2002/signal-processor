"""Main window: toolbar, control sidebar, real-time plots, metrics and latency telemetry."""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, QSettings, QSignalBlocker, QSize, Qt, QTemporaryDir, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QGuiApplication, QIcon, QKeySequence, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from dsp import FFT_SIZES, FILTER_TYPES, SCALES, WINDOW_FUNCTIONS, DSPConfig, FrameResult, SignalProcessor, __version__
from io_handlers import sky_catalog
from io_handlers import (
    SampleRateRequiredError,
    SignalData,
    StreamSimulator,
    SyntheticGenerator,
    load_csv,
    load_wav,
)

from .audio import AudioPlayer, MicrophoneInput
from .sky_panel import SkyPanel
from .solar_panel import SolarPanel
from .sound_panel import SoundKitPanel

APP_NAME = "Signal Processor"
LATENCY_BUDGET_MS = 10.0
GENERATOR_SAMPLE_RATE = 48000.0
TEXT_REFRESH_S = 0.1                 # numeric read-outs are refreshed 10 times per second
LATENCY_WINDOW = 120                 # frames used for the average / worst-case latency
WARMUP_FRAMES = 30                   # first frames (window creation, first paints) are not rated
FILTER_CURVE_POINTS = 400
SIDEBAR_WIDTH = 300
LAYOUT_VERSION = 3                   # bump when docks change, so an old saved layout is ignored

# Palette: a bench instrument at night. Two trace colours carry meaning everywhere:
# teal = time domain / DSP, amber = frequency domain / rendering.
COLOR_BG = "#0f141b"
COLOR_PANEL = "#161d27"
COLOR_RAISED = "#1e2733"
COLOR_BORDER = "#2a3544"
COLOR_TEXT = "#dfe6ee"
COLOR_MUTED = "#8595a8"
COLOR_FAINT = "#566477"
COLOR_ACCENT = "#4dd0c4"             # waveform trace, controls, DSP share of the frame time
COLOR_SPECTRUM = "#f2b24a"           # spectrum trace, render share of the frame time
COLOR_PEAK = "#ff6b8b"
COLOR_OK = "#57d08a"
COLOR_WARN = "#f0c04a"
COLOR_BAD = "#ff6b6b"
FONT_STACK = '"Segoe UI", "SF Pro Text", "Inter", "Noto Sans", "DejaVu Sans", sans-serif'


def _write_control_images(folder: str) -> dict[str, str]:
    """Draw the small arrows / tick used by spin boxes, combo boxes and check boxes.

    Qt style sheets can only take such glyphs from image files, so they are
    painted once at start-up into a temporary folder (nothing ships as a file).
    """
    def chevron(name: str, color: str, up: bool) -> str:
        pix = QPixmap(18, 12)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(color), 2.0)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        path = QPainterPath()
        if up:
            path.moveTo(4, 8.5); path.lineTo(9, 3.5); path.lineTo(14, 8.5)
        else:
            path.moveTo(4, 3.5); path.lineTo(9, 8.5); path.lineTo(14, 3.5)
        painter.drawPath(path)
        painter.end()
        target = os.path.join(folder, name)
        pix.save(target, "PNG")
        return target.replace("\\", "/")

    def tick(name: str) -> str:
        pix = QPixmap(24, 24)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(COLOR_BG), 3.4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        path = QPainterPath()
        path.moveTo(5.5, 12.5); path.lineTo(10, 17); path.lineTo(18.5, 7.5)
        painter.drawPath(path)
        painter.end()
        target = os.path.join(folder, name)
        pix.save(target, "PNG")
        return target.replace("\\", "/")

    return {
        "up": chevron("up.png", COLOR_MUTED, True),
        "down": chevron("down.png", COLOR_MUTED, False),
        "up_off": chevron("up_off.png", COLOR_BORDER, True),
        "down_off": chevron("down_off.png", COLOR_BORDER, False),
        "tick": tick("tick.png"),
    }


def build_style_sheet(img: dict[str, str]) -> str:
    return f"""
QMainWindow, QWidget {{ background: {COLOR_BG}; color: {COLOR_TEXT}; font-family: {FONT_STACK}; font-size: 13px; }}
QToolTip {{ background: {COLOR_RAISED}; color: {COLOR_TEXT}; border: 1px solid {COLOR_BORDER}; padding: 4px 6px; }}

QToolBar {{ background: {COLOR_PANEL}; border: none; border-bottom: 1px solid {COLOR_BORDER}; padding: 6px 8px; spacing: 2px; }}
QToolBar::separator {{ background: {COLOR_BORDER}; width: 1px; margin: 5px 8px; }}
QToolBar QWidget {{ background: transparent; }}
QToolButton {{ color: {COLOR_TEXT}; border: 1px solid transparent; border-radius: 5px; padding: 6px 11px; }}
QToolButton:hover {{ background: {COLOR_RAISED}; border-color: {COLOR_BORDER}; }}
QToolButton:pressed {{ background: {COLOR_BORDER}; }}
QToolButton:checked {{ background: #14323a; border-color: {COLOR_ACCENT}; color: {COLOR_ACCENT}; }}
QToolButton:focus {{ border-color: {COLOR_FAINT}; }}
QLabel#chip {{ background: {COLOR_RAISED}; border: 1px solid {COLOR_BORDER}; border-radius: 9px; padding: 1px 10px; color: {COLOR_TEXT}; font-size: 12px; }}
QLabel#version {{ color: {COLOR_FAINT}; padding: 0 4px 0 10px; }}

QGroupBox {{ background: {COLOR_PANEL}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; margin-top: 0px; padding: 32px 12px 10px 12px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: padding; subcontrol-position: top left; left: 12px; top: 10px; color: {COLOR_TEXT}; }}
QGroupBox QLabel, QGroupBox QCheckBox {{ background: transparent; font-weight: 400; color: {COLOR_MUTED}; }}
QGroupBox QLabel#colHead {{ color: {COLOR_FAINT}; font-size: 11px; }}
QGroupBox:disabled::title {{ color: {COLOR_FAINT}; }}

QComboBox, QSpinBox, QDoubleSpinBox {{ background: {COLOR_BG}; color: {COLOR_TEXT}; border: 1px solid {COLOR_BORDER}; border-radius: 5px; padding: 4px 8px; min-height: 20px; font-weight: 400; selection-background-color: #1f4d55; }}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {COLOR_FAINT}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {COLOR_ACCENT}; }}
QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{ color: {COLOR_FAINT}; background: {COLOR_PANEL}; }}
QLabel:disabled, QCheckBox:disabled {{ color: {COLOR_FAINT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QComboBox::down-arrow {{ image: url({img['down']}); width: 9px; height: 6px; }}
QComboBox::down-arrow:disabled {{ image: url({img['down_off']}); }}
QComboBox QAbstractItemView {{ background: {COLOR_RAISED}; border: 1px solid {COLOR_BORDER}; outline: none; selection-background-color: #1f4d55; selection-color: {COLOR_TEXT}; padding: 2px; }}
QSpinBox, QDoubleSpinBox {{ padding-right: 24px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 20px; border-left: 1px solid {COLOR_BORDER}; border-top-right-radius: 5px; background: {COLOR_BG}; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 20px; border-left: 1px solid {COLOR_BORDER}; border-top: 1px solid {COLOR_BORDER}; border-bottom-right-radius: 5px; background: {COLOR_BG}; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ background: {COLOR_RAISED}; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({img['up']}); width: 9px; height: 6px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({img['down']}); width: 9px; height: 6px; }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{ image: url({img['up_off']}); }}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ image: url({img['down_off']}); }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border: 1px solid {COLOR_FAINT}; border-radius: 4px; background: {COLOR_BG}; }}
QCheckBox::indicator:checked {{ background: {COLOR_ACCENT}; border-color: {COLOR_ACCENT}; image: url({img['tick']}); }}
QCheckBox::indicator:disabled {{ border-color: {COLOR_BORDER}; background: {COLOR_PANEL}; }}

QSlider::groove:horizontal {{ height: 4px; background: {COLOR_BORDER}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {COLOR_ACCENT}; border-radius: 2px; }}
QSlider::sub-page:horizontal:disabled {{ background: {COLOR_BORDER}; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; background: {COLOR_TEXT}; border: 2px solid {COLOR_ACCENT}; }}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}
QSlider::handle:horizontal:disabled {{ background: {COLOR_FAINT}; border-color: {COLOR_BORDER}; }}

QStatusBar {{ background: {COLOR_PANEL}; border-top: 1px solid {COLOR_BORDER}; color: {COLOR_MUTED}; }}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ background: transparent; padding: 2px 8px; color: {COLOR_MUTED}; }}
QScrollArea {{ border: none; }}
QScrollBar:vertical {{ background: {COLOR_BG}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {COLOR_BORDER}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {COLOR_FAINT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QSplitter::handle:vertical {{ height: 8px; background: {COLOR_BG}; }}

QFrame#pane, QFrame#strip {{ background: {COLOR_PANEL}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; }}
QFrame#pane QLabel, QFrame#strip QLabel, QFrame#strip QWidget {{ background: transparent; }}
QLabel#paneTitle {{ font-weight: 600; }}
QLabel#paneInfo {{ color: {COLOR_MUTED}; font-size: 12px; }}
QLabel#verdict {{ font-size: 12px; font-weight: 600; }}
QLabel#readoutValue {{ font-size: 17px; font-weight: 600; }}
QLabel#readoutTitle {{ color: {COLOR_MUTED}; font-size: 12px; }}
QLabel#readoutSub {{ color: {COLOR_FAINT}; font-size: 12px; }}
QFrame#divider {{ background: {COLOR_BORDER}; border: none; max-width: 1px; min-width: 1px; }}
QDockWidget {{ color: {COLOR_MUTED}; font-weight: 600; titlebar-close-icon: none; titlebar-normal-icon: none; }}
QDockWidget::title {{ background: {COLOR_BG}; padding: 6px 4px; text-align: left; }}
QMainWindow::separator {{ background: {COLOR_BG}; width: 8px; height: 8px; }}
QMainWindow::separator:hover {{ background: {COLOR_BORDER}; }}
QTabBar::tab {{ background: {COLOR_PANEL}; color: {COLOR_MUTED}; border: 1px solid {COLOR_BORDER}; padding: 5px 12px; border-radius: 5px; margin: 2px; }}
QTabBar::tab:selected {{ color: {COLOR_ACCENT}; border-color: {COLOR_ACCENT}; }}
QTreeWidget {{ background: {COLOR_PANEL}; border: 1px solid {COLOR_BORDER}; border-radius: 8px; outline: none; padding: 4px; font-size: 13px; }}
QTreeWidget::item {{ padding: 3px 4px; border-radius: 4px; }}
QTreeWidget::item:hover {{ background: {COLOR_RAISED}; }}
QTreeWidget::item:selected {{ background: #1f4d55; color: {COLOR_TEXT}; }}
QLabel#skyName {{ font-size: 17px; font-weight: 600; }}
QLabel#skyFacts {{ color: {COLOR_MUTED}; font-size: 12px; }}
QLabel#skyStory {{ font-size: 14px; }}
QListWidget {{ background: {COLOR_BG}; border: 1px solid {COLOR_BORDER}; border-radius: 6px; outline: none; padding: 3px; font-size: 13px; }}
QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
QListWidget::item:hover {{ background: {COLOR_RAISED}; }}
QListWidget::item:selected {{ background: #1f4d55; }}
QLabel#legendDsp {{ color: {COLOR_ACCENT}; font-size: 12px; }}
QLabel#legendRender {{ color: {COLOR_SPECTRUM}; font-size: 12px; }}
QLabel#legendWorst {{ color: {COLOR_MUTED}; font-size: 12px; }}
"""


def _make_icon(kind: str, color: str = COLOR_TEXT) -> QIcon:
    """Toolbar icons, drawn as vector strokes so they stay sharp at any display scale."""
    scale = 3
    pix = QPixmap(18 * scale, 18 * scale)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(scale, scale)
    pen = QPen(QColor(color), 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    path = QPainterPath()
    if kind == "wav":                       # a short burst of audio bars
        for x, h in ((3, 3), (6, 6.5), (9, 4), (12, 7), (15, 2.5)):
            path.moveTo(x, 9 - h); path.lineTo(x, 9 + h)
    elif kind == "csv":                     # a table
        path.addRoundedRect(QRectF(2.5, 3.5, 13, 11), 1.5, 1.5)
        path.moveTo(2.5, 7.5); path.lineTo(15.5, 7.5)
        path.moveTo(2.5, 11); path.lineTo(15.5, 11)
        path.moveTo(7.5, 7.5); path.lineTo(7.5, 14.5)
    elif kind == "live":                    # one period of a sine
        path.moveTo(2, 9)
        path.cubicTo(4.5, 0.5, 6.5, 0.5, 9, 9)
        path.cubicTo(11.5, 17.5, 13.5, 17.5, 16, 9)
    elif kind == "play":
        painter.setBrush(QColor(color))
        path.moveTo(5.5, 3.5); path.lineTo(14.5, 9); path.lineTo(5.5, 14.5); path.closeSubpath()
    elif kind == "pause":
        painter.setBrush(QColor(color))
        path.addRoundedRect(QRectF(4.5, 3.5, 3, 11), 0.8, 0.8)
        path.addRoundedRect(QRectF(10.5, 3.5, 3, 11), 0.8, 0.8)
    elif kind == "sky":                     # a four-pointed star
        painter.setBrush(QColor(color))
        path.moveTo(9, 1.5); path.lineTo(10.8, 7.2); path.lineTo(16.5, 9); path.lineTo(10.8, 10.8)
        path.lineTo(9, 16.5); path.lineTo(7.2, 10.8); path.lineTo(1.5, 9); path.lineTo(7.2, 7.2); path.closeSubpath()
    elif kind == "mic":
        path.addRoundedRect(QRectF(6.5, 2, 5, 9), 2.5, 2.5)
        path.moveTo(4, 8.5); path.cubicTo(4, 15, 14, 15, 14, 8.5)
        path.moveTo(9, 13.4); path.lineTo(9, 16.5)
    elif kind == "controls":                # three sliders
        for y, x in ((4.5, 11.5), (9, 6), (13.5, 10)):
            path.moveTo(2.5, y); path.lineTo(15.5, y)
            path.addEllipse(QPointF(x, y), 1.8, 1.8)
    elif kind == "planet":
        path.addEllipse(QPointF(9, 9), 4.2, 4.2)
        path.addEllipse(QPointF(9, 9), 8, 2.6)
    elif kind == "sounds":                  # a note
        painter.setBrush(QColor(color))
        path.addEllipse(QPointF(6, 13.5), 2.6, 2.1)
        path.moveTo(8.5, 13.5); path.lineTo(8.5, 3); path.lineTo(14, 4.5); path.lineTo(14, 7)
    elif kind == "speaker":
        path.moveTo(3, 7); path.lineTo(6, 7); path.lineTo(10, 3.5); path.lineTo(10, 14.5); path.lineTo(6, 11); path.lineTo(3, 11); path.closeSubpath()
        path.moveTo(12.5, 6.5); path.cubicTo(14.5, 8, 14.5, 10, 12.5, 11.5)
    elif kind == "export":                  # arrow leaving a tray
        path.moveTo(9, 11); path.lineTo(9, 2.5)
        path.moveTo(5.5, 6); path.lineTo(9, 2.5); path.lineTo(12.5, 6)
        path.moveTo(3, 11); path.lineTo(3, 15); path.lineTo(15, 15); path.lineTo(15, 11)
    painter.drawPath(path)
    painter.end()
    pix.setDevicePixelRatio(scale)
    return QIcon(pix)


def _tabular_font(pixel_size: int, weight: QFont.Weight) -> QFont:
    """Font for live numbers: equal-width digits, so values do not jitter sideways."""
    font = QFont()
    font.setFamilies(["Segoe UI", "SF Pro Text", "Inter", "Noto Sans", "DejaVu Sans"])
    font.setPixelSize(pixel_size)
    font.setWeight(weight)
    try:
        font.setFeature(QFont.Tag("tnum"), 1)
    except (AttributeError, TypeError):      # Qt older than 6.7
        pass
    return font


# ----------------------------------------------------------------- utilities
def _json_safe(value):
    """Replace NaN / infinity (not valid JSON) by ``None``, recursively."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def export_metrics(path: str, payload: dict) -> str:
    """Write a metrics snapshot to ``path`` as JSON or CSV, chosen by the file extension.

    ``payload`` is a dict of sections (dicts) and scalars. The CSV has the
    columns ``section,key,value``. Returns the format written ("json" / "csv").
    """
    payload = _json_safe(payload)
    if path.lower().endswith(".csv"):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["section", "key", "value"])
            for section, content in payload.items():
                if isinstance(content, dict):
                    for key, value in content.items():
                        writer.writerow([section, key, "" if value is None else value])
                else:
                    writer.writerow(["general", section, "" if content is None else content])
        return "csv"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return "json"


def _format_frequency(hz: float) -> str:
    if abs(hz) >= 1000.0:
        return f"{hz / 1000.0:.4f} kHz"
    if abs(hz) < 10.0:                                 # slow pulsars: 0.7478 Hz, not 0.75 Hz
        return f"{hz:.4f} Hz"
    return f"{hz:.2f} Hz"


class TimedPlotWidget(pg.PlotWidget):
    """PlotWidget that measures how long its paint events really take."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.paint_ms_accum = 0.0

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        t0 = time.perf_counter()
        super().paintEvent(event)
        self.paint_ms_accum += (time.perf_counter() - t0) * 1e3

    def take_paint_ms(self) -> float:
        """Paint time accumulated since the previous call, in ms."""
        value, self.paint_ms_accum = self.paint_ms_accum, 0.0
        return value


class Readout(QWidget):
    """One numeric read-out: title, large value, optional sub-line."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 10, 9)
        layout.setSpacing(1)
        self._title = QLabel(title)
        self._title.setObjectName("readoutTitle")
        self._value = QLabel("-")
        self._value.setObjectName("readoutValue")
        self._value.setFont(_tabular_font(17, QFont.Weight.DemiBold))
        self._sub = QLabel(" ")
        self._sub.setObjectName("readoutSub")
        self._sub.setFont(_tabular_font(12, QFont.Weight.Normal))
        for label in (self._title, self._value, self._sub):
            layout.addWidget(label)
        self.setMinimumWidth(126)

    def set_value(self, text: str, sub: str = "") -> None:
        self._value.setText(text)
        self._sub.setText(sub or " ")


class ReadoutStrip(QFrame):
    """Read-out row that drops the less important values when it gets narrow."""

    optional: list[QWidget] = []

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        roomy = event.size().width() >= 1010
        for widget in self.optional:
            widget.setVisible(roomy)
        super().resizeEvent(event)


class BudgetBar(QWidget):
    """Frame time against the budget: DSP share, render share and a worst-case tick."""

    def __init__(self, budget_ms: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._budget = budget_ms
        self._dsp = 0.0
        self._render = 0.0
        self._worst = 0.0
        self.setFixedHeight(12)
        self.setMinimumWidth(180)

    def set_values(self, dsp_ms: float, render_ms: float, worst_ms: float) -> None:
        self._dsp, self._render, self._worst = dsp_ms, render_ms, worst_ms
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = float(self.width()), float(self.height())
        full = self._budget * 1.25                       # the bar shows 0 .. 125 % of the budget
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLOR_BG))
        painter.drawRoundedRect(QRectF(0, 2, w, h - 4), 4, 4)
        x_dsp = min(self._dsp / full, 1.0) * w
        x_total = min((self._dsp + self._render) / full, 1.0) * w
        if x_total > 0:
            painter.setBrush(QColor(COLOR_SPECTRUM))
            painter.drawRoundedRect(QRectF(0, 2, x_total, h - 4), 4, 4)
        if x_dsp > 0:
            painter.setBrush(QColor(COLOR_ACCENT))
            painter.drawRoundedRect(QRectF(0, 2, x_dsp, h - 4), 4, 4)
            if x_total > x_dsp + 4:                      # square off the joint between the shares
                painter.drawRect(QRectF(x_dsp - 4, 2, 4, h - 4))
        x_budget = self._budget / full * w
        painter.setPen(QPen(QColor(COLOR_TEXT), 1.5))
        painter.drawLine(QPointF(x_budget, 0), QPointF(x_budget, h))
        x_worst = min(self._worst / full, 1.0) * w
        painter.setPen(QPen(QColor(COLOR_MUTED), 1.5))
        painter.drawLine(QPointF(x_worst, 2), QPointF(x_worst, h - 2))
        painter.end()


class LatencyMeter(QWidget):
    """Frame-time read-out with a budget bar; the proof for the 10 ms requirement."""

    def __init__(self, budget_ms: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._budget = budget_ms
        self._color = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 9, 14, 9)
        layout.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(8)
        title = QLabel("Frame time")
        title.setObjectName("readoutTitle")
        self._total = QLabel("-")
        self._total.setObjectName("readoutValue")
        self._total.setFont(_tabular_font(17, QFont.Weight.DemiBold))
        self._verdict = QLabel("")
        self._verdict.setObjectName("verdict")
        self._verdict.setFont(_tabular_font(12, QFont.Weight.DemiBold))
        top.addWidget(title)
        top.addStretch(1)
        top.addWidget(self._verdict)
        layout.addLayout(top)

        middle = QHBoxLayout()
        middle.setSpacing(12)
        middle.addWidget(self._total)
        self.bar = BudgetBar(budget_ms)
        middle.addWidget(self.bar, 1)
        layout.addLayout(middle)

        legend = QHBoxLayout()
        legend.setSpacing(10)
        self._dsp = QLabel("DSP -")
        self._dsp.setObjectName("legendDsp")
        self._render = QLabel("Render -")
        self._render.setObjectName("legendRender")
        self._worst = QLabel("worst -")
        self._worst.setObjectName("legendWorst")
        for label in (self._dsp, self._render, self._worst):
            label.setFont(_tabular_font(12, QFont.Weight.Normal))
            legend.addWidget(label)
        legend.addStretch(1)
        layout.addLayout(legend)
        self.setMinimumWidth(280)
        self.setToolTip(
            "Average time per frame over the last 120 frames.\n"
            "Teal: DSP (filter + FFT + metrics). Amber: rendering (plot update + paint).\n"
            f"White line: the {budget_ms:.0f} ms budget. Grey tick: slowest frame in the window."
        )

    def set_values(self, dsp_ms: float, render_ms: float, worst_ms: float, color: str, verdict: str) -> None:
        self._total.setText(f"{dsp_ms + render_ms:.2f} ms")
        self._verdict.setText(verdict)
        self._dsp.setText(f"DSP {dsp_ms:.2f}")
        self._render.setText(f"Render {render_ms:.2f}")
        self._worst.setText(f"worst {worst_ms:.2f} ms")
        self.bar.set_values(dsp_ms, render_ms, worst_ms)
        if color != self._color:                         # restyling is slow; only on a change
            self._color = color
            self._verdict.setStyleSheet(f"color: {color};")


# --------------------------------------------------------------- main window
class MainWindow(QMainWindow):
    """Application main window."""

    benchmark_finished = pyqtSignal(dict)

    def __init__(self, benchmark_frames: int = 0) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{__version__}")
        self.resize(1360, 860)
        self.setMinimumSize(1200, 680)
        self.setAcceptDrops(True)
        self._assets = QTemporaryDir()                      # arrow / tick images for the style sheet
        self.setStyleSheet(build_style_sheet(_write_control_images(self._assets.path())))

        self._benchmark_frames = int(benchmark_frames)
        self._benchmark_log: list[tuple[float, float]] = []

        self.processor = SignalProcessor(GENERATOR_SAMPLE_RATE, DSPConfig())
        self.generator = SyntheticGenerator(GENERATOR_SAMPLE_RATE)
        self.signal: SignalData | None = None
        self.stream = StreamSimulator(self.generator, frame_rate=60.0)
        self.playing = True
        self.spectrum_kind = "Magnitude"          # or "PSD"

        self._result: FrameResult | None = None
        self._spec_y: np.ndarray | None = None
        self._time_x = np.zeros(0)
        self._last_tick = time.perf_counter()
        self._last_text_update = 0.0
        self._update_ms_prev = 0.0
        self._dsp_hist: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._render_hist: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._tick_times: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._cursor: tuple[float, float] | None = None
        self._time_top = 0.0
        self._spec_top: float | None = None        # None = Y range must be set on the next frame
        self._status_color = ""
        self._restoring = False
        self._moving_cut_line = False
        self._view_fmax: float | None = None        # spectrum zoom for sky objects
        self._sky_object = None                      # object chosen on the sky map / solar system
        self._kit_entry: dict | None = None          # sound chosen in the sound kit
        self._loop_audio: np.ndarray | None = None   # what the loudspeakers play for the current source
        self.audio = AudioPlayer(self)
        self.microphone = MicrophoneInput(self)
        self._frames_seen = 0

        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._on_tick)

        self._sync_controls_to_source()
        self._update_filter_controls()
        self._apply_frame_rate()
        if not self._benchmark_frames and QGuiApplication.platformName() != "offscreen":
            self._restore_layout()
        self._timer.start()

    # ------------------------------------------------------------- toolbar
    def _build_toolbar(self) -> None:
        bar = QToolBar("Main")
        bar.setMovable(False)
        bar.setIconSize(QSize(18, 18))
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        bar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.addToolBar(bar)

        self._icon_play = _make_icon("play")
        self._icon_pause = _make_icon("pause")
        live_icon = QIcon()
        live_icon.addPixmap(_make_icon("live").pixmap(54, 54), QIcon.Mode.Normal, QIcon.State.Off)
        live_icon.addPixmap(_make_icon("live", COLOR_ACCENT).pixmap(54, 54), QIcon.Mode.Normal, QIcon.State.On)

        self.act_load_wav = QAction(_make_icon("wav"), "Load WAV", self)
        self.act_load_wav.setShortcut(QKeySequence("Ctrl+O"))
        self.act_load_wav.setToolTip("Open a .wav file (Ctrl+O). Files can also be dropped on the window.")
        self.act_load_wav.triggered.connect(self._on_load_wav)

        self.act_load_csv = QAction(_make_icon("csv"), "Load CSV", self)
        self.act_load_csv.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.act_load_csv.setToolTip("Open a .csv file: amplitude only, or time and amplitude (Ctrl+Shift+O)")
        self.act_load_csv.triggered.connect(self._on_load_csv)

        self.act_live = QAction(live_icon, "Live Generator", self)
        self.act_live.setCheckable(True)
        self.act_live.setChecked(True)
        self.act_live.setShortcut(QKeySequence("Ctrl+G"))
        self.act_live.setToolTip("Built-in two-tone + noise generator (Ctrl+G)")
        self.act_live.toggled.connect(self._on_live_toggled)

        def two_state(kind: str) -> QIcon:
            icon = QIcon()
            icon.addPixmap(_make_icon(kind).pixmap(54, 54), QIcon.Mode.Normal, QIcon.State.Off)
            icon.addPixmap(_make_icon(kind, COLOR_ACCENT).pixmap(54, 54), QIcon.Mode.Normal, QIcon.State.On)
            return icon

        self._two_state_icon = two_state
        self.act_mic = QAction(two_state("mic"), "Microfon", self)
        self.act_mic.setCheckable(True)
        self.act_mic.setShortcut(QKeySequence("Ctrl+R"))
        self.act_mic.setToolTip("Analizează în direct sunetul de la microfon: vorbește, cântă, bate din palme (Ctrl+R)")
        self.act_mic.toggled.connect(self._on_mic_toggled)

        self.act_sound = QAction(two_state("speaker"), "Sunet", self)
        self.act_sound.setCheckable(True)
        self.act_sound.setChecked(self.audio.available)
        self.act_sound.setEnabled(self.audio.available)
        self.act_sound.setToolTip("Redă în boxe sunetul ales din Cer, Planete sau Sunete"
                                  if self.audio.available else f"Sunet indisponibil: {self.audio.error}")
        self.act_sound.toggled.connect(lambda _: self._apply_audio())

        self.act_play = QAction(self._icon_pause, "Pause", self)
        self.act_play.setShortcut(QKeySequence(Qt.Key.Key_Space))
        self.act_play.setToolTip("Pause or resume the stream (Space)")
        self.act_play.triggered.connect(self._on_play_pause)

        self.act_export = QAction(_make_icon("export"), "Export Metrics", self)
        self.act_export.setShortcut(QKeySequence("Ctrl+E"))
        self.act_export.setToolTip("Save the current metrics as CSV or JSON (Ctrl+E)")
        self.act_export.triggered.connect(self._on_export)

        bar.addAction(self.act_load_wav)
        bar.addAction(self.act_load_csv)
        bar.addAction(self.act_live)
        bar.addAction(self.act_mic)
        bar.addSeparator()
        bar.addAction(self.act_play)
        bar.addSeparator()
        bar.addAction(self.act_export)

        bar.addSeparator()
        self._panel_bar = bar                              # panel buttons are added in _build_central
        self._panel_anchor = bar.addSeparator()
        bar.addAction(self.act_sound)
        self.sld_volume = QSlider(Qt.Orientation.Horizontal)
        self.sld_volume.setRange(0, 100)
        self.sld_volume.setValue(60)
        self.sld_volume.setFixedWidth(80)
        self.sld_volume.setEnabled(self.audio.available)
        self.sld_volume.setToolTip("Volum")
        self.sld_volume.valueChanged.connect(lambda value: self.audio.set_volume(value / 100.0))
        bar.addWidget(self.sld_volume)

        # Shown in the waveform header (source name) and in the status bar (channel selector).
        self.cmb_channel = QComboBox()
        self.cmb_channel.setToolTip("Channel of the loaded WAV file")
        self.cmb_channel.setMinimumWidth(170)
        self.cmb_channel.currentIndexChanged.connect(self._on_channel_changed)
        self.lbl_source = QLabel("")
        self.lbl_source.setObjectName("chip")

    # ------------------------------------------------------------- central
    def _build_central(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        right = QVBoxLayout(central)
        right.setContentsMargins(10, 10, 10, 10)
        right.setSpacing(10)
        right.addWidget(self._build_plots(), 1)
        right.addWidget(self._build_metrics_panel())

        sidebar = QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sidebar.setMinimumWidth(SIDEBAR_WIDTH + 12)
        sidebar.setWidget(self._build_sidebar())

        self.sky_panel = SkyPanel()
        self.solar_panel = SolarPanel()
        self.kit_panel = SoundKitPanel()
        self.sky_panel.object_selected.connect(self._on_sky_object)
        self.solar_panel.object_selected.connect(self._on_sky_object)
        self.kit_panel.sound_selected.connect(self._on_kit_sound)

        # Every panel is a dock: drag its title to another edge, drop it on another panel to
        # stack them as tabs, or pull it out as a separate window (for a second screen).
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowTabbedDocks
                            | QMainWindow.DockOption.AllowNestedDocks)
        self.dock_controls = self._add_dock("Controls", "controls", "controls", sidebar, "Ctrl+1",
                                            "Setările analizei: FFT, fereastră, filtru, generator")
        self.dock_sky = self._add_dock("Cer", "sky", "sky", self.sky_panel, "Ctrl+2",
                                       "Harta cerului: pulsari și stele de ascultat")
        self.dock_solar = self._add_dock("Planete", "solar", "planet", self.solar_panel, "Ctrl+3",
                                         "Sistemul Solar: fiecare planetă are nota ei")
        self.dock_kit = self._add_dock("Sunete", "kit", "sounds", self.kit_panel, "Ctrl+4",
                                       "Kit de sunete: tonuri, instrumente, muzică, voce, natură")
        self._learning_docks = (self.dock_sky, self.dock_solar, self.dock_kit)
        self.tabifyDockWidget(self.dock_sky, self.dock_solar)
        self.tabifyDockWidget(self.dock_solar, self.dock_kit)
        for dock in self._learning_docks:
            dock.hide()
            dock.visibilityChanged.connect(lambda visible, d=dock: self._on_learning_dock(d, visible))

    def _add_dock(self, title: str, name: str, icon: str, widget: QWidget, shortcut: str, tip: str) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"dock_{name}")
        dock.setWidget(widget)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                         | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        action = dock.toggleViewAction()
        action.setIcon(self._two_state_icon(icon))
        action.setShortcut(QKeySequence(shortcut))
        action.setToolTip(f"{tip} ({shortcut}). Trage de titlul panoului ca să-l muți.")
        self._panel_bar.insertAction(self._panel_anchor, action)
        return dock

    def _on_learning_dock(self, dock: QDockWidget, visible: bool) -> None:
        """Opening a wide learning panel docked in the window makes room by hiding the controls."""
        if visible and not dock.isFloating() and not self.dock_controls.isFloating() and not self._restoring:
            self.dock_controls.hide()
        self._restart_latency_stats()

    def _restore_layout(self) -> None:
        """Bring back the window and panel arrangement of the previous session."""
        settings = QSettings("SignalProcessor", "SignalProcessor")
        self._restoring = True
        try:
            geometry = settings.value("geometry")
            state = settings.value("state")
            if geometry is not None:
                self.restoreGeometry(geometry)
            if state is not None:
                self.restoreState(state, LAYOUT_VERSION)
        except Exception:                                  # a damaged setting must not stop the start-up
            pass
        self._restoring = False

    def reset_layout(self) -> None:
        """Put every panel back where it was on first start."""
        self._restoring = True
        for dock in (self.dock_controls, *self._learning_docks):
            dock.setFloating(False)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.tabifyDockWidget(self.dock_sky, self.dock_solar)
        self.tabifyDockWidget(self.dock_solar, self.dock_kit)
        for dock in self._learning_docks:
            dock.hide()
        self.dock_controls.show()
        self._restoring = False

    @staticmethod
    def _grid(box: QGroupBox) -> QGridLayout:
        grid = QGridLayout(box)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(7)
        grid.setColumnStretch(1, 1)
        return grid

    def _build_sidebar(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(SIDEBAR_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # --- analysis
        box = QGroupBox("Analysis")
        grid = self._grid(box)
        self.cmb_fft = QComboBox()
        for size in FFT_SIZES:
            self.cmb_fft.addItem(f"{size} points", size)
        self.cmb_fft.setCurrentIndex(FFT_SIZES.index(self.processor.config.fft_size))
        self.cmb_fft.currentIndexChanged.connect(lambda _: self._apply_config(fft_size=self.cmb_fft.currentData()))
        self.cmb_window = QComboBox()
        self.cmb_window.addItems(WINDOW_FUNCTIONS)
        self.cmb_window.currentTextChanged.connect(lambda text: self._apply_config(window=text))
        self.cmb_kind = QComboBox()
        self.cmb_kind.addItems(["Magnitude", "PSD"])
        self.cmb_kind.setToolTip("Magnitude: amplitude of tones. PSD: power per Hz, the right view for noise.")
        self.cmb_kind.currentTextChanged.connect(self._on_kind_changed)
        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems(SCALES)
        self.cmb_scale.currentTextChanged.connect(lambda text: self._apply_config(scale=text))
        self.spn_fps = QSpinBox()
        self.spn_fps.setRange(10, 240)
        self.spn_fps.setValue(60)
        self.spn_fps.setSuffix(" fps")
        self.spn_fps.valueChanged.connect(lambda _: self._apply_frame_rate())
        for row, (text, widget) in enumerate((
            ("FFT size", self.cmb_fft), ("Window", self.cmb_window), ("Spectrum", self.cmb_kind),
            ("Scale", self.cmb_scale), ("Frame rate", self.spn_fps),
        )):
            grid.addWidget(QLabel(text), row, 0)
            grid.addWidget(widget, row, 1)
        layout.addWidget(box)

        # --- filter
        box = QGroupBox("Filter")
        grid = self._grid(box)
        self.cmb_filter = QComboBox()
        self.cmb_filter.addItems(FILTER_TYPES)
        self.cmb_filter.setToolTip("Butterworth IIR filter applied before the analysis")
        self.cmb_filter.currentTextChanged.connect(self._on_filter_type_changed)
        grid.addWidget(QLabel("Type"), 0, 0)
        grid.addWidget(self.cmb_filter, 0, 1)

        self.lbl_cut1 = QLabel("Cutoff")
        self.spn_cut1, self.sld_cut1 = self._make_frequency_control(self.processor.config.cutoff_hz)
        self.spn_cut1.valueChanged.connect(lambda v: self._on_cutoff_spin(self.sld_cut1, "cutoff_hz", v))
        self.sld_cut1.valueChanged.connect(lambda v: self._on_cutoff_slider(self.spn_cut1, v))
        grid.addWidget(self.lbl_cut1, 1, 0)
        grid.addWidget(self.spn_cut1, 1, 1)
        grid.addWidget(self.sld_cut1, 2, 0, 1, 2)

        self.lbl_cut2 = QLabel("High edge")
        self.spn_cut2, self.sld_cut2 = self._make_frequency_control(self.processor.config.cutoff_high_hz)
        self.spn_cut2.valueChanged.connect(lambda v: self._on_cutoff_spin(self.sld_cut2, "cutoff_high_hz", v))
        self.sld_cut2.valueChanged.connect(lambda v: self._on_cutoff_slider(self.spn_cut2, v))
        grid.addWidget(self.lbl_cut2, 3, 0)
        grid.addWidget(self.spn_cut2, 3, 1)
        grid.addWidget(self.sld_cut2, 4, 0, 1, 2)

        self.lbl_order = QLabel("Order")
        self.spn_order = QSpinBox()
        self.spn_order.setRange(1, 12)
        self.spn_order.setValue(self.processor.config.filter_order)
        self.spn_order.valueChanged.connect(lambda v: self._apply_config(filter_order=int(v)))
        self.lbl_q = QLabel("Notch Q")
        self.spn_q = QDoubleSpinBox()
        self.spn_q.setRange(0.5, 200.0)
        self.spn_q.setDecimals(1)
        self.spn_q.setValue(self.processor.config.notch_q)
        self.spn_q.setToolTip("Stop-band width = centre frequency / Q")
        self.spn_q.valueChanged.connect(lambda v: self._apply_config(notch_q=float(v)))
        pair = QHBoxLayout()
        pair.setSpacing(8)
        pair.addWidget(self.lbl_order)
        pair.addWidget(self.spn_order, 1)
        pair.addSpacing(6)
        pair.addWidget(self.lbl_q)
        pair.addWidget(self.spn_q, 1)
        grid.addLayout(pair, 5, 0, 1, 2)

        self.chk_zero_phase = QCheckBox("Zero-phase (no delay)")
        self.chk_zero_phase.setToolTip(
            "Off: causal filter (sosfilt) running continuously over the stream.\n"
            "On: forward-backward filter (sosfiltfilt) on each displayed frame:\n"
            "no phase shift, twice the attenuation in dB."
        )
        self.chk_zero_phase.toggled.connect(lambda on: self._apply_config(zero_phase=bool(on)))
        grid.addWidget(self.chk_zero_phase, 6, 0, 1, 2)
        layout.addWidget(box)

        # --- generator
        self.box_generator = QGroupBox("Live generator")
        grid = QGridLayout(self.box_generator)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(7)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 2)
        for column, text in ((1, "Frequency"), (2, "Amplitude")):
            head = QLabel(text)
            head.setObjectName("colHead")
            grid.addWidget(head, 0, column)
        nyquist = 0.5 * GENERATOR_SAMPLE_RATE
        self.spn_tone_freq: list[QDoubleSpinBox] = []
        self.spn_tone_amp: list[QDoubleSpinBox] = []
        for index, (freq, amp) in enumerate(self.generator.tones, start=1):
            f_box = QDoubleSpinBox()
            f_box.setRange(1.0, nyquist - 1.0)
            f_box.setDecimals(1)
            f_box.setSingleStep(10.0)
            f_box.setSuffix(" Hz")
            f_box.setValue(freq)
            f_box.setKeyboardTracking(False)
            a_box = QDoubleSpinBox()
            a_box.setRange(0.0, 1.0)
            a_box.setDecimals(2)
            a_box.setSingleStep(0.05)
            a_box.setValue(amp)
            f_box.valueChanged.connect(lambda _: self._on_generator_changed())
            a_box.valueChanged.connect(lambda _: self._on_generator_changed())
            grid.addWidget(QLabel(f"Tone {index}"), index, 0)
            grid.addWidget(f_box, index, 1)
            grid.addWidget(a_box, index, 2)
            self.spn_tone_freq.append(f_box)
            self.spn_tone_amp.append(a_box)
        self.spn_noise = QDoubleSpinBox()
        self.spn_noise.setRange(0.0, 1.0)
        self.spn_noise.setDecimals(3)
        self.spn_noise.setSingleStep(0.005)
        self.spn_noise.setValue(self.generator.noise_std)
        self.spn_noise.setToolTip("RMS level of the added Gaussian noise")
        self.spn_noise.valueChanged.connect(lambda _: self._on_generator_changed())
        row = len(self.spn_tone_freq) + 1
        grid.addWidget(QLabel("Noise"), row, 0)
        grid.addWidget(self.spn_noise, row, 2)
        layout.addWidget(self.box_generator)

        layout.addStretch(1)
        for widget in panel.findChildren((QComboBox, QSpinBox, QDoubleSpinBox)):
            widget.setMinimumWidth(64)                   # let the grid, not the content, set widths
        return panel

    def _make_frequency_control(self, value: float) -> tuple[QDoubleSpinBox, QSlider]:
        spin = QDoubleSpinBox()
        spin.setDecimals(1)
        spin.setSuffix(" Hz")
        spin.setRange(1.0, self.processor.nyquist - 1.0)
        spin.setSingleStep(10.0)
        spin.setValue(value)
        spin.setKeyboardTracking(False)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 1000)
        return spin, slider

    def _make_pane(self, title: str, plot: pg.PlotWidget, extra: QWidget | None = None) -> tuple[QFrame, QLabel]:
        """A titled frame around a plot; returns it with its right-hand info label."""
        pane = QFrame()
        pane.setObjectName("pane")
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(12, 9, 10, 8)
        layout.setSpacing(4)
        header = QHBoxLayout()
        name = QLabel(title)
        name.setObjectName("paneTitle")
        info = QLabel("")
        info.setObjectName("paneInfo")
        info.setFont(_tabular_font(12, QFont.Weight.Normal))
        info.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(name)
        if extra is not None:
            header.addSpacing(8)
            header.addWidget(extra)
        header.addWidget(info, 1)
        layout.addLayout(header)
        layout.addWidget(plot, 1)
        return pane, info

    def _style_plot(self, plot: pg.PlotWidget) -> None:
        plot.setBackground(COLOR_PANEL)
        plot.showGrid(x=True, y=True, alpha=0.16)
        plot.setMenuEnabled(False)
        plot.disableAutoRange()
        plot.getPlotItem().hideButtons()
        for name in ("left", "bottom"):
            axis = plot.getPlotItem().getAxis(name)
            axis.setPen(pg.mkPen(COLOR_BORDER))
            axis.setTextPen(pg.mkPen(COLOR_MUTED))
        plot.getPlotItem().getAxis("left").setWidth(58)        # both plots line up exactly

    def _build_plots(self) -> QWidget:
        pg.setConfigOptions(antialias=False, background=COLOR_PANEL, foreground=COLOR_MUTED)

        self.time_plot = TimedPlotWidget()
        self._style_plot(self.time_plot)
        self._set_plain_label(self.time_plot, "Amplitude")
        self.time_plot.setLabel("bottom", "Time", units="s")
        self.time_curve = self.time_plot.plot(pen=pg.mkPen(COLOR_ACCENT, width=1))
        self._tune_curve(self.time_curve)
        time_pane, self.lbl_time_info = self._make_pane("Waveform", self.time_plot, self.lbl_source)

        self.spec_plot = TimedPlotWidget()
        self._style_plot(self.spec_plot)
        self.spec_plot.setLabel("bottom", "Frequency", units="Hz")

        # Filter response, drawn behind the spectrum (dB scales only).
        self.filter_curve = pg.PlotCurveItem(pen=pg.mkPen(COLOR_FAINT, width=1))
        self.filter_curve.setZValue(1)
        self.spec_plot.addItem(self.filter_curve, ignoreBounds=True)

        self.spec_curve = self.spec_plot.plot(pen=pg.mkPen(COLOR_SPECTRUM, width=1))
        self.spec_curve.setZValue(5)
        self._tune_curve(self.spec_curve)

        # Cutoff handles: drag them in the plot instead of using the sidebar.
        handle_pen = pg.mkPen(COLOR_ACCENT, width=1)
        hover_pen = pg.mkPen(COLOR_TEXT, width=2)
        self.cut_line1 = pg.InfiniteLine(angle=90, movable=True, pen=handle_pen, hoverPen=hover_pen)
        self.cut_line2 = pg.InfiniteLine(angle=90, movable=True, pen=handle_pen, hoverPen=hover_pen)
        self.cut_line1.sigPositionChanged.connect(lambda line: self._on_cut_line_moved(self.spn_cut1, line))
        self.cut_line2.sigPositionChanged.connect(lambda line: self._on_cut_line_moved(self.spn_cut2, line))
        for line in (self.cut_line1, self.cut_line2):
            line.setZValue(8)
            line.setVisible(False)
            self.spec_plot.addItem(line, ignoreBounds=True)

        # Peak marker: a small arrow whose tip sits on the peak. Moving a cached
        # item is far cheaper per frame than re-building a scatter plot.
        cached = pg.QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache
        self.peak_marker = pg.ArrowItem(
            angle=-90, headLen=11, tipAngle=45, tailLen=0, pen=pg.mkPen(COLOR_PEAK), brush=pg.mkBrush(COLOR_PEAK)
        )
        self.peak_marker.setZValue(10)
        self.peak_marker.setCacheMode(cached)
        self.spec_plot.addItem(self.peak_marker, ignoreBounds=True)
        self.peak_label = pg.TextItem("", color=COLOR_PEAK, anchor=(0.0, 1.2))
        self.peak_label.setZValue(11)
        self.peak_label.textItem.setCacheMode(cached)
        self.spec_plot.addItem(self.peak_label, ignoreBounds=True)

        cross_pen = pg.mkPen(COLOR_FAINT, width=1)
        self.cross_v = pg.InfiniteLine(angle=90, movable=False, pen=cross_pen)
        self.cross_h = pg.InfiniteLine(angle=0, movable=False, pen=cross_pen)
        for line in (self.cross_v, self.cross_h):
            line.setZValue(2)
            line.setVisible(False)
            self.spec_plot.addItem(line, ignoreBounds=True)
        self._cache_axes(self.time_plot)
        self._cache_axes(self.spec_plot)
        self._mouse_proxy = pg.SignalProxy(self.spec_plot.scene().sigMouseMoved, rateLimit=60, slot=self._on_mouse_moved)
        spec_pane, self.lbl_readout = self._make_pane("Spectrum", self.spec_plot)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(time_pane)
        splitter.addWidget(spec_pane)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([260, 400])
        return splitter

    @staticmethod
    def _set_plain_label(plot: pg.PlotWidget, text: str, units: str = "") -> None:
        """Label the left axis and switch its automatic SI prefix off for good.

        dB values must never be shown as "k" or "m" multiples. An empty range
        list is used because ``enableAutoSIPrefix(False)`` alone still leaves
        behind the scale factor of whatever range was displayed before.
        """
        axis = plot.getPlotItem().getAxis("left")
        axis.setLabel(text, units=units or None, siPrefixEnableRanges=())
        axis.updateAutoSIPrefix()

    @staticmethod
    def _cache_axes(plot: pg.PlotWidget) -> None:
        """Let Qt keep the axes, tick labels and grid as pixmaps.

        They only change when a range changes, but would otherwise be redrawn
        (text included) on every frame because the curve next to them changed.
        """
        for name in ("left", "bottom"):
            plot.getPlotItem().getAxis(name).setCacheMode(pg.QtWidgets.QGraphicsItem.CacheMode.DeviceCoordinateCache)

    @staticmethod
    def _tune_curve(curve: pg.PlotDataItem) -> None:
        # Min/max ("peak") decimation to about 2 points per screen pixel: no visible
        # difference, but the painter gets a few thousand vertices at most.
        curve.opts["autoDownsampleFactor"] = 1.0
        curve.setDownsampling(auto=True, method="peak")
        curve.setClipToView(True)
        curve.setSkipFiniteCheck(True)

    def _build_metrics_panel(self) -> QWidget:
        strip = ReadoutStrip()
        strip.setObjectName("strip")
        row = QHBoxLayout(strip)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.card_freq = Readout("Dominant tone")
        self.card_peak = Readout("Peak magnitude")
        self.card_rms = Readout("RMS amplitude")
        self.card_snr = Readout("Estimated SNR")
        self.card_thd = Readout("THD")
        self.card_snr.setToolTip("Power of the dominant tone over the broadband noise power.\nHarmonics and other steady tones are not counted as noise.")
        self.card_thd.setToolTip("Harmonics 2 to 10 of the dominant tone, relative to the fundamental.")
        self.latency_meter = LatencyMeter(LATENCY_BUDGET_MS)
        self._optional_readouts: list[QWidget] = []       # hidden while the sky map takes the space
        for readout in (self.card_freq, self.card_peak, self.card_rms, self.card_snr, self.card_thd):
            row.addWidget(readout, 3)
            divider = QFrame()
            divider.setObjectName("divider")
            row.addWidget(divider)
            if readout in (self.card_rms, self.card_snr, self.card_thd):
                self._optional_readouts += [readout, divider]
        strip.optional = self._optional_readouts
        row.addWidget(self.latency_meter, 4)
        return strip

    def _build_statusbar(self) -> None:
        bar = self.statusBar()
        bar.setSizeGripEnabled(False)
        self.lbl_status_source = QLabel("")
        bar.addWidget(self.lbl_status_source, 1)
        bar.addPermanentWidget(self.cmb_channel)
        self.lbl_status_latency = QLabel("DSP - ms | Render - ms")
        self.lbl_status_latency.setFont(_tabular_font(12, QFont.Weight.Normal))
        self.lbl_status_latency.setToolTip(
            "DSP: filtering + FFT + metrics for one frame.\n"
            "Render: plot data upload + the actual paint events of both plots.\n"
            f"Budget: {LATENCY_BUDGET_MS:.0f} ms per frame for DSP + Render."
        )
        bar.addPermanentWidget(self.lbl_status_latency)

    # ---------------------------------------------------------- drag & drop
    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(url.isLocalFile() for url in urls):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        for url in event.mimeData().urls():
            if url.isLocalFile():
                event.acceptProposedAction()
                self.load_file(url.toLocalFile())
                return

    # ------------------------------------------------------------- sources
    def _set_source(self, source, sky_object=None, kit_entry: dict | None = None) -> None:
        """Switch the stream to a new source and rebuild everything that depends on it."""
        live = isinstance(source, SyntheticGenerator)
        mic = isinstance(source, MicrophoneInput)
        self._sky_object = sky_object
        self._kit_entry = kit_entry
        if sky_object is None and kit_entry is None:       # an ordinary source: nothing to play back
            self._view_fmax = 6000.0 if mic else None
            self._loop_audio = None
            self.audio.stop()
        if sky_object is None or sky_object.kind == "planet" or sky_object.key == "sun":
            self.sky_panel.clear_selection()
        if sky_object is None or not (sky_object.kind == "planet" or sky_object.key == "sun"):
            self.solar_panel.clear_selection()
        if kit_entry is None:
            self.kit_panel.clear_selection()
        if not mic and self.act_mic.isChecked():           # leaving the microphone
            with QSignalBlocker(self.act_mic):
                self.act_mic.setChecked(False)
            self.microphone.stop()
        if not live and not mic:
            self.signal = source                           # kept while live, to switch back to it
        self.stream = StreamSimulator(source, frame_rate=float(self.spn_fps.value()))
        if live:
            self.generator.reset()
        self.processor.set_sample_rate(self.stream.sample_rate)
        with QSignalBlocker(self.act_live):
            self.act_live.setChecked(live)
        self._sync_controls_to_source()
        self._last_tick = time.perf_counter()
        if sky_object is not None:                         # slow pulsars would take seconds to fill the frame
            self.processor.push(self.stream.read(self.processor.config.fft_size))
        if not self.playing:                               # show something even while paused
            if sky_object is None:
                self.processor.push(self.stream.read(self.processor.config.fft_size))
            self._refresh()

    def _sync_controls_to_source(self) -> None:
        """Update ranges, labels and axes after a sample-rate or source change."""
        fs = self.processor.sample_rate
        nyquist = self.processor.nyquist
        top = max(nyquist - min(1.0, 0.01 * nyquist), 0.002)
        bottom = min(1.0, 0.001 * nyquist)
        cfg = self.processor.config
        for spin, slider, key in ((self.spn_cut1, self.sld_cut1, "cutoff_hz"), (self.spn_cut2, self.sld_cut2, "cutoff_high_hz")):
            value = min(max(getattr(cfg, key), bottom), top)
            with QSignalBlocker(spin), QSignalBlocker(slider):
                spin.setRange(bottom, top)
                spin.setDecimals(1 if nyquist > 200 else 4)
                spin.setSingleStep(max(nyquist / 500.0, 0.0001))
                spin.setValue(value)
                slider.setRange(0, 1000)
                slider.setValue(self._freq_to_slider(value))
            if value != getattr(cfg, key):
                self.processor.update_config(**{key: value})
        for line in (self.cut_line1, self.cut_line2):
            line.setBounds((bottom, top))

        mic = isinstance(self.stream.source, MicrophoneInput)
        live = self.stream.is_live and not mic
        self.box_generator.setEnabled(live)
        multi = not live and not mic and self.signal is not None and self.signal.num_channels > 1
        with QSignalBlocker(self.cmb_channel):
            self.cmb_channel.clear()
            if multi:
                self.cmb_channel.addItem("Mono mix (all channels)", None)
                for ch in range(self.signal.num_channels):
                    self.cmb_channel.addItem(f"Channel {ch + 1}", ch)
                current = self.signal.channel
                self.cmb_channel.setCurrentIndex(0 if current is None else current + 1)
            else:
                self.cmb_channel.addItem("Mono", None)
        self.cmb_channel.setEnabled(multi)
        self.cmb_channel.setVisible(multi)

        rate = f"{fs / 1000.0:g} kHz" if fs >= 1000 else f"{fs:g} Hz"
        if live:
            chip = f"Live generator, {rate}"
            status = f"Source: live generator | Fs {fs:.0f} Hz"
        elif mic:
            chip = f"Microfon, {rate}"
            status = f"Source: microphone, live | Fs {fs:.0f} Hz"
        elif self._kit_entry is not None:
            chip = self._kit_entry["title"]
            status = f"Source: {self._kit_entry['title']} | synthesised sound, not a recording | Fs {fs:g} Hz"
        elif self._sky_object is not None:
            chip = self._sky_object.nickname
            status = f"Source: {self.signal.name} | synthesised from published values, not a recording | Fs {fs:g} Hz"
        else:
            sig = self.signal
            chip = f"{sig.name}, {rate}, {sig.duration:.2f} s"
            status = f"Source: {sig.name} | Fs {fs:g} Hz | {sig.num_channels} ch | {sig.duration:.2f} s, looping"
        self.lbl_source.setText(chip)
        self.lbl_source.setToolTip("" if live or mic or not self.signal.path else self.signal.path)
        self.lbl_status_source.setText(status)
        self._rebuild_axes()
        self._update_filter_overlay()

    def _rebuild_axes(self) -> None:
        cfg = self.processor.config
        fs = self.processor.sample_rate
        n = cfg.fft_size
        self._time_x = np.arange(n, dtype=np.float64) / fs
        # Limits first: a range outside the previous source's limits would be clamped.
        self.spec_plot.getPlotItem().getViewBox().setLimits(xMin=0.0, xMax=self.processor.nyquist)
        self.time_plot.getPlotItem().getViewBox().setLimits(xMin=0.0, xMax=n / fs)
        self.time_plot.setXRange(0.0, n / fs, padding=0.0)
        self.spec_plot.setXRange(0.0, min(self._view_fmax or self.processor.nyquist, self.processor.nyquist), padding=0.0)
        self._time_top = 0.0
        self._spec_top = None
        span = n / fs
        window = f"{span * 1e3:.1f} ms" if span < 1.0 else f"{span:.2f} s"
        self.lbl_time_info.setText(f"{window} window, {fs / n:.3g} Hz per FFT bin")
        self._update_spectrum_labels()

    def _update_spectrum_labels(self) -> None:
        db = self.processor.config.scale == "dB"
        if self.spectrum_kind == "PSD":
            label, units = ("PSD", "dBFS/Hz") if db else ("PSD (FS²/Hz)", "")
        else:
            label, units = ("Magnitude", "dBFS") if db else ("Magnitude (1.0 = full scale)", "")
        self._set_plain_label(self.spec_plot, label, units)
        self._spec_top = None
        self._restart_latency_stats()

    def _restart_latency_stats(self) -> None:
        """Latency statistics describe steady streaming: drop the one-off cost of a
        layout change (new axes, labels, ranges) from the rolling window."""
        self._dsp_hist.clear()
        self._render_hist.clear()
        self._frames_seen = min(self._frames_seen, WARMUP_FRAMES - 5)

    # ------------------------------------------------------------- actions
    def _on_load_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load WAV file", "", "WAV audio (*.wav *.wave);;All files (*)")
        if path:
            self.load_file(path)

    def _on_load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CSV file", "", "CSV / text (*.csv *.txt *.tsv *.dat);;All files (*)"
        )
        if path:
            self.load_file(path)

    def load_file(self, path: str, sample_rate: float | None = None) -> bool:
        """Load a WAV or CSV file and start streaming it. Returns ``True`` on success."""
        try:
            if os.path.splitext(path)[1].lower() in (".wav", ".wave"):
                signal = load_wav(path)
            else:
                try:
                    signal = load_csv(path, sample_rate=sample_rate)
                except SampleRateRequiredError:
                    fs, ok = QInputDialog.getDouble(
                        self,
                        "Sample rate",
                        f"'{os.path.basename(path)}' has no time column.\nSample rate in Hz:",
                        44100.0, 0.001, 1e9, 3,
                    )
                    if not ok:
                        return False
                    signal = load_csv(path, sample_rate=fs)
        except Exception as exc:
            QMessageBox.warning(self, "Could not load file", str(exc))
            return False
        self._set_source(signal)
        if signal.notes:
            self.statusBar().showMessage(" ".join(signal.notes), 8000)
        return True

    def _on_live_toggled(self, checked: bool) -> None:
        if checked:
            self._set_source(self.generator)
        elif self.signal is not None:                      # back to the last loaded file
            self._set_source(self.signal)
        else:                                              # nothing else to switch to
            with QSignalBlocker(self.act_live):
                self.act_live.setChecked(True)
            self.statusBar().showMessage("Load a WAV or CSV file to switch away from the live generator.", 5000)

    # ---------------------------------------------------- sky, planets, sounds
    def _apply_audio(self) -> None:
        """Start or stop the loudspeakers according to the source, the Sunet button and Play / Pause."""
        if self._loop_audio is not None and self.act_sound.isChecked() and self.playing:
            self.audio.play(self._loop_audio)
        else:
            self.audio.stop()

    def _tune(self, fft_size: int) -> None:
        for combo, text in ((self.cmb_filter, "Bypass"), (self.cmb_window, "Hann"), (self.cmb_kind, "Magnitude")):
            combo.setCurrentText(text)
        with QSignalBlocker(self.cmb_fft):
            self.cmb_fft.setCurrentIndex(FFT_SIZES.index(fft_size))
        self.processor.update_config(fft_size=fft_size)

    def _on_sky_object(self, obj) -> None:
        """Tune the analyser to a sky object or planet and start its sound."""
        signal = sky_catalog.analysis_signal(obj)
        factor = {"pulsar": 12.0, "planet": 12.0, "variable": 6.0, "star": 2.0}.get(obj.kind, 2.5)
        self._view_fmax = factor * obj.display_frequency_hz
        self._tune(obj.fft_size)
        self._loop_audio = sky_catalog.audio_signal(obj)
        self._set_source(signal, sky_object=obj)
        self._apply_audio()

    def _on_kit_sound(self, entry: dict) -> None:
        """Play a sound from the kit and show it in the analyser."""
        try:
            signal = load_wav(entry["path"])
        except Exception as exc:
            QMessageBox.warning(self, "Sunetul nu a putut fi încărcat", str(exc))
            return
        self._view_fmax = float(entry.get("view_hz", 5000.0))
        self._tune(4096)
        self._loop_audio = signal.samples if signal.sample_rate == 44100.0 else None
        self._set_source(signal, kit_entry=entry)
        self._apply_audio()

    def _on_mic_toggled(self, checked: bool) -> None:
        if not checked:
            self.microphone.stop()
            if isinstance(self.stream.source, MicrophoneInput):
                self._set_source(self.generator)
            return
        if not self.microphone.start():
            with QSignalBlocker(self.act_mic):
                self.act_mic.setChecked(False)
            QMessageBox.information(self, "Microfon", f"Microfonul nu poate fi folosit.\n{self.microphone.error}")
            return
        self._tune(4096)
        self._set_source(self.microphone)

    def _on_play_pause(self) -> None:
        self.playing = not self.playing
        self._apply_audio()
        self.act_play.setText("Pause" if self.playing else "Play")
        self.act_play.setIcon(self._icon_pause if self.playing else self._icon_play)
        self._last_tick = time.perf_counter()

    def _on_channel_changed(self, _index: int) -> None:
        if self.stream.is_live or self.signal is None or self.signal.num_channels < 2:
            return
        self._set_source(self.signal.select_channel(self.cmb_channel.currentData()))

    def _on_generator_changed(self) -> None:
        tones = [(f.value(), a.value()) for f, a in zip(self.spn_tone_freq, self.spn_tone_amp)]
        self.generator.set_tones(tones)
        self.generator.set_noise(self.spn_noise.value())

    def _on_kind_changed(self, kind: str) -> None:
        self.spectrum_kind = kind
        self._update_spectrum_labels()
        self._update_filter_overlay()
        if not self.playing:
            self._refresh()

    def _on_filter_type_changed(self, filter_type: str) -> None:
        self._update_filter_controls()
        self._apply_config(filter_type=filter_type)

    def _update_filter_controls(self) -> None:
        kind = self.cmb_filter.currentText()
        active = kind != "Bypass"
        self.lbl_cut1.setText({"Band-pass": "Low edge", "Notch": "Centre"}.get(kind, "Cutoff"))
        for widget in (self.lbl_cut1, self.spn_cut1, self.sld_cut1, self.lbl_order, self.spn_order, self.chk_zero_phase):
            widget.setEnabled(active)
        for widget in (self.lbl_cut2, self.spn_cut2, self.sld_cut2):
            widget.setEnabled(kind == "Band-pass")
        for widget in (self.lbl_q, self.spn_q):
            widget.setEnabled(kind == "Notch")

    def _update_filter_overlay(self) -> None:
        """Show the cutoff handles and the filter response that match the current filter."""
        cfg = self.processor.config
        active = cfg.filter_type != "Bypass"
        self._moving_cut_line = True                       # programmatic moves, not drags
        self.cut_line1.setValue(cfg.cutoff_hz)
        self.cut_line2.setValue(cfg.cutoff_high_hz)
        self._moving_cut_line = False
        self.cut_line1.setVisible(active)
        self.cut_line2.setVisible(cfg.filter_type == "Band-pass")

        show_curve = active and cfg.scale == "dB"
        self.filter_curve.setVisible(show_curve)
        if show_curve:
            freqs, gain = self.processor.frequency_response(FILTER_CURVE_POINTS)
            floor = -180.0 if self.spectrum_kind == "PSD" else -140.0
            self.filter_curve.setData(freqs, np.maximum(20.0 * np.log10(np.maximum(gain, 1e-12)), floor))

    def _on_cut_line_moved(self, spin: QDoubleSpinBox, line: pg.InfiniteLine) -> None:
        if not self._moving_cut_line:
            spin.setValue(round(float(line.value()), spin.decimals()))     # runs the normal cutoff path

    # Sliders work on a 0..1000 logarithmic scale so low frequencies stay adjustable.
    def _freq_to_slider(self, hz: float) -> int:
        lo, hi = self.spn_cut1.minimum(), self.spn_cut1.maximum()
        if hi <= lo:
            return 0
        hz = min(max(hz, lo), hi)
        return int(round(1000.0 * math.log(hz / lo) / math.log(hi / lo)))

    def _slider_to_freq(self, position: int) -> float:
        lo, hi = self.spn_cut1.minimum(), self.spn_cut1.maximum()
        return lo * (hi / lo) ** (position / 1000.0)

    def _on_cutoff_spin(self, slider: QSlider, key: str, value: float) -> None:
        with QSignalBlocker(slider):
            slider.setValue(self._freq_to_slider(value))
        self._apply_config(**{key: float(value)})

    def _on_cutoff_slider(self, spin: QDoubleSpinBox, position: int) -> None:
        spin.setValue(round(self._slider_to_freq(position), spin.decimals()))   # triggers _on_cutoff_spin

    def _apply_config(self, **changes) -> None:
        try:
            self.processor.update_config(**changes)
        except ValueError as exc:
            self.statusBar().showMessage(f"Invalid setting: {exc}", 6000)
            return
        if "fft_size" in changes:
            self._rebuild_axes()
        if "scale" in changes:
            self._update_spectrum_labels()
        self._update_filter_overlay()
        self._restart_latency_stats()                      # a settings change is not steady state
        if not self.playing:
            self._refresh()

    def _apply_frame_rate(self) -> None:
        fps = int(self.spn_fps.value())
        self.stream.set_frame_rate(fps)
        self._timer.setInterval(max(1, int(round(1000.0 / fps))))

    def _on_export(self) -> None:
        if self._result is None:
            QMessageBox.information(self, "Export metrics", "There is nothing to export yet.")
            return
        payload = self.metrics_snapshot()
        default = f"metrics_{_dt.datetime.now():%Y%m%d_%H%M%S}.json"
        path, selected = QFileDialog.getSaveFileName(
            self, "Export metrics", default, "JSON (*.json);;CSV (*.csv)"
        )
        if not path:
            return
        if not path.lower().endswith((".json", ".csv")):
            path += ".csv" if "csv" in selected.lower() else ".json"
        try:
            export_metrics(path, payload)
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"Metrics exported to {path}", 6000)

    def metrics_snapshot(self) -> dict:
        """Everything "Export Metrics" writes: source, settings, metrics and latency."""
        result = self._result
        dsp, render = self._latency_stats()
        return {
            "application": {"name": APP_NAME, "version": __version__},
            "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": {
                "name": self.stream.name,
                "type": "live generator" if self.stream.is_live else "file",
                "path": None if self.stream.is_live or self.signal is None else self.signal.path,
                "sample_rate_hz": self.processor.sample_rate,
            },
            "settings": {**asdict(self.processor.config), "spectrum": self.spectrum_kind},
            "metrics": {} if result is None else result.metrics.to_dict(),
            "latency": {
                "dsp_ms_avg": dsp[0], "dsp_ms_max": dsp[1],
                "render_ms_avg": render[0], "render_ms_max": render[1],
                "total_ms_avg": dsp[0] + render[0],
                "budget_ms": LATENCY_BUDGET_MS,
                "frames_averaged": len(self._dsp_hist),
            },
        }

    # --------------------------------------------------------- frame update
    def _on_tick(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_tick
        self._last_tick = now
        if not self.playing:
            return
        self._tick_times.append(now)

        # Paint events of the previous frame have run by now: collect their real duration.
        paint_ms = self.time_plot.take_paint_ms() + self.spec_plot.take_paint_ms()
        if self._dsp_hist:
            self._render_hist.append(self._update_ms_prev + paint_ms)

        chunk = self.stream.chunk_for_elapsed(elapsed)
        t0 = time.perf_counter()
        self.processor.push(chunk)
        result = self.processor.analyze()
        t1 = time.perf_counter()
        self._draw(result)
        t2 = time.perf_counter()

        self._dsp_hist.append((t1 - t0) * 1e3)
        self._update_ms_prev = (t2 - t1) * 1e3
        self._frames_seen += 1
        if self._frames_seen == WARMUP_FRAMES:             # forget start-up costs
            self._dsp_hist.clear()
            self._render_hist.clear()
            self._benchmark_log.clear()

        if now - self._last_text_update >= TEXT_REFRESH_S:
            self._last_text_update = now
            self._update_texts(result)

        if self._benchmark_frames:
            if self._render_hist:
                self._benchmark_log.append((self._dsp_hist[-1], self._render_hist[-1]))
            if len(self._benchmark_log) >= self._benchmark_frames:
                self._finish_benchmark()

    def _refresh(self) -> None:
        """Re-analyse and redraw the current buffer (used while paused)."""
        result = self.processor.analyze()
        self._draw(result)
        self._update_texts(result)

    def _draw(self, result: FrameResult) -> None:
        self._result = result
        frame = result.time_signal
        y = result.display_psd() if self.spectrum_kind == "PSD" else result.display_magnitude()
        self._spec_y = y
        self.time_curve.setData(self._time_x, frame)
        self.spec_curve.setData(result.frequencies, y)

        # Y ranges are managed by hand (auto-range would re-scan the data every frame).
        peak = float(np.max(np.abs(frame))) if frame.size else 0.0
        if peak > self._time_top or peak < 0.5 * self._time_top or self._time_top == 0.0:
            self._time_top = max(1.15 * peak, 1e-9) if peak > 0 else 1.0
            self.time_plot.setYRange(-self._time_top, self._time_top, padding=0.0)

        y_max = float(y.max()) if y.size else 0.0
        if result.scale == "dB":
            floor = -180.0 if self.spectrum_kind == "PSD" else -140.0
            top = max(10.0 if self.spectrum_kind == "Magnitude" else 0.0, 10.0 * math.ceil((y_max + 3.0) / 10.0))
            if top != self._spec_top:
                self._spec_top = top
                self.spec_plot.setYRange(floor, top, padding=0.0)
        elif self._spec_top is None or y_max > self._spec_top or y_max < 0.3 * self._spec_top:
            self._spec_top = max(1.2 * y_max, 1e-30) if y_max > 0 else 1.0
            self.spec_plot.setYRange(0.0, self._spec_top, padding=0.0)

        self.peak_marker.setPos(result.metrics.dominant_frequency_hz, self._peak_display_value(result))

    def _peak_display_value(self, result: FrameResult) -> float:
        """Y position of the peak marker in the units currently plotted."""
        m = result.metrics
        if self.spectrum_kind == "Magnitude":
            return m.peak_magnitude_dbfs if result.scale == "dB" else m.peak_magnitude
        df = result.frequencies[1] - result.frequencies[0] if result.frequencies.size > 1 else 1.0
        index = int(round(m.dominant_frequency_hz / df)) if df > 0 else 0
        return float(self._spec_y[min(max(index, 0), self._spec_y.size - 1)])

    def _value_text(self, value: float, scale: str) -> str:
        if self.spectrum_kind == "PSD":
            return f"{value:.1f} dBFS/Hz" if scale == "dB" else f"{value:.3e} FS²/Hz"
        return f"{value:.2f} dBFS" if scale == "dB" else f"{value:.5f}"

    def _latency_stats(self) -> tuple[tuple[float, float], tuple[float, float]]:
        def stats(history) -> tuple[float, float]:
            return (float(np.mean(history)), float(np.max(history))) if history else (0.0, 0.0)
        return stats(self._dsp_hist), stats(self._render_hist)

    def _update_texts(self, result: FrameResult) -> None:
        m = result.metrics
        self.card_freq.set_value(_format_frequency(m.dominant_frequency_hz), f"bin width {result.sample_rate / self.processor.config.fft_size:.3g} Hz")
        self.card_peak.set_value(f"{m.peak_magnitude_dbfs:.2f} dBFS", f"amplitude {m.peak_magnitude:.4f}")
        self.card_rms.set_value(f"{m.rms_dbfs:.2f} dBFS", f"amplitude {m.rms_amplitude:.4f}")
        self.card_snr.set_value(f"{m.snr_db:.1f} dB")
        self.card_thd.set_value(f"{m.thd_percent:.3f} %")

        peak_y = self._peak_display_value(result)
        self.peak_label.setText(f"{_format_frequency(m.dominant_frequency_hz)}  {self._value_text(peak_y, result.scale)}")
        self.peak_label.setPos(m.dominant_frequency_hz, peak_y)

        (dsp_avg, dsp_max), (render_avg, render_max) = self._latency_stats()
        total_avg = dsp_avg + render_avg
        total_max = dsp_max + render_max
        if total_max < LATENCY_BUDGET_MS:
            color, verdict = COLOR_OK, f"OK < {LATENCY_BUDGET_MS:.0f} ms"
        elif total_avg < LATENCY_BUDGET_MS:
            color, verdict = COLOR_WARN, f"spikes > {LATENCY_BUDGET_MS:.0f} ms"
        else:
            color, verdict = COLOR_BAD, f"over {LATENCY_BUDGET_MS:.0f} ms"
        fps = 0.0
        if len(self._tick_times) > 1:
            span = self._tick_times[-1] - self._tick_times[0]
            fps = (len(self._tick_times) - 1) / span if span > 0 else 0.0
        self.latency_meter.set_values(dsp_avg, render_avg, total_max, color, verdict)
        self.lbl_status_latency.setText(
            f"DSP {dsp_avg:.2f} ms | Render {render_avg:.2f} ms | Total {total_avg:.2f} ms "
            f"({verdict}) | {fps:.0f} fps"
        )
        if color != self._status_color:
            self._status_color = color
            self.lbl_status_latency.setStyleSheet(f"color: {color};")
        self._update_readout()

    # ------------------------------------------------------------ crosshair
    def _on_mouse_moved(self, event: tuple) -> None:
        position = event[0]
        plot_item = self.spec_plot.getPlotItem()
        inside = plot_item.sceneBoundingRect().contains(position)
        self.cross_v.setVisible(inside)
        self.cross_h.setVisible(inside)
        if not inside:
            self._cursor = None
            self._update_readout()
            return
        point = plot_item.getViewBox().mapSceneToView(position)
        self._cursor = (point.x(), point.y())
        self.cross_v.setPos(point.x())
        self.cross_h.setPos(point.y())
        self._update_readout()

    def _update_readout(self) -> None:
        result = self._result
        if self._cursor is None or result is None or self._spec_y is None:
            hint = "Hover for a read-out"
            if self.processor.config.filter_type != "Bypass":
                hint += ", drag the teal line to move the cutoff"
            self.lbl_readout.setText(hint)
            return
        x, y = self._cursor
        freqs = result.frequencies
        df = freqs[1] - freqs[0] if freqs.size > 1 else 1.0
        index = min(max(int(round(x / df)), 0), freqs.size - 1)
        self.lbl_readout.setText(
            f"Cursor {_format_frequency(x)}, {self._value_text(y, result.scale)}      "
            f"Spectrum there {self._value_text(float(self._spec_y[index]), result.scale)}"
        )

    # ------------------------------------------------------------ benchmark
    def _finish_benchmark(self) -> None:
        log = np.array(self._benchmark_log)
        self._benchmark_frames = 0
        total = log.sum(axis=1)
        report = {
            "frames": int(log.shape[0]),
            "warmup_frames_discarded": WARMUP_FRAMES,
            "fft_size": self.processor.config.fft_size,
            "dsp_ms_avg": float(log[:, 0].mean()), "dsp_ms_max": float(log[:, 0].max()),
            "render_ms_avg": float(log[:, 1].mean()), "render_ms_max": float(log[:, 1].max()),
            "total_ms_avg": float(total.mean()),
            "total_ms_p99": float(np.percentile(total, 99)),
            "total_ms_max": float(total.max()),
            "frames_over_budget": int(np.count_nonzero(total >= LATENCY_BUDGET_MS)),
            "budget_ms": LATENCY_BUDGET_MS,
        }
        self.benchmark_finished.emit(report)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._timer.stop()
        self.audio.stop()
        self.microphone.stop()
        if not self._benchmark_frames and QGuiApplication.platformName() != "offscreen":
            settings = QSettings("SignalProcessor", "SignalProcessor")
            settings.setValue("geometry", self.saveGeometry())
            settings.setValue("state", self.saveState(LAYOUT_VERSION))
        super().closeEvent(event)
