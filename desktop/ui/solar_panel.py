"""Solar-system panel: the planets on their orbits; click one to hear its "note"."""

from __future__ import annotations

import datetime as _dt
import math

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget

from io_handlers.sky_catalog import PLANET_ORBITS, PLANETS, SKY_BY_KEY, planet_longitude

from .sky_panel import KIND_COLORS, SKY_BG, SKY_GRID, SKY_TEXT, ObjectInfo

_J2000 = _dt.datetime(2000, 1, 1, 12, 0, 0)
_MAX_AU = 30.07


class SolarSystemWidget(QWidget):
    """Top view of the Solar System. Distances are compressed (logarithmic), speeds are in proportion."""

    object_clicked = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.days = (_dt.datetime.now() - _J2000).total_seconds() / 86400.0
        self.days_per_second = 20.0
        self._selected: str | None = None
        self._blink = 0
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._advance)
        self._timer.start()

    def date(self) -> _dt.date:
        return (_J2000 + _dt.timedelta(days=self.days)).date()

    def set_selected(self, key: str | None) -> None:
        self._selected = key
        self.update()

    def _advance(self) -> None:
        if not self.isVisible():
            return
        self.days += self.days_per_second * 0.05
        self._blink = (self._blink + 1) % 20
        self.update()

    def _radius(self, au: float) -> float:
        limit = 0.5 * min(self.width() - 50.0, self.height()) - 18.0     # room for the labels
        return 26.0 + (limit - 26.0) * math.log1p(au / 0.3) / math.log1p(_MAX_AU / 0.3)

    def _position(self, key: str) -> QPointF:
        au = PLANET_ORBITS[key][0]
        angle = math.radians(planet_longitude(key, self.days))
        r = self._radius(au)
        return QPointF(self.width() / 2.0 + r * math.cos(angle), self.height() / 2.0 - r * math.sin(angle))

    def _object_at(self, pos: QPointF) -> str | None:
        centre = QPointF(self.width() / 2.0, self.height() / 2.0)
        if math.hypot(pos.x() - centre.x(), pos.y() - centre.y()) < 16.0:
            return "sun"
        best, best_distance = None, 16.0
        for key in PLANET_ORBITS:
            p = self._position(key)
            distance = math.hypot(p.x() - pos.x(), p.y() - pos.y())
            if distance < best_distance:
                best, best_distance = key, distance
        return best

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        key = self._object_at(event.position())
        if key is not None:
            self.object_clicked.emit(key)

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(SKY_BG))
        centre = QPointF(self.width() / 2.0, self.height() / 2.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(SKY_GRID), 1.0))
        for key, (au, *_rest) in PLANET_ORBITS.items():
            painter.drawEllipse(centre, self._radius(au), self._radius(au))

        glow = QRadialGradient(centre, 24.0)
        glow.setColorAt(0.0, QColor("#fff3c4"))
        glow.setColorAt(0.45, QColor(KIND_COLORS["star"]))
        glow.setColorAt(1.0, QColor(242, 178, 74, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(centre, 24.0, 24.0)
        if self._selected == "sun":
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(KIND_COLORS["star"]), 2.0))
            painter.drawEllipse(centre, 18.0 + self._blink * 0.6, 18.0 + self._blink * 0.6)

        font = QFont(self.font())
        font.setPixelSize(11)
        painter.setFont(font)
        for obj in PLANETS:
            _, _, _, colour, size = PLANET_ORBITS[obj.key]
            p = self._position(obj.key)
            selected = obj.key == self._selected
            if obj.key == "saturn":
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(colour), 1.4))
                painter.drawEllipse(p, size * 1.9, size * 0.6)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(colour))
            painter.drawEllipse(p, size, size)
            if selected:
                grow = self._blink / 20.0
                ring = QColor(colour)
                ring.setAlphaF(1.0 - grow)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(ring, 2.0))
                painter.drawEllipse(p, size + 4.0 + 14.0 * grow, size + 4.0 + 14.0 * grow)
            painter.setPen(QColor("#ffffff") if selected else QColor(SKY_TEXT))
            painter.drawText(QRectF(p.x() - 40, p.y() + size + 2, 80, 14), Qt.AlignmentFlag.AlignCenter, obj.nickname)
        painter.end()


class SolarPanel(QWidget):
    object_selected = pyqtSignal(object)              # SkyObject

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(430)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        frame = QFrame()
        frame.setObjectName("pane")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(12, 9, 12, 10)
        frame_layout.setSpacing(6)
        header = QHBoxLayout()
        title = QLabel("Sistemul Solar")
        title.setObjectName("paneTitle")
        self.lbl_date = QLabel("")
        self.lbl_date.setObjectName("paneInfo")
        self.lbl_date.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(title)
        header.addWidget(self.lbl_date, 1)
        frame_layout.addLayout(header)
        self.view = SolarSystemWidget()
        self.view.object_clicked.connect(self.select)
        frame_layout.addWidget(self.view, 1)
        speed = QHBoxLayout()
        speed.setSpacing(10)
        label = QLabel("Viteza timpului")
        label.setObjectName("paneInfo")
        self.sld_speed = QSlider(Qt.Orientation.Horizontal)
        self.sld_speed.setRange(0, 100)
        self.sld_speed.setValue(35)
        self.sld_speed.setToolTip("Stânga: timpul stă pe loc. Dreapta: ani întregi pe secundă.")
        self.sld_speed.valueChanged.connect(self._on_speed)
        speed.addWidget(label)
        speed.addWidget(self.sld_speed, 1)
        frame_layout.addLayout(speed)
        layout.addWidget(frame, 6)

        lower = QFrame()
        lower.setObjectName("pane")
        lower_layout = QVBoxLayout(lower)
        lower_layout.setContentsMargins(12, 10, 12, 10)
        self.info = ObjectInfo(
            "Alege o planetă sau Soarele",
            "Planetele sunt desenate acolo unde se află astăzi (aproximativ). Distanțele sunt strânse ca să "
            "încapă toate; vitezele sunt în proporția reală.",
        )
        lower_layout.addWidget(self.info)
        layout.addWidget(lower, 3)

        self._clock = QTimer(self)
        self._clock.setInterval(250)
        self._clock.timeout.connect(self._show_date)
        self._clock.start()
        self._on_speed(self.sld_speed.value())

    def _on_speed(self, value: int) -> None:
        self.view.days_per_second = 0.0 if value == 0 else 0.5 * (2000.0 ** (value / 100.0))

    def _show_date(self) -> None:
        if self.isVisible():
            self.lbl_date.setText(f"Data: {self.view.date():%d.%m.%Y}")

    def clear_selection(self) -> None:
        self.view.set_selected(None)

    def select(self, key: str) -> None:
        obj = SKY_BY_KEY[key]
        self.view.set_selected(key)
        self.info.show_object(obj)
        self.object_selected.emit(obj)
