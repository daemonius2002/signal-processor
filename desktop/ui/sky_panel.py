"""Sky panel: a clickable map of the sky with pulsars and stars to listen to."""

from __future__ import annotations

import math

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

from io_handlers.sky_catalog import (
    BRIGHT_STARS,
    CONSTELLATION_LINES,
    LABELLED_STARS,
    SKY_BY_KEY,
    SKY_OBJECTS,
    SkyObject,
    frequency_text,
    galactic_plane,
)

SKY_BG = "#0a0f1a"
SKY_GRID = "#1b2636"
SKY_LINE = "#32465f"
SKY_STAR = "#e8eef7"
SKY_TEXT = "#8595a8"
SKY_MILKY = "#27374d"
KIND_COLORS = {"pulsar": "#4dd0c4", "star": "#f2b24a", "variable": "#b79cf2", "planet": "#7fb3ff", "fiction": "#ff6b8b"}
SKY_PANEL_WIDTH = 520
VIEWS = (("Tot cerul", "all"), ("Emisfera nordică", "north"), ("Emisfera sudică", "south"))
HEMISPHERE_SPAN = 115.0          # degrees from the pole shown in a hemisphere view


class ObjectInfo(QWidget):
    """Name, facts and a short story about the selected object (shared by the learning panels)."""

    def __init__(self, title: str, hint: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.lbl_name = QLabel(title)
        self.lbl_name.setObjectName("skyName")
        self.lbl_name.setWordWrap(True)
        self.lbl_facts = QLabel(hint)
        self.lbl_facts.setObjectName("skyFacts")
        self.lbl_facts.setWordWrap(True)
        self.lbl_story = QLabel("")
        self.lbl_story.setObjectName("skyStory")
        self.lbl_story.setWordWrap(True)
        self.lbl_story.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.lbl_name)
        layout.addWidget(self.lbl_facts)
        layout.addWidget(self.lbl_story, 1)

    def show_text(self, name: str, facts: str, story: str) -> None:
        self.lbl_name.setText(name)
        self.lbl_facts.setText(facts)
        self.lbl_story.setText(story)

    def show_object(self, obj: SkyObject) -> None:
        if obj.kind == "planet":
            where = "Sistemul Solar"
        elif obj.ra_hours is None:
            where = "pe cer se mișcă în fiecare zi"
        else:
            where = f"constelația {obj.constellation}"
        text = frequency_text(obj)
        self.show_text(obj.nickname, f"{obj.kind_ro}, {obj.name}, {where}.\n{text[0].upper()}{text[1:]}.", obj.story_ro)


class SkyMapWidget(QWidget):
    """Whole-sky map (right ascension 24h..0h left to right, declination -90..+90)."""

    object_clicked = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._selected: str | None = None
        self._hover: str | None = None
        self.show_all_labels = False                  # used when the map is exported as a picture
        self._view = "all"
        self._background: QPixmap | None = None
        self._blink = 0
        self._timer = QTimer(self)                    # the selected source pulses on the map
        self._timer.setInterval(125)
        self._timer.timeout.connect(self._on_blink)
        self._timer.start()

    # ---------------------------------------------------------------- geometry
    def set_view(self, view: str) -> None:
        """"all" (whole sky), "north" or "south" (a hemisphere, as on a planisphere)."""
        self._view = view
        self._background = None
        self.update()

    def _map_rect(self) -> QRectF:
        if self._view == "all":
            return QRectF(8.0, 6.0, self.width() - 16.0, self.height() - 24.0)
        side = min(self.width() - 16.0, self.height() - 12.0)
        return QRectF((self.width() - side) / 2.0, (self.height() - side) / 2.0, side, side)

    def _project(self, ra_hours: float, dec_deg: float) -> QPointF | None:
        """Sky position -> widget position; ``None`` if it is outside the current view."""
        rect = self._map_rect()
        if self._view == "all":
            return QPointF(rect.left() + rect.width() * (1.0 - ra_hours / 24.0),
                           rect.top() + rect.height() * (0.5 - dec_deg / 180.0))
        from_pole = 90.0 - dec_deg if self._view == "north" else 90.0 + dec_deg
        if from_pole > HEMISPHERE_SPAN:
            return None
        radius = 0.5 * rect.width() * from_pole / HEMISPHERE_SPAN
        angle = math.radians(ra_hours * 15.0)
        # Seen from inside the sky, right ascension runs anticlockwise around the north pole
        # and clockwise around the south pole.
        direction = -1.0 if self._view == "north" else 1.0
        return QPointF(rect.center().x() + radius * math.cos(angle),
                       rect.center().y() + direction * radius * math.sin(angle))

    def _object_at(self, pos: QPointF) -> str | None:
        best, best_distance = None, 15.0
        for obj in SKY_OBJECTS:
            if obj.ra_hours is None:
                continue
            p = self._project(obj.ra_hours, obj.dec_deg)
            if p is None:
                continue
            distance = math.hypot(p.x() - pos.x(), p.y() - pos.y())
            if distance < best_distance:
                best, best_distance = obj.key, distance
        return best

    # ------------------------------------------------------------------ state
    def set_selected(self, key: str | None) -> None:
        self._selected = key
        self.update()

    def _on_blink(self) -> None:
        if self._selected is not None and self.isVisible():
            self._blink = (self._blink + 1) % 8
            self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        hover = self._object_at(event.position())
        if hover != self._hover:
            self._hover = hover
            self.setToolTip(SKY_BY_KEY[hover].nickname if hover else "")
            self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        key = self._object_at(event.position())
        if key is not None:
            self.object_clicked.emit(key)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self._background = None
        super().resizeEvent(event)

    # --------------------------------------------------------------- painting
    def _render_background(self) -> QPixmap:
        ratio = self.devicePixelRatioF()
        pix = QPixmap(int(self.width() * ratio), int(self.height() * ratio))
        pix.setDevicePixelRatio(ratio)
        pix.fill(QColor(SKY_BG))
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self._map_rect()
        whole = self._view == "all"
        small = QFont(self.font())
        small.setPixelSize(10)
        painter.setFont(small)
        clip = QPainterPath()
        if whole:
            clip.addRect(rect)
        else:
            clip.addEllipse(rect)

        # Milky Way: a soft band along the galactic plane.
        ra, dec = galactic_plane(361)
        painter.setPen(QPen(QColor(SKY_MILKY), 16.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        path = QPainterPath()
        previous = None
        for r, d in zip(ra, dec):
            point = self._project(float(r), float(d))
            if point is None:
                previous = None
                continue
            if previous is None or abs(point.x() - previous.x()) > rect.width() / 2:   # gap or wrap at 0h / 24h
                path.moveTo(point)
            else:
                path.lineTo(point)
            previous = point
        painter.save()
        painter.setClipPath(clip)
        painter.setOpacity(0.55)
        painter.drawPath(path)
        painter.restore()

        # Grid and axis labels.
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if whole:
            painter.setPen(QPen(QColor(SKY_GRID), 1.0))
            for hour in range(0, 25, 3):
                x = self._project(hour, 0).x()
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            for declination in (-60, -30, 0, 30, 60):
                y = self._project(0, declination).y()
                painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QPen(QColor(SKY_LINE), 1.0))
            painter.drawRect(rect)
            painter.setPen(QColor(SKY_TEXT))
            for hour in range(0, 24, 3):
                x = self._project(hour, 0).x()
                painter.drawText(QRectF(x - 20, rect.bottom() + 2, 40, 14), Qt.AlignmentFlag.AlignCenter, f"{hour}h")
            for declination in (-60, -30, 0, 30, 60):
                y = self._project(0, declination).y()
                painter.drawText(QRectF(rect.left() + 3, y - 13, 40, 12), Qt.AlignmentFlag.AlignLeft, f"{declination:+d}°")
        else:
            sign = 1.0 if self._view == "north" else -1.0
            centre = rect.center()
            painter.setPen(QPen(QColor(SKY_GRID), 1.0))
            for declination in (60, 30, 0):
                radius = 0.5 * rect.width() * (90.0 - declination) / HEMISPHERE_SPAN
                painter.drawEllipse(centre, radius, radius)
            for hour in range(0, 24, 3):
                edge = self._project(hour, sign * (90.0 - HEMISPHERE_SPAN))
                painter.drawLine(centre, edge)
            painter.setPen(QPen(QColor(SKY_LINE), 1.2))
            painter.drawEllipse(rect)
            painter.setPen(QColor(SKY_TEXT))
            for hour in range(0, 24, 3):
                p = self._project(hour, sign * (90.0 - HEMISPHERE_SPAN + 7.0))
                painter.drawText(QRectF(p.x() - 14, p.y() - 7, 28, 14), Qt.AlignmentFlag.AlignCenter, f"{hour}h")
            for declination in (60, 30, 0):
                p = self._project(1.5, sign * declination)
                painter.drawText(QRectF(p.x() - 16, p.y() - 7, 32, 14), Qt.AlignmentFlag.AlignCenter, f"{sign * declination:+.0f}°")

        # Constellation stick figures.
        position = {name: (ra_h, dec_d) for name, ra_h, dec_d, _ in BRIGHT_STARS}
        painter.setPen(QPen(QColor(SKY_LINE), 1.2))
        for _label, _label_at, names in CONSTELLATION_LINES:
            points = [self._project(*position[name]) for name in names]
            for a, b in zip(points, points[1:]):
                if a is not None and b is not None and (not whole or abs(a.x() - b.x()) < rect.width() / 2):
                    painter.drawLine(a, b)
        painter.setPen(QColor("#5f7896"))
        for label, label_at, _names in CONSTELLATION_LINES:
            if label_at != (0.0, 0.0):
                p = self._project(*label_at)
                if p is not None:
                    painter.drawText(QRectF(p.x() - 50, p.y() - 7, 100, 14), Qt.AlignmentFlag.AlignCenter, label)

        # Stars, sized by brightness.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(SKY_STAR))
        for _name, ra_h, dec_d, magnitude in BRIGHT_STARS:
            p = self._project(ra_h, dec_d)
            if p is not None:
                radius = max(0.9, 3.1 - 0.62 * magnitude)
                painter.drawEllipse(p, radius, radius)
        painter.setPen(QColor(SKY_TEXT))
        for name, ra_h, dec_d, _ in BRIGHT_STARS:
            if name in LABELLED_STARS:
                p = self._project(ra_h, dec_d)
                if p is None:
                    continue
                width = painter.fontMetrics().horizontalAdvance(name)
                if p.x() + 8 + width > rect.right():   # no room on the right: label on the left
                    painter.drawText(QPointF(p.x() - 6 - width, p.y() + 4), name)
                else:
                    painter.drawText(QPointF(p.x() + 6, p.y() + 4), name)
        painter.end()
        return pix

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt naming)
        if self._background is None:
            self._background = self._render_background()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._background)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        label_font = QFont(self.font())
        label_font.setPixelSize(11)
        label_font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(label_font)
        for obj in SKY_OBJECTS:
            if obj.ra_hours is None:
                continue
            p = self._project(obj.ra_hours, obj.dec_deg)
            if p is None:
                continue
            color = QColor(KIND_COLORS[obj.kind])
            selected = obj.key == self._selected
            if selected:                               # expanding ring, like a beacon
                grow = self._blink / 8.0
                ring = QColor(color)
                ring.setAlphaF(1.0 - grow)
                painter.setPen(QPen(ring, 2.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(p, 7.0 + 16.0 * grow, 7.0 + 16.0 * grow)
            painter.setPen(QPen(color, 2.0))
            painter.setBrush(color if selected else QColor(SKY_BG))
            painter.drawEllipse(p, 5.0, 5.0)
            if selected or obj.key == self._hover or self.show_all_labels:
                text = obj.nickname
                width = painter.fontMetrics().horizontalAdvance(text) + 10
                x = min(max(p.x() - width / 2, 4.0), self.width() - width - 4.0)
                above = p.y() > 40 and not (self.show_all_labels and obj.key == "lgm1")   # two neighbours
                y = p.y() - 26 if above else p.y() + 10
                box = QRectF(x, y, width, 17)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(10, 15, 26, 230))
                painter.drawRoundedRect(box, 4, 4)
                painter.setPen(color)
                painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)
        painter.end()


class SkyPanel(QWidget):
    """Map + object list + explanation. Emits ``object_selected`` when the user picks a source."""

    object_selected = pyqtSignal(object)              # SkyObject

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(430)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        map_frame = QFrame()
        map_frame.setObjectName("pane")
        map_layout = QVBoxLayout(map_frame)
        map_layout.setContentsMargins(12, 9, 12, 10)
        map_layout.setSpacing(6)
        header = QHBoxLayout()
        self.cmb_view = QComboBox()
        for text, view in VIEWS:
            self.cmb_view.addItem(text, view)
        self.cmb_view.setToolTip("Tot cerul, sau o singură emisferă, ca pe un planiglob")
        legend = QLabel("&nbsp;&nbsp;".join(
            f"<span style='color:{KIND_COLORS[kind]}'>● {text}</span>"
            for kind, text in (("pulsar", "pulsar"), ("star", "stea"), ("variable", "stea pulsantă"), ("fiction", "film"))
        ))
        legend.setObjectName("paneInfo")
        legend.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.cmb_view)
        header.addWidget(legend, 1)
        map_layout.addLayout(header)
        self.map = SkyMapWidget()
        self.map.object_clicked.connect(self.select)
        self.cmb_view.currentIndexChanged.connect(lambda _: self.map.set_view(self.cmb_view.currentData()))
        map_layout.addWidget(self.map, 1)
        layout.addWidget(map_frame, 5)

        lower = QFrame()
        lower.setObjectName("pane")
        lower_layout = QHBoxLayout(lower)
        lower_layout.setContentsMargins(10, 10, 12, 10)
        lower_layout.setSpacing(12)
        self.list = QListWidget()
        self.list.setFixedWidth(190)
        for obj in SKY_OBJECTS:
            item = QListWidgetItem(obj.nickname)
            item.setData(Qt.ItemDataRole.UserRole, obj.key)
            item.setForeground(QColor(KIND_COLORS[obj.kind]))
            self.list.addItem(item)
        self.list.currentItemChanged.connect(self._on_list_changed)
        lower_layout.addWidget(self.list)
        self.info = ObjectInfo("Alege un punct de pe hartă",
                               "Fiecare punct colorat este un pulsar sau o stea pe care o poți asculta.")
        lower_layout.addWidget(self.info, 1)
        layout.addWidget(lower, 4)

    def _on_list_changed(self, current: QListWidgetItem | None, _previous) -> None:
        if current is not None:
            self.select(current.data(Qt.ItemDataRole.UserRole))

    def clear_selection(self) -> None:
        """Forget the selected object (the user switched to another source)."""
        self.map.set_selected(None)
        self.list.blockSignals(True)
        self.list.setCurrentItem(None)
        self.list.blockSignals(False)

    def select(self, key: str) -> None:
        obj: SkyObject = SKY_BY_KEY[key]
        if obj.dec_deg is not None and self.map._project(obj.ra_hours, obj.dec_deg) is None:
            self.cmb_view.setCurrentIndex(0)           # not visible in this hemisphere: show the whole sky
        self.map.set_selected(key)
        self.list.blockSignals(True)
        for row in range(self.list.count()):
            if self.list.item(row).data(Qt.ItemDataRole.UserRole) == key:
                self.list.setCurrentRow(row)
        self.list.blockSignals(False)
        self.info.show_object(obj)
        self.object_selected.emit(obj)
