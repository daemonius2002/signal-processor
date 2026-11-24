"""Sound-kit panel: example sounds by category (tones, instruments, music, voice, nature...)."""

from __future__ import annotations

import json
import os

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QFrame, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from .sky_panel import ObjectInfo

KIT_FOLDER = "samples_sunete"


def find_kit_folder() -> str | None:
    """The kit sits next to the application: in the project folder, or one level up when installed."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for base in (here, os.path.dirname(here)):
        candidate = os.path.join(base, KIT_FOLDER)
        if os.path.isfile(os.path.join(candidate, "catalog.json")):
            return candidate
    return None


class SoundKitPanel(QWidget):
    """Emits ``sound_selected(dict)`` with the catalogue entry plus the absolute ``path``."""

    sound_selected = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(380)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.currentItemChanged.connect(self._on_item)
        layout.addWidget(self.tree, 5)

        lower = QFrame()
        lower.setObjectName("pane")
        lower_layout = QVBoxLayout(lower)
        lower_layout.setContentsMargins(12, 10, 12, 10)
        self.info = ObjectInfo("Alege un sunet", "Sunetele sunt făcute pe calculator, nu înregistrate. "
                               "Poți adăuga și sunetele tale cu Load WAV sau cu microfonul.")
        lower_layout.addWidget(self.info)
        layout.addWidget(lower, 3)

        self.folder = find_kit_folder()
        if self.folder is None:
            self.info.show_text("Kitul de sunete lipsește",
                                f"Nu am găsit folderul «{KIT_FOLDER}».",
                                "Rulează «python tests/generate_sound_kit.py» în folderul proiectului, apoi repornește aplicația.")
            return
        try:
            with open(os.path.join(self.folder, "catalog.json"), encoding="utf-8") as fh:
                sounds = json.load(fh)["sounds"]
        except (OSError, ValueError, KeyError) as exc:
            self.info.show_text("Kitul de sunete nu poate fi citit", str(exc), "")
            return
        groups: dict[str, QTreeWidgetItem] = {}
        for entry in sounds:
            group = groups.get(entry["category"])
            if group is None:
                group = QTreeWidgetItem([entry["category"]])
                group.setForeground(0, QColor("#8595a8"))
                group.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.tree.addTopLevelItem(group)
                groups[entry["category"]] = group
            item = QTreeWidgetItem([entry["title"]])
            item.setData(0, Qt.ItemDataRole.UserRole, entry)
            group.addChild(item)
        self.tree.expandAll()

    def _on_item(self, current: QTreeWidgetItem | None, _previous) -> None:
        entry = current.data(0, Qt.ItemDataRole.UserRole) if current is not None else None
        if not entry:
            return
        self.info.show_text(entry["title"], f"{entry['category']}, {entry['seconds']:.0f} secunde.", entry["story"])
        self.sound_selected.emit({**entry, "path": os.path.join(self.folder, entry["file"])})

    def clear_selection(self) -> None:
        self.tree.blockSignals(True)
        self.tree.setCurrentItem(None)
        self.tree.blockSignals(False)
