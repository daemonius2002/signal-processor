"""Write the sky samples: one WAV per pulsar / star / planet, a catalogue and a sky map.

Run from the project folder:

    python tests/generate_sky_samples.py                # writes into samples_stele/
    python tests/generate_sky_samples.py some/folder

All signals are SYNTHESISED from published, rounded values (pulsar spin
frequencies, stellar oscillation frequencies). They are not telescope
recordings. The Vega entry is fiction (the film "Contact").
"""

from __future__ import annotations

import csv
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from io_handlers.sky_catalog import AUDIO_SAMPLE_RATE, PLANETS, SKY_OBJECTS, audio_signal, frequency_text  # noqa: E402


def _slug(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9+-]+", "_", plain).strip("_")


def _hms(hours: float) -> str:
    total = round(hours * 3600)
    return f"{total // 3600:02d}h {total % 3600 // 60:02d}m {total % 60:02d}s"


def _dms(degrees: float) -> str:
    sign = "-" if degrees < 0 else "+"
    total = round(abs(degrees) * 60)
    return f"{sign}{total // 60:02d}° {total % 60:02d}'"


def generate_all(out_dir: str, seconds: float = 16.0, with_map: bool = True) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    rows = []
    for index, obj in enumerate(SKY_OBJECTS + PLANETS, start=1):
        freq = obj.display_frequency_hz
        tag = f"{freq:.5g}Hz" if obj.kind in ("pulsar", "fiction") else f"{freq:.4g}Hz_accelerat"
        name = f"{index:02d}_{_slug(obj.nickname)}_{tag}.wav"
        path = os.path.join(out_dir, name)
        wavfile.write(path, AUDIO_SAMPLE_RATE, np.round(audio_signal(obj, seconds) * 32767).astype(np.int16))
        written.append(path)
        rows.append([
            name, obj.nickname, obj.name, obj.kind_ro, obj.constellation,
            "" if obj.ra_hours is None else _hms(obj.ra_hours),
            "" if obj.dec_deg is None else _dms(obj.dec_deg),
            frequency_text(obj), obj.story_ro,
        ])
    catalogue = os.path.join(out_dir, "catalog_stele.csv")
    with open(catalogue, "w", newline="", encoding="utf-8-sig") as fh:          # BOM: Excel reads the diacritics
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["Fișier", "Nume", "Denumire catalog", "Tip", "Constelație", "Ascensie dreaptă (J2000)",
                         "Declinație (J2000)", "Frecvența reală", "Poveste"])
        writer.writerows(rows)
    written.append(catalogue)
    if with_map:
        written.append(_write_map(os.path.join(out_dir, "harta_celesta.png")))
    return written


def _write_map(path: str) -> str:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    from ui.sky_panel import SkyMapWidget

    widget = SkyMapWidget()
    widget.show_all_labels = True
    widget.resize(1500, 780)
    widget.setStyleSheet("font-family: 'Segoe UI', 'Noto Sans', 'DejaVu Sans', sans-serif; font-size: 13px;")
    widget.grab().save(path, "PNG")
    del app
    return path


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples_stele")
    for item in generate_all(target):
        print(item)
