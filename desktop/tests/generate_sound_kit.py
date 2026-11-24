"""Write the children's sound kit: WAV files by category plus catalog.json.

Run from the project folder:

    python tests/generate_sound_kit.py                 # writes into samples_sunete/

All sounds are synthesised (see io_handlers/sound_kit.py). The spoken items are
made with the eSpeak NG speech synthesiser if it is installed (``espeak-ng``);
without it they are skipped.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from scipy import signal as sp_signal  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from io_handlers.sound_kit import CATEGORIES, FS, KIT, render  # noqa: E402


def _slug(text: str) -> str:
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", plain).strip("_")


def _speak(text: str, pitch: int) -> np.ndarray | None:
    program = shutil.which("espeak-ng") or shutil.which("espeak")
    if program is None:
        return None
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "speech.wav")
        subprocess.run([program, "-v", "ro", "-s", "125", "-p", str(pitch), "-w", path, text], check=True)
        rate, data = wavfile.read(path)
    samples = data.astype(np.float64) / 32768.0
    samples = sp_signal.resample_poly(samples, FS, rate)
    pad = np.zeros(int(0.4 * FS))
    samples = np.concatenate([pad, samples, pad])
    return (0.8 * samples / np.max(np.abs(samples))).astype(np.float32)


def generate_all(out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    catalogue = []
    written = []
    for sound in KIT:
        samples = render(sound) if sound.make is not None else _speak(sound.speech, sound.speech_pitch)
        if samples is None:
            print(f"skipped (no speech synthesiser): {sound.title}")
            continue
        number = CATEGORIES.index(sound.category) + 1
        folder = f"{number}_{_slug(sound.category)}"
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)
        name = f"{folder}/{_slug(sound.title)}.wav"
        path = os.path.join(out_dir, name)
        wavfile.write(path, FS, np.round(samples * 32767).astype(np.int16))
        written.append(path)
        catalogue.append({"key": sound.key, "category": sound.category, "title": sound.title, "story": sound.story,
                          "file": name, "view_hz": sound.view_hz, "seconds": round(samples.size / FS, 2)})
    index = os.path.join(out_dir, "catalog.json")
    with open(index, "w", encoding="utf-8") as fh:
        json.dump({"note": "Sunete sintetizate pe calculator, nu înregistrări.", "sounds": catalogue}, fh, ensure_ascii=False, indent=1)
    written.append(index)
    return written


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples_sunete")
    files = generate_all(target)
    print(f"{len(files) - 1} sounds written to {target}")
