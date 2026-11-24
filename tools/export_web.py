"""Export the data used by the web version (web/data) from the desktop modules.

    python tools/export_web.py

Needs the desktop requirements (numpy, scipy) and desktop/samples_sunete (the sound kit).
"""

from __future__ import annotations

import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "desktop"))

import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from io_handlers import sky_catalog as sky  # noqa: E402

OUT = os.path.join(ROOT, "web", "data")
AUDIO_SECONDS = 6.0


def export() -> None:
    for sub in ("analysis", "audio"):
        shutil.rmtree(os.path.join(OUT, sub), ignore_errors=True)
        os.makedirs(os.path.join(OUT, sub))
    objects = []
    for obj in sky.SKY_OBJECTS + sky.PLANETS:
        seconds = max(6.0, 2.5 * obj.fft_size / obj.analysis_rate)
        if obj.kind == "fiction":
            seconds = 26.0
        data = sky.analysis_signal(obj, seconds)
        np.round(np.clip(data.samples, -1, 1) * 32767).astype("<i2").tofile(os.path.join(OUT, "analysis", f"{obj.key}.i16"))
        audio = sky.audio_signal(obj, 26.0 if obj.kind == "fiction" else AUDIO_SECONDS)
        wavfile.write(os.path.join(OUT, "audio", f"{obj.key}.wav"), sky.AUDIO_SAMPLE_RATE, np.round(audio * 32767).astype(np.int16))
        factor = {"pulsar": 12.0, "planet": 12.0, "variable": 6.0, "star": 2.0}.get(obj.kind, 2.5)
        text = sky.frequency_text(obj)
        objects.append({
            "key": obj.key, "name": obj.name, "nick": obj.nickname, "kind": obj.kind, "kindRo": obj.kind_ro,
            "constellation": obj.constellation, "ra": obj.ra_hours, "dec": obj.dec_deg,
            "hz": round(obj.display_frequency_hz, 6), "fs": obj.analysis_rate, "fft": obj.fft_size,
            "viewHz": round(factor * obj.display_frequency_hz, 3), "facts": text[0].upper() + text[1:] + ".",
            "story": obj.story_ro,
        })
    ra, dec = sky.galactic_plane(181)
    catalogue = {
        "objects": objects,
        "stars": [[name, ra_h, dec_d, mag] for name, ra_h, dec_d, mag in sky.BRIGHT_STARS],
        "labelled": list(sky.LABELLED_STARS),
        "lines": [[label, list(at), list(names)] for label, at, names in sky.CONSTELLATION_LINES],
        "milkyWay": [[round(float(r), 3), round(float(d), 2)] for r, d in zip(ra, dec)],
        "orbits": {key: list(value) for key, value in sky.PLANET_ORBITS.items()},
    }
    kit_src = os.path.join(ROOT, "desktop", "samples_sunete")
    kit_dst = os.path.join(OUT, "sounds")
    shutil.rmtree(kit_dst, ignore_errors=True)
    shutil.copytree(kit_src, kit_dst)
    with open(os.path.join(kit_dst, "catalog.json"), encoding="utf-8") as fh:
        catalogue["sounds"] = json.load(fh)["sounds"]
    os.remove(os.path.join(kit_dst, "catalog.json"))
    with open(os.path.join(OUT, "catalog.json"), "w", encoding="utf-8") as fh:
        json.dump(catalogue, fh, ensure_ascii=False)
    print(f"{len(objects)} sky objects, {len(catalogue['sounds'])} sounds -> {OUT}")


if __name__ == "__main__":
    export()
