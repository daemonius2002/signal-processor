"""Generate the test signals used by the test-suite and by the User Guide examples.

Run from the project folder:

    python tests/generate_samples.py              # writes into tests/samples/
    python tests/generate_samples.py some/folder  # writes somewhere else

Files written (all contain 440 Hz at amplitude 0.5 + 1200 Hz at amplitude 0.25 + noise):

    tone_440_1200_mono.wav      44.1 kHz, 16-bit PCM, mono, 3 s
    tone_440_1200_stereo.wav    44.1 kHz, 16-bit PCM, left = 440 Hz only, right = 1200 Hz only
    tone_440_1200_2col.csv      "time_s,amplitude" header, 8 kHz, 2 s
    tone_440_1200_1col.csv      amplitude only, no header, 8 kHz, 2 s (needs sample rate 8000)
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.io import wavfile

TONES = ((440.0, 0.5), (1200.0, 0.25))      # (frequency in Hz, amplitude)
NOISE_STD = 0.01
WAV_SAMPLE_RATE = 44100
CSV_SAMPLE_RATE = 8000
WAV_SECONDS = 3.0
CSV_SECONDS = 2.0
SEED = 20240229


def make_signal(sample_rate: float, seconds: float, tones=TONES, noise_std: float = NOISE_STD, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return (time axis, multi-tone signal with Gaussian noise) as float64 arrays."""
    n = int(round(sample_rate * seconds))
    t = np.arange(n) / float(sample_rate)
    signal = np.zeros(n)
    for frequency, amplitude in tones:
        signal += amplitude * np.sin(2.0 * np.pi * frequency * t)
    if noise_std > 0:
        signal += np.random.default_rng(seed).normal(0.0, noise_std, n)
    return t, signal


def _to_pcm16(signal: np.ndarray) -> np.ndarray:
    return np.round(np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)


def generate_all(out_dir: str) -> dict[str, str]:
    """Write all sample files into ``out_dir`` and return ``{key: path}``."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {
        "wav_mono": os.path.join(out_dir, "tone_440_1200_mono.wav"),
        "wav_stereo": os.path.join(out_dir, "tone_440_1200_stereo.wav"),
        "csv_2col": os.path.join(out_dir, "tone_440_1200_2col.csv"),
        "csv_1col": os.path.join(out_dir, "tone_440_1200_1col.csv"),
    }

    _, mono = make_signal(WAV_SAMPLE_RATE, WAV_SECONDS)
    wavfile.write(paths["wav_mono"], WAV_SAMPLE_RATE, _to_pcm16(mono))

    _, left = make_signal(WAV_SAMPLE_RATE, WAV_SECONDS, tones=TONES[:1])
    _, right = make_signal(WAV_SAMPLE_RATE, WAV_SECONDS, tones=TONES[1:], seed=SEED + 1)
    wavfile.write(paths["wav_stereo"], WAV_SAMPLE_RATE, np.column_stack([_to_pcm16(left), _to_pcm16(right)]))

    t, signal = make_signal(CSV_SAMPLE_RATE, CSV_SECONDS)
    np.savetxt(paths["csv_2col"], np.column_stack([t, signal]), delimiter=",", header="time_s,amplitude", comments="", fmt="%.9f")
    np.savetxt(paths["csv_1col"], signal, fmt="%.9f")
    return paths


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    for key, path in generate_all(target).items():
        print(f"{key:11s} {path}")
