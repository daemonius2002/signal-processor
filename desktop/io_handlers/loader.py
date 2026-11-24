"""Signal sources: ``.wav`` / ``.csv`` loaders, a synthetic multi-tone generator
and a real-time stream simulator. GUI-free; depends on NumPy, SciPy and
(optionally, preferred) soundfile.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


class SampleRateRequiredError(ValueError):
    """Raised when a CSV file has no usable time column and no sample rate was given."""


@dataclass
class SignalData:
    """A loaded signal: mono float32 samples plus what is needed to re-derive them."""

    samples: np.ndarray                      # mono, float32
    sample_rate: float
    name: str = "signal"
    path: str | None = None
    channels: np.ndarray | None = None       # (frames, n_channels) float32, WAV files only
    channel: int | None = None               # selected channel index, None = mono mix
    notes: list[str] = field(default_factory=list)

    @property
    def num_channels(self) -> int:
        return 1 if self.channels is None else int(self.channels.shape[1])

    @property
    def num_samples(self) -> int:
        return int(self.samples.size)

    @property
    def duration(self) -> float:
        return self.num_samples / self.sample_rate if self.sample_rate > 0 else 0.0

    def select_channel(self, channel: int | None) -> "SignalData":
        """Return a copy using one channel (0-based) or the mono mix (``None``)."""
        if self.channels is None:
            return self
        return SignalData(
            samples=_to_mono(self.channels, channel),
            sample_rate=self.sample_rate,
            name=self.name,
            path=self.path,
            channels=self.channels,
            channel=channel,
            notes=list(self.notes),
        )


# ------------------------------------------------------------------------ WAV
def _to_mono(channels: np.ndarray, channel: int | None) -> np.ndarray:
    n_channels = channels.shape[1]
    if channel is None:
        mono = channels[:, 0] if n_channels == 1 else channels.mean(axis=1)
    else:
        if not 0 <= channel < n_channels:
            raise ValueError(f"Channel {channel + 1} does not exist; the file has {n_channels} channel(s)")
        mono = channels[:, channel]
    return np.ascontiguousarray(mono, dtype=np.float32)


def _normalise_pcm(data: np.ndarray) -> np.ndarray:
    """Convert integer PCM (as returned by scipy.io.wavfile) to float32 in [-1, 1)."""
    if data.dtype == np.uint8:
        return ((data.astype(np.float32) - 128.0) / 128.0).astype(np.float32)
    if data.dtype == np.int16:
        return (data.astype(np.float32) / 32768.0).astype(np.float32)
    if data.dtype == np.int32:                       # also 24-bit, which scipy left-justifies
        return (data.astype(np.float64) / 2147483648.0).astype(np.float32)
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32)
    raise ValueError(f"Unsupported WAV sample format: {data.dtype}")


def _read_wav_channels(path: str) -> tuple[np.ndarray, float]:
    """Read a WAV file as (frames x channels float32 array, sample rate)."""
    errors: list[str] = []
    try:
        import soundfile as sf

        data, fs = sf.read(path, dtype="float32", always_2d=True)
        return data, float(fs)
    except ImportError as exc:                       # soundfile or libsndfile not installed
        errors.append(f"soundfile unavailable ({exc})")
    except Exception as exc:                         # unreadable by libsndfile: try SciPy
        errors.append(f"soundfile: {exc}")
    try:
        from scipy.io import wavfile

        fs, data = wavfile.read(path)
        data = _normalise_pcm(np.asarray(data))
        if data.ndim == 1:
            data = data[:, np.newaxis]
        return data, float(fs)
    except Exception as exc:
        errors.append(f"scipy.io.wavfile: {exc}")
    raise ValueError(f"Could not read WAV file '{os.path.basename(path)}': " + "; ".join(errors))


def load_wav(path: str | os.PathLike, channel: int | None = None, normalize_peak: bool = False) -> SignalData:
    """Load a WAV file.

    The sample rate comes from the file header. Samples are converted to
    float32 where +/-1.0 is digital full scale. ``channel=None`` mixes all
    channels down to mono; an integer selects one channel (0-based).
    ``normalize_peak=True`` additionally rescales the signal so its peak is 1.0.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    channels, fs = _read_wav_channels(path)
    if channels.shape[0] == 0:
        raise ValueError(f"'{os.path.basename(path)}' contains no audio samples")
    if fs <= 0:
        raise ValueError(f"'{os.path.basename(path)}' has an invalid sample rate ({fs})")
    notes: list[str] = []
    if not np.all(np.isfinite(channels)):
        channels = np.nan_to_num(channels, nan=0.0, posinf=0.0, neginf=0.0)
        notes.append("Non-finite samples were replaced by 0.")
    if normalize_peak:
        peak = float(np.max(np.abs(channels)))
        if peak > 0:
            channels = (channels / peak).astype(np.float32)
            notes.append(f"Peak-normalised (original peak {peak:.4g}).")
    return SignalData(
        samples=_to_mono(channels, channel),
        sample_rate=fs,
        name=os.path.basename(path),
        path=path,
        channels=channels,
        channel=channel,
        notes=notes,
    )


# ------------------------------------------------------------------------ CSV
_NUMBER_RE = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


def _is_number(token: str) -> bool:
    token = token.strip().strip('"').strip("'")
    return bool(_NUMBER_RE.match(token)) or token.lower() in ("nan", "inf", "-inf", "+inf")


def _detect_delimiter(lines: Sequence[str]) -> str | None:
    """Return ';', tab, ',' or ``None`` (= any whitespace / single column)."""
    sample = lines[:50]
    for delim in (";", "\t"):
        if sum(delim in line for line in sample) > len(sample) / 2:
            return delim
    if sum("," in line for line in sample) > len(sample) / 2:
        return ","
    return None


def _split(line: str, delimiter: str | None) -> list[str]:
    parts = line.split(delimiter) if delimiter else line.split()
    return [p.strip().strip('"').strip("'") for p in parts]


def _parse_csv(path: str) -> tuple[np.ndarray, list[str]]:
    """Read a CSV file into (rows x columns float64 array, header names or [])."""
    with open(path, "r", encoding="utf-8-sig", errors="replace", newline=None) as fh:
        lines = [ln.strip() for ln in fh]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    if not lines:
        raise ValueError("The file is empty")

    delimiter = _detect_delimiter(lines)
    decimal_comma = delimiter != "," and any("," in ln for ln in lines[:50])

    header: list[str] = []
    skip = 0
    while skip < min(len(lines), 10):                # header = leading rows that are not all numeric
        tokens = [t for t in _split(lines[skip], delimiter) if t != ""]
        if tokens and all(_is_number(t) for t in tokens):
            break
        if not header:
            header = _split(lines[skip], delimiter)
        skip += 1
    body = lines[skip:]
    if not body:
        raise ValueError("No numeric data rows were found")
    if decimal_comma:
        body = [ln.replace(",", ".") for ln in body]
    if delimiter:                                    # tolerate a trailing delimiter on each row
        body = [ln.rstrip(delimiter).rstrip() if ln.endswith(delimiter) else ln for ln in body]
    try:
        data = np.loadtxt(body, delimiter=delimiter, ndmin=2, dtype=np.float64, quotechar='"')
    except ValueError as exc:
        raise ValueError(f"Could not parse numeric data: {exc}") from exc
    return data, header


def _time_scale_from_header(name: str) -> float | None:
    """Seconds per unit for a time column, guessed from its header; ``None`` if unknown."""
    label = name.strip().lower()
    if not label:
        return None
    units = re.findall(r"[a-zµμ]+", label)
    if "ms" in units or "millisecond" in label or "msec" in label:
        return 1e-3
    if "us" in units or "µs" in units or "μs" in units or "microsecond" in label:
        return 1e-6
    if "ns" in units or "nanosecond" in label:
        return 1e-9
    if "s" in units or "sec" in units or "second" in label or "time" in label or label in ("t", "timp"):
        return 1.0
    return None


def load_csv(
    path: str | os.PathLike,
    sample_rate: float | None = None,
    time_column: int = 0,
    amplitude_column: int | None = None,
) -> SignalData:
    """Load a signal from a CSV / text file.

    * Header rows are detected automatically (and skipped).
    * Delimiter is detected automatically: comma, semicolon, tab or spaces.
      With semicolon / tab / space delimiters a decimal comma is accepted.
    * 1 column: amplitude only - ``sample_rate`` must be given, otherwise
      :class:`SampleRateRequiredError` is raised.
    * 2+ columns: ``time, amplitude``; the sample rate is computed from the
      median time step. If the first column is just a sample counter
      (0, 1, 2, ...) and nothing says it is in seconds, ``sample_rate`` is
      required as in the 1-column case.
    * An explicitly passed ``sample_rate`` always wins over the computed one.
    """
    path = os.fspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    name = os.path.basename(path)
    try:
        data, header = _parse_csv(path)
    except ValueError as exc:
        raise ValueError(f"Could not read CSV file '{name}': {exc}") from exc

    if sample_rate is not None and (not math.isfinite(sample_rate) or sample_rate <= 0):
        raise ValueError(f"Sample rate must be a positive number, got {sample_rate!r}")

    n_cols = data.shape[1]
    notes: list[str] = []
    if n_cols == 1:
        amplitude = data[:, 0]
        if sample_rate is None:
            raise SampleRateRequiredError(
                f"'{name}' has a single column (amplitude only); a sample rate must be supplied"
            )
        fs = float(sample_rate)
    else:
        amp_col = amplitude_column if amplitude_column is not None else (1 if time_column == 0 else 0)
        if not (0 <= time_column < n_cols and 0 <= amp_col < n_cols) or amp_col == time_column:
            raise ValueError(f"Invalid column selection for '{name}' ({n_cols} columns)")
        keep = np.isfinite(data[:, time_column]) & np.isfinite(data[:, amp_col])
        t = data[keep, time_column]
        amplitude = data[keep, amp_col]
        if t.size < 2:
            raise ValueError(f"'{name}' needs at least two valid rows to work out the sample rate")
        steps = np.diff(t)
        dt = float(np.median(steps))
        if dt <= 0:
            raise ValueError(f"The time column of '{name}' is not increasing")
        time_header = header[time_column] if time_column < len(header) else ""
        unit = _time_scale_from_header(time_header)
        looks_like_index = abs(dt - 1.0) < 1e-9 and unit is None
        if sample_rate is not None:
            fs = float(sample_rate)
        elif looks_like_index:
            raise SampleRateRequiredError(
                f"The first column of '{name}' is a sample counter, not a time axis; "
                "a sample rate must be supplied"
            )
        else:
            fs = 1.0 / (dt * (unit if unit is not None else 1.0))
            if abs(fs - round(fs)) < 1e-6 * fs:        # 7999.9999996 Hz from rounded time stamps
                fs = float(round(fs))
        jitter = float(np.max(np.abs(steps - dt))) / dt
        if jitter > 0.01:
            notes.append(
                f"Time steps are not uniform (up to {100 * jitter:.1f}% deviation); "
                "the median step was used for the sample rate."
            )

    amplitude = amplitude[np.isfinite(amplitude)]
    if amplitude.size == 0:
        raise ValueError(f"'{name}' contains no valid amplitude values")
    return SignalData(
        samples=np.ascontiguousarray(amplitude, dtype=np.float32),
        sample_rate=fs,
        name=name,
        path=path,
        notes=notes,
    )


def load_signal(path: str | os.PathLike, sample_rate: float | None = None, channel: int | None = None) -> SignalData:
    """Load ``.wav`` or ``.csv`` / ``.txt`` / ``.tsv`` / ``.dat`` according to the file extension."""
    ext = os.path.splitext(os.fspath(path))[1].lower()
    if ext in (".wav", ".wave"):
        return load_wav(path, channel=channel)
    if ext in (".csv", ".txt", ".tsv", ".dat"):
        return load_csv(path, sample_rate=sample_rate)
    raise ValueError(f"Unsupported file type '{ext}'. Supported: .wav, .csv")


# ------------------------------------------------------------------ generator
GENERATOR_MODES = ("tones", "white", "pink")
# Pinking filter (white -> pink, about -3 dB per octave from 10 Hz up); coefficients after J. O. Smith.
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])


class SyntheticGenerator:
    """Phase-continuous multi-tone + Gaussian noise generator.

    ``tones`` is a sequence of ``(frequency_hz, amplitude)`` pairs. Successive
    calls to :meth:`generate` join seamlessly, also across parameter changes.
    """

    def __init__(
        self,
        sample_rate: float = 48000.0,
        tones: Sequence[tuple[float, float]] = ((440.0, 0.5), (1200.0, 0.25), (2500.0, 0.0)),
        noise_std: float = 0.01,
        seed: int | None = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        self.sample_rate = float(sample_rate)
        self.noise_std = float(noise_std)
        self.name = "Live generator"
        self.mode = "tones"                           # "tones", "white" or "pink"
        self._pink_state = np.zeros(3)
        self._rng = np.random.default_rng(seed)
        self._freqs = np.zeros(0)
        self._amps = np.zeros(0)
        self._phases = np.zeros(0)
        self.set_tones(tones)

    @property
    def tones(self) -> list[tuple[float, float]]:
        return [(float(f), float(a)) for f, a in zip(self._freqs, self._amps)]

    def set_tones(self, tones: Sequence[tuple[float, float]]) -> None:
        """Replace the tone list; tones that keep their position keep their phase."""
        freqs = np.array([float(f) for f, _ in tones], dtype=np.float64)
        amps = np.array([float(a) for _, a in tones], dtype=np.float64)
        if np.any(freqs < 0) or np.any(freqs >= 0.5 * self.sample_rate):
            raise ValueError("Tone frequencies must lie between 0 and the Nyquist frequency")
        phases = np.zeros(freqs.size)
        keep = min(freqs.size, self._phases.size)
        phases[:keep] = self._phases[:keep]
        self._freqs, self._amps, self._phases = freqs, amps, phases

    def set_noise(self, noise_std: float) -> None:
        self.noise_std = max(0.0, float(noise_std))

    def reset(self) -> None:
        self._phases[:] = 0.0

    def set_mode(self, mode: str) -> None:
        """"tones" (multi-tone + noise), "white" (white noise) or "pink" (pink noise, -3 dB per octave)."""
        if mode not in GENERATOR_MODES:
            raise ValueError(f"Unknown generator mode {mode!r}; choose one of {GENERATOR_MODES}")
        self.mode = mode
        self._pink_state = np.zeros(3)

    def _noise(self, n: int) -> np.ndarray:
        """Full-band test noise at about -20 dBFS RMS."""
        white = self._rng.standard_normal(n)
        if self.mode == "white":
            return 0.1 * white
        from scipy.signal import lfilter

        pink, self._pink_state = lfilter(_PINK_B, _PINK_A, white, zi=self._pink_state)
        return 1.2 * pink                             # the pinking filter attenuates; bring it back to about -20 dBFS

    def generate(self, n: int) -> np.ndarray:
        """Return the next ``n`` samples as float32."""
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        if self.mode != "tones":
            return self._noise(n).astype(np.float32)
        out = np.zeros(n, dtype=np.float64)
        if self._freqs.size:
            step = 2.0 * np.pi * self._freqs / self.sample_rate          # radians per sample
            phase = self._phases[:, np.newaxis] + step[:, np.newaxis] * np.arange(n)
            out = self._amps @ np.sin(phase)
            self._phases = (self._phases + step * n) % (2.0 * np.pi)
        if self.noise_std > 0:
            out = out + self._rng.normal(0.0, self.noise_std, n)
        return out.astype(np.float32)


# --------------------------------------------------------------------- stream
class StreamSimulator:
    """Feeds a signal chunk by chunk, as an acquisition device would.

    ``source`` is either a :class:`SignalData` (played back, looping by default)
    or a :class:`SyntheticGenerator` (endless). ``frame_rate`` is the number of
    chunks per second, so each chunk holds ``sample_rate / frame_rate`` samples.
    The simulator has no clock of its own: call :meth:`next_chunk` once per
    frame, or :meth:`chunk_for_elapsed` with the measured wall-clock time since
    the previous call to stay locked to real time even when timers jitter.
    """

    MIN_FRAME_RATE = 1.0
    MAX_FRAME_RATE = 1000.0

    def __init__(self, source: SignalData | SyntheticGenerator, frame_rate: float = 60.0, loop: bool = True) -> None:
        self.source = source
        self.loop = bool(loop)
        self._is_generator = hasattr(source, "generate")      # generator, microphone, ...
        if not self._is_generator and source.num_samples == 0:
            raise ValueError("Cannot stream an empty signal")
        self._position = 0
        self._carry = 0.0
        self._frame_rate = 60.0
        self.set_frame_rate(frame_rate)

    @property
    def sample_rate(self) -> float:
        return float(self.source.sample_rate)

    @property
    def name(self) -> str:
        return self.source.name

    @property
    def is_live(self) -> bool:
        return self._is_generator

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    def set_frame_rate(self, frame_rate: float) -> None:
        self._frame_rate = float(min(max(frame_rate, self.MIN_FRAME_RATE), self.MAX_FRAME_RATE))

    @property
    def chunk_size(self) -> int:
        """Nominal number of samples per frame."""
        return max(1, int(round(self.sample_rate / self._frame_rate)))

    @property
    def position(self) -> int:
        """Index of the next sample to be delivered (file sources)."""
        return self._position

    @property
    def progress(self) -> float:
        """Playback position as a fraction 0..1 (always 0 for the live generator)."""
        return 0.0 if self._is_generator else self._position / self.source.num_samples

    @property
    def finished(self) -> bool:
        return (not self._is_generator) and (not self.loop) and self._position >= self.source.num_samples

    def reset(self) -> None:
        self._position = 0
        self._carry = 0.0
        if self._is_generator:
            self.source.reset()

    def read(self, n: int) -> np.ndarray:
        """Deliver the next ``n`` samples (fewer at the end of a non-looping file)."""
        n = int(n)
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        if self._is_generator:
            return self.source.generate(n)
        samples = self.source.samples
        total = samples.size
        if self.loop:
            idx = (self._position + np.arange(n)) % total
            self._position = (self._position + n) % total
            return samples[idx]
        chunk = samples[self._position:self._position + n]
        self._position = min(self._position + n, total)
        return chunk.copy()

    def next_chunk(self) -> np.ndarray:
        """Deliver one nominal frame worth of samples."""
        return self.read(self.chunk_size)

    def chunk_for_elapsed(self, elapsed_seconds: float, max_frames: float = 4.0) -> np.ndarray:
        """Deliver exactly the samples that correspond to ``elapsed_seconds`` of real time.

        Fractions of a sample are carried over to the next call, so the stream
        does not drift. After a stall, at most ``max_frames`` nominal frames
        are delivered at once (the stream skips ahead instead of piling up).
        """
        wanted = max(0.0, float(elapsed_seconds)) * self.sample_rate + self._carry
        limit = max_frames * self.chunk_size
        if wanted > limit:
            wanted = float(limit)
        n = int(wanted)
        self._carry = wanted - n
        return self.read(n)
