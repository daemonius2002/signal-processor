"""Vectorised frequency-domain signal processing engine.

The module is GUI-free and depends only on NumPy and SciPy.

Conventions
-----------
* Signals are real, mono and expressed in "full scale" units: a sine wave of
  amplitude 1.0 reads 0 dBFS in the magnitude spectrum.
* The magnitude spectrum is single-sided and compensated for the coherent gain
  of the window, so a tone of amplitude ``A`` peaks at ``A`` whatever the window.
* The PSD is single-sided, in ``FS^2/Hz``; integrating it over frequency gives
  the mean power of the windowed frame.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy import fft as sp_fft
from scipy import ndimage as sp_ndimage
from scipy import signal as sp_signal

WINDOW_FUNCTIONS: tuple[str, ...] = ("Hann", "Hamming", "Blackman", "Rectangular")
FILTER_TYPES: tuple[str, ...] = ("Bypass", "Low-pass", "High-pass", "Band-pass", "Notch")
FFT_SIZES: tuple[int, ...] = (512, 1024, 2048, 4096, 8192, 16384)
SCALES: tuple[str, ...] = ("dB", "Linear")

_SCIPY_WINDOW_NAMES = {
    "Hann": "hann",
    "Hamming": "hamming",
    "Blackman": "blackman",
    "Rectangular": "boxcar",
}

# Half width of the window main lobe, in FFT bins (first null on each side).
_MAIN_LOBE_HALF_WIDTH = {"Rectangular": 1, "Hann": 2, "Hamming": 2, "Blackman": 3}

_MAX_HARMONICS = 10          # THD uses harmonics 2 .. 10 that lie below Nyquist
_DB_FLOOR = 1e-12            # -240 dB amplitude floor, avoids log10(0)
_POWER_FLOOR = 1e-30
_TONE_WINDOW_BINS = 129      # neighbourhood used to tell discrete tones from the noise floor
_TONE_THRESHOLD = 12.0       # a bin this many times above its neighbourhood mean is a tone
_RING_CAPACITY = 2 * max(FFT_SIZES)
_ZERO_PHASE_HISTORY = 1024   # extra past samples filtered with each zero-phase frame


# Nominal centre frequencies of the 28 third-octave bands from 31.5 Hz to 16 kHz
# (the range of classic LED real-time analysers).
THIRD_OCTAVE_CENTRES: tuple[float, ...] = (
    31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0,
    1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0, 6300.0, 8000.0, 10000.0, 12500.0, 16000.0,
)


def third_octave_levels(psd: np.ndarray, sample_rate: float, floor_db: float = -120.0) -> np.ndarray:
    """Level of each third-octave band in dBFS (RMS), from a single-sided PSD.

    The power in a band is the PSD integrated between the band edges
    ``fc * 2**(-1/6)`` and ``fc * 2**(1/6)``; bins that straddle an edge count
    in proportion, so low bands narrower than one FFT bin still get a value.
    Bands that reach beyond the Nyquist frequency are returned as NaN.
    Pink noise reads the same level in every band; white noise rises 1 dB per band.
    """
    n_bins = psd.size
    df = sample_rate / (2.0 * (n_bins - 1))
    edges = np.arange(n_bins + 1) * df - 0.5 * df                  # each bin covers f +/- df/2
    cumulative = np.concatenate(([0.0], np.cumsum(psd) * df))      # power below each edge
    # A band holds mean-square = sum(PSD * df); the windowed PSD is normalised so this is the band power.
    centres = np.asarray(THIRD_OCTAVE_CENTRES)
    lo = np.interp(centres * 2.0 ** (-1.0 / 6.0), edges, cumulative)
    hi = np.interp(centres * 2.0 ** (1.0 / 6.0), edges, cumulative)
    power = np.maximum(hi - lo, 10.0 ** (floor_db / 10.0))
    levels = 10.0 * np.log10(power)
    levels[centres * 2.0 ** (1.0 / 6.0) > 0.5 * sample_rate] = np.nan
    return levels


@dataclass(frozen=True)
class DSPConfig:
    """All user-selectable processing parameters."""

    fft_size: int = 4096
    window: str = "Hann"
    filter_type: str = "Bypass"
    cutoff_hz: float = 1000.0        # low-pass / high-pass cutoff, band-pass lower edge, notch centre
    cutoff_high_hz: float = 4000.0   # band-pass upper edge
    filter_order: int = 4            # Butterworth order (band filters have 2x this many poles)
    notch_q: float = 10.0            # notch quality factor: stop-band width = centre / Q
    zero_phase: bool = False         # False: causal sosfilt, True: zero-phase sosfiltfilt
    scale: str = "dB"                # "dB" or "Linear" (used by FrameResult.display_*)

    def validate(self) -> None:
        if self.fft_size < 16 or self.fft_size > _RING_CAPACITY:
            raise ValueError(f"fft_size must be between 16 and {_RING_CAPACITY}, got {self.fft_size}")
        if self.window not in WINDOW_FUNCTIONS:
            raise ValueError(f"Unknown window {self.window!r}; choose one of {WINDOW_FUNCTIONS}")
        if self.filter_type not in FILTER_TYPES:
            raise ValueError(f"Unknown filter {self.filter_type!r}; choose one of {FILTER_TYPES}")
        if self.scale not in SCALES:
            raise ValueError(f"Unknown scale {self.scale!r}; choose one of {SCALES}")
        if not 1 <= int(self.filter_order) <= 12:
            raise ValueError("filter_order must be between 1 and 12")
        if self.notch_q <= 0:
            raise ValueError("notch_q must be positive")


@dataclass(frozen=True)
class Metrics:
    """Numeric read-outs for one analysis frame."""

    dominant_frequency_hz: float
    peak_magnitude: float            # linear amplitude of the dominant tone (1.0 = full scale)
    peak_magnitude_dbfs: float
    rms_amplitude: float             # time-domain RMS of the (filtered) frame
    rms_dbfs: float
    snr_db: float
    thd_percent: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class FrameResult:
    """Everything produced by one call to the analysis engine."""

    sample_rate: float
    time_signal: np.ndarray          # filtered frame, length fft_size
    frequencies: np.ndarray          # Hz, length fft_size // 2 + 1
    magnitude: np.ndarray            # linear amplitude spectrum (coherent-gain compensated)
    magnitude_db: np.ndarray         # dBFS
    psd: np.ndarray                  # FS^2 / Hz
    psd_db: np.ndarray               # dBFS / Hz
    metrics: Metrics
    dsp_ms: float
    scale: str = "dB"

    def display_magnitude(self) -> np.ndarray:
        """Magnitude spectrum in the scale selected in the configuration."""
        return self.magnitude_db if self.scale == "dB" else self.magnitude

    def display_psd(self) -> np.ndarray:
        """PSD in the scale selected in the configuration."""
        return self.psd_db if self.scale == "dB" else self.psd


class SignalProcessor:
    """FFT engine + Butterworth filters + metrics, with a streaming ring buffer.

    Two ways of using it:

    * stateless: ``process_frame(samples)`` analyses the last ``fft_size`` samples
      of the array it is given;
    * streaming: ``push(chunk)`` feeds consecutive chunks, ``analyze()`` analyses
      the most recent ``fft_size`` samples. The causal filter keeps its state
      between chunks, so the output is identical to filtering the whole stream
      in one go.
    """

    def __init__(self, sample_rate: float = 48000.0, config: DSPConfig | None = None) -> None:
        self._config = config if config is not None else DSPConfig()
        self._config.validate()
        self._fs = self._check_fs(sample_rate)
        self._raw = np.zeros(_RING_CAPACITY, dtype=np.float64)
        self._filtered = np.zeros(_RING_CAPACITY, dtype=np.float64)
        self._sos: np.ndarray | None = None
        self._zi: np.ndarray | None = None
        self._rebuild_tables()
        self._design_filter()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _check_fs(sample_rate: float) -> float:
        fs = float(sample_rate)
        if not math.isfinite(fs) or fs <= 0:
            raise ValueError(f"Sample rate must be a positive number, got {sample_rate!r}")
        return fs

    @property
    def config(self) -> DSPConfig:
        return self._config

    @property
    def sample_rate(self) -> float:
        return self._fs

    @property
    def nyquist(self) -> float:
        return 0.5 * self._fs

    @property
    def frequencies(self) -> np.ndarray:
        return self._freqs

    def set_sample_rate(self, sample_rate: float) -> None:
        """Change the sample rate. Clears the stream buffer (old samples no longer apply)."""
        self._fs = self._check_fs(sample_rate)
        self._rebuild_tables()
        self._design_filter()
        self.reset()

    def update_config(self, **changes) -> DSPConfig:
        """Change one or more configuration fields, e.g. ``update_config(window="Blackman")``."""
        new = replace(self._config, **changes)
        new.validate()
        old, self._config = self._config, new
        if (new.fft_size, new.window) != (old.fft_size, old.window):
            self._rebuild_tables()
        filter_fields = ("filter_type", "cutoff_hz", "cutoff_high_hz", "filter_order", "notch_q", "zero_phase")
        if any(getattr(new, f) != getattr(old, f) for f in filter_fields):
            self._design_filter()
            self._refilter_ring()
        return new

    def reset(self) -> None:
        """Clear the stream buffer and the filter state."""
        self._raw[:] = 0.0
        self._filtered[:] = 0.0
        self._reset_filter_state()

    def _rebuild_tables(self) -> None:
        n = self._config.fft_size
        self._window = sp_signal.get_window(_SCIPY_WINDOW_NAMES[self._config.window], n, fftbins=True).astype(np.float64)
        self._coherent_gain = float(self._window.mean())
        self._window_power = float(np.sum(self._window ** 2))
        self._freqs = sp_fft.rfftfreq(n, d=1.0 / self._fs)
        self._amp_scale = 2.0 / (n * self._coherent_gain)
        self._psd_scale = 1.0 / (self._fs * self._window_power)
        # Tables for the sub-bin peak search (see _locate_peak).
        self._n_int = np.arange(n, dtype=np.int64)
        self._n_float = np.arange(n, dtype=np.float64)
        self._unit_circle = np.exp((-2j * np.pi / n) * np.arange(n))
        block = max(1, n // 512)
        if n % block:
            block = 1
        self._block = block
        self._block_centres = np.arange(n // block, dtype=np.float64) * block + 0.5 * (block - 1)
        self._band_half_width = _MAIN_LOBE_HALF_WIDTH[self._config.window] + 1

    @property
    def coherent_gain(self) -> float:
        """Mean of the window; the amplitude spectrum is divided by it."""
        return self._coherent_gain

    @property
    def enbw_bins(self) -> float:
        """Equivalent noise bandwidth of the window, in bins."""
        n = self._config.fft_size
        return n * self._window_power / float(self._window.sum()) ** 2

    # ---------------------------------------------------------------- filters
    def effective_band(self) -> tuple[float, float]:
        """Band edges actually used by the filter after clamping to (0, Nyquist)."""
        cfg = self._config
        nyq = self.nyquist
        lo_limit, hi_limit = 1e-4 * nyq, 0.999 * nyq
        if cfg.filter_type == "Band-pass":
            lo = min(max(cfg.cutoff_hz, lo_limit), hi_limit)
            hi = min(max(cfg.cutoff_high_hz, lo_limit), hi_limit)
            if hi < lo:
                lo, hi = hi, lo
            if hi - lo < 1e-3 * nyq:                       # keep the band from collapsing
                centre = 0.5 * (lo + hi)
                lo = max(centre - 5e-4 * nyq, lo_limit)
                hi = min(lo + 1e-3 * nyq, hi_limit)
            return lo, hi
        if cfg.filter_type == "Notch":
            f0 = min(max(cfg.cutoff_hz, 10 * lo_limit), hi_limit)
            half = 0.5 * f0 / cfg.notch_q
            lo = max(f0 - half, lo_limit)
            hi = min(f0 + half, hi_limit)
            if hi <= lo:
                lo = max(hi - 1e-3 * nyq, 0.5 * lo_limit)
            return lo, hi
        fc = min(max(cfg.cutoff_hz, lo_limit), hi_limit)
        return fc, fc

    def _design_filter(self) -> None:
        cfg = self._config
        if cfg.filter_type == "Bypass":
            self._sos = None
        else:
            lo, hi = self.effective_band()
            order = int(cfg.filter_order)
            if cfg.filter_type == "Low-pass":
                self._sos = sp_signal.butter(order, lo, btype="lowpass", fs=self._fs, output="sos")
            elif cfg.filter_type == "High-pass":
                self._sos = sp_signal.butter(order, lo, btype="highpass", fs=self._fs, output="sos")
            elif cfg.filter_type == "Band-pass":
                self._sos = sp_signal.butter(order, [lo, hi], btype="bandpass", fs=self._fs, output="sos")
            else:  # Notch = narrow Butterworth band-stop
                self._sos = sp_signal.butter(order, [lo, hi], btype="bandstop", fs=self._fs, output="sos")
        self._reset_filter_state()

    def _reset_filter_state(self) -> None:
        self._zi = None if self._sos is None else np.zeros((self._sos.shape[0], 2), dtype=np.float64)

    def _refilter_ring(self) -> None:
        """Re-run the causal filter over the buffered raw samples after a filter change."""
        self._reset_filter_state()
        if self._sos is not None and not self._config.zero_phase:
            self._filtered[:], self._zi = sp_signal.sosfilt(self._sos, self._raw, zi=self._zi)

    def frequency_response(self, n_points: int = 1024) -> tuple[np.ndarray, np.ndarray]:
        """Filter magnitude response as (frequencies in Hz, linear gain)."""
        if self._sos is None:
            f = np.linspace(0.0, self.nyquist, n_points)
            return f, np.ones(n_points)
        f, h = sp_signal.sosfreqz(self._sos, worN=n_points, fs=self._fs)
        gain = np.abs(h)
        return f, gain * gain if self._config.zero_phase else gain

    def apply_filter(self, samples: np.ndarray) -> np.ndarray:
        """Filter a whole array (stateless). Causal or zero-phase according to the config."""
        x = np.asarray(samples, dtype=np.float64)
        if self._sos is None or x.size == 0:
            return x.copy()
        if self._config.zero_phase:
            return self._zero_phase(x)
        return sp_signal.sosfilt(self._sos, x)

    def _zero_phase(self, x: np.ndarray) -> np.ndarray:
        default_pad = 3 * (2 * self._sos.shape[0] + 1)
        return sp_signal.sosfiltfilt(self._sos, x, padlen=max(0, min(default_pad, x.size - 1)))

    # -------------------------------------------------------------- streaming
    def push(self, chunk: np.ndarray) -> None:
        """Append a chunk of new samples to the stream buffer."""
        x = np.asarray(chunk, dtype=np.float64).ravel()
        k = x.size
        if k == 0:
            return
        causal = self._sos is not None and not self._config.zero_phase
        if causal:
            y, self._zi = sp_signal.sosfilt(self._sos, x, zi=self._zi)
        cap = _RING_CAPACITY
        if k >= cap:
            self._raw[:] = x[-cap:]
            if causal:
                self._filtered[:] = y[-cap:]
            return
        self._raw[:-k] = self._raw[k:]
        self._raw[-k:] = x
        if causal:
            self._filtered[:-k] = self._filtered[k:]
            self._filtered[-k:] = y

    def analyze(self) -> FrameResult:
        """Analyse the most recent ``fft_size`` samples of the stream."""
        t0 = time.perf_counter()
        n = self._config.fft_size
        if self._sos is None:
            frame = self._raw[-n:].copy()
        elif self._config.zero_phase:
            # Filter some extra history so the start-of-block transient of the
            # forward-backward filter falls outside the analysed frame.
            extra = min(_ZERO_PHASE_HISTORY, _RING_CAPACITY - n)
            frame = self._zero_phase(self._raw[-(n + extra):])[-n:]
        else:
            frame = self._filtered[-n:].copy()
        return self._analyze_frame(frame, t0)

    def process_frame(self, samples: np.ndarray) -> FrameResult:
        """Stateless analysis of the last ``fft_size`` samples of ``samples``.

        The filter is applied to the whole array first, so passing more than
        ``fft_size`` samples keeps the causal filter's start-up transient out of
        the analysed frame. Shorter inputs are zero-padded at the front.
        """
        t0 = time.perf_counter()
        n = self._config.fft_size
        y = self.apply_filter(samples)
        if y.size >= n:
            frame = y[-n:]
        else:
            frame = np.zeros(n, dtype=np.float64)
            if y.size:
                frame[-y.size:] = y
        return self._analyze_frame(frame, t0)

    # ------------------------------------------------------------ FFT engine
    def _analyze_frame(self, frame: np.ndarray, t0: float) -> FrameResult:
        n = self._config.fft_size
        windowed = frame * self._window
        spectrum = sp_fft.rfft(windowed)
        power = spectrum.real ** 2 + spectrum.imag ** 2          # |X|^2
        raw_mag = np.sqrt(power)

        magnitude = raw_mag * self._amp_scale
        magnitude[0] *= 0.5                                      # DC has no mirrored twin
        psd = power * (2.0 * self._psd_scale)
        psd[0] *= 0.5
        if n % 2 == 0:                                           # neither has the Nyquist bin
            magnitude[-1] *= 0.5
            psd[-1] *= 0.5

        magnitude_db = 20.0 * np.log10(np.maximum(magnitude, _DB_FLOOR))
        psd_db = 10.0 * np.log10(np.maximum(psd, _POWER_FLOOR))
        metrics = self._metrics(frame, windowed, raw_mag, power)

        return FrameResult(
            sample_rate=self._fs,
            time_signal=frame,
            frequencies=self._freqs,
            magnitude=magnitude,
            magnitude_db=magnitude_db,
            psd=psd,
            psd_db=psd_db,
            metrics=metrics,
            dsp_ms=(time.perf_counter() - t0) * 1e3,
            scale=self._config.scale,
        )

    def _baseband_blocks(self, windowed: np.ndarray, k0: int) -> np.ndarray:
        """Shift integer bin ``k0`` down to DC and sum the result in short blocks.

        The mixing phasor for an integer bin is periodic in the FFT length, so
        it is read from a cached table instead of being recomputed. After the
        shift, the phase of anything within a bin of ``k0`` turns by a few
        milliradians per block, so the block sums (at most 512 of them) keep
        everything the peak search needs while making each DTFT probe about
        ``fft_size / 512`` times cheaper.
        """
        n = windowed.size
        mixed = windowed * self._unit_circle[(k0 * self._n_int) % n]
        if self._block == 1:
            return mixed
        return mixed.reshape(-1, self._block).sum(axis=1)

    def _dtft(self, blocks: np.ndarray, offsets: np.ndarray) -> np.ndarray:
        """Complex DTFT of the windowed frame at ``k0 + offsets`` bins (vectorised)."""
        phase = np.outer(offsets * (-2.0 * np.pi / self._config.fft_size), self._block_centres)
        return np.exp(1j * phase) @ blocks

    def _locate_peak(self, windowed: np.ndarray, raw_mag: np.ndarray) -> tuple[float, float, float]:
        """Sub-bin estimate of the dominant tone: (fractional bin, raw |X| there, phase of X there).

        Start from the strongest FFT bin, get a first sub-bin estimate from a
        parabola through the log magnitudes of the three bins around it, then
        polish by evaluating the DTFT itself on two successively finer 3-point
        grids. Unlike closed-form interpolators this works for every window.
        """
        n_bins = raw_mag.size
        if n_bins < 3:
            k0 = int(np.argmax(raw_mag))
            return float(k0), float(raw_mag[k0]), 0.0
        k0 = int(np.argmax(raw_mag[1:])) + 1                     # skip DC
        peak = float(raw_mag[k0])
        if peak <= 0.0 or k0 >= n_bins - 1:
            return float(k0), peak, 0.0

        a, b, c = np.log(np.maximum(raw_mag[k0 - 1:k0 + 2], 1e-300))
        curvature = a - 2.0 * b + c
        d = 0.0                                                  # offset from k0, in bins
        if curvature < 0.0:
            d = float(np.clip(0.5 * (a - c) / curvature, -0.5, 0.5))

        blocks = self._baseband_blocks(windowed, k0)
        for step in (0.1, 0.01):
            m = np.abs(self._dtft(blocks, np.array([d - step, d, d + step])))
            curvature = m[0] - 2.0 * m[1] + m[2]
            if curvature >= 0.0:
                break
            d += step * float(np.clip(0.5 * (m[0] - m[2]) / curvature, -1.0, 1.0))
        d = min(max(d, -float(k0)), n_bins - 1.0 - k0)
        at_peak = complex(self._dtft(blocks, np.array([d]))[0])
        return k0 + d, abs(at_peak), math.atan2(at_peak.imag, at_peak.real)

    def _metrics(self, frame: np.ndarray, windowed: np.ndarray, raw_mag: np.ndarray, power: np.ndarray) -> Metrics:
        n = self._config.fft_size
        n_bins = raw_mag.size
        rms = float(np.sqrt(np.mean(frame * frame)))
        k_peak, peak_raw, peak_phase = self._locate_peak(windowed, raw_mag)
        amplitude = peak_raw * self._amp_scale
        frequency = k_peak * self._fs / n

        hw = self._band_half_width

        def band(centre: float) -> slice:
            kc = int(round(centre))
            return slice(max(kc - hw, 1), min(kc + hw + 1, n_bins))

        fundamental = band(k_peak)
        signal_power = float(power[fundamental].sum())

        # Take the fitted dominant tone out in the time domain and look at what is
        # left. Without this step the tone's own window side lobes would be counted
        # as noise and distortion, capping the SNR read-out at 40-50 dB.
        if amplitude > 0.0:
            tone = amplitude * np.cos((2.0 * np.pi * k_peak / n) * self._n_float + peak_phase)
            residual = sp_fft.rfft((frame - tone) * self._window)
            residual_power = residual.real ** 2 + residual.imag ** 2
        else:
            residual_power = power

        # Power bookkeeping on the bin grid: harmonic bands, DC band, everything else = noise.
        excluded = np.zeros(n_bins, dtype=bool)
        excluded[:min(hw + 1, n_bins)] = True                   # DC and its leakage
        excluded[fundamental] = True                            # leftovers of an imperfect fit
        harmonic_power = 0.0
        harmonic_bins = 0
        for h in range(2, _MAX_HARMONICS + 1):
            centre = h * k_peak
            if centre >= n_bins - 1:
                break
            sl = band(centre)
            fresh = ~excluded[sl]
            harmonic_power += float(residual_power[sl][fresh].sum())
            harmonic_bins += int(np.count_nonzero(fresh))
            excluded[sl] = True

        # Other discrete tones (a second generator tone, mains hum, ...) are neither noise
        # nor distortion of the dominant tone: bins that stand far above their
        # neighbourhood are left out of the noise sum.
        if n_bins > 2 * _TONE_WINDOW_BINS:
            local_mean = sp_ndimage.uniform_filter1d(residual_power, _TONE_WINDOW_BINS, mode="nearest")
            tones = residual_power > _TONE_THRESHOLD * local_mean
            if tones.any():
                excluded |= sp_ndimage.maximum_filter1d(tones.view(np.uint8), 2 * hw + 1, mode="constant").view(bool)

        # Noise also lives under the excluded bands; scale the measured part up to the full band.
        noise_bins = n_bins - int(np.count_nonzero(excluded))
        noise_power = float(residual_power[~excluded].sum()) * (n_bins / noise_bins) if noise_bins > 0 else 0.0
        # Each harmonic band holds its share of that noise too; do not report it as distortion.
        if noise_bins > 0:
            harmonic_power = max(harmonic_power - noise_power * harmonic_bins / n_bins, 0.0)

        if signal_power <= _POWER_FLOOR:
            snr_db, thd = 0.0, 0.0
        else:
            snr_db = 10.0 * math.log10(signal_power / max(noise_power, signal_power * 1e-20))
            thd = 100.0 * math.sqrt(harmonic_power / signal_power)

        return Metrics(
            dominant_frequency_hz=float(frequency),
            peak_magnitude=float(amplitude),
            peak_magnitude_dbfs=20.0 * math.log10(max(amplitude, _DB_FLOOR)),
            rms_amplitude=rms,
            rms_dbfs=20.0 * math.log10(max(rms, _DB_FLOOR)),
            snr_db=float(snr_db),
            thd_percent=float(thd),
        )
