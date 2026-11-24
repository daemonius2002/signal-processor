"""Unit tests: FFT accuracy, window compensation, PSD, filters, metrics, loaders, streaming.

Run from the project folder:  python -m pytest -v
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest
from scipy import signal as sp_signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp import FFT_SIZES, WINDOW_FUNCTIONS, DSPConfig, SignalProcessor  # noqa: E402
from io_handlers import (  # noqa: E402
    SampleRateRequiredError,
    StreamSimulator,
    SyntheticGenerator,
    load_csv,
    load_signal,
    load_wav,
)
from tests import generate_samples  # noqa: E402

FS = 44100.0


def sine(freq: float, n: int, fs: float = FS, amp: float = 1.0, phase: float = 0.0) -> np.ndarray:
    return amp * np.sin(2.0 * np.pi * freq * np.arange(n) / fs + phase)


def band_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


@pytest.fixture(scope="module")
def samples(tmp_path_factory) -> dict[str, str]:
    return generate_samples.generate_all(str(tmp_path_factory.mktemp("samples")))


# ------------------------------------------------------------ FFT accuracy
@pytest.mark.parametrize("window", ["Hann", "Blackman"])
@pytest.mark.parametrize("fft_size", FFT_SIZES)
@pytest.mark.parametrize("freq", [440.0, 1200.0, 997.3, 5000.7, 15000.1])
def test_frequency_error_below_0p1_percent(window, fft_size, freq):
    proc = SignalProcessor(FS, DSPConfig(fft_size=fft_size, window=window))
    for phase in (0.0, 1.1, 2.5):
        m = proc.process_frame(sine(freq, fft_size, amp=0.5, phase=phase)).metrics
        assert abs(m.dominant_frequency_hz - freq) / freq < 1e-3
        assert abs(m.peak_magnitude - 0.5) / 0.5 < 5e-3


@pytest.mark.parametrize("window", WINDOW_FUNCTIONS)
@pytest.mark.parametrize("freq", [440.0, 1200.0])
def test_frequency_error_all_windows_at_long_fft(window, freq):
    """All four windows meet the 0.1 % limit on the 440 / 1200 Hz test tones at 16384 points."""
    proc = SignalProcessor(FS, DSPConfig(fft_size=16384, window=window))
    for phase in (0.0, 0.7, 1.9, 3.0):
        m = proc.process_frame(sine(freq, 16384, amp=0.5, phase=phase)).metrics
        assert abs(m.dominant_frequency_hz - freq) / freq < 1e-3
        assert abs(m.peak_magnitude - 0.5) / 0.5 < 1e-2


def test_two_tones_plus_noise_finds_the_stronger_tone():
    _, x = generate_samples.make_signal(FS, 1.0)
    proc = SignalProcessor(FS, DSPConfig(fft_size=8192))
    result = proc.process_frame(x)
    assert abs(result.metrics.dominant_frequency_hz - 440.0) / 440.0 < 1e-3
    assert abs(result.metrics.peak_magnitude - 0.5) < 0.01
    # the second tone is visible at its own amplitude
    k = int(round(1200.0 / (FS / 8192)))
    assert abs(result.magnitude[k - 2:k + 3].max() - 0.25) < 0.04


@pytest.mark.parametrize("window", WINDOW_FUNCTIONS)
def test_coherent_gain_compensation(window):
    """A bin-centred full-scale sine reads 1.0 (0 dBFS) with every window."""
    n = 4096
    freq = 100 * FS / n
    proc = SignalProcessor(FS, DSPConfig(fft_size=n, window=window))
    result = proc.process_frame(sine(freq, n))
    assert result.magnitude[100] == pytest.approx(1.0, abs=1e-6)
    assert result.magnitude_db[100] == pytest.approx(0.0, abs=1e-4)
    assert result.metrics.peak_magnitude_dbfs == pytest.approx(0.0, abs=1e-3)
    assert result.frequencies[100] == pytest.approx(freq)
    assert result.frequencies[-1] == pytest.approx(FS / 2)


def test_dc_is_not_doubled():
    proc = SignalProcessor(FS, DSPConfig(fft_size=1024, window="Hann"))
    result = proc.process_frame(np.full(1024, 0.3))
    assert result.magnitude[0] == pytest.approx(0.3, rel=1e-9)


@pytest.mark.parametrize("window", WINDOW_FUNCTIONS)
def test_psd_integrates_to_signal_power(window):
    """White noise of variance s^2: PSD ~ s^2 / (fs/2) and its integral ~ s^2."""
    n = 16384
    sigma = 0.2
    x = np.random.default_rng(1).normal(0.0, sigma, n)
    proc = SignalProcessor(FS, DSPConfig(fft_size=n, window=window))
    result = proc.process_frame(x)
    df = FS / n
    assert np.sum(result.psd) * df == pytest.approx(sigma ** 2, rel=0.05)
    assert np.mean(result.psd[10:-10]) == pytest.approx(sigma ** 2 / (FS / 2), rel=0.05)


def test_scale_selection():
    proc = SignalProcessor(FS, DSPConfig(fft_size=1024, scale="Linear"))
    result = proc.process_frame(sine(1000.0, 1024))
    assert result.display_magnitude() is result.magnitude
    assert result.display_psd() is result.psd
    proc.update_config(scale="dB")
    result = proc.process_frame(sine(1000.0, 1024))
    assert result.display_magnitude() is result.magnitude_db
    assert result.display_psd() is result.psd_db
    assert np.allclose(result.magnitude_db, 20 * np.log10(np.maximum(result.magnitude, 1e-12)))


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        SignalProcessor(FS, DSPConfig(window="Kaiser"))
    proc = SignalProcessor(FS)
    with pytest.raises(ValueError):
        proc.update_config(filter_type="Comb")
    with pytest.raises(ValueError):
        SignalProcessor(0.0)


# ----------------------------------------------------------------- metrics
def test_rms_of_sine():
    proc = SignalProcessor(FS, DSPConfig(fft_size=4410))
    m = proc.process_frame(sine(1000.0, 4410, amp=0.8)).metrics      # whole number of periods
    assert m.rms_amplitude == pytest.approx(0.8 / np.sqrt(2), rel=1e-6)


@pytest.mark.parametrize("noise_std, expected_snr", [(0.1, 16.99), (0.01, 36.99), (0.001, 56.99)])
def test_snr_estimate(noise_std, expected_snr):
    n = 16384
    x = sine(1000.0, n) + np.random.default_rng(7).normal(0.0, noise_std, n)
    m = SignalProcessor(FS, DSPConfig(fft_size=n)).process_frame(x).metrics
    assert m.snr_db == pytest.approx(expected_snr, abs=1.0)


def test_snr_ignores_other_tones():
    """440 Hz (0.5) + 1200 Hz (0.25) + noise 0.01: SNR = 0.125 / 1e-4 = 31 dB, the 1200 Hz tone is not noise."""
    _, x = generate_samples.make_signal(FS, 1.0)
    m = SignalProcessor(FS, DSPConfig(fft_size=16384)).process_frame(x).metrics
    assert m.snr_db == pytest.approx(30.97, abs=1.0)
    assert m.thd_percent < 0.5


def test_snr_with_band_limited_noise():
    """Filtered (non-white) noise must still be counted as noise."""
    n = 16384
    rng = np.random.default_rng(11)
    proc = SignalProcessor(FS, DSPConfig(fft_size=n, filter_type="Low-pass", cutoff_hz=4000.0, filter_order=8))
    x = sine(1000.0, 3 * n) + rng.normal(0.0, 0.05, 3 * n)
    m = proc.process_frame(x).metrics
    expected = 10 * np.log10(0.5 / (0.05 ** 2 * 4000.0 / (FS / 2)))       # noise power left in 0-4 kHz
    assert m.snr_db == pytest.approx(expected, abs=1.5)


@pytest.mark.parametrize("window", WINDOW_FUNCTIONS)
def test_thd_of_known_distortion(window):
    """Fundamental + 10 % third harmonic + 5 % fifth harmonic -> THD = 11.18 %."""
    n = 8192
    f0 = 200 * FS / n                                 # bin-centred, so it also works unwindowed
    x = sine(f0, n) + 0.10 * sine(3 * f0, n) + 0.05 * sine(5 * f0, n)
    m = SignalProcessor(FS, DSPConfig(fft_size=n, window=window)).process_frame(x).metrics
    assert m.thd_percent == pytest.approx(100 * np.hypot(0.10, 0.05), rel=0.01)


def test_thd_of_clean_sine_is_negligible():
    n = 8192
    m = SignalProcessor(FS, DSPConfig(fft_size=n)).process_frame(sine(1000.0, n)).metrics
    assert m.thd_percent < 0.01
    assert m.snr_db > 80.0


def test_silence_does_not_crash():
    result = SignalProcessor(FS, DSPConfig(fft_size=1024)).process_frame(np.zeros(1024))
    assert np.all(np.isfinite(result.magnitude_db))
    assert result.metrics.rms_amplitude == 0.0
    assert np.isfinite(result.metrics.snr_db) and np.isfinite(result.metrics.thd_percent)


# ----------------------------------------------------------------- filters
def _filtered_tone_gain_db(proc: SignalProcessor, freq: float) -> float:
    n = 3 * 16384
    y = proc.apply_filter(sine(freq, n))
    return 20.0 * np.log10(band_rms(y[n // 3:2 * n // 3]) / (1.0 / np.sqrt(2)))   # away from both ends


@pytest.mark.parametrize("zero_phase", [False, True])
def test_low_pass_attenuation(zero_phase):
    proc = SignalProcessor(FS, DSPConfig(filter_type="Low-pass", cutoff_hz=1000.0, filter_order=4, zero_phase=zero_phase))
    assert abs(_filtered_tone_gain_db(proc, 100.0)) < 0.1                    # pass-band untouched
    at_cutoff = _filtered_tone_gain_db(proc, 1000.0)
    assert at_cutoff == pytest.approx(-6.02 if zero_phase else -3.01, abs=0.2)
    assert _filtered_tone_gain_db(proc, 8000.0) < -60.0                      # stop-band


@pytest.mark.parametrize("zero_phase", [False, True])
def test_high_pass_attenuation(zero_phase):
    proc = SignalProcessor(FS, DSPConfig(filter_type="High-pass", cutoff_hz=2000.0, filter_order=4, zero_phase=zero_phase))
    assert abs(_filtered_tone_gain_db(proc, 10000.0)) < 0.1
    assert _filtered_tone_gain_db(proc, 200.0) < -60.0


def test_band_pass_attenuation():
    proc = SignalProcessor(FS, DSPConfig(filter_type="Band-pass", cutoff_hz=800.0, cutoff_high_hz=1600.0, filter_order=4))
    assert abs(_filtered_tone_gain_db(proc, 1130.0)) < 0.2
    assert _filtered_tone_gain_db(proc, 100.0) < -60.0
    assert _filtered_tone_gain_db(proc, 10000.0) < -60.0
    # swapped edges are tolerated
    swapped = SignalProcessor(FS, DSPConfig(filter_type="Band-pass", cutoff_hz=1600.0, cutoff_high_hz=800.0))
    assert swapped.effective_band() == pytest.approx((800.0, 1600.0))


def test_notch_removes_only_the_target_tone():
    proc = SignalProcessor(FS, DSPConfig(filter_type="Notch", cutoff_hz=1200.0, notch_q=10.0, filter_order=4))
    assert _filtered_tone_gain_db(proc, 1200.0) < -60.0
    assert abs(_filtered_tone_gain_db(proc, 440.0)) < 0.1
    assert abs(_filtered_tone_gain_db(proc, 3000.0)) < 0.1


def test_notch_changes_the_dominant_frequency():
    """440 Hz (0.5) + 1200 Hz (0.25): notching out 440 Hz leaves 1200 Hz as the peak."""
    _, x = generate_samples.make_signal(FS, 2.0)
    proc = SignalProcessor(FS, DSPConfig(fft_size=8192, filter_type="Notch", cutoff_hz=440.0, notch_q=5.0))
    m = proc.process_frame(x).metrics
    assert abs(m.dominant_frequency_hz - 1200.0) / 1200.0 < 1e-3
    assert m.peak_magnitude == pytest.approx(0.25, abs=0.01)


def test_bypass_returns_input_unchanged():
    x = np.random.default_rng(3).normal(size=5000)
    proc = SignalProcessor(FS, DSPConfig(filter_type="Bypass"))
    assert np.array_equal(proc.apply_filter(x), x)


def test_zero_phase_has_no_delay_and_causal_has():
    n = 8192
    pulse = np.zeros(n)
    pulse[n // 2] = 1.0
    proc = SignalProcessor(FS, DSPConfig(filter_type="Low-pass", cutoff_hz=500.0, filter_order=4, zero_phase=True))
    assert int(np.argmax(proc.apply_filter(pulse))) == n // 2
    proc.update_config(zero_phase=False)
    assert int(np.argmax(proc.apply_filter(pulse))) > n // 2 + 10


def test_frequency_response_matches_design():
    proc = SignalProcessor(FS, DSPConfig(filter_type="Low-pass", cutoff_hz=1000.0, filter_order=6))
    f, gain = proc.frequency_response(4096)
    assert gain[0] == pytest.approx(1.0, abs=1e-6)
    assert gain[np.argmin(np.abs(f - 1000.0))] == pytest.approx(1 / np.sqrt(2), abs=0.01)


# --------------------------------------------------------------- streaming
def test_streaming_filter_equals_one_shot_filter():
    """Chunked causal filtering (state carried across chunks) == filtering the whole stream."""
    x = np.random.default_rng(5).normal(size=20000)
    cfg = DSPConfig(fft_size=4096, filter_type="Band-pass", cutoff_hz=500.0, cutoff_high_hz=3000.0)
    proc = SignalProcessor(FS, cfg)
    for start in range(0, x.size, 733):
        proc.push(x[start:start + 733])
    sos = sp_signal.butter(4, [500.0, 3000.0], btype="bandpass", fs=FS, output="sos")
    expected = sp_signal.sosfilt(sos, x)[-4096:]
    assert np.allclose(proc.analyze().time_signal, expected, atol=1e-10)


def test_streaming_matches_stateless_analysis():
    x = sine(997.0, 30000, amp=0.4) + np.random.default_rng(6).normal(0, 0.01, 30000)
    proc = SignalProcessor(FS, DSPConfig(fft_size=8192))
    for start in range(0, x.size, 1000):
        proc.push(x[start:start + 1000])
    streamed = proc.analyze()
    direct = SignalProcessor(FS, DSPConfig(fft_size=8192)).process_frame(x)
    assert np.allclose(streamed.magnitude, direct.magnitude)
    assert streamed.metrics.dominant_frequency_hz == pytest.approx(direct.metrics.dominant_frequency_hz)


def test_filter_change_reprocesses_buffer():
    x = sine(440.0, 40000, amp=0.5) + sine(6000.0, 40000, amp=0.5)
    proc = SignalProcessor(FS, DSPConfig(fft_size=4096))
    proc.push(x)
    proc.update_config(filter_type="High-pass", cutoff_hz=3000.0)       # no new samples pushed
    m = proc.analyze().metrics
    assert abs(m.dominant_frequency_hz - 6000.0) / 6000.0 < 1e-3


def test_generator_is_phase_continuous_and_accurate():
    gen = SyntheticGenerator(48000.0, tones=[(440.0, 0.5), (1200.0, 0.25)], noise_std=0.0)
    chunks = [gen.generate(n) for n in (800, 1, 799, 123, 5000, 8192)]
    assert all(c.dtype == np.float32 for c in chunks)
    joined = np.concatenate(chunks)
    t = np.arange(joined.size) / 48000.0
    expected = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 1200 * t)
    assert np.allclose(joined, expected, atol=1e-5)
    with pytest.raises(ValueError):
        gen.set_tones([(30000.0, 0.1)])                                 # above Nyquist


def test_generator_noise_level():
    gen = SyntheticGenerator(48000.0, tones=[], noise_std=0.05, seed=1)
    assert np.std(gen.generate(100000)) == pytest.approx(0.05, rel=0.02)


def test_stream_simulator_generator_frames():
    stream = StreamSimulator(SyntheticGenerator(48000.0), frame_rate=60.0)
    assert stream.is_live and stream.chunk_size == 800
    assert stream.next_chunk().size == 800
    stream.set_frame_rate(100.0)
    assert stream.next_chunk().size == 480


def test_stream_simulator_file_looping_and_end(samples):
    data = load_wav(samples["wav_mono"])
    stream = StreamSimulator(data, frame_rate=50.0)
    assert stream.chunk_size == 882
    first = stream.next_chunk()
    assert np.array_equal(first, data.samples[:882])
    stream.read(data.num_samples - 882 - 10)
    wrapped = stream.read(20)                                           # crosses the end, loops
    assert np.array_equal(wrapped, np.concatenate([data.samples[-10:], data.samples[:10]]))
    assert 0.0 <= stream.progress < 1.0

    once = StreamSimulator(data, frame_rate=50.0, loop=False)
    once.read(data.num_samples - 5)
    assert once.read(100).size == 5 and once.finished and once.read(100).size == 0
    once.reset()
    assert once.position == 0 and not once.finished


def test_stream_simulator_real_time_pacing():
    stream = StreamSimulator(SyntheticGenerator(44100.0), frame_rate=60.0)
    total = sum(stream.chunk_for_elapsed(1.0 / 60.0).size for _ in range(600))    # 10 s of frames
    assert abs(total - 441000) <= 1                                     # no drift from rounding
    assert stream.chunk_for_elapsed(5.0).size == 4 * stream.chunk_size  # stall: capped, no pile-up


# ----------------------------------------------------------------- loaders
def test_wav_loader_mono(samples):
    data = load_wav(samples["wav_mono"])
    assert data.sample_rate == 44100.0
    assert data.samples.dtype == np.float32 and data.samples.ndim == 1
    assert data.num_channels == 1
    assert data.duration == pytest.approx(3.0)
    assert 0.7 < np.max(np.abs(data.samples)) <= 1.0                    # normalised to +/-1
    m = SignalProcessor(data.sample_rate, DSPConfig(fft_size=16384)).process_frame(data.samples).metrics
    assert abs(m.dominant_frequency_hz - 440.0) / 440.0 < 1e-3
    assert m.peak_magnitude == pytest.approx(0.5, abs=0.005)


def test_wav_loader_stereo_mix_and_channel_select(samples):
    mix = load_wav(samples["wav_stereo"])
    assert mix.num_channels == 2 and mix.channel is None
    left = load_wav(samples["wav_stereo"], channel=0)
    right = mix.select_channel(1)
    assert np.allclose(mix.samples, 0.5 * (left.samples + right.samples), atol=1e-6)
    proc = SignalProcessor(mix.sample_rate, DSPConfig(fft_size=16384))
    assert proc.process_frame(left.samples).metrics.dominant_frequency_hz == pytest.approx(440.0, rel=1e-3)
    assert proc.process_frame(right.samples).metrics.dominant_frequency_hz == pytest.approx(1200.0, rel=1e-3)
    with pytest.raises(ValueError):
        load_wav(samples["wav_stereo"], channel=2)


def test_wav_loader_scipy_fallback_formats(tmp_path, monkeypatch):
    """Integer PCM normalisation used when soundfile is not available."""
    from scipy.io import wavfile

    monkeypatch.setitem(sys.modules, "soundfile", None)                 # makes "import soundfile" fail
    x = sine(1000.0, 8000, fs=8000.0, amp=0.5)
    for dtype, scale, tol in ((np.int16, 32767.0, 1e-4), (np.int32, 2147483647.0, 1e-6), (np.float32, 1.0, 1e-7)):
        path = str(tmp_path / f"t_{np.dtype(dtype).name}.wav")
        wavfile.write(path, 8000, (x * scale).astype(dtype))
        data = load_wav(path)
        assert data.sample_rate == 8000.0 and data.samples.dtype == np.float32
        assert np.allclose(data.samples, x, atol=tol)
    path = str(tmp_path / "t_u8.wav")
    wavfile.write(path, 8000, np.round(x * 127 + 128).astype(np.uint8))
    assert np.allclose(load_wav(path).samples, x, atol=0.01)


def test_csv_two_columns_with_header(samples):
    data = load_csv(samples["csv_2col"])
    assert data.sample_rate == pytest.approx(8000.0, rel=1e-6)
    assert data.num_samples == 16000 and data.samples.dtype == np.float32
    m = SignalProcessor(data.sample_rate, DSPConfig(fft_size=8192)).process_frame(data.samples).metrics
    assert abs(m.dominant_frequency_hz - 440.0) / 440.0 < 1e-3


def test_csv_one_column_needs_sample_rate(samples):
    with pytest.raises(SampleRateRequiredError):
        load_csv(samples["csv_1col"])
    data = load_csv(samples["csv_1col"], sample_rate=8000.0)
    assert data.sample_rate == 8000.0 and data.num_samples == 16000
    m = SignalProcessor(8000.0, DSPConfig(fft_size=8192)).process_frame(data.samples).metrics
    assert abs(m.dominant_frequency_hz - 440.0) / 440.0 < 1e-3


def test_csv_dialects(tmp_path):
    t = np.arange(200) / 1000.0
    x = np.sin(2 * np.pi * 50 * t)

    semicolon = tmp_path / "semicolon_decimal_comma.csv"                # European Excel export
    semicolon.write_text("Timp (s);Amplitudine\n" + "\n".join(f"{a:.6f};{b:.6f}".replace(".", ",") for a, b in zip(t, x)), encoding="utf-8")
    data = load_csv(str(semicolon))
    assert data.sample_rate == pytest.approx(1000.0) and np.allclose(data.samples, x, atol=1e-5)

    tabs = tmp_path / "tabs_no_header.txt"
    tabs.write_text("\n".join(f"{a:.6f}\t{b:.6f}" for a, b in zip(t, x)))
    assert load_csv(str(tabs)).sample_rate == pytest.approx(1000.0)

    millis = tmp_path / "milliseconds.csv"
    millis.write_text("# exported by a scope\ntime (ms),ch1\n" + "\n".join(f"{1000 * a:.3f},{b:.6f}" for a, b in zip(t, x)))
    assert load_csv(str(millis)).sample_rate == pytest.approx(1000.0)

    counter = tmp_path / "index.csv"
    counter.write_text("n,value\n" + "\n".join(f"{i},{b:.6f}" for i, b in enumerate(x)))
    with pytest.raises(SampleRateRequiredError):
        load_csv(str(counter))
    assert load_csv(str(counter), sample_rate=1000.0).sample_rate == 1000.0

    one = tmp_path / "one_col_header.csv"
    one.write_text("amplitude\n" + "\n".join(f"{b:.6f}" for b in x))
    assert np.allclose(load_csv(str(one), sample_rate=500.0).samples, x, atol=1e-5)


def test_csv_errors(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError):
        load_csv(str(empty))
    text = tmp_path / "text.csv"
    text.write_text("a,b\nfoo,bar\n")
    with pytest.raises(ValueError):
        load_csv(str(text))
    with pytest.raises(FileNotFoundError):
        load_csv(str(tmp_path / "missing.csv"))
    with pytest.raises(ValueError):
        load_signal(str(tmp_path / "file.mp3"))


def test_load_signal_dispatch(samples):
    assert load_signal(samples["wav_mono"]).sample_rate == 44100.0
    assert load_signal(samples["csv_2col"]).sample_rate == pytest.approx(8000.0)
    assert load_signal(samples["csv_1col"], sample_rate=8000.0).num_samples == 16000


# ----------------------------------------------------------------- latency
@pytest.mark.parametrize("zero_phase", [False, True])
def test_dsp_frame_time_is_far_below_10_ms(zero_phase):
    """Worst case: largest FFT, band-pass filter, 60 fps chunks. Median must stay under 5 ms."""
    cfg = DSPConfig(fft_size=16384, filter_type="Band-pass", cutoff_hz=300.0, cutoff_high_hz=5000.0, zero_phase=zero_phase)
    proc = SignalProcessor(48000.0, cfg)
    gen = SyntheticGenerator(48000.0)
    times = []
    for _ in range(100):
        chunk = gen.generate(800)
        t0 = time.perf_counter()
        proc.push(chunk)
        proc.analyze()
        times.append((time.perf_counter() - t0) * 1e3)
    assert float(np.median(times)) < 5.0


# --------------------------------------------------------------- sky objects
def test_sky_catalogue_is_consistent():
    from io_handlers import sky_catalog as sky

    keys = [obj.key for obj in sky.SKY_OBJECTS]
    assert len(keys) == len(set(keys)) >= 10
    star_names = {name for name, *_ in sky.BRIGHT_STARS}
    for _label, _at, names in sky.CONSTELLATION_LINES:
        assert set(names) <= star_names
    for obj in sky.SKY_OBJECTS:
        assert obj.fft_size in FFT_SIZES and obj.story_ro and sky.frequency_text(obj)
        if obj.ra_hours is not None:
            assert 0.0 <= obj.ra_hours < 24.0 and -90.0 <= obj.dec_deg <= 90.0
    ra, dec = sky.galactic_plane(73)
    assert np.all((ra >= 0) & (ra < 24)) and np.max(np.abs(dec)) < 63.0           # plane is tilted 62.9 deg


@pytest.mark.parametrize("key", ["crab", "vela", "b0329", "lgm1", "b1937", "j0437", "b0950", "fastest", "sun", "alfcen", "procyon"])
def test_sky_object_shows_its_real_frequency(key):
    """The analyser's dominant tone is the pulsar's spin rate (or the star's sped-up vibration)."""
    from io_handlers import sky_catalog as sky

    obj = sky.SKY_BY_KEY[key]
    data = sky.analysis_signal(obj)
    assert data.samples.dtype == np.float32 and np.max(np.abs(data.samples)) <= 1.0
    proc = SignalProcessor(data.sample_rate, DSPConfig(fft_size=obj.fft_size))
    tolerance = 2e-3 if obj.kind == "pulsar" else 5e-3
    for start in np.linspace(0, data.num_samples - obj.fft_size, 6).astype(int):
        shown = proc.process_frame(data.samples[start:start + obj.fft_size]).metrics.dominant_frequency_hz
        assert abs(shown - obj.display_frequency_hz) / obj.display_frequency_hz < tolerance


def test_sky_audio_and_prime_message():
    from io_handlers import sky_catalog as sky

    for obj in sky.SKY_OBJECTS:
        audio = sky.audio_signal(obj, seconds=2.0)
        assert audio.dtype == np.float32 and audio.size == 2 * sky.AUDIO_SAMPLE_RATE
        assert 0.3 < float(np.max(np.abs(audio))) <= 1.0
    # the film signal: bursts of beeps counted as 2, 3, 5, 7
    fs = 8000.0
    message = sky._prime_message(int(fs * 12), fs, 880.0)
    loud = (np.abs(message) > 0.5).astype(int)
    beeps_start = np.flatnonzero(np.diff((np.convolve(loud, np.ones(80), "same") > 0).astype(int)) == 1) / fs
    groups, count = [], 1
    for gap in np.diff(beeps_start):
        if gap > 0.5:
            groups.append(count)
            count = 1
        else:
            count += 1
    assert groups[:4] == [2, 3, 5, 7]


# ------------------------------------------- third-octave bands, noise, room
def test_third_octave_band_of_a_tone():
    from dsp import THIRD_OCTAVE_CENTRES, third_octave_levels

    fs, n = 48000.0, 16384
    result = SignalProcessor(fs, DSPConfig(fft_size=n)).process_frame(sine(1000.0, n, fs=fs, amp=0.5))
    levels = third_octave_levels(result.psd, fs)
    assert len(levels) == len(THIRD_OCTAVE_CENTRES) == 28
    band = THIRD_OCTAVE_CENTRES.index(1000.0)
    assert levels[band] == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.1)      # RMS level of the tone
    assert np.nanmax(np.delete(levels, band)) < levels[band] - 60.0
    low_rate = third_octave_levels(SignalProcessor(8000.0, DSPConfig(fft_size=4096)).process_frame(sine(500.0, 4096, fs=8000.0)).psd, 8000.0)
    assert np.isnan(low_rate[-1]) and np.isfinite(low_rate[THIRD_OCTAVE_CENTRES.index(500.0)])


@pytest.mark.parametrize("mode, slope", [("white", 1.0), ("pink", 0.0)])
def test_generator_noise_colours(mode, slope):
    """White noise rises 1 dB per third-octave band; pink noise is flat (-3 dB per octave in PSD)."""
    from dsp import third_octave_levels

    fs, n = 48000.0, 16384
    gen = SyntheticGenerator(fs, seed=4)
    gen.set_mode(mode)
    proc = SignalProcessor(fs, DSPConfig(fft_size=n))
    power = sum(10 ** (third_octave_levels(proc.process_frame(gen.generate(n)).psd, fs) / 10) for _ in range(40))
    levels = 10 * np.log10(power / 40)
    fitted = np.polyfit(np.arange(5, 26), levels[5:26], 1)[0]                 # 100 Hz ... 10 kHz
    assert fitted == pytest.approx(slope, abs=0.08)
    assert 0.05 < float(np.std(gen.generate(48000))) < 0.2                     # about -20 dBFS
    with pytest.raises(ValueError):
        gen.set_mode("brown")


def test_room_deviation_and_test_signals():
    pytest.importorskip("PyQt6")
    from dsp import THIRD_OCTAVE_CENTRES
    from ui.lab_panels import band_deviation, noise_buffer, tone_buffer

    centres = np.array(THIRD_OCTAVE_CENTRES)
    flat = np.full(28, 1e-4)
    assert np.allclose(band_deviation(flat, "pink"), 0.0)
    assert np.allclose(band_deviation(flat * centres / 1000.0, "white"), 0.0, atol=1e-9)      # white: +1 dB per band removed
    bumped = flat.copy()
    bumped[list(THIRD_OCTAVE_CENTRES).index(125.0)] *= 10 ** 0.6                                # +6 dB at 125 Hz
    deviation = band_deviation(bumped, "pink")
    assert deviation[list(THIRD_OCTAVE_CENTRES).index(125.0)] == pytest.approx(6.0, abs=0.4)
    assert np.all(np.isnan(band_deviation(np.zeros(28))))
    for colour in ("white", "pink"):
        noise = noise_buffer(colour, seconds=2.0)
        assert noise.dtype == np.float32 and float(np.sqrt(np.mean(noise.astype(float) ** 2))) == pytest.approx(0.1, rel=0.05)
    loop = tone_buffer([(440.4, 0.5), (1200.0, 0.25)])
    assert loop.size == 44100 and abs(float(loop[0])) < 1e-6                                    # whole periods: seamless loop
