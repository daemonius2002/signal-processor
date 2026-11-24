# Signal Processor v1.3.0 - Developer Notes

## 1. Layout

```text
main.py                  entry point: arguments, High-DPI policy, error dialogs, benchmark mode
requirements.txt         pinned dependencies
build.spec               PyInstaller configuration (Windows, macOS .app, Linux)
dsp/processor.py         DSPConfig, SignalProcessor, FrameResult, Metrics   (NumPy / SciPy only)
io_handlers/loader.py    load_wav, load_csv, SyntheticGenerator, StreamSimulator (no Qt)
ui/main_window.py        MainWindow, TimedPlotWidget, Readout, LatencyMeter, export_metrics (PyQt6 + PyQtGraph)
tests/generate_samples.py   writes the 440 Hz + 1200 Hz test WAV / CSV files
tests/test_dsp.py        133 pytest cases
docs/                    USER_GUIDE.md, DEVELOPER_NOTES.md
```

`dsp` and `io_handlers` do not import Qt and can be used on their own:

```python
from dsp import DSPConfig, SignalProcessor
from io_handlers import load_wav

data = load_wav("tests/samples/tone_440_1200_mono.wav")
proc = SignalProcessor(data.sample_rate, DSPConfig(fft_size=8192, window="Hann"))
result = proc.process_frame(data.samples)
print(result.metrics.dominant_frequency_hz, result.metrics.peak_magnitude_dbfs)
```

The version number lives in `dsp/__init__.py` (`__version__`), `build.spec` (`APP_VERSION`) and the
headings of the two guides.

## 2. Data flow per frame

```text
QTimer (PreciseTimer, 1000 / fps ms)
  -> StreamSimulator.chunk_for_elapsed(dt)     samples for the real time that passed (no drift)
  -> SignalProcessor.push(chunk)               ring buffer + causal sosfilt with carried state
  -> SignalProcessor.analyze()                 window, rfft, scaling, metrics      = "DSP ms"
  -> MainWindow._draw(result)                  setData on two curves, move marker  \
  -> Qt paint events of both plots             timed inside TimedPlotWidget        / "Render ms"
```

Everything runs on the GUI thread. A frame costs a few milliseconds, so a worker thread would add
hand-over latency without removing any work.

## 3. DSP details (`dsp/processor.py`)

* **Windows**: `scipy.signal.get_window(..., fftbins=True)` (periodic form), cached per FFT size.
* **Magnitude**: `2 |X[k]| / (N * CG)` with the coherent gain `CG = mean(w)`; DC and Nyquist bins
  are not doubled. A sine of amplitude A reads A (20 log10 A dBFS) with every window.
* **PSD**: `2 |X[k]|^2 / (Fs * sum(w^2))` in FS^2/Hz, single-sided; `sum(PSD) * Fs/N` equals the
  power of the windowed frame.
* **Filters**: `scipy.signal.butter(..., output="sos")`. Low-pass, high-pass, band-pass; the notch is
  a band-stop of width `f0 / Q`. Causal mode uses `sosfilt` with the state `zi` kept between chunks,
  so chunked output equals one-shot filtering of the stream (tested). Zero-phase mode uses
  `sosfiltfilt` on the frame plus 1024 samples of history. After a filter change the buffered raw
  samples are filtered again, so the display is consistent immediately.
* **Peak search**: strongest bin, log-parabolic first guess, then two rounds of 3-point parabolic
  refinement on the DTFT itself (steps 0.1 and 0.01 bin). The DTFT probes are cheap because the
  frame is first mixed down by the integer bin (phasor read from a cached table) and summed in
  blocks, leaving at most 512 values per probe.
* **SNR / THD**: the fitted dominant tone is subtracted in the time domain and the residual is
  transformed again. Harmonic bands 2-10 of the residual give THD (minus their share of noise).
  Bins standing 12x above their 129-bin neighbourhood are treated as other tones; the rest is noise,
  scaled up for the excluded bins.

### Measured accuracy (synthetic tones, random phases, Fs = 44.1 kHz)

| Window | N = 512 | N = 4096 | N = 16384 |
|---|---|---|---|
| Hann | 0.010 % | < 0.0001 % | < 0.0001 % |
| Blackman | 0.005 % | < 0.0001 % | < 0.0001 % |
| Hamming | 0.12 % | 0.001 % | 0.0002 % |
| Rectangular | 0.38 % | 0.056 % | 0.003 % |

Worst frequency error over 440, 1200, 997.3, 5000.7 and 15000.1 Hz. The Hamming and Rectangular
figures at short N come from side-lobe leakage of the negative-frequency image, not from the
interpolation; they are a property of those windows.

## 4. How the 10 ms budget is met

**DSP**

* One `scipy.fft.rfft` per frame plus one for the residual; no Python loops over samples.
* Windows, frequency axes, scaling factors, SOS coefficients and phasor tables are computed only
  when a setting changes.
* The causal filter processes only the new chunk (about `Fs / fps` samples), not the whole frame.
* Sub-bin peak search on block sums instead of full-length DTFTs (about 0.8 ms -> under 0.1 ms per frame at N = 16384).

**Rendering (PyQtGraph)**

* Axes, tick labels and grid are cached as pixmaps (`DeviceCoordinateCache`). Uncached, they were
  redrawn with text on every frame and cost about 11 ms per frame - the largest single item.
* Curves are min/max ("peak") decimated to about two points per screen pixel
  (`autoDownsampleFactor = 1`, `clipToView`), so the painter never gets more than a few thousand
  vertices; finite checks are skipped; pens are 1 px, antialiasing off.
* Auto-range is off. Y ranges are adjusted by the application only when the data leaves the range.
* The peak marker is a cached arrow that is moved, not rebuilt. Its label, all numeric read-outs and
  the frame-time bar are refreshed 10 times per second, not every frame; style sheets are reapplied
  only on colour change.
* The filter response curve (400 points) and the cutoff handles are only updated when a setting
  changes, and are hidden in Bypass.
* OpenGL is deliberately not required: PyQtGraph's OpenGL path needs the extra `PyOpenGL` package
  and a working GPU driver, which is the usual source of missing-library problems in frozen builds.
  With the optimisations above the raster painter stays inside the budget. To experiment, install
  `PyOpenGL` and add `pg.setConfigOptions(useOpenGL=True)` in `MainWindow._build_plots`.

**How latency is measured**

* DSP ms: wall time around `push()` + `analyze()`.
* Render ms: wall time of `_draw()` plus the measured duration of the paint events it causes
  (`TimedPlotWidget.paintEvent`). The paint time is real, not an estimate.
* The frame-time read-out shows the mean and maximum of the last 120 frames. The first 30 frames
  after start are not rated, and the window restarts after every settings or source change.

**Reference measurement** - `python main.py --benchmark 600 --fft-size N`, Linux, Python 3.13,
2 virtual cores (Xeon 2.3 GHz, shared), Qt offscreen software rendering, 1360 x 860 window, 60 fps:

| FFT size | DSP avg ms | Render avg ms | Total avg ms | Total p99 ms | Total max ms | Frames >= 10 ms |
|---|---|---|---|---|---|---|
| 512 | 0.40 | 1.80 | 2.20 | 3.84 | 6.23 | 0 / 600 |
| 1024 | 0.49 | 1.94 | 2.43 | 5.60 | 7.42 | 0 / 600 |
| 2048 | 0.58 | 2.69 | 3.28 | 5.47 | 6.39 | 0 / 600 |
| 4096 | 0.64 | 3.15 | 3.79 | 6.63 | 9.73 | 0 / 600 |
| 8192 | 0.92 | 3.87 | 4.79 | 8.92 | 11.91 | 1 / 600 |
| 16384 | 1.37 | 4.53 | 5.89 | 9.75 | 12.64 | 4 / 600 |

Averages are well inside the budget at every size. On that small shared machine 5 of 3600 frames
went over 10 ms (scheduling spikes, largest two FFT sizes). Run the same command on the target
machine to get its own figures; the exit code is 0 when no frame exceeded the budget and 2 otherwise.

## 5. Tests

```bash
python tests/generate_samples.py     # optional: sample files for manual use
python -m pytest -v                  # 133 tests, about 2 s
```

Covered: frequency / amplitude accuracy per window and FFT size, coherent-gain compensation, DC
handling, PSD normalisation, dB / linear scaling, RMS, SNR (white and band-limited noise, extra
tones), THD, filter attenuation (pass band, cutoff, stop band; causal and zero-phase), notch,
streaming equals one-shot filtering, generator phase continuity, stream pacing and looping, WAV
(mono, stereo, channel select, SciPy fallback formats), CSV dialects and error cases, DSP time.

The tests need no display. The GUI can be exercised without a screen with
`QT_QPA_PLATFORM=offscreen python main.py --benchmark 300`.

## 6. Building the standalone executable

PyInstaller does not cross-compile: build on each target system, in the virtual environment that has
`requirements.txt` installed.

```bash
python -m PyInstaller build.spec --noconfirm --clean
```

| System | Result | Start |
|---|---|---|
| Windows | `dist\SignalProcessor\SignalProcessor.exe` | double-click, or `dist\SignalProcessor\SignalProcessor.exe` |
| Linux | `dist/SignalProcessor/SignalProcessor` | `./dist/SignalProcessor/SignalProcessor` |
| macOS | `dist/SignalProcessor.app` | `open dist/SignalProcessor.app` |

Distribute the whole `dist/SignalProcessor` folder (or the `.app`). Check a build with
`SignalProcessor --benchmark 300 --report report.json`
(the Windows executable has no console, so the report is only visible through `--report`).

Notes

* `build.spec` bundles the libsndfile library from the `_soundfile_data` package explicitly and
  excludes other Qt bindings, Tk, matplotlib and PyOpenGL.
* macOS: the bundle has the architecture of the Python that built it (arm64 or x86_64). Unsigned
  apps must be opened once with right-click > Open, or signed with `codesign`.
* Linux: the target machine needs `libxcb-cursor0` (Qt's xcb platform plugin).
* Windows: SmartScreen may warn about an unsigned executable.
* Build status of v1.3.0: the Linux build was produced and run (`--version`, `--benchmark` with a WAV
  file). The Windows and macOS builds use the same spec but were not run by the author of this
  version.

## 7. User interface notes (v1.3.0)

* All colours are constants at the top of `ui/main_window.py`. Two of them carry meaning across the
  window: teal = time domain and DSP time, amber = frequency domain and render time.
* The style sheet is built by `build_style_sheet()`. Spin-box arrows, combo arrows and the check-box
  tick are painted at start-up into a `QTemporaryDir` (`_write_control_images`), because Qt style
  sheets only accept such glyphs as image files. No image files ship with the application.
* Toolbar icons are vector strokes drawn by `_make_icon()`; no icon files or icon fonts.
* Live numbers use tabular digits (`_tabular_font`, OpenType `tnum`, Qt 6.7+) so they do not jitter.
* The cutoff handles are `pg.InfiniteLine` items; dragging one sets the matching spin box, which runs
  the same code path as typing a value.

## 8. Sky Map (v1.3.0)

* `io_handlers/sky_catalog.py` - the catalogue (`SKY_OBJECTS`), bright stars and constellation lines
  for the map, and the synthesis: `analysis_signal(obj)` (sample rate chosen per object so its
  frequency is well resolved) and `audio_signal(obj)` (44.1 kHz, made audible). No Qt.
* `ui/sky_panel.py` - `SkyMapWidget` (static sky cached in a pixmap, only the markers repainted,
  8 times per second) and `SkyPanel` (map, list, story, sound controls).
* `ui/audio.py` - `AudioPlayer`: QtMultimedia `QAudioSink` fed by a looping `QIODevice`. It is
  skipped on the `offscreen` platform and with `SIGNAL_PROCESSOR_NO_AUDIO=1`, and reports
  `available = False` instead of raising when there is no device.
* `MainWindow._on_sky_object` loads the analysis signal as an ordinary file source, selects the FFT
  size and zooms the spectrum to about 12 harmonics. Audio runs in Qt's audio thread and is not part
  of the frame-time figures.
* `tests/generate_sky_samples.py` writes the WAV files, the catalogue and the map picture.

## 9. Windows installer (v1.3.0)

`SignalProcessor_Setup_v1.3.0.exe` was assembled on Linux, not with PyInstaller: a conda-forge
CPython 3.13 for Windows, the pinned `win_amd64` wheels unpacked into its `site-packages`, the
application sources, and an NSIS script (installs per user under
`%LOCALAPPDATA%\Programs\SignalProcessor`, no administrator rights). Unused Qt modules (QML, Quick,
Designer, Bluetooth and similar) were removed from the bundle. It could not be run on Windows from
the build machine, so its first start on Windows is the first real test. `build.spec` remains the
supported way to build on the target system itself.
