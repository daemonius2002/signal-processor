# Signal Processor v1.4.2 - User Guide

Real-time frequency analysis of WAV files, CSV files and a built-in signal generator:
FFT spectrum, PSD, Butterworth filters, numeric metrics and a live latency monitor.

## 1. Installation

Requirements: Python 3.11, 3.12 or 3.13 (64-bit) on Windows, macOS (Apple Silicon or Intel) or Linux.

Open a terminal **in the project folder** (the one that contains `main.py`).

Windows (PowerShell):

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

Linux only: Qt needs one system library that is not part of the Python packages:
`sudo apt install libxcb-cursor0` (Debian / Ubuntu).

> If the folder name contains special characters such as `&&`, put the path in quotes when you
> `cd` into it: `cd "D:\path\with && in it"`.

### Dependencies (all pinned in `requirements.txt`)

| Package | Version | Used for |
|---|---|---|
| PyQt6 | 6.12.1 | user interface |
| PyQt6-Qt6 | 6.12.0 | Qt runtime libraries for PyQt6 |
| PyQt6_sip | 13.13.1 | binding runtime for PyQt6 |
| pyqtgraph | 0.14.0 | real-time plots |
| numpy | 2.4.6 | array maths |
| scipy | 1.17.1 | FFT, window functions, Butterworth filters, WAV fallback reader |
| soundfile | 0.14.0 | WAV reading (bundles the libsndfile library) |
| pyinstaller | 6.22.3 | building the standalone executable (build only) |
| pytest | 9.2.0 | unit tests (development only) |

Nothing else is imported. `soundfile` brings `cffi`/`pycparser`, `pyinstaller` and `pytest` bring a
few small helper packages; pip installs them automatically.

## 2. Supported inputs

### WAV (`Load WAV`, Ctrl+O)

* PCM 8 / 16 / 24 / 32-bit and 32 / 64-bit float, any sample rate, any number of channels.
* The sample rate is read from the file.
* Samples are converted to 32-bit float where +/-1.0 is digital full scale (0 dBFS).
* Multi-channel files are mixed to mono by default; pick a single channel under **Source > Channel**.

### CSV (`Load CSV`, Ctrl+Shift+O) - also `.txt`, `.tsv`, `.dat`

| Layout | Example | Sample rate |
|---|---|---|
| 1 column: amplitude | `0.0131` | you are asked for it |
| 2 columns: time, amplitude | `0.000125,0.0131` | computed from the median time step |

* A header row is detected and skipped automatically. Lines starting with `#` are ignored.
* The delimiter is detected automatically: comma, semicolon, tab or spaces.
* With semicolon / tab / space delimiters a decimal comma (`0,0131`) is accepted.
* Time is in seconds unless the header names another unit: `time (ms)`, `t [us]`, `ns`.
* If the first column is only a sample counter (0, 1, 2, ...) you are asked for the sample rate.
* CSV amplitudes are used as they are (no rescaling); 1.0 is treated as full scale for dBFS.

### Live generator (`Live Generator`, Ctrl+G)

Two sine tones plus Gaussian noise at 48 kHz. Frequency and amplitude of each tone and the noise
level are set under **Live generator** in the sidebar and take effect immediately.
Turning `Live Generator` off returns to the last loaded file.

## 3. The window

**Toolbar**

| Button | Shortcut | Action |
|---|---|---|
| Load WAV | Ctrl+O | open a WAV file and stream it in a loop |
| Load CSV | Ctrl+Shift+O | open a CSV file and stream it in a loop |
| Live Generator | Ctrl+G | switch to / from the built-in generator |
| Pause / Play | Space | freeze / resume the stream (settings still apply while paused) |
| Export Metrics | Ctrl+E | save the current read-outs as JSON or CSV |

A WAV or CSV file can also be dropped anywhere on the window. The right end of the toolbar shows
the current source (name, sample rate, length) and, for multi-channel WAV files, the channel selector.

**Sidebar**

* **Analysis**
  * *FFT size* - 512 to 16384 points. Larger = finer frequency resolution (`Fs / N` per bin), longer time window.
  * *Window* - Hann (default), Hamming, Blackman, Rectangular. Levels are corrected for the window, so a tone reads the same amplitude with every window.
  * *Spectrum* - Magnitude (amplitude of tones) or PSD (power per Hz, the right view for noise).
  * *Scale* - dB (dBFS) or Linear.
  * *Frame rate* - how often new data is fetched and the display updated (10-240 fps).
* **Filter** - Bypass, Low-pass, High-pass, Band-pass, Notch (Butterworth).
  * *Cutoff / Low edge / Centre* and *High edge*: spin box or slider (logarithmic).
  * *Order*: 1-12. *Notch Q*: stop-band width = centre / Q.
  * *Zero-phase*: off = causal filter running continuously over the stream (as a real device would);
    on = forward-backward filter on each displayed frame: no phase shift, twice the attenuation in dB.
* **Live generator** - frequency and amplitude of the two tones, and the noise level.

**Plots**

* Waveform: the analysed frame (after the filter). Its header shows the window length and the FFT bin width.
* Spectrum: the pink arrow and label mark the strongest peak. Move the mouse over the plot for a
  crosshair; the header shows the cursor position and the spectrum value at that frequency.
* With a filter active, a teal vertical line marks each cutoff frequency - **drag it** to change the
  cutoff - and a thin grey curve shows the filter's response (dB scales).
* Mouse wheel zooms, drag pans, right-button drag zooms one axis. Drag the gap between the two plots to resize them.

**Read-out strip**

| Read-out | Meaning |
|---|---|
| Dominant tone | frequency of the strongest tone, interpolated between FFT bins |
| Peak magnitude | level of that tone in dBFS, amplitude below |
| RMS amplitude | RMS of the displayed frame in dBFS, amplitude below |
| Estimated SNR | power of the dominant tone / broadband noise power. Harmonics and other steady tones are not counted as noise |
| THD | harmonics 2-10 of the dominant tone relative to the fundamental, in % |
| Frame time | average DSP + render time per frame over the last 120 frames |

The frame-time bar runs from 0 to 12.5 ms: teal = DSP share, amber = render share, white line = the
10 ms budget, grey tick = the slowest frame in the window. The verdict is green (always under 10 ms),
amber (average under 10 ms, isolated spikes above) or red (average over 10 ms). The status bar repeats
`DSP ms | Render ms | Total`. The statistics restart after every settings change.

## 4. Example workflows

Create the sample files first: `python tests/generate_samples.py` (writes to `tests/samples/`).

**Measure a tone in a WAV file**

1. `Load WAV` > `tests/samples/tone_440_1200_mono.wav`.
2. Set FFT size 8192. Dominant frequency reads 440.00 Hz, peak magnitude -6.02 dBFS (amplitude 0.5).
3. Hover over the second peak: 1.2 kHz at about -12 dBFS (amplitude 0.25).

**Remove a tone with the notch filter**

1. With the same file: Filter type `Notch`, Centre `440 Hz`, Q `5`.
2. The 440 Hz peak disappears; the dominant frequency becomes 1200.00 Hz.

**Load a CSV without a time column**

1. `Load CSV` > `tests/samples/tone_440_1200_1col.csv`.
2. Enter `8000` when asked for the sample rate.

**Look at noise density**

1. `Live Generator`, set both tone amplitudes to 0 and noise to 0.01.
2. Spectrum `PSD`, Scale `dB`: the floor sits at about -84 dBFS/Hz (`0.01^2 / 24000 Hz`).

**Export**

`Export Metrics` > choose JSON or CSV. The file holds the source, all settings, the five metrics and
the latency statistics, with a time stamp and the application version.

## 4a. Sky Map - listening to pulsars and stars (v1.4.1)

`Sky Map` (Ctrl+M) replaces the control sidebar with a map of the whole sky. Click a coloured dot, or
a name in the list, and the analyser tunes to that object: the waveform shows its pulses, the
spectrum peak sits at its frequency, and (with **Sunet** ticked) the signal plays on the loudspeakers.
The texts of this panel are in Romanian, written for children.

| Object | What it is | Real frequency | Shown at |
|---|---|---|---|
| Pulsarul Crab (PSR B0531+21) | pulsar, Taurus | 29.6 turns per second | 29.6 Hz |
| Pulsarul Vela (PSR B0833-45) | pulsar, Vela | 11.19 | 11.19 Hz |
| PSR B0329+54 | pulsar, Camelopardalis | 1.3995 | 1.3995 Hz |
| Primul pulsar, LGM-1 (PSR B1919+21) | pulsar, Vulpecula | 0.7478 | 0.7478 Hz |
| Pulsarul milisecundă (PSR B1937+21) | pulsar, Vulpecula | 641.93 | 641.93 Hz |
| PSR J0437-4715 | pulsar, Pictor | 173.69 | 173.69 Hz |
| PSR B0950+08 | pulsar, Leo | 3.951 | 3.951 Hz |
| Cel mai rapid pulsar (PSR J1748-2446ad) | pulsar, Sagittarius | 716.36 | 716.36 Hz |
| Soarele | star | one vibration per 5.4 min (3090 µHz) | 309 Hz (sped up 100 000 x) |
| Alpha Centauri A | star, Centaurus | 2410 µHz | 241 Hz (sped up 100 000 x) |
| Procyon | star, Canis Minor | about 1000 µHz | 100 Hz (sped up 100 000 x) |
| Mesajul din «Contact» (Vega) | **fiction** | - | 880 Hz beeps counting 2, 3, 5, 7, 11, 13 |

**What is real and what is not.** The frequencies are published, rounded values; the Crab pulsar is
slowing down, so its figure drifts slightly over the years. The signals themselves are *synthesised*
from those values - they are not recordings from a radio telescope. Pulsars really do send radio
pulses at their spin rate, and astronomers really do play them through loudspeakers; here slow pulses
are turned into short bursts of hiss so they can be heard. The stars' vibrations are far too slow to
hear and are sped up 100 000 times. The Vega signal is invented, after the 1997 film "Contact".

The same signals are available as files: `python tests/generate_sky_samples.py` writes
`samples_stele/` with 12 WAV files (16 s each, 44.1 kHz), `catalog_stele.csv` (names, coordinates,
frequencies, stories) and `harta_celesta.png`. The WAV files play in any audio player and can be
opened with `Load WAV`.

Sound uses QtMultimedia, which is part of PyQt6. If no audio device is found, the **Sunet** box is
greyed out and everything else still works.

## 4b. Movable panels, planets, sound kit, microphone (v1.4.1)

**Panels.** Controls, Cer (sky), Planete (Solar System) and Sunete (sound kit) are dockable panels,
switched with the toolbar buttons or Ctrl+1 ... Ctrl+4. Drag a panel by its title to another edge of
the window, drop it on another panel to stack them as tabs, or pull it out as a separate window (for
a second screen). The arrangement is remembered between sessions.

**Cer.** The view selector switches between the whole sky and the northern or southern hemisphere
(as on a planisphere). New objects: four more pulsars (Geminga, the pulsar with planets, the
Hulse-Taylor pulsar, the double pulsar) and four pulsating stars (Delta Cephei, Polaris, RR Lyrae,
Mira), whose brightness cycles are sped up 100 million times.

**Planete.** The eight planets where they are today (approximate: circular orbits, mean motion), with
a time-speed slider. Distances are compressed logarithmically. Click a planet: its orbital period is
turned into a note, with one Earth year = the note A (440 Hz) and the real proportions kept. Mercury
1827 Hz, Venus 715 Hz, Earth 440 Hz, Mars 234 Hz, Jupiter 37 Hz, Saturn 15 Hz, Uranus 5.2 Hz,
Neptune 2.7 Hz. Planets do not make these sounds; it is a way of hearing how much slower the far
planets move.

**Sunete.** 36 example sounds in 7 categories: simple tones, instruments playing the same note,
music (public-domain melodies), voice, nature, everyday sounds, noises. All are synthesised by
`io_handlers/sound_kit.py` - they are not recordings; the spoken items use the eSpeak NG speech
synthesiser. The files are in `samples_sunete/` (regenerate with `python tests/generate_sound_kit.py`).
For real recordings use `Load WAV` or the microphone.

**Microfon** (Ctrl+R). Analyses the default microphone live: speak, sing, whistle, clap. Windows may
ask for microphone permission (Settings > Privacy > Microphone).

**Sunet** and the volume slider in the toolbar switch the loudspeakers for Cer, Planete and Sunete.

## 4c. LED bands, spectrogram, noise, waves and room acoustics (v1.4.1)

**View** (Controls > Analysis) switches the lower pane between three displays:

* *Spectrum (line)* - the classic trace with peak marker and crosshair.
* *1/3-octave LED bands* - 28 bands from 31.5 Hz to 16 kHz as LED columns, 3 dB per LED, with
  falling peak-hold markers, like a hardware real-time analyser. Each band shows the RMS level in
  dBFS of everything between `fc / 2^(1/6)` and `fc * 2^(1/6)`.
* *Spectrogram* - frequency across (logarithmic), time down, level as colour (-110 ... -10 dBFS).
  It shows how a sound changes, and things a single spectrum hides, such as a faint constant whistle
  at 18 kHz appearing as a thin vertical line.

**Live generator > Signal**: *Tones + noise* (now three tones), *White noise* (same power at every
frequency: rises 1 dB per band on the LED view) or *Pink noise* (3 dB less per octave: flat on the LED
view). *Play it on the loudspeakers* makes the generator audible.

**Unde** (Ctrl+5): three simple waves a, b, c and their sum d, drawn one above the other, with a
frequency and a level for each. The analyser shows the three peaks again: a complex wave is only a
sum of simple ones (Fourier, 1822; the fast algorithm, FFT, by Cooley and Tukey, 1965).

**Acustică** (Ctrl+6) - room response, as done with a real-time analyser and pink noise:

1. Put the microphone where you listen. Choose *Zgomot roz* and press *Pornește măsurarea*. The
   loudspeakers play the noise, the analyser switches to the microphone (FFT 8192 or more).
2. The chart shows each band relative to the average of the 100 Hz - 10 kHz bands, averaged over all
   frames since the start (*Reia media* restarts it). Orange = too loud, teal = too weak.
3. *Salvează CSV* writes the deviation and the equaliser correction (the deviation with the sign
   changed, limited to +/-12 dB) for each band.

The result contains the response of the loudspeakers and of the microphone too. With a laptop or
headset microphone it is indicative only; a measurement microphone is needed for real calibration.

## 5. Command line

```text
python main.py FILE                    open a .wav / .csv at start-up
python main.py data.csv --sample-rate 8000
python main.py --benchmark 600         run 600 frames, print a latency report, exit
python main.py --benchmark 600 --fft-size 16384
python main.py --benchmark 600 --report report.json
python main.py --version
```

The standalone executable accepts the same arguments. The Windows executable has no console window:
add `--report report.json` to a benchmark run to get the result as a file.

## 6. Accuracy notes and limits

* Frequency error is below 0.1 % with the Hann and Blackman windows at every FFT size, and with all
  four windows at 16384 points (covered by the unit tests). The Rectangular and Hamming windows at
  short FFT sizes can exceed 0.1 % for low tones that do not fall on a bin, because of their high
  side lobes - use Hann or Blackman for measurements.
* A tone must span at least a few periods inside the frame: use a larger FFT for low frequencies.
* SNR and THD are estimates from one frame; with the Rectangular window they are limited by leakage.
* Zero-phase filtering works on the frame being displayed, so the newest few milliseconds of each
  frame carry a small edge transient (hidden by every window except Rectangular).
* Files are streamed in a loop; the loop point can produce a click in the display once per pass.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| "Missing dependency" at start-up | activate the virtual environment and run `python -m pip install -r requirements.txt` |
| Linux: `Could not load the Qt platform plugin "xcb"` | `sudo apt install libxcb-cursor0` |
| "Could not read WAV file" | the file is not PCM / float WAV (for example MP3 renamed to .wav); convert it to WAV first |
| Latency read-out amber / red | lower the frame rate or FFT size, make the window smaller, close other heavy programs |
