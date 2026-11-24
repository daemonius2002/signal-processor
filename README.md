# Signal Processor / Ascultă Universul

Two versions of the same idea: see what a sound is made of.

| Folder | What it is |
|---|---|
| `web/` | **Web version** (Romanian, for children): sky map with pulsars and stars, the Solar System, a sound kit and a live microphone view, with waveform and spectrum. Plain HTML + JavaScript, no build step. |
| `desktop/` | **Desktop application** v1.4.1 (Python 3.11+, PyQt6): the full real-time analyser with filters, metrics, latency monitor, LED third-octave bands, spectrogram, room-response measurement, plus the same sky, planets and sound panels. See `desktop/docs/USER_GUIDE.md`. |
| `tools/export_web.py` | Regenerates `web/data` from the desktop modules. |
| `server.js` | Tiny static server for `web/` (no dependencies). |

## Run the web version

```bash
npm start            # http://localhost:3000   (PORT environment variable is honoured)
```

Any static host works too: publish the `web/` folder. The microphone needs https (or localhost).

Link straight to an object with a hash: `/#crab`, `/#jupiter`, `/#vega`.

## Run the desktop version

```bash
cd desktop
python -m pip install -r requirements.txt
python tests/generate_sky_samples.py     # optional: WAV files of the sky objects
python main.py
```

The Windows installer is published under Releases.

## What is real and what is not

The frequencies are published, rounded values (pulsar spin rates, stellar oscillations, orbital
periods). The signals are **synthesised** from them; they are not telescope recordings. Stars and
planets are sped up so they can be heard (the factor is shown for each). The sound kit is synthesised
too; its spoken items use the eSpeak NG speech synthesiser. "Mesajul din Contact" is fiction, after
the 1997 film.
