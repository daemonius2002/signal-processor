"""Sound kit for children: synthesised examples of tones, instruments, music, nature and everyday sounds.

Every sound here is computed, not recorded. Melodies are in the public domain.
``tests/generate_sound_kit.py`` writes them as WAV files with a ``catalog.json``;
the application's "Sunete" panel plays those files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import signal as sp_signal

FS = 44100
NOTE = {"C4": 261.63, "D4": 293.66, "E4": 329.63, "F4": 349.23, "G4": 392.00, "A4": 440.00, "B4": 493.88,
        "C5": 523.25, "D5": 587.33, "E5": 659.25, "F5": 698.46, "G5": 783.99, "A3": 220.00, "C3": 130.81, "G3": 196.00}


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(FS * seconds)) / FS


def _fade(x: np.ndarray, ms: float = 15.0) -> np.ndarray:
    n = min(int(FS * ms / 1000.0), x.size // 2)
    if n > 0:
        x[:n] *= np.linspace(0.0, 1.0, n)
        x[-n:] *= np.linspace(1.0, 0.0, n)
    return x


def _norm(x: np.ndarray, peak: float = 0.8) -> np.ndarray:
    top = float(np.max(np.abs(x)))
    return (x * (peak / top) if top > 0 else x).astype(np.float32)


def _band(noise: np.ndarray, lo: float, hi: float, order: int = 2) -> np.ndarray:
    sos = sp_signal.butter(order, [lo, hi], btype="bandpass", fs=FS, output="sos")
    return sp_signal.sosfilt(sos, noise)


def _lowpass(x: np.ndarray, cutoff: float, order: int = 2) -> np.ndarray:
    return sp_signal.sosfilt(sp_signal.butter(order, cutoff, btype="lowpass", fs=FS, output="sos"), x)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# ------------------------------------------------------------- instruments
def piano(freq: float, seconds: float = 1.6) -> np.ndarray:
    t = _t(seconds)
    out = np.zeros(t.size)
    for k in range(1, 9):
        partial = freq * k * np.sqrt(1.0 + 0.0004 * k * k)          # stiff strings: slightly sharp partials
        if partial < 0.45 * FS:
            out += (1.0 / k ** 1.3) * np.exp(-t * (1.6 + 0.55 * k)) * np.sin(2 * np.pi * partial * t)
    out *= 1.0 - np.exp(-t * 400.0)
    return _fade(out, 5.0)


def guitar(freq: float, seconds: float = 2.0, seed: int = 1) -> np.ndarray:
    """Plucked string (Karplus-Strong): a burst of noise circulating in a delay line."""
    n = int(FS * seconds)
    period = int(round(FS / freq))
    line = _rng(seed).uniform(-1.0, 1.0, period)
    out = np.empty(n)
    for start in range(0, n, period):
        size = min(period, n - start)
        out[start:start + size] = line[:size]
        line = 0.4985 * (line + np.roll(line, -1))
    return _fade(out, 3.0)


def flute(freq: float, seconds: float = 1.6, seed: int = 2) -> np.ndarray:
    t = _t(seconds)
    vibrato = 1.0 + 0.004 * np.sin(2 * np.pi * 5.0 * t)
    phase = 2 * np.pi * freq * np.cumsum(vibrato) / FS
    tone = np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.08 * np.sin(3 * phase)
    breath = 0.05 * _band(_rng(seed).standard_normal(t.size), 1500.0, 6000.0)
    envelope = np.minimum(1.0, t / 0.08) * np.minimum(1.0, (seconds - t) / 0.12)
    return (tone + breath) * envelope


def violin(freq: float, seconds: float = 1.8) -> np.ndarray:
    t = _t(seconds)
    vibrato = 1.0 + 0.006 * np.sin(2 * np.pi * 5.5 * t) * np.minimum(1.0, t / 0.4)
    phase = 2 * np.pi * freq * np.cumsum(vibrato) / FS
    out = np.zeros(t.size)
    for k in range(1, 16):
        if freq * k < 0.45 * FS:
            body = 1.0 + 1.5 * np.exp(-0.5 * ((freq * k - 2800.0) / 900.0) ** 2)     # resonance of the body
            out += (body / k) * np.sin(k * phase)
    envelope = np.minimum(1.0, t / 0.12) * np.minimum(1.0, (seconds - t) / 0.15)
    return out * envelope


def bell(freq: float = 660.0, seconds: float = 3.5) -> np.ndarray:
    t = _t(seconds)
    out = np.zeros(t.size)
    for ratio, amp, decay in ((0.5, 0.6, 0.6), (1.0, 1.0, 0.9), (1.19, 0.7, 1.1), (1.5, 0.5, 1.4), (2.0, 0.6, 1.8),
                              (2.51, 0.35, 2.4), (3.01, 0.25, 3.0), (4.1, 0.15, 4.0)):
        out += amp * np.exp(-t * decay) * np.sin(2 * np.pi * freq * ratio * t)
    return _fade(out, 2.0)


def drum_beat(seconds: float = 6.0, seed: int = 3) -> np.ndarray:
    n = int(FS * seconds)
    out = np.zeros(n)
    rng = _rng(seed)
    step = 0.25
    pattern = "k.h.s.h.k.khs.h."
    for index in range(int(seconds / step)):
        hit = pattern[index % len(pattern)]
        start = int(index * step * FS)
        if hit == "k":
            t = _t(0.35)
            sound = np.sin(2 * np.pi * (55.0 * t + 60.0 * (1 - np.exp(-t * 28.0)) / 28.0)) * np.exp(-t * 9.0)
        elif hit == "s":
            t = _t(0.22)
            sound = 0.7 * _band(rng.standard_normal(t.size), 900.0, 7000.0) * np.exp(-t * 22.0) + 0.5 * np.sin(2 * np.pi * 190.0 * t) * np.exp(-t * 30.0)
        elif hit == "h":
            t = _t(0.07)
            sound = 0.35 * _band(rng.standard_normal(t.size), 6000.0, 14000.0) * np.exp(-t * 70.0)
        else:
            continue
        end = min(start + sound.size, n)
        out[start:end] += sound[:end - start]
    return out


def melody(notes: list[tuple[str, float]], instrument: Callable[..., np.ndarray], beat: float = 0.42) -> np.ndarray:
    total = sum(length for _, length in notes) * beat + 1.5
    out = np.zeros(int(FS * total))
    position = 0.0
    for name, length in notes:
        if name != "-":
            sound = instrument(NOTE[name], max(length * beat * 1.6, 0.5))
            start = int(position * FS)
            end = min(start + sound.size, out.size)
            out[start:end] += sound[:end - start]
        position += length * beat
    return out


TWINKLE = [("C4", 1), ("C4", 1), ("G4", 1), ("G4", 1), ("A4", 1), ("A4", 1), ("G4", 2),
           ("F4", 1), ("F4", 1), ("E4", 1), ("E4", 1), ("D4", 1), ("D4", 1), ("C4", 2)]
ODE_TO_JOY = [("E4", 1), ("E4", 1), ("F4", 1), ("G4", 1), ("G4", 1), ("F4", 1), ("E4", 1), ("D4", 1),
              ("C4", 1), ("C4", 1), ("D4", 1), ("E4", 1), ("E4", 1.5), ("D4", 0.5), ("D4", 2)]


def chord(names: tuple[str, ...], seconds: float = 2.4) -> np.ndarray:
    return sum(piano(NOTE[name], seconds) for name in names)


# ------------------------------------------------------------------ makers
def pure_tone() -> np.ndarray:
    return _fade(np.sin(2 * np.pi * 440.0 * _t(5.0)), 30.0)


def low_and_high() -> np.ndarray:
    parts = [_fade(np.sin(2 * np.pi * f * _t(1.6)), 30.0) for f in (110.0, 440.0, 1760.0, 7040.0)]
    gap = np.zeros(int(0.3 * FS))
    return np.concatenate([np.concatenate([p, gap]) for p in parts])


def scale() -> np.ndarray:
    return melody([(n, 1) for n in ("C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5")], lambda f, s: _fade(np.sin(2 * np.pi * f * _t(0.4)), 20.0), beat=0.5)


def sweep() -> np.ndarray:
    t = _t(10.0)
    return _fade(sp_signal.chirp(t, f0=40.0, t1=10.0, f1=12000.0, method="logarithmic"), 30.0)


def beats() -> np.ndarray:
    t = _t(7.0)
    return _fade(0.5 * (np.sin(2 * np.pi * 440.0 * t) + np.sin(2 * np.pi * 443.0 * t)), 30.0)


def loud_and_quiet() -> np.ndarray:
    t = _t(8.0)
    return _fade((0.08 + 0.92 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 4.0))) * np.sin(2 * np.pi * 330.0 * t), 30.0)


def major_minor() -> np.ndarray:
    gap = np.zeros(int(0.4 * FS))
    return np.concatenate([chord(("C4", "E4", "G4")), gap, chord(("C4", "E4", "G4")), gap,
                           chord(("A3", "C4", "E4")), gap, chord(("A3", "C4", "E4"))])


def vowels_sung() -> np.ndarray:
    """Vowels A E I O U from a buzzing source shaped by two or three resonances (formants)."""
    formants = {"a": (800, 1200, 2500), "e": (500, 1900, 2500), "i": (300, 2300, 3000), "o": (500, 900, 2400), "u": (320, 800, 2300)}
    out = []
    for vowel in "aeiou":
        t = _t(1.1)
        pitch = 150.0 * (1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * t))
        source = sp_signal.sawtooth(2 * np.pi * np.cumsum(pitch) / FS)
        voiced = sum(_band(source, f * 0.88, f * 1.12, 1) * w for f, w in zip(formants[vowel], (1.0, 0.6, 0.3)))
        out += [_fade(voiced / np.max(np.abs(voiced)), 60.0), np.zeros(int(0.25 * FS))]
    return np.concatenate(out)


def rain(seed: int = 10) -> np.ndarray:
    rng = _rng(seed)
    n = int(FS * 8.0)
    hiss = 0.35 * _band(rng.standard_normal(n), 1500.0, 9000.0)
    drops = np.zeros(n)
    for start in rng.integers(0, n - 400, 900):
        length = int(rng.integers(60, 300))
        drops[start:start + length] += rng.uniform(0.2, 1.0) * rng.standard_normal(length) * np.exp(-np.arange(length) / (length / 5.0))
    return _fade(hiss + 0.5 * _band(drops, 2500.0, 12000.0), 300.0)


def wind(seed: int = 11) -> np.ndarray:
    rng = _rng(seed)
    n = int(FS * 9.0)
    t = np.arange(n) / FS
    gust = 0.55 + 0.45 * np.sin(2 * np.pi * 0.13 * t) * np.sin(2 * np.pi * 0.31 * t + 1.0)
    noise = rng.standard_normal(n)
    low = _band(noise, 120.0, 600.0)
    whistle = _band(noise, 700.0, 1100.0, 1) * (0.3 + 0.7 * np.clip(gust, 0, 1) ** 3)
    return _fade((low + 0.8 * whistle) * gust, 500.0)


def waves(seed: int = 12) -> np.ndarray:
    rng = _rng(seed)
    n = int(FS * 10.0)
    t = np.arange(n) / FS
    swell = np.clip(np.sin(2 * np.pi * t / 5.0 - 1.0), 0, 1) ** 1.5 + 0.12
    return _fade(_lowpass(rng.standard_normal(n), 2500.0) * swell + 0.3 * _band(rng.standard_normal(n), 3000.0, 9000.0) * swell ** 3, 500.0)


def thunder(seed: int = 13) -> np.ndarray:
    rng = _rng(seed)
    n = int(FS * 7.0)
    t = np.arange(n) / FS
    rumble = _lowpass(rng.standard_normal(n), 140.0, 4) * (np.exp(-np.maximum(t - 0.6, 0) * 0.7) * (t > 0.55))
    crack = _band(rng.standard_normal(n), 300.0, 5000.0) * np.exp(-np.abs(t - 0.6) * 18.0)
    rolls = 1.0 + 0.6 * np.sin(2 * np.pi * 1.7 * t) * np.sin(2 * np.pi * 0.6 * t)
    return _fade(4.0 * rumble * rolls + 0.8 * crack, 50.0)


def birdsong(seed: int = 14) -> np.ndarray:
    rng = _rng(seed)
    n = int(FS * 7.0)
    out = np.zeros(n)
    position = 0.3
    while position < 6.4:
        for _ in range(int(rng.integers(2, 6))):
            length = rng.uniform(0.07, 0.16)
            t = _t(length)
            f0, f1 = rng.uniform(2500, 4200), rng.uniform(3200, 6000)
            freq = f0 + (f1 - f0) * (t / length) ** rng.uniform(0.5, 2.0)
            chirp = np.sin(2 * np.pi * np.cumsum(freq) / FS) * np.sin(np.pi * t / length) ** 2
            start = int(position * FS)
            out[start:start + chirp.size] += chirp[:max(0, n - start)]
            position += length + rng.uniform(0.02, 0.07)
        position += rng.uniform(0.35, 0.9)
    return out


def cricket() -> np.ndarray:
    t = _t(6.0)
    pulse = (np.sin(2 * np.pi * 62.0 * t) > 0.2).astype(float)                # wing strokes
    chirp = ((t % 0.55) < 0.26).astype(float)                                 # chirps
    return _fade(_lowpass(pulse * chirp, 400.0) * np.sin(2 * np.pi * 4300.0 * t), 20.0)


def bee() -> np.ndarray:
    t = _t(6.0)
    wobble = 230.0 * (1.0 + 0.04 * np.sin(2 * np.pi * 0.7 * t) + 0.015 * np.sin(2 * np.pi * 7.0 * t))
    phase = 2 * np.pi * np.cumsum(wobble) / FS
    buzz = sum(np.sin(k * phase) / k ** 0.8 for k in range(1, 12))
    return _fade(buzz * (0.75 + 0.25 * np.sin(2 * np.pi * 0.5 * t)), 300.0)


def heartbeat() -> np.ndarray:
    n = int(FS * 7.0)
    out = np.zeros(n)
    t = _t(0.16)
    for beat in np.arange(0.3, 6.6, 0.85):                                    # about 70 beats per minute
        for offset, pitch, level in ((0.0, 52.0, 1.0), (0.3, 68.0, 0.7)):     # "lub" - "dub"
            thump = level * np.sin(2 * np.pi * pitch * t) * np.exp(-t * 26.0) * np.sin(np.pi * t / 0.16) ** 0.5
            start = int((beat + offset) * FS)
            out[start:start + thump.size] += thump[:max(0, n - start)]
    return out


def siren() -> np.ndarray:
    t = _t(7.0)
    freq = 950.0 + 450.0 * np.sin(2 * np.pi * 0.45 * t)
    phase = 2 * np.pi * np.cumsum(freq) / FS
    return _fade(np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase), 50.0)


def phone_ring() -> np.ndarray:
    t = _t(6.0)
    ringing = ((t % 3.0) < 1.6).astype(float)
    trill = 0.5 + 0.5 * sp_signal.square(2 * np.pi * 22.0 * t)
    tone = np.sin(2 * np.pi * 1400.0 * t) + 0.7 * np.sin(2 * np.pi * 1850.0 * t) + 0.3 * np.sin(2 * np.pi * 2900.0 * t)
    return _lowpass(ringing * trill, 300.0) * tone


def clock_tick() -> np.ndarray:
    n = int(FS * 7.0)
    out = np.zeros(n)
    rng = _rng(20)
    t = _t(0.03)
    for index, beat in enumerate(np.arange(0.2, 6.8, 0.5)):
        pitch = 2600.0 if index % 2 == 0 else 2100.0                          # tic - tac
        click = (np.sin(2 * np.pi * pitch * t) + 0.4 * rng.standard_normal(t.size)) * np.exp(-t * 220.0)
        start = int(beat * FS)
        out[start:start + click.size] += click
    return out


def whistle() -> np.ndarray:
    t = _t(5.0)
    glide = 1500.0 + 500.0 * np.sin(2 * np.pi * 0.5 * t)
    tone = np.sin(2 * np.pi * np.cumsum(glide) / FS)
    air = 0.06 * _band(_rng(21).standard_normal(t.size), 1200.0, 2600.0, 1)
    return _fade(tone + air, 80.0)


def white_noise() -> np.ndarray:
    return _fade(_rng(30).standard_normal(int(FS * 5.0)), 100.0)


def pink_noise() -> np.ndarray:
    n = int(FS * 5.0)
    spectrum = np.fft.rfft(_rng(31).standard_normal(n))
    spectrum[1:] /= np.sqrt(np.arange(1, spectrum.size))
    return _fade(np.fft.irfft(spectrum, n), 100.0)


def brown_noise() -> np.ndarray:
    n = int(FS * 5.0)
    spectrum = np.fft.rfft(_rng(32).standard_normal(n))
    spectrum[1:] /= np.arange(1, spectrum.size)
    spectrum[:int(20 * n / FS)] = 0.0
    return _fade(np.fft.irfft(spectrum, n), 100.0)


# ----------------------------------------------------------------- catalogue
@dataclass(frozen=True)
class KitSound:
    key: str
    category: str
    title: str
    story: str
    make: Callable[[], np.ndarray] | None      # None = spoken text (made by a speech synthesiser)
    view_hz: float = 5000.0                    # spectrum range that shows it well
    speech: str = ""
    speech_pitch: int = 50


CATEGORIES = ("Tonuri simple", "Instrumente (aceeași notă)", "Muzică", "Voce", "Natură", "Din jurul nostru", "Zgomote")

KIT: tuple[KitSound, ...] = (
    KitSound("ton_la", CATEGORIES[0], "Un singur ton: La (440 Hz)",
             "Cel mai simplu sunet: aerul vibrează de 440 de ori pe secundă. Pe spectru apare un singur vârf.", pure_tone, 2000.0),
    KitSound("gros_subtire", CATEGORIES[0], "De la gros la subțire",
             "Patru tonuri: 110, 440, 1760 și 7040 Hz. Cu cât vibrația e mai rapidă, cu atât sunetul e mai subțire, iar vârful se mută spre dreapta.", low_and_high, 8000.0),
    KitSound("gama", CATEGORIES[0], "Gama Do-Re-Mi",
             "Opt note, fiecare puțin mai sus decât cea dinainte. Ultimul Do vibrează exact de două ori mai repede decât primul.", scale, 1500.0),
    KitSound("alunecare", CATEGORIES[0], "Alunecare: 40 Hz - 12 000 Hz",
             "Un ton care urcă încet. Copiii aud și capătul de sus; adulții îl pierd mai devreme. Test de auz!", sweep, 13000.0),
    KitSound("batai", CATEGORIES[0], "Două tonuri aproape la fel (bătăi)",
             "440 și 443 Hz cântate împreună. Urechea aude un singur sunet care crește și scade de 3 ori pe secundă.", beats, 1000.0),
    KitSound("tare_incet", CATEGORIES[0], "Tare și încet",
             "Aceeași notă, mai tare și mai încet. Vârful nu se mută stânga-dreapta, doar urcă și coboară.", loud_and_quiet, 1500.0),
    KitSound("pian", CATEGORIES[1], "Pian", "Nota La la pian: un vârf la 440 Hz și multe vârfuri mai mici la 880, 1320... Se sting repede.",
             lambda: piano(440.0, 4.0), 5000.0),
    KitSound("chitara", CATEGORIES[1], "Chitară", "O coardă ciupită. Aceeași notă La, dar vârfurile au alte înălțimi, de aceea sună altfel decât pianul.",
             lambda: guitar(440.0, 4.0), 5000.0),
    KitSound("flaut", CATEGORIES[1], "Flaut", "Flautul e aproape un ton curat: un vârf mare și doar câteva mici. De aceea sună moale.",
             lambda: flute(440.0, 4.0), 5000.0),
    KitSound("vioara", CATEGORIES[1], "Vioară", "Vioara are foarte multe vârfuri (armonice), de aceea sună bogat și puțin aspru.",
             lambda: violin(440.0, 4.0), 8000.0),
    KitSound("clopot", CATEGORIES[1], "Clopot", "La clopot vârfurile nu sunt la distanțe egale. De aceea sună «metalic».", lambda: bell(440.0, 5.0), 5000.0),
    KitSound("toba", CATEGORIES[1], "Tobe", "Tobele nu au o notă anume: sunt lovituri scurte de zgomot. Toba mare e jos, cinelul e sus.", drum_beat, 12000.0),
    KitSound("stelute", CATEGORIES[2], "Cântecel la pian («Ah! vous dirai-je, maman»)",
             "Melodia cunoscută de la «Twinkle, Twinkle». Urmărește cum sare vârful de la o notă la alta.", lambda: melody(TWINKLE, piano), 3000.0),
    KitSound("oda", CATEGORIES[2], "Beethoven: Odă bucuriei, la flaut", "Tema din Simfonia a 9-a, cântată la flaut.", lambda: melody(ODE_TO_JOY, flute), 3000.0),
    KitSound("chitara_melodie", CATEGORIES[2], "Odă bucuriei, la chitară", "Aceleași note ca la flaut, alt instrument: vârfurile principale sunt în același loc.",
             lambda: melody(ODE_TO_JOY, guitar), 3000.0),
    KitSound("acorduri", CATEGORIES[2], "Acord vesel și acord trist", "Trei note deodată. Primele două acorduri (Do major) sună vesel, ultimele două (La minor) sună trist.",
             major_minor, 2000.0),
    KitSound("voce_robot", CATEGORIES[3], "Voce de robot: «Bună!»", "O voce făcută de calculator. Vocea are multe vârfuri care se schimbă la fiecare sunet rostit.",
             None, 5000.0, "Bună! Eu sunt o voce de robot. Vocea mea este făcută din multe frecvențe, care se schimbă tot timpul."),
    KitSound("voce_groasa", CATEGORIES[3], "Voce groasă", "Aceeași propoziție cu voce groasă: vârfurile sunt mai la stânga și mai dese.",
             None, 5000.0, "Aceasta este o voce groasă, ca de urs.", 15),
    KitSound("voce_subtire", CATEGORIES[3], "Voce subțire", "Și cu voce subțire: vârfurile sunt mai la dreapta și mai rare.",
             None, 5000.0, "Aceasta este o voce subțire, ca de șoricel.", 95),
    KitSound("numaratoare", CATEGORIES[3], "Numărătoare: unu, doi, trei...", "Fiecare cuvânt arată altfel pe ecran. Încearcă să ghicești cifra fără să asculți!",
             None, 5000.0, "Unu. Doi. Trei. Patru. Cinci. Șase. Șapte. Opt. Nouă. Zece."),
    KitSound("vocale", CATEGORIES[3], "Vocalele A, E, I, O, U", "Aceeași notă, dar gura își schimbă forma. Fiecare vocală întărește alte vârfuri.", vowels_sung, 4000.0),
    KitSound("ploaie", CATEGORIES[4], "Ploaie", "Mii de picături, fiecare cu sunetul ei: împreună dau un fâșâit întins pe tot ecranul.", rain, 14000.0),
    KitSound("vant", CATEGORIES[4], "Vânt", "Vântul e un zgomot gros, care se umflă și se liniștește. Uneori șuieră.", wind, 3000.0),
    KitSound("valuri", CATEGORIES[4], "Valurile mării", "Fiecare val vine, se sparge și se retrage, cam o dată la 5 secunde.", waves, 10000.0),
    KitSound("tunet", CATEGORIES[4], "Tunet", "Un pocnet, apoi un huruit foarte gros, sub 150 Hz. Sunetele groase ajung departe.", thunder, 3000.0),
    KitSound("pasare", CATEGORIES[4], "Pasăre", "Ciripit: tonuri foarte subțiri (3000-6000 Hz) care alunecă repede în sus sau în jos.", birdsong, 8000.0),
    KitSound("greier", CATEGORIES[4], "Greier", "Greierul își freacă aripile: un ton de vreo 4300 Hz, întrerupt de multe ori pe secundă.", cricket, 8000.0),
    KitSound("albina", CATEGORIES[4], "Albină", "Bâzâitul e chiar bătaia aripilor: cam 230 de bătăi pe secundă, deci un vârf la 230 Hz.", bee, 3000.0),
    KitSound("inima", CATEGORIES[4], "Bătăile inimii", "«Lub-dub», de vreo 70 de ori pe minut. Sunete foarte groase, sub 100 Hz.", heartbeat, 400.0),
    KitSound("sirena", CATEGORIES[5], "Sirenă", "Tonul urcă și coboară fără oprire, ca să fie greu de ignorat.", siren, 5000.0),
    KitSound("telefon", CATEGORIES[5], "Telefon vechi", "Două tonuri întrerupte foarte repede: așa «zbârnâie» soneria.", phone_ring, 5000.0),
    KitSound("ceas", CATEGORIES[5], "Ceas: tic-tac", "Pocnituri scurte, de două ori pe secundă. Un sunet scurt se întinde pe multe frecvențe.", clock_tick, 8000.0),
    KitSound("fluierat", CATEGORIES[5], "Fluierat", "Fluieratul e aproape un ton curat, ca un flaut mic făcut din buze.", whistle, 4000.0),
    KitSound("zgomot_alb", CATEGORIES[6], "Zgomot alb", "Toate frecvențele, la fel de tare: spectrul e plat. Sună ca un televizor fără semnal.", white_noise, 20000.0),
    KitSound("zgomot_roz", CATEGORIES[6], "Zgomot roz", "Mai mult din cele groase, mai puțin din cele subțiri. Seamănă cu o cascadă.", pink_noise, 20000.0),
    KitSound("zgomot_maro", CATEGORIES[6], "Zgomot maro", "Aproape numai sunete groase: un huruit, ca un tunet îndepărtat.", brown_noise, 5000.0),
)


def render(sound: KitSound) -> np.ndarray | None:
    """Float32 samples at 44.1 kHz, or ``None`` for speech items (see generate_sound_kit.py)."""
    return None if sound.make is None else _norm(np.asarray(sound.make(), dtype=np.float64))
