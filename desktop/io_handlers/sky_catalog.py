"""Sky catalogue: pulsars and stars whose signals can be "listened to".

Everything here is SYNTHESISED from published, rounded physical values (spin
frequency of pulsars, oscillation frequencies of stars). These are not
recordings from a radio telescope. One entry (Vega) is fiction, modelled on
the film "Contact" (1997), and is labelled as such.

Each object gives two signals:

* ``analysis_signal``: what the analyser shows. Its sample rate is chosen so
  the object's real frequency is well resolved by the FFT.
* ``audio_signal``: a 44.1 kHz version made audible (pulses become clicks,
  slow star vibrations are sped up), for loudspeakers and for WAV export.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np

from .loader import SignalData

AUDIO_SAMPLE_RATE = 44100


@dataclass(frozen=True)
class SkyObject:
    key: str
    name: str                    # catalogue / proper name
    nickname: str                # short Romanian label used on the map and in file names
    kind: str                    # "pulsar", "star", "variable", "planet" or "fiction"
    constellation: str
    ra_hours: float | None       # right ascension (J2000), None = not fixed on the map (the Sun)
    dec_deg: float | None        # declination (J2000)
    frequency_hz: float          # real frequency: pulsar spin, or nu_max of the star
    analysis_rate: float         # sample rate of the analysis signal
    fft_size: int                # FFT size that shows it well
    story_ro: str                # explanation for children, in Romanian
    speedup: float = 1.0         # stars only: how much the vibration is sped up
    spacing_hz: float = 0.0      # stars only: large frequency separation (real, in Hz)
    duty: float = 0.04           # pulsars: pulse width as a fraction of the period (audio version)

    @property
    def display_frequency_hz(self) -> float:
        """Frequency the analyser will show (sped-up for stars)."""
        return self.frequency_hz * self.speedup

    @property
    def kind_ro(self) -> str:
        return {"pulsar": "Pulsar", "star": "Stea", "variable": "Stea pulsantă", "planet": "Planetă",
                "fiction": "Ficțiune (film)"}[self.kind]


SKY_OBJECTS: tuple[SkyObject, ...] = (
    SkyObject(
        "crab", "PSR B0531+21", "Pulsarul Crab", "pulsar", "Taurul", 5.575, 22.01, 29.6, 4000.0, 4096,
        "O stea a explodat în anul 1054, iar astronomii chinezi au văzut-o ziua în amiaza mare. "
        "A rămas un miez cât un oraș, care se învârte de aproape 30 de ori pe secundă. "
        "La fiecare rotire ne trimite un fulger radio, ca un far.",
        duty=0.03,
    ),
    SkyObject(
        "vela", "PSR B0833-45", "Pulsarul Vela", "pulsar", "Velele", 8.589, -45.18, 11.19, 2000.0, 4096,
        "Se învârte de 11 ori pe secundă. Este unul dintre cei mai puternici pulsari de pe cerul sudic "
        "și sună ca un motor care toarce.",
        duty=0.04,
    ),
    SkyObject(
        "b0329", "PSR B0329+54", "Pulsarul B0329+54", "pulsar", "Girafa", 3.550, 54.58, 1.3995, 500.0, 4096,
        "Un pulsar lent: cam o bătaie și jumătate pe secundă, ca o inimă liniștită. "
        "Este printre cei mai strălucitori pulsari în unde radio de pe cerul nordic.",
        duty=0.03,
    ),
    SkyObject(
        "lgm1", "PSR B1919+21", "Primul pulsar (LGM-1)", "pulsar", "Vulpea", 19.363, 21.88, 0.7478, 500.0, 4096,
        "Primul pulsar descoperit, în 1967, de studenta Jocelyn Bell. Bate o dată la 1,34 secunde, atât de "
        "regulat încât l-au poreclit în glumă LGM-1: «omuleții verzi». Era doar o stea care se rotește.",
        duty=0.03,
    ),
    SkyObject(
        "b1937", "PSR B1937+21", "Pulsarul milisecundă", "pulsar", "Vulpea", 19.661, 21.58, 641.93, 44100.0, 4096,
        "Se învârte de 642 de ori pe secundă! Suprafața lui aleargă cu o parte bună din viteza luminii. "
        "Nu mai auzi bătăi separate, ci o notă muzicală, aproape de un Mi.",
        duty=0.08,
    ),
    SkyObject(
        "j0437", "PSR J0437-4715", "Pulsarul J0437-4715", "pulsar", "Pictorul", 4.621, -47.25, 173.69, 22050.0, 4096,
        "Unul dintre cele mai apropiate și mai precise ceasuri din Univers: 174 de rotații pe secundă, "
        "aproape fără nicio abatere. Sună ca o notă joasă, aproape de un Fa.",
        duty=0.10,
    ),
    SkyObject(
        "b0950", "PSR B0950+08", "Pulsarul B0950+08", "pulsar", "Leul", 9.886, 7.93, 3.951, 1000.0, 4096,
        "Un pulsar apropiat, care bate de aproape 4 ori pe secundă, ca un ceas grăbit.",
        duty=0.04,
    ),
    SkyObject(
        "fastest", "PSR J1748-2446ad", "Cel mai rapid pulsar", "pulsar", "Săgetătorul", 17.80, -24.78, 716.36, 44100.0, 4096,
        "Recordul cunoscut: 716 rotații pe secundă. O stea mai grea decât Soarele, strânsă într-o bilă "
        "de vreo 30 km, care se învârte mai repede decât un blender.",
        duty=0.10,
    ),
    SkyObject(
        "geminga", "Geminga (PSR J0633+1746)", "Geminga", "pulsar", "Gemenii", 6.566, 17.77, 4.2176, 1000.0, 4096,
        "Un pulsar aproape mut în unde radio, dar foarte strălucitor în raze gamma. Numele vine dintr-o "
        "glumă în dialectul milanez: «gh'è minga» înseamnă «nu e acolo», fiindcă ani de zile nu l-a găsit nimeni.",
        duty=0.05,
    ),
    SkyObject(
        "b1257", "PSR B1257+12 (Lich)", "Pulsarul cu planete", "pulsar", "Fecioara", 13.004, 12.68, 160.81, 22050.0, 4096,
        "În jurul lui s-au găsit, în 1992, primele planete din afara Sistemului Solar. Au fost descoperite "
        "pentru că trag de pulsar și îi strică puțin ritmul perfect de 161 de bătăi pe secundă.",
        duty=0.10,
    ),
    SkyObject(
        "hulse", "PSR B1913+16", "Pulsarul Hulse-Taylor", "pulsar", "Vulturul", 19.259, 16.11, 16.94, 2000.0, 4096,
        "Doi sori morți care se rotesc unul în jurul celuilalt în mai puțin de 8 ore. Se apropie încet, "
        "exact cum a prezis Einstein, pentru că pierd energie prin unde gravitaționale. Premiul Nobel, 1993.",
        duty=0.05,
    ),
    SkyObject(
        "double", "PSR J0737-3039A", "Pulsarul dublu", "pulsar", "Pupa", 7.627, -30.66, 44.054, 8000.0, 4096,
        "Singurul sistem cunoscut în care s-au văzut doi pulsari care se rotesc unul în jurul celuilalt. "
        "Acesta este cel rapid: 44 de rotații pe secundă.",
        duty=0.06,
    ),
    SkyObject(
        "sun", "Soarele", "Soarele", "star", "-", None, None, 3090e-6, 8000.0, 8192,
        "Soarele vibrează ca un clopot uriaș: suprafața lui urcă și coboară o dată la 5 minute. "
        "E prea lent ca să-l auzim, așa că am accelerat vibrația de 100 000 de ori.",
        speedup=1e5, spacing_hz=135.1e-6,
    ),
    SkyObject(
        "alfcen", "Alpha Centauri A", "Alpha Centauri A", "star", "Centaurul", 14.660, -60.83, 2410e-6, 8000.0, 8192,
        "Vecina noastră, la 4,4 ani-lumină, seamănă cu Soarele. E puțin mai mare, așa că vibrează "
        "puțin mai jos. Sunet accelerat de 100 000 de ori.",
        speedup=1e5, spacing_hz=105.5e-6,
    ),
    SkyObject(
        "procyon", "Procyon A", "Procyon", "star", "Câinele Mic", 7.655, 5.22, 1000e-6, 8000.0, 8192,
        "O stea mai mare și mai caldă decât Soarele. Stelele mari vibrează mai rar, deci sună mai gros. "
        "Sunet accelerat de 100 000 de ori.",
        speedup=1e5, spacing_hz=55.0e-6,
    ),
    SkyObject(
        "delcep", "Delta Cephei", "Delta Cephei", "variable", "Cefeu", 22.486, 58.42, 1.0 / (5.3663 * 86400), 8000.0, 8192,
        "O stea care respiră: se umflă și se strânge o dată la 5 zile și 9 ore, iar lumina ei crește și scade. "
        "Cu astfel de stele au măsurat astronomii distanța până la alte galaxii. Accelerat de 100 de milioane de ori.",
        speedup=1e8,
    ),
    SkyObject(
        "polaris", "Polaris (Alpha UMi)", "Steaua Polară", "variable", "Carul Mic", 2.530, 89.26, 1.0 / (3.9696 * 86400), 8000.0, 8192,
        "Steaua care arată nordul pulsează și ea, foarte puțin, o dată la aproape 4 zile. "
        "Accelerat de 100 de milioane de ori.",
        speedup=1e8,
    ),
    SkyObject(
        "rrlyr", "RR Lyrae", "RR Lyrae", "variable", "Lira", 19.424, 42.78, 1.0 / (0.56687 * 86400), 22050.0, 4096,
        "O stea bătrână care pulsează repede: o dată la 13 ore și jumătate. Fiind mai rapidă decât "
        "Delta Cephei, sună mult mai subțire. Accelerat de 100 de milioane de ori.",
        speedup=1e8,
    ),
    SkyObject(
        "mira", "Mira (Omicron Ceti)", "Mira", "variable", "Balena", 2.322, -2.98, 1.0 / (332.0 * 86400), 1000.0, 4096,
        "«Minunata»: o gigantă roșie care aproape dispare și reapare pe cer o dată la 11 luni. "
        "Chiar și accelerată de 100 de milioane de ori, bate doar de 3-4 ori pe secundă.",
        speedup=1e8,
    ),
    SkyObject(
        "vega", "Vega", "Mesajul din «Contact»", "fiction", "Lira", 18.616, 38.78, 880.0, 8000.0, 4096,
        "FICȚIUNE. În filmul «Contact» (1997), Ellie aude dinspre Vega bătăi grupate în numere prime: "
        "2, 3, 5, 7, 11, 13... Natura nu face numere prime, deci ar fi semnul cuiva inteligent. "
        "În realitate, de la Vega nu s-a primit niciodată un asemenea semnal.",
    ),
)

_YEAR_SECONDS = 365.25 * 86400.0
PLANET_SPEEDUP = 440.0 * _YEAR_SECONDS       # one Earth year becomes one period of the note A (440 Hz)


def _planet(key, name, period_years, rate, fft, story):
    return SkyObject(key, name, name, "planet", "Sistemul Solar", None, None, 1.0 / (period_years * _YEAR_SECONDS),
                     rate, fft, story, speedup=PLANET_SPEEDUP, duty=0.06)


PLANETS: tuple[SkyObject, ...] = (
    _planet("mercury", "Mercur", 0.2408467, 44100.0, 4096,
            "Cea mai apropiată de Soare și cea mai grăbită: un an durează doar 88 de zile. De aceea are nota cea mai înaltă."),
    _planet("venus", "Venus", 0.61519726, 44100.0, 4096,
            "Un an pe Venus ține 225 de zile. E aproape cât Pământul, dar acoperită de nori groși și fierbinți."),
    _planet("earth", "Pământul", 1.0000174, 22050.0, 4096,
            "Casa noastră. Am ales ca un an pământesc să devină nota La (440 Hz); toate celelalte planete păstrează proporția reală."),
    _planet("mars", "Marte", 1.8808476, 22050.0, 4096,
            "Planeta roșie are nevoie de aproape 2 ani pentru un ocol, deci sună cu aproape o octavă mai jos decât Pământul."),
    _planet("jupiter", "Jupiter", 11.862615, 4000.0, 4096,
            "Uriașul: un an ține aproape 12 ani de-ai noștri. Nota lui e un bâzâit gros, abia de 37 de bătăi pe secundă."),
    _planet("saturn", "Saturn", 29.447498, 2000.0, 4096,
            "Planeta cu inele ocolește Soarele în 29 de ani și jumătate. Aici nu mai auzi o notă, ci bătăi dese."),
    _planet("uranus", "Uranus", 84.016846, 1000.0, 4096,
            "84 de ani pentru un singur ocol: un om vede cel mult un an de Uranus într-o viață. Cam 5 bătăi pe secundă."),
    _planet("neptune", "Neptun", 164.79132, 500.0, 4096,
            "Cea mai îndepărtată: de când a fost descoperită, în 1846, a făcut abia un ocol complet. Bate rar, de 2-3 ori pe secundă."),
)

# Drawing data: key -> (distance from the Sun in AU, mean longitude at J2000 in degrees,
#                       degrees per day, colour, dot radius in pixels)
PLANET_ORBITS: dict[str, tuple[float, float, float, str, float]] = {
    "mercury": (0.387, 252.25, 4.09233, "#b8b0a4", 3.0),
    "venus": (0.723, 181.98, 1.60213, "#e8cf9a", 4.5),
    "earth": (1.000, 100.46, 0.98561, "#5aa9ff", 4.8),
    "mars": (1.524, 355.43, 0.52403, "#e0694a", 3.8),
    "jupiter": (5.203, 34.40, 0.08309, "#d9a66b", 9.0),
    "saturn": (9.537, 49.94, 0.03346, "#e3cf8f", 8.0),
    "uranus": (19.19, 313.23, 0.01173, "#8fd6dc", 6.5),
    "neptune": (30.07, 304.88, 0.00598, "#5f7fe8", 6.3),
}


def planet_longitude(key: str, days_since_j2000: float) -> float:
    """Approximate heliocentric longitude in degrees (circular orbit, mean motion)."""
    _, longitude0, rate, _, _ = PLANET_ORBITS[key]
    return (longitude0 + rate * days_since_j2000) % 360.0


SKY_BY_KEY = {obj.key: obj for obj in SKY_OBJECTS + PLANETS}

# Bright stars for the map: (name, RA hours, Dec degrees, visual magnitude). J2000, rounded.
BRIGHT_STARS: tuple[tuple[str, float, float, float], ...] = (
    ("Sirius", 6.752, -16.72, -1.46), ("Canopus", 6.399, -52.70, -0.74), ("Arcturus", 14.261, 19.18, -0.05),
    ("Vega", 18.616, 38.78, 0.03), ("Capella", 5.278, 46.00, 0.08), ("Rigel", 5.242, -8.20, 0.13),
    ("Procyon", 7.655, 5.22, 0.34), ("Betelgeuse", 5.919, 7.41, 0.42), ("Achernar", 1.629, -57.24, 0.46),
    ("Hadar", 14.064, -60.37, 0.61), ("Altair", 19.846, 8.87, 0.76), ("Acrux", 12.443, -63.10, 0.76),
    ("Aldebaran", 4.599, 16.51, 0.86), ("Antares", 16.490, -26.43, 0.96), ("Spica", 13.420, -11.16, 0.97),
    ("Pollux", 7.755, 28.03, 1.14), ("Fomalhaut", 22.961, -29.62, 1.16), ("Deneb", 20.690, 45.28, 1.25),
    ("Mimosa", 12.795, -59.69, 1.25), ("Regulus", 10.140, 11.97, 1.35), ("Castor", 7.577, 31.89, 1.58),
    ("Gacrux", 12.519, -57.11, 1.63), ("Bellatrix", 5.419, 6.35, 1.64), ("Elnath", 5.438, 28.61, 1.65),
    ("Alnilam", 5.604, -1.20, 1.69), ("Alnitak", 5.679, -1.94, 1.77), ("Mintaka", 5.533, -0.30, 2.23),
    ("Saiph", 5.796, -9.67, 2.09), ("Polaris", 2.530, 89.26, 1.98), ("Dubhe", 11.062, 61.75, 1.79),
    ("Merak", 11.031, 56.38, 2.37), ("Phecda", 11.897, 53.69, 2.44), ("Megrez", 12.257, 57.03, 3.31),
    ("Alioth", 12.900, 55.96, 1.77), ("Mizar", 13.399, 54.93, 2.27), ("Alkaid", 13.792, 49.31, 1.86),
    ("Caph", 0.153, 59.15, 2.27), ("Schedar", 0.675, 56.54, 2.23), ("Gamma Cas", 0.945, 60.72, 2.47),
    ("Ruchbah", 1.430, 60.24, 2.68), ("Segin", 1.907, 63.67, 3.38), ("Sadr", 20.370, 40.26, 2.20),
    ("Albireo", 19.512, 27.96, 3.08), ("Gienah", 20.770, 33.97, 2.46), ("Delta Cyg", 19.750, 45.13, 2.87),
    ("Sheliak", 18.835, 33.36, 3.45), ("Sulafat", 18.982, 32.69, 3.24), ("Rigil Kent", 14.660, -60.83, -0.27),
    ("Alphard", 9.460, -8.66, 1.98), ("Denebola", 11.818, 14.57, 2.14), ("Algol", 3.136, 40.96, 2.12),
    ("Mirfak", 3.405, 49.86, 1.79), ("Alpheratz", 0.140, 29.09, 2.06), ("Shaula", 17.560, -37.10, 1.63),
    ("Dschubba", 16.006, -22.62, 2.32), ("Sargas", 17.622, -43.00, 1.87), ("Delta Cru", 12.252, -58.75, 2.80),
)

# Stick figures: (constellation label, label position (RA h, Dec deg), [star names in drawing order]).
CONSTELLATION_LINES: tuple[tuple[str, tuple[float, float], tuple[str, ...]], ...] = (
    ("Orion", (5.9, 13.5), ("Betelgeuse", "Bellatrix", "Mintaka", "Alnilam", "Alnitak", "Saiph", "Rigel", "Mintaka")),
    ("Orion", (0.0, 0.0), ("Betelgeuse", "Alnitak")),
    ("Carul Mare", (12.4, 66.0), ("Alkaid", "Mizar", "Alioth", "Megrez", "Phecda", "Merak", "Dubhe", "Megrez")),
    ("Cassiopeia", (1.0, 70.0), ("Caph", "Schedar", "Gamma Cas", "Ruchbah", "Segin")),
    ("Lebăda", (20.4, 53.0), ("Deneb", "Sadr", "Albireo")),
    ("Lebăda", (0.0, 0.0), ("Gienah", "Sadr", "Delta Cyg")),
    ("Lira", (18.3, 46.0), ("Vega", "Sheliak", "Sulafat", "Vega")),
    ("Crucea Sudului", (12.5, -70.0), ("Gacrux", "Acrux")),
    ("Crucea Sudului", (0.0, 0.0), ("Mimosa", "Delta Cru")),
    ("Scorpionul", (16.6, -17.0), ("Dschubba", "Antares", "Shaula", "Sargas")),
    ("Gemenii", (7.6, 38.5), ("Castor", "Pollux")),
)

LABELLED_STARS = ("Sirius", "Canopus", "Arcturus", "Capella", "Rigel", "Betelgeuse", "Altair", "Aldebaran",
                  "Antares", "Spica", "Fomalhaut", "Deneb", "Regulus", "Polaris", "Achernar")


def galactic_plane(points: int = 181) -> tuple[np.ndarray, np.ndarray]:
    """The Milky Way's mid-line (galactic latitude 0) as (RA hours, Dec degrees)."""
    lon = np.radians(np.linspace(0.0, 360.0, points))
    ra_pole, dec_pole, lon_ncp = np.radians(192.85948), np.radians(27.12825), np.radians(122.93192)
    dec = np.arcsin(np.cos(dec_pole) * np.cos(lon_ncp - lon))                  # latitude b = 0
    ra = ra_pole + np.arctan2(np.sin(lon_ncp - lon), -np.sin(dec_pole) * np.cos(lon_ncp - lon))
    return (np.degrees(ra) / 15.0) % 24.0, np.degrees(dec)


# ------------------------------------------------------------------ synthesis
def _rng(obj: SkyObject, salt: int) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(obj.key.encode()) + salt)


def _pulse_envelope(obj: SkyObject, n: int, fs: float, rng: np.random.Generator, jitter: float, duty: float) -> np.ndarray:
    """Periodic Gaussian pulses at the pulsar's spin frequency, with pulse-to-pulse variation."""
    rate = obj.display_frequency_hz
    phase = (np.arange(n) * (rate / fs)) % 1.0
    sigma = duty / 2.355                                                       # duty = full width at half maximum
    distance = np.minimum(phase, 1.0 - phase)
    env = np.exp(-0.5 * (distance / sigma) ** 2)
    if jitter > 0:                                                             # real pulses differ in strength
        turns = int(n * rate / fs) + 2
        gains = 1.0 + jitter * rng.standard_normal(turns).clip(-2, 2)
        env *= gains[np.floor((np.arange(n) * (rate / fs)) + 0.5).astype(int)]
    return env


def _star_modes(obj: SkyObject, n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Solar-like oscillations: a comb of modes around nu_max, sped up to audio."""
    centre = obj.frequency_hz * obj.speedup
    spacing = obj.spacing_hz * obj.speedup
    width = 0.25 * centre                                                      # envelope of excited modes
    t = np.arange(n) / fs
    out = np.zeros(n)
    for order in range(-9, 10):
        for degree, (shift, strength) in enumerate(((0.0, 1.0), (0.5, 0.85), (-0.12, 0.45))):
            freq = centre + (order + shift) * spacing
            if freq <= 0 or freq >= 0.45 * fs:
                continue
            amp = strength * math.exp(-0.5 * ((freq - centre) / (width / 2.355)) ** 2)
            if amp < 0.02:
                continue
            if order == 0 and degree == 0:
                amp *= 1.8                                                     # the mode at nu_max leads
            out += amp * np.sin(2.0 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))
    return out / np.max(np.abs(out))


def _light_curve(obj: SkyObject, n: int, fs: float) -> np.ndarray:
    """Pulsating star: quick brightening, slow fading, repeated at the (sped-up) pulsation rate."""
    t = np.arange(n) / fs
    out = np.zeros(n)
    for harmonic, amp in ((1, 1.0), (2, 0.42), (3, 0.22), (4, 0.11), (5, 0.06)):
        freq = harmonic * obj.display_frequency_hz
        if freq < 0.45 * fs:
            out += amp * np.sin(2.0 * np.pi * freq * t + 0.6 * (harmonic - 1))
    return out / np.max(np.abs(out))


def _prime_message(n: int, fs: float, tone_hz: float) -> np.ndarray:
    """Groups of beeps counting the prime numbers 2, 3, 5, 7, 11, 13 - as in the film "Contact"."""
    beep, gap, pause = 0.11, 0.09, 0.75
    out = np.zeros(n)
    t0 = 0.4
    ramp = int(0.008 * fs)
    while t0 * fs < n:
        for prime in (2, 3, 5, 7, 11, 13):
            for _ in range(prime):
                a, b = int(t0 * fs), min(int((t0 + beep) * fs), n)
                if a >= n:
                    return out
                seg = np.sin(2.0 * np.pi * tone_hz * np.arange(b - a) / fs)
                window = np.ones(b - a)
                edge = min(ramp, (b - a) // 2)
                if edge > 0:
                    window[:edge] = np.linspace(0, 1, edge)
                    window[-edge:] = np.linspace(1, 0, edge)
                out[a:b] = seg * window
                t0 += beep + gap
            t0 += pause
        t0 += 1.5
    return out


def analysis_signal(obj: SkyObject, seconds: float | None = None) -> SignalData:
    """The signal shown in the analyser; its dominant frequency is the object's (sped-up) frequency."""
    fs = obj.analysis_rate
    if seconds is None:                                                        # at least ~40 periods, 20 s minimum
        seconds = max(20.0, 60.0 / obj.display_frequency_hz) if obj.kind != "fiction" else 26.0
        seconds = min(seconds, 90.0)
    n = int(fs * seconds)
    rng = _rng(obj, 1)
    if obj.kind == "variable":
        x = 0.6 * _light_curve(obj, n, fs) + 0.02 * rng.standard_normal(n)
    elif obj.kind in ("pulsar", "planet"):
        # Wider pulses than in the audio version: the spin frequency itself then stands
        # clearly above its harmonics in the spectrum.
        x = 0.8 * _pulse_envelope(obj, n, fs, rng, jitter=0.08, duty=max(obj.duty, 0.14))
        x -= x.mean()
        x += 0.02 * rng.standard_normal(n)
    elif obj.kind == "star":
        x = 0.6 * _star_modes(obj, n, fs, rng) + 0.02 * rng.standard_normal(n)
    else:
        x = 0.7 * _prime_message(n, fs, obj.frequency_hz) + 0.03 * rng.standard_normal(n)
    return SignalData(samples=x.astype(np.float32), sample_rate=fs, name=f"{obj.nickname} ({obj.name})")


def audio_signal(obj: SkyObject, seconds: float = 16.0) -> np.ndarray:
    """A 44.1 kHz float32 version meant for loudspeakers and WAV export."""
    fs = float(AUDIO_SAMPLE_RATE)
    n = int(fs * seconds)
    rng = _rng(obj, 2)
    if obj.kind == "variable" and obj.display_frequency_hz >= 60.0:
        x = 0.7 * _light_curve(obj, n, fs) + 0.01 * rng.standard_normal(n)
    elif obj.kind in ("pulsar", "planet", "variable"):
        env = _pulse_envelope(obj, n, fs, rng, jitter=0.0 if obj.kind == "planet" else 0.15, duty=obj.duty)
        if obj.display_frequency_hz < 60.0:
            # Slow pulsars: each radio pulse becomes a short burst of hiss, as on a telescope's loudspeaker.
            noise = rng.standard_normal(n)
            spectrum = np.fft.rfft(noise)
            freqs = np.fft.rfftfreq(n, 1.0 / fs)
            spectrum[(freqs < 250.0) | (freqs > 3500.0)] = 0.0
            carrier = np.fft.irfft(spectrum, n)
            x = env * carrier / np.max(np.abs(carrier))
        else:
            x = env - env.mean()                                               # fast pulsars are a tone by themselves
        x = 0.85 * x / np.max(np.abs(x)) + 0.012 * rng.standard_normal(n)
    elif obj.kind == "star":
        x = 0.7 * _star_modes(obj, n, fs, rng) + 0.01 * rng.standard_normal(n)
    else:
        x = 0.7 * _prime_message(n, fs, obj.frequency_hz) + 0.02 * rng.standard_normal(n)
    fade = int(0.02 * fs)
    x[:fade] *= np.linspace(0, 1, fade)
    x[-fade:] *= np.linspace(1, 0, fade)
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def frequency_text(obj: SkyObject) -> str:
    """Human-readable real frequency, in Romanian."""
    if obj.kind == "pulsar":
        period = 1.0 / obj.frequency_hz
        period_text = f"{period * 1e3:.3f} ms" if period < 1.0 else f"{period:.3f} s"
        return f"{obj.frequency_hz:g} rotații pe secundă (o rotație la {period_text})"
    if obj.kind == "variable":
        days = 1.0 / obj.frequency_hz / 86400.0
        period = f"{days:.2f} zile" if days >= 2 else f"{days * 24:.1f} ore"
        return (f"o pulsație la {period}; accelerat de {obj.speedup:,.0f} ori apare la "
                f"{obj.display_frequency_hz:.4g} Hz").replace(",", " ")
    if obj.kind == "planet":
        years = 1.0 / obj.frequency_hz / _YEAR_SECONDS
        period = f"{years * 365.25:.0f} zile" if years < 2 else f"{years:.1f} ani"
        return f"un ocol în jurul Soarelui în {period}; dacă un an pământesc ar fi nota La, ar suna la {obj.display_frequency_hz:.4g} Hz"
    if obj.kind == "star":
        minutes = 1.0 / obj.frequency_hz / 60.0
        return (f"o vibrație la {minutes:.1f} minute ({obj.frequency_hz * 1e6:.0f} µHz); "
                f"accelerat de {obj.speedup:,.0f} ori apare la {obj.display_frequency_hz:.0f} Hz").replace(",", " ")
    return f"ton de {obj.frequency_hz:.0f} Hz, grupat în numere prime (semnal inventat pentru film)"
