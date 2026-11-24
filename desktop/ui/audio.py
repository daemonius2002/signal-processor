"""Loudspeaker output for the sky signals (QtMultimedia, part of PyQt6 - no extra dependency).

The player degrades gracefully: without QtMultimedia or without an audio
device ``AudioPlayer.available`` is False and every call is a no-op.
"""

from __future__ import annotations

import os

import numpy as np
from PyQt6.QtCore import QIODevice, QObject
from PyQt6.QtGui import QGuiApplication

try:
    from PyQt6.QtMultimedia import QAudioFormat, QAudioSink, QAudioSource, QMediaDevices

    _HAVE_MULTIMEDIA = True
except ImportError:                                   # QtMultimedia missing on this system
    _HAVE_MULTIMEDIA = False

AUDIO_SAMPLE_RATE = 44100


class _LoopDevice(QIODevice):
    """Endless read-only stream that repeats one block of PCM bytes."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._data = b""
        self._position = 0

    def set_data(self, data: bytes) -> None:
        self._data = data
        self._position = 0

    def isSequential(self) -> bool:  # noqa: N802 (Qt naming)
        return True

    def bytesAvailable(self) -> int:  # noqa: N802 (Qt naming)
        return (1 << 20) if self._data else 0

    def readData(self, max_len: int) -> bytes:  # noqa: N802 (Qt naming)
        size = len(self._data)
        if size == 0 or max_len <= 0:
            return b""
        max_len -= max_len % 4                        # whole frames for mono and stereo 16-bit
        out = bytearray()
        while len(out) < max_len:
            take = min(max_len - len(out), size - self._position)
            out += self._data[self._position:self._position + take]
            self._position = (self._position + take) % size
        return bytes(out)

    def writeData(self, _data: bytes) -> int:  # noqa: N802 (Qt naming)
        return 0


class AudioPlayer(QObject):
    """Plays a mono float32 buffer (44.1 kHz) in a loop on the default output device."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._sink = None
        self._channels = 1
        self._volume = 0.6
        self._device = _LoopDevice(self)
        self._device.open(QIODevice.OpenModeFlag.ReadOnly)
        self.error = ""
        if not _HAVE_MULTIMEDIA:
            self.error = "QtMultimedia is not available"
            return
        # Headless runs (tests, benchmarks) and SIGNAL_PROCESSOR_NO_AUDIO=1 skip the sound system:
        # probing for devices can abort the process on machines without any audio service.
        if os.environ.get("SIGNAL_PROCESSOR_NO_AUDIO") or QGuiApplication.platformName() in ("offscreen", "minimal"):
            self.error = "audio is switched off in this session"
            return
        try:
            output = QMediaDevices.defaultAudioOutput()
            if output.isNull():
                self.error = "No audio output device was found"
                return
            for channels in (1, 2):
                fmt = QAudioFormat()
                fmt.setSampleRate(AUDIO_SAMPLE_RATE)
                fmt.setChannelCount(channels)
                fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
                if output.isFormatSupported(fmt):
                    self._channels = channels
                    self._sink = QAudioSink(output, fmt, self)
                    self._sink.setVolume(self._volume)
                    return
            self.error = "The audio device does not accept 44.1 kHz 16-bit sound"
        except Exception as exc:                      # audio must never take the application down
            self._sink = None
            self.error = f"Audio could not be started: {exc}"

    @property
    def available(self) -> bool:
        return self._sink is not None

    def play(self, samples: np.ndarray) -> None:
        """Start looping ``samples`` (mono float32 in -1..1 at 44.1 kHz)."""
        if self._sink is None:
            return
        pcm = np.round(np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
        if self._channels == 2:
            pcm = np.repeat(pcm, 2)
        try:
            self._sink.stop()
            self._device.set_data(pcm.tobytes())
            self._sink.start(self._device)
        except Exception as exc:
            self.error = f"Audio playback failed: {exc}"

    def stop(self) -> None:
        if self._sink is not None:
            self._sink.stop()
            self._device.set_data(b"")

    def set_paused(self, paused: bool) -> None:
        if self._sink is None:
            return
        if paused:
            self._sink.suspend()
        else:
            self._sink.resume()

    def set_volume(self, volume: float) -> None:
        self._volume = float(min(max(volume, 0.0), 1.0))
        if self._sink is not None:
            self._sink.setVolume(self._volume)


def _audio_blocked() -> str:
    """Reason why the sound system must not be touched in this session, or ""."""
    if not _HAVE_MULTIMEDIA:
        return "QtMultimedia is not available"
    if os.environ.get("SIGNAL_PROCESSOR_NO_AUDIO") or QGuiApplication.platformName() in ("offscreen", "minimal"):
        return "audio is switched off in this session"
    return ""


class MicrophoneInput(QObject):
    """Live microphone as a signal source (``generate`` returns whatever has arrived since the last call)."""

    name = "Microfon"

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.sample_rate = float(AUDIO_SAMPLE_RATE)
        self.error = _audio_blocked()
        self._source = None
        self._device = None
        self._channels = 1
        self._float = False

    @staticmethod
    def devices() -> list[str]:
        """Names of the available input devices, the system default first."""
        if _audio_blocked():
            return []
        try:
            default = QMediaDevices.defaultAudioInput().description()
            names = [d.description() for d in QMediaDevices.audioInputs()]
        except Exception:
            return []
        return ([default] if default in names else []) + [n for n in names if n != default]

    def start(self, device_name: str | None = None) -> bool:
        """Open a microphone (the default one if no name is given). Returns False and sets ``error`` on failure."""
        self.stop()
        self.error = _audio_blocked()                 # a failed earlier attempt may succeed now
        if self.error:
            return False
        try:
            device = QMediaDevices.defaultAudioInput()
            for candidate in QMediaDevices.audioInputs():
                if device_name and candidate.description() == device_name:
                    device = candidate
            self.name = f"Microfon ({device.description()})" if not device.isNull() else "Microfon"
            if device.isNull():
                self.error = "Nu am găsit niciun microfon"
                return False
            fmt = QAudioFormat()
            fmt.setSampleRate(AUDIO_SAMPLE_RATE)
            fmt.setChannelCount(1)
            fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
            if not device.isFormatSupported(fmt):
                fmt = device.preferredFormat()
            if fmt.sampleFormat() not in (QAudioFormat.SampleFormat.Int16, QAudioFormat.SampleFormat.Float):
                self.error = "Formatul microfonului nu este suportat"
                return False
            self._float = fmt.sampleFormat() == QAudioFormat.SampleFormat.Float
            self._channels = max(1, fmt.channelCount())
            self.sample_rate = float(fmt.sampleRate())
            self._source = QAudioSource(device, fmt, self)
            self._device = self._source.start()
            if self._device is None:
                self.error = "Microfonul nu a putut fi pornit (verifică permisiunile din Windows)"
                return False
            return True
        except Exception as exc:
            self.error = f"Microfonul nu a putut fi pornit: {exc}"
            return False

    def stop(self) -> None:
        if self._source is not None:
            self._source.stop()
        self._source = None
        self._device = None

    def reset(self) -> None:
        self.generate(0)                              # drop what is waiting

    def generate(self, _n: int) -> np.ndarray:
        if self._device is None:
            return np.zeros(0, dtype=np.float32)
        raw = bytes(self._device.readAll())
        frame = (4 if self._float else 2) * self._channels
        raw = raw[:len(raw) - len(raw) % frame]
        if not raw:
            return np.zeros(0, dtype=np.float32)
        data = np.frombuffer(raw, dtype="<f4") if self._float else np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if self._channels > 1:
            data = data.reshape(-1, self._channels).mean(axis=1)
        return np.ascontiguousarray(data, dtype=np.float32)
