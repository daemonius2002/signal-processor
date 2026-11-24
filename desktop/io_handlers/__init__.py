"""Signal sources: WAV / CSV loaders, synthetic generator and real-time stream simulator."""

from .loader import (
    SampleRateRequiredError,
    SignalData,
    StreamSimulator,
    SyntheticGenerator,
    load_csv,
    load_signal,
    load_wav,
)

__all__ = [
    "SampleRateRequiredError",
    "SignalData",
    "StreamSimulator",
    "SyntheticGenerator",
    "load_csv",
    "load_signal",
    "load_wav",
]
