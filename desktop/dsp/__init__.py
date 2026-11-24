"""DSP package: FFT engine, digital filters and numeric metrics."""

from .processor import (
    FFT_SIZES,
    FILTER_TYPES,
    SCALES,
    WINDOW_FUNCTIONS,
    DSPConfig,
    FrameResult,
    Metrics,
    SignalProcessor,
    THIRD_OCTAVE_CENTRES,
    third_octave_levels,
)

__version__ = "1.4.2"

__all__ = [
    "FFT_SIZES",
    "FILTER_TYPES",
    "SCALES",
    "WINDOW_FUNCTIONS",
    "DSPConfig",
    "FrameResult",
    "Metrics",
    "SignalProcessor",
    "THIRD_OCTAVE_CENTRES",
    "third_octave_levels",
    "__version__",
]
