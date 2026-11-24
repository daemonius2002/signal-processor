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
)

__version__ = "1.3.0"

__all__ = [
    "FFT_SIZES",
    "FILTER_TYPES",
    "SCALES",
    "WINDOW_FUNCTIONS",
    "DSPConfig",
    "FrameResult",
    "Metrics",
    "SignalProcessor",
    "__version__",
]
