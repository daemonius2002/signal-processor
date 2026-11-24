"""Signal Processor - real-time frequency signal processing.

Usage:
    python main.py                      start the application
    python main.py FILE                 start and open a .wav / .csv file
    python main.py --benchmark 600      run 600 frames, print the latency report as JSON, exit
                                        (--fft-size N: other FFT size, --report PATH: also save it)
    python main.py --version
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback


def _fatal(title: str, message: str) -> None:
    """Report a start-up problem on stderr and, if Qt is usable, in a dialog."""
    if sys.stderr is not None:
        print(f"{title}: {message}", file=sys.stderr)
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication(sys.argv[:1])
        QMessageBox.critical(None, title, message)
        del app
    except Exception:
        pass


def _install_exception_dialog() -> None:
    """Show unexpected errors in a dialog instead of silently killing the app."""
    from PyQt6.QtWidgets import QApplication, QMessageBox

    showing = {"active": False}

    def handler(exc_type, exc, tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        if sys.stderr is not None:
            print(details, file=sys.stderr)
        if showing["active"] or QApplication.instance() is None:
            return
        showing["active"] = True                      # a failing frame timer must not stack dialogs
        try:
            box = QMessageBox(QMessageBox.Icon.Critical, "Unexpected error", f"{exc_type.__name__}: {exc}")
            box.setInformativeText("The application will try to keep running. Details are below.")
            box.setDetailedText(details)
            box.exec()
        finally:
            showing["active"] = False

    sys.excepthook = handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="signal_processor", description="Real-time frequency signal processing")
    parser.add_argument("file", nargs="?", help=".wav or .csv file to open at start-up")
    parser.add_argument("--sample-rate", type=float, default=None, help="sample rate in Hz for 1-column CSV files")
    parser.add_argument("--benchmark", type=int, default=0, metavar="FRAMES", help="measure latency for FRAMES frames and exit")
    parser.add_argument("--report", metavar="PATH", default=None, help="also write the benchmark report to this JSON file")
    parser.add_argument("--fft-size", type=int, default=None, help="initial FFT size")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    # Make PyQtGraph use PyQt6 even if another Qt binding is installed as well.
    os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

    try:
        from dsp import FFT_SIZES, __version__
    except ImportError as exc:
        _fatal("Missing dependency", f"{exc}\n\nInstall the requirements with:\n  python -m pip install -r requirements.txt")
        return 1
    if args.version:
        print(f"Signal Processor {__version__}")
        return 0
    if args.fft_size is not None and args.fft_size not in FFT_SIZES:
        parser.error(f"--fft-size must be one of {', '.join(map(str, FFT_SIZES))}")

    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QGuiApplication
        from PyQt6.QtWidgets import QApplication

        from ui import MainWindow
    except ImportError as exc:
        _fatal("Missing dependency", f"{exc}\n\nInstall the requirements with:\n  python -m pip install -r requirements.txt")
        return 1

    # Qt 6 always enables High-DPI scaling; keep fractional factors (125 %, 150 %) exact.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Signal Processor")
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "icon.ico")
    if os.path.isfile(icon_path):
        from PyQt6.QtGui import QIcon

        app.setWindowIcon(QIcon(icon_path))
    _install_exception_dialog()

    try:
        window = MainWindow(benchmark_frames=args.benchmark)
    except Exception as exc:
        _fatal("Start-up failed", f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
        return 1

    if args.fft_size is not None:
        window.cmb_fft.setCurrentIndex(FFT_SIZES.index(args.fft_size))
    window.show()
    if args.file:
        window.load_file(args.file, sample_rate=args.sample_rate)

    exit_code = {"value": None}
    if args.benchmark:
        def on_done(report: dict) -> None:
            text = json.dumps(report, indent=2)
            print(text)
            if args.report:
                try:
                    with open(args.report, "w", encoding="utf-8") as fh:
                        fh.write(text + "\n")
                except OSError as exc:
                    _fatal("Could not write the report", str(exc))
            exit_code["value"] = 0 if report["frames_over_budget"] == 0 else 2
            app.quit()

        window.benchmark_finished.connect(on_done)

    result = app.exec()
    return result if exit_code["value"] is None else exit_code["value"]


if __name__ == "__main__":
    sys.exit(main())
