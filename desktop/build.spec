# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build configuration for Signal Processor v1.4.2
#
# Run on the operating system you are building for (PyInstaller does not cross-compile):
#
#     python -m PyInstaller build.spec --noconfirm --clean
#
# Result:
#     Windows  dist/SignalProcessor/SignalProcessor.exe
#     Linux    dist/SignalProcessor/SignalProcessor
#     macOS    dist/SignalProcessor.app   (Apple Silicon or Intel, matching the Python used)
#
# The dist/SignalProcessor folder (or the .app) is self-contained: copy it to a
# machine without Python and run it.

import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

APP_NAME = "SignalProcessor"
APP_VERSION = "1.4.2"

# soundfile ships the libsndfile shared library inside the "_soundfile_data"
# package; make sure it is bundled on every platform.
binaries = []
datas = []
for package in ("_soundfile_data",):
    try:
        binaries += collect_dynamic_libs(package)
        datas += collect_data_files(package)
    except Exception:
        pass  # wheel without bundled libsndfile (system library is used instead)

datas += [("ui/icon.ico", "ui")]

hiddenimports = [
    "PyQt6.QtMultimedia",
    "_soundfile_data",
    "soundfile",
    "_cffi_backend",
    "scipy.io.wavfile",
    "scipy.signal",
    "scipy.fft",
]

# Other Qt bindings and heavy optional packages must not be pulled in by accident.
excludes = [
    "PyQt5", "PySide2", "PySide6",
    "tkinter", "matplotlib", "IPython", "jupyter", "notebook",
    "pytest", "OpenGL", "h5py", "pandas", "PIL",
]

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,            # GUI application: no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,         # macOS: architecture of the running Python (arm64 or x86_64)
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=None,
        bundle_identifier="com.signalprocessor.app",
        version=APP_VERSION,
        info_plist={
            "CFBundleName": "Signal Processor",
            "CFBundleDisplayName": "Signal Processor",
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
        },
    )
