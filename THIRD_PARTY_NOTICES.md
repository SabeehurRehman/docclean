# Third-party notices

DocClean.exe bundles the following open-source components:

| Component | License | Source |
|---|---|---|
| U²-Net model (`models/u2netp.onnx`) – Qin et al., ONNX export from the rembg project | Apache-2.0 | https://github.com/xuebinqin/U-2-Net, https://github.com/danielgatis/rembg |
| OpenCV (opencv-python-headless) | Apache-2.0 | https://github.com/opencv/opencv-python |
| Qt for Python (PySide6) and Qt | LGPL-3.0 | https://code.qt.io/cgit/pyside/pyside-setup.git/ |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy |
| Pillow | MIT-CMU (HPND) | https://github.com/python-pillow/Pillow |
| Python | PSF License | https://www.python.org |
| PyInstaller bootloader | GPL-2.0 with bootloader exception | https://github.com/pyinstaller/pyinstaller |

Full license texts are available at the source links above. Qt/PySide6 are used
unmodified under the LGPL; you may replace them by rebuilding DocClean from this
repository's source (see "Build it yourself" in README.md).
