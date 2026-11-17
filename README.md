# DocClean

Windows desktop app that takes raw photos of documents, removes the background
(desk, hands, table), straightens the page, and makes the text sharp and readable.

## Features
- Add photos by button or drag & drop (files or whole folders)
- Automatic page/card detection using a built-in AI model (U²-Net, runs offline) with a
  classic edge-detection fallback – works on busy backgrounds like patterned fabric
- Perspective correction ("scanner" view)
- Drag the blue corners to fix the crop by hand; "Use whole image" to skip cropping
- Shadow / uneven-lighting removal that keeps photos on ID cards intact, sharpening (adjustable), denoising
- Styles: **Enhanced color**, **Grayscale**, **Black & white** (best for text), **Original**
- Rotate pages; save one page, all pages as PNG, or all pages as one PDF

## Get the .exe

**Option A – build on any Windows PC**
1. Install Python 3.12 from python.org (tick *Add python.exe to PATH*).
2. Copy this folder to the PC and double-click `build_windows.bat`.
3. The app appears at `dist\DocClean.exe` – copy it anywhere and run it.

**Option B – let GitHub build it**
Push this folder to a GitHub repo, open *Actions → Build Windows app → Run workflow*,
then download `DocClean.exe` from the finished run's artifacts.

**Run from source (no build)**: double-click `run_windows.bat`.

Note: Windows SmartScreen may warn the first time because the .exe is unsigned –
click *More info → Run anyway*.

## Credits
Object detection model `models/u2netp.onnx` – U²-Net (Qin et al., Apache-2.0),
ONNX export from the rembg project.
