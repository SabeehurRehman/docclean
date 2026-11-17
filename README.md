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

## Download

**[⬇ Download DocClean.exe for Windows](https://github.com/SabeehurRehman/docclean/releases/latest/download/DocClean.exe)**
(Windows 10/11, 64-bit, ~110 MB, no installation needed)

All versions: [Releases page](https://github.com/SabeehurRehman/docclean/releases)

First launch: if Windows shows *"Windows protected your PC"*, click **More info → Run anyway**
(the app isn't code-signed). It takes a few seconds to start the first time.

Your photos never leave your computer – all processing, including the AI page
detection, runs offline.

## How to use
1. Click **Add images** (or drag photos/folders onto the window).
2. Check each page – drag the blue corners if the crop needs adjusting.
3. Pick a **Style** and **Sharpness**, rotate if needed.
4. **Save page**, **Save all** (PNG files) or **Save all as PDF**.

## Build it yourself

**Option A – build on any Windows PC**
1. Install Python 3.12 from python.org (tick *Add python.exe to PATH*).
2. Copy this folder to the PC and double-click `build_windows.bat`.
3. The app appears at `dist\DocClean.exe` – copy it anywhere and run it.

**Option B – GitHub Actions**
Every push builds and self-tests `DocClean.exe`; pushing a tag like `v1.0.1` also
publishes it as a new public release.

**Run from source (no build)**: double-click `run_windows.bat`.

Note: Windows SmartScreen may warn the first time because the .exe is unsigned –
click *More info → Run anyway*.

## Credits
Object detection model `models/u2netp.onnx` – U²-Net (Qin et al., Apache-2.0),
ONNX export from the rembg project. See THIRD_PARTY_NOTICES.md for all bundled components.
