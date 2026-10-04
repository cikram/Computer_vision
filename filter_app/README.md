# Filter Studio

Apply OpenCV filters to a photo or to a live camera feed, and tune every parameter with sliders.
Built for the ENSAM *Vision par Ordinateur* mini-project (TP7). Light and dark themes.

## Features

- **Photo mode:** upload a JPG or PNG. All filters appear at once in a grid. Each card has its own
  sliders and its own PNG download button. Moving a slider updates only that card.
- **Camera mode:** pick one filter and see it applied in real time on your webcam. The sliders act on every frame.
- **View:** side by side, filtered only, or original only (photo and camera).
- **Color space:** original (BGR), RGB, grayscale, HSV, HLS, LAB, YCrCb, or a single channel (B, G, R, H, S, V, ...).
  Channels are drawn the way `cv2.imshow` draws them, so HSV or LAB can look unusual. That is expected.
- **Filters:** Mean, Gaussian, Median, Bilateral, Sobel, Laplacian, Canny.
  - Sobel and Laplacian have an optional Gaussian blur before them, plus auto contrast.
  - Canny has a "Smooth before" choice (Gaussian, Median, Bilateral) for the Gaussian+Canny,
    Median+Canny and Bilateral+Canny tests of the TP.
- **Dark mode** toggle in the top right.

## Setup (Windows, PowerShell, from the `Computer_Vision` folder)

Reuse the existing `cv` virtual environment and add the two missing packages:

```powershell
.\cv\Scripts\python.exe -m pip install streamlit streamlit-webrtc
cd filter_app
..\cv\Scripts\python.exe -m streamlit run app.py
```

Calling `python.exe` directly avoids PowerShell blocking `Activate.ps1`.
To start from a new environment instead:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens at http://localhost:8501.

Do not install `opencv-python-headless` next to `opencv-python`: both provide `cv2` and they conflict.

## Camera tips

- Click **Start** and allow camera access when the browser asks.
- If it was blocked, click the lock icon in the address bar, allow the camera, and reload.
- Frames are limited to 480 px wide so Bilateral stays smooth.

## Files

| File | Role |
|---|---|
| `app.py` | Interface: sidebar, photo grid, camera |
| `filters.py` | OpenCV logic: color spaces, the 7 filters, `apply_filter`, `process_frame`. No Streamlit code |
| `theme.py` | Light and dark CSS, animated header |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Base theme and accent color |

## Notes

- Uploaded photos are shrunk to 800 px wide so the grid stays fast.
- Odd kernel sizes are enforced; even values are bumped to the next odd number.
- Canny keeps the high threshold at or above the low one.
- Sobel and Laplacian are computed in 64-bit float, then converted with `convertScaleAbs`.
  With *Auto contrast* on, the strongest response is stretched to 255. Turn it off to get the plain TP output.
