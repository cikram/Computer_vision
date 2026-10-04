# Filter Studio

Apply OpenCV filters to a photo or to a live camera feed, and tune every parameter with sliders.
Built for the ENSAM *Vision par Ordinateur* mini-project (TP7). Light and dark themes.

## Features

- **Photo mode:** upload a JPG or PNG. All filters appear at once in a grid. Each card has its own
  sliders, and two round icon buttons at the top right: reset (↺) and download PNG (⤓). Moving a slider updates only that card.
- **Camera mode:** apply one, two or three filters at once to the live webcam. Choose how many with
  *Filters on screen*, then pick a filter and set its sliders in each *Filter A / B / C* box.
  The picture shows the original plus one labelled panel per filter (1 filter: side by side, 2: a row of three,
  3: a 2x2 grid). The sliders act on every frame. Panels are cropped to 4:3 and kept small so the video stays smooth.
- **Header:** the conveyor shows your own photo going in and its filtered versions coming out.
  It shrinks once a photo is loaded or the camera is on, and keeps animating.
- **Sidebar:**
  - *Mode* and *View* (Compare, Filtered, Original) are one-click buttons.
  - *Filters to show* are pills you click on and off, with shortcuts **Smooth**, **Edges** and **All**.
  - **Reset all filters** puts every slider back to its default.
  - Every slider has a **?** tooltip that explains what it does.
- **Color space:** original (BGR), RGB, grayscale, HSV, HLS, LAB, YCrCb, or a single channel (B, G, R, H, S, V, ...).
  Channels are drawn the way `cv2.imshow` draws them, so HSV or LAB can look unusual. That is expected.
- **Filters:** Mean, Gaussian, Median, Bilateral, Sobel, Laplacian, Canny.
  - Sobel, Laplacian and Canny share a **Smooth before** checkbox. When ticked, pick the smoothing filter
    (Gaussian, Median or Bilateral) and its sliders. Canny starts ticked with Gaussian; this covers the
    Gaussian+Canny, Median+Canny and Bilateral+Canny tests of the TP.
  - Sobel and Laplacian also have auto contrast.
- **Dark mode** toggle in the top right.
- Your photo is kept when you switch to the camera and back.

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
Streamlit 1.50 or newer is required (segmented buttons and pills); `pip install streamlit` gets the latest.

## Camera tips

- Click **Start** and allow camera access when the browser asks.
- If it was blocked, click the lock icon in the address bar, allow the camera, and reload.
- Each panel is at most 540 px wide (360 px with three panels) so Bilateral stays smooth.

## Files

| File | Role |
|---|---|
| `app.py` | Interface: sidebar, photo grid, camera |
| `filters.py` | OpenCV logic: color spaces, the 7 filters, `apply_filter`, `process_frame`. No Streamlit code |
| `theme.py` | Light and dark CSS, animated header with your photo |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Base theme and accent color |

## Notes

- Uploaded photos are shrunk to 800 px wide so the grid stays fast.
- Odd kernel sizes are enforced; even values are bumped to the next odd number.
- Canny keeps the high threshold at or above the low one.
- Sobel and Laplacian are computed in 64-bit float, then converted with `convertScaleAbs`.
  With *Auto contrast* on, the strongest response is stretched to 255. Turn it off to get the plain TP output.
