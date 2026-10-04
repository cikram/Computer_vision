"""OpenCV logic for the filter app. No Streamlit code here, so it can be read and tested on its own.

Conventions
- Every image is a uint8 NumPy array in OpenCV order (BGR), or a 2D array for one channel.
- Pipeline: image -> convert_space() -> apply_filter() -> display.
- Images are shown the way cv2.imshow() would show them. A converted space such as HSV
  therefore looks "strange" on purpose: its channels are not colors any more.
"""
from typing import Callable, NamedTuple, Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------------------
# Color spaces
# --------------------------------------------------------------------------------------
SPACES = {
    "Original (BGR)": None,
    "RGB": cv2.COLOR_BGR2RGB,
    "Grayscale": cv2.COLOR_BGR2GRAY,
    "HSV": cv2.COLOR_BGR2HSV,
    "HLS": cv2.COLOR_BGR2HLS,
    "LAB": cv2.COLOR_BGR2LAB,
    "YCrCb": cv2.COLOR_BGR2YCrCb,
}

# channel name -> (conversion code or None for BGR, channel index)
CHANNELS = {
    "Blue (BGR)": (None, 0),
    "Green (BGR)": (None, 1),
    "Red (BGR)": (None, 2),
    "Hue (HSV)": (cv2.COLOR_BGR2HSV, 0),
    "Saturation (HSV)": (cv2.COLOR_BGR2HSV, 1),
    "Value (HSV)": (cv2.COLOR_BGR2HSV, 2),
    "Hue (HLS)": (cv2.COLOR_BGR2HLS, 0),
    "Lightness (HLS)": (cv2.COLOR_BGR2HLS, 1),
    "Saturation (HLS)": (cv2.COLOR_BGR2HLS, 2),
    "L (LAB)": (cv2.COLOR_BGR2LAB, 0),
    "a (LAB)": (cv2.COLOR_BGR2LAB, 1),
    "b (LAB)": (cv2.COLOR_BGR2LAB, 2),
    "Y (YCrCb)": (cv2.COLOR_BGR2YCrCb, 0),
    "Cr (YCrCb)": (cv2.COLOR_BGR2YCrCb, 1),
    "Cb (YCrCb)": (cv2.COLOR_BGR2YCrCb, 2),
}

SINGLE_CHANNEL = "Single channel"
SPACE_OPTIONS = list(SPACES) + [SINGLE_CHANNEL]


def convert_space(img, space, channel=None):
    """Convert a BGR image to the chosen color space (or to one gray channel)."""
    if space == SINGLE_CHANNEL:
        code, idx = CHANNELS[channel or "Blue (BGR)"]
        x = img if code is None else cv2.cvtColor(img, code)
        return np.ascontiguousarray(x[:, :, idx])
    code = SPACES[space]
    return img.copy() if code is None else cv2.cvtColor(img, code)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def resize_max(img, max_w):
    """Shrink the image so its width is at most max_w pixels (never enlarges)."""
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, max(1, int(h * max_w / w))), interpolation=cv2.INTER_AREA)


def to_display(x):
    """Array for st.image(): gray stays 2D, 3-channel goes BGR -> RGB."""
    return x if x.ndim == 2 else cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


def to_bgr(x):
    """3-channel array for video frames."""
    return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR) if x.ndim == 2 else x


def encode_png(x):
    """PNG bytes of the image, as displayed."""
    ok, buf = cv2.imencode(".png", x)
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def _odd(k):
    """Force a positive odd kernel size."""
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def _gray(img):
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _to_u8(arr, auto_contrast):
    """Float derivative -> uint8. Auto contrast stretches the strongest response to 255."""
    arr = np.abs(arr)
    if auto_contrast:
        peak = float(arr.max())
        alpha = 255.0 / peak if peak > 0 else 1.0
        return cv2.convertScaleAbs(arr, alpha=alpha)
    return cv2.convertScaleAbs(arr)


# --------------------------------------------------------------------------------------
# Filter parameter specs (the UI is generated from these, so sliders, tooltips and
# defaults always match the filter code below)
#   kind "odd"    -> slider, odd values only, options = (min, max)
#   kind "int"    -> slider, options = (min, max)
#   kind "float"  -> slider, options = (min, max, step)
#   kind "choice" -> radio / select slider, options = list
#   kind "bool"   -> checkbox
#   show_if       -> None, or a function(values_so_far) -> bool
#   help          -> tooltip text
# --------------------------------------------------------------------------------------
class Param(NamedTuple):
    key: str
    label: str
    kind: str
    options: object
    default: object
    show_if: Optional[Callable] = None
    help: str = ""


_SIGMA_HELP = "Spread of the blur. Higher means a stronger blur. 0 lets OpenCV choose it from the kernel size."


def _smooth_params(default_on):
    """'Smooth before' checkbox + the smoothing filter to use, shared by Sobel, Laplacian and Canny.

    A derivative amplifies noise, so these three filters can blur the image first.
    The type selector and its sliders only appear once the box is ticked.
    """
    on = lambda v: v["pre"]
    return [
        Param("pre", "Smooth before", "bool", None, default_on, None,
              "Smooth the image first. A derivative amplifies noise, so this usually gives cleaner edges."),
        Param("smooth", "Smoothing filter", "choice", ["Gaussian", "Median", "Bilateral"], "Gaussian", on,
              "Which filter smooths the image first. The TP tests Gaussian, Median and Bilateral before Canny."),
        Param("g_ksize", "Blur kernel size", "odd", (3, 31), 5,
              lambda v: v["pre"] and v["smooth"] == "Gaussian",
              "Window of the Gaussian blur applied first. Odd values only."),
        Param("g_sigma", "Blur sigma (0 = auto)", "float", (0.0, 20.0, 0.5), 0.0,
              lambda v: v["pre"] and v["smooth"] == "Gaussian", _SIGMA_HELP),
        Param("m_ksize", "Median kernel size", "odd", (3, 31), 5,
              lambda v: v["pre"] and v["smooth"] == "Median", "Window of the median filter applied first."),
        Param("b_d", "Bilateral diameter", "int", (1, 15), 9,
              lambda v: v["pre"] and v["smooth"] == "Bilateral", "Neighborhood diameter of the bilateral filter."),
        Param("b_sc", "Bilateral sigma color", "int", (10, 200), 75,
              lambda v: v["pre"] and v["smooth"] == "Bilateral", "How different two colors can be and still be mixed."),
        Param("b_ss", "Bilateral sigma space", "int", (10, 200), 75,
              lambda v: v["pre"] and v["smooth"] == "Bilateral", "How far apart two pixels can influence each other."),
    ]


_AUTO_HELP = "Stretch the strongest response to white so edges are easy to see. Turn it off for the plain TP output."

FILTERS = {
    "Mean": [
        Param("ksize", "Kernel size", "odd", (3, 31), 5, None,
              "Size of the square window. Bigger means a stronger blur and less detail."),
    ],
    "Gaussian": [
        Param("ksize", "Kernel size", "odd", (3, 31), 7, None,
              "Size of the window. It should be about 6 x sigma + 1, or sigma has no extra effect."),
        Param("sigma", "Sigma (0 = auto)", "float", (0.0, 20.0, 0.5), 0.0, None, _SIGMA_HELP),
    ],
    "Median": [
        Param("ksize", "Kernel size", "odd", (3, 31), 5, None,
              "Size of the window. Larger removes more noise but also erases fine details."),
    ],
    "Bilateral": [
        Param("d", "Diameter", "int", (1, 15), 9, None,
              "Neighborhood diameter in pixels. Bigger is smoother and slower."),
        Param("sigma_color", "Sigma color", "int", (10, 200), 75, None,
              "How different two colors can be and still be mixed. Higher blurs across edges more."),
        Param("sigma_space", "Sigma space", "int", (10, 200), 75, None,
              "How far apart two pixels can be and still influence each other."),
    ],
    "Sobel": [
        Param("direction", "Direction", "choice", ["X", "Y", "Magnitude"], "Magnitude", None,
              "X finds vertical edges, Y finds horizontal edges, Magnitude combines both."),
        Param("ksize", "Kernel size", "choice", [1, 3, 5, 7], 3, None,
              "Size of the derivative window. Bigger is less sensitive to noise but gives thicker edges."),
        Param("auto", "Auto contrast", "bool", None, True, None, _AUTO_HELP),
    ]
    + _smooth_params(False),
    "Laplacian": [
        Param("ksize", "Kernel size", "choice", [1, 3, 5, 7], 3, None,
              "Size of the second-derivative window. Bigger is less sensitive to noise."),
        Param("auto", "Auto contrast", "bool", None, True, None, _AUTO_HELP),
    ]
    + _smooth_params(False),
    "Canny": [
        Param("low", "Low threshold", "int", (0, 255), 100, None,
              "Edges weaker than this are dropped."),
        Param("high", "High threshold", "int", (0, 255), 200, None,
              "Edges stronger than this are always kept. In between, an edge is kept only if it "
              "touches a strong one. Usually 2 to 3 times the low value."),
    ]
    + _smooth_params(True),  # Canny does not smooth by itself, so the TP's Gaussian + Canny is on by default
}

FILTER_NAMES = list(FILTERS)

# Shortcuts for the sidebar
GROUPS = {
    "Smooth": ["Mean", "Gaussian", "Median", "Bilateral"],
    "Edges": ["Sobel", "Laplacian", "Canny"],
    "All": FILTER_NAMES,
}


def defaults(name):
    """Default parameter values of a filter."""
    return {p.key: p.default for p in FILTERS[name]}


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------
def _smooth_before(img, p):
    """Optional smoothing before a derivative filter, chosen by the parameters."""
    if not p.get("pre"):
        return img
    mode = p.get("smooth", "Gaussian")
    if mode == "Median":
        return cv2.medianBlur(img, _odd(p.get("m_ksize", 5)))
    if mode == "Bilateral":
        return cv2.bilateralFilter(
            img, int(p.get("b_d", 9)), float(p.get("b_sc", 75)), float(p.get("b_ss", 75))
        )
    k = _odd(p.get("g_ksize", 5))
    return cv2.GaussianBlur(img, (k, k), float(p.get("g_sigma", 0.0)))


def apply_filter(img, name, p=None):
    """Apply one filter. Used by the photo grid and the live camera."""
    p = {**defaults(name), **(p or {})} if name in FILTERS else {}

    if name == "Mean":
        k = _odd(p["ksize"])
        return cv2.blur(img, (k, k))

    if name == "Gaussian":
        k = _odd(p["ksize"])
        return cv2.GaussianBlur(img, (k, k), float(p["sigma"]))

    if name == "Median":
        return cv2.medianBlur(img, _odd(p["ksize"]))

    if name == "Bilateral":
        return cv2.bilateralFilter(
            img, int(p["d"]), float(p["sigma_color"]), float(p["sigma_space"])
        )

    if name == "Sobel":
        gray = _gray(_smooth_before(img, p))
        k = int(p["ksize"])
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        if p["direction"] == "X":
            out = gx
        elif p["direction"] == "Y":
            out = gy
        else:
            out = cv2.magnitude(gx, gy)
        return _to_u8(out, p["auto"])

    if name == "Laplacian":
        gray = _gray(_smooth_before(img, p))
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=int(p["ksize"]))
        return _to_u8(lap, p["auto"])

    if name == "Canny":
        gray = _gray(_smooth_before(img, p))
        low = int(p["low"])
        high = max(int(p["high"]), low)  # the high threshold must not be below the low one
        return cv2.Canny(gray, low, high)

    return img.copy()  # "None" or unknown name


# --------------------------------------------------------------------------------------
# Live camera: several filters side by side on the same frame
# --------------------------------------------------------------------------------------
PANEL_RATIO = 4 / 3  # every panel is cropped to 4:3, so the layout (and its shape) is predictable


def panel_count(view, n_filters):
    """How many panels the live view shows."""
    if view == "Original only":
        return 1
    return n_filters if view == "Filtered only" else n_filters + 1


def layout(n_panels):
    """(columns, rows, panel width) for 1 to 4 panels. More panels means smaller ones, to stay fast."""
    return {1: (1, 1, 540), 2: (2, 1, 480), 3: (3, 1, 360), 4: (2, 2, 400)}[n_panels]


def aspect(n_panels):
    """Width / height of the whole live picture."""
    cols, rows, _ = layout(n_panels)
    return cols * 4 / (rows * 3)


def _center_crop(img, ratio):
    h, w = img.shape[:2]
    if w / h > ratio:  # too wide: trim the sides
        new_w = int(h * ratio)
        x = (w - new_w) // 2
        return img[:, x : x + new_w]
    new_h = int(w / ratio)  # too tall: trim top and bottom
    y = (h - new_h) // 2
    return img[y : y + new_h]


def _label(tile, text):
    """Write the panel name in its top-left corner (white text with a dark outline)."""
    scale = max(0.45, tile.shape[1] / 800)
    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(tile, text, (10, int(24 * scale / 0.6)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thickness, cv2.LINE_AA)
    return tile


def process_frame(frame, space, channel, panels, view):
    """One live frame: the original and/or one panel per filter, as a single BGR picture.

    panels is a list of (filter name, parameters), one to three entries.
    """
    n = panel_count(view, len(panels))
    cols, rows, width = layout(n)
    height = width * 3 // 4
    frame = _center_crop(frame, PANEL_RATIO)
    shrinking = frame.shape[1] > width
    frame = cv2.resize(frame, (width, height),
                       interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)
    src = convert_space(frame, space, channel)

    tiles = []
    if view != "Filtered only":
        tiles.append(_label(to_bgr(src).copy(), "Original"))
    if view != "Original only":
        for name, params in panels:
            tiles.append(_label(to_bgr(apply_filter(src, name, params)).copy(), name))
    return np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])
