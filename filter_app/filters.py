"""OpenCV logic for the filter app. No Streamlit code here, so it can be read and tested on its own.

Conventions
- Every image is a uint8 NumPy array in OpenCV order (BGR), or a 2D array for one channel.
- Pipeline: image -> convert_space() -> apply_filter() -> display.
- Images are shown the way cv2.imshow() would show them. A converted space such as HSV
  therefore looks "strange" on purpose: its channels are not colors any more.
"""
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
# Filter parameter specs (the UI is generated from these, so the sliders always match)
# Each param: (key, label, kind, options, default, show_if)
#   kind "odd"    -> slider, odd values only, options = (min, max)
#   kind "int"    -> slider, options = (min, max)
#   kind "float"  -> slider, options = (min, max, step)
#   kind "choice" -> radio / select, options = list
#   kind "bool"   -> checkbox
#   show_if       -> None or a function(values_so_far) -> bool
# --------------------------------------------------------------------------------------
def _gauss_params(show_if):
    return [
        ("g_ksize", "Blur kernel size", "odd", (3, 31), 5, show_if),
        ("g_sigma", "Blur sigma (0 = auto)", "float", (0.0, 20.0, 0.5), 0.0, show_if),
    ]


def _pre_gauss():
    """Optional Gaussian blur before a derivative filter (Sobel, Laplacian)."""
    return [("pre", "Gaussian blur before", "bool", None, False, None)] + _gauss_params(
        lambda v: v["pre"]
    )


FILTERS = {
    "Mean": [
        ("ksize", "Kernel size", "odd", (3, 31), 5, None),
    ],
    "Gaussian": [
        ("ksize", "Kernel size", "odd", (3, 31), 7, None),
        ("sigma", "Sigma (0 = auto)", "float", (0.0, 20.0, 0.5), 0.0, None),
    ],
    "Median": [
        ("ksize", "Kernel size", "odd", (3, 31), 5, None),
    ],
    "Bilateral": [
        ("d", "Diameter", "int", (1, 15), 9, None),
        ("sigma_color", "Sigma color", "int", (10, 200), 75, None),
        ("sigma_space", "Sigma space", "int", (10, 200), 75, None),
    ],
    "Sobel": [
        ("direction", "Direction", "choice", ["X", "Y", "Magnitude"], "Magnitude", None),
        ("ksize", "Kernel size", "choice", [1, 3, 5, 7], 3, None),
        ("auto", "Auto contrast", "bool", None, True, None),
    ]
    + _pre_gauss(),
    "Laplacian": [
        ("ksize", "Kernel size", "choice", [1, 3, 5, 7], 3, None),
        ("auto", "Auto contrast", "bool", None, True, None),
    ]
    + _pre_gauss(),
    "Canny": [
        ("low", "Low threshold", "int", (0, 255), 100, None),
        ("high", "High threshold", "int", (0, 255), 200, None),
        ("smooth", "Smooth before", "choice", ["None", "Gaussian", "Median", "Bilateral"],
         "Gaussian", None),
    ]
    + _gauss_params(lambda v: v["smooth"] == "Gaussian")
    + [
        ("m_ksize", "Median kernel size", "odd", (3, 31), 5, lambda v: v["smooth"] == "Median"),
        ("b_d", "Bilateral diameter", "int", (1, 15), 9, lambda v: v["smooth"] == "Bilateral"),
        ("b_sc", "Bilateral sigma color", "int", (10, 200), 75,
         lambda v: v["smooth"] == "Bilateral"),
        ("b_ss", "Bilateral sigma space", "int", (10, 200), 75,
         lambda v: v["smooth"] == "Bilateral"),
    ],
}

FILTER_NAMES = list(FILTERS)

# One line per filter, shown on the cards
DESCRIPTIONS = {
    "Mean": "Average of the neighbors. Fast, blurs edges.",
    "Gaussian": "Weighted average, closer pixels count more.",
    "Median": "Median of the neighbors. Best for salt and pepper noise.",
    "Bilateral": "Smooths flat areas, keeps edges sharp.",
    "Sobel": "First derivative. Finds edges and their direction.",
    "Laplacian": "Second derivative. Highlights fast intensity changes.",
    "Canny": "Thin, clean edges with double thresholding.",
}


def defaults(name):
    """Default parameter values of a filter."""
    return {key: default for key, _, _, _, default, _ in FILTERS[name]}


# --------------------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------------------
def _smooth_before(img, p):
    """Optional smoothing before a derivative filter, chosen by the parameters."""
    mode = p.get("smooth", "Gaussian" if p.get("pre") else "None")
    if mode == "Gaussian":
        k = _odd(p.get("g_ksize", 5))
        return cv2.GaussianBlur(img, (k, k), float(p.get("g_sigma", 0.0)))
    if mode == "Median":
        return cv2.medianBlur(img, _odd(p.get("m_ksize", 5)))
    if mode == "Bilateral":
        return cv2.bilateralFilter(
            img, int(p.get("b_d", 9)), float(p.get("b_sc", 75)), float(p.get("b_ss", 75))
        )
    return img


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


def process_frame(frame, space, channel, name, params, view, max_w=480):
    """One video frame for the live camera: source and/or filtered result, as BGR."""
    frame = resize_max(frame, max_w)
    src = convert_space(frame, space, channel)
    if view == "Original only":
        return to_bgr(src)
    out = to_bgr(apply_filter(src, name, params))
    if view == "Filtered only":
        return out
    return np.hstack([to_bgr(src), out])
