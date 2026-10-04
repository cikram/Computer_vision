"""Filter Studio: apply OpenCV filters to a photo or to a live camera feed.

Run with:  streamlit run app.py
Files:     app.py (interface)  |  filters.py (OpenCV logic)  |  theme.py (look)
"""
import base64

import cv2
import numpy as np
import streamlit as st

import filters as F
import theme

MAX_WIDTH = 800  # uploaded photos are shrunk to this width so the grid stays fast
VIEWS = ["Side by side", "Filtered only", "Original only"]
CAM_DEFAULTS = ["Gaussian", "Canny", "Sobel"]  # what Filter A, B and C start with in camera mode

st.set_page_config(page_title="Filter Studio", layout="wide", initial_sidebar_state="auto")

# st.fragment reruns one card alone when its slider moves; plain call on older versions
fragment = getattr(st, "fragment", lambda fn: fn)

# --------------------------------------------------------------------------------------
# Live camera (needs streamlit-webrtc). The processor is defined only if the package exists.
# --------------------------------------------------------------------------------------
try:
    import av
    from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

    CAMERA_READY = True
except ImportError:
    CAMERA_READY = False

if CAMERA_READY:

    class FilterProcessor(VideoProcessorBase):
        """Runs on every frame. The sliders update self.cfg, so changes show instantly."""

        def __init__(self):
            self.cfg = ("Original (BGR)", None, [("None", {})], "Compare")

        def update(self, cfg):
            self.cfg = cfg  # one assignment, safe across threads

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            try:
                out = F.process_frame(img, *self.cfg)
            except Exception:  # never stop the stream because of one bad frame
                out = img
            return av.VideoFrame.from_ndarray(out, format="bgr24")


# --------------------------------------------------------------------------------------
# Small UI helpers
# --------------------------------------------------------------------------------------
def render_params(spec, prefix, box):
    """Draw the widgets of one filter inside `box` and return {param: value}."""
    values = {}
    for p in spec:
        if p.show_if is not None and not p.show_if(values):
            continue
        wkey = f"{prefix}_{p.key}"  # unique per filter, so cards never share values
        if p.kind == "odd":  # odd sizes only: odd minimum + step 2
            values[p.key] = box.slider(p.label, p.options[0], p.options[1], p.default,
                                       step=2, key=wkey, help=p.help)
        elif p.kind == "int":
            values[p.key] = box.slider(p.label, p.options[0], p.options[1], p.default,
                                       key=wkey, help=p.help)
        elif p.kind == "float":
            values[p.key] = box.slider(p.label, p.options[0], p.options[1], p.default,
                                       step=p.options[2], key=wkey, help=p.help)
        elif p.kind == "bool":
            values[p.key] = box.checkbox(p.label, value=p.default, key=wkey, help=p.help)
        elif all(isinstance(o, int) for o in p.options):
            values[p.key] = box.select_slider(p.label, options=p.options, value=p.default,
                                              key=wkey, help=p.help)
        elif len(p.options) <= 3:
            values[p.key] = box.radio(p.label, p.options, index=p.options.index(p.default),
                                      horizontal=True, key=wkey, help=p.help)
        else:
            values[p.key] = box.selectbox(p.label, p.options, index=p.options.index(p.default),
                                          key=wkey, help=p.help)
    return values


def reset_params(prefix, name):
    """Put the sliders of one filter back to their defaults (runs as a button callback)."""
    for p in F.FILTERS.get(name, []):
        key = f"{prefix}_{p.key}"
        if key in st.session_state:
            st.session_state[key] = p.default


def reset_all():
    for name in F.FILTER_NAMES:
        reset_params(name, name)


def pick_group(group):
    st.session_state["selected"] = list(F.GROUPS[group])


def keep_one(key, fallback):
    """Segmented controls and pills can be un-clicked. Put the last real choice back instead."""
    value = st.session_state.get(key)
    if value:
        st.session_state[f"{key}__last"] = value
    else:
        st.session_state[key] = st.session_state.get(f"{key}__last", fallback)


def show(img, **kwargs):
    st.image(F.to_display(img), width="stretch", **kwargs)


@st.cache_data(show_spinner=False)
def load_image(data):
    """Decode uploaded bytes to a BGR array (None if the file is not an image)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else F.resize_max(img, MAX_WIDTH)


def _data_uri(x):
    ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


@st.cache_data(show_spinner=False, max_entries=8)
def hero_photo(data, name):
    """Small copies of the upload for the header belt: the raw photo, then six filtered versions."""
    img = load_image(data)
    if img is None:
        return None
    small = F.resize_max(img, 260)
    steps = [  # card title, detail, result
        ("Gaussian", "ksize 11", F.apply_filter(small, "Gaussian", {"ksize": 11})),
        ("Canny", "100 / 200", F.apply_filter(small, "Canny", {})),
        ("Grayscale", "BGR2GRAY", F.convert_space(small, "Grayscale")),
        ("Median", "ksize 7", F.apply_filter(small, "Median", {"ksize": 7})),
        ("Sobel", "ksize 3", F.apply_filter(small, "Sobel", {"pre": True})),
        ("Laplacian", "ksize 3", F.apply_filter(small, "Laplacian", {"pre": True})),
    ]
    images = [small] + [result for _, _, result in steps]
    css = "".join(f".pic.th{i}{{background-image:url({_data_uri(im)})}}" for i, im in enumerate(images))
    shown = name if len(name) <= 13 else name[:12] + "…"
    return {
        "css": css,
        "label": shown,
        "size": f"{img.shape[1]}×{img.shape[0]}",
        "filtered": [(f"th{i + 1}", t, d) for i, (t, d, _) in enumerate(steps)],
    }


# --------------------------------------------------------------------------------------
# Photo mode
# --------------------------------------------------------------------------------------
@fragment
def filter_card(name, src, view):
    """One filter: title with reset and download icons, image(s), and its own sliders."""
    with st.container(key=f"card_{name}"):
        head = st.container(key=f"head_{name}")  # filled below, once the result exists
        image_slot = st.container()
        params = render_params(F.FILTERS[name], name, st)
        out, error = None, None
        try:
            out = F.apply_filter(src, name, params)
        except Exception as err:
            error = err
        with head:
            title, reset_col, save_col = st.columns(3, vertical_alignment="center")
            title.markdown(f'<span class="tag">{name}</span>', unsafe_allow_html=True)
            reset_col.button(":material/restart_alt:", key=f"rs_{name}", help="Reset this filter",
                             on_click=reset_params, args=(name, name))
            if out is not None:
                save_col.download_button(":material/download:", data=F.encode_png(out),
                                         file_name=f"{name.lower()}.png", mime="image/png",
                                         key=f"dl_{name}", help="Download PNG", on_click="ignore")
        if error is not None:
            st.error(f"{name} failed with these settings: {error}")
            return
        with image_slot:
            if view == "Side by side":
                left, right = st.columns(2)
                with left:
                    show(src, caption="Source")
                with right:
                    show(out, caption="Filtered")
            else:
                show(out)


def photo_mode(upload, space, channel, view, selected):
    if upload is None:
        st.markdown(
            '<p class="empty">Drop a JPG or PNG above. Every filter appears here at once.</p>',
            unsafe_allow_html=True,
        )
        return
    img = load_image(upload.getvalue())
    if img is None:
        st.error("This file could not be read as an image. Upload a JPG or PNG.")
        return

    src = F.convert_space(img, space, channel)
    if space != "Original (BGR)":
        label = channel if space == F.SINGLE_CHANNEL else space
        st.caption(
            f"Source in {label}. Channels are drawn as cv2.imshow would draw them, "
            "so colors can look unusual."
        )

    if view == "Original only":
        show(src, caption="Source")
        return
    if not selected:
        st.info("Pick at least one filter in the sidebar.")
        return

    per_row = 2 if view == "Side by side" else 3
    for start in range(0, len(selected), per_row):
        cols = st.columns(per_row)
        for col, name in zip(cols, selected[start : start + per_row]):
            with col:
                filter_card(name, src, view)


# --------------------------------------------------------------------------------------
# Camera mode
# --------------------------------------------------------------------------------------
def camera_mode(cfg, n_panels):
    if not CAMERA_READY:
        st.error("Camera mode needs streamlit-webrtc. Run: pip install streamlit-webrtc")
        return
    with st.container(key="camera_box"):
        ctx = webrtc_streamer(
            key="camera",
            video_processor_factory=FilterProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_html_attrs={"style": {"width": "100%", "display": "block"}, "controls": False, "autoPlay": True},
            async_processing=True,
        )
    if ctx.video_processor:
        ctx.video_processor.update(cfg)  # slider values reach the next frame
    if ctx.state.playing:
        theme.fit_camera(F.aspect(n_panels))  # no empty band under the video
    st.caption(
        "Press Start and allow camera access. If the browser blocks it, click the lock icon "
        "in the address bar, allow the camera, and reload. Panels are limited in size so the video stays smooth."
    )


# --------------------------------------------------------------------------------------
# Page. The sidebar is built first so the header knows the mode; it still shows on the left.
# --------------------------------------------------------------------------------------
with st.sidebar:
    mode = st.segmented_control("Mode", ["Photo", "Camera"], default="Photo", key="mode",
                                on_change=keep_one, args=("mode", "Photo"), width="stretch") or "Photo"
    view = st.segmented_control("View", VIEWS, default=VIEWS[0], key="view",
                                format_func=lambda v: {"Side by side": "Compare"}.get(v, v.replace(" only", "")),
                                on_change=keep_one, args=("view", VIEWS[0]), width="stretch") or VIEWS[0]
    space = st.selectbox("Color space", F.SPACE_OPTIONS, key="space")
    channel = None
    if space == F.SINGLE_CHANNEL:
        channel = st.selectbox("Channel", list(F.CHANNELS), key="channel")
    st.divider()

    if mode == "Photo":
        selected = st.pills("Filters to show", F.FILTER_NAMES, selection_mode="multi",
                            default=F.FILTER_NAMES, key="selected") or []
        shortcuts = st.columns(3)
        for col, group in zip(shortcuts, F.GROUPS):
            col.button(group, key=f"grp_{group}", on_click=pick_group, args=(group,), width="stretch")
        st.button("Reset all filters", key="reset_all", on_click=reset_all, width="stretch")
    else:
        n_filters = st.segmented_control("Filters on screen", [1, 2, 3], default=1, key="n_filters",
                                         on_change=keep_one, args=("n_filters", 1), width="stretch") or 1
        panels = []
        for i in range(n_filters):
            with st.expander(f"Filter {'ABC'[i]}", expanded=True):
                name = st.pills(f"Filter {'ABC'[i]}", ["None"] + F.FILTER_NAMES, selection_mode="single",
                                default=CAM_DEFAULTS[i], key=f"cam_filter_{i}", label_visibility="collapsed",
                                on_change=keep_one, args=(f"cam_filter_{i}", CAM_DEFAULTS[i])) or CAM_DEFAULTS[i]
                params = render_params(F.FILTERS.get(name, []), f"cam{i}_{name}", st)
                if name != "None":
                    st.button("Reset filter", key=f"reset_cam{i}", on_click=reset_params,
                              args=(f"cam{i}_{name}", name), width="stretch")
            panels.append((name, params))

theme.inject(st.session_state.get("dark", False), hide_uploader=(mode == "Camera"))

top_left, top_right = st.columns([6, 1.4], vertical_alignment="center")
with top_left:
    theme.brand()
with top_right:
    st.toggle("Dark mode", key="dark")

# The uploader always exists (hidden in camera mode) so a photo survives a trip to the camera.
saved = st.session_state.get("upload")
photo = hero_photo(saved.getvalue(), saved.name) if (saved is not None and mode == "Photo") else None
theme.hero(compact=(mode == "Camera" or photo is not None), photo=photo)

with st.container(key="uploader_box"):
    upload = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="upload",
                              label_visibility="collapsed")

if mode == "Photo":
    photo_mode(upload, space, channel, view, selected)
else:
    live_view = "Compare" if view == "Side by side" else view
    camera_mode((space, channel, panels, live_view), F.panel_count(live_view, len(panels)))
