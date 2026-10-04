"""Filter Studio: apply OpenCV filters to a photo or to a live camera feed.

Run with:  streamlit run app.py
Files:     app.py (interface)  |  filters.py (OpenCV logic)  |  theme.py (look)
"""
import cv2
import numpy as np
import streamlit as st

import filters as F
import theme

MAX_WIDTH = 800  # uploaded photos are shrunk to this width so the grid stays fast
VIEWS = ["Side by side", "Filtered only", "Original only"]

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
            self.cfg = ("Original (BGR)", None, "None", {}, "Filtered only")

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
    for key, label, kind, opts, default, show_if in spec:
        if show_if is not None and not show_if(values):
            continue
        wkey = f"{prefix}_{key}"  # unique per filter, so cards never share values
        if kind == "odd":  # odd sizes only: odd minimum + step 2
            values[key] = box.slider(label, opts[0], opts[1], default, step=2, key=wkey)
        elif kind == "int":
            values[key] = box.slider(label, opts[0], opts[1], default, key=wkey)
        elif kind == "float":
            values[key] = box.slider(label, opts[0], opts[1], default, step=opts[2], key=wkey)
        elif kind == "bool":
            values[key] = box.checkbox(label, value=default, key=wkey)
        elif all(isinstance(o, int) for o in opts):
            values[key] = box.select_slider(label, options=opts, value=default, key=wkey)
        elif len(opts) <= 3:
            values[key] = box.radio(
                label, opts, index=opts.index(default), horizontal=True, key=wkey
            )
        else:
            values[key] = box.selectbox(label, opts, index=opts.index(default), key=wkey)
    return values


@st.cache_data(show_spinner=False)
def load_image(data):
    """Decode uploaded bytes to a BGR array (None if the file is not an image)."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else F.resize_max(img, MAX_WIDTH)


# --------------------------------------------------------------------------------------
# Photo mode
# --------------------------------------------------------------------------------------
@fragment
def filter_card(name, src, view):
    """One filter: title, image(s), its own sliders, download button."""
    with st.container(key=f"card_{name}"):
        st.markdown(f'<span class="tag">{name}</span>', unsafe_allow_html=True)
        image_slot = st.container()  # filled after the sliders are read
        params = render_params(F.FILTERS[name], name, st)
        try:
            out = F.apply_filter(src, name, params)
        except Exception as err:
            st.error(f"{name} failed with these settings: {err}")
            return
        with image_slot:
            if view == "Side by side":
                left, right = st.columns(2)
                left.image(F.to_display(src), caption="Source", width="stretch")
                right.image(F.to_display(out), caption="Filtered", width="stretch")
            else:
                st.image(F.to_display(out), width="stretch")
        st.download_button(
            "Download PNG",
            data=F.encode_png(out),
            file_name=f"{name.lower()}.png",
            mime="image/png",
            key=f"dl_{name}",
            on_click="ignore",
        )


def photo_mode(space, channel, view, selected):
    upload = st.file_uploader(
        "Upload a photo", type=["jpg", "jpeg", "png"], key="upload", label_visibility="collapsed"
    )
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
        st.image(F.to_display(src), caption="Source", width="stretch")
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
def camera_mode(cfg):
    if not CAMERA_READY:
        st.error("Camera mode needs streamlit-webrtc. Run: pip install streamlit-webrtc")
        return
    ctx = webrtc_streamer(
        key="camera",
        video_processor_factory=FilterProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )
    if ctx.video_processor:
        ctx.video_processor.update(cfg)  # slider values reach the next frame
    st.caption(
        "Press Start and allow camera access. If the browser blocks it, click the lock icon "
        "in the address bar, allow the camera, and reload. Frames are limited to 480 px wide."
    )


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------
theme.inject(st.session_state.get("dark", False))  # read before the toggle is drawn

top_left, top_right = st.columns([6, 1.4], vertical_alignment="center")
with top_left:
    theme.brand()
with top_right:
    st.toggle("Dark mode", key="dark")

theme.hero()

with st.sidebar:
    mode = st.radio("Mode", ["Photo", "Camera"], horizontal=True, key="mode")
    view = st.radio("View", VIEWS, key="view")
    space = st.selectbox("Color space", F.SPACE_OPTIONS, key="space")
    channel = None
    if space == F.SINGLE_CHANNEL:
        channel = st.selectbox("Channel", list(F.CHANNELS), key="channel")
    st.divider()
    if mode == "Photo":
        selected = st.multiselect(
            "Filters to show", F.FILTER_NAMES, default=F.FILTER_NAMES, key="selected"
        )
    else:
        options = ["None"] + F.FILTER_NAMES
        cam_filter = st.selectbox("Filter", options, index=options.index("Gaussian"), key="cam_filter")
        cam_params = render_params(F.FILTERS.get(cam_filter, []), f"cam_{cam_filter}", st.sidebar)

if mode == "Photo":
    photo_mode(space, channel, view, selected)
else:
    camera_mode((space, channel, cam_filter, cam_params, view))
