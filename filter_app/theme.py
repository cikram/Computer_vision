"""Look of the app: light and dark themes, top bar, and the animated hero.

Pure CSS injected with st.markdown. No images, no JavaScript.
"""
import html

import streamlit as st

LIGHT = {
    "bg": "#E9E8E4",
    "panel": "#F1F0EC",
    "card": "#F7F6F3",
    "field": "#FFFFFF",
    "text": "#1A1A19",
    "muted": "#6B6B66",
    "border": "rgba(26,26,25,.11)",
    "accent": "#1E7B4F",
    "accent_ink": "#FFFFFF",
    "btn": "#2A2A29",
    "btn_ink": "#FFFFFF",
    "btn_hover": "#454542",
    "shadow": "0 1px 2px rgba(26,26,25,.06), 0 10px 28px rgba(26,26,25,.07)",
    "band": "#DCDBD6",
    "bars": "rgba(255,255,255,.80)",
    "glass": "rgba(255,255,255,.62)",
    "chip": "#FFFFFF",
    "line": "rgba(26,26,25,.35)",
}

DARK = {
    "bg": "#0F0F10",
    "panel": "#151517",
    "card": "#1A1A1C",
    "field": "#242427",
    "text": "#F1F1EF",
    "muted": "#A0A09B",
    "border": "rgba(255,255,255,.11)",
    "accent": "#2FA36B",
    "accent_ink": "#08150E",
    "btn": "#F1F1EF",
    "btn_ink": "#121212",
    "btn_hover": "#D6D6D2",
    "shadow": "0 1px 2px rgba(0,0,0,.45), 0 10px 28px rgba(0,0,0,.40)",
    "band": "#1B1B1D",
    "bars": "rgba(255,255,255,.07)",
    "glass": "rgba(255,255,255,.07)",
    "chip": "#222225",
    "line": "rgba(255,255,255,.35)",
}

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: __bg__; --panel: __panel__; --card: __card__; --field: __field__;
  --text: __text__; --muted: __muted__; --border: __border__;
  --accent: __accent__; --accent-ink: __accent_ink__;
  --btn: __btn__; --btn-ink: __btn_ink__; --btn-hover: __btn_hover__;
  --shadow: __shadow__; --band: __band__; --bars: __bars__;
  --glass: __glass__; --chip: __chip__; --line: __line__;
  color-scheme: __scheme__;
}

/* ---------- page ---------- */
html, body, .stApp, .stApp *:not([data-testid="stIconMaterial"]):not([class*="material" i]) {
  font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
}
.stApp, [data-testid="stAppViewContainer"] { background: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background: transparent; pointer-events: none; }
[data-testid="stHeader"] button, [data-testid="stHeader"] a { pointer-events: auto; }
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], .stAppDeployButton, [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
.block-container { max-width: 1180px; padding-top: 3.6rem; padding-bottom: 4rem; }

.stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp span,
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] { color: var(--text); }
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stCaptionContainer"] p,
.stApp small { color: var(--muted); }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: .6rem; }
[data-testid="stSidebarCollapseButton"] button, [data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapsedControl"] button { color: var(--text); }

/* ---------- inputs (Streamlit 1.5x+ builds selects with react-aria; old baseweb kept as fallback) ---------- */
[data-testid="stSelectbox"] [role="group"], [data-testid="stMultiSelect"] [role="group"],
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"] {
  background: var(--field) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important; color: var(--text) !important;
}
[data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input, [data-baseweb="select"] * { color: var(--text) !important; }
[data-testid="stSelectbox"] [role="group"] svg, [data-testid="stMultiSelect"] [role="group"] svg { color: var(--muted); fill: currentColor; }
[data-testid$="VirtualDropdown"], [role="listbox"], [data-baseweb="popover"] > div, [data-baseweb="menu"] {
  background: var(--field) !important; color: var(--text) !important;
  border: 1px solid var(--border); border-radius: 12px;
}
[role="option"] { background: transparent !important; color: var(--text) !important; }
[role="option"] * { color: var(--text) !important; }
[role="option"]:hover, [role="option"][data-focused="true"], [role="option"][aria-selected="true"] { background: var(--border) !important; }
[data-tag], [data-baseweb="tag"] { background: var(--accent) !important; border-radius: 8px !important; }
[data-tag] *, [data-baseweb="tag"] * { color: var(--accent-ink) !important; }
[data-tag] svg, [data-baseweb="tag"] svg { fill: none; stroke: var(--accent-ink); }
[data-testid="stSidebar"] hr { border-color: var(--border); }

[data-testid="stSliderTickBar"] p, [data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: var(--muted) !important; }
[data-testid="stSliderThumbValue"] p, [data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--accent) !important; font-weight: 600; }

[data-testid="stRadio"] label, [data-testid="stCheckbox"] label, [data-testid="stToggle"] label { color: var(--text); }
[data-testid="stToggle"] label, [data-testid="stToggle"] label *, [data-testid="stCheckbox"] label { background-color: transparent; }
[data-testid="stRadio"] [role="radiogroup"] { gap: .15rem; }

/* unselected radio and checkbox: follow the theme instead of staying white */
[data-testid="stRadioOption"]:not([data-selected="true"]) > div > div:first-child { background: var(--field); border-color: var(--line); }
[data-testid="stCheckbox"] label:not(:has(input:checked)) > div:first-of-type { background: var(--field); border-color: var(--line); }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFileUploaderDropzone"] button {
  background: var(--btn); color: var(--btn-ink); border: 0; border-radius: 999px;
  padding: .45rem 1.1rem; font-weight: 500; font-size: .85rem; transition: background .15s;
}
.stButton > button *, .stDownloadButton > button *, [data-testid="stFileUploaderDropzone"] button * { color: var(--btn-ink) !important; }
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFileUploaderDropzone"] button:hover { background: var(--btn-hover); color: var(--btn-ink); }
.stDownloadButton > button { width: 100%; }

/* segmented controls (Mode, View) and pills (filters) */
button[data-variant="segmented_control"], button[data-variant="pills"] {
  background: var(--field) !important; color: var(--text) !important; border: 1px solid var(--border) !important;
  font-size: .8rem; box-shadow: none !important;
}
button[data-variant="segmented_control"] *, button[data-variant="pills"] * { color: inherit !important; background: transparent !important; }
button[data-variant="segmented_control"][data-selected="true"], button[data-variant="pills"][data-selected="true"] {
  background: var(--accent) !important; color: var(--accent-ink) !important; border-color: var(--accent) !important;
}
button[data-variant="segmented_control"] { padding: .3rem .3rem !important; flex: 1 1 0; min-width: 0; white-space: nowrap; font-size: .76rem; }
button[data-variant="pills"] { border-radius: 999px !important; font-size: .78rem; }
button[data-variant="segmented_control"]:hover, button[data-variant="pills"]:hover { border-color: var(--accent) !important; }

/* quiet buttons in the sidebar: filter groups and reset */
.stApp [class*="st-key-reset_"] button, .stApp [class*="st-key-grp_"] button {
  background: transparent; color: var(--muted); border: 1px solid var(--border);
  padding: .18rem .3rem; font-size: .74rem; min-height: 0; width: 100%; white-space: nowrap;
}
.stApp [class*="st-key-reset_"] button *, .stApp [class*="st-key-grp_"] button * { color: var(--muted) !important; }
.stApp [class*="st-key-reset_"] button:hover, .stApp [class*="st-key-grp_"] button:hover { background: var(--border); }
.stApp [class*="st-key-reset_"] button:hover *, .stApp [class*="st-key-grp_"] button:hover * { color: var(--text) !important; }

/* card header: the filter name on the left, reset and download as round icon buttons on the right */
[class*="st-key-head_"] [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; gap: .4rem; align-items: center; }
[class*="st-key-head_"] [data-testid="stColumn"] { flex: 0 0 auto !important; width: auto !important; min-width: 0 !important; }
[class*="st-key-head_"] [data-testid="stColumn"]:first-child { flex: 1 1 auto !important; }
.stApp [class*="st-key-rs_"] button, .stApp [class*="st-key-dl_"] button {
  width: 34px; height: 34px; min-height: 0; padding: 0; border-radius: 999px;
  background: transparent; border: 1px solid var(--border); color: var(--muted);
}
.stApp [class*="st-key-rs_"] button *, .stApp [class*="st-key-dl_"] button * { color: var(--muted) !important; font-size: 1.1rem; }
.stApp [class*="st-key-rs_"] button:hover, .stApp [class*="st-key-dl_"] button:hover { background: var(--border); border-color: var(--line); }
.stApp [class*="st-key-rs_"] button:hover *, .stApp [class*="st-key-dl_"] button:hover * { color: var(--text) !important; }

/* the fullscreen button Streamlit adds on hover: inside the image corner, dark glass, readable on any theme */
[data-testid="stElementToolbar"] { top: 6px !important; right: 6px !important; }
[data-testid="stElementToolbarButtonContainer"] {
  background: rgba(15, 15, 16, .62) !important; border: 1px solid rgba(255, 255, 255, .22) !important;
  border-radius: 9px !important; box-shadow: none !important; backdrop-filter: blur(6px);
}
[data-testid="stElementToolbar"] button, [data-testid="stElementToolbar"] button * { color: #fff !important; background: transparent !important; }

/* tooltips */
[data-testid="stTooltipContent"] { background: var(--btn) !important; color: var(--btn-ink) !important; border-radius: 10px; }
[data-testid="stTooltipContent"] * { color: var(--btn-ink) !important; }
[data-testid="stTooltipIcon"], [data-testid="stTooltipIcon"] svg { color: var(--text) !important; opacity: .6 !important; }

/* ---------- uploader ---------- */
[data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"] {
  background: var(--card); border: 1px dashed var(--line); border-radius: 18px; padding: 1.1rem 1.3rem;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--muted); }
[data-testid="stFileUploaderDropzone"] button * { color: var(--btn-ink) !important; }
[data-testid="stFileUploaderFile"] *, [data-testid="stFileChipName"] { color: var(--text) !important; }
[data-testid="stFileChip"] { background: var(--field) !important; border-radius: 12px; }

/* ---------- cards ---------- */
[class*="st-key-card_"], [data-testid="stVerticalBlockBorderWrapper"]:has(> div > [class*="st-key-card_"]) {
  background: var(--card); border: 1px solid var(--border); border-radius: 20px;
  box-shadow: var(--shadow); padding: 1rem 1.05rem 1.1rem;
}
[class*="st-key-card_"] { gap: .7rem; }
/* cards in the same row share one height */
[data-testid="stColumn"] :is([data-testid="stVerticalBlock"], [data-testid="stLayoutWrapper"]):has([class*="st-key-card_"]) { flex: 1 1 auto; flex-direction: column; }
[data-testid="stColumn"] [class*="st-key-card_"] { flex: 1 1 auto; }

/* live camera panel (a separate iframe that keeps its own light look) */
iframe[title*="webrtc" i] { border-radius: 18px; border: 1px solid var(--border); background: var(--card); box-shadow: var(--shadow); }
.st-key-camera_box { container-type: inline-size; margin: 0 auto; }

/* expanders (one per camera filter) */
[data-testid="stExpander"] details { background: var(--card); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
[data-testid="stExpander"] summary { background: var(--field) !important; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: var(--text) !important; font-size: .85rem; font-weight: 600; }
[data-testid="stExpander"] summary:hover { filter: brightness(.96); }
[data-testid="stImage"] img { border-radius: 14px; }
[data-testid="stImage"] [data-testid="stImageCaption"] { color: var(--muted); font-size: .72rem; }

.stApp .tag { display: inline-block; background: var(--accent); color: var(--accent-ink); font-weight: 600;
  font-size: .95rem; line-height: 1; padding: .34rem .6rem; border-radius: 8px; }
.stApp p.card-note { color: var(--muted); font-size: .78rem; margin: .35rem 0 0; line-height: 1.35; }
.stApp p.empty { color: var(--muted); font-size: .9rem; text-align: center; padding: .6rem 0 0; }

/* ---------- top bar ---------- */
.brand { display: flex; align-items: center; gap: .55rem; font-weight: 700; font-size: 1.05rem; letter-spacing: -.01em; color: var(--text); }
.logo { display: grid; grid-template-columns: 1fr 1fr; gap: 2px; width: 16px; height: 16px; }
.logo i { background: var(--accent); border-radius: 1.5px; display: block; }
.logo i:nth-child(2), .logo i:nth-child(3) { opacity: .45; }

/* ---------- hero ---------- */
.hero { text-align: center; padding: 2.2rem 0 .4rem; }
.hero h1 { font-size: clamp(1.7rem, 3.6vw, 2.5rem); font-weight: 500; letter-spacing: -.025em; line-height: 1.15; margin: 0; padding: 0; color: var(--text); }
.hero h1 .hl { background: var(--accent); color: var(--accent-ink); padding: .02em .22em .06em; border-radius: .18em; }
.hero p { color: var(--muted); font-size: .95rem; margin: .9rem auto 0; max-width: 34rem; }

.belt { position: relative; height: 210px; margin: 2.2rem 0 1.4rem; }
.belt::before, .belt::after { content: ""; position: absolute; left: 0; right: 0; border-top: 1px dashed var(--line); z-index: 0; }
.belt::before { top: 2px; } .belt::after { bottom: 2px; }
.band { position: absolute; left: 0; right: 0; top: 14px; bottom: 14px; background: var(--band); overflow: hidden; border-radius: 4px; }
.bars { position: absolute; inset: 18px 0; background: repeating-linear-gradient(90deg, var(--bars) 0 7px, transparent 7px 21px); border-radius: 6px; animation: bars 2.4s linear infinite; }
.lane { position: absolute; top: 0; bottom: 0; overflow: hidden; }
.lane.l { left: 0; right: 50%; } .lane.r { left: 50%; right: 0; }
.track { display: flex; align-items: center; gap: 34px; height: 100%; width: max-content; animation: slide 30s linear infinite; padding-right: 34px; }
.lane.l .track { padding-left: 0; }

.pc { flex: none; width: 150px; height: 112px; border-radius: 16px; background: var(--glass); border: 1px solid var(--border);
  box-shadow: var(--shadow); backdrop-filter: blur(6px); padding: 8px; display: flex; flex-direction: column; gap: 6px; }
.pc:nth-child(odd) { transform: rotate(-4deg); } .pc:nth-child(even) { transform: rotate(3deg) translateY(4px); }
.pic { flex: 1; border-radius: 9px; position: relative; overflow: hidden;
  background: radial-gradient(circle at 72% 30%, #f6c860 0 13%, transparent 14%), linear-gradient(#86b6e8 0 58%, #6c9c62 58% 100%); }
.pic.noisy::after { content: ""; position: absolute; inset: 0; background-image: radial-gradient(rgba(0,0,0,.35) 1px, transparent 1.4px), radial-gradient(rgba(255,255,255,.5) 1px, transparent 1.4px); background-size: 5px 5px, 7px 7px; background-position: 0 0, 2px 3px; opacity: .55; }
.pic.blur { filter: blur(2.2px); }
.pic.gray { filter: grayscale(1); }
.pic.edge { background: #111; }
.pic.edge::before { content: ""; position: absolute; width: 22px; height: 22px; border: 2px solid #f2f2f2; border-radius: 50%; right: 18%; top: 14%; }
.pic.edge::after { content: ""; position: absolute; left: -10%; right: -10%; bottom: -35%; height: 72%; border-top: 2px solid #f2f2f2; border-radius: 50%; }
.pic.ph { background-size: cover; background-position: center; background-repeat: no-repeat; }
.lane.l .pc:nth-child(3n+2) .pic.ph { background-position: 18% 40%; }
.lane.l .pc:nth-child(3n) .pic.ph { background-position: 85% 60%; }
.pcl { font-size: 10.5px; line-height: 1; color: var(--text); font-weight: 600; display: flex; justify-content: space-between; opacity: .85; }
.pcl span:last-child { color: var(--muted); font-weight: 500; }

.chip { position: absolute; left: 50%; top: 50%; width: 156px; height: 156px; transform: translate(-50%, -50%); z-index: 3;
  background: var(--chip); border: 1px solid var(--border); border-radius: 22px; box-shadow: var(--shadow), 0 0 0 7px var(--glass);
  display: flex; flex-direction: column; align-items: center; justify-content: center; gap: .55rem; }
.chip .logo { width: 22px; height: 22px; gap: 3px; }
.chip b { font-size: 1.08rem; font-weight: 600; letter-spacing: -.01em; color: var(--text); }
.pipe { position: absolute; left: 50%; width: 1px; height: 26px; background: var(--line); z-index: 2; }
.pipe.t { top: -10px; } .pipe.b { bottom: -10px; }

/* compact header: shown once a photo is loaded or the camera is on */
.hero.compact { padding: .9rem 0 0; }
.hero.compact h1 { font-size: clamp(1.2rem, 2.3vw, 1.6rem); }
.hero.compact p { display: none; }
.hero.compact .belt { height: 108px; margin: 1.05rem 0 .8rem; }
.hero.compact .band { top: 9px; bottom: 9px; }
.hero.compact .bars { inset: 11px 0; }
.hero.compact .track { gap: 22px; padding-right: 22px; }
.hero.compact .pc { width: 104px; height: 78px; padding: 5px; gap: 4px; border-radius: 12px; }
.hero.compact .pic { border-radius: 7px; }
.hero.compact .pcl { font-size: 9px; }
.hero.compact .pcl span:last-child { display: none; }
.hero.compact .chip { width: 86px; height: 86px; border-radius: 16px; gap: .3rem; box-shadow: var(--shadow), 0 0 0 5px var(--glass); }
.hero.compact .chip .logo { width: 14px; height: 14px; gap: 2px; }
.hero.compact .chip b { font-size: .68rem; }
.hero.compact .pipe { height: 14px; } .hero.compact .pipe.t { top: -5px; } .hero.compact .pipe.b { bottom: -5px; }

@keyframes slide { from { transform: translateX(-50%); } to { transform: translateX(0); } }
@keyframes bars { from { background-position: 0 0; } to { background-position: 21px 0; } }

@media (max-width: 720px) {
  .belt { height: 150px; } .pc { width: 104px; height: 80px; padding: 6px; } .pcl { font-size: 9px; }
  .chip { width: 104px; height: 104px; border-radius: 16px; } .chip b { font-size: .85rem; } .chip .logo { width: 16px; height: 16px; }
  .track { gap: 22px; }
  .hero.compact .belt { height: 92px; } .hero.compact .pc { width: 84px; height: 64px; } .hero.compact .chip { width: 70px; height: 70px; } .hero.compact .chip b { display: none; }
}
@media (prefers-reduced-motion: reduce) { .track, .bars { animation: none; } }
"""


def _compact(html):
    """Remove indentation and blank lines so Markdown never turns the HTML into a code block."""
    return "".join(line.strip() for line in html.strip().splitlines())


def _card(kind, left, right):
    return (
        f'<div class="pc"><div class="pic {kind}"></div>'
        f'<div class="pcl"><span>{left}</span><span>{right}</span></div></div>'
    )


def _hero_html(compact, photo):
    """photo: None, or {"css", "label", "size", "filtered": [(css class, name, detail), ...]}."""
    if photo:  # the visitor's own picture rides the belt: raw on the left, filtered on the right
        raw = [("ph th0", html.escape(photo["label"]), photo["size"])] * 6
        done = [(f"ph {cls}", html.escape(name), html.escape(detail)) for cls, name, detail in photo["filtered"]]
        extra_css = f"<style>{photo['css']}</style>"
    else:
        raw = [("noisy", "photo.jpg", "800 px")] * 6
        done = [
            ("blur", "Gaussian", "σ 3"),
            ("edge", "Canny", "100/200"),
            ("gray", "Grayscale", "BGR2GRAY"),
            ("blur", "Median", "ksize 5"),
            ("edge", "Sobel", "ksize 3"),
            ("gray", "HSV", "Value"),
        ]
        extra_css = ""
    # each track holds its cards twice so the loop is seamless
    left = "".join(_card(*c) for c in raw * 2)
    right = "".join(_card(*c) for c in done * 2)
    size = " compact" if compact else ""
    return extra_css + _compact(
        f"""
        <div class="hero{size}">
          <h1><span class="hl">Image filters</span> in real time</h1>
          <p>Upload a photo or use your camera, then tune every OpenCV filter with a slider.</p>
          <div class="belt">
            <div class="band"><div class="bars"></div></div>
            <div class="lane l"><div class="track">{left}</div></div>
            <div class="lane r"><div class="track">{right}</div></div>
            <div class="pipe t"></div><div class="pipe b"></div>
            <div class="chip"><span class="logo"><i></i><i></i><i></i><i></i></span><b>Filter Studio</b></div>
          </div>
        </div>
        """
    )


def inject(dark, hide_uploader=False):
    """Write the theme CSS for the current mode."""
    palette = dict(DARK if dark else LIGHT)
    palette["scheme"] = "dark" if dark else "light"
    css = CSS
    for key, value in palette.items():
        css = css.replace(f"__{key}__", value)
    if hide_uploader:  # keep the widget alive (so the photo is remembered) but not visible
        css += ".st-key-uploader_box { display: none; }"
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def brand():
    st.markdown(
        _compact('<div class="brand"><span class="logo"><i></i><i></i><i></i><i></i></span>Filter Studio</div>'),
        unsafe_allow_html=True,
    )


def hero(compact=False, photo=None):
    st.markdown(_hero_html(compact, photo), unsafe_allow_html=True)


def fit_camera(ratio):
    """Size the camera panel to the video, so no empty band is left under it.

    ratio is width / height of the live picture. The picture is built with that exact ratio,
    so the height follows from the width: width / ratio, plus the START/STOP bar (about 58 px).
    """
    max_width = min(1180, round(540 * ratio))  # keep tall layouts from getting huge
    st.markdown(
        f"<style>.st-key-camera_box {{ max-width: {max_width}px; }}"
        f".st-key-camera_box iframe {{ height: calc(100cqw / {ratio:.4f} + 58px) !important; }}</style>",
        unsafe_allow_html=True,
    )
