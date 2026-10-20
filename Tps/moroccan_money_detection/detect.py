#!/usr/bin/env python3
"""
Moroccan money detector: finds coins and banknotes in a photo or a live camera,
classifies each one with the fine-tuned EfficientNet-B0 and adds up the total in DH.

Usage
    python detect.py --source photo.jpg      # one image  -> results/photo_detected.jpg
    python detect.py --source webcam         # live camera (q = quit, s = save snapshot)
    python detect.py --source clip.mp4       # a video file works too

Needs models/best_model.pt (written by notebooks/02_train_and_evaluate.ipynb).
v2: flip TTA is off by default (use --tta to enable it) and the default --conf is 0.6.
v3: two-tone coins (5 and 10 DH) get a soft prior from a centre-vs-ring colour gap, and objects below --show-min
    confidence are not drawn at all (hides logos and other round objects that look like nothing in particular).
v4: the classifier is trained on three datasets (catalogue photos, phone coins, phone notes); FastSAM weights are looked
    up in weights/ first; --model chooses another checkpoint (for example models/holdout_model.pt).
v5: optional coin size prior (--size-prior): the apparent diameters of the coins in one picture are compared with the real
    ones. Off by default because it did not help measurably (notebook 04).
"""
import argparse
import os
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
GREEN, ORANGE, WHITE, DARK = (90, 200, 80), (0, 150, 255), (255, 255, 255), (28, 28, 28)  # BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX
torch.set_num_threads(os.cpu_count() or 1)


def fmt(v):
    return f"{v:g}"


# --------------------------------------------------------------------------- classifier
class Classifier:
    """Loads best_model.pt and predicts denominations for BGR crops."""

    def __init__(self, models_dir, name="best_model.pt"):
        ck = torch.load(Path(models_dir) / name, map_location="cpu")
        self.classes = ck["classes"]
        self.size = ck["img_size"]
        self.values = np.array([float(c) for c in self.classes])
        self.note_mask = torch.tensor(self.values >= 20)          # 20, 50, 100, 200 DH are notes
        self.mean = torch.tensor(ck["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(ck["std"]).view(1, 3, 1, 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = timm.create_model(ck["arch"], pretrained=False)
        model.reset_classifier(len(self.classes))
        model.load_state_dict(ck["state_dict"])
        self.model = model.eval().to(self.device).to(memory_format=torch.channels_last)

    def _tensor(self, bgr):
        # identical to training: RGB, square resize with bicubic interpolation
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb).resize((self.size, self.size), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)

    @torch.inference_mode()
    def predict(self, crops, kinds, tta=False):
        """crops[i] is one BGR crop or a list of crops of the same object (averaged). Returns (probs, mass).
        kinds[i] is 'coin', 'note' or None: a round object can only be a coin and a rectangle only a note, so the
        impossible classes are removed. mass = how much the model itself believed in that coin / note family."""
        flat, owner = [], []
        for i, c in enumerate(crops):
            for cc in (c if isinstance(c, list) else [c]):
                flat.append(cc)
                owner.append(i)
        if not flat:
            return np.zeros((0, len(self.classes))), np.zeros(0)
        out = []
        for s in range(0, len(flat), 16):
            x = torch.stack([self._tensor(c) for c in flat[s:s + 16]]).float() / 255
            x = ((x - self.mean) / self.std).to(self.device).contiguous(memory_format=torch.channels_last)
            p = self.model(x).softmax(1)
            if tta:
                p = (p + self.model(torch.flip(x, [3])).softmax(1)) / 2
            out.append(p.cpu())
        p = torch.cat(out)
        owner = torch.tensor(owner)
        P = torch.zeros(len(crops), p.shape[1]).index_add_(0, owner, p)
        P /= torch.bincount(owner, minlength=len(crops)).unsqueeze(1).float()
        mass = torch.ones(len(P))
        for i, k in enumerate(kinds):
            if k == "coin":
                P[i, self.note_mask] = 0
            elif k == "note":
                P[i, ~self.note_mask] = 0
            mass[i] = P[i].sum()
            P[i] /= P[i].sum()
        return P.numpy(), mass.numpy()


# --------------------------------------------------------------------------- localisation
_LW = np.array([0.7, 1.0, 1.0], np.float32)                     # lightness counts less than colour


def edge_region(small):
    """Where the picture has dense, sharp edges (coin rims, engraving, printing). Soft shadows and smooth
    lighting gradients have no such edges, so they are NOT part of this region."""
    h, w = small.shape[:2]
    g = cv2.GaussianBlur(small, (5, 5), 1.2).astype(np.float32)
    mag = np.zeros((h, w), np.float32)
    for c in range(3):
        gx = cv2.Sobel(g[..., c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g[..., c], cv2.CV_32F, 0, 1, ksize=3)
        mag = np.maximum(mag, np.hypot(gx, gy))
    t = max(40.0, 3.0 * float(np.median(mag)) + 25)
    k = max(5, int(0.035 * max(h, w))) | 1
    dens = cv2.blur((mag > t).astype(np.float32), (k, k))
    m = (dens > 0.10).astype(np.uint8) * 255
    kk = max(3, int(0.02 * max(h, w))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker), cv2.MORPH_OPEN, ker)
    return cv2.dilate(m, ker)


def background_estimate(lab, certain):
    """Background colour at every pixel, interpolated from the pixels that are certainly background (outside the
    edge region). Follows lighting gradients and shadows, because it is built from the nearby background."""
    h, w = lab.shape[:2]
    s = 160.0 / max(h, w)
    size = (max(8, int(w * s)), max(8, int(h * s)))
    tl = cv2.resize(lab, size, interpolation=cv2.INTER_AREA)
    tc = cv2.resize(certain.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    if tc.mean() < 0.04:                                          # almost no clean background: use the dominant colour
        return np.broadcast_to(np.median(lab.reshape(-1, 3), axis=0), lab.shape).copy()
    est, filled = None, None
    for sig in (3, 8, 24):
        n = cv2.GaussianBlur(tl * tc[..., None], (0, 0), sig)
        dn = cv2.GaussianBlur(tc, (0, 0), sig)
        e = n / np.maximum(dn[..., None], 1e-4)
        ok = dn > 0.06
        if est is None:
            est, filled = e, ok
        else:
            est = np.where((~filled)[..., None] & ok[..., None], e, est)
            filled = filled | ok
    est = np.where(filled[..., None], est, np.median(lab.reshape(-1, 3), axis=0))
    return cv2.resize(est.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def foreground_mask(small):
    """Objects vs background: distance to the interpolated background colour, then clean-up."""
    h, w = small.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    certain = edge_region(small) == 0
    bg = background_estimate(lab, certain)
    d = np.linalg.norm((lab - bg) * _LW, axis=2)
    sigma = max(1.4826 * float(np.median(d[certain])) if certain.any() else 4.0, 1.5)
    mask = (d > max(12.0, 6.0 * sigma)).astype(np.uint8) * 255
    k = max(3, int(0.006 * max(h, w))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, cnts, -1, 255, -1)               # fill holes (engraving, pale areas of notes)
    return filled


def describe(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    circ = 4 * np.pi * area / (peri * peri + 1e-9)
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    return dict(area=area, circ=circ, aspect=max(rw, rh) / (min(rw, rh) + 1e-9),
                extent=area / (rw * rh + 1e-9), circle=(cx, cy, r), rect=rect)


def kind_of(d, img_area):
    if d["circ"] >= 0.72 and d["aspect"] <= 1.35 and d["extent"] >= 0.65:
        return "coin"
    if (d["aspect"] >= 1.4 and d["extent"] >= 0.80 and d["area"] >= 0.01 * img_area
            and min(d["rect"][1]) >= 0.10 * np.sqrt(img_area)):
        return "note"
    return None


def split_blob(cnt):
    """Splits touching coins (distance transform + watershed). Returns a list of contours or None."""
    x, y, w, h = cv2.boundingRect(cnt)
    pad = 4
    blob = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv2.drawContours(blob, [cnt - [x - pad, y - pad]], -1, 255, -1)
    dt = cv2.distanceTransform(blob, cv2.DIST_L2, 5)
    _, peaks = cv2.threshold(dt, 0.55 * dt.max(), 255, cv2.THRESH_BINARY)
    peaks = peaks.astype(np.uint8)
    n, markers = cv2.connectedComponents(peaks)
    if n - 1 < 2:
        return None
    markers = markers + 1
    markers[(blob > 0) & (peaks == 0)] = 0                      # unknown zone between the peaks
    markers[blob == 0] = 1                                      # background
    cv2.watershed(cv2.cvtColor(blob, cv2.COLOR_GRAY2BGR), markers)
    out = []
    for lab in range(2, n + 1):
        seg = ((markers == lab) * 255).astype(np.uint8)
        cs, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            out.append(max(cs, key=cv2.contourArea) + np.array([x - pad, y - pad]))
    return out if len(out) >= 2 else None


def split_by_colour(small, cnt, min_part=0.22, min_dist=28):
    """Splits a big blob into two objects when it is clearly two differently coloured, compact regions
    (e.g. two notes lying side by side). Returns two contours or None. Checked on 31 single-note photos: 0 false splits."""
    h, w = small.shape[:2]
    blob = np.zeros((h, w), np.uint8)
    cv2.drawContours(blob, [cnt], -1, 255, -1)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (9, 9), 0), cv2.COLOR_BGR2LAB)
    ys, xs = np.nonzero(blob)
    if len(ys) < 800:
        return None
    sel = np.random.RandomState(0).choice(len(ys), min(len(ys), 12000), replace=False)
    ab = lab[ys[sel], xs[sel], 1:3].astype(np.float32)
    _, _, cen = cv2.kmeans(ab, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    if np.linalg.norm(cen[0] - cen[1]) < min_dist:
        return None
    ab_all = lab[..., 1:3].astype(np.float32)
    part1 = np.linalg.norm(ab_all - cen[1], axis=2) < np.linalg.norm(ab_all - cen[0], axis=2)
    k = max(3, int(0.012 * max(h, w))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    total, out = float((blob > 0).sum()), []
    for m in (part1 & (blob > 0), ~part1 & (blob > 0)):
        m8 = cv2.morphologyEx(m.astype(np.uint8) * 255, cv2.MORPH_OPEN, ker)
        cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cs:
            return None
        c = max(cs, key=cv2.contourArea)
        a = cv2.contourArea(c)
        if a < min_part * total or a / (cv2.contourArea(cv2.convexHull(c)) + 1e-9) < 0.80:
            return None
        out.append(c)
    return out


def order_points(pts):
    pts = np.asarray(pts, np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def crop_circle(img, cx, cy, r, pad=1.15):
    half = int(r * pad) + 1
    p = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REPLICATE)
    cx, cy = int(round(cx)) + half, int(round(cy)) + half
    return p[cy - half:cy + half, cx - half:cx + half]


def crop_box(img, pts):
    tl, tr, br, bl = order_points(pts)
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    out = cv2.warpPerspective(img, cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32), dst), (w, h))
    return cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE) if h > w else out   # notes are landscape in the training data


def crop_rect(img, x, y, w, h, pad=0.05):
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x - pad * w)), max(0, int(y - pad * h))
    x1, y1 = min(W, int(x + w + pad * w)), min(H, int(y + h + pad * h))
    return img[y0:y1, x0:x1]


def _coin_ellipse(c, circle, inv):
    """Ellipse that hugs the coin (better than a circle for tilted coins). Falls back to the circle."""
    cx, cy, r = [v * inv for v in circle]
    fallback = ((cx, cy), (2 * r, 2 * r), 0.0)
    if len(c) < 5:
        return fallback
    (ex, ey), (ew, eh), ang = cv2.fitEllipse(c)
    ew, eh = ew * inv, eh * inv
    if not (0.6 * 2 * r <= min(ew, eh) and max(ew, eh) <= 1.15 * 2 * r):
        return fallback
    return ((ex * inv, ey * inv), (ew, eh), ang)


BIM_HI = 12.0                          # centre-vs-ring colour gap (Lab a/b) above which a coin is clearly two-tone
BIM_LO = 2.5                           # centre-vs-ring gap below which a coin is clearly one-tone (silver or bronze all over)
BIM_FACTOR = 0.1                       # soft prior: the impossible group keeps 10% of its weight on a clearly two-tone coin
BIM_FACTOR_LO = 0.25                   # gentler on the one-tone side: a dark or glary 5/10 DH can read as one-tone
BIMETAL_VALUES = (5.0, 10.0)           # only the 5 and 10 DH coins have a different metal in the centre and in the ring
USE_BIMETAL = True                     # --no-bimetal turns the two-tone prior off
SHOW_MIN = 0.5                         # below this confidence an object is not drawn (set by --show-min)


def bimetal_score(bgr):
    """Colour gap between the centre of a coin crop and its outer ring (Lab a/b medians). Both are in the same image,
    so a global colour cast or the light level hardly matters. Expects the crop with the coin filling the circle."""
    h, w = bgr.shape[:2]
    R = min(h, w) / 2
    yy, xx = np.mgrid[:h, :w]
    rr = np.hypot(xx - w / 2, yy - h / 2) / R
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    centre, ring = lab[rr < 0.35][:, 1:], lab[(rr > 0.55) & (rr < 0.85)][:, 1:]
    if len(centre) < 20 or len(ring) < 20:
        return 0.0
    return float(np.linalg.norm(np.median(centre, 0) - np.median(ring, 0)))


def apply_bimetal_prior(clf, it, probs):
    b = it.get("bim")
    if not USE_BIMETAL or it.get("kind") != "coin" or b is None:
        return probs
    two_tone = np.isin(clf.values, BIMETAL_VALUES)
    p = probs.copy()
    if b >= BIM_HI:
        p[~two_tone] *= BIM_FACTOR
    elif b <= BIM_LO:
        p[two_tone] *= BIM_FACTOR_LO
    else:
        return probs
    return p / p.sum()


# Size prior. Coins in one picture share a scale: a coin read with high confidence (say a 10 DH) tells how many pixels one
# millimetre is, and an uncertain coin can then be compared with the real diameters. It mainly separates 0.5 / 1 DH (21 / 24 mm)
# and 0.1 / 0.2 DH (20 / 23 mm); 0.2, 1, 5 and 2 DH are within 3 mm of each other, so the prior is soft and never decides alone.
# Diameters come from numismatic catalogues (Numista, uCoin, exchange sites), not from Bank Al-Maghrib; 5 and 10 DH differ by about
# 1 mm between series, so the middle value is used.
COIN_MM = {0.1: 20.0, 0.2: 23.0, 0.5: 21.0, 1.0: 24.0, 2.0: 26.0, 5.0: 25.5, 10.0: 27.5}
USE_SIZE = False                       # off by default, --size-prior turns it on: on the test photos it gained at most 1-2 points (notebook 04)
SIZE_REF_CONF = 0.9                    # a coin read at least this confidently can serve as the scale reference
SIZE_SIGMA = 0.11                      # measured spread of log(measured / real diameter) on the 76 coin photos (notebook 04)
SIZE_FLOOR = 0.3                       # a wrong-sized class keeps at least this share of its weight: the prior is an indication, not a rule
SIZE_SPREAD = 1.25                     # references that disagree by more than this (tilted camera, coins at different heights): no prior


def coin_diameter_px(it):
    """Apparent diameter in pixels: the long axis of the fitted ellipse (a tilted coin is an ellipse), else the circle."""
    g = it.get("geom")
    if g is not None and g[0] == "ellipse":
        return float(max(g[1][1]))
    return 2.0 * float(it["size"])


def set_size_scale(clf, items, probs_list):
    """Sets it['mm_scale'] (pixels per millimetre) on every coin that is not itself a reference. items: dicts (items or tracks)."""
    for it in items:
        it["mm_scale"] = None
    if not USE_SIZE:
        return
    refs = []
    for it, p in zip(items, probs_list):
        if it.get("kind") != "coin" or it.get("weak") or it.get("whole"):
            continue
        p = apply_bimetal_prior(clf, it, p)
        i = int(np.argmax(p))
        if p[i] >= SIZE_REF_CONF and float(clf.values[i]) in COIN_MM:
            refs.append((id(it), coin_diameter_px(it) / COIN_MM[float(clf.values[i])]))
    for it in items:
        if it.get("kind") != "coin":
            continue
        scales = [s for ident, s in refs if ident != id(it)]
        if not scales or (len(scales) > 1 and max(scales) / min(scales) > SIZE_SPREAD):
            continue
        if any(ident == id(it) for ident, _ in refs):                # a reference keeps its own reading
            continue
        it["mm_scale"] = float(np.median(scales))


def apply_size_prior(clf, it, probs):
    s = it.get("mm_scale")
    if not USE_SIZE or not s or it.get("kind") != "coin":
        return probs
    d = coin_diameter_px(it) / s
    w = np.ones(len(probs))
    for k, v in enumerate(clf.values):
        D = COIN_MM.get(float(v))
        if D:
            w[k] = SIZE_FLOOR + (1 - SIZE_FLOOR) * np.exp(-0.5 * (np.log(d / D) / SIZE_SIGMA) ** 2)
    p = probs * w
    return p / p.sum() if p.sum() > 0 else probs


COIN_PADS = (0.85, 1.0, 1.15)          # training photos are tight close-ups: classify a few tight crops and average


def coin_item(img, cx, cy, r, geom, weak=False):
    crops = [crop_circle(img, cx, cy, r, p) for p in COIN_PADS]
    return dict(kind="coin", geom=geom, center=(cx, cy), size=r, crop=crops[1], crops=crops, weak=weak,
                bim=bimetal_score(crops[1]))


def find_items(img, min_frac=0.002, max_side=800):
    """Returns (items, mask, status). status: 'ok', 'none' (no objects) or 'fills' (one object fills the image).
    item = dict(kind, geom, center, size, crop) with geom in ORIGINAL image coordinates:
    ('ellipse', ((cx,cy),(w,h),angle)) | ('box', 4x2 points) | ('poly', Nx2 points)"""
    H, W = img.shape[:2]
    scale = min(1.0, max_side / max(H, W))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    mask = foreground_mask(small)
    area_img = small.shape[0] * small.shape[1]
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_frac * area_img]
    if not cnts:
        return [], mask, "none"
    if max(cv2.contourArea(c) for c in cnts) > 0.7 * area_img:
        return [], mask, "fills"

    infos = [(c, describe(c)) for c in cnts]
    infos = [(c, d, kind_of(d, area_img)) for c, d in infos]
    parts = [(c, d, k, False) for c, d, k in infos if k == "coin"]
    # pieces of one big object (a folded note, a note with pale areas) lie close together: merge them into one hull
    loose = [c for c, _, k in infos if k != "coin"]
    if loose:
        canvas = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(canvas, loose, -1, 255, -1)
        g = max(3, int(0.03 * max(small.shape[:2]))) | 1
        n, lab = cv2.connectedComponents(cv2.dilate(canvas, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g))))
        for l in range(1, n):
            cs, _ = cv2.findContours((((lab == l) & (canvas > 0)) * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            merged = len(cs) > 1
            c = cv2.convexHull(np.vstack(cs)) if merged else cs[0]
            d = describe(c)
            if d["area"] >= min_frac * area_img:
                parts.append((c, d, kind_of(d, area_img), merged))

    final = []
    for c, d, k, merged in parts:
        if k is None and not merged:                            # ragged mask: judge the shape by its convex hull
            dh = describe(cv2.convexHull(c))
            kh = kind_of(dh, area_img)
            if kh == "note" or (kh == "coin" and d["area"] / dh["area"] > 0.85):
                c, d, k = cv2.convexHull(c), dh, kh
        if k is None and not merged:                            # irregular blob: maybe touching coins
            segs = split_blob(c)
            if segs:
                seg_parts = []
                for sg in segs:
                    if cv2.contourArea(sg) < min_frac * area_img:
                        continue
                    ds = describe(sg)
                    seg_parts.append((sg, ds, "coin" if ds["circ"] >= 0.7 and ds["aspect"] <= 1.4 else None))
                if seg_parts:
                    final += seg_parts
                    continue
        final.append((c, d, k))
    parts = final

    expanded = []
    for c, d, k in parts:
        if k == "coin" and d["area"] > 0.12 * area_img:
            k = None
        if d["area"] >= 0.06 * area_img and k != "coin":
            segs = split_by_colour(small, c)                    # two notes next to / on each other
            if segs:
                for sg in segs:
                    ds = describe(sg)
                    ks = kind_of(ds, area_img)
                    expanded.append((sg, ds, ks if ks == "note" else ("noteblob" if ds["area"] >= 0.03 * area_img else None)))
                continue
            if k is None:
                k = "noteblob"                                  # big, irregular, cut by the frame: a note that is partly visible
        expanded.append((c, d, k))
    parts = expanded

    inv = 1.0 / scale
    items = build_items(img, parts, inv)
    items += hough_coins(img, small, inv, items)
    return items, mask, "ok"


def build_items(img, parts, inv):
    """parts = [(contour in working scale, descriptor, kind)] -> items in original image coordinates."""
    items = []
    for c, d, k in parts[:40]:
        if k == "coin":
            cx, cy, r = [v * inv for v in d["circle"]]
            items.append(coin_item(img, cx, cy, r, ("ellipse", _coin_ellipse(c, d["circle"], inv))))
        elif k == "note":
            pts = cv2.boxPoints(d["rect"]) * inv
            items.append(dict(kind="note", geom=("box", pts), center=tuple(pts.mean(0)),
                              size=max(d["rect"][1]) * inv / 2, crop=crop_box(img, pts)))
        else:
            x, y, w, h = [v * inv for v in cv2.boundingRect(c)]
            hull = cv2.convexHull(c).reshape(-1, 2).astype(np.float32) * inv
            items.append(dict(kind="note" if k == "noteblob" else None, geom=("poly", hull), center=(x + w / 2, y + h / 2),
                              size=max(w, h) / 2, crop=crop_rect(img, x, y, w, h)))
    return [it for it in items if it["crop"].size > 0 and min(it["crop"].shape[:2]) >= 8]


class Segmenter:
    """FastSAM: a pretrained 'segment everything' model. It knows nothing about money, it only finds objects
    (also in shadows, because it works from edges and shapes). pip install ultralytics; the weights (23 MB) are
    downloaded automatically the first time."""

    def __init__(self, weights=None):
        from ultralytics import FastSAM
        local = HERE / "weights" / "FastSAM-s.pt"
        self.model = FastSAM(str(weights or (local if local.exists() else "FastSAM-s.pt")))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.polygons(np.full((160, 160, 3), 128, np.uint8))          # warm-up (the first call is slow)

    def polygons(self, bgr):
        r = self.model(bgr, device=self.device, retina_masks=True, imgsz=640, conf=0.4, iou=0.9, verbose=False)[0]
        return [] if r.masks is None else [np.asarray(p, np.float32) for p in r.masks.xy if len(p) >= 5]


def find_items_sam(img, seg, min_frac=0.002, max_side=800):
    """Candidates from FastSAM instead of colour thresholds. Same output as find_items: (items, mask, status)."""
    H, W = img.shape[:2]
    scale = min(1.0, max_side / max(H, W))
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    area_img = small.shape[0] * small.shape[1]
    cands = []
    for p in seg.polygons(small):
        c = np.round(p).astype(np.int32).reshape(-1, 1, 2)
        a = cv2.contourArea(c)
        if a < min_frac * area_img or a > 0.40 * area_img:      # specks, and masks of the whole table / sheet
            continue
        d = describe(c)
        k = kind_of(d, area_img)
        if k is None:                                           # ragged mask: judge the shape by its convex hull
            dh = describe(cv2.convexHull(c))
            kh = kind_of(dh, area_img)
            if kh == "note" or (kh == "coin" and d["area"] / dh["area"] > 0.85):
                c, d, k = cv2.convexHull(c), dh, kh
        if k == "coin" and d["area"] > 0.12 * area_img:
            k = None
        if k is None and d["area"] < 0.06 * area_img:           # small, a bit ragged (shadow tail): round enough to be a coin?
            dh = describe(cv2.convexHull(c))
            fill = dh["area"] / (np.pi * dh["circle"][2] ** 2 + 1e-9)
            if fill >= 0.62 and dh["aspect"] <= 1.5:
                mom = cv2.moments(c)
                cx, cy = mom["m10"] / (mom["m00"] + 1e-9), mom["m01"] / (mom["m00"] + 1e-9)
                d = dict(d, circle=(cx, cy, float(np.sqrt(d["area"] / np.pi) * 1.05)))   # circle of equal area: ignores thin tails
                k = "coin"
        if k is None and d["area"] >= 0.06 * area_img:
            k = "noteblob"                                      # big irregular object: maybe a note that is cut / folded
        cands.append([c, d, k])
    cands.sort(key=lambda x: -x[1]["area"])
    drawn, kept = [], []
    for c, d, k in cands:                                       # parts of an object (print on a note, a portrait on a coin)
        m = np.zeros(small.shape[:2], np.uint8)
        cv2.drawContours(m, [c], -1, 1, -1)
        # drop parts of an object: anything inside a coin, and non-round pieces inside a note. A round object lying on
        # a note (or inside a shadow region that looks like a note) is kept: the classifier decides later.
        inside = any((kk == "coin" or (kk == "note" and k != "coin")) and (m & bm).sum() > 0.8 * m.sum() for bm, kk in drawn)
        if not inside:
            kept.append((c, d, k))
            drawn.append((m, k))
    mask = np.zeros(small.shape[:2], np.uint8)
    for c, _, _ in kept:
        cv2.drawContours(mask, [c], -1, 255, -1)
    items = build_items(img, kept, 1.0 / scale)
    return items, mask, ("ok" if items else "none")


def hough_coins(img, small, inv, items, max_new=10):
    """Round rims found from edges, independent of colour and of shadows. Skips circles that are already covered
    by a detected coin or that lie on a note. These coins are 'weak': the classifier must clearly agree."""
    h, w = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (9, 9), 2)
    m = max(h, w)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=0.05 * m, param1=80, param2=30,
                               minRadius=int(0.028 * m), maxRadius=int(0.1 * m))
    out = []
    if circles is None:
        return out
    for cx, cy, r in circles[0][:max_new]:
        cx, cy, r = cx * inv, cy * inv, r * inv
        if any(it["kind"] == "coin" and np.hypot(cx - it["center"][0], cy - it["center"][1]) < 0.9 * max(r, it["size"])
               for it in items):
            continue
        if any(it["kind"] == "note" and cv2.pointPolygonTest(np.float32(it["geom"][1]).reshape(-1, 1, 2), (float(cx), float(cy)), False) >= 0
               for it in items):
            continue
        it = coin_item(img, cx, cy, r, ("ellipse", ((cx, cy), (2 * r, 2 * r), 0.0)), weak=True)
        if it["crop"].size and min(it["crop"].shape[:2]) >= 8:
            out.append(it)
    return out


def whole_image_item(img):
    h, w = img.shape[:2]
    m = 0.03
    pts = np.array([[m * w, m * h], [(1 - m) * w, m * h], [(1 - m) * w, (1 - m) * h], [m * w, (1 - m) * h]], np.float32)
    return dict(kind=None, geom=("box", pts), center=(w / 2, h / 2), size=max(w, h) / 2, crop=img, whole=True)


# --------------------------------------------------------------------------- analysis
def make_result(clf, it, probs, conf_thr, mass=1.0):
    probs = apply_size_prior(clf, it, apply_bimetal_prior(clf, it, probs))
    i = int(np.argmax(probs))
    conf, whole = float(probs[i]), it.get("whole", False)
    need = conf_thr if (it["kind"] is not None or whole) else max(conf_thr, 0.75)   # unknown shape: stronger evidence
    need_mass = 0.6 if it.get("weak") else 0.35
    accepted = conf >= need and mass >= need_mass               # also: the model must believe it is a coin / note at all
    show = accepted or whole or (it["kind"] is not None and conf >= SHOW_MIN and mass >= need_mass - 0.1)
    return dict(kind=it["kind"], geom=it["geom"], center=it["center"], size=it["size"], idx=i, conf=conf,
                value=float(clf.values[i]), accepted=accepted, show=show, probs=probs, whole=whole, mass=float(mass),
                crop=it.get("crop"), weak=bool(it.get("weak")))


class Tracker:
    """Keeps objects between frames so a coin that has not moved is NOT classified again (the main speed-up for
    live use), averages its probabilities over time (stabler answer) and hides one-frame noise."""

    def __init__(self, recheck=2.0, fast=0.35, max_miss=2, min_hits=2):
        self.tracks, self.recheck, self.fast, self.max_miss, self.min_hits = [], recheck, fast, max_miss, min_hits

    def update(self, clf, items, conf_thr, tta):
        now = time.time()
        free = list(range(len(self.tracks)))
        pairs = []
        for j, it in enumerate(items):
            for i in free:
                t = self.tracks[i]
                dist = np.hypot(t["center"][0] - it["center"][0], t["center"][1] - it["center"][1])
                if t["kind"] == it["kind"] and dist < 0.6 * max(t["size"], it["size"]):
                    pairs.append((dist, i, j))
        used_t, used_d, todo = set(), set(), []
        for dist, i, j in sorted(pairs):
            if i in used_t or j in used_d:
                continue
            used_t.add(i); used_d.add(j)
            t, it = self.tracks[i], items[j]
            t.update(geom=it["geom"], center=it["center"], size=it["size"], miss=0, hits=t["hits"] + 1,
                     whole=it.get("whole", False), crop=it["crop"])
            if it.get("bim") is not None:                       # smooth the two-tone score over frames
                t["bim"] = it["bim"] if t.get("bim") is None else 0.7 * t["bim"] + 0.3 * it["bim"]
            if now - t["t_class"] > (self.fast if t["n"] < 4 else self.recheck):   # first checks are quick, later ones rare
                todo.append((t, it, False))
        for j, it in enumerate(items):
            if j not in used_d:
                t = dict(kind=it["kind"], geom=it["geom"], center=it["center"], size=it["size"], probs=None, t_class=0,
                         hits=1, miss=0, n=0, whole=it.get("whole", False), crop=it["crop"], weak=it.get("weak", False), bim=it.get("bim"))
                self.tracks.append(t)
                todo.append((t, it, True))
        for i, t in enumerate(list(self.tracks)):
            if i < len(free) and i not in used_t:
                t["miss"] += 1
        self.tracks = [t for t in self.tracks if t["miss"] <= self.max_miss]
        for is_new in (True, False):                            # new objects and re-checks are classified the same way (flip TTA only with --tta)
            batch = [(t, it) for t, it, new in todo if new == is_new]
            if not batch:
                continue
            crops = [it.get("crops") or it["crop"] if is_new else (it["crops"][t["n"] % len(it["crops"])] if it.get("crops") else it["crop"])
                     for t, it in batch]
            probs, mass = clf.predict(crops, [t["kind"] for t, _ in batch], tta)
            for (t, _), p, m in zip(batch, probs, mass):
                n = t["n"]
                if t["probs"] is None:
                    t["probs"], t["mass"] = p, float(m)
                else:                                           # running mean over all checks of this object
                    t["probs"] = (t["probs"] * n + p) / (n + 1)
                    t["mass"] = (t["mass"] * n + float(m)) / (n + 1)
                t["n"] = n + 1
                t["t_class"] = now
        live = [t for t in self.tracks if t["miss"] == 0 and t["hits"] >= self.min_hits and t["probs"] is not None]
        set_size_scale(clf, live, [t["probs"] for t in live])
        return [make_result(clf, t, t["probs"], conf_thr, t.get("mass", 1.0)) for t in live]


def _poly(g):
    kind, data = g
    if kind == "ellipse":
        (cx, cy), (w, h), ang = data
        return cv2.ellipse2Poly((int(cx), int(cy)), (max(1, int(w / 2)), max(1, int(h / 2))), int(ang), 0, 360, 15).astype(np.float32)
    return np.asarray(data, np.float32)


def suppress_overlaps(results, shape, thr=0.4):
    """If two detections overlap mostly, keep one. The bigger wins (a note beats its own fragments), except that
    an unknown-shape blob never beats a confident coin or note sitting inside it."""
    if len(results) < 2:
        return results
    H, W = shape[:2]
    s = 200.0 / max(H, W)
    masks = []
    for r in results:
        m = np.zeros((int(H * s) + 2, int(W * s) + 2), np.uint8)
        cv2.fillPoly(m, [np.int32(_poly(r["geom"]) * s)], 1)
        masks.append(m.astype(bool))
    areas = [m.sum() for m in masks]
    keep = [True] * len(results)
    order = sorted(range(len(results)), key=lambda i: -areas[i])
    for a, i in enumerate(order):
        for j in order[a + 1:]:
            if not (keep[i] and keep[j]) or areas[j] == 0:
                continue
            if (masks[i] & masks[j]).sum() / areas[j] > thr:
                big, small = results[i], results[j]
                if big["accepted"] and not small["accepted"]:
                    keep[j] = False
                elif small["accepted"] and not big["accepted"]:
                    keep[i] = False
                elif big.get("weak") and not small.get("weak"):
                    keep[i] = False
                elif big["kind"] is None and small["kind"] is not None and small["accepted"]:
                    keep[i] = False
                elif not big["accepted"] and not small["accepted"] and big["conf"] * big["mass"] < small["conf"] * small["mass"]:
                    keep[i] = False                             # neither is sure: keep the one the model believes in more
                else:
                    keep[j] = False
    return [r for r, k in zip(results, keep) if k]


SEGMENTER = None                                                # set in main() when --detector is sam / auto


def analyse(clf, img, conf_thr=0.6, tracker=None, tta=False, min_frac=0.002, max_side=800):
    """Returns (results, mask). One result per object: kind, geom, value, conf, accepted, probs."""
    items, mask, status = (find_items_sam(img, SEGMENTER, min_frac, max_side) if SEGMENTER is not None
                           else find_items(img, min_frac, max_side))
    if status == "fills":                                       # close-up: the object fills the picture
        items = [whole_image_item(img)]
    if tracker is not None:
        return suppress_overlaps(tracker.update(clf, items, conf_thr, tta), img.shape), mask
    probs, mass = clf.predict([it.get("crops") or it["crop"] for it in items], [it["kind"] for it in items], tta)
    set_size_scale(clf, items, probs)
    return suppress_overlaps([make_result(clf, it, p, conf_thr, m) for it, p, m in zip(items, probs, mass)], img.shape), mask


def total_of(results):
    return round(sum(r["value"] for r in results if r["accepted"]), 2)


def breakdown(results):
    ok = Counter(r["value"] for r in results if r["accepted"])
    return " + ".join(f"{n} x {fmt(v)} DH" for v, n in sorted(ok.items(), reverse=True))


def summary(results):
    unsure = sum(1 for r in results if not r["accepted"] and r["show"])
    if not results:
        return "no coins or notes found"
    text = breakdown(results) or "nothing recognised with enough confidence"
    return f"{text} = {fmt(total_of(results))} DH" + (f"  ({unsure} uncertain, not counted)" if unsure else "")


# --------------------------------------------------------------------------- drawing
def _round_rect(img, p1, p2, col, r):
    (x1, y1), (x2, y2) = p1, p2
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), col, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), col, -1)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(img, (cx, cy), r, col, -1, cv2.LINE_AA)


def _label(img, text, cx, top, bottom, col, fs):
    H, W = img.shape[:2]
    th = max(1, int(round(fs * 2)))
    (tw, tht), base = cv2.getTextSize(text, FONT, fs, th)
    pad = max(4, int(fs * 7))
    w, h = tw + 2 * pad, tht + base + 2 * pad
    x = int(min(max(cx - w / 2, 2), W - w - 2))
    y = int(top - h - 4)
    if y < 2:                                                    # no room above: put the label below
        y = int(min(bottom + 4, H - h - 2))
    _round_rect(img, (x, y), (x + w, y + h), col, pad)
    cv2.putText(img, text, (x + pad, y + pad + tht), FONT, fs, DARK, th, cv2.LINE_AA)


def draw(img, results, total=None, fps=None):
    out = img.copy()
    H, W = out.shape[:2]
    fs = max(0.55, min(H, W) / 1000)
    lw = max(2, int(min(H, W) / 330))
    for r in results:
        if not r["show"]:
            continue                                            # too unsure to show (probably background noise)
        col = GREEN if r["accepted"] else ORANGE
        label = f"{fmt(r['value'])} DH  {r['conf'] * 100:.0f}%" if r["accepted"] else f"? {fmt(r['value'])} DH  {r['conf'] * 100:.0f}%"
        kind, g = r["geom"]
        if kind == "ellipse":
            box = ((float(g[0][0]), float(g[0][1])), (float(g[1][0]), float(g[1][1])), float(g[2]))
            cv2.ellipse(out, box, DARK, lw + 3, cv2.LINE_AA)
            cv2.ellipse(out, box, col, lw, cv2.LINE_AA)
            top, bottom, cx = g[0][1] - max(g[1]) / 2, g[0][1] + max(g[1]) / 2, g[0][0]
        else:
            pts = np.int32(g)
            cv2.polylines(out, [pts], True, DARK, lw + 3, cv2.LINE_AA)
            cv2.polylines(out, [pts], True, col, lw, cv2.LINE_AA)
            top, bottom, cx = float(g[:, 1].min()), float(g[:, 1].max()), float(g[:, 0].mean())
        _label(out, label, cx, top, bottom, col, fs)
    tot = total_of(results) if total is None else total
    line1, line2 = f"Total  {fmt(tot)} DH", breakdown(results)
    s1, s2 = fs * 1.5, fs * 0.8
    (w1, h1), _ = cv2.getTextSize(line1, FONT, s1, 3)
    (w2, h2), _ = cv2.getTextSize(line2, FONT, s2, 2) if line2 else ((0, 0), 0)
    bw, bh = max(w1, w2) + 28, h1 + (h2 + 14 if line2 else 0) + 26
    overlay = out.copy()
    _round_rect(overlay, (8, 8), (8 + bw, 8 + bh), DARK, 12)
    out = cv2.addWeighted(overlay, 0.78, out, 0.22, 0)
    cv2.putText(out, line1, (22, 8 + 13 + h1), FONT, s1, WHITE, 3, cv2.LINE_AA)
    if line2:
        cv2.putText(out, line2, (22, 8 + 13 + h1 + 14 + h2), FONT, s2, (200, 200, 200), 2, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(out, f"{fps:.0f} fps", (10, H - 12), FONT, 0.6, WHITE, 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- run modes
def show(window, img):
    try:
        cv2.imshow(window, img)
        return True
    except cv2.error:
        return False


def run_image(clf, args):
    path = Path(args.source)
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"Could not read image: {path}")
    t0 = time.time()
    results, mask = analyse(clf, img, args.conf, None, args.tta, args.min_area)
    print(f"{path.name}: {summary(results)}   [{time.time() - t0:.2f}s]")
    for r in results:
        top = np.argsort(-r["probs"])[:3]
        print("   ", r["kind"] or "object", "->", ", ".join(f"{clf.classes[j]} DH {r['probs'][j]:.2f}" for j in top))
    out = draw(img, results)
    res_dir = Path(args.out) if args.out else HERE / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    out_path = res_dir / f"{path.stem}_detected.jpg"
    cv2.imwrite(str(out_path), out)
    if args.debug:
        cv2.imwrite(str(res_dir / f"{path.stem}_mask.jpg"), mask)
    print("saved:", out_path)
    if not args.no_show:
        if show("Moroccan money detector (press any key to close)", out):
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("(no display available, open the saved image instead)")


class Worker(threading.Thread):
    """Runs detection + classification in the background so the video never freezes."""

    def __init__(self, clf, args):
        super().__init__(daemon=True)
        self.clf, self.args, self.tracker = clf, args, Tracker()
        self.frame, self.results, self.totals = None, [], deque(maxlen=5)
        self.alive, self.runs, self.busy_time = True, 0, 0.0
        self.lock = threading.Lock()

    def submit(self, frame):
        with self.lock:
            if self.frame is None and not self.busy:
                self.frame = frame.copy()

    busy = False

    def run(self):
        while self.alive:
            with self.lock:
                f, self.frame = self.frame, None
                self.busy = f is not None
            if f is None:
                time.sleep(0.003)
                continue
            t0 = time.time()
            res, _ = analyse(self.clf, f, self.args.conf, self.tracker, self.args.tta, self.args.min_area, max_side=640)
            with self.lock:
                self.results = res
                self.totals.append(total_of(res))
                self.busy = False
            self.runs += 1
            self.busy_time += time.time() - t0

    def latest(self):
        with self.lock:
            return list(self.results), (float(np.median(self.totals)) if self.totals else 0.0)


def run_stream(clf, args, src, live):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"Could not open {'camera ' + str(src) if live else src}." + (" Try --camera 1." if live else ""))
    if live:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame_time = 0.0 if live else 1.0 / (cap.get(cv2.CAP_PROP_FPS) or 25)   # video files play at their own speed
    worker = Worker(clf, args)
    worker.start()
    res_dir = Path(args.out) if args.out else HERE / "results"
    print("Running: q or Esc = quit | s = save snapshot | c = save the crops (for retraining)" if not args.no_show else "Running (no window)...")
    fps, frames, t_start, last = 0.0, 0, time.time(), time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        worker.submit(frame)
        results, total = worker.latest()
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-3) if frames else 0.0
        last, frames = now, frames + 1
        view = draw(frame, results, total, fps)
        if frame_time:
            time.sleep(max(0.0, frame_time - (time.time() - now)))
        if args.no_show:
            continue
        if not show("Moroccan money detector", view):
            sys.exit("No display available for the live window (use --no-show or an image).")
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            res_dir.mkdir(parents=True, exist_ok=True)
            p = res_dir / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(p), view)
            print("saved:", p, "|", summary(results))
        if key == ord("c"):
            n_saved = 0
            for r in results:
                if r["show"] and r.get("crop") is not None:
                    d = res_dir / "collected" / fmt(r["value"])
                    d.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(d / f"{int(time.time() * 1000)}_{n_saved}.jpg"), r["crop"])
                    n_saved += 1
            print(f"saved {n_saved} crops in {res_dir / 'collected'} (sort them into the right folder, then retrain)")
    if not live:                                                # video file: let the detector finish the last frame
        for _ in range(400):
            if worker.frame is None and not worker.busy:
                break
            time.sleep(0.01)
    worker.alive = False
    cap.release()
    cv2.destroyAllWindows()
    el = time.time() - t_start
    if worker.runs:
        print(f"{frames} frames in {el:.1f}s ({frames / el:.0f} fps shown) | detector ran {worker.runs}x, "
              f"{1000 * worker.busy_time / worker.runs:.0f} ms each | last result: {summary(worker.latest()[0])}")


def main():
    ap = argparse.ArgumentParser(description="Moroccan coin and banknote detector")
    ap.add_argument("--source", default="webcam", help="image path, video path or 'webcam' (default)")
    ap.add_argument("--models", default=str(HERE / "models"), help="folder containing the checkpoint")
    ap.add_argument("--model", default="best_model.pt", help="checkpoint file name inside --models (default best_model.pt)")
    ap.add_argument("--detector", choices=["auto", "sam", "rules"], default="auto",
                    help="sam = FastSAM finds the objects (best, needs ultralytics); rules = colour/shape rules; auto = sam if available")
    ap.add_argument("--conf", type=float, default=0.6, help="minimum confidence to count an item (default 0.6)")
    ap.add_argument("--show-min", type=float, default=0.5, help="do not draw objects below this confidence (default 0.5; 0.4 shows more uncertain ones)")
    ap.add_argument("--no-bimetal", action="store_true", help="turn off the two-tone (5/10 DH) colour prior")
    ap.add_argument("--size-prior", action="store_true", help="turn on the coin size prior (needs a confidently read coin in the picture; small gain, see notebook 04)")
    ap.add_argument("--tta", action="store_true", help="average each prediction with its mirrored image (2x slower, no measured gain)")
    ap.add_argument("--min-area", type=float, default=0.002, help="smallest object as a fraction of the image (default 0.002)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    ap.add_argument("--out", default=None, help="folder for results (default: ./results next to detect.py)")
    ap.add_argument("--no-show", action="store_true", help="do not open a window (just save / print)")
    ap.add_argument("--debug", action="store_true", help="image mode: also save the foreground mask")
    args = ap.parse_args()

    if not (Path(args.models) / args.model).exists():
        sys.exit(f"{args.model} not found in {args.models}. Run notebooks/02_train_and_evaluate.ipynb first.")
    clf = Classifier(args.models, args.model)
    global SEGMENTER, SHOW_MIN, USE_BIMETAL, USE_SIZE
    SHOW_MIN, USE_BIMETAL, USE_SIZE = args.show_min, not args.no_bimetal, args.size_prior
    if args.detector in ("auto", "sam"):
        try:
            SEGMENTER = Segmenter()
            print("detector: FastSAM")
        except Exception as e:
            if args.detector == "sam":
                sys.exit(f"FastSAM unavailable ({e}). Run: pip install ultralytics")
            print(f"FastSAM unavailable ({type(e).__name__}), using the colour/shape rules instead (pip install ultralytics for better results)")
    src = args.source
    if src.lower() == "webcam":
        run_stream(clf, args, args.camera, live=True)
    elif Path(src).suffix.lower() in VIDEO_EXT:
        run_stream(clf, args, src, live=False)
    else:
        run_image(clf, args)


if __name__ == "__main__":
    main()