#!/usr/bin/env python3
"""
Builds everything the detector needs from the datasets already in data/ (about 10 minutes on 2 CPU cores):
  det/coin, det/note, det/web   RGBA cut-outs (coins from the Roboflow polygons, notes and 0.1 / 2 DH coins with GrabCut)
  det/bg                        real backgrounds (coin photos with the coins inpainted + note-free strips of note photos)
  det/cutouts.csv, bg.csv, real.json
Folds come from data/merged/index.csv (the corrected one), so a fold-0 object never shows up in a training scene.
Usage:  python make_detector_data.py [project_root]      (default: the folder of this file)
"""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import cv2, numpy as np, pandas as pd

P = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
O = P / "det"
NAMES = ["five", "half", "one", "quarter", "ten"]; MAP = {"five": "5", "half": "0.5", "one": "1", "quarter": "0.2", "ten": "10"}


def save_rgba(img, mask, path, pad=4):
    ys, xs = np.where(mask > 0)
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0]), max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    cv2.imwrite(str(path), np.dstack([img[y0:y1, x0:x1], (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255]))     # PNG: the alpha channel must survive


def grabcut(img, rect_frac=0.04, it=4):
    h, w = img.shape[:2]; m = np.zeros((h, w), np.uint8)
    rect = (int(rect_frac * w), int(rect_frac * h), int((1 - 2 * rect_frac) * w), int((1 - 2 * rect_frac) * h))
    cv2.grabCut(img, m, rect, np.zeros((1, 65)), np.zeros((1, 65)), it, cv2.GC_INIT_WITH_RECT)
    mk = cv2.morphologyEx(np.where((m == 1) | (m == 3), 255, 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(mk)
    if n <= 1:
        return None
    mk = np.where(lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); out = np.zeros_like(mk); cv2.drawContours(out, cnts, -1, 255, -1)
    return out


def solidity(mk):
    cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); c = max(cnts, key=cv2.contourArea)
    return cv2.contourArea(c) / max(cv2.contourArea(cv2.convexHull(c)), 1)


def job_note(args):
    rel, label, fold, group = args
    img = cv2.imread(str(P / "data/merged" / rel)); mk = grabcut(img)
    if mk is None:
        return None
    a = float((mk > 0).mean())
    if not (0.12 < a < 0.80 and solidity(mk) > 0.90):
        return None
    x, y, w, h = cv2.boundingRect(mk); stem = Path(rel).stem
    save_rgba(img, mk, O / "note" / f"{stem}.png")
    strips = []                                                                       # note-free rectangles (outside the note's bounding box)
    H, W = img.shape[:2]
    for k, (a0, b0, c0, d0) in enumerate([(0, 0, x - 6, H), (x + w + 6, 0, W, H), (0, 0, W, y - 6), (0, y + h + 6, W, H)]):
        if c0 - a0 >= 110 and d0 - b0 >= 110 and (c0 - a0) * (d0 - b0) >= 20000:
            fp = O / "bg" / f"{stem}_s{k}.jpg"; cv2.imwrite(str(fp), img[b0:d0, a0:c0]); strips.append(dict(file=f"bg/{fp.name}", fold=fold))
    return dict(file=f"note/{stem}.png", kind="note", label=label, fold=fold, group=group), \
        dict(file=str(P / "data/merged" / rel), fold=fold, group=group, boxes=[[1, x, y, x + w, y + h, label]], kind="note_photo"), strips


def job_web(args):
    rel, label, fold, group = args
    img = cv2.imread(str(P / "data/merged" / rel)); h, w = img.shape[:2]
    if max(h, w) > 512:
        s = 512 / max(h, w); img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mk = grabcut(img, 0.03)
    if mk is None or not (0.15 < (mk > 0).mean() < 0.95 and solidity(mk) > 0.9):
        return None
    name = f"{label}_{Path(rel).stem}.png"                                            # the label is part of the name: file names repeat across class folders
    save_rgba(img, mk, O / "web" / name)
    return dict(file=f"web/{name}", kind="coin", label=label, fold=fold, group=group, web=True)


def main():
    for d in ["coin", "note", "web", "bg"]:
        (O / d).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(P / "data/merged/index.csv", dtype={"label": str}); assert df.path.is_unique, "index.csv has duplicate rows"
    fog = df.drop_duplicates("group").set_index("group").fold.to_dict(); meta, real, bgs = [], [], []
    seen = set()
    for split in ["train", "valid", "test"]:                                          # coins: first copy of each source photo, polygons -> masks
        for f in sorted((P / "data/raw/moroccan_coins_yolo" / split / "images").glob("*.jpg")):
            src = f.name.split("_jpg.rf.")[0]
            if src in seen or "coin:" + src not in fog:
                continue
            seen.add(src); fold = int(fog["coin:" + src]); img = cv2.imread(str(f)); H, W = img.shape[:2]; hole = np.zeros((H, W), np.uint8); objs = []
            for line in (f.parent.parent / "labels" / (f.stem + ".txt")).read_text().splitlines():
                p = line.split()
                if len(p) < 5 or len(p) <= 5:
                    continue
                pts = np.array(list(map(float, p[1:]))).reshape(-1, 2) * [W, H]; x0, y0 = pts.min(0); x1, y1 = pts.max(0)
                if min(x1 - x0, y1 - y0) < 14:
                    continue
                m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.int32(pts)], 255); hole |= m; objs.append((MAP[NAMES[int(p[0])]], m, (x0, y0, x1, y1)))
            for k, (lab, m, b) in enumerate(objs):
                save_rgba(img, m, O / "coin" / f"{src}__{k}.png")
                meta.append(dict(file=f"coin/{src}__{k}.png", kind="coin", label=lab, fold=fold, group="coin:" + src, diam=float(max(b[2] - b[0], b[3] - b[1]))))
            real.append(dict(file=str(f), fold=fold, group="coin:" + src, boxes=[[0, *b, lab] for lab, m, b in objs], kind="coin_photo"))
            cv2.imwrite(str(O / "bg" / f"{src}.jpg"), cv2.inpaint(img, cv2.dilate(hole, np.ones((15, 15), np.uint8)), 5, cv2.INPAINT_TELEA)); bgs.append(dict(file=f"bg/{src}.jpg", fold=fold))
    notes = [(r.path, r.label, int(r.fold), r.group) for r in df[df.source == "phone_note"].itertuples()]
    web = [(r.path, r.label, int(r.fold), r.group) for r in df[(df.source == "web") & df.label.isin(["0.1", "2"])].itertuples()]
    with Pool(2) as pool:
        for r in pool.map(job_note, notes, chunksize=8):
            if r:
                meta.append(r[0]); real.append(r[1]); bgs += r[2]
        meta += [r for r in pool.map(job_web, web) if r]
    pd.DataFrame(meta).to_csv(O / "cutouts.csv", index=False); pd.DataFrame(bgs).to_csv(O / "bg.csv", index=False); json.dump(real, open(O / "real.json", "w"))
    m = pd.DataFrame(meta); print(m.groupby(["kind", "label"]).size().unstack(0).fillna(0).T.to_string()); print("backgrounds:", len(bgs))


if __name__ == "__main__":
    main()
