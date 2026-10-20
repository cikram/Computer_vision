#!/usr/bin/env python3
"""
Moroccan money counter, v2: a trained YOLO detector finds every coin and note, the EfficientNet-B0 classifier
reads the value of each one, and the total is printed in DH.

    python detect_yolo.py --source photo.jpg
    python detect_yolo.py --source clip.mp4
    python detect_yolo.py --source webcam           (q = quit)

Thresholds are per kind: a wrong note costs 20 to 200 DH, a wrong coin 0.1 to 10 DH, so notes need more confidence.
An object below its threshold is drawn in orange and NOT added to the total.
"""
import argparse, sys, time
from pathlib import Path
import cv2, numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from detect import Classifier                       # classifier loader of the original project (kept unchanged)
GREEN, ORANGE, WHITE, DARK = (90, 200, 80), (0, 150, 255), (255, 255, 255), (28, 28, 28)
FONT = cv2.FONT_HERSHEY_SIMPLEX
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def contain_suppress(res, thr=0.7):
    """Drop a box lying mostly inside a more confident box of the same kind (a fold or a corner of a note found twice)."""
    keep = [True] * len(res)
    order = sorted(range(len(res)), key=lambda i: -res[i]["det_conf"])
    for a, i in enumerate(order):
        for j in order[a + 1:]:
            if not (keep[i] and keep[j]) or res[i]["kind"] != res[j]["kind"]:
                continue
            bi, bj = res[i]["box"], res[j]["box"]
            iw, ih = min(bi[2], bj[2]) - max(bi[0], bj[0]), min(bi[3], bj[3]) - max(bi[1], bj[1])
            if iw <= 0 or ih <= 0:
                continue
            small = min((bi[2] - bi[0]) * (bi[3] - bi[1]), (bj[2] - bj[0]) * (bj[3] - bj[1]))
            if iw * ih / small > thr:
                keep[j] = False
    return [r for r, k in zip(res, keep) if k]


def crop(img, box, kind, pad_coin=1.15, pad_note=1.05):
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; w, h = x1 - x0, y1 - y0
    if kind == "coin":
        w = h = max(w, h) * pad_coin
    else:
        w, h = w * pad_note, h * pad_note
    H, W = img.shape[:2]; a, b, c, d = int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)
    p = max(0, -a, -b, c - W, d - H)
    if p > 0:
        img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REPLICATE); a, b, c, d = a + p, b + p, c + p, d + p
    return img[b:d, a:c]


class Counter:
    def __init__(self, det_path, models_dir, clf_name, det_conf=0.25, imgsz=416, thr_coin=0.5, thr_note=0.8, coin_det_min=0.5, contain=0.7):
        from ultralytics import YOLO
        self.det = YOLO(str(det_path)); self.clf = Classifier(models_dir, clf_name)
        self.det_conf, self.imgsz, self.thr = det_conf, imgsz, {"coin": thr_coin, "note": thr_note}
        self.coin_det_min, self.contain = coin_det_min, contain

    def __call__(self, img):
        r = self.det.predict(img, imgsz=self.imgsz, conf=self.det_conf, iou=0.5, verbose=False, device="cpu" if not torch.cuda.is_available() else 0)[0]
        res = []
        for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            kind = "coin" if int(k) == 0 else "note"
            if kind == "coin" and c < self.coin_det_min:       # weak 'coin' boxes are mostly parts of a note (emblem, hologram)
                continue
            res.append(dict(box=[int(v) for v in b], kind=kind, det_conf=float(c)))
        res = contain_suppress(res, self.contain)
        crops = [crop(img, d["box"], d["kind"]) for d in res]; kinds = [d["kind"] for d in res]
        if crops:
            P, _ = self.clf.predict(crops, kinds)
            for d, p in zip(res, P):
                i = int(p.argmax()); d.update(value=float(self.clf.values[i]), conf=float(p[i]))
                d["accepted"] = d["conf"] >= self.thr[d["kind"]]
        return res


def drop_ignored(res, zones):
    """Remove detections whose centre lies inside an ignore rectangle (x0,y0,x1,y1), e.g. a fixed logo or watermark."""
    return [d for d in res if not any(z[0] <= (d["box"][0] + d["box"][2]) / 2 <= z[2] and z[1] <= (d["box"][1] + d["box"][3]) / 2 <= z[3] for z in zones)]


def total(res):
    return round(sum(d["value"] for d in res if d["accepted"]), 2)


def total_all(res):
    return round(sum(d["value"] for d in res), 2)


def draw(img, res):
    out = img.copy(); H, W = out.shape[:2]; fs = max(0.5, W / 1100)
    for d in res:
        col = GREEN if d["accepted"] else ORANGE; x0, y0, x1, y1 = d["box"]
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)
        t = f"{d['value']:g} DH {d['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(t, FONT, fs, 2); cv2.rectangle(out, (x0, max(y0 - th - 8, 0)), (x0 + tw + 6, max(y0, th + 8)), col, -1)
        cv2.putText(out, t, (x0 + 3, max(y0 - 4, th + 4)), FONT, fs, DARK, 2, cv2.LINE_AA)
    lines = [(f"Trusted total {total(res):g} DH", WHITE), (f"All detected {total_all(res):g} DH", ORANGE)]
    y = 8
    for t, col in lines:
        (tw, th), _ = cv2.getTextSize(t, FONT, fs * 1.2, 3)
        cv2.rectangle(out, (8, y), (tw + 24, y + th + 18), DARK, -1); cv2.putText(out, t, (16, y + th + 8), FONT, fs * 1.2, col, 3, cv2.LINE_AA); y += th + 24
    return out


def summary(res):
    from collections import Counter as C
    c = C(d["value"] for d in res if d["accepted"]); unsure = sum(1 for d in res if not d["accepted"])
    txt = " + ".join(f"{n} x {v:g}" for v, n in sorted(c.items(), reverse=True)) or "nothing counted"
    return f"trusted: {txt} = {total(res):g} DH | all detected: {total_all(res):g} DH" + (f"  ({unsure} uncertain)" if unsure else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcam"); ap.add_argument("--det", default=str(HERE / "models" / "detector.pt"))
    ap.add_argument("--models", default=str(HERE / "models")); ap.add_argument("--model", default="best_model_v2.pt")
    ap.add_argument("--conf-coin", type=float, default=0.5); ap.add_argument("--conf-note", type=float, default=0.8)
    ap.add_argument("--det-conf", type=float, default=0.25); ap.add_argument("--coin-det-min", type=float, default=0.5); ap.add_argument("--contain", type=float, default=0.7); ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--ignore", action="append", default=[], help="x0,y0,x1,y1 pixel rectangle to ignore (repeatable)")
    ap.add_argument("--out", default=str(HERE / "results")); ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    ctr = Counter(a.det, a.models, a.model, a.det_conf, a.imgsz, a.conf_coin, a.conf_note, a.coin_det_min, a.contain); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    zones = [tuple(int(v) for v in z.split(",")) for z in a.ignore]
    src = a.source
    if src != "webcam" and Path(src).suffix.lower() not in VIDEO_EXT:
        img = cv2.imread(src)
        if img is None: sys.exit(f"cannot read {src}")
        res = drop_ignored(ctr(img), zones); print(summary(res)); p = out / (Path(src).stem + "_detected.jpg"); cv2.imwrite(str(p), draw(img, res)); print("saved", p)
        if not a.no_show: cv2.imshow("money", draw(img, res)); cv2.waitKey(0)
        return
    cap = cv2.VideoCapture(0 if src == "webcam" else src); writer = None; last = ""
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        res = drop_ignored(ctr(frame), zones); vis = draw(frame, res); s = summary(res)
        if s != last: print(s); last = s
        if src != "webcam" and writer is None:
            writer = cv2.VideoWriter(str(out / (Path(src).stem + "_detected.mp4")), cv2.VideoWriter_fourcc(*"mp4v"), cap.get(cv2.CAP_PROP_FPS) or 25, (vis.shape[1], vis.shape[0]))
        if writer is not None: writer.write(vis)
        if not a.no_show:
            cv2.imshow("money", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release()
    if writer is not None: writer.release()


if __name__ == "__main__":
    main()
