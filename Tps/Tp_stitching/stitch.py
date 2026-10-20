"""Image stitching with OpenCV.

Methods (--method):
  manual    SIFT + ratio test + RANSAC, greedy incremental assembly against the growing
            mosaic (order of files does not matter), distance-transform feather blending.
  scans     OpenCV Stitcher, SCANS mode (flat surface, camera translation).
  panorama  OpenCV Stitcher, PANORAMA mode (camera rotating in place).
  stitcher  OpenCV Stitcher only, mode chosen automatically: tries scans then panorama.
  auto      tries manual, scans, panorama in this order.
            For stitcher and auto, each result is scored with a fidelity measure; the
            search stops at the first result above --good, otherwise the best one is kept.
            Built-in Stitcher runs are killed after --timeout seconds.

Fidelity: each input image is searched in the final panorama with normalized
cross-correlation and the worst piece gives the score. A faithful stitch contains every
piece almost unchanged (score close to 1); ghosting, distortion or a missing piece lower it.

The result is cropped to the largest rectangle free of black pixels.

Input: one folder containing the image files to stitch together.
Output: a single panorama image.

Usage:
    python stitch.py parts/ -o pano.jpg --scale 2
    python stitch.py parts/ --method manual --model affine
    python stitch.py parts/ --method stitcher --timeout 60
    python stitch.py parts/ --method scans --timeout 120
"""
import argparse
import glob
import multiprocessing
import os
import sys
from queue import Empty

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)


def load_images(folder, scale):
    paths = []
    for ext in ("jpg", "jpeg", "png", "bmp", "tif"):
        paths += glob.glob(os.path.join(folder, f"*.{ext}"))
        paths += glob.glob(os.path.join(folder, f"*.{ext.upper()}"))
    paths = sorted(set(paths))
    imgs = [cv2.imread(p) for p in paths]
    if len(imgs) < 2 or any(i is None for i in imgs):
        sys.exit(f"Need at least 2 valid images in '{folder}'")
    if scale != 1:
        imgs = [cv2.resize(i, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                for i in imgs]
    print(f"Loaded {len(imgs)} images, first size: {imgs[0].shape[1]}x{imgs[0].shape[0]}")
    return imgs


def builtin_worker(imgs, mode, conf, queue):
    stitcher = cv2.Stitcher_create(mode)
    stitcher.setPanoConfidenceThresh(conf)
    if mode == cv2.Stitcher_SCANS:
        stitcher.setWaveCorrection(False)
    try:
        status, pano = stitcher.stitch(imgs)
    except cv2.error as e:
        queue.put((-1, str(e).splitlines()[0]))
        return
    queue.put((status, pano))


def stitch_builtin(imgs, name, timeout, conf):
    """Run cv2.Stitcher in a separate process so a hang or crash cannot freeze the script."""
    mode = cv2.Stitcher_SCANS if name == "scans" else cv2.Stitcher_PANORAMA
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=builtin_worker, args=(imgs, mode, conf, queue))
    proc.start()
    try:
        status, pano = queue.get(timeout=timeout)
    except Empty:
        status, pano = None, f"timeout after {timeout}s"
    proc.terminate()
    proc.join()
    if status == cv2.Stitcher_OK:
        return pano
    print(f"Built-in Stitcher {name} failed ({pano if isinstance(pano, str) else 'status ' + str(status)})")
    return None


def fidelity(pano, imgs, max_side=800):
    s = min(1.0, max_side / max(pano.shape[:2]))
    g = cv2.cvtColor(cv2.resize(pano, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                     cv2.COLOR_BGR2GRAY)
    if min(g.shape) < 8:
        return 0.0
    scores = []
    for img in imgs:
        t = cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        f = min(1.0, (g.shape[0] - 1) / t.shape[0], (g.shape[1] - 1) / t.shape[1])
        if f < 1.0:
            t = cv2.resize(t, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        if min(t.shape) < 4:
            scores.append(0.0)
            continue
        scores.append(float(cv2.matchTemplate(g, t, cv2.TM_CCOEFF_NORMED).max()))
    return float(min(scores))


def prep_gray(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def estimate_transform(mosaic, mask, img, sift, model, ratio=0.75, min_inliers=12):
    """Return (3x3 matrix img->mosaic, inliers) or (None, 0)."""
    k1, d1 = sift.detectAndCompute(prep_gray(mosaic), mask)
    k2, d2 = sift.detectAndCompute(prep_gray(img), None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None, 0
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    good = [m for m, n in matcher.knnMatch(d2, d1, k=2) if m.distance < ratio * n.distance]
    if len(good) < min_inliers:
        return None, 0
    src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    if model == "affine":
        A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0)
        if A is None:
            return None, 0
        H = np.vstack([A, [0, 0, 1]])
    else:
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            return None, 0
    n = int(inl.sum())
    return (H, n) if n >= min_inliers else (None, n)


def blend_into(mosaic, mask, img, H):
    """Warp img with H into an enlarged canvas and feather-blend with the mosaic."""
    h, w = img.shape[:2]
    mh, mw = mosaic.shape[:2]
    corners_img = cv2.perspectiveTransform(
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2), H)
    corners_mos = np.float32([[0, 0], [mw, 0], [mw, mh], [0, mh]]).reshape(-1, 1, 2)
    allc = np.vstack([corners_img, corners_mos])
    x0, y0 = np.floor(allc.min(axis=0).ravel()).astype(int)
    x1, y1 = np.ceil(allc.max(axis=0).ravel()).astype(int)
    T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    size = (int(x1 - x0), int(y1 - y0))
    if size[0] * size[1] > 150_000_000:
        return None, None

    a = cv2.warpPerspective(mosaic, T, size).astype(np.float32)
    ma = cv2.warpPerspective(mask, T, size, flags=cv2.INTER_NEAREST)
    b = cv2.warpPerspective(img, T @ H, size, flags=cv2.INTER_CUBIC).astype(np.float32)
    mb = cv2.warpPerspective(np.full((h, w), 255, np.uint8), T @ H, size,
                             flags=cv2.INTER_NEAREST)
    ma = (ma > 0).astype(np.uint8)
    mb = cv2.erode((mb > 0).astype(np.uint8), np.ones((3, 3), np.uint8))

    wa = cv2.distanceTransform(ma, cv2.DIST_L2, 3)
    wb = cv2.distanceTransform(mb, cv2.DIST_L2, 3)
    total = wa + wb
    total[total == 0] = 1
    alpha = (wb / total)[..., None]
    out = a * (1 - alpha) + b * alpha
    out[(ma == 1) & (mb == 0)] = a[(ma == 1) & (mb == 0)]
    out[(ma == 0) & (mb == 1)] = b[(ma == 0) & (mb == 1)]
    return out.astype(np.uint8), ((ma | mb) * 255).astype(np.uint8)


def stitch_manual(imgs, model="homography"):
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02)
    start = max(range(len(imgs)), key=lambda i: len(sift.detect(prep_gray(imgs[i]), None)))
    mosaic = imgs[start].copy()
    mask = np.full(mosaic.shape[:2], 255, np.uint8)
    remaining = [i for i in range(len(imgs)) if i != start]
    print(f"Manual pipeline, starting from image #{start}")

    while remaining:
        best = None
        for i in remaining:
            H, n = estimate_transform(mosaic, mask, imgs[i], sift, model)
            if H is not None and (best is None or n > best[2]):
                best = (i, H, n)
        if best is None:
            print(f"Could not place {len(remaining)} image(s): {remaining}")
            break
        i, H, n = best
        res, new_mask = blend_into(mosaic, mask, imgs[i], H)
        if res is None:
            remaining.remove(i)
            print(f"Image #{i}: degenerate transform, skipped")
            continue
        mosaic, mask = res, new_mask
        remaining.remove(i)
        print(f"Added image #{i} ({n} inliers) -> mosaic {mosaic.shape[1]}x{mosaic.shape[0]}")
    return mosaic if len(remaining) < len(imgs) - 1 else None


def largest_inner_rect(valid):
    """Largest axis-aligned rectangle with no invalid pixel. Returns (x0, y0, x1, y1)."""
    h, w = valid.shape
    heights = np.zeros(w, dtype=int)
    best_area, best = 0, (0, 0, w, h)
    for y in range(h):
        heights = np.where(valid[y], heights + 1, 0)
        stack = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] >= cur:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best_area:
                    best_area, best = area, (sx, y - sh + 1, x, y + 1)
                start = sx
            stack.append((start, cur))
    return best


def crop_black(img, max_side=600):
    h, w = img.shape[:2]
    void = (~np.any(img > 0, axis=2)).astype(np.uint8)
    _, labels = cv2.connectedComponents(void, connectivity=4)
    edge = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    valid = (~np.isin(labels, edge[edge > 0])).astype(np.uint8)
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8))
    if valid.sum() == 0:
        return img
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(valid, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
    x0, y0, x1, y1 = largest_inner_rect(small > 0)
    m = int(np.ceil(1 / s)) + 1
    x0, y0 = int(x0 / s) + m, int(y0 / s) + m
    x1, y1 = min(w, int(x1 / s) - m), min(h, int(y1 / s) - m)
    return img[y0:y1, x0:x1] if x1 > x0 and y1 > y0 else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder containing the image files to stitch")
    ap.add_argument("-o", "--output", default="pano.jpg", help="output image file")
    ap.add_argument("--scale", type=float, default=3.0, help="upscale factor for tiny pieces")
    ap.add_argument("--method", choices=["auto", "manual", "stitcher", "scans", "panorama"],
                    default="auto")
    ap.add_argument("--model", choices=["homography", "affine"], default="homography",
                    help="transformation model of the manual method")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="seconds before a built-in Stitcher run is abandoned")
    ap.add_argument("--conf", type=float, default=0.6,
                    help="confidence threshold of the built-in Stitcher (lower accepts weaker pairs)")
    ap.add_argument("--good", type=float, default=0.9,
                    help="auto: stop at the first result with at least this fidelity")
    args = ap.parse_args()

    imgs = load_images(args.folder, args.scale)
    groups = {"auto": ["manual", "scans", "panorama"], "stitcher": ["scans", "panorama"]}
    names = groups.get(args.method, [args.method])

    results = []
    for name in names:
        print(f"\n--- {name} ---")
        if name == "manual":
            pano = stitch_manual(imgs, args.model)
        else:
            pano = stitch_builtin(imgs, name, args.timeout, args.conf)
        if pano is None:
            continue
        pano = crop_black(pano)
        score = fidelity(pano, imgs)
        print(f"{name}: {pano.shape[1]}x{pano.shape[0]}, fidelity {score:.3f}")
        results.append((score, name, pano))
        if args.method in groups and score >= args.good:
            break

    if not results:
        sys.exit("Stitching failed: not enough overlap or texture between pieces.")

    score, name, pano = max(results, key=lambda r: r[0])
    cv2.imwrite(args.output, pano, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\nSelected: {name} (fidelity {score:.3f})")
    print(f"Saved {args.output} ({pano.shape[1]}x{pano.shape[0]})")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()