"""Shared code for the Moroccan money notebooks: data index, augmentation, training and evaluation.

The notebooks import this file so that the training recipe lives in one place and the notebooks stay readable.
Checkpoints are written in the format that detect.py expects (state_dict, arch, img_size, classes, mean, std).
"""
import io
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision.transforms import v2
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
MERGED = ROOT / "data" / "merged"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"

SEED = 42
ARCH = "efficientnet_b0"
IMG_SIZE = 256
PRE = int(IMG_SIZE * 1.15)            # training images are stored a bit larger so random crops keep their detail
CLASSES = ["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100", "200"]
VALUES = np.array([float(c) for c in CLASSES])
IS_NOTE = VALUES >= 20                # 20, 50, 100, 200 DH are notes; everything below is a coin
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- data
def load_index():
    """data/merged/index.csv: path, label, source, group, fold (written by notebook 01)."""
    df = pd.read_csv(MERGED / "index.csv", dtype={"label": str})
    df["y"] = df.label.map({c: i for i, c in enumerate(CLASSES)})
    df["is_note"] = df.y.map(lambda i: bool(IS_NOTE[i]))
    return df


def load_tensor(path, size):
    """Same preprocessing as detect.py: RGB, squashed to a square, bicubic. Returns a uint8 CHW tensor."""
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).contiguous()


def load_images(df, size):
    return [load_tensor(MERGED / p, size) for p in df.path]


def norm(x):
    return (x.float() / 255 - MEAN) / STD


# --------------------------------------------------------------------------- augmentation
def camera_aug(x):
    """Cheap imitation of a different camera: JPEG compression and sensor noise (uint8 CHW in and out)."""
    if random.random() < 0.5:
        buf = io.BytesIO()
        Image.fromarray(x.permute(1, 2, 0).numpy()).save(buf, "JPEG", quality=random.randint(30, 85))
        buf.seek(0)
        x = torch.from_numpy(np.asarray(Image.open(buf).convert("RGB")).copy()).permute(2, 0, 1)
    if random.random() < 0.3:
        x = (x.float() + torch.randn(x.shape) * random.uniform(2, 8)).clamp(0, 255).to(torch.uint8)
    return x


def _blur():
    return v2.RandomApply([v2.GaussianBlur(5, (0.1, 1.5))], p=0.3)


# recipe "old": the original notebook (one policy for everything, with mirror flips and a 25 degree rotation)
AUG_OLD = v2.Compose([
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25), antialias=True),
    v2.RandomHorizontalFlip(), v2.RandomRotation(25), v2.ColorJitter(0.4, 0.4, 0.4, 0.05), _blur()])

# recipe "new": one policy per kind of picture, each tied to something observed in the data
AUG_COIN = v2.Compose([                       # coins lie at any angle and are never mirrored
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True),
    v2.RandomRotation(180), v2.ColorJitter(0.3, 0.3, 0.3, 0.03), _blur()])
AUG_NOTE = v2.Compose([                       # catalogue notes: landscape, upright, mirrored Arabic text never occurs
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.8, 1.25), antialias=True),
    v2.RandomRotation(5), v2.RandomPerspective(0.2, p=0.5), v2.ColorJitter(0.3, 0.3, 0.2, 0.0), _blur()])
AUG_FRAME = v2.Compose([                      # whole phone photos: the note is smaller and can be at any orientation
    v2.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0), ratio=(0.8, 1.25), antialias=True),
    v2.RandomRotation(15), v2.RandomPerspective(0.2, p=0.5), v2.ColorJitter(0.3, 0.3, 0.2, 0.0), _blur()])


def policy_of(row):
    if not row.is_note:
        return "coin"
    return "frame" if row.source == "phone_note" else "note"


def augment(x, recipe, policy):
    if recipe == "old":
        return AUG_OLD(x)
    if policy == "frame":
        x = torch.rot90(x, random.randint(0, 3), (1, 2)).contiguous()
    x = {"coin": AUG_COIN, "note": AUG_NOTE, "frame": AUG_FRAME}[policy](x)
    return camera_aug(x)


# --------------------------------------------------------------------------- model
def build_model(num_classes=None, pretrained=True):
    """EfficientNet-B0. ImageNet weights come from weights/ (no download needed), else from the timm hub."""
    m = timm.create_model(ARCH, pretrained=False)
    local = WEIGHTS / "efficientnet_b0_ra-3dd342df.pth"
    if pretrained:
        if local.exists():
            m.load_state_dict(torch.load(local, map_location="cpu"))
        else:
            m = timm.create_model(ARCH, pretrained=True)
    m.reset_classifier(num_classes if num_classes else 0)
    return m


def freeze_early_blocks(model, n_blocks):
    """Freeze the stem and the first n_blocks blocks: about 1.7x faster on CPU and a mild regulariser on 1,000 images."""
    frozen = [model.conv_stem, model.bn1] + list(model.blocks[:n_blocks])
    for mod in frozen:
        for p in mod.parameters():
            p.requires_grad = False
    return frozen


def train_model(X_raw, y, policies, recipe="new", epochs=10, lr=1e-3, batch=16, freeze_blocks=3, log=print, seed=SEED):
    """Fine-tune EfficientNet-B0. X_raw: list of uint8 CHW tensors (PRE x PRE). Returns (model, history)."""
    set_seed(seed)
    n, k = len(X_raw), len(CLASSES)
    model = build_model(k)
    model.drop_rate = 0.2
    frozen = freeze_early_blocks(model, freeze_blocks)
    model = model.to(memory_format=torch.channels_last)
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": head, "lr": lr * 5}], weight_decay=1e-2)
    steps = epochs * math.ceil(n / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 5], total_steps=steps, pct_start=0.15)
    cnt = np.bincount(y, minlength=k)
    sw = 1.0 / cnt[y]
    sw /= sw.sum()
    history = []
    for ep in range(epochs):
        model.train()
        for mod in frozen:                                   # frozen blocks keep their ImageNet batch-norm statistics
            mod.eval()
        t0, run, seen = time.time(), 0.0, 0
        ids = np.random.choice(n, n, p=sw)
        for b in range(0, n, batch):
            bi = ids[b:b + batch]
            if len(bi) < 2:
                continue
            xb = torch.stack([augment(X_raw[i], recipe, policies[i]) for i in bi])
            xb = norm(xb).contiguous(memory_format=torch.channels_last)
            yb = torch.tensor(y[bi])
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item() * len(bi)
            seen += len(bi)
        history.append({"epoch": ep + 1, "train_loss": run / max(seen, 1), "seconds": time.time() - t0})
        log(f"epoch {ep + 1:2d}/{epochs} | loss {run / max(seen, 1):.3f} | {time.time() - t0:.0f}s")
    return model.eval(), history


@torch.no_grad()
def predict_proba(model, X, batch=32):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        x = norm(torch.stack(X[i:i + batch])).contiguous(memory_format=torch.channels_last)
        out.append(model(x).softmax(1))
    return torch.cat(out).numpy()


@torch.no_grad()
def embed(X, batch=32):
    """Frozen ImageNet EfficientNet-B0 features, L2-normalised (for the baseline)."""
    m = build_model(None).eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(m(norm(torch.stack(X[i:i + batch]))))
    f = torch.cat(out).numpy()
    return f / np.linalg.norm(f, axis=1, keepdims=True)


def save_checkpoint(model, path, extra=None):
    """Format read by detect.py (Classifier.__init__)."""
    ck = {"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "arch": ARCH, "img_size": IMG_SIZE,
          "classes": CLASSES, "mean": MEAN.flatten().tolist(), "std": STD.flatten().tolist()}
    if extra:
        ck.update(extra)
    torch.save(ck, path)


def load_model(path):
    """Rebuild a model from a checkpoint written by save_checkpoint."""
    ck = torch.load(path, map_location="cpu")
    m = timm.create_model(ck["arch"], pretrained=False)
    m.reset_classifier(len(ck["classes"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval().to(memory_format=torch.channels_last)


# --------------------------------------------------------------------------- metrics
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def evaluate(P, df, name=""):
    """P: probabilities (n, 11) aligned with df. Accuracy, macro-F1 and the cost of the mistakes in DH."""
    y, pred = df.y.values, P.argmax(1)
    ok = pred == y
    lo, hi = wilson(int(ok.sum()), len(y))
    rec = {"name": name, "n": len(y), "acc": ok.mean(), "acc_ci": f"{lo:.3f}-{hi:.3f}",
           "macro_f1": f1_score(y, pred, average="macro"), "mae_dh": np.abs(VALUES[pred] - VALUES[y]).mean(),
           "coin_acc": ok[~IS_NOTE[y]].mean(), "note_acc": ok[IS_NOTE[y]].mean(),
           "coin_note_errors": int((IS_NOTE[pred] != IS_NOTE[y]).sum())}
    for s in sorted(df.source.unique()):
        m = (df.source == s).values
        rec[f"acc_{s}"] = ok[m].mean() if m.any() else float("nan")
    return rec


def paired_bootstrap(Pa, Pb, df, n_boot=1000, seed=0):
    """B minus A on the same resamples of GROUPS (twins and bursts stay together). Returns mean and 95% interval."""
    rng = np.random.RandomState(seed)
    groups = df.group.to_numpy(dtype=str)
    gl = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in gl}
    y, pa, pb = df.y.values, Pa.argmax(1), Pb.argmax(1)
    d_f1, d_mae = [], []
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.choice(gl, len(gl))])
        d_f1.append(f1_score(y[idx], pb[idx], average="macro") - f1_score(y[idx], pa[idx], average="macro"))
        d_mae.append(np.abs(VALUES[pb[idx]] - VALUES[y[idx]]).mean() - np.abs(VALUES[pa[idx]] - VALUES[y[idx]]).mean())
    q = lambda v: tuple(float(x) for x in np.round(np.percentile(v, [2.5, 97.5]), 3))
    return {"delta_macro_f1": round(float(np.mean(d_f1)), 3), "f1_95ci": q(d_f1),
            "delta_mae_dh": round(float(np.mean(d_mae)), 3), "mae_95ci": q(d_mae)}


def threshold_table(P, df, thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9)):
    conf, pred, y = P.max(1), P.argmax(1), df.y.values
    rows = []
    for t in thresholds:
        keep = conf >= t
        wrong = keep & (pred != y)
        rows.append({"threshold": t, "coverage": keep.mean(),
                     "acc_accepted": (pred == y)[keep].mean() if keep.any() else float("nan"),
                     "wrong_accepted": int(wrong.sum()),
                     "mean_err_dh": np.abs(VALUES[pred[keep]] - VALUES[y[keep]]).mean() if keep.any() else float("nan")})
    return pd.DataFrame(rows)
