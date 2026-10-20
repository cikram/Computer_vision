"""Detector: YOLO11n, 2 classes (coin, note), 416 px. Usage: python scripts/train_detector.py NAME EPOCHS [START_WEIGHTS]
The delivered detector.pt was trained in three runs (30 epochs on the first scene set, 28 more, then 14 on the corrected scene set
from compose.py), then a fourth run of 7 epochs on scenes with synthetic false objects (compose.py ... syn_d, lr0 0.003, start = previous best).
Usage for that step: python scripts/train_detector.py det3 7 runs/det2/weights/best.pt 0.003"""
import sys
from pathlib import Path
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parent.parent
name, epochs = sys.argv[1], int(sys.argv[2]); start = sys.argv[3] if len(sys.argv) > 3 else "yolo11n.pt"; lr0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01
YOLO(start).train(data=str(ROOT / "det/yolo/data.yaml"), epochs=epochs, lr0=lr0, imgsz=416, batch=16, workers=2, device="cpu", project=str(ROOT / "runs"), name=name,
                  exist_ok=True, patience=0, flipud=0.5, fliplr=0.5, degrees=0, mosaic=1.0, close_mosaic=3, plots=False, amp=False, seed=0, cache=False)
