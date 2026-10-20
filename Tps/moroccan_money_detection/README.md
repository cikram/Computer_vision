# Moroccan money detection

Finds every coin and banknote in a photo, a video or the webcam, reads each value and prints the total in DH.

Pipeline: **YOLO11n detector** (2 classes: coin / note) → **EfficientNet-B0 classifier** (11 classes: 0.1, 0.2, 0.5, 1, 2, 5, 10 DH coins and 20, 50, 100, 200 DH notes) → total.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```
The trained weights are in `models/`, so no training is needed to run the detector.

## Run
```bash
python detect_yolo.py --source photo.jpg     # an image
python detect_yolo.py --source clip.mp4      # a video
python detect_yolo.py                        # webcam (q = quit)
```
Example output:
```
trusted: 2 x 10 + 5 x 5 + 1 x 1 + 4 x 0.5 + 1 x 0.2 = 48.2 DH | all detected: 48.2 DH
saved results/coins_detected.jpg
```
- **Green box** = trusted (classifier confidence ≥ 0.5 for coins, ≥ 0.8 for notes). **Orange box** = uncertain.
- Two totals are shown: *Trusted* (green only) and *All detected* (green + orange).
- Annotated images are saved to `results/`.

Useful options:

| option | default | meaning |
|---|---|---|
| `--conf-coin` / `--conf-note` | 0.5 / 0.8 | classifier threshold for a "trusted" coin / note |
| `--det-conf` | 0.25 | detector confidence threshold |
| `--imgsz` | 416 | detector input size |
| `--ignore x0,y0,x1,y1` | – | pixel rectangle to ignore (repeatable) |
| `--out` | `results/` | output folder |
| `--no-show` | off | do not open a window |

## Project layout
```
detect_yolo.py            main script (detector + classifier)
moneylib.py               classes, augmentations, EfficientNet-B0 builder, dataset loading
detect.py                 classifier loader + the original FastSAM pipeline (used as baseline)
make_detector_data.py     builds det/ (cut-outs, backgrounds) from the raw datasets
models/
  detector.pt             YOLO11n coin/note detector (v3, trained with false objects)
  best_model_v2.pt        final classifier (all folds)
  holdout_model_v2.pt     classifier trained without fold 0, for honest tests
  classes.json            class names
scripts/
  compose.py, distract.py synthetic scenes (objects pasted on real backgrounds, + false objects)
  build_yolo.py           converts the scenes to the YOLO format
  train_detector.py       trains the YOLO detector
  train_classifier_v2.py  trains the classifier
  pipe_eval.py, eval_all.py, eval_d.py   evaluation
notebooks/
  01_build_merged_dataset.ipynb   merges the 3 datasets, removes duplicates, builds the folds
data/merged/index.csv     list of all images with label, source, group and fold (fold 0 = test)
```

## Data (not in the repo)
The images are not committed. Put them in `data/` to retrain:

| dataset | content | folder |
|---|---|---|
| `web_moroccan_money` | 355 web/catalogue photos, one object per image | `data/raw/web_moroccan_money/` |
| `moroccan_coins_yolo` | 184 phone photos of several coins, polygons (Roboflow, MIT) | `data/raw/moroccan_coins_yolo/` |
| `moroccan_currency_yolo` | 658 phone photos of one note (Roboflow, CC BY 4.0) | `data/raw/moroccan_currency_yolo/` |

The notebook `01_build_merged_dataset.ipynb` then creates `data/merged/` (`web/`, `phone_coin/`, `phone_note/`).
Folds are built on **groups** (copies or near-copies of the same photo stay in the same fold), so test images never leak into training.

## Retrain
```bash
python make_detector_data.py .
python scripts/compose.py 600 150 150
python scripts/build_yolo.py
python scripts/train_detector.py det 60 yolo11n.pt          # yolo11n.pt is downloaded automatically
# false-object step (logos, icons, rings, shadows as background), fine-tunes the detector:
python scripts/compose.py 1500 300 300 syn_d
python scripts/build_yolo.py syn_d
python scripts/train_detector.py det3 7 runs/det/weights/best.pt 0.003
python scripts/eval_d.py runs/det3/weights/best.pt           # held-out false-object test
python scripts/train_classifier_v2.py all 2 12 models/best_model_v2.pt
```
Generated folders (`det/`, `runs/`, `results/`) are git-ignored.
