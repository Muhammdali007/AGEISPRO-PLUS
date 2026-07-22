# Weapon Training Pipeline

This project uses standard Ultralytics YOLO detection models at runtime, not OBB inference.
If your source weapon dataset is YOLO OBB, convert it into normal detect labels before training
the candidate checkpoint.

## Recommended model size

Use these defaults unless benchmarking proves otherwise:

| Hardware target | Recommended base model | Why |
| --- | --- | --- |
| CPU-only or very weak GPU | `yolo11s` | Acceptable for experiments, not the production default. |
| 8 GB VRAM | `yolo11s` | Usable when latency matters more than recall. |
| 10-12 GB VRAM | `yolo11m` | Best default for CCTV weapon detection in this repo. |
| 16-24 GB VRAM | `yolo11l` | Better small-object recall, slower but stronger. |
| 24+ GB VRAM | `yolo11x` | Use only if your validation set shows a real gain. |

For knives, scissors, pistols, and guns in CCTV footage, `yolo11m` is the safest default.
`yolo11n` is too small for production-grade weapon work.

## Class strategy

Train distinct classes:

- `knife`
- `scissors`
- `pistol`
- `rifle`
- `shotgun`
- `other_weapon`

The AegisPro inference layer already normalizes weapon-family labels into the `weapon`
detector channel, so the runtime can still alert on all of them while preserving finer
class detail in the model. It canonicalizes `handgun`/`revolver` as `pistol`, keeps `shotgun` and
`rifle`, and uses `Other weapon` when a single-class checkpoint cannot determine a subtype.

The bundled single-class specialist can be paired with the general model by setting
`AI_MODEL_WEAPON_ENSEMBLE_GENERAL=true`. In that mode, a general `knife` or `scissors` result names
the object only when it overlaps a specialist weapon box. This is an interim labeling aid, not a
replacement for a validated multiclass weapon checkpoint; COCO cannot name pistols, rifles, or
shotguns. Set `AI_MODEL_WEAPON_EXCLUDED_LABELS=scissor,scissors` while using this interim ensemble when the
general model confuses camera knives with scissors. The specialist alert remains active and displays
`Other weapon` until a validated multiclass checkpoint supplies the subtype.

The small July 2026 feedback dataset is an exception: its overlapping source aliases were
collapsed into one `weapon` class because duplicate boxes and very sparse per-class examples were
hurting training. Return to distinct classes after enough independently labeled examples exist for
each weapon type.

## Feedback-assisted retraining

Do not train directly from the model's own alerts. First have a person confirm the weapon box and
mark false-alert frames as negatives; otherwise errors are fed back into the next model. Build the
cleaned weapon, fire/smoke, and smoke-only datasets with:

```powershell
python apps/ai/scripts/prepare_feedback_datasets.py `
  --weapon-base storage/datasets/weapons_openimages_yolo `
  --fire-smoke-base storage/datasets/fire-smoke-yolo `
  --output-root storage/datasets/feedback-YYYYMMDD `
  --knife-positive "storage/incidents/example/snapshot.jpg,0,303,568,413" `
  --fire-positive "storage/incidents/fire-example/snapshot.jpg,120,90,360,410" `
  --smoke-positive "storage/incidents/smoke-example/snapshot.jpg,20,0,420,390" `
  --weapon-hard-negative storage/incidents/phone-false-alert/snapshot.jpg `
  --fire-smoke-hard-negative storage/incidents/lamp-false-alert/snapshot.jpg
```

This is supervised feedback retraining, not reinforcement learning or autonomous live learning.
Keep training candidates separate from live checkpoints until holdout metrics and camera replay
checks pass. Use `--hard-negative` only after a reviewer confirms that a frame contains no weapon,
fire, or smoke at all. Detector-specific negative flags avoid poisoning an unrelated class; for
example, a false weapon alert is not automatically a negative fire example. The generated
`feedback-manifest.json` records every reviewed source image and the labeling policy. The builder
also clamps inherited boxes to the image plane, rejects zero-area labels, and writes smoke positives
to both the combined fire/smoke dataset and the smoke-only dataset.

## Reproducible Open Images expansion

For subtype recognition, use `--preserve-classes`. It canonicalizes the overlapping Open Images
labels into `knife`, `scissors`, `pistol`, `rifle`, `shotgun`, and `other_weapon`, collapses duplicate
source boxes, and adds balanced hard negatives for phones, tools, flashlights, drills, pens,
screwdrivers, cameras, remotes, hair dryers, and toothbrushes:

```powershell
apps/ai/.venv/Scripts/python.exe tools/download_openimages_weapons.py `
  --out-dir storage/datasets/weapons_openimages_multiclass_yolo `
  --preserve-classes `
  --train-hard-negatives 500 `
  --val-hard-negatives 150 `
  --workers 16

apps/ai/.venv/Scripts/python.exe apps/ai/scripts/audit_yolo_dataset.py `
  --data storage/datasets/weapons_openimages_multiclass_yolo/data.yaml `
  --report storage/training-runs/weapon-openimages-multiclass-audit.json
```

The existing single-class 2026-07-18 build contains 1,402 training images (902 positive and 500
hard-negative), 456 validation images (306 positive and 150 hard-negative), and 2,406 weapon boxes.
Its integrity audit found no missing labels, invalid boxes, corrupt images, duplicate content,
cross-split leakage, or conflicting annotations. Rebuild and audit the multiclass variant before
training; do not treat the single-class metrics below as subtype metrics.
Open Images annotation licensing and the image-level licenses must still be reviewed for the
intended deployment.

The locally gated three-epoch v2 run improved validation mAP50 from 0.239 to 0.333 and mAP50-95
from 0.152 to 0.220. Its 640 px OpenVINO artifact reached mAP50 0.294 and recall 0.316 on the same
456-image split at 18.1 ms average Intel GPU inference. Longer CUDA training and independent CCTV
holdouts are still required before production promotion.

## 1. Convert OBB labels into detect labels

Example:

```powershell
cd apps/ai
python scripts/prepare_weapon_dataset.py `
  --source-root C:\Users\Hp\Desktop\AegisPro\storage\datasets\fasdd\your-extracted-dataset `
  --output-root C:\Users\Hp\Desktop\AegisPro\storage\datasets\weapon-detect-v1 `
  --source-names knife,scissors,pistol,gun `
  --class-map C:\Users\Hp\Desktop\AegisPro\apps\ai\training\weapon-class-map.example.json `
  --copy-images
```

That command writes:

- `storage/datasets/weapon-detect-v1/images/...`
- `storage/datasets/weapon-detect-v1/labels/...`
- `storage/datasets/weapon-detect-v1/dataset.yaml`

## 2. Train and gate the candidate checkpoint

`train_detector.py` always retains a candidate checkpoint, compares it with the current checkpoint
on the same evaluation data, and only copies it to the requested output when every configured gate
passes:

```powershell
apps/ai/.venv/Scripts/python.exe apps/ai/scripts/train_detector.py `
  --data storage/datasets/weapons_openimages_multiclass_yolo/data.yaml `
  --weights storage/models/weapon.pt `
  --baseline-weights storage/models/weapon.pt `
  --epochs 30 `
  --image-size 640 `
  --batch 8 `
  --device cpu `
  --run-name weapon-openimages-multiclass `
  --output-weights storage/models/weapon-multiclass.pt `
  --minimum-recall 0.50 `
  --minimum-map50-improvement 0.01 `
  --operating-class pistol `
  --operating-threshold 0.45 `
  --minimum-operating-precision 0.70 `
  --minimum-operating-recall 0.10 `
  --defer-intermediate-checkpoints `
  --export-openvino
```

The operating-point gates are separate from Ultralytics' best-F1 summary. This prevents a model
from passing promotion metrics at one confidence value while the application deploys a noisier
threshold. For a multi-class model, repeat calibration per class before promotion.

CPU fine-tuning is suitable for short local experiments. Use a CUDA training host for the full
30-80 epoch run, then bring the checkpoint back through the same holdout and promotion gates.
`--defer-intermediate-checkpoints` avoids expensive per-epoch optimizer serialization and still
writes the final checkpoint. Omit it when resumable intermediate checkpoints matter more than
throughput.

## 3. Wire the checkpoint into AegisPro

Set:

```env
AI_MODEL_WEAPON_WEIGHTS_PATH=/app/storage/models/weapon-multiclass_openvino_model
```

If your weapon model also includes person detection and you want one shared checkpoint:

```env
AI_MODEL_PERSON_WEAPON_WEIGHTS_PATH=/app/storage/models/weapon.pt
```

## Training quality checklist

- Prefer CCTV-like footage over web photos.
- Keep small weapons visible at the same scale they appear in your cameras.
- Include negative frames with phones, tools, wallets, and metal objects.
- Validate separately on indoor, outdoor, day, night, and occluded scenes.
- Track per-class precision and recall for `knife`, `scissors`, `pistol`, `rifle`, and `shotgun`.
- Record the dataset license for every source that contributes to training or tuning.
- Reserve a truly independent holdout that is not used for training or threshold selection.
- Require a `weapon_recall_at_selected_threshold` gate of at least `0.90` before promotion.
- Do not promote a model to production until false positives and missed threats are reviewed on real camera footage.
- Sign the promotion manifest described in [`model-promotion.md`](model-promotion.md).
