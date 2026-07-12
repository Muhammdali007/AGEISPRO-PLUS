# Weapon Training Pipeline

This project uses standard Ultralytics YOLO detection models at runtime, not OBB inference.
If your source weapon dataset is YOLO OBB, convert it into normal detect labels before training
the production checkpoint.

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
- `gun`

The AegisPro inference layer already normalizes weapon-family labels into the `weapon`
detector channel, so the runtime can still alert on all of them while preserving finer
class detail in the model.

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

## 2. Train the checkpoint

Recommended default:

```powershell
cd apps/ai
python scripts/train_weapon_model.py `
  --data C:\Users\Hp\Desktop\AegisPro\storage\datasets\weapon-detect-v1\dataset.yaml `
  --model-size m `
  --epochs 80 `
  --device 0 `
  --output-weights C:\Users\Hp\Desktop\AegisPro\storage\models\weapon.pt
```

## 3. Wire the checkpoint into AegisPro

Set:

```env
AI_MODEL_WEAPON_WEIGHTS_PATH=/app/storage/models/weapon.pt
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
- Track per-class precision and recall for `knife`, `scissors`, `pistol`, and `gun`.
- Do not promote a model to production until false positives are reviewed on real camera footage.
