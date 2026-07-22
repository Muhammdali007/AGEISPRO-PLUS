# Model Promotion

Production deployment and model promotion are now separate concerns in AegisPro.
The platform can be production-hardened while a detector remains non-promoted.

Weapon and fire/smoke checkpoints require a signed promotion manifest before the AI service
will start in production. By default the manifest is stored beside the checkpoint:

- `storage/models/weapon.pt.promotion.json`
- `storage/models/fire-smoke.pt.promotion.json`

## Required evidence

Every promotion manifest must include:

- the checkpoint SHA-256 hash so approval is tied to the exact file being deployed
- every dataset used for training or tuning, with an explicit license for each source
- an `independent_holdout` section stating `independent_from_training=true`
- `per_class_metrics` for the detector classes being promoted
- `gates` evaluated at the selected operating threshold
- at least two signatures by default

The default weapon policy requires a `weapon_recall_at_selected_threshold` gate with
`minimum >= 0.90`.

## Minimal example

```json
{
  "model_id": "weapon-2026-07-14",
  "checkpoint_sha256": "<sha256>",
  "selected_operating_threshold": 0.27,
  "datasets": [
    {
      "name": "Licensed CCTV weapon corpus",
      "license": "Internal collection release 2026-07"
    },
    {
      "name": "Open Images weapon subset",
      "license": "Open Images Terms of Use"
    }
  ],
  "independent_holdout": {
    "name": "weapon-holdout-jul2026",
    "license": "Internal collection release 2026-07",
    "independent_from_training": true
  },
  "per_class_metrics": {
    "weapon": {
      "precision": 0.94,
      "recall": 0.91
    },
    "knife": {
      "precision": 0.92,
      "recall": 0.90
    }
  },
  "gates": [
    {
      "name": "weapon_recall_at_selected_threshold",
      "minimum": 0.90,
      "actual": 0.91,
      "passed": true
    }
  ],
  "signatures": [
    {
      "name": "Model owner",
      "role": "Model owner",
      "signed_at": "2026-07-14T10:00:00Z"
    },
    {
      "name": "Independent reviewer",
      "role": "Independent reviewer",
      "signed_at": "2026-07-14T11:00:00Z"
    }
  ]
}
```

## Review flow

1. Train a candidate checkpoint.
2. Evaluate it on a truly independent holdout that is excluded from training and threshold tuning.
3. Record per-class metrics and operating-threshold gates in the promotion manifest.
4. Confirm dataset provenance and licenses.
5. Collect signatures from the model owner and an independent reviewer.
6. Deploy only the checkpoint whose hash matches the signed manifest.
