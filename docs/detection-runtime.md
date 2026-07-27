# Detection runtime

AegisPro uses four detector roles. Local Windows development points these roles at exported
OpenVINO model directories, while the original `.pt` checkpoints remain available for training:

- `yolo11s_openvino_model`: person detection.
- `weapon-v2_openvino_model`: the expanded single-class weapon detector.
- `fire-smoke-v3_openvino_model`: the 640-pixel fire detector (with a smoke fallback class).
- `smoke_openvino_model`: the smoke specialist.

`AI_MODEL_FIRE_WEIGHTS_PATH` and `AI_MODEL_SMOKE_WEIGHTS_PATH` can route fire and smoke to separate
checkpoints. Pointing all three hazard paths at the same combined checkpoint groups them into one
model invocation. If either class-specific setting is empty, it falls back to
`AI_MODEL_FIRE_SMOKE_WEIGHTS_PATH`.

The smoke specialist is intentionally routed separately. Fire uses the 640-pixel export so flames
that occupy a small part of a camera frame retain enough detail for inference. On the 400-image
independent v5 holdout,
at the 0.10 runtime threshold, it reached 0.444 precision and 0.091 recall (20 true positives). The
combined model's sampled smoke path reached 0.286 precision and 0.018 recall. The combined v3
model remains selected for fire because it outperformed the available fire specialist. These are
checkpoint-selection measurements, not production camera acceptance results.

When `AI_MODEL_WEAPON_ENSEMBLE_GENERAL=true`, the dedicated checkpoint remains the weapon-alert
gate and the general checkpoint supplies a finer label only when its box overlaps the specialist
box. This permits `knife` and `scissors` labels from COCO without accepting an unconfirmed generic
scissors prediction as an alert. The current dedicated checkpoint is single-class, so detections
without subtype evidence display as `Other weapon`. Pistol, shotgun, rifle, and other names are
normalized automatically when a promoted multiclass checkpoint exposes those classes. Detector-
specific confidence thresholds are configured in the AI environment. The `/health/runtime`
endpoint reports every configured model path, whether the file exists, its size, the effective
thresholds, the resolved inference device, batch capacity, and the configured soak/load gate status.

The API continuous-detection worker starts when `API_CONTINUOUS_DETECTION_ENABLED=true`. It is the
single inference owner for enabled HTTP, RTSP, and file cameras; camera pages only read the worker's
latest overlay state. The API rejects automatic browser live-scan requests for these server-readable
sources, preventing duplicate frame decoding and model execution. The worker scans according to the
requested inference FPS, while actual throughput is naturally bounded by camera decoding and model
latency. It keeps a bounded per-camera backlog and dispatches up to
`API_CONTINUOUS_DETECTION_BATCH_SIZE` frames to the AI service in one request. Per-camera queue depth
is capped by `API_CONTINUOUS_DETECTION_MAX_PENDING_PER_CAMERA`, so one slow camera can no longer pile
up unbounded work.

Every fast scan runs person and weapon detection. Fire and smoke use a sub-second hazard lane
controlled by `API_CONTINUOUS_DETECTION_HAZARD_INTERVAL_SECONDS` (0.5 seconds by default). This stays
inside the temporal-confirmation window, allowing a borderline hazard to confirm on the next pass
instead of expiring between observations. The AI endpoint runs blocking model work in a worker thread
behind a single inference lock, so health checks and the async event loop remain responsive while
the non-thread-safe tracker/model state stays serialized.

Each completed scan also feeds the operator sound-alert gate. Weapon, fire, and smoke detections
publish `sound.alert` immediately, then use `API_SOUND_ALERT_HAZARD_COOLDOWN_SECONDS` to avoid alarm
spam. Unknown people must be present in `API_SOUND_ALERT_UNKNOWN_SCAN_THRESHOLD` consecutive person
scans (three by default); a clear scan resets the streak. Persistent unknown-person reminders use
`API_SOUND_ALERT_UNKNOWN_COOLDOWN_SECONDS`. The authenticated dashboard receives these events over
its existing websocket and synthesizes distinct hazard and unknown-person tones in the browser.

Browser-local USB cameras are the transport exception: a supervisor or administrator must keep the
camera page open so it can submit the current frame, because the API worker cannot access a device
owned by the browser. The browser does not run inference. It asynchronously encodes a maximum
960-pixel-edge snapshot, while the server requests person, weapon, fire, and smoke on every available
200 ms live pass. Returned boxes are scaled to the preview's natural
coordinates. Explicit manual scans remain restricted to supervisors/administrators and are
rate-limited by `API_MANUAL_CAMERA_SCAN_COOLDOWN_SECONDS`.

Latest-frame overlays are stored separately from incident history and shared through Redis. They
expire after `API_CAMERA_OVERLAY_TTL_SECONDS`. A person box can bridge one short missed frame for
`API_CAMERA_OVERLAY_PERSON_GRACE_SECONDS`. Browser threat boxes remain visible only between their
bounded lane scans and are removed as soon as the next scan for that detector no longer sees them.
This prevents duplicate-incident cooldowns from freezing old box coordinates on a moving feed.

Repeated detections with the same track, known identity, or a highly overlapping box are suppressed
for `API_DETECTION_DUPLICATE_WINDOW_SECONDS`. Production currently uses one API worker so only one
scheduler owns camera processing.

Continuous monitoring applies temporal confirmation to borderline weapon, fire, smoke, and known-
person detections. The confirmation counts are configurable. The calibrated local policy requires
two matching weapon, fire, smoke, and known-identity observations. Weapon scores of at least 0.60
bypass confirmation; the selected weapon calibration measured about 0.80 precision at that point.
Fire and smoke retain confirmation because their current single-frame checkpoints do not meet the
precision gate at a useful bypass threshold. Scale-aware matching lets growing flames, smoke, and
moving hand-held weapons confirm without accepting a distant same-class box as the same candidate.
One missed detector frame is tolerated between matching hits to prevent partial occlusion from
continually restarting confirmation. Threat candidates are returned to the latest-frame overlay on
their first observation with `provisional=true`; provisional boxes never enter incident, alert,
callback, evidence-clip, or sound-alert ingestion. This separates fast operator visibility from the
stricter alert decision. Manual scans are never delayed by this filter, but detector-specific
confidence thresholds still apply. The browser-
transported USB live loop also uses confirmation because it submits consecutive frames; a one-off
privileged manual scan remains the only bypass.

## Throughput controls

- `AI_MODEL_BATCH_SIZE` controls the maximum multi-camera inference batch sent to the GPU-backed
  Ultralytics backend.
- `AI_MODEL_PRELOAD_ON_STARTUP=true` loads every effective checkpoint and performs one real warmup
  inference per model before the service starts serving. This moves the slow first CPU inference
  into startup instead of delaying the first operator scan.
- `AI_MODEL_RUNTIME_AUTOINSTALL=false` prevents Ultralytics from attempting runtime `pip install`
  calls. Tracker dependencies are preinstalled through the `aegispro-ai[model]` extras.
- `AI_MODEL_SNAPSHOT_TRACK_*` controls the per-camera velocity/IoU association used when frames arrive
  as independent browser requests or batch predictions. It produces stable person IDs without
  sharing ByteTrack state across cameras and leaves raw detector coordinates unsmoothed.
- `AI_MODEL_DEVICE=0` is the recommended production setting when the AI container should target the
  first CUDA device.
- On a Windows Intel system with exported OpenVINO models, `AI_MODEL_DEVICE=intel:gpu` selects the
  Intel GPU. Runtime validation falls back to OpenVINO CPU, then regular CPU, when that device is not
  present.
- `AI_MODEL_IOU_THRESHOLD` and `AI_MODEL_MAX_DETECTIONS` bound NMS work. Same-label boxes are also
  suppressed when one covers at least 80% of the smaller box, which removes nested duplicate people.
- Fixed-shape OpenVINO exports read their own `metadata.yaml` input dimensions. This permits 640 px
  person/weapon models and the 320 px hazard model in the same runtime without tensor-shape errors.

### Local latency benchmark (2026-07-18)

On the development machine (i5-1145G7, Intel Iris Xe, CPU-only PyTorch), the same weapon checkpoint
and incident image produced these median model times after warmup:

| Runtime | Median | p95 |
| --- | ---: | ---: |
| PyTorch CPU | 71.0 ms | 97.9 ms |
| OpenVINO CPU | 80.2 ms | - |
| OpenVINO Intel GPU | 21.3 ms | 22.7 ms |

The expanded weapon v2 OpenVINO artifact retained mAP50 0.294 at 640 px versus 0.249 for the prior
artifact, with 18.1 ms average GPU inference across the full 456-image validation split. This is a
single-machine validation result; representative camera replays are still required.

On July 19, three hot-service probes measured 129-147 ms of backend work for the person/weapon lane,
95-122 ms for the separately routed fire/smoke lane, and 216-248 ms when all four roles ran together.
The measurements include model orchestration and differ from isolated per-model timings. A post-warm
fresh `buffalo_m` recognition took 199.8 ms, while
the same spatial match inside the short cache window took 2.6 ms and 47.3 ms end-to-end. These
figures are local single-frame observations, not multi-camera soak results.

## Runtime gates

The AI service can read a runtime validation report from `AI_RUNTIME_GATE_REPORT_PATH`. The report
tracks four promotion gates:

- `load`
- `soak_8h`
- `soak_24h`
- `soak_72h`

The optimization report warns when these gates are missing and blocks promotion when any gate is
failing.

## Model provisioning

Production startup fails if any required checkpoint is missing. Production promotion now also
requires a signed manifest for the weapon and fire/smoke checkpoints. Place these files in
`storage/models` before deployment:

```text
storage/models/yolo11n.pt
storage/models/weapon.pt
storage/models/weapon.pt.promotion.json
storage/models/fire-smoke.pt
storage/models/fire-smoke.pt.promotion.json
storage/models/fire.pt
storage/models/fire.pt.promotion.json
storage/models/smoke.pt
storage/models/smoke.pt.promotion.json
```

Weapon v2 was trained from 1,402 images including 500 balanced hard negatives and gated on 456
validation images. At the 416 px training gate, mAP50 increased from 0.239 to 0.333, mAP50-95 from
0.152 to 0.220, precision from 0.414 to 0.510, and recall from 0.306 to 0.335. The exported 640 px
OpenVINO checkpoint improved mAP50 from 0.249 to 0.294 and recall from 0.284 to 0.316 on Intel GPU.
The July 19 runtime calibration found that the old 0.15 weapon threshold produced only 0.251
precision and 0.422 recall on that validation split. The single-frame precision gate selected a
0.45 operating point with 0.701 precision but only 0.184 recall. Development continuous monitoring
uses a 0.25 candidate floor behind immediate provisional overlays, two-frame temporal
confirmation, and oversized-box rejection. This exposes a possible weapon on its first weak frame
while preventing a one-frame candidate from creating an alert. A separate representative CCTV
holdout is still required; this calibration is not a production acceptance result.

The deployed combined hazard v3 artifact uses a fixed 320 px input. On the independent 400-image
D-Fire test split its OpenVINO mAP50 improved from 0.064 to 0.096 and recall from 0.105 to 0.191;
smoke AP50 improved from 0.006 to 0.057 and fire AP50 from 0.123 to 0.135. Intel GPU inference
averaged 9.4 ms per image. Precision fell from 0.613 to 0.328, so temporal confirmation and camera
replay review remain essential. These staged results improve the development runtime but are not
sufficient for production promotion without signed manifests and representative camera holdouts.
At the current 0.10 fire threshold, the v3 OpenVINO artifact measured 0.166 precision and 0.351 recall
for fire on the independent split. The previous 0.20 operating point measured 0.356 precision and
0.146 recall. First-frame provisional display plus two-frame temporal confirmation is used to retain
the earlier candidate without allowing its lower single-frame precision to create an immediate alert.
Smoke uses a 0.05 candidate threshold with the separately routed specialist and the same provisional/
confirmed split. The checkpoints still fail the production precision target and remain staged pending
stronger retrained models.

A five-epoch low-learning-rate hard-negative candidate was trained on 2026-07-19 and deliberately
not promoted. On the holdout it improved aggregate mAP50 from 0.113 to 0.118 and recall from 0.169
to 0.189, but precision fell from 0.201 to 0.194 and failed the 0.25 training gate. Its exported
fire path reached 0.438 precision and 0.124 recall at threshold 0.20, versus v3's 0.356/0.146. That
trade is retained as `fire-smoke-v4.candidate_openvino_model` for analysis, not live deployment.

Reproduce the operating-point reports with:

```powershell
apps/ai/.venv/Scripts/python.exe apps/ai/scripts/calibrate_detector_thresholds.py `
  --weights storage/models/weapon-v2_openvino_model `
  --data storage/datasets/weapons_openimages_v2_yolo/data.yaml `
  --image-size 640 --device intel:gpu `
  --minimum-precision 0.70 --minimum-recall 0.10
```

The current local weapon and fire/smoke checkpoints should be treated as staged evaluation artifacts,
not production-ready detectors, until all of the following are true:

- licensed datasets have been expanded with representative CCTV footage and hard negatives
- an independent holdout has been evaluated for every promoted detector class
- per-class promotion gates pass at the selected operating threshold
- the promotion manifests are signed and tied to the checkpoint hashes

## Fire/smoke dataset expansion

The reproducible D-Fire normalizer remaps the downloaded source order (`smoke`, `fire`) to the
runtime order (`fire`, `smoke`), clips invalid source boxes, preserves hard-negative frames, and
creates balanced category subsets:

```powershell
apps/ai/.venv/Scripts/python.exe apps/ai/scripts/prepare_fire_smoke_dataset.py `
  --source-data storage/datasets/dfire-21k/data.yaml `
  --source-license "CC0 1.0; preserve original D-Fire citation" `
  --source-data storage/datasets/fire-smoke-yolo/data.yaml `
  --source-license "review required before promotion" `
  --out-dir storage/datasets/fire-smoke-dfire-v5 `
  --max-train 1200 --max-val 400 --max-test 400
```

The v5 staged subset contains equal fire-only, smoke-only, combined, and no-hazard image counts:
1,200 training images, 400 validation images, and 400 test images. All 2,000 images and 3,459 boxes
passed the integrity audit. The previous local combined checkpoint reached only mAP50 0.0227 on the
new independent test split, confirming that its earlier small-dataset score did not generalize to
this domain. Two gated training stages raised the 320 px PyTorch holdout mAP50 to 0.1128 and recall
to 0.1686. Export shape was validated explicitly: the 640 px export was rejected for regression,
while the fixed 320 px OpenVINO export passed the deployment comparison and was promoted locally.

The default policy requires a `weapon_recall_at_selected_threshold` gate with a minimum of `0.90`
before a weapon model can be promoted. See [`model-promotion.md`](model-promotion.md)
for the manifest format and review workflow.

Known-person enrollment and runtime recognition use InsightFace `buffalo_m` with 512-element
embeddings. Its recognition network is byte-identical to the previous local `buffalo_l` recognition
network, so existing embeddings remain compatible; the lighter RetinaFace detector reduced fresh
face analysis from 478.6 ms to 179.3 ms on the local benchmark. Recognition now analyzes a frame
once, associates each detected face with one person box, and follows the stable motion track. Unknown
results refresh after 0.5 seconds; confirmed known identities refresh after 1 second to avoid a
periodic face-analysis stall on every few overlay frames. Runtime faces below the configured
detection score or 32 px minimum size are treated as unknown instead of risking a weak match. A
known label is carried across a borderline frame only
when the same identity remains the best candidate with a valid margin. Multi-angle templates use
top-k/centroid scoring, while near-duplicates and templates below the configured face-detection
quality are ignored. New enrollment photos must contain exactly one clear face of at least 48 px.

The API also curates profiles before JSON serialization and caps runtime templates per identity.
For the current three-person development store this reduced each inference request from 76 raw
templates / 412,246 JSON bytes to 29 templates / 157,286 bytes. A leave-one-template-out check had
100% nearest-identity accuracy and accepted 94.3% at the local 0.60 threshold plus 0.10 margin. That
check measures consistency of enrolled templates; it is not a substitute for an independent known/
unknown camera holdout.

InsightFace's distributed pretrained model weights have a non-commercial research licensing caveat.
Obtain the appropriate license or replace the weights before commercial deployment.
