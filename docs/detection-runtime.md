# Detection runtime

AegisPro uses three model roles:

- `yolo11n.pt`: person detection and COCO knife/scissors coverage.
- `weapon.pt`: the locally trained handgun, knife, rifle, scissors, shotgun, and weapon detector.
- `fire-smoke.pt`: the locally trained fire and smoke detector.

The AI backend ensembles the generic and dedicated weapon checkpoints, normalizes their labels,
and removes overlapping duplicate boxes. Detector-specific confidence thresholds are configured in
the AI environment. The `/health/runtime` endpoint reports every configured model path, whether the
file exists, its size, and the effective thresholds.

The API continuous-detection worker starts when `API_CONTINUOUS_DETECTION_ENABLED=true`. It scans
enabled cameras according to their requested inference FPS, while actual throughput is naturally
bounded by camera decoding and model latency. HTTP snapshots, file images/videos, RTSP sources, and
locally accessible USB devices are supported. Docker deployments must explicitly expose USB devices
to the API container when USB capture is required.

Repeated detections with the same track, known identity, or a highly overlapping box are suppressed
for `API_DETECTION_DUPLICATE_WINDOW_SECONDS`. Production currently uses one API worker so only one
scheduler owns camera processing.

## Model provisioning

Production startup fails if any required checkpoint is missing. Place these files in
`storage/models` before deployment:

```text
storage/models/yolo11n.pt
storage/models/weapon.pt
storage/models/fire-smoke.pt
```

The local fire/smoke checkpoint was evaluated on the held-out split with overall precision 0.905,
recall 0.35, and mAP50 0.511. Fire performance is materially stronger than smoke performance. The
small local datasets establish a functional pipeline but should be expanded with representative
CCTV footage and hard-negative examples before a safety-critical deployment.

Known-person enrollment and runtime recognition both use InsightFace `buffalo_l` with 512-element
embeddings. Runtime recognition analyzes the full detected-person crop and maps InsightFace's face
box back to the original frame.
