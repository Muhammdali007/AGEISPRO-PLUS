"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, Pencil, Play, Radar, Radio, RefreshCw, Square, Trash2 } from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/button";
import { CameraStreamPanel } from "@/components/camera-stream-panel";
import type { CameraStreamPanelHandle, CapturedCameraFrame } from "@/components/camera-stream-panel";
import type { CameraDetectionScanSummary } from "@/lib/api";
import { EmptyState, InlineLink, SectionCard } from "@/components/dashboard-ui";
import { deleteCamera, getCamera, getCameraDetectionOverlays, getCameraStream, runCameraDetectionScan, runCameraLiveDetectionScan, testCameraConnection, updateCamera } from "@/lib/api";
import { useAuthStore } from "@/lib/auth-store";
import { formatDateTime, labelize, statusTone } from "@/lib/format";
import { cn } from "@/lib/cn";

const LIVE_DETECTION_EMPTY_GRACE_MS = 2500;
// Keep safety models on a sub-second lane while running the considerably more
// expensive face-recognition work less often.
const LIVE_WEAPON_SCAN_INTERVAL_MS = 500;
const LIVE_HAZARD_SCAN_INTERVAL_MS = 500;
const LIVE_RECOGNITION_SCAN_INTERVAL_MS = 2000;
const LIVE_KNOWN_IDENTITY_HOLD_MS = 8000;
const LIVE_PERSON_DETECTION_TYPES = new Set(["person", "known_person", "unknown_person"]);
const LIVE_RECOGNITION_OBSERVED_AT_KEY = "client_recognition_observed_at_ms";

function boxIoU(
  left: CameraDetectionScanSummary["bounding_box"],
  right: CameraDetectionScanSummary["bounding_box"]
) {
  if (!left || !right) {
    return 0;
  }

  const intersectionWidth = Math.max(0, Math.min(left.x2, right.x2) - Math.max(left.x1, right.x1));
  const intersectionHeight = Math.max(0, Math.min(left.y2, right.y2) - Math.max(left.y1, right.y1));
  const intersection = intersectionWidth * intersectionHeight;
  const leftArea = Math.max(0, left.x2 - left.x1) * Math.max(0, left.y2 - left.y1);
  const rightArea = Math.max(0, right.x2 - right.x1) * Math.max(0, right.y2 - right.y1);
  const union = leftArea + rightArea - intersection;
  return union > 0 ? intersection / union : 0;
}

function projectFaceBox(
  faceBox: CameraDetectionScanSummary["face_bounding_box"],
  previousPersonBox: CameraDetectionScanSummary["bounding_box"],
  currentPersonBox: CameraDetectionScanSummary["bounding_box"]
) {
  if (!faceBox || !previousPersonBox || !currentPersonBox) {
    return faceBox;
  }

  const previousWidth = previousPersonBox.x2 - previousPersonBox.x1;
  const previousHeight = previousPersonBox.y2 - previousPersonBox.y1;
  if (previousWidth <= 0 || previousHeight <= 0) {
    return faceBox;
  }

  const currentWidth = currentPersonBox.x2 - currentPersonBox.x1;
  const currentHeight = currentPersonBox.y2 - currentPersonBox.y1;
  return {
    ...faceBox,
    x1: currentPersonBox.x1 + ((faceBox.x1 - previousPersonBox.x1) / previousWidth) * currentWidth,
    y1: currentPersonBox.y1 + ((faceBox.y1 - previousPersonBox.y1) / previousHeight) * currentHeight,
    x2: currentPersonBox.x1 + ((faceBox.x2 - previousPersonBox.x1) / previousWidth) * currentWidth,
    y2: currentPersonBox.y1 + ((faceBox.y2 - previousPersonBox.y1) / previousHeight) * currentHeight
  };
}

function retainPersonIdentity(
  current: CameraDetectionScanSummary,
  previousPeople: CameraDetectionScanSummary[]
) {
  if (current.recognition_status === "known") {
    return current;
  }

  const identityCandidates = previousPeople.filter(
    (detection) =>
      detection.detection_type === "known_person"
      || detection.detection_type === "unknown_person"
      || Boolean(detection.recognition_status)
  );
  const trackMatch = current.track_id
    ? identityCandidates.find((detection) => detection.track_id === current.track_id)
    : undefined;
  const spatialMatch = identityCandidates
    .map((detection) => ({ detection, overlap: boxIoU(current.bounding_box, detection.bounding_box) }))
    .filter(({ overlap }) => overlap >= 0.35)
    .sort((left, right) => right.overlap - left.overlap)[0]?.detection;
  const previousIdentity = trackMatch ?? spatialMatch;
  if (!previousIdentity) {
    return current;
  }
  const previousIsKnown = previousIdentity.recognition_status === "known";
  const candidateIdentityId = typeof current.metadata.candidate_identity_id === "string"
    ? current.metadata.candidate_identity_id
    : "";
  const recognitionError = typeof current.metadata.recognition_error === "string"
    ? current.metadata.recognition_error
    : "";
  const isUnrecognizedPersonPass =
    current.detection_type === "person"
    && !current.recognition_status
    && !current.identity_label;
  const isSameKnownCandidate =
    current.recognition_status === "unknown"
    && previousIsKnown
    && Boolean(previousIdentity.identity_id)
    && candidateIdentityId === previousIdentity.identity_id;
  const isTemporaryFaceMiss =
    current.recognition_status === "unknown"
    && previousIsKnown
    && recognitionError.toLowerCase().includes("no face");

  if (!isUnrecognizedPersonPass && !isSameKnownCandidate && !isTemporaryFaceMiss) {
    return current;
  }

  return {
    ...current,
    detection_type: previousIdentity.detection_type,
    recognition_status: previousIdentity.recognition_status,
    identity_id: previousIdentity.identity_id,
    identity_label: previousIdentity.identity_label,
    match_confidence: previousIdentity.match_confidence,
    person_type: previousIdentity.person_type,
    department: previousIdentity.department,
    reference_id: previousIdentity.reference_id,
    title: previousIdentity.title,
    face_bounding_box: projectFaceBox(
      previousIdentity.face_bounding_box,
      previousIdentity.bounding_box,
      current.bounding_box
    ),
    metadata: {
      ...previousIdentity.metadata,
      ...current.metadata,
      recognition_carried_forward: true
    }
  };
}

function mergeLiveDetectionFrame(
  previous: CameraDetectionScanSummary[] | null,
  incoming: CameraDetectionScanSummary[],
  requestedDetectors: string[]
) {
  const observedAt = Date.now();
  const previousPeople = (previous ?? []).filter((detection) =>
    LIVE_PERSON_DETECTION_TYPES.has(detection.detection_type)
  );
  const currentThreats = incoming.filter(
    (detection) => !LIVE_PERSON_DETECTION_TYPES.has(detection.detection_type)
  );
  const currentPeople = incoming
    .filter((detection) => LIVE_PERSON_DETECTION_TYPES.has(detection.detection_type))
    .map((detection) => detection.recognition_status
      ? {
          ...detection,
          metadata: {
            ...detection.metadata,
            [LIVE_RECOGNITION_OBSERVED_AT_KEY]: observedAt
          }
        }
      : detection)
    .map((detection) => retainPersonIdentity(detection, previousPeople));
  const retainedKnownLabels = new Set(
    currentPeople
      .filter((detection) => detection.recognition_status === "known")
      .map((detection) => detection.identity_label)
      .filter(Boolean)
  );
  const unmatchedKnownPeople = previousPeople.filter((detection) => {
    if (
      detection.recognition_status !== "known"
      || !detection.identity_label
      || retainedKnownLabels.has(detection.identity_label)
    ) {
      return false;
    }
    const rawObservedAt = detection.metadata[LIVE_RECOGNITION_OBSERVED_AT_KEY];
    return typeof rawObservedAt === "number"
      && observedAt - rawObservedAt <= LIVE_KNOWN_IDENTITY_HOLD_MS;
  });
  const retainedPeople = currentPeople.length > 0
    ? [...currentPeople, ...unmatchedKnownPeople]
    : previousPeople;
  const requested = new Set(requestedDetectors);
  const incomingThreatTypes = new Set(currentThreats.map((detection) => detection.detection_type));
  const retainedUnscannedThreats = (previous ?? []).filter((detection) =>
    !LIVE_PERSON_DETECTION_TYPES.has(detection.detection_type)
      && !requested.has(detection.detection_type)
      && !incomingThreatTypes.has(detection.detection_type)
  );
  return [...currentThreats, ...retainedUnscannedThreats, ...retainedPeople];
}

function scaleDetectionsToSource(
  detections: CameraDetectionScanSummary[],
  frame: CapturedCameraFrame
) {
  const scaleX = frame.sourceWidth / frame.width;
  const scaleY = frame.sourceHeight / frame.height;
  if (scaleX === 1 && scaleY === 1) {
    return detections;
  }

  const scaleBox = (box: CameraDetectionScanSummary["bounding_box"]) => box
    ? {
        ...box,
        x1: box.x1 * scaleX,
        y1: box.y1 * scaleY,
        x2: box.x2 * scaleX,
        y2: box.y2 * scaleY
      }
    : box;

  return detections.map((detection) => ({
    ...detection,
    bounding_box: scaleBox(detection.bounding_box),
    face_bounding_box: scaleBox(detection.face_bounding_box)
  }));
}

function formatLiveScanStatusMessage(
  currentFrameCount: number,
  visibleCount: number,
  alertCount: number
) {
  const detectionLabel = visibleCount === 1 ? "detection" : "detections";
  const alertLabel = alertCount === 1 ? "alert" : "alerts";
  if (visibleCount !== currentFrameCount) {
    return `Continuous AI scan active: ${visibleCount} visible ${detectionLabel} (${currentFrameCount} current-frame), ${alertCount} ${alertLabel}.`;
  }
  return `Continuous AI scan active: ${visibleCount} ${detectionLabel}, ${alertCount} ${alertLabel}.`;
}

export default function CameraDetailPage() {
  const params = useParams<{ cameraId: string }>();
  const router = useRouter();
  const { accessToken, user, logout } = useAuthStore();
  const queryClient = useQueryClient();
  const canManageCamera = user?.role === "administrator" || user?.role === "supervisor";
  const streamPanelRef = useRef<CameraStreamPanelHandle | null>(null);
  const liveScanPendingRef = useRef(false);
  const lastLiveWeaponScanRef = useRef(0);
  const lastLiveHazardScanRef = useRef(0);
  const lastLiveRecognitionScanRef = useRef(0);
  const liveDetectionsRef = useRef<CameraDetectionScanSummary[] | null>(null);
  const [liveDetections, setLiveDetections] = useState<CameraDetectionScanSummary[] | null>(null);
  const [liveScanStatus, setLiveScanStatus] = useState<{
    state: "idle" | "waiting" | "scanning" | "ok" | "error";
    message: string;
  }>({
    state: "idle",
    message: "Continuous AI scanning starts when the camera preview is running."
  });

  const cameraQuery = useQuery({
    queryKey: ["camera", params.cameraId, accessToken],
    queryFn: async () => getCamera(accessToken!, params.cameraId),
    enabled: Boolean(accessToken && params.cameraId),
    retry: false
  });
  const streamQuery = useQuery({
    queryKey: ["camera-stream", params.cameraId, accessToken],
    queryFn: async () => getCameraStream(accessToken!, params.cameraId),
    enabled: Boolean(accessToken && params.cameraId),
    retry: false
  });
  const overlayQuery = useQuery({
    queryKey: ["camera-overlays", params.cameraId, accessToken],
    queryFn: async () => getCameraDetectionOverlays(accessToken!, params.cameraId),
    enabled: Boolean(
      accessToken
      && params.cameraId
      && streamQuery.data
      && streamQuery.data.stream_kind !== "browser-camera"
      && !(cameraQuery.data?.source_type === "file" && streamQuery.data.stream_kind !== "image")
    ),
    // This endpoint exposes short-lived latest-frame state, not incident
    // history. Frequent reads are cheap and keep boxes aligned to the worker.
    refetchInterval: cameraQuery.data?.detection_enabled
      ? Math.max(200, Math.round(1000 / Math.max(1, cameraQuery.data.inference_fps)))
      : 1000,
    retry: false
  });
  const testConnection = useMutation({
    mutationFn: async () => testCameraConnection(accessToken!, params.cameraId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["camera", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["camera-stream", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] })
      ]);
    }
  });
  const updateCameraState = useMutation({
    mutationFn: async (nextRunning: boolean) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before changing this camera state.");
      }

      return updateCamera(accessToken, params.cameraId, {
        detection_enabled: nextRunning,
        status: nextRunning ? "unknown" : "disabled"
      });
    },
    onSuccess: async (_, nextRunning) => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["camera", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["camera-stream", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] })
      ]);

      if (nextRunning) {
        testConnection.mutate();
      }
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
      }
    }
  });
  const deleteCameraMutation = useMutation({
    mutationFn: async () => {
      if (!accessToken) {
        throw new Error("You need to sign in again before deleting this camera.");
      }

      return deleteCamera(accessToken, params.cameraId);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] });
      router.push("/dashboard/cameras");
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
      }
    }
  });
  const scanMutation = useMutation({
    mutationFn: async (occurrenceHint?: string) => {
      if (!accessToken) {
        throw new Error("You need to sign in again before running an AI scan.");
      }

      const frame = await streamPanelRef.current?.captureFrame();
      if (stream?.stream_kind === "browser-camera" && !frame) {
        throw new Error("Wait until the live preview is visible, then run the AI scan again.");
      }

      return runCameraDetectionScan(accessToken, params.cameraId, {
        ...(frame
          ? {
              frame_content_base64: frame.contentBase64,
              frame_content_type: frame.contentType
            }
          : {}),
        occurrence_hint: occurrenceHint ?? "privileged_manual_scan"
      });
    },
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["incidents", accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["alerts", accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["camera-overlays", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["camera", params.cameraId, accessToken] }),
        queryClient.invalidateQueries({ queryKey: ["cameras", "list", accessToken] })
      ]);
    },
    onError: (cause) => {
      if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
        logout();
        router.push("/login");
      }
    }
  });
  const camera = cameraQuery.data;
  const stream = streamQuery.data;
  const usesPreviewFrameTransport = Boolean(
    stream?.stream_kind === "browser-camera"
    || (camera?.source_type === "file" && stream?.stream_kind !== "image")
  );
  const isRecordedVideo = Boolean(
    camera?.source_type === "file" && stream?.stream_kind !== "image"
  );
  // Browser-camera and recorded-file detections belong to the exact preview
  // frame just submitted. Do not paint persisted boxes from another frame over
  // moving media.
  const overlays = usesPreviewFrameTransport
    ? liveDetections ?? []
    : overlayQuery.data?.overlays ?? [];
  const isRunning = Boolean(camera?.detection_enabled && camera.status !== "disabled");
  // Server-readable HTTP/RTSP/file sources are already owned by the backend
  // worker. Only transport browser-local frames that the server cannot open.
  const isLiveScanActive = Boolean(
    accessToken
    && camera
    && usesPreviewFrameTransport
    && canManageCamera
    && isRunning
  );
  const liveScanInferenceFps = camera?.inference_fps ?? 1;
  const displayedLiveScanStatus = !isRunning
    ? {
        state: "idle" as const,
        message: "Continuous AI scanning starts when the camera is running."
      }
    : !usesPreviewFrameTransport
      ? {
          state: "ok" as const,
          message: "Backend worker owns inference; this page consumes its latest overlay data."
        }
      : isLiveScanActive
        ? liveScanStatus
        : {
            state: "idle" as const,
            message: "A supervisor or administrator must keep this preview open to transport frames."
          };

  useEffect(() => {
    if (cameraQuery.error instanceof Error && cameraQuery.error.message.includes("Camera not found")) {
      router.replace("/dashboard/cameras");
    }
  }, [cameraQuery.error, router]);

  useEffect(() => {
    if (!accessToken || !isLiveScanActive) {
      liveScanPendingRef.current = false;
      return;
    }

    let stopped = false;
    const token = accessToken;
    let overlayExpiry: number | undefined;
    // A browser camera cannot queue frames like a media worker. Start the next
    // scan only after the previous one completes and discard intermediate
    // frames. This keeps inference latency from turning into video latency.
    const minimumIntervalMs = isRecordedVideo ? 100 : 200;
    const intervalMs = Math.max(
      minimumIntervalMs,
      Math.round(1000 / Math.max(1, liveScanInferenceFps))
    );
    let nextScan: number | undefined;
    // The first readable preview frame must run every safety detector. Delaying
    // the hazard lane here made a newly visible fire/smoke/weapon wait before
    // it could even receive its first observation.
    const laneStartedAt = performance.now();
    lastLiveWeaponScanRef.current = laneStartedAt - LIVE_WEAPON_SCAN_INTERVAL_MS;
    lastLiveHazardScanRef.current = laneStartedAt - LIVE_HAZARD_SCAN_INTERVAL_MS;
    lastLiveRecognitionScanRef.current = laneStartedAt;

    function cancelOverlayExpiry() {
      if (overlayExpiry !== undefined) {
        window.clearTimeout(overlayExpiry);
        overlayExpiry = undefined;
      }
    }

    function setCurrentLiveDetections(detections: CameraDetectionScanSummary[] | null) {
      liveDetectionsRef.current = detections;
      setLiveDetections(detections);
    }

    function clearPeopleAfterGraceFromLiveDetections() {
      setLiveDetections((current) => {
        const retained = (current ?? []).filter(
          (detection) => !LIVE_PERSON_DETECTION_TYPES.has(detection.detection_type)
        );
        liveDetectionsRef.current = retained;
        return retained;
      });
    }

    function expireOverlayAfterGrace() {
      if (stopped || overlayExpiry !== undefined) {
        return;
      }
      overlayExpiry = window.setTimeout(() => {
        overlayExpiry = undefined;
        if (!stopped) {
          clearPeopleAfterGraceFromLiveDetections();
        }
      }, LIVE_DETECTION_EMPTY_GRACE_MS);
    }

    function scheduleNextScan(delayMs = intervalMs) {
      if (stopped) {
        return;
      }
      if (nextScan !== undefined) {
        window.clearTimeout(nextScan);
      }
      nextScan = window.setTimeout(() => {
        nextScan = undefined;
        void scanLiveFrame();
      }, delayMs);
    }

    async function scanLiveFrame() {
      if (stopped) {
        return;
      }
      if (liveScanPendingRef.current || document.visibilityState !== "visible") {
        scheduleNextScan(Math.max(intervalMs, 500));
        return;
      }

      const scanStartedAt = performance.now();
      const frame = await streamPanelRef.current?.captureFrame();
      if (!frame || stopped) {
        if (!stopped) {
          setLiveScanStatus({
            state: "waiting",
            message: "Waiting for a readable camera frame from the live preview."
          });
          scheduleNextScan(Math.max(intervalMs, 500));
        }
        return;
      }

      liveScanPendingRef.current = true;
      // Once this live loop owns the overlay, never fall back to coordinates
      // from an older persisted incident.
      setCurrentLiveDetections(liveDetectionsRef.current ?? []);
      setLiveScanStatus({
        state: "scanning",
        message: "Continuous AI scan is analyzing the latest camera frame."
      });
      try {
        const scanClock = performance.now();
        // Recorded footage is finite and may contain a hazard for only a few
        // frames, so run the complete AI suite on every analyzed video frame.
        // Live sources retain the lower-cost specialist cadence.
        const weaponScanDue =
          isRecordedVideo
          || scanClock - lastLiveWeaponScanRef.current >= LIVE_WEAPON_SCAN_INTERVAL_MS;
        const hazardScanDue =
          isRecordedVideo
          || scanClock - lastLiveHazardScanRef.current >= LIVE_HAZARD_SCAN_INTERVAL_MS;
        const recognitionScanDue =
          isRecordedVideo
          || scanClock - lastLiveRecognitionScanRef.current >= LIVE_RECOGNITION_SCAN_INTERVAL_MS;
        const requestedDetectors = [
          "person",
          ...(weaponScanDue ? ["weapon"] : []),
          ...(hazardScanDue ? ["fire", "smoke"] : [])
        ];
        const result = await runCameraLiveDetectionScan(token, params.cameraId, {
          frame_content_base64: frame.contentBase64,
          frame_content_type: frame.contentType,
          // All safety detectors share the low-latency lane. Threat candidates
          // are visible immediately; alerts still require temporal confirmation.
          requested_detectors: requestedDetectors,
          recognition_enabled: recognitionScanDue,
          occurrence_hint: "dashboard_live_scan"
        });
        // Measure each specialist lane from completion. If a combined scan
        // itself takes longer than its interval, measuring from start makes
        // every following request combined as well and starves person-only
        // frames indefinitely.
        const scanCompletedAt = performance.now();
        if (weaponScanDue) {
          lastLiveWeaponScanRef.current = scanCompletedAt;
        }
        if (hazardScanDue) {
          lastLiveHazardScanRef.current = scanCompletedAt;
        }
        if (recognitionScanDue) {
          lastLiveRecognitionScanRef.current = scanCompletedAt;
        }
        if (!stopped) {
          const currentDetections = scaleDetectionsToSource(result.detections, frame);
          const mergedDetections = mergeLiveDetectionFrame(
            liveDetectionsRef.current,
            currentDetections,
            requestedDetectors
          );
          const hasCurrentPeople = currentDetections.some((detection) =>
            LIVE_PERSON_DETECTION_TYPES.has(detection.detection_type)
          );
          setCurrentLiveDetections(mergedDetections);
          if (hasCurrentPeople) {
            cancelOverlayExpiry();
          } else {
            // People receive a bounded missed-frame grace period. Threat boxes
            // are retained only until that detector's next scheduled lane pass.
            expireOverlayAfterGrace();
          }
          setLiveScanStatus({
            state: "ok",
            message: formatLiveScanStatusMessage(
              result.detection_count,
              mergedDetections.length,
              result.alert_count
            )
          });

          if (result.incident_count > 0 || result.alert_count > 0) {
            void queryClient.invalidateQueries({ queryKey: ["incidents", token] });
            void queryClient.invalidateQueries({ queryKey: ["alerts", token] });
          }
        }
      } catch (cause) {
        if (cause instanceof Error && (cause.message === "Invalid credentials" || cause.message === "Session expired")) {
          logout();
          router.push("/login");
          return;
        }
        if (!stopped) {
          expireOverlayAfterGrace();
          setLiveScanStatus({
            state: "error",
            message: cause instanceof Error ? cause.message : "Continuous AI scan failed."
          });
        }
      } finally {
        liveScanPendingRef.current = false;
        // Hold a start-to-start cadence. Adding the full interval after a slow
        // response makes configured FPS drift by the inference latency itself.
        const elapsedMs = performance.now() - scanStartedAt;
        scheduleNextScan(Math.max(0, intervalMs - elapsedMs));
      }
    }

    scheduleNextScan(0);

    return () => {
      stopped = true;
      if (nextScan !== undefined) {
        window.clearTimeout(nextScan);
      }
      cancelOverlayExpiry();
      setCurrentLiveDetections(null);
    };
  }, [
    accessToken,
    isLiveScanActive,
    isRecordedVideo,
    liveScanInferenceFps,
    logout,
    params.cameraId,
    queryClient,
    router
  ]);

  async function handleDeleteCamera() {
    if (!window.confirm("Delete this camera? This action cannot be undone.")) {
      return;
    }

    try {
      await deleteCameraMutation.mutateAsync();
    } catch {
      // The mutation error is rendered elsewhere if needed.
    }
  }

  return (
    <div className="space-y-6">
      <SectionCard
        title={camera?.name ?? "Camera details"}
        description="Live preview consumes server-owned detection overlays while backend workers own continuous inference."
        action={
          <div className="flex flex-wrap items-center gap-3">
            <Button
              type="button"
              variant="ghost"
              onClick={() => testConnection.mutate()}
              disabled={testConnection.isPending || !accessToken}
            >
              <RefreshCw size={16} className={cn(testConnection.isPending && "animate-spin")} />
              {testConnection.isPending ? "Testing..." : "Test connection"}
            </Button>
            {canManageCamera ? (
              <>
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => scanMutation.mutate(undefined)}
                  disabled={scanMutation.isPending || !accessToken || !camera}
                >
                  <Radar size={16} aria-hidden="true" />
                  {scanMutation.isPending ? "Scanning..." : "Run manual scan"}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => updateCameraState.mutate(!isRunning)}
                  disabled={updateCameraState.isPending || !camera}
                >
                  {isRunning ? <Square size={16} aria-hidden="true" /> : <Play size={16} aria-hidden="true" />}
                  {updateCameraState.isPending ? (isRunning ? "Turning off..." : "Starting...") : isRunning ? "Turn off camera" : "Start camera"}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  onClick={() => router.push(`/dashboard/cameras/${params.cameraId}/edit`)}
                  disabled={!camera}
                >
                  <Pencil size={16} aria-hidden="true" />
                  Edit
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  className="border-danger/50 text-red-200 hover:bg-danger/10"
                  onClick={handleDeleteCamera}
                  disabled={deleteCameraMutation.isPending || !camera}
                >
                  <Trash2 size={16} aria-hidden="true" />
                  {deleteCameraMutation.isPending ? "Deleting..." : "Delete"}
                </Button>
              </>
            ) : null}
            <InlineLink href="/dashboard/cameras" label="Back to cameras" />
          </div>
        }
      >
        {cameraQuery.error instanceof Error ? (
          <EmptyState title="Camera unavailable" description={cameraQuery.error.message} />
        ) : !camera ? (
          <EmptyState
            title="Loading camera"
            description="Fetching the camera configuration and health metadata from the API."
          />
        ) : (
          <div className="space-y-6">
            <div className="grid gap-6 xl:grid-cols-[1.5fr_1fr]">
              <div className="rounded-[26px] border border-white/10 bg-black/20 p-4">
                <div className="mb-4 flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-slate-200">Live preview</p>
                    <p className="mt-1 text-sm text-slate-400">
                      {stream?.health_message ?? "Waiting for stream configuration details from the API."}
                    </p>
                  </div>
                  <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs uppercase tracking-[0.18em] text-slate-300">
                    <Radio size={14} className="text-accent" />
                    {stream?.is_live ? "Live" : "Recorded"}
                  </div>
                </div>

                {streamQuery.error instanceof Error ? (
                  <EmptyState title="Stream metadata unavailable" description={streamQuery.error.message} />
                ) : stream ? (
                  <CameraStreamPanel
                    ref={streamPanelRef}
                    accessToken={accessToken!}
                    camera={camera}
                    stream={stream}
                    detections={overlays}
                  />
                ) : (
                  <EmptyState
                    title="Preparing stream profile"
                    description="Resolving how this camera source should be previewed in the browser."
                  />
                )}
              </div>

              <div className="space-y-4">
                <HealthCard
                  title="Health"
                  value={labelize(stream?.health_status ?? camera.status)}
                  tone={statusTone(stream?.health_status ?? camera.status)}
                  detail={formatDateTime(stream?.checked_at ?? camera.health_checked_at)}
                />
                <HealthCard
                  title="Detection"
                  value={isRunning ? "Running" : "Stopped"}
                  tone={
                    isRunning
                      ? "bg-emerald-500/15 text-emerald-200"
                      : "bg-slate-500/20 text-slate-200"
                  }
                  detail={isRunning ? `${camera.inference_fps} inference FPS configured on backend worker` : "Camera is paused for detection and monitoring workflows"}
                />
                <HealthCard
                  title="Continuous scan"
                  value={labelize(displayedLiveScanStatus.state)}
                  tone={
                    displayedLiveScanStatus.state === "ok" || displayedLiveScanStatus.state === "scanning"
                      ? "bg-emerald-500/15 text-emerald-200"
                      : displayedLiveScanStatus.state === "error"
                        ? "bg-red-500/15 text-red-200"
                        : "bg-slate-500/20 text-slate-200"
                  }
                  detail={displayedLiveScanStatus.message}
                />
                <HealthCard
                  title="Playback mode"
                  value={stream ? labelize(stream.stream_kind) : "Loading"}
                  tone="bg-cyan-500/15 text-cyan-100"
                  detail={stream?.requires_relay ? "Relay required for browser playback" : "Direct playback supported"}
                />
              </div>
            </div>

            {testConnection.data ? (
              <div className="rounded-[22px] border border-white/10 bg-black/15 p-4 text-sm text-slate-300">
                <div className="flex items-center gap-2">
                  <Activity size={16} className="text-accent" />
                  Most recent test: {testConnection.data.message}
                </div>
                <p className="mt-2 text-slate-400">
                  Completed {formatDateTime(testConnection.data.checked_at)}
                  {typeof testConnection.data.latency_ms === "number"
                    ? ` | ${testConnection.data.latency_ms} ms`
                    : ""}
                </p>
              </div>
            ) : null}

            {testConnection.error instanceof Error ? (
              <EmptyState title="Connection test failed" description={testConnection.error.message} />
            ) : null}

            {scanMutation.data ? (
              <div className="rounded-[22px] border border-emerald-400/20 bg-emerald-500/10 p-4 text-sm text-emerald-100">
                <div className="flex items-center gap-2">
                  <Radar size={16} aria-hidden="true" />
                  Scan complete: {scanMutation.data.detection_count} detections, {scanMutation.data.incident_count} incidents, {scanMutation.data.alert_count} alerts.
                </div>
                <p className="mt-2 text-emerald-200/80">
                  Backend: {scanMutation.data.backend ?? "unknown"} | Model: {scanMutation.data.model_name}
                </p>
                {scanMutation.data.backend === "simulated" ? (
                  <p className="mt-2 text-amber-200/90">
                    The AI service is running in simulated mode, so it will not perform real face or knife detection from the camera feed.
                  </p>
                ) : null}
                <p className="mt-2 text-emerald-200/80">
                  {scanMutation.data.detections.length > 0
                    ? scanMutation.data.detections
                        .map((detection) =>
                          `${labelize(detection.object_label || detection.detection_type)} ${Math.round(detection.confidence * 100)}%${detection.identity_label ? ` (${detection.identity_label})` : ""}`
                        )
                        .join(" | ")
                    : scanMutation.data.ignored_reasons.join(" | ") || "No detections were produced for this frame."}
                </p>
              </div>
            ) : null}

            {scanMutation.error instanceof Error ? (
              <EmptyState title="AI scan failed" description={scanMutation.error.message} />
            ) : null}

            {updateCameraState.error instanceof Error ? (
              <EmptyState title="Unable to update camera state" description={updateCameraState.error.message} />
            ) : null}

            {deleteCameraMutation.error instanceof Error ? (
              <EmptyState title="Unable to delete camera" description={deleteCameraMutation.error.message} />
            ) : null}

            <div className="grid gap-4 lg:grid-cols-2">
              <DetailBlock label="Status" value={labelize(camera.status)} tone={statusTone(camera.status)} />
              <DetailBlock
                label="Source type"
                value={labelize(camera.source_type)}
                tone="bg-cyan-500/15 text-cyan-100"
              />
              <DetailBlock label="Source" value={camera.source} tone="bg-black/20 text-slate-200" />
              <DetailBlock
                label="Credential status"
                value={camera.credentials_rotation_required ? "Rotation required" : camera.source_redacted ? "Protected" : "Not redacted"}
                tone={camera.credentials_rotation_required ? "bg-amber-500/15 text-amber-100" : "bg-black/20 text-slate-200"}
              />
              <DetailBlock label="Location" value={camera.location ?? "Not set"} tone="bg-black/20 text-slate-200" />
              <DetailBlock label="Group" value={camera.group ?? "Ungrouped"} tone="bg-black/20 text-slate-200" />
              <DetailBlock label="Inference FPS" value={`${camera.inference_fps}`} tone="bg-emerald-500/15 text-emerald-200" />
              <DetailBlock label="Health checked" value={formatDateTime(camera.health_checked_at)} tone="bg-black/20 text-slate-200" />
              <DetailBlock label="Last seen" value={formatDateTime(camera.last_seen_at)} tone="bg-black/20 text-slate-200" />
              <DetailBlock label="Tags" value={camera.tags.length ? camera.tags.join(", ") : "No tags"} tone="bg-black/20 text-slate-200" />
              <DetailBlock
                label="Stream notes"
                value={stream?.notes.join(" | ") || "No stream notes yet"}
                tone="bg-black/20 text-slate-200"
              />
            </div>
          </div>
        )}
      </SectionCard>
    </div>
  );
}

function HealthCard({
  title,
  value,
  detail,
  tone
}: {
  title: string;
  value: string;
  detail: string;
  tone: string;
}) {
  return (
    <div className="rounded-[22px] border border-white/10 bg-black/15 p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{title}</p>
      <p className={cn("mt-3 inline-flex rounded-full px-2.5 py-1 text-sm font-medium", tone)}>{value}</p>
      <p className="mt-3 text-sm text-slate-400">{detail}</p>
    </div>
  );
}

function DetailBlock({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="rounded-[22px] border border-white/10 bg-black/15 p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className={cn("mt-3 inline-flex max-w-full rounded-full px-2.5 py-1 text-sm font-medium", tone)}>{value}</p>
    </div>
  );
}
