"use client";

import { ShieldAlert, Volume2, VolumeX } from "lucide-react";
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  useState
} from "react";
import { Button } from "@/components/button";
import type { LiveEvent } from "@/lib/api";

const SOUND_ALERT_PREFERENCE_KEY = "aegispro:sound-alerts-enabled";

export type SoundAlertSystemHandle = {
  handleEvent: (event: LiveEvent) => void;
};

type BrowserWindowWithAudio = Window & {
  webkitAudioContext?: typeof AudioContext;
};

export const SoundAlertSystem = forwardRef<SoundAlertSystemHandle>(function SoundAlertSystem(_, ref) {
  const audioContextRef = useRef<AudioContext | null>(null);
  const enabledRef = useRef(true);
  const nextPatternAtRef = useRef(0);
  const [enabled, setEnabled] = useState(true);
  const [ready, setReady] = useState(false);
  const [lastAlert, setLastAlert] = useState<LiveEvent | null>(null);

  const ensureAudioReady = useCallback(async () => {
    if (typeof window === "undefined") {
      return false;
    }

    const AudioContextConstructor =
      window.AudioContext ?? (window as BrowserWindowWithAudio).webkitAudioContext;
    if (!AudioContextConstructor) {
      return false;
    }

    const context = audioContextRef.current ?? new AudioContextConstructor();
    audioContextRef.current = context;
    if (context.state === "suspended") {
      try {
        await context.resume();
      } catch {
        setReady(false);
        return false;
      }
    }

    const isReady = context.state === "running";
    setReady(isReady);
    return isReady;
  }, []);

  const handleEvent = useCallback((event: LiveEvent) => {
    if (event.type !== "sound.alert") {
      return;
    }

    setLastAlert(event);
    if (!enabledRef.current) {
      return;
    }

    void ensureAudioReady().then((isReady) => {
      const context = audioContextRef.current;
      if (!isReady || !context) {
        return;
      }
      nextPatternAtRef.current = playAlertPattern(
        context,
        event.detection_type,
        nextPatternAtRef.current
      );
    });
  }, [ensureAudioReady]);

  useImperativeHandle(ref, () => ({ handleEvent }), [handleEvent]);

  useEffect(() => {
    const storedPreference = window.localStorage.getItem(SOUND_ALERT_PREFERENCE_KEY);
    const shouldEnable = storedPreference !== "false";
    enabledRef.current = shouldEnable;
    setEnabled(shouldEnable);
  }, []);

  useEffect(() => () => {
    const context = audioContextRef.current;
    audioContextRef.current = null;
    if (context) {
      void context.close();
    }
  }, []);

  async function toggleSoundAlerts() {
    if (enabledRef.current && ready) {
      enabledRef.current = false;
      setEnabled(false);
      setReady(false);
      window.localStorage.setItem(SOUND_ALERT_PREFERENCE_KEY, "false");
      if (audioContextRef.current?.state === "running") {
        await audioContextRef.current.suspend();
      }
      return;
    }

    enabledRef.current = true;
    setEnabled(true);
    window.localStorage.setItem(SOUND_ALERT_PREFERENCE_KEY, "true");
    const isReady = await ensureAudioReady();
    const context = audioContextRef.current;
    if (isReady && context) {
      // A confirmation chime proves that this browser/tab has granted audio
      // playback before the operator relies on it for a real camera event.
      nextPatternAtRef.current = playAlertPattern(
        context,
        "unknown_person",
        nextPatternAtRef.current
      );
    }
  }

  const buttonLabel = enabled && ready
    ? "Sound alerts on"
    : enabled
      ? "Enable sound alerts"
      : "Sound alerts muted";

  return (
    <div className="flex flex-wrap items-center gap-2">
      {lastAlert ? (
        <div
          role="status"
          aria-live="assertive"
          title={lastAlert.message}
          className="hidden max-w-72 items-center gap-2 rounded-full border border-red-400/30 bg-red-500/10 px-3 py-2 text-xs text-red-100 xl:flex"
        >
          <ShieldAlert size={15} aria-hidden="true" className="shrink-0 text-red-300" />
          <span className="truncate">{lastAlert.message ?? "Security alert detected"}</span>
        </div>
      ) : null}
      <Button
        type="button"
        variant="ghost"
        onClick={() => void toggleSoundAlerts()}
        aria-pressed={enabled && ready}
        title={
          enabled && ready
            ? "Mute camera sound alerts"
            : "Enable and test camera sound alerts in this browser"
        }
        className={enabled && ready ? "border-emerald-400/30 text-emerald-100" : undefined}
      >
        {enabled && ready
          ? <Volume2 size={16} aria-hidden="true" />
          : <VolumeX size={16} aria-hidden="true" />}
        {buttonLabel}
      </Button>
    </div>
  );
});

function playAlertPattern(
  context: AudioContext,
  detectionType: LiveEvent["detection_type"],
  queuedStart: number
) {
  const isUnknownPerson = detectionType === "unknown_person";
  const frequencies = isUnknownPerson
    ? [440, 554, 659]
    : detectionType === "smoke"
      ? [520, 760, 520, 760]
      : [880, 660, 880, 660];
  const noteDuration = isUnknownPerson ? 0.18 : 0.16;
  const noteGap = isUnknownPerson ? 0.10 : 0.07;
  const now = context.currentTime;
  // Keep simultaneous camera events intelligible, while bounding the queue so
  // a burst from many feeds cannot delay a critical alarm indefinitely.
  const startAt = Math.max(now + 0.03, Math.min(queuedStart, now + 1.5));

  frequencies.forEach((frequency, index) => {
    const noteStart = startAt + index * (noteDuration + noteGap);
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    oscillator.type = isUnknownPerson ? "sine" : "square";
    oscillator.frequency.setValueAtTime(frequency, noteStart);
    gain.gain.setValueAtTime(0.0001, noteStart);
    gain.gain.exponentialRampToValueAtTime(isUnknownPerson ? 0.16 : 0.22, noteStart + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.0001, noteStart + noteDuration);
    oscillator.connect(gain);
    gain.connect(context.destination);
    oscillator.start(noteStart);
    oscillator.stop(noteStart + noteDuration + 0.01);
  });

  return startAt + frequencies.length * (noteDuration + noteGap) + 0.08;
}
