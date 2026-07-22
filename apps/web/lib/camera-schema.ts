import { z } from "zod";
import { cameraSourceTypes, registrationRoles } from "@/components/camera-form";

export function buildCameraFormSchema({ requireSource }: { requireSource: boolean }) {
  return z.object({
    name: z.string().min(1, "Camera name is required").max(160, "Camera name is too long"),
    registration_role: z.enum(registrationRoles),
    source_type: z.enum(cameraSourceTypes),
    source: requireSource ? z.string().min(1, "Source is required") : z.string(),
    location: z.string().max(255, "Location is too long").optional().or(z.literal("")),
    group: z.string().max(120, "Group is too long").optional().or(z.literal("")),
    tags: z.string(),
    inference_fps: z.coerce.number().int().min(1, "Inference FPS must be at least 1").max(30, "Inference FPS cannot exceed 30"),
    detection_enabled: z.boolean(),
    metadata: z.string().superRefine((value, ctx) => {
      if (!value.trim()) {
        return;
      }

      try {
        const parsed = JSON.parse(value) as unknown;
        if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            message: "Metadata must be a JSON object."
          });
        }
      } catch {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "Metadata must be valid JSON."
        });
      }
    })
  }).superRefine((values, ctx) => {
    if (!values.source.trim() || values.source_type !== "http") {
      return;
    }
    try {
      const url = new URL(values.source.trim().replace(/^(ipv4|ipv6)\s*:\s*/i, ""));
      if (url.protocol !== "http:" && url.protocol !== "https:") {
        throw new Error("invalid protocol");
      }
    } catch {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["source"],
        message: "HTTP cameras require a complete http:// or https:// URL."
      });
    }
  });
}

export const cameraFormSchema = buildCameraFormSchema({ requireSource: true });
