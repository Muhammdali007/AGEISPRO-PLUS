import { z } from "zod";
import { cameraSourceTypes, registrationRoles } from "@/components/camera-form";

export const cameraFormSchema = z.object({
  name: z.string().min(1, "Camera name is required").max(160, "Camera name is too long"),
  registration_role: z.enum(registrationRoles),
  source_type: z.enum(cameraSourceTypes),
  source: z.string().min(1, "Source is required"),
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
});
