import type { ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/cn";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "ghost";
};

export function Button({ className, variant = "primary", ...props }: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex h-10 items-center justify-center gap-2 rounded-md px-4 text-sm font-medium transition focus:outline-none focus:ring-2 focus:ring-accent disabled:cursor-not-allowed disabled:opacity-60",
        variant === "primary" && "bg-accent text-slate-950 hover:bg-emerald-300",
        variant === "ghost" && "border border-border bg-transparent text-slate-200 hover:bg-panelSoft",
        className
      )}
      {...props}
    />
  );
}
