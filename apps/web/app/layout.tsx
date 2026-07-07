import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";
import { Providers } from "@/app/providers";

export const metadata: Metadata = {
  title: "AegisPro",
  description: "AI surveillance management platform"
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body className="bg-background text-slate-100 antialiased">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
