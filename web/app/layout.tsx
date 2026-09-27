import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Reprise - Synthesia Reconstruction",
  description:
    "Forensic reconstruction of piano performances from Synthesia-style videos.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
