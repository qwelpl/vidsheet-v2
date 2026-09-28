import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vidsheet",
  description: "Turn piano videos into sheet music.",
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
