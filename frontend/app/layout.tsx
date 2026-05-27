import type { Metadata } from "next";
import "./globals.css";
import GlobalWebcam from "../components/GlobalWebcam";

export const metadata: Metadata = {
  title: "Attention Dashboard — Facial Expression Analysis",
  description: "Real-time facial expression analysis dashboard with live camera monitoring and engagement tracking",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <GlobalWebcam />
        {children}
      </body>
    </html>
  );
}
