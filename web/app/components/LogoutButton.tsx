"use client";
import { useState } from "react";

export default function LogoutButton() {
  const [busy, setBusy] = useState(false);
  async function logout() {
    setBusy(true);
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } finally {
      window.location.href = "/login";
    }
  }
  return (
    <button
      onClick={logout}
      disabled={busy}
      title="Log out"
      style={{
        position: "fixed", right: 10, bottom: 10, zIndex: 50,
        background: "#0b0c10cc", color: "var(--text-dim, #9aa0ad)",
        border: "1px solid var(--border, #23262d)", borderRadius: 7,
        padding: "5px 10px", fontSize: 11.5, cursor: "pointer",
        backdropFilter: "blur(6px)",
      }}
    >
      {busy ? "…" : "Log out"}
    </button>
  );
}
