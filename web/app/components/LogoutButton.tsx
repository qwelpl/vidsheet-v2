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
        background: "#e0533a", color: "#fff",
        border: "1px solid #e0533a", borderRadius: 7,
        padding: "5px 10px", fontSize: 11.5, fontWeight: 600, cursor: "pointer",
      }}
    >
      {busy ? "…" : "Log out"}
    </button>
  );
}
