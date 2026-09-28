"use client";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

export default function AccountMenu() {
  const router = useRouter();
  const [username, setUsername] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/auth/me")
      .then((r) => (r.ok ? r.json() : { user: null }))
      .then((d) => setUsername(d.user))
      .catch(() => {});
  }, []);

  useEffect(() => {
    const h = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
    window.location.href = "/login";
  }

  const initial = (username ?? "?").charAt(0).toUpperCase();

  return (
    <div ref={box} style={{ position: "fixed", top: 10, right: 12, zIndex: 60 }}>
      <button
        onClick={() => setOpen((o) => !o)}
        title={username ?? undefined}
        style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer",
          background: "#16181fcc", border: "1px solid var(--border, #23262d)", borderRadius: 20,
          padding: "4px 10px 4px 4px", color: "var(--text, #e6e8ee)", backdropFilter: "blur(6px)" }}
      >
        <span style={{ width: 24, height: 24, borderRadius: "50%", background: "var(--accent, #4c8bf5)",
          color: "#fff", display: "grid", placeItems: "center", fontSize: 12, fontWeight: 700 }}>
          {initial}
        </span>
        <span style={{ fontSize: 12.5, maxWidth: 140, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
          {username ?? "…"}
        </span>
        <span style={{ fontSize: 10, color: "var(--text-dim, #9aa0ad)" }}>▾</span>
      </button>

      {open && (
        <div className="panel" style={{ position: "absolute", right: 0, top: 40, minWidth: 170,
          borderRadius: 9, padding: 6, boxShadow: "0 10px 30px #000a" }}>
          <MenuItem onClick={() => { setOpen(false); router.push("/settings"); }}>Account settings</MenuItem>
          <div className="divider" style={{ margin: "5px 4px" }} />
          <MenuItem onClick={logout} danger>Log out</MenuItem>
        </div>
      )}
    </div>
  );
}

function MenuItem({ children, onClick, danger }: {
  children: React.ReactNode; onClick: () => void; danger?: boolean;
}) {
  return (
    <button onClick={onClick}
      style={{ display: "block", width: "100%", textAlign: "left", background: "transparent",
        border: "none", cursor: "pointer", padding: "8px 10px", borderRadius: 6, fontSize: 13,
        color: danger ? "#e0533a" : "var(--text, #e6e8ee)" }}
      onMouseEnter={(e) => (e.currentTarget.style.background = "var(--panel-2, #1c1f28)")}
      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
      {children}
    </button>
  );
}
