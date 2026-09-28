"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

export default function SettingsPage() {
  const router = useRouter();
  const [username, setUsername] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/auth/me")
      .then((r) => (r.ok ? r.json() : { user: null }))
      .then((d) => setUsername(d.user))
      .catch(() => setUsername(null));
  }, []);

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg, #0a0b0e)", color: "var(--text, #e6e8ee)" }}>
      <div style={{ maxWidth: 560, margin: "0 auto", padding: "28px 20px 60px" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 22 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
            <button className="btn btn-ghost" onClick={() => router.push("/")}>&larr; Back</button>
            <h1 style={{ fontSize: 20, fontWeight: 700, margin: 0 }}>Account settings</h1>
          </div>
          <span className="chip mono">{username ?? "…"}</span>
        </div>

        <ChangeUsername current={username} onChanged={setUsername} />
        <ChangePassword />
        <DangerZone />
      </div>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="panel" style={{ borderRadius: 10, padding: 18, marginBottom: 16 }}>
      <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 12 }}>{title}</div>
      {children}
    </div>
  );
}

function Row({ label, ...rest }: { label: string } & React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 10 }}>
      <span style={{ fontSize: 12, letterSpacing: ".02em", color: "var(--text-faint)" }}>{label}</span>
      <input {...rest} spellCheck={false}
        style={{ background: "#0b0c10", border: "1px solid var(--border, #23262d)",
          borderRadius: 7, padding: "9px 11px", color: "inherit", fontSize: 14, outline: "none" }} />
    </label>
  );
}

function Status({ msg }: { msg: { ok: boolean; text: string } | null }) {
  if (!msg) return null;
  return <div style={{ fontSize: 12.5, marginTop: 4, color: msg.ok ? "#35b36b" : "#e0533a" }}>{msg.text}</div>;
}

function ChangeUsername({ current, onChanged }: { current: string | null; onChanged: (u: string) => void }) {
  const [newUsername, setNew] = useState("");
  const [currentPassword, setPw] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setMsg(null);
    const r = await fetch("/api/auth/account", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "username", newUsername, currentPassword }),
    });
    const d = await r.json().catch(() => ({}));
    setBusy(false);
    if (!r.ok) { setMsg({ ok: false, text: d.error || "Failed" }); return; }
    onChanged(d.username); setNew(""); setPw("");
    setMsg({ ok: true, text: "Username updated" });
  }

  return (
    <Card title="Username">
      <form onSubmit={submit}>
        <Row label="New username" value={newUsername} placeholder={current ?? ""}
          onChange={(e) => setNew(e.target.value)} autoComplete="off" />
        <Row label="Current password" type="password" value={currentPassword}
          onChange={(e) => setPw(e.target.value)} autoComplete="current-password" />
        <button className="btn btn-primary" disabled={busy || !newUsername || !currentPassword}
          style={{ marginTop: 4 }}>{busy ? "…" : "Update username"}</button>
        <Status msg={msg} />
      </form>
    </Card>
  );
}

function ChangePassword() {
  const [currentPassword, setCur] = useState("");
  const [newPassword, setNew] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true); setMsg(null);
    const r = await fetch("/api/auth/account", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "password", currentPassword, newPassword }),
    });
    const d = await r.json().catch(() => ({}));
    setBusy(false);
    if (!r.ok) { setMsg({ ok: false, text: d.error || "Failed" }); return; }
    setCur(""); setNew("");
    setMsg({ ok: true, text: "Password updated" });
  }

  return (
    <Card title="Password">
      <form onSubmit={submit}>
        <Row label="Current password" type="password" value={currentPassword}
          onChange={(e) => setCur(e.target.value)} autoComplete="current-password" />
        <Row label="New password" type="password" value={newPassword}
          onChange={(e) => setNew(e.target.value)} autoComplete="new-password" />
        <button className="btn btn-primary" disabled={busy || !currentPassword || !newPassword}
          style={{ marginTop: 4 }}>{busy ? "…" : "Update password"}</button>
        <Status msg={msg} />
      </form>
    </Card>
  );
}

function DangerZone() {
  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" }).catch(() => {});
    window.location.href = "/login";
  }
  return (
    <Card title="Session">
      <button onClick={logout} className="btn"
        style={{ background: "#e0533a", color: "#fff", border: "1px solid #e0533a", fontWeight: 600 }}>
        Log out
      </button>
    </Card>
  );
}
