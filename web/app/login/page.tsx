"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import NoteRain from "../components/NoteRain";

type Mode = "login" | "redeem";

export default function LoginPage() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("login");
  const [key, setKey] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const endpoint = mode === "redeem" ? "/api/auth/signup" : "/api/auth/login";
      const payload =
        mode === "redeem" ? { key, username, password } : { username, password };
      const r = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        setError(d.error || "Something went wrong");
        setBusy(false);
        return;
      }
      router.replace("/");
      router.refresh();
    } catch {
      setError("Network error");
      setBusy(false);
    }
  }

  return (
    <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", position: "relative", overflow: "hidden",
      background: "var(--bg, #0a0b0e)", color: "var(--text, #e6e8ee)", padding: 20 }}>
      <NoteRain />
      <div className="panel vs-panel vs-rise" style={{ width: 360, maxWidth: "100%", borderRadius: 12,
        padding: 24, position: "relative", zIndex: 1 }}>
        <div style={{ fontSize: 26, fontWeight: 800, letterSpacing: "-0.03em", marginBottom: 2, color: "var(--text)" }}>Vidsheet</div>
        <div style={{ fontSize: 13, color: "var(--text-dim, #9aa0ad)", marginBottom: 18 }}>
          {mode === "login" ? "Sign in to continue" : "Redeem your license key"}
        </div>

        <div className="seg" style={{ display: "flex", marginBottom: 16 }}>
          <button className={mode === "login" ? "active" : ""} style={{ flex: 1 }}
            onClick={() => { setMode("login"); setError(null); }}>Log in</button>
          <button className={mode === "redeem" ? "active" : ""} style={{ flex: 1 }}
            onClick={() => { setMode("redeem"); setError(null); }}>Sign up</button>
        </div>

        <form onSubmit={submit} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {mode === "redeem" && (
            <Field label="License key" value={key} onChange={setKey}
              placeholder="XXXX-XXXX-XXXX" autoFocus mono />
          )}
          <Field label="Username" value={username} onChange={setUsername}
            placeholder="your name" autoFocus={mode === "login"} />
          <Field label="Password" value={password} onChange={setPassword}
            placeholder="••••••••" type="password" />

          {error && (
            <div style={{ color: "#e0533a", fontSize: 12.5, marginTop: 2 }}>{error}</div>
          )}

          <button type="submit" className="btn btn-primary" disabled={busy}
            style={{ marginTop: 8, justifyContent: "center", height: 38 }}>
            {busy ? "…" : mode === "redeem" ? "Create account" : "Log in"}
          </button>
        </form>

        <div style={{ fontSize: 11.5, color: "var(--text-faint, #6b7280)", marginTop: 14, lineHeight: 1.5 }}>
          {mode === "login"
            ? "First time here? Use “Sign up” with the license key from the developer."
            : "You only enter the key once - after that, log in with your username and password."}
        </div>
      </div>
    </div>
  );
}

function Field({ label, value, onChange, placeholder, type = "text", autoFocus, mono }: {
  label: string; value: string; onChange: (v: string) => void;
  placeholder?: string; type?: string; autoFocus?: boolean; mono?: boolean;
}) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 11, color: "var(--text-dim, #9aa0ad)", textTransform: "uppercase",
        letterSpacing: ".05em" }}>{label}</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        type={type}
        autoFocus={autoFocus}
        autoComplete={type === "password" ? "current-password" : "off"}
        spellCheck={false}
        className={mono ? "mono" : undefined}
        style={{ background: "#0b0c10", border: "1px solid var(--border, #23262d)",
          borderRadius: 7, padding: "9px 11px", color: "inherit", fontSize: 14, outline: "none" }}
      />
    </label>
  );
}
