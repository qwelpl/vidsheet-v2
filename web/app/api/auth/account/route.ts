import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { changePassword, changeUsername } from "@/lib/auth";
import {
  readSession, createSessionToken, SESSION_COOKIE, sessionCookieOptions,
} from "@/lib/session";

export const runtime = "nodejs";

export async function POST(req: Request) {
  const current = await readSession((await cookies()).get(SESSION_COOKIE)?.value);
  if (!current) return NextResponse.json({ error: "Not signed in" }, { status: 401 });

  let body: { action?: string; currentPassword?: string; newPassword?: string; newUsername?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Bad request" }, { status: 400 });
  }

  if (body.action === "password") {
    const r = await changePassword(current, body.currentPassword ?? "", body.newPassword ?? "");
    if (!r.ok) return NextResponse.json({ error: r.error }, { status: 400 });
    return NextResponse.json({ ok: true });
  }

  if (body.action === "username") {
    const r = await changeUsername(current, body.currentPassword ?? "", body.newUsername ?? "");
    if (!r.ok) return NextResponse.json({ error: r.error }, { status: 400 });
    const res = NextResponse.json({ ok: true, username: r.username });
    res.cookies.set(SESSION_COOKIE, await createSessionToken(r.username), sessionCookieOptions);
    return res;
  }

  return NextResponse.json({ error: "Unknown action" }, { status: 400 });
}
