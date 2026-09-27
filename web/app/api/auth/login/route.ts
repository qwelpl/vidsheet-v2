import { NextResponse } from "next/server";
import { login } from "@/lib/auth";
import { createSessionToken, SESSION_COOKIE, sessionCookieOptions } from "@/lib/session";

export const runtime = "nodejs";

export async function POST(req: Request) {
  let body: { username?: string; password?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Bad request" }, { status: 400 });
  }
  const result = await login(body.username ?? "", body.password ?? "");
  if (!result.ok) return NextResponse.json({ error: result.error }, { status: 401 });

  const res = NextResponse.json({ username: result.username });
  res.cookies.set(SESSION_COOKIE, await createSessionToken(result.username), sessionCookieOptions);
  return res;
}
