import { NextResponse } from "next/server";
import { cookies } from "next/headers";
import { readSession, SESSION_COOKIE } from "@/lib/session";

export const runtime = "nodejs";

export async function GET() {
  const token = (await cookies()).get(SESSION_COOKIE)?.value;
  const username = await readSession(token);
  if (!username) return NextResponse.json({ user: null }, { status: 401 });
  return NextResponse.json({ user: username });
}
