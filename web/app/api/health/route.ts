import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Unauthenticated liveness endpoint for uptime monitors. The app root (/)
// redirects to /login behind the auth gate, which a monitor reads as a failure;
// this always returns a plain 200 so UptimeRobot can point here instead.
export async function GET() {
  return NextResponse.json({ status: "ok" });
}
