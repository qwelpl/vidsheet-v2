// User store (Vercel Blob JSON), password hashing (Web Crypto PBKDF2) and
// license-key validation. Imported only from Node-runtime route handlers.
import "server-only";
import { get, put, BlobNotFoundError } from "@vercel/blob";

const DB_PATH = "auth/db.json";

function token(): string {
  const t = process.env.BLOB_READ_WRITE_TOKEN;
  if (!t) throw new Error("BLOB_READ_WRITE_TOKEN is not set");
  return t;
}

export interface StoredUser {
  salt: string;
  hash: string;
  keyUsed: string;
  createdAt: number;
}
interface DB {
  users: Record<string, StoredUser>;
  usedKeys: string[];
}

async function readDB(): Promise<DB> {
  try {
    const res = await get(DB_PATH, { access: "private", token: token() });
    if (!res) return { users: {}, usedKeys: [] };
    const data = (await new Response(res.stream).json()) as Partial<DB>;
    return { users: data.users ?? {}, usedKeys: data.usedKeys ?? [] };
  } catch (e) {
    if (e instanceof BlobNotFoundError) return { users: {}, usedKeys: [] };
    throw e;
  }
}

async function writeDB(db: DB): Promise<void> {
  await put(DB_PATH, JSON.stringify(db), {
    access: "private",
    token: token(),
    addRandomSuffix: false,
    allowOverwrite: true,
    contentType: "application/json",
    cacheControlMaxAge: 0,
  });
}

// --- password hashing (PBKDF2-SHA256, edge/node compatible) ---------------
function toHex(b: Uint8Array): string {
  return Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");
}
function fromHex(h: string): Uint8Array {
  const out = new Uint8Array(h.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(h.slice(i * 2, i * 2 + 2), 16);
  return out;
}
async function derive(password: string, salt: Uint8Array): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(password), "PBKDF2", false, ["deriveBits"]);
  const bits = await crypto.subtle.deriveBits(
    { name: "PBKDF2", salt: salt as unknown as BufferSource, iterations: 120000, hash: "SHA-256" },
    key, 256);
  return toHex(new Uint8Array(bits));
}
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let r = 0;
  for (let i = 0; i < a.length; i++) r |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return r === 0;
}

// --- license keys ---------------------------------------------------------
function validKeys(): Set<string> {
  return new Set(
    (process.env.LICENSE_KEYS ?? "")
      .split(",")
      .map((k) => k.trim().toUpperCase())
      .filter(Boolean));
}

const USERNAME_RE = /^[a-zA-Z0-9_.-]{3,32}$/;

export type AuthResult =
  | { ok: true; username: string }
  | { ok: false; error: string };

export async function redeemKey(
  rawKey: string, username: string, password: string): Promise<AuthResult> {
  const key = (rawKey || "").trim().toUpperCase();
  username = (username || "").trim();
  if (!USERNAME_RE.test(username))
    return { ok: false, error: "Username must be 3-32 chars: letters, numbers, . _ -" };
  if ((password || "").length < 8)
    return { ok: false, error: "Password must be at least 8 characters" };
  if (!validKeys().has(key))
    return { ok: false, error: "Invalid license key" };

  const db = await readDB();
  if (db.usedKeys.includes(key))
    return { ok: false, error: "This license key has already been used" };
  if (db.users[username.toLowerCase()])
    return { ok: false, error: "That username is taken" };

  const salt = crypto.getRandomValues(new Uint8Array(16));
  const hash = await derive(password, salt);
  db.users[username.toLowerCase()] = {
    salt: toHex(salt), hash, keyUsed: key, createdAt: Date.now(),
  };
  db.usedKeys.push(key);
  await writeDB(db);
  return { ok: true, username };
}

export async function login(username: string, password: string): Promise<AuthResult> {
  username = (username || "").trim();
  const db = await readDB();
  const user = db.users[username.toLowerCase()];
  if (!user) return { ok: false, error: "Wrong username or password" };
  const hash = await derive(password || "", fromHex(user.salt));
  if (!timingSafeEqual(hash, user.hash))
    return { ok: false, error: "Wrong username or password" };
  return { ok: true, username };
}
