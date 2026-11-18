// REST client and the streaming call for the v1 API.
import { readSseEvents } from "./sse.js";

const API_BASE = "/api/v1";
const TOO_LONG_TEXT = "Message is too long for the 8K context window.";
const STREAM_BROKEN_TEXT = "The reply stream is not valid.";
const STREAM_ENDED_TEXT = "The reply stream ended before the reply finished.";

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function withRequestId(text, requestId) {
  return requestId ? `${text} (request ${requestId})` : text;
}

function errorText(status, data) {
  if (status === 413) return TOO_LONG_TEXT;
  if (data && typeof data.detail === "string") return data.detail;
  if (data && Array.isArray(data.detail)) return "The request is not valid.";
  return `Request failed (status ${status}).`;
}

async function toApiError(response) {
  const data = await response.json().catch(() => null);
  const requestId = response.headers.get("X-Request-ID");
  return new ApiError(response.status, withRequestId(errorText(response.status, data), requestId));
}

async function request(method, path, { body, signal } = {}) {
  const options = { method, headers: {}, signal };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  return fetch(`${API_BASE}${path}`, options);
}

async function requestJson(method, path) {
  const response = await request(method, path);
  if (!response.ok) throw await toApiError(response);
  return response.json();
}

export function listSessions() {
  return requestJson("GET", "/sessions");
}

export function createSession() {
  return requestJson("POST", "/sessions");
}

export function getMessages(sessionId) {
  return requestJson("GET", `/sessions/${encodeURIComponent(sessionId)}/messages`);
}

function parseData(raw) {
  try {
    const data = JSON.parse(raw);
    return data !== null && typeof data === "object" ? data : null;
  } catch {
    return null;
  }
}

// One reader per known event name. A reader returns null when the data is not valid.
const ITEM_READERS = {
  tool_call: (data) =>
    typeof data.call_id === "string" && typeof data.name === "string"
      ? { type: "tool_call", callId: data.call_id, name: data.name }
      : null,
  tool_result: (data) =>
    typeof data.call_id === "string" && typeof data.ok === "boolean"
      ? { type: "tool_result", callId: data.call_id, ok: data.ok }
      : null,
  delta: (data) => (typeof data.text === "string" ? { type: "delta", text: data.text } : null),
  done: (data) => (Number.isInteger(data.reply_chars) ? { type: "done", replyChars: data.reply_chars } : null),
  error: (data, requestId) => {
    const message = typeof data.message === "string" ? data.message : "The model request failed.";
    const eventRequestId = typeof data.request_id === "string" ? data.request_id : requestId;
    return { type: "error", message: withRequestId(message, eventRequestId) };
  },
};

// Map one SSE event to a stream item. Return null for an unknown event name.
function toStreamItem({ event, data: raw }, requestId) {
  if (!Object.hasOwn(ITEM_READERS, event)) return null;
  const brokenItem = { type: "error", message: withRequestId(STREAM_BROKEN_TEXT, requestId) };
  const data = parseData(raw);
  if (data === null) return brokenItem;
  return ITEM_READERS[event](data, requestId) ?? brokenItem;
}

function isFinalItem(item) {
  return item.type === "done" || item.type === "error";
}

/**
 * Send a message and yield the reply as items:
 * { type: "tool_call" | "tool_result" | "delta" | "done" | "error", ... }.
 * Throw ApiError when the server rejects the message before the stream starts.
 * The generator ends after the first "done" or "error" item.
 */
export async function* streamMessage(sessionId, content, { signal } = {}) {
  const path = `/sessions/${encodeURIComponent(sessionId)}/messages`;
  const response = await request("POST", path, { body: { content }, signal });
  if (!response.ok) throw await toApiError(response);
  const requestId = response.headers.get("X-Request-ID");
  if (!response.body) {
    yield { type: "error", message: withRequestId(STREAM_BROKEN_TEXT, requestId) };
    return;
  }
  for await (const event of readSseEvents(response.body)) {
    const item = toStreamItem(event, requestId);
    if (item === null) continue;
    yield item;
    if (isFinalItem(item)) return;
  }
  yield { type: "error", message: withRequestId(STREAM_ENDED_TEXT, requestId) };
}
