// A pure parser for the Server-Sent Events text format. No DOM, no fetch.

const DEFAULT_EVENT = "message";

function parseField(line) {
  const colon = line.indexOf(":");
  if (colon === -1) return { field: line, value: "" };
  const value = line.slice(colon + 1);
  return { field: line.slice(0, colon), value: value.startsWith(" ") ? value.slice(1) : value };
}

/**
 * Create an incremental SSE parser.
 * push(text) returns the events that the text completes; flush() returns a final frame
 * that has no trailing blank line.
 */
export function createSseParser() {
  let buffer = "";
  let eventName = DEFAULT_EVENT;
  let dataLines = [];

  function dispatch(events) {
    if (dataLines.length > 0) events.push({ event: eventName, data: dataLines.join("\n") });
    eventName = DEFAULT_EVENT;
    dataLines = [];
  }

  function handleLine(line, events) {
    if (line === "") {
      dispatch(events);
      return;
    }
    if (line.startsWith(":")) return;
    const { field, value } = parseField(line);
    if (field === "event") eventName = value;
    if (field === "data") dataLines.push(value);
  }

  // Take complete lines from the buffer. A "\r" at the very end waits for the next
  // chunk, because a "\n" can still follow it.
  function drainLines(events, isFinal) {
    for (;;) {
      const end = buffer.search(/[\r\n]/);
      if (end === -1) return;
      const isCr = buffer[end] === "\r";
      if (isCr && end === buffer.length - 1 && !isFinal) return;
      const terminatorLength = isCr && buffer[end + 1] === "\n" ? 2 : 1;
      const line = buffer.slice(0, end);
      buffer = buffer.slice(end + terminatorLength);
      handleLine(line, events);
    }
  }

  return {
    push(text) {
      const events = [];
      buffer += text;
      drainLines(events, false);
      return events;
    },
    flush() {
      const events = [];
      drainLines(events, true);
      if (buffer !== "") handleLine(buffer, events);
      buffer = "";
      dispatch(events);
      return events;
    },
  };
}

/** Read a ReadableStream<Uint8Array> and yield its parsed SSE events. */
export async function* readSseEvents(body) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = createSseParser();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      yield* parser.push(decoder.decode(value, { stream: true }));
    }
    yield* parser.push(decoder.decode());
    yield* parser.flush();
  } finally {
    reader.releaseLock();
  }
}
