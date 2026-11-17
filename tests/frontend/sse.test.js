import { describe, expect, test } from "bun:test";
import { createSseParser } from "../../src/frontend/js/sse.js";

const DELTA_FRAME = 'event: delta\r\ndata: {"v":1,"text":"Hel"}\r\n\r\n';
const DELTA_EVENT = { event: "delta", data: '{"v":1,"text":"Hel"}' };

function pushAll(chunks) {
  const parser = createSseParser();
  const events = chunks.flatMap((chunk) => parser.push(chunk));
  return [...events, ...parser.flush()];
}

describe("createSseParser", () => {
  test("one complete frame gives one event with its name and data", () => {
    const parser = createSseParser();
    expect(parser.push('event: done\ndata: {"v":1,"reply_chars":13}\n\n')).toEqual([
      { event: "done", data: '{"v":1,"reply_chars":13}' },
    ]);
  });

  test.each([
    ["middle of the event name", [3]],
    ["middle of the data line", [20]],
    ["middle of a CRLF", [DELTA_FRAME.indexOf("\r\n") + 1]],
    ["between the two final line ends", [DELTA_FRAME.length - 2]],
    ["every character", Array.from({ length: DELTA_FRAME.length - 1 }, (_, i) => i + 1)],
  ])("a frame split at the %s gives the same event", (_name, splitPoints) => {
    const bounds = [0, ...splitPoints, DELTA_FRAME.length];
    const chunks = bounds.slice(1).map((end, i) => DELTA_FRAME.slice(bounds[i], end));
    expect(pushAll(chunks)).toEqual([DELTA_EVENT]);
  });

  test("multiple data lines join with a newline and comment lines are ignored", () => {
    expect(pushAll([": keep-alive\rdata: one\rdata:two\r: note\r\r"])).toEqual([
      { event: "message", data: "one\ntwo" },
    ]);
  });

  test("two frames in one push give two events in order", () => {
    const parser = createSseParser();
    expect(parser.push("event: delta\ndata: a\n\nevent: done\ndata: b\n\n")).toEqual([
      { event: "delta", data: "a" },
      { event: "done", data: "b" },
    ]);
  });

  test("flush returns a final frame with no trailing blank line", () => {
    const parser = createSseParser();
    expect(parser.push("event: done\ndata: last")).toEqual([]);
    expect(parser.flush()).toEqual([{ event: "done", data: "last" }]);
  });
});
