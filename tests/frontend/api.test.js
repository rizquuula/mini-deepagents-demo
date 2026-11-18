import { afterEach, describe, expect, test } from "bun:test";
import { streamMessage } from "../../src/frontend/js/api.js";

const realFetch = globalThis.fetch;

function serveSse(body) {
  globalThis.fetch = async () =>
    new Response(body, { headers: { "Content-Type": "text/event-stream", "X-Request-ID": "rid" } });
}

async function collect(stream) {
  const items = [];
  for await (const item of stream) items.push(item);
  return items;
}

afterEach(() => {
  globalThis.fetch = realFetch;
});

describe("streamMessage", () => {
  test("tool frames become tool items and the stream goes on to the reply", async () => {
    serveSse(
      'event: tool_call\ndata: {"v":1,"call_id":"c1","name":"calculate"}\n\n' +
        'event: tool_result\ndata: {"v":1,"call_id":"c1","ok":false}\n\n' +
        'event: delta\ndata: {"v":1,"text":"42"}\n\n' +
        'event: done\ndata: {"v":1,"reply_chars":2}\n\n',
    );

    expect(await collect(streamMessage(1, "hi"))).toEqual([
      { type: "tool_call", callId: "c1", name: "calculate" },
      { type: "tool_result", callId: "c1", ok: false },
      { type: "delta", text: "42" },
      { type: "done", replyChars: 2 },
    ]);
  });
});
