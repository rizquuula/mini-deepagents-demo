// Boot: wire the store, the API, and the UI modules.
import * as api from "./api.js";
import { createStore } from "./state.js";
import { createComposer } from "./ui/composer.js";
import { createMessageView } from "./ui/messages.js";
import { createSessionList } from "./ui/sessions.js";

const NO_SESSIONS_TEXT = 'No chats yet. Click "New chat" or type a message to start a new chat.';
const NO_MESSAGES_TEXT = "No messages yet. Type a message to start.";
const CONNECTION_FAILED_TEXT = "The connection to the server failed.";

function errorMessage(error) {
  return error instanceof api.ApiError ? error.message : CONNECTION_FAILED_TEXT;
}

function boot() {
  const store = createStore({ sessions: [], activeId: null, isStreaming: false });
  let streamController = null;

  const messages = createMessageView({ containerEl: document.getElementById("messages") });
  const sessionList = createSessionList({
    listEl: document.getElementById("session-list"),
    newChatEl: document.getElementById("new-chat"),
    onOpen: (id) => openSession(id),
    onNewChat: () => startNewChat(),
  });
  const composer = createComposer({
    formEl: document.getElementById("composer"),
    inputEl: document.getElementById("input"),
    sendEl: document.getElementById("send"),
    onSend: (content) => sendMessage(content),
  });

  store.subscribe((state) => {
    sessionList.render(state.sessions, state.activeId);
    composer.setSendDisabled(state.isStreaming);
    messages.setBusy(state.isStreaming);
  });

  // Stop the current reply. The user caused this, so no error shows.
  function abortStream() {
    if (streamController === null) return;
    streamController.abort();
    streamController = null;
    store.set({ isStreaming: false });
  }

  function openSession(id) {
    abortStream();
    return loadSession(id);
  }

  async function loadSession(id) {
    store.set({ activeId: id });
    try {
      const history = await api.getMessages(id);
      if (store.get().activeId !== id) return;
      if (history.length === 0) {
        messages.showEmpty(NO_MESSAGES_TEXT);
        return;
      }
      messages.showMessages(history);
    } catch (error) {
      if (store.get().activeId !== id) return;
      messages.clear();
      messages.showError(errorMessage(error));
    }
  }

  async function createAndOpenSession() {
    const session = await api.createSession();
    store.set({ sessions: [session, ...store.get().sessions] });
    await loadSession(session.id);
    return session.id;
  }

  async function startNewChat() {
    abortStream();
    try {
      await createAndOpenSession();
      composer.focus();
    } catch (error) {
      messages.showError(errorMessage(error));
    }
  }

  async function streamReply(sessionId, content, controller) {
    const userEl = messages.appendMessage({ role: "user", content });
    const reply = messages.startReply();
    const removeBubbles = () => {
      userEl.remove();
      reply.el.remove();
    };
    try {
      for await (const item of api.streamMessage(sessionId, content, { signal: controller.signal })) {
        if (item.type === "tool_call") reply.showToolCall(item.callId, item.name);
        if (item.type === "tool_result") reply.showToolResult(item.callId, item.ok);
        if (item.type === "delta") reply.appendText(item.text);
        if (item.type === "done") composer.clear();
        if (item.type === "error") {
          removeBubbles();
          messages.showError(item.message);
        }
      }
    } catch (error) {
      removeBubbles();
      if (!controller.signal.aborted) messages.showError(errorMessage(error));
    }
  }

  async function sendMessage(content) {
    if (store.get().isStreaming || content.trim() === "") return;
    const controller = new AbortController();
    streamController = controller;
    store.set({ isStreaming: true });
    try {
      const sessionId = store.get().activeId ?? (await createAndOpenSession());
      if (!controller.signal.aborted) await streamReply(sessionId, content, controller);
    } catch (error) {
      if (!controller.signal.aborted) messages.showError(errorMessage(error));
    } finally {
      if (streamController === controller) {
        streamController = null;
        store.set({ isStreaming: false });
      }
      composer.focus();
    }
  }

  async function init() {
    try {
      store.set({ sessions: await api.listSessions() });
    } catch (error) {
      messages.showError(errorMessage(error));
      return;
    }
    const [newest] = store.get().sessions;
    if (newest === undefined) {
      messages.showEmpty(NO_SESSIONS_TEXT);
      return;
    }
    await loadSession(newest.id);
  }

  init();
}

boot();
