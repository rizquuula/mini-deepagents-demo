// The message area: history, the streamed reply, the empty state, and errors.

const PENDING_TEXT = "Thinking…";

function messageElement(kind, roleText, text) {
  const el = document.createElement("div");
  el.className = `message message-${kind}`;
  const role = document.createElement("span");
  role.className = "role";
  role.textContent = roleText;
  const body = document.createElement("span");
  body.textContent = text;
  el.append(role, body);
  return { el, body };
}

export function createMessageView({ containerEl }) {
  function scrollToEnd() {
    containerEl.scrollTop = containerEl.scrollHeight;
  }

  function removeEmptyState() {
    containerEl.querySelector(".empty-state")?.remove();
  }

  function append(kind, roleText, text) {
    removeEmptyState();
    const message = messageElement(kind, roleText, text);
    containerEl.append(message.el);
    scrollToEnd();
    return message;
  }

  function appendMessage({ role, content }) {
    const isUser = role === "user";
    return append(isUser ? "user" : "assistant", isUser ? "You" : "Assistant", content).el;
  }

  return {
    showMessages(messages) {
      containerEl.replaceChildren();
      messages.forEach(appendMessage);
    },
    showEmpty(text) {
      const el = document.createElement("p");
      el.className = "empty-state";
      el.textContent = text;
      containerEl.replaceChildren(el);
    },
    showError(text) {
      append("error", "Error", text).el.setAttribute("role", "alert");
    },
    clear() {
      containerEl.replaceChildren();
    },
    setBusy(isBusy) {
      containerEl.setAttribute("aria-busy", String(isBusy));
    },
    appendMessage,

    /** Add an assistant bubble with "Thinking…". The first appendText call replaces it. */
    startReply() {
      const { el, body } = append("assistant message-pending", "Assistant", PENDING_TEXT);
      let hasText = false;
      return {
        el,
        appendText(text) {
          if (!hasText) {
            hasText = true;
            el.classList.remove("message-pending");
            body.textContent = "";
          }
          body.textContent += text;
          scrollToEnd();
        },
      };
    },
  };
}
