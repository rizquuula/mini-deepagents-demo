// The sidebar: the session list and the "New chat" button.

function sessionItem(session, isActive, onOpen) {
  const item = document.createElement("li");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "session-button";
  button.setAttribute("aria-current", String(isActive));
  const label = document.createElement("span");
  label.className = "session-label";
  label.textContent = `Chat #${session.id}`;
  const date = document.createElement("span");
  date.className = "session-date";
  date.textContent = session.created_at;
  button.append(label, date);
  button.addEventListener("click", () => onOpen(session.id));
  item.append(button);
  return item;
}

export function createSessionList({ listEl, newChatEl, onOpen, onNewChat }) {
  newChatEl.addEventListener("click", () => onNewChat());

  return {
    render(sessions, activeId) {
      listEl.replaceChildren(...sessions.map((s) => sessionItem(s, s.id === activeId, onOpen)));
    },
  };
}
