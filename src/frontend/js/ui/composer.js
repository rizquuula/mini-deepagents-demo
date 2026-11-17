// The composer: the message textarea and the Send button.

export function createComposer({ formEl, inputEl, sendEl, onSend }) {
  formEl.addEventListener("submit", (event) => {
    event.preventDefault();
    onSend(inputEl.value);
  });

  inputEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      onSend(inputEl.value);
    }
  });

  return {
    setSendDisabled(isDisabled) {
      sendEl.disabled = isDisabled;
    },
    clear() {
      inputEl.value = "";
    },
    focus() {
      inputEl.focus();
    },
  };
}
