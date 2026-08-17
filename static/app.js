// Study Buddy Agent - static chat frontend
// Talks to POST /api/chat, no dependencies.

(function () {
  "use strict";

  var chatLog = document.getElementById("chat-log");
  var inputRow = document.getElementById("input-row");
  var messageInput = document.getElementById("message-input");
  var sendButton = document.getElementById("send-button");
  var scoreDisplay = document.getElementById("score-display");

  /**
   * Convert **bold** markdown and literal newlines into safe HTML.
   * Escapes everything else so user/agent text can't inject markup.
   */
  function formatReply(text) {
    var escaped = escapeHtml(String(text));
    // Bold: **text** -> <strong>text</strong>
    var withBold = escaped.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    // Newlines -> <br>
    var withBreaks = withBold.replace(/\n/g, "<br>");
    return withBreaks;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function scrollToBottom() {
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function appendUserMessage(text) {
    var row = document.createElement("div");
    row.className = "message user";
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    chatLog.appendChild(row);
    scrollToBottom();
  }

  function appendAgentMessage(text) {
    var row = document.createElement("div");
    row.className = "message agent";
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = formatReply(text);
    row.appendChild(bubble);
    chatLog.appendChild(row);
    scrollToBottom();
  }

  function appendErrorMessage(text) {
    var row = document.createElement("div");
    row.className = "message agent error";
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    row.appendChild(bubble);
    chatLog.appendChild(row);
    scrollToBottom();
  }

  function showLoadingIndicator() {
    var row = document.createElement("div");
    row.className = "message agent loading";
    row.id = "loading-indicator";
    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = '<span class="typing-dots"><span></span><span></span><span></span></span>';
    row.appendChild(bubble);
    chatLog.appendChild(row);
    scrollToBottom();
    return row;
  }

  function removeLoadingIndicator() {
    var el = document.getElementById("loading-indicator");
    if (el && el.parentNode) {
      el.parentNode.removeChild(el);
    }
  }

  function updateScore(score) {
    if (!score) {
      return;
    }
    var correct = typeof score.correct === "number" ? score.correct : 0;
    var incorrect = typeof score.incorrect === "number" ? score.incorrect : 0;
    var total = typeof score.total === "number" ? score.total : 0;
    scoreDisplay.textContent = "✓ " + correct + "  ✗ " + incorrect + "  Total: " + total;
  }

  function setSending(isSending) {
    sendButton.disabled = isSending;
    messageInput.disabled = isSending;
  }

  function sendMessage(text) {
    appendUserMessage(text);
    messageInput.value = "";
    setSending(true);
    showLoadingIndicator();

    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text })
    })
      .then(function (response) {
        return response
          .json()
          .catch(function () {
            // Response wasn't valid JSON.
            return {};
          })
          .then(function (data) {
            return { ok: response.ok, data: data };
          });
      })
      .then(function (result) {
        removeLoadingIndicator();
        if (result.ok && result.data && typeof result.data.reply === "string") {
          appendAgentMessage(result.data.reply);
          updateScore(result.data.score);
        } else {
          var errMsg =
            (result.data && result.data.error) || "Something went wrong.";
          appendErrorMessage(errMsg);
        }
      })
      .catch(function () {
        // Network failure or unexpected error.
        removeLoadingIndicator();
        appendErrorMessage("Something went wrong.");
      })
      .then(function () {
        setSending(false);
        messageInput.focus();
      });
  }

  inputRow.addEventListener("submit", function (event) {
    event.preventDefault();
    var text = messageInput.value.trim();
    if (!text) {
      return;
    }
    sendMessage(text);
  });

  // Initial state.
  scrollToBottom();
  messageInput.focus();
})();
