// Study Buddy Agent - static chat frontend
// Talks to POST /api/chat, no dependencies.

(function () {
  "use strict";

  var chatLog = document.getElementById("chat-log");
  var inputRow = document.getElementById("input-row");
  var messageInput = document.getElementById("message-input");
  var sendButton = document.getElementById("send-button");
  var scoreDisplay = document.getElementById("score-display");
  var mcToggle = document.getElementById("mc-toggle");

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

  function appendAgentMessage(text, hint, choices) {
    var row = document.createElement("div");
    row.className = "message agent";

    var wrap = document.createElement("div");
    wrap.className = "agent-wrap";

    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = formatReply(text);
    wrap.appendChild(bubble);

    if (hint) {
      wrap.appendChild(buildHintToggle(hint));
    }

    if (choices && choices.length) {
      wrap.appendChild(buildChoiceButtons(choices));
    }

    row.appendChild(wrap);
    chatLog.appendChild(row);
    scrollToBottom();
  }

  function buildChoiceButtons(choices) {
    var letters = "ABCD";
    var group = document.createElement("div");
    group.className = "choice-group";

    choices.forEach(function (choice, i) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "choice-btn";
      btn.textContent = letters[i] + ") " + choice;
      btn.addEventListener("click", function () {
        var buttons = group.querySelectorAll(".choice-btn");
        for (var j = 0; j < buttons.length; j++) {
          buttons[j].disabled = true;
        }
        sendMessage(choice);
      });
      group.appendChild(btn);
    });

    return group;
  }

  function buildHintToggle(hint) {
    var hintBtn = document.createElement("button");
    hintBtn.type = "button";
    hintBtn.className = "hint-toggle";
    hintBtn.textContent = "💡 HINT";

    var hintBox = document.createElement("div");
    hintBox.className = "hint-box hidden";
    hintBox.textContent = hint;

    hintBtn.addEventListener("click", function () {
      var revealing = hintBox.classList.contains("hidden");
      hintBox.classList.toggle("hidden");
      hintBtn.textContent = revealing ? "💡 HIDE HINT" : "💡 HINT";
      if (revealing) {
        scrollToBottom();
      }
    });

    var group = document.createElement("div");
    group.className = "hint-group";
    group.appendChild(hintBtn);
    group.appendChild(hintBox);
    return group;
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

  function sendMessage(displayText, apiText) {
    appendUserMessage(displayText);
    messageInput.value = "";
    setSending(true);
    showLoadingIndicator();

    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: apiText || displayText })
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
          appendAgentMessage(result.data.reply, result.data.hint, result.data.choices);
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
    var apiText = mcToggle && mcToggle.checked ? text + " (multiple choice with 4 options)" : text;
    sendMessage(text, apiText);
  });

  // Initial state.
  scrollToBottom();
  messageInput.focus();
})();
