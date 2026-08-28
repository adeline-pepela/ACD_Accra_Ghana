const messagesEl = document.getElementById("messages");
const formEl = document.getElementById("chat-form");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const clientSelect = document.getElementById("client-select");
const resetBtn = document.getElementById("reset-btn");

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addUserMessage(text) {
  const row = document.createElement("div");
  row.className = "msg user";
  row.innerHTML = `<div class="bubble"></div>`;
  row.querySelector(".bubble").textContent = text;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function addTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg assistant";
  row.id = "typing-indicator";
  row.innerHTML = `<div class="bubble typing">Delegating to specialist(s)...</div>`;
  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
}

function removeTypingIndicator() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function addAssistantMessage(result) {
  const row = document.createElement("div");

  if (result.blocked) {
    row.className = "msg blocked";
    row.innerHTML = `
      <div class="bubble">
        <strong>Guardrail blocked this request (${escapeHtml(result.stage)} stage)</strong><br>
        ${escapeHtml(result.reason || "")}
      </div>`;
    messagesEl.appendChild(row);
    scrollToBottom();
    return;
  }

  row.className = "msg assistant";
  const domains = result.domains_consulted || [];
  const chips = domains.map(d => `<span class="chip">${escapeHtml(d)}</span>`).join("");
  const traceId = "trace-" + Math.random().toString(36).slice(2);
  const traceLines = (result.delegation_log || []).map(escapeHtml).join("\n");

  row.innerHTML = `
    <div class="bubble">
      ${escapeHtml(result.response || "")}
      ${domains.length ? `<div class="meta-row">${chips}</div>` : ""}
      ${traceLines ? `<span class="trace-toggle" data-target="${traceId}">Show delegation trace</span>
      <div class="trace-log" id="${traceId}">${traceLines}</div>` : ""}
    </div>`;
  messagesEl.appendChild(row);

  const toggle = row.querySelector(".trace-toggle");
  if (toggle) {
    toggle.addEventListener("click", () => {
      const log = document.getElementById(traceId);
      const open = log.classList.toggle("open");
      toggle.textContent = open ? "Hide delegation trace" : "Show delegation trace";
    });
  }
  scrollToBottom();
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

async function sendMessage(text) {
  addUserMessage(text);
  inputEl.value = "";
  sendBtn.disabled = true;
  addTypingIndicator();

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, client_id: clientSelect.value }),
    });
    const result = await resp.json();
    removeTypingIndicator();
    if (!resp.ok) {
      addAssistantMessage({ blocked: true, stage: "request", reason: result.error || "Something went wrong." });
    } else {
      addAssistantMessage(result);
    }
  } catch (err) {
    removeTypingIndicator();
    addAssistantMessage({ blocked: true, stage: "network", reason: String(err) });
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

formEl.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = inputEl.value.trim();
  if (!text) return;
  sendMessage(text);
});

document.querySelectorAll(".quick-prompt").forEach((btn) => {
  btn.addEventListener("click", () => {
    sendMessage(btn.dataset.text);
  });
});

resetBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  messagesEl.innerHTML = `
    <div class="msg assistant">
      <div class="bubble">New conversation started -- memory has been cleared.</div>
    </div>`;
});
