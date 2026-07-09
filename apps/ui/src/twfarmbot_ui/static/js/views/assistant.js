import { api, ui, sse, postAction, errorMessage } from "../api.js";
import { h, icon, snack, md, page, card, toolbar, expander, jsonBlock, actionSummary } from "../ui.js";
import * as state from "../state.js";

const CHAT_TIMEOUT_MS = 90000;
const APPROVAL_WORDS = new Set(["yes", "y", "approve", "approved", "ok", "okay", "sure",
  "go ahead", "do it", "confirm", "confirmed", "execute", "run it"]);
const REJECTION_WORDS = new Set(["no", "n", "reject", "rejected", "cancel", "cancelled",
  "don't", "dont", "stop", "abort"]);

const normalize = (text) => text.replace(/[!.? ]+$/g, "").trim().toLowerCase();
const isApproval = (text) => APPROVAL_WORDS.has(normalize(text));
const isRejection = (text) => REJECTION_WORDS.has(normalize(text));

function metricsFooter(metrics) {
  if (!metrics) return "";
  const parts = [];
  if (metrics.total_latency_s != null) parts.push(`total ${metrics.total_latency_s.toFixed(2)}s`);
  if (metrics.ttft_s) parts.push(`ttft ${metrics.ttft_s.toFixed(2)}s`);
  if (metrics.tokens_per_s) parts.push(`${metrics.tokens_per_s.toFixed(1)} tok/s`);
  if (metrics.total_tokens) {
    parts.push(`tokens ${metrics.prompt_tokens || 0}+${metrics.completion_tokens || 0}=${metrics.total_tokens}`);
  }
  if (metrics.resireg_latency_s) parts.push(`resireg ${metrics.resireg_latency_s.toFixed(2)}s`);
  return parts.join(" · ");
}

function toolCallLabel(name, args = {}) {
  let label = name;
  if (name === "analyze_image" && args.prompt) label += ` · ${args.prompt}`;
  else if (name === "segment_image" && args.classes) label += ` · ${args.classes}`;
  else if (name === "visualize_image_features") label += ` · ${args.n_clusters ?? 6} clusters`;
  else if (name === "estimate_traversability" && args.prompt) label += ` · ${args.prompt}`;
  return label;
}

function toolCallEl(name, args, result) {
  const parts = [expander(toolCallLabel(name, args || {}), jsonBlock({ args, result }))];
  if (result && typeof result === "object") {
    const urls = result.image_urls || (result.image_url ? [result.image_url] : []);
    if (urls.length) parts.push(h("div", { class: "chat-images" }, urls.map((src) => h("img", { src, alt: "" }))));
  }
  return h("div", {}, ...parts);
}

async function fetchLatestImage() {
  const r = await api("/images", { params: { limit: "1", refresh: "true" }, timeoutMs: 15000 });
  return ((r.ok && r.body?.images) || [])[0] || null;
}

function imageIsNewer(image, previous) {
  if (!previous) return true;
  if (image.id != null && previous.id != null) return image.id !== previous.id;
  return (image.created_at || "") > (previous.created_at || "");
}

async function waitForNewImage(previous, { attempts = 15, delayMs = 2000 } = {}) {
  for (let i = 0; i < attempts; i++) {
    await new Promise((resolve) => setTimeout(resolve, delayMs));
    const image = await fetchLatestImage();
    if (image && imageIsNewer(image, previous)) return image;
  }
  return null;
}

async function executeProposedActions(actions, message) {
  const willCapture = message && actions.some((a) => a.kind === "take_photo");
  const previousImage = willCapture ? await fetchLatestImage() : null;
  const results = [];
  for (const action of actions) {
    const r = await postAction(action.kind, action.params || {}, { wait: true });
    results.push({ kind: action.kind, ok: r.ok, detail: typeof r.body === "string" ? r.body : r.body?.detail });
  }
  if (willCapture) {
    const image = await waitForNewImage(previousImage);
    if (image) (message.images = message.images || []).push({ attachment_url: image.attachment_url });
  }
  return results;
}

function formatExecutionResults(results) {
  if (!results.length) return "✅ Approved (no actions).";
  return results.map((res) => res.ok
    ? `✅ ${actionSummary({ kind: res.kind })}`
    : `❌ ${actionSummary({ kind: res.kind })} — ${res.detail || "error"}`).join("\n");
}

export async function render(container) {
  const session = state.store.session;
  const chatScroll = h("div", { class: "chat-scroll" });
  const metricsBar = h("div", { class: "assistant-metrics" });
  const sessionBox = h("div");
  const pickerBox = h("div", { class: "model-picker" });
  let busy = false;

  const clearBtn = h("md-outlined-button", {
    onClick: async () => {
      session.assistant_messages = [];
      await state.persistSession();
      drawMessages();
    },
  }, icon("mop"), "Clear chat");

  const { root, body } = page("Assistant", "TWFarmBot · UAS Technikum Wien", { actions: clearBtn });
  body.append(
    sessionBox,
    pickerBox,
    h("div", { class: "chat-panel" }, chatScroll),
  );
  container.append(root);

  // Fixed chat input bar — attached to body so it spans the full main area.
  const input = h("md-outlined-text-field", { label: "Message", placeholder: "Ask the FarmBot assistant…" });
  const sendBtn = h("md-filled-button", {}, icon("send"), "Send");
  const chatBar = h("div", { class: "chat-input-bar" },
    h("div", { class: "chat-input-inner" },
      metricsBar,
      h("div", { class: "chat-input-row" }, input, sendBtn)));
  document.body.append(chatBar);

  const send = async () => {
    const prompt = input.value.trim();
    if (!prompt) return;
    input.value = "";
    await sendPrompt(prompt);
  };
  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });

  function messageEl(message) {
    if (message.role === "tool") {
      return h("div", { class: "msg" },
        h("md-icon", { class: "avatar" }, "build"),
        h("div", { class: "bubble" }, toolCallEl(message.name, message.args, message.result)));
    }
    if (message.role === "user") {
      return h("div", { class: "msg user" },
        h("md-icon", { class: "avatar" }, "person"),
        h("div", { class: "bubble", html: md(message.content) }));
    }
    const bubble = h("div", { class: "bubble" });
    if (message.thinking) bubble.append(expander("Thinking", h("div", { html: md(message.thinking) })));
    bubble.append(h("div", { html: md(message.content) }));
    if (message.images?.length) {
      bubble.append(h("div", { class: "chat-images" },
        message.images.map((img) => h("img", { src: img.attachment_url, alt: "" }))));
    }
    const proposals = message.proposed_actions || [];
    if (proposals.length && !message.approved && !message.rejected) {
      bubble.append(h("div", { class: "proposal-card" },
        h("p", { class: "caption", style: "margin:0 0 8px" }, "Proposed actions:"),
        ...proposals.map((a) => h("p", { class: "caption", style: "margin:0" }, actionSummary(a))),
        toolbar(
          h("md-filled-button", { onClick: () => resolveProposal(message, true) }, icon("check"), "Approve"),
          h("md-outlined-button", { onClick: () => resolveProposal(message, false) }, icon("close"), "Reject"))));
    } else if (message.approved) {
      bubble.append(h("p", { class: "caption" }, "Approved"));
    } else if (message.rejected) {
      bubble.append(h("p", { class: "caption" }, "Rejected"));
    }
    return h("div", { class: "msg" }, h("md-icon", { class: "avatar" }, "smart_toy"), bubble);
  }

  function drawMessages() {
    chatScroll.replaceChildren(...session.assistant_messages.map(messageEl));
    metricsBar.textContent = metricsFooter(session.assistant_metrics);
    chatScroll.lastElementChild?.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  async function resolveProposal(message, approved) {
    if (approved) {
      const results = await executeProposedActions(message.proposed_actions, message);
      message.approved = true;
      message.content += `\n\n${formatExecutionResults(results)}`;
    } else {
      message.rejected = true;
      message.content += "\n\n❌ Cancelled.";
    }
    await state.persistSession();
    drawMessages();
  }

  async function sendPrompt(prompt) {
    if (busy) return;
    const messages = session.assistant_messages;
    const last = messages[messages.length - 1];
    const pending = last?.role === "assistant" && (last.proposed_actions || []).length
      && !last.approved && !last.rejected;

    if (isApproval(prompt) || isRejection(prompt)) {
      if (pending) {
        messages.push({ role: "user", content: prompt });
        drawMessages();
        await resolveProposal(last, isApproval(prompt));
        return;
      }
      snack("No pending proposal to approve or reject.");
      return;
    }

    busy = true;
    messages.push({ role: "user", content: prompt });
    drawMessages();

    const liveBubble = h("div", { class: "bubble" });
    const liveMsg = h("div", { class: "msg" },
      h("md-icon", { class: "avatar" }, "smart_toy"),
      h("div", { class: "stack" },
        h("p", { class: "caption", style: "margin:0" }, "Thinking…"),
        liveBubble));
    chatScroll.append(liveMsg);

    let textDiv = null;
    let accumulated = "";
    let segment = "";
    const meta = { tool_calls: [], proposed_actions: [], metrics: {} };
    const thinkingParts = [];
    let streamError = null;
    const newSegment = () => { textDiv = null; segment = ""; };
    const appendText = (chunk) => {
      accumulated += chunk;
      segment += chunk;
      if (!textDiv) { textDiv = h("div"); liveBubble.append(textDiv); }
      textDiv.innerHTML = md(segment);
      liveMsg.scrollIntoView({ behavior: "smooth", block: "end" });
    };

    try {
      for await (const event of sse("/chat/stream",
        { messages, model: session.assistant_selected_model },
        { timeoutMs: CHAT_TIMEOUT_MS })) {
        if (event.type === "delta") appendText(event.content || "");
        else if (event.type === "thinking") {
          newSegment();
          thinkingParts.push(String(event.content || ""));
          liveBubble.append(expander("Thinking", h("div", { html: md(event.content) })));
        } else if (event.type === "tool_call") {
          newSegment();
          const { name, args, result } = event;
          messages.push({ role: "tool", name, args, result });
          if (name === "take_photo" && result?.status === "ok") {
            const image = await fetchLatestImage();
            if (image) result.image_url = image.attachment_url;
          }
          liveBubble.append(toolCallEl(name, args, result));
        } else if (event.type === "meta") {
          meta.tool_calls = event.tool_calls || [];
          meta.proposed_actions = event.proposed_actions || [];
          meta.metrics = event.metrics || {};
        } else if (event.type === "error") {
          streamError = event.error || "stream error";
        }
      }
    } catch (err) {
      streamError = `${err.name}: ${err.message}`;
    }

    if (!accumulated && !meta.tool_calls.length && !meta.proposed_actions.length) {
      const r = await api("/chat", {
        method: "POST",
        json: { messages, model: session.assistant_selected_model },
        timeoutMs: CHAT_TIMEOUT_MS,
      });
      if (r.ok && typeof r.body === "object") {
        accumulated = String(r.body.response || "");
        meta.tool_calls = r.body.tool_calls || [];
        meta.metrics = r.body.metrics || {};
        for (const tc of meta.tool_calls) {
          messages.push({ role: "tool", name: tc.name, args: tc.args, result: tc.result });
        }
        meta.proposed_actions = meta.tool_calls
          .filter((tc) => tc.result?.status === "proposed")
          .map((tc) => ({ kind: tc.result.kind ?? tc.name, params: tc.result.params ?? tc.args ?? {} }));
        thinkingParts.splice(0, thinkingParts.length, String(r.body.thinking || ""));
        streamError = null;
      } else if (!streamError) {
        streamError = `Fallback failed: ${errorMessage(r)}`;
      }
    }

    if (Object.keys(meta.metrics).length) session.assistant_metrics = meta.metrics;
    if (streamError) snack(`Assistant error: ${streamError}`, { error: true, ms: 8000 });
    if (accumulated || meta.tool_calls.length || meta.proposed_actions.length) {
      messages.push({
        role: "assistant", content: accumulated, thinking: thinkingParts.join(""),
        tool_calls: meta.tool_calls, proposed_actions: meta.proposed_actions,
        images: [], metrics: meta.metrics,
      });
    }
    busy = false;
    await state.persistSession();
    drawMessages();
    drawSessionBox();
  }

  async function drawPicker() {
    const provRes = await api("/providers");
    const providers = (provRes.ok && provRes.body?.providers) || ["openrouter", "local"];
    let provider = (provRes.ok && provRes.body?.current) || providers[0];
    const providerSelect = h("md-outlined-select", { label: "Provider" },
      providers.map((p) => h("md-select-option", { value: p, selected: p === provider },
        h("div", { slot: "headline" }, p))));
    const modelBox = h("div");
    pickerBox.replaceChildren(providerSelect, modelBox);

    async function loadModels() {
      const r = await api("/models", { params: { provider } });
      const models = (r.ok && r.body?.models) || [];
      let current = session.assistant_selected_model || (r.ok && r.body?.current) || null;
      if (models.length) {
        if (!models.includes(current)) {
          const preferred = ["openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-3.5-sonnet"];
          current = preferred.find((m) => models.includes(m)) || models[0];
        }
        session.assistant_selected_model = current;
        const modelSelect = h("md-outlined-select", { label: "Model", class: "grow" },
          models.map((m) => h("md-select-option", { value: m, selected: m === current },
            h("div", { slot: "headline" }, m))));
        modelSelect.addEventListener("change", () => { session.assistant_selected_model = modelSelect.value; });
        modelBox.replaceChildren(modelSelect);
      } else {
        modelBox.replaceChildren(h("md-outlined-text-field", {
          label: "Model", value: current || "", class: "grow",
          onInput: (e) => { session.assistant_selected_model = e.target.value || null; },
        }));
      }
    }
    providerSelect.addEventListener("change", () => { provider = providerSelect.value; loadModels(); });
    await loadModels();
  }

  async function drawSessionBox() {
    const labelField = h("md-outlined-text-field", {
      label: "Session label", value: session.label || "", class: "grow",
      onInput: (e) => { session.label = e.target.value.trim() || null; },
    });
    const listBox = h("div", { class: "stack" });
    sessionBox.replaceChildren(expander("Session",
      toolbar(labelField,
        h("md-outlined-button", {
          onClick: async () => { await state.persistSession(); state.newSession(); location.reload(); },
        }, icon("add"), "New"),
        h("md-outlined-button", {
          onClick: async () => { await state.persistSession(); snack("Session saved"); },
        }, icon("save"), "Save")),
      listBox));

    const r = await ui("/sessions");
    const sessions = ((r.ok && r.body?.sessions) || [])
      .filter((s) => s.session_id !== session.session_id).slice(0, 20);
    listBox.replaceChildren(...sessions.map((s) => h("div", { class: "session-list-item" },
      h("span", { class: "caption" }, (s.label || s.session_id) + (s.preview ? ` · ${s.preview}` : "")),
      h("md-text-button", {
        onClick: async () => {
          const res = await ui(`/sessions/${encodeURIComponent(s.session_id)}`);
          if (!res.ok) { snack("Session not found", { error: true }); return; }
          state.loadSessionSnapshot(res.body);
          location.reload();
        },
      }, "Load"),
      h("md-icon-button", {
        onClick: async () => {
          await ui(`/sessions/${encodeURIComponent(s.session_id)}`, { method: "DELETE" });
          drawSessionBox();
        },
      }, icon("delete")))));
  }

  drawMessages();
  drawSessionBox();
  drawPicker();

  return () => chatBar.remove();
}
