"""Native Astryx chat surface for the TWFarmBot assistant.

The surrounding dashboard remains Streamlit.  This component owns only the
assistant workspace so the chat can have one scroll container and no nested
Streamlit widget chrome.
"""

from __future__ import annotations

import json
from typing import Any

import streamlit.components.v1 as components


_BOOTSTRAP_MARKER = "__TWFARMBOT_BOOTSTRAP__"


_ASSISTANT_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TWFarmBot Assistant</title>
  <style>
    :root {
      --ink: #f4f6fb;
      --muted: #9aa1b1;
      --faint: #687082;
      --canvas: #0a0c11;
      --panel: #12151c;
      --panel-soft: #0f1218;
      --line: rgba(255, 255, 255, .09);
      --line-strong: rgba(255, 255, 255, .17);
      --blue: #225bff;
      --blue-bright: #628bff;
      --blue-wash: rgba(34, 91, 255, .15);
      --success: #72d7ac;
      --danger: #ff8585;
      --radius: 14px;
      --mono: "SFMono-Regular", "Cascadia Code", "Roboto Mono", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      --type-caption: 12px;
      --type-body: 14px;
      --type-heading: 20px;
      --type-title: 28px;
      --line-tight: 1.2;
      --line-body: 1.5;
    }

    *, *::before, *::after { box-sizing: border-box; }
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      overflow: hidden;
      background: var(--canvas);
    }
    body {
      color: var(--ink);
      font-family: var(--sans);
      font-size: var(--type-body);
      line-height: var(--line-body);
    }
    button, input, textarea, select { font: inherit; }
    button { cursor: pointer; }
    button:focus-visible, textarea:focus-visible, select:focus-visible {
      outline: 2px solid var(--blue-bright);
      outline-offset: 2px;
    }

    .app {
      width: 100%;
      height: 100%;
      min-height: 0;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      background:
        radial-gradient(800px 360px at 72% -12%, rgba(34, 91, 255, .14), transparent 66%),
        var(--canvas);
    }
    .main {
      display: flex;
      flex: 1 1 auto;
      flex-direction: column;
      min-width: 0;
      min-height: 0;
      overflow: hidden;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex: 0 0 auto;
      min-height: 75px;
      padding: 17px 34px 14px;
      border-bottom: 1px solid var(--line);
    }
    .topbar-title {
      margin-top: 0;
      font-size: var(--type-title);
      line-height: var(--line-tight);
      font-weight: 700;
      letter-spacing: -.03em;
    }
    .topbar-subtitle {
      margin-top: 5px;
      color: var(--muted);
      font-size: var(--type-caption);
      line-height: 1.4;
    }
    .topbar-tools { display: flex; align-items: center; gap: 15px; }
    .model-select {
      max-width: 180px;
      padding: 7px 22px 7px 2px;
      border: 0;
      border-bottom: 1px solid var(--line-strong);
      border-radius: 0;
      color: var(--muted);
      background: transparent;
      font-size: var(--type-caption);
      line-height: 1.4;
      font-weight: 600;
    }
    .model-select:hover,
    .model-select:focus {
      color: var(--ink);
      border-bottom-color: var(--blue-bright);
      outline: 0;
    }
    .model-select:disabled { cursor: wait; opacity: .55; }
    .model-select option { color: var(--ink); background: var(--panel); }
    .clear-button {
      padding: 8px 11px;
      border: 1px solid var(--line-strong);
      border-radius: 8px;
      color: var(--muted);
      background: transparent;
      font-size: var(--type-caption);
      line-height: 1.4;
      font-weight: 600;
    }
    .clear-button:hover { color: var(--ink); border-color: var(--blue-bright); background: var(--blue-wash); }

    /* This is the only scrolling region in the assistant. */
    .conversation {
      flex: 1 1 auto;
      min-height: 0;
      overflow-y: auto;
      overscroll-behavior: contain;
      padding: 28px max(34px, calc((100% - 920px) / 2)) 26px;
      scrollbar-color: rgba(255,255,255,.16) transparent;
      scrollbar-width: thin;
    }
    .conversation::-webkit-scrollbar { width: 8px; }
    .conversation::-webkit-scrollbar-thumb { border-radius: 99px; background: rgba(255,255,255,.14); }

    .welcome { max-width: 620px; margin: 11vh auto 0; }
    .welcome h2 {
      margin: 0;
      font-size: var(--type-heading);
      line-height: 1.25;
      font-weight: 700;
      letter-spacing: -.015em;
    }
    .welcome-copy {
      max-width: 500px;
      margin: 13px 0 24px;
      color: var(--muted);
      font-size: var(--type-body);
      line-height: var(--line-body);
    }
    .suggestions {
      display: grid;
      border-top: 1px solid var(--line);
    }
    .suggestion {
      display: flex;
      align-items: center;
      min-height: 0;
      padding: 13px 2px;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      color: var(--muted);
      background: transparent;
      text-align: left;
      transition: color .14s ease, padding-left .14s ease;
    }
    .suggestion::after {
      content: "→";
      margin-left: auto;
      color: var(--faint);
    }
    .suggestion:hover {
      padding-left: 7px;
      color: var(--ink);
      background: transparent;
    }
    .suggestion-text {
      display: block;
      font-size: var(--type-body);
      line-height: var(--line-body);
      font-weight: 600;
    }

    .message-list { display: grid; gap: 20px; max-width: 920px; margin: 0 auto; }
    .message { display: flex; align-items: flex-start; gap: 17px; animation: rise .2s ease both; }
    @keyframes rise { from { opacity: 0; transform: translateY(5px); } to { opacity: 1; transform: translateY(0); } }
    .message-author {
      width: 62px;
      flex: 0 0 62px;
      padding-top: 12px;
      color: var(--faint);
      font-family: var(--sans);
      font-size: var(--type-caption);
      line-height: 1.4;
      font-weight: 600;
      letter-spacing: .06em;
      text-transform: uppercase;
    }
    .message-body { display: flex; min-width: 0; max-width: min(760px, calc(100% - 79px)); flex-direction: column; align-items: flex-start; }
    .bubble {
      padding: 13px 16px;
      border: 1px solid var(--line);
      border-radius: 4px 15px 15px 15px;
      background: var(--panel);
      font-size: var(--type-body);
      line-height: var(--line-body);
    }
    .bubble p { margin: 0 0 8px; }
    .bubble p:last-child { margin-bottom: 0; }
    .bubble strong { font-weight: 750; color: #fff; }
    .bubble code { padding: 2px 5px; border-radius: 5px; color: var(--blue-bright); background: var(--panel-soft); font: .9em var(--mono); }
    .bubble pre { overflow-x: auto; margin: 10px 0 0; padding: 11px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel-soft); color: #dce2ef; font: var(--type-caption)/1.5 var(--mono); }
    .bubble a { color: var(--blue-bright); }
    .user-message { justify-content: flex-end; }
    .user-message .message-author { order: 2; text-align: right; }
    .user-message .message-body { order: 1; align-items: flex-end; }
    .user-message .bubble {
      border-radius: 15px 4px 15px 15px;
      border-color: rgba(98,139,255,.46);
      background: linear-gradient(145deg, rgba(34,91,255,.27), rgba(18,21,28,.88));
    }
    .cursor { display: inline-block; width: 7px; height: 16px; margin-left: 3px; vertical-align: -3px; border-radius: 1px; background: var(--blue-bright); animation: blink 1s step-end infinite; }
    @keyframes blink { 50% { opacity: 0; } }

    .thinking {
      width: 100%;
      margin-top: 11px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel-soft);
    }
    .thinking summary { padding: 9px 11px; color: var(--muted); cursor: pointer; font: 600 var(--type-caption)/1.4 var(--sans); list-style: none; }
    .thinking summary::-webkit-details-marker { display: none; }
    .thinking summary::before { content: ">"; display: inline-block; width: 15px; color: var(--blue-bright); }
    .thinking[open] summary::before { content: "v"; }
    .thinking-copy { padding: 0 11px 11px 26px; color: var(--faint); font-size: var(--type-caption); line-height: 1.5; white-space: pre-wrap; }

    .tool-stack { display: grid; width: 100%; gap: 7px; margin-top: 12px; }
    .tool-card { border: 1px solid rgba(98,139,255,.25); border-radius: 10px; background: rgba(15,18,24,.88); }
    .tool-summary { display: flex; align-items: center; gap: 10px; padding: 9px 11px; color: var(--blue-bright); font: 600 var(--type-caption)/1.4 var(--sans); list-style: none; }
    .tool-summary::-webkit-details-marker { display: none; }
    .tool-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .tool-args { overflow: hidden; color: var(--faint); text-overflow: ellipsis; white-space: nowrap; }
    .tool-details { padding: 0 11px 11px; }
    .tool-details pre { overflow: auto; max-height: 130px; margin: 0; padding: 9px; border-radius: 7px; color: var(--muted); background: var(--canvas); font: var(--type-caption)/1.45 var(--mono); white-space: pre-wrap; word-break: break-word; }
    .tool-image { display: block; width: 100%; max-height: 260px; margin-top: 9px; border-radius: 8px; object-fit: contain; background: var(--canvas); }

    .proposal {
      width: 100%;
      margin-top: 12px;
      padding: 14px;
      border: 1px solid rgba(98,139,255,.48);
      border-radius: 12px;
      background: linear-gradient(145deg, rgba(34,91,255,.11), rgba(18,21,28,.78));
    }
    .proposal-head { color: var(--blue-bright); font: 700 var(--type-caption)/1.4 var(--sans); letter-spacing: .08em; text-transform: uppercase; }
    .proposal-copy { margin: 8px 0 11px; color: var(--muted); font-size: var(--type-caption); line-height: 1.5; }
    .action-list { display: grid; gap: 6px; margin-bottom: 13px; }
    .action-line { display: flex; gap: 9px; align-items: baseline; color: var(--ink); font: var(--type-caption)/1.5 var(--sans); }
    .action-index { width: 19px; color: var(--blue-bright); }
    .action-params { overflow: hidden; color: var(--muted); text-overflow: ellipsis; white-space: nowrap; }
    .proposal-actions { display: flex; gap: 8px; }
    .btn { padding: 9px 13px; border: 1px solid var(--line-strong); border-radius: 8px; color: var(--ink); background: transparent; font-size: var(--type-caption); line-height: 1.4; font-weight: 600; }
    .btn:hover { border-color: var(--blue-bright); background: var(--blue-wash); }
    .btn-approve { border-color: var(--blue); color: #fff; background: var(--blue); }
    .btn-approve:hover { border-color: var(--blue-bright); background: var(--blue-bright); }
    .btn:disabled { cursor: wait; opacity: .55; }
    .result-note { margin-top: 10px; color: var(--muted); font: var(--type-caption)/1.4 var(--sans); }
    .result-note.ok { color: var(--success); }
    .result-note.bad { color: var(--danger); }
    .metric-line { margin-top: 9px; color: var(--faint); font: var(--type-caption)/1.4 var(--sans); }

    .composer-wrap { flex: 0 0 auto; padding: 0 max(34px, calc((100% - 920px) / 2)) 18px; }
    .composer {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: end;
      gap: 10px;
      padding: 8px 9px 8px 12px;
      border: 1px solid var(--line-strong);
      border-radius: 14px;
      background: rgba(18,21,28,.96);
      box-shadow: 0 12px 38px rgba(0,0,0,.24);
      transition: border-color .16s ease, box-shadow .16s ease;
    }
    .composer:focus-within { border-color: rgba(98,139,255,.72); box-shadow: 0 0 0 3px rgba(34,91,255,.1), 0 12px 38px rgba(0,0,0,.24); }
    .composer textarea { width: 100%; min-height: 22px; max-height: 125px; resize: none; padding: 8px 0; border: 0; outline: 0; color: var(--ink); background: transparent; font-size: var(--type-body); line-height: var(--line-body); }
    .composer textarea::placeholder { color: var(--faint); }
    .send {
      display: grid;
      place-items: center;
      width: 35px;
      height: 35px;
      padding: 0;
      border: 0;
      border-radius: 9px;
      color: #fff;
      background: var(--blue);
      font-size: 17px;
      transition: transform .16s ease, background .16s ease;
    }
    .send:hover { background: var(--blue-bright); transform: translateY(-1px); }
    .send:disabled { cursor: wait; opacity: .5; transform: none; }

    .toast { position: fixed; right: 22px; bottom: 88px; z-index: 4; max-width: 340px; padding: 11px 13px; border: 1px solid var(--line-strong); border-radius: 9px; color: var(--ink); background: #1a1e28; box-shadow: 0 10px 30px rgba(0,0,0,.35); font-size: var(--type-caption); line-height: 1.4; }
    .toast.error { border-color: rgba(255,133,133,.4); color: #ffabab; }
    .hidden { display: none !important; }

    @media (max-width: 850px) {
      .topbar, .conversation, .composer-wrap { padding-left: 20px; padding-right: 20px; }
      .message { gap: 11px; }
      .message-author { width: 49px; flex-basis: 49px; }
      .message-body { max-width: calc(100% - 60px); }
    }
    @media (max-width: 560px) {
      .topbar { min-height: 65px; }
      .topbar-subtitle { display: none; }
      .model-select { max-width: 130px; }
      .welcome { margin-top: 5vh; }
      .welcome h2 { font-size: var(--type-heading); }
      .conversation { padding-top: 18px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <main class="main">
      <header class="topbar">
        <div>
          <div class="topbar-title">Assistant</div>
          <div class="topbar-subtitle">Ask about the garden or prepare an action.</div>
        </div>
        <div class="topbar-tools">
          <select id="model" class="model-select" aria-label="Assistant model">
            <option>Loading models…</option>
          </select>
          <button id="top-clear" class="clear-button" type="button">Clear</button>
        </div>
      </header>

      <section id="conversation" class="conversation" aria-live="polite"></section>

      <div class="composer-wrap">
        <form id="composer" class="composer">
          <textarea id="prompt" rows="1" placeholder="Ask about your garden…" aria-label="Message"></textarea>
          <button id="send" class="send" type="submit" aria-label="Send message">↑</button>
        </form>
      </div>
    </main>
  </div>
  <div id="toast" class="toast hidden" role="status"></div>

  <script>
    const BOOT = __TWFARMBOT_BOOTSTRAP__;
    const storageKey = `twfarmbot:assistant:v3:${BOOT.apiUrl}`;
    const modelKey = `${storageKey}:model`;
    const state = {
      messages: [],
      model: BOOT.initialModel || "",
      models: [],
      busy: false,
    };
    const $ = (id) => document.getElementById(id);
    const conversation = $("conversation");
    const prompt = $("prompt");
    const send = $("send");
    const modelSelect = $("model");
    const toast = $("toast");

    function apiBase() {
      const configured = String(BOOT.apiUrl || "").replace(/\/$/, "");
      try {
        const url = new URL(configured);
        const parentUrl = new URL(document.referrer || window.parent.location.href);
        if ((url.hostname === "127.0.0.1" || url.hostname === "localhost") &&
            parentUrl.hostname !== "127.0.0.1" && parentUrl.hostname !== "localhost") {
          url.hostname = parentUrl.hostname;
        }
        return url.toString().replace(/\/$/, "");
      } catch (_error) {
        return configured;
      }
    }


    function showToast(message, isError = false) {
      toast.textContent = message;
      toast.className = `toast${isError ? " error" : ""}`;
      window.clearTimeout(showToast.timer);
      showToast.timer = window.setTimeout(() => toast.className = "toast hidden", 4200);
    }

    function escapeHtml(value) {
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      })[char]);
    }

    function markdown(value) {
      let text = escapeHtml(value || "");
      const codeBlocks = [];
      text = text.replace(/```(?:[a-zA-Z0-9_-]+)?\n?([\s\S]*?)```/g, (_match, code) => {
        codeBlocks.push(`<pre>${code.trim()}</pre>`);
        return `\u0000CODE${codeBlocks.length - 1}\u0000`;
      });
      text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
      text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      text = text.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
      text = text.split("\n").map((line) => line || "<br>").join("<br>");
      return text.replace(/\u0000CODE(\d+)\u0000/g, (_match, index) => codeBlocks[Number(index)]);
    }

    function prettyJson(value) {
      try { return JSON.stringify(value ?? {}, null, 2); }
      catch (_error) { return String(value ?? ""); }
    }

    function metricsLabel(metrics) {
      if (!metrics) return "";
      const items = [];
      if (metrics.total_latency_s != null) items.push(`total ${Number(metrics.total_latency_s).toFixed(2)}s`);
      if (metrics.ttft_s) items.push(`ttft ${Number(metrics.ttft_s).toFixed(2)}s`);
      if (metrics.tokens_per_s) items.push(`${Number(metrics.tokens_per_s).toFixed(1)} tok/s`);
      if (metrics.resireg_latency_s) items.push(`vision ${Number(metrics.resireg_latency_s).toFixed(2)}s`);
      return items.join(" · ");
    }

    function renderTool(tool) {
      const name = escapeHtml(tool.name || "tool");
      const args = tool.args && Object.keys(tool.args).length
        ? escapeHtml(Object.entries(tool.args).map(([key, value]) => `${key}=${typeof value === "object" ? JSON.stringify(value) : value}`).join(" · "))
        : "";
      const result = tool.result || {};
      const imageUrl = result.image_url || "";
      return `<details class="tool-card"><summary class="tool-summary"><span class="tool-label">${name}</span>${args ? `<span class="tool-args">${args}</span>` : ""}</summary><div class="tool-details"><pre>${escapeHtml(prettyJson(result))}</pre>${imageUrl ? `<img class="tool-image" src="${escapeHtml(imageUrl)}" alt="Tool result">` : ""}</div></details>`;
    }

    function renderProposal(message, index) {
      const actions = message.proposed_actions || [];
      if (!actions.length) return "";
      if (message.approved) {
        const failed = (message.execution_results || []).some((item) => item.status === "error");
        return `<div class="result-note ${failed ? "bad" : "ok"}">${failed ? "Action finished with errors" : "Actions approved and dispatched"}</div>`;
      }
      if (message.rejected) return `<div class="result-note">Proposal dismissed</div>`;
      if (message.executing) return `<div class="result-note">Dispatching approved actions…</div>`;
      const lines = actions.map((action, actionIndex) => {
        const params = Object.keys(action.params || {}).length ? prettyJson(action.params).replace(/[\n{}"]+/g, " ").trim() : "no parameters";
        return `<div class="action-line"><span class="action-index">${String(actionIndex + 1).padStart(2, "0")}</span><span>${escapeHtml(action.kind || "action")}</span><span class="action-params">${escapeHtml(params)}</span></div>`;
      }).join("");
      return `<div class="proposal"><div class="proposal-head">Approval required</div><div class="proposal-copy">The assistant prepared this sequence. Nothing will move until you approve it.</div><div class="action-list">${lines}</div><div class="proposal-actions"><button class="btn btn-approve" data-approve="${index}" type="button">Approve sequence</button><button class="btn" data-reject="${index}" type="button">Reject</button></div></div>`;
    }

    function renderMessageBody(message, index) {
      const thinking = message.thinking ? `<details class="thinking"><summary>Reasoning trace</summary><div class="thinking-copy">${escapeHtml(message.thinking)}</div></details>` : "";
      const tools = message.tool_calls?.length ? `<div class="tool-stack">${message.tool_calls.map(renderTool).join("")}</div>` : "";
      const waiting = message.streaming && !message.content ? '<span class="cursor"></span>' : "";
      const error = message.error ? `<span style="color:var(--danger)">${escapeHtml(message.error)}</span>` : "";
      const content = message.content ? markdown(message.content) : waiting || error;
      const proposal = message.role === "assistant" ? renderProposal(message, index) : "";
      const metrics = message.metrics ? `<div class="metric-line">${escapeHtml(metricsLabel(message.metrics))}</div>` : "";
      return `<div class="message-content">${content}</div>${thinking}${tools}${proposal}${metrics}`;
    }

    function bindMessageActions(root = conversation) {
      root.querySelectorAll("[data-approve]").forEach((button) => {
        button.addEventListener("click", () => approve(Number(button.dataset.approve)));
      });
      root.querySelectorAll("[data-reject]").forEach((button) => {
        button.addEventListener("click", () => reject(Number(button.dataset.reject)));
      });
    }

    function renderMessage(message, index) {
      const isUser = message.role === "user";
      const isAssistant = message.role === "assistant";
      if (!isUser && !isAssistant) return "";
      const author = isUser ? "You" : "Assistant";
      return `<article class="message ${isUser ? "user-message" : "assistant-message"}" data-message-index="${index}"><div class="message-author">${author}</div><div class="message-body"><div class="bubble">${renderMessageBody(message, index)}</div></div></article>`;
    }

    function patchMessage(index, contentOnly = false) {
      const message = state.messages[index];
      const article = conversation.querySelector(`[data-message-index="${index}"]`);
      if (!message || !article) return;
      const distanceFromBottom = conversation.scrollHeight - conversation.scrollTop - conversation.clientHeight;
      const followOutput = distanceFromBottom < 80;
      if (contentOnly) {
        const content = article.querySelector(".message-content");
        if (content) {
          const waiting = message.streaming && !message.content ? '<span class="cursor"></span>' : "";
          const error = message.error ? `<span style="color:var(--danger)">${escapeHtml(message.error)}</span>` : "";
          content.innerHTML = message.content ? markdown(message.content) : waiting || error;
        }
      } else {
        const bubble = article.querySelector(".bubble");
        if (bubble) {
          bubble.innerHTML = renderMessageBody(message, index);
          bindMessageActions(bubble);
        }
      }
      if (followOutput) {
        requestAnimationFrame(() => {
          conversation.scrollTop = conversation.scrollHeight;
        });
      }
    }

    function render() {
      if (!state.messages.length) {
        conversation.innerHTML = `<div class="welcome"><h2>What would you like to do?</h2><p class="welcome-copy">Ask about the garden, inspect robot state, analyze a camera image, or prepare a safe action.</p><div class="suggestions"><button class="suggestion" type="button" data-suggestion="Give me a quick status report on the garden."><span class="suggestion-text">Give me a quick status report on the garden</span></button><button class="suggestion" type="button" data-suggestion="Analyze the latest garden image for weeds."><span class="suggestion-text">Analyze the latest image for weeds</span></button><button class="suggestion" type="button" data-suggestion="What is the current position of the FarmBot?"><span class="suggestion-text">Show the current FarmBot position</span></button><button class="suggestion" type="button" data-suggestion="Water the tomato zone for 30 seconds, then go home."><span class="suggestion-text">Prepare a tomato-zone watering run</span></button></div></div>`;
      } else {
        conversation.innerHTML = `<div class="message-list">${state.messages.map(renderMessage).join("")}</div>`;
        bindMessageActions();
      }
      conversation.querySelectorAll("[data-suggestion]").forEach((button) => button.addEventListener("click", () => {
        prompt.value = button.dataset.suggestion;
        resizePrompt();
        prompt.focus();
      }));
      conversation.scrollTop = conversation.scrollHeight;
    }

    function save() {
      try { localStorage.setItem(storageKey, JSON.stringify(state.messages)); }
      catch (_error) { /* localStorage is optional */ }
    }

    function load() {
      try {
        const stored = JSON.parse(localStorage.getItem(storageKey) || "null");
        if (Array.isArray(stored)) state.messages = stored;
      } catch (_error) {
        state.messages = [];
      }
      if (!state.messages.length && Array.isArray(BOOT.initialMessages) && BOOT.initialMessages.length) state.messages = BOOT.initialMessages;
    }

    function resizePrompt() {
      prompt.style.height = "auto";
      prompt.style.height = `${Math.min(prompt.scrollHeight, 125)}px`;
    }

    async function getJson(path, options = {}) {
      const response = await fetch(`${apiBase()}${path}`, { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
      let body = {};
      try { body = await response.json(); } catch (_error) { /* non-JSON error */ }
      if (!response.ok) throw new Error(body.detail || body.error || `Request failed (${response.status})`);
      return body;
    }

    async function loadModels() {
      try {
        const providers = await getJson("/providers");
        const provider = providers.current || "local";
        const result = await getJson(`/models?provider=${encodeURIComponent(provider)}`);
        state.models = result.models || [];
        const remembered = localStorage.getItem(modelKey);
        if (remembered && state.models.includes(remembered)) {
          state.model = remembered;
        } else if (!state.model || !state.models.includes(state.model)) {
          state.model = result.current || state.models[0] || "";
        }
        modelSelect.innerHTML = state.models.map((model) => (
          `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`
        )).join("");
        modelSelect.value = state.model;
        modelSelect.disabled = false;
      } catch (error) {
        modelSelect.innerHTML = "<option>API unavailable</option>";
        modelSelect.disabled = true;
        showToast(`Could not load models: ${error.message}`, true);
      }
    }

    async function sendMessage(value) {
      const text = String(value || "").trim();
      if (!text || state.busy) return;
      state.busy = true;
      send.disabled = true;
      prompt.value = "";
      resizePrompt();
      const requestMessages = [...state.messages, { role: "user", content: text }];
      state.messages.push({ role: "user", content: text });
      const assistant = { role: "assistant", content: "", thinking: "", tool_calls: [], proposed_actions: [], streaming: true };
      state.messages.push(assistant);
      const assistantIndex = state.messages.length - 1;
      render();
      save();
      modelSelect.disabled = true;

      try {
        const response = await fetch(`${apiBase()}/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "Accept": "text/event-stream" },
          body: JSON.stringify({ messages: requestMessages, model: state.model || null, allow_actions: true }),
        });
        if (!response.ok) {
          let detail = `Request failed (${response.status})`;
          try { detail = (await response.json()).detail || detail; } catch (_error) { /* SSE error */ }
          throw new Error(detail);
        }
        if (!response.body) throw new Error("The API returned no stream");
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        const handleLine = (line) => {
          if (!line.startsWith("data: ")) return;
          const raw = line.slice(6).trim();
          if (!raw || raw === "[DONE]") return;
          try {
            const event = JSON.parse(raw);
            let contentOnly = false;
            if (event.type === "delta") {
              assistant.content += event.content || "";
              contentOnly = true;
            } else if (event.type === "thinking") {
              assistant.thinking += event.content || "";
            } else if (event.type === "tool_call") {
              assistant.tool_calls.push({ name: event.name, args: event.args, result: event.result });
            } else if (event.type === "meta") {
              assistant.proposed_actions = event.proposed_actions || [];
              assistant.metrics = event.metrics || {};
              assistant.tool_calls = event.tool_calls || assistant.tool_calls;
            } else if (event.type === "error") {
              assistant.error = event.error || "Assistant failed";
            }
            patchMessage(assistantIndex, contentOnly);
          } catch (error) {
            assistant.error = `Invalid assistant event: ${error.message}`;
            patchMessage(assistantIndex);
          }
        };
        while (true) {
          const chunk = await reader.read();
          if (chunk.done) break;
          buffer += decoder.decode(chunk.value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";
          lines.forEach(handleLine);
        }
        if (buffer) handleLine(buffer);
        assistant.streaming = false;
        if (!assistant.content && !assistant.error && !assistant.proposed_actions.length) assistant.content = "I could not produce a response. Try again with a more specific request.";
        patchMessage(assistantIndex);
      } catch (error) {
        assistant.streaming = false;
        assistant.error = error.message || "Assistant request failed";
        showToast(assistant.error, true);
      } finally {
        state.busy = false;
        send.disabled = false;
        save();
        modelSelect.disabled = !state.models.length;
        patchMessage(assistantIndex);
        prompt.focus();
      }
    }

    async function approve(index) {
      const message = state.messages[index];
      if (!message || !message.proposed_actions?.length || message.executing) return;
      message.executing = true;
      render();
      const results = [];
      for (const action of message.proposed_actions) {
        try {
          const result = await getJson("/actions?wait=true", { method: "POST", body: JSON.stringify({ kind: action.kind, params: action.params || {} }) });
          results.push({ kind: action.kind, status: result.status || "ok", result });
        } catch (error) {
          results.push({ kind: action.kind, status: "error", error: error.message });
        }
      }
      message.execution_results = results;
      message.approved = true;
      message.executing = false;
      save();
      render();
    }

    function reject(index) {
      const message = state.messages[index];
      if (!message) return;
      message.rejected = true;
      save();
      render();
    }

    function clearConversation() {
      state.messages = [];
      save();
      render();
      prompt.focus();
    }

    $("composer").addEventListener("submit", (event) => { event.preventDefault(); sendMessage(prompt.value); });
    prompt.addEventListener("input", resizePrompt);
    prompt.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); sendMessage(prompt.value); }
    });
    modelSelect.addEventListener("change", () => {
      state.model = modelSelect.value;
      localStorage.setItem(modelKey, state.model);
      showToast(`Using ${state.model}`);
    });
    $("top-clear").addEventListener("click", clearConversation);

    load();
    render();
    resizePrompt();
    loadModels();
  </script>
</body>
</html>
"""


def render_assistant_component(
    *,
    api_url: str,
    initial_messages: list[dict[str, Any]] | None = None,
    initial_model: str | None = None,
    height: int = 760,
) -> None:
    """Render the self-contained Astryx assistant surface."""
    bootstrap = {
        "apiUrl": api_url,
        "initialMessages": initial_messages or [],
        "initialModel": initial_model or "",
    }
    html = _ASSISTANT_HTML.replace(
        _BOOTSTRAP_MARKER,
        json.dumps(bootstrap, separators=(",", ":"), ensure_ascii=True),
    )
    components.html(html, height=height, scrolling=False)
