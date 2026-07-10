import { ui } from "../api.js";
import { h, snack, page, section, card, toolbar, expander, emptyState, actionSummary } from "../ui.js";
import * as state from "../state.js";

export async function render(container) {
  const { root, body } = page("History");
  const r = await ui("/sessions");
  const sessions = (r.ok && r.body?.sessions) || [];

  if (!sessions.length) {
    body.append(emptyState("No saved sessions yet. Chat is saved automatically.", { iconName: "history", compact: true }));
    container.append(root);
    return;
  }

  const sessionList = h("div", { class: "stack" });
  body.append(
    section("Chat sessions", sessionList),
  );

  sessionList.replaceChildren(...sessions.map((sess) => {
    const updated = (sess.updated_at || "").slice(0, 19).replace("T", " ");
    return card(null, h("div", { class: "toolbar", style: "justify-content:space-between" },
      h("div", {},
        h("b", {}, sess.label || sess.session_id),
        h("p", { class: "caption" }, `Updated ${updated}` + (sess.preview ? ` · ${sess.preview}` : ""))),
      h("md-filled-tonal-button", {
        onClick: async () => {
          const res = await ui(`/sessions/${encodeURIComponent(sess.session_id)}`);
          if (!res.ok) { snack("Session not found", { error: true }); return; }
          state.loadSessionSnapshot(res.body);
          const params = new URLSearchParams(location.search);
          params.set("tab", "assistant");
          location.search = params.toString();
        },
      }, "Load")));
  }));

  const executed = state.store.session?.executed_plans || [];
  if (executed.length) {
    body.append(section("Executed plans", h("div", { class: "stack" },
      executed.slice().reverse().map((plan, index) => {
        const status = plan.status || "unknown";
        const icon = { ok: "✅", partial: "⚠️", failed: "❌" }[status] || "❓";
        const ts = (plan.queued_at || "").slice(0, 19).replace("T", " ");
        return card(`${icon} Plan ${index + 1}`,
          h("p", { class: "caption" }, plan.request || ""),
          h("p", { class: "caption" }, `${ts} · ${(plan.actions || []).length} action(s) · ${status}`),
          expander("Actions", ...(plan.actions || []).map((a) => h("p", { class: "caption" }, actionSummary(a)))),
          (plan.results || []).length ? expander("Results", ...plan.results.map((res) =>
            h("p", { class: "caption" },
              `${res.ok ? "✅" : "❌"} ${actionSummary({ kind: res.kind })}`
              + (res.detail ? ` — ${res.detail}` : "")))) : null);
      }))));
  }

  container.append(root);
}
