/* ============================================================
   EXL3 web console — frontend (vanilla JS, no build step)
   ============================================================ */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const state = {
  view: "dashboard",
  specs: [],
  models: [],
  jobs: [],
  legacy: [],
  system: null,
  drawerJob: null,
  chat: { messages: [], busy: false, ctrl: null },
};

/* ---------------- api & utils ---------------- */

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch (_) { /* non-json */ }
  if (!res.ok) throw new Error((body && body.error) || `${res.status} ${res.statusText}`);
  return body;
}
const post = (path, data) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(data),
});

function toast(msg, kind = "") {
  const t = document.createElement("div");
  t.className = `toast ${kind}`;
  t.textContent = msg;
  $("#toasts").appendChild(t);
  setTimeout(() => t.remove(), 5200);
}

const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function fmtBytes(n) {
  if (n == null) return "–";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n >= 100 ? n.toFixed(0) : n.toFixed(1)} ${u[i]}`;
}
function fmtDur(s) {
  if (s == null) return "";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h}h ${m}m` : m ? `${m}m ${s % 60}s` : `${s}s`;
}
const pill = (status) => `<span class="pill ${esc(status)}">${esc(status)}</span>`;

/* ---------------- spec-driven forms ---------------- */

function ensureDatalist() {
  if ($("#dl-paths")) return;
  const dl = document.createElement("datalist");
  dl.id = "dl-paths";
  document.body.appendChild(dl);
}
function refreshDatalist() {
  ensureDatalist();
  const dl = $("#dl-paths");
  dl.innerHTML = state.models.map((m) =>
    `<option value="${esc(m.path)}">${esc(m.name)}</option>`).join("");
}

function renderForm(container, spec, overrides = {}) {
  container.innerHTML = "";
  const inputs = {};
  for (const fld of spec.fields) {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const isWide = ["path", "str"].includes(fld.kind) && !fld.choices;
    if (isWide) wrap.classList.add("full");
    const val = overrides[fld.name] ?? fld.default ?? "";
    let input;
    if (fld.kind === "bool") {
      wrap.className = "field";
      wrap.innerHTML = `<label>&nbsp;</label>
        <label class="check-field"><input type="checkbox" ${fld.default ? "checked" : ""}>
        <span>${esc(fld.label)}</span></label>`;
      input = wrap.querySelector("input");
      if (fld.help) wrap.insertAdjacentHTML("beforeend", `<div class="help">${esc(fld.help)}</div>`);
    } else if (fld.kind === "choice") {
      input = document.createElement("select");
      input.innerHTML = (fld.choices || []).map((c) =>
        `<option ${String(c) === String(val) ? "selected" : ""}>${esc(c)}</option>`).join("");
      wrap.innerHTML = `<label>${esc(fld.label)}${fld.required ? ' <span class="req">*</span>' : ""}</label>`;
      wrap.appendChild(input);
    } else {
      input = document.createElement("input");
      input.type = (fld.kind === "int" || fld.kind === "float") ? "number" : "text";
      if (fld.kind === "int") input.step = "1";
      if (fld.kind === "float") input.step = "0.1";
      if (fld.kind === "path") { ensureDatalist(); input.setAttribute("list", "dl-paths"); }
      input.value = val;
      if (fld.placeholder) input.placeholder = fld.placeholder;
      wrap.innerHTML = `<label>${esc(fld.label)}${fld.required ? ' <span class="req">*</span>' : ""}</label>`;
      wrap.appendChild(input);
    }
    if (fld.kind !== "bool" && fld.help)
      wrap.insertAdjacentHTML("beforeend", `<div class="help">${esc(fld.help)}</div>`);
    inputs[fld.name] = { el: input, fld };
    container.appendChild(wrap);
  }
  return () => {
    const out = {};
    for (const [name, { el, fld }] of Object.entries(inputs))
      out[name] = fld.kind === "bool" ? el.checked : el.value;
    return out;
  };
}

const spec = (key) => state.specs.find((s) => s.key === key);
const specsInGroup = (g) => state.specs.filter((s) => s.group === g);

/* ---------------- jobs ---------------- */

async function startJob(specKey, params) {
  try {
    const { job } = await post("/api/jobs/start", { spec: specKey, params });
    toast(`Started: ${job.name}`, "ok");
    openDrawer(job.id);
    pollJobs();
  } catch (e) {
    toast(`Could not start: ${e.message}`, "err");
  }
}

function jobTable(jobs, { legacy = [] } = {}) {
  if (!jobs.length && !legacy.length) return `<div class="empty">No jobs yet.</div>`;
  const row = (j) => `
    <tr class="clickable" data-job="${esc(j.id)}">
      <td class="mono dim">${esc(j.started || "")}</td>
      <td><span class="pill kind ${esc(j.kind)}">${esc(j.kind)}</span></td>
      <td>${esc(j.name || "")}</td>
      <td>${pill(j.status)}</td>
      <td class="dim">${j.meta?.bits ? esc(j.meta.bits) + " bpw" : ""}${j.meta?.port ? "port " + esc(j.meta.port) : ""}</td>
      <td class="mono dim">${j.pid ? esc(j.pid) : ""}</td>
    </tr>`;
  const legacyRows = legacy.map((j) => `
    <tr data-logpath="${esc(j.log || "")}">
      <td class="mono dim">${esc(j.started || "")}</td>
      <td><span class="pill kind quant">quant</span></td>
      <td>${esc(j.name || "")}</td>
      <td>${pill(j.status)} <span class="pill legacy">legacy</span></td>
      <td class="dim">${j.meta?.bits ? esc(j.meta.bits) + " bpw" : ""}</td>
      <td class="mono dim">${j.meta?.elapsed_s ? fmtDur(j.meta.elapsed_s) : ""}</td>
    </tr>`);
  return `<table><thead><tr>
      <th>started</th><th>kind</th><th>name</th><th>status</th><th></th><th>pid / took</th>
    </tr></thead><tbody>${jobs.map(row).join("")}${legacyRows.join("")}</tbody></table>`;
}

function bindJobRows(container) {
  $$("tr[data-job]", container).forEach((tr) =>
    tr.addEventListener("click", () => openDrawer(tr.dataset.job)));
  $$("tr[data-logpath]", container).forEach((tr) =>
    tr.addEventListener("click", () => {
      if (tr.dataset.logpath) openStaticLog(tr.dataset.logpath);
    }));
}

async function pollJobs() {
  try {
    const data = await api("/api/jobs");
    state.jobs = data.jobs;
    state.legacy = data.legacy || [];
  } catch (_) { return; }
  const running = state.jobs.filter((j) => j.status === "running").length;
  $("#badge-jobs").textContent = running;
  $("#chip-jobs").textContent = running ? `${running} running` : "idle";
  renderJobsView();
  renderDashJobs();
  renderServeList();
  renderHistories();
}

/* ---------------- drawer ---------------- */

let drawerTimer = null;

function openDrawer(jobId) {
  state.drawerJob = jobId;
  $("#overlay").classList.add("on");
  refreshDrawer();
  clearInterval(drawerTimer);
  drawerTimer = setInterval(refreshDrawer, 2000);
}
function closeDrawer() {
  state.drawerJob = null;
  $("#overlay").classList.remove("on");
  clearInterval(drawerTimer);
}
async function refreshDrawer() {
  const id = state.drawerJob;
  if (!id) return;
  try {
    const data = await api(`/api/jobs/${id}/log?tail=150000`);
    const job = state.jobs.find((j) => j.id === id);
    $("#drawer-title").innerHTML = `${esc(job?.name || id)} ${pill(data.status)}`;
    $("#drawer-meta").innerHTML = job ? `
      <span>kind <span class="mono">${esc(job.kind)}</span></span>
      <span>pid <span class="mono">${esc(job.pid || "–")}</span></span>
      <span>started <span class="mono">${esc(job.started || "")}</span></span>
      ${job.ended ? `<span>ended <span class="mono">${esc(job.ended)}</span></span>` : ""}
      ${job.exit_code != null ? `<span>exit <span class="mono">${esc(job.exit_code)}</span></span>` : ""}
    ` : "";
    $("#drawer-status").textContent = data.status;
    $("#drawer-kill").disabled = data.status !== "running";
    $("#drawer-path").textContent = data.log_path || "";
    const box = $("#drawer-log");
    const stick = $("#drawer-autoscroll").checked &&
      box.scrollTop + box.clientHeight >= box.scrollHeight - 60;
    box.textContent = data.log || "(no output yet)";
    if (stick) box.scrollTop = box.scrollHeight;
  } catch (e) { /* transient */ }
}
async function openStaticLog(path) {
  // legacy jobs.jsonl records: show their convert.log in the drawer (read-only)
  $("#overlay").classList.add("on");
  clearInterval(drawerTimer);
  state.drawerJob = null;
  $("#drawer-title").textContent = "Legacy job log";
  $("#drawer-meta").innerHTML = "";
  $("#drawer-kill").disabled = true;
  $("#drawer-path").textContent = path;
  try {
    const r = await post("/api/readfile", { path });
    $("#drawer-log").textContent = (r.text || "(empty)").slice(-150000);
  } catch (e) {
    $("#drawer-log").textContent = `Cannot read log through the console: ${e.message}\nOpen it on the host: ${path}`;
  }
  const box = $("#drawer-log");
  box.scrollTop = box.scrollHeight;
}

$("#drawer-close").onclick = closeDrawer;
$("#overlay").addEventListener("click", (e) => { if (e.target.id === "overlay") closeDrawer(); });
$("#drawer-refresh").onclick = refreshDrawer;
$("#drawer-kill").onclick = async () => {
  if (!state.drawerJob) return;
  if (!confirm("Kill this job (SIGTERM, then SIGKILL)?")) return;
  try {
    await post(`/api/jobs/${state.drawerJob}/kill`, {});
    toast("Kill signal sent", "ok");
  } catch (e) { toast(e.message, "err"); }
  refreshDrawer();
  pollJobs();
};

/* ---------------- view switching ---------------- */

const VIEW_TITLES = {
  dashboard: ["Dashboard", "system health, active work"],
  jobs: ["Jobs", "everything the console has run"],
  system: ["System", "GPU, RAM, disk, versions"],
  quantize: ["Quantize", "HF model → EXL3 (convert.py pipeline)"],
  serve: ["Serve", "OpenAI-compatible inference server"],
  chat: ["Chat", "playground for a running server"],
  evaluate: ["Evaluate", "eval/*.py harnesses"],
  tools: ["Tools", "tools/*.py utilities"],
  models: ["Models", "test_models/ browser"],
};

function switchView(v) {
  state.view = v;
  $$(".view").forEach((el) => el.classList.toggle("on", el.id === `view-${v}`));
  $$(".nav-item").forEach((el) => el.classList.toggle("active", el.dataset.view === v));
  const [t, s] = VIEW_TITLES[v] || [v, ""];
  $("#tb-title").textContent = t;
  $("#tb-sub").textContent = s;
  location.hash = v;
  onEnter(v);
}
$$(".nav-item").forEach((el) =>
  el.addEventListener("click", () => switchView(el.dataset.view)));

function onEnter(v) {
  if (v === "models") renderModels();
  if (v === "jobs") renderJobsView();
  if (v === "dashboard") renderDashboard();
  if (v === "system") renderSystem();
  if (v === "serve") renderServeList();
}

/* ---------------- dashboard ---------------- */

function renderDashboard() {
  const s = state.system;
  $("#dash-stats").innerHTML = !s ? `<div class="empty">loading…</div>` : "";
  if (!s) return;
  const g = s.gpus[0];
  const hasGpuMem = g && g.mem_used != null && g.mem_total;
  const gpuPct = hasGpuMem ? Math.round(100 * g.mem_used / g.mem_total) : null;
  const ramPct = s.ram?.total ? Math.round(100 * (1 - s.ram.available / s.ram.total)) : 0;
  const gpuVal = hasGpuMem
    ? `${g.mem_used / 1024 < 10 ? (g.mem_used / 1024).toFixed(1) : Math.round(g.mem_used / 1024)} / ${(g.mem_total / 1024).toFixed(0)} GB`
    : "unified <small>shared with RAM</small>";
  $("#dash-stats").innerHTML = `
    <div class="card stat">
      <div class="lbl">GPU — ${esc(g?.name || "n/a")}</div>
      <div class="val">${gpuVal}</div>
      ${gpuPct != null ? `<div class="bar"><i class="${gpuPct > 90 ? "bad" : ""}" style="width:${gpuPct}%"></i></div>` : ""}
      <div class="hint">utilization ${g?.util ?? "–"}% · ${g?.temp ?? "–"}°C</div>
    </div>
    <div class="card stat">
      <div class="lbl">System RAM</div>
      <div class="val">${s.ram?.total ? Math.round(s.ram.total / 1024) + " GB" : "–"} <small>${ramPct}% used</small></div>
      <div class="bar"><i class="${ramPct > 90 ? "bad" : ramPct > 75 ? "warn" : ""}" style="width:${ramPct}%"></i></div>
      <div class="hint">${s.ram?.available ? Math.round(s.ram.available / 1024) + " GB available (unified)" : ""}</div>
    </div>
    <div class="card stat">
      <div class="lbl">Workspace disk</div>
      <div class="val mono" style="font-size: 17px;line-height:1.7">${esc((s.disk || "").split("\n").slice(1, 2).join("") || "–")}</div>
      <div class="hint">exllamav3 ${esc(s.exllamav3 || "?")} · python ${esc(s.python || "?")}</div>
    </div>`;
}

function renderDashJobs() {
  const running = state.jobs.filter((j) => j.status === "running");
  $("#dash-jobs").innerHTML = running.length
    ? jobTable(running)
    : `<div class="empty">Nothing running. Start a quantization or a server.</div>`;
  bindJobRows($("#dash-jobs"));
}

function renderDashServe() {
  const serves = state.jobs.filter((j) => j.kind === "serve" && j.status === "running");
  $("#dash-serve").innerHTML = serves.length
    ? serves.map((j) => `<div class="kv"><span class="k">${esc(j.name)}</span>
        <span class="v">port ${esc(j.meta?.port || "?")} · pid ${esc(j.pid || "?")}</span></div>`).join("")
    : `<div class="empty">No server started by the console. The Health probe on the Serve tab can check any endpoint.</div>`;
}

/* ---------------- system view ---------------- */

function renderSystem() {
  const s = state.system;
  if (!s) { $("#sys-cards").innerHTML = `<div class="empty">loading…</div>`; return; }
  const gpuRows = s.gpus.map((g, i) => {
    const memOk = g.mem_used != null && g.mem_total;
    return `
    <div class="kv"><span class="k">GPU ${i}</span><span class="v">${esc(g.name)}</span></div>
    <div class="kv"><span class="k">memory</span><span class="v">${memOk
      ? `${g.mem_used} / ${g.mem_total} MiB (${Math.round(100 * g.mem_used / g.mem_total)}%)`
      : "unified (see RAM)"}</span></div>
    <div class="kv"><span class="k">utilization</span><span class="v">${g.util ?? "–"}% · ${g.temp ?? "–"}°C</span></div>`;
  }).join("");
  $("#sys-cards").innerHTML = `
    <div class="card">
      <h3>GPU <span class="rt"><button class="btn sm" onclick="pollSystem()">↻</button></span></h3>
      ${gpuRows || '<div class="empty">nvidia-smi unavailable</div>'}
    </div>
    <div class="card">
      <h3>Memory & disk</h3>
      <div class="kv"><span class="k">RAM total</span><span class="v">${s.ram?.total ? Math.round(s.ram.total / 1024) + " GB" : "–"}</span></div>
      <div class="kv"><span class="k">RAM available</span><span class="v">${s.ram?.available ? Math.round(s.ram.available / 1024) + " GB" : "–"}</span></div>
      <div class="kv"><span class="k">load avg</span><span class="v">${s.load?.map((x) => x.toFixed(2)).join("  ") || "–"}</span></div>
      <pre class="logbox" style="margin-top:10px;min-height:0">${esc(s.disk || "")}</pre>
    </div>
    <div class="card">
      <h3>Runtime</h3>
      <div class="kv"><span class="k">exllamav3</span><span class="v">${esc(s.exllamav3 || "?")}</span></div>
      <div class="kv"><span class="k">python</span><span class="v">${esc(s.python)}</span></div>
      <div class="kv"><span class="k">repo</span><span class="v">${esc(s.repo)}</span></div>
    </div>`;
}

async function pollSystem() {
  try {
    state.system = await api("/api/system");
    const s = state.system;
    const g = s.gpus[0];
    if (g) {
      const memTxt = g.mem_used != null ? ` · ${(g.mem_used / 1024).toFixed(1)}G` : "";
      $("#chip-gpu").textContent = `${g.util ?? 0}%${memTxt}`;
      $("#chip-gpu-dot").className = "dot " + ((g.util ?? 0) > 80 ? "warn" : "ok");
    } else {
      $("#chip-gpu").textContent = "–";
      $("#chip-gpu-dot").className = "dot bad";
    }
    $("#chip-ram").textContent = s.ram?.total ? `${Math.round(s.ram.total / 1024)}G` : "–";
    $("#foot-version").textContent = s.exllamav3 || "";
    if (state.view === "dashboard") renderDashboard();
    if (state.view === "system") renderSystem();
  } catch (_) { /* console server restarting? */ }
}

/* ---------------- quantize view ---------------- */

let collectQuant, collectResume, collectAttach;

function initQuantize() {
  const s = spec("quant");
  if (!s) return;
  $("#quant-desc").textContent = s.description;
  collectQuant = renderForm($("#quant-form"), s);
  collectResume = renderForm($("#resume-form"), spec("quant_resume"));
  collectAttach = renderForm($("#attach-form"), spec("attach_draft"));

  $("#quant-start").onclick = () => startJob("quant", collectQuant());
  $("#quant-preflight").onclick = () =>
    startJob("quant_preflight", { model: collectQuant().model });
  $("#resume-btn").onclick = () => startJob("quant_resume", collectResume());
  $("#attach-btn").onclick = () => startJob("attach_draft", collectAttach());
}

function renderQuantHistory() {
  const quants = state.jobs.filter((j) => ["quant", "preflight"].includes(j.kind));
  const legacy = state.legacy;
  $("#quant-history").innerHTML = jobTable(quants, { legacy });
  bindJobRows($("#quant-history"));
}

/* ---------------- serve view ---------------- */

let collectServe;

function initServe() {
  const s = spec("serve");
  if (!s) return;
  $("#serve-desc").textContent = s.description;
  collectServe = renderForm($("#serve-form"), s);
  $("#serve-start").onclick = () => startJob("serve", collectServe());
  $("#health-btn").onclick = async () => {
    const url = $("#health-url").value.trim();
    $("#health-out").innerHTML = '<span class="muted">probing…</span>';
    try {
      const r = await api(`/api/serve/health?url=${encodeURIComponent(url)}`);
      $("#health-out").innerHTML = r.up
        ? `<span class="pill ok">up</span> <span class="mono small">busy=${esc(r.health.busy)} · ctx=${esc(r.health.context_length)} · prompt_tok=${esc(r.health.prompt_tokens_total)} · gen_tok=${esc(r.health.completion_tokens_total)}</span>`
        : `<span class="pill failed">down</span> <span class="muted small">${esc(r.error)}</span>`;
    } catch (e) { $("#health-out").innerHTML = `<span class="pill failed">error</span> ${esc(e.message)}`; }
  };
}

async function renderServeList() {
  const serves = state.jobs.filter((j) => j.kind === "serve");
  if (!serves.length) { $("#serve-list").innerHTML = `<div class="empty">No servers started from the console yet.</div>`; $("#dash-serve") && renderDashServe(); return; }
  const rows = [];
  for (const j of serves) {
    let health = "";
    if (j.status === "running") {
      try {
        const p = j.meta?.port || 8888;
        const r = await api(`/api/serve/health?url=${encodeURIComponent(`http://127.0.0.1:${p}`)}`);
        health = r.up
          ? `<span class="pill ok">up</span> <span class="small muted">busy=${esc(r.health.busy)} ctx=${esc(r.health.context_length)}</span>`
          : `<span class="pill failed">down</span>`;
      } catch (_) { health = `<span class="pill failed">?</span>`; }
    }
    rows.push(`<tr>
      <td class="mono dim">${esc(j.started)}</td>
      <td>${esc(j.name)}</td>
      <td>${pill(j.status)}</td>
      <td>${health}</td>
      <td><button class="btn sm" data-open="${esc(j.id)}">log</button>
          ${j.status === "running" ? `<button class="btn sm danger" data-kill="${esc(j.id)}">stop</button>` : ""}</td>
    </tr>`);
  }
  $("#serve-list").innerHTML = `<table><thead><tr><th>started</th><th>name</th><th>status</th><th>health</th><th></th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
  $$("[data-open]", $("#serve-list")).forEach((b) => b.onclick = () => openDrawer(b.dataset.open));
  $$("[data-kill]", $("#serve-list")).forEach((b) => b.onclick = async () => {
    if (!confirm("Stop this server?")) return;
    await post(`/api/jobs/${b.dataset.kill}/kill`, {}).catch((e) => toast(e.message, "err"));
    pollJobs();
  });
  renderDashServe();
}

/* ---------------- chat ---------------- */

function chatBubble(role, content, reasoning, toolCalls) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  let html = "";
  if (reasoning) html += `<div class="reasoning"></div>`;
  html += `<div class="content"></div>`;
  if (toolCalls?.length) html += `<div class="tools"></div>`;
  el.innerHTML = html;
  if (reasoning) $(".reasoning", el).textContent = reasoning;
  $(".content", el).textContent = content;
  if (toolCalls?.length)
    $(".tools", el).textContent = toolCalls.map((c) =>
      `⚙ ${c.function?.name}(${c.function?.arguments})`).join("\n");
  return el;
}

function renderChat() {
  const log = $("#chat-log");
  log.innerHTML = "";
  if (!state.chat.messages.length) {
    log.innerHTML = `<div class="empty">No messages yet — say hi to the model.</div>`;
    return;
  }
  for (const m of state.chat.messages)
    log.appendChild(chatBubble(m.role, m.content || "", m.reasoning || "", m.tool_calls));
  log.scrollTop = log.scrollHeight;
}

async function sendChat() {
  const c = state.chat;
  if (c.busy) return;
  const text = $("#chat-input").value.trim();
  if (!text) return;
  $("#chat-input").value = "";
  c.messages.push({ role: "user", content: text });
  renderChat();

  const payload = {
    model: "exl3",
    stream: true,
    messages: [
      ...($("#chat-system").value.trim() ? [{ role: "system", content: $("#chat-system").value.trim() }] : []),
      ...c.messages.map(({ role, content, reasoning, tool_calls }) => {
        const m = { role, content };
        if (tool_calls) m.tool_calls = tool_calls;
        return m;
      }),
    ],
    temperature: parseFloat($("#chat-temp").value || "0.6"),
    top_p: parseFloat($("#chat-topp").value || "0.95"),
    top_k: parseInt($("#chat-topk").value || "20", 10),
    max_tokens: parseInt($("#chat-maxtok").value || "1024", 10),
  };

  const base = $("#chat-base").value.trim().replace(/\/$/, "");
  $("#chat-target").textContent = base;
  c.busy = true;
  $("#chat-send").disabled = true;
  $("#chat-stop").disabled = false;
  c.ctrl = new AbortController();

  const asst = { role: "assistant", content: "", reasoning: "", tool_calls: [] };
  c.messages.push(asst);
  const log = $("#chat-log");
  log.innerHTML = "";
  for (const m of c.messages.slice(0, -1))
    log.appendChild(chatBubble(m.role, m.content || "", m.reasoning || "", m.tool_calls));
  const live = chatBubble("assistant", "", "", []);
  log.appendChild(live);
  const paint = () => {
    $(".reasoning", live)?.remove();
    if (asst.reasoning) live.insertAdjacentHTML("afterbegin", `<div class="reasoning"></div>`);
    if ($(".reasoning", live)) $(".reasoning", live).textContent = asst.reasoning;
    $(".content", live).textContent = asst.content;
    const tools = $(".tools", live);
    if (asst.tool_calls.length) {
      if (!tools) live.insertAdjacentHTML("beforeend", `<div class="tools"></div>`);
      $(".tools", live).textContent = asst.tool_calls.map((tc) =>
        `⚙ ${tc.function?.name}(${tc.function?.arguments})`).join("\n");
    }
    log.scrollTop = log.scrollHeight;
  };

  try {
    const res = await fetch("/api/chat", {
      method: "POST", signal: c.ctrl.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_url: base, payload }),
    });
    if (!res.ok || !res.body) {
      let msg = `HTTP ${res.status}`;
      try { msg = (await res.json()).error?.message || msg; } catch (_) { }
      throw new Error(msg);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let obj;
        try { obj = JSON.parse(data); } catch (_) { continue; }
        if (obj.error) throw new Error(obj.error.message || "server error");
        const d = obj.choices?.[0]?.delta || {};
        if (d.reasoning_content) asst.reasoning += d.reasoning_content;
        if (d.content) asst.content += d.content;
        if (d.tool_calls) for (const tc of d.tool_calls) {
          const i = tc.index ?? asst.tool_calls.length;
          asst.tool_calls[i] = tc;
        }
        paint();
      }
    }
  } catch (e) {
    if (e.name === "AbortError") { asst.content += "\n[stopped]"; paint(); }
    else {
      c.messages.pop();
      const eb = document.createElement("div");
      eb.className = "msg error";
      eb.textContent = `⚠ ${e.message}`;
      log.appendChild(eb);
      log.scrollTop = log.scrollHeight;
    }
  } finally {
    c.busy = false;
    $("#chat-send").disabled = false;
    $("#chat-stop").disabled = true;
    c.ctrl = null;
    renderChat();
  }
}

function initChat() {
  $("#chat-send").onclick = sendChat;
  $("#chat-stop").onclick = () => state.chat.ctrl?.abort();
  $("#chat-clear").onclick = () => { state.chat.messages = []; renderChat(); };
  $("#chat-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendChat(); }
  });
}

/* ---------------- evaluate & tools ---------------- */

function initSpecRunner(prefix, group) {
  const sel = $(`#${prefix}-select`);
  const list = specsInGroup(group);
  if (!list.length) {
    sel.innerHTML = `<option value="">(none registered)</option>`;
    return;
  }
  sel.innerHTML = list.map((s) => `<option value="${esc(s.key)}">${esc(s.label)}</option>`).join("");
  let collect = null;
  const paint = () => {
    const s = spec(sel.value);
    $(`#${prefix}-desc`).textContent = s?.description || "";
    collect = s ? renderForm($(`#${prefix}-form`), s) : null;
  };
  sel.onchange = paint;
  paint();
  $(`#${prefix}-run`).onclick = () => {
    if (!collect) return;
    startJob(sel.value, collect());
  };
}

function renderHistories() {
  if ($("#eval-history")) {
    $("#eval-history").innerHTML = jobTable(state.jobs.filter((j) => j.kind === "eval"));
    bindJobRows($("#eval-history"));
  }
  if ($("#tool-history")) {
    $("#tool-history").innerHTML = jobTable(state.jobs.filter((j) => j.kind === "tool"));
    bindJobRows($("#tool-history"));
  }
  if ($("#quant-history")) renderQuantHistory();
}

/* ---------------- models ---------------- */

async function renderModels() {
  if (!state.models.length) {
    try { state.models = (await api("/api/models")).models; refreshDatalist(); }
    catch (_) { }
  }
  const rows = state.models.map((m) => `
    <tr>
      <td><span class="pill kind ${esc(m.kind)}">${esc(m.kind)}</span></td>
      <td>${esc(m.name)} ${m.has_draft ? '<span class="pill kind" title="draft bundled">draft</span>' : ""}</td>
      <td class="mono dim">${esc(m.path)}</td>
      <td class="dim">${fmtBytes(m.size)}</td>
      <td><button class="btn sm ghost" data-copy="${esc(m.path)}">copy path</button>
          <button class="btn sm danger" data-del="${esc(m.path)}" data-name="${esc(m.name)}" data-size="${m.size}">delete</button></td>
    </tr>`);
  $("#models-table").innerHTML = state.models.length
    ? `<table><thead><tr><th>kind</th><th>name</th><th>path</th><th>size</th><th></th></tr></thead><tbody>${rows.join("")}</tbody></table>`
    : `<div class="empty">test_models/ is empty.</div>`;
  $$("[data-copy]", $("#models-table")).forEach((b) => b.onclick = () => {
    navigator.clipboard?.writeText(b.dataset.copy).then(
      () => toast("Path copied", "ok"), () => toast(b.dataset.copy));
  });
  $$("[data-del]", $("#models-table")).forEach((b) => b.onclick = async () => {
    const name = b.dataset.name;
    const size = fmtBytes(+b.dataset.size);
    if (!confirm(`Delete "${name}" (${size})?\n\nThis permanently removes the directory. This cannot be undone.`)) return;
    try {
      await post("/api/models/delete", { path: b.dataset.del });
      toast(`Deleted ${name}`, "ok");
      state.models = [];
      renderModels();
    } catch (e) { toast(e.message, "err"); }
  });
}

/* ---------------- jobs view ---------------- */

function renderJobsView() {
  const f = $("#jobs-filter")?.value || "all";
  const jobs = f === "all" ? state.jobs : state.jobs.filter((j) => j.kind === f);
  $("#jobs-table").innerHTML = jobTable(jobs, { legacy: f === "all" || f === "quant" ? state.legacy : [] });
  bindJobRows($("#jobs-table"));
}
$("#jobs-filter")?.addEventListener("change", renderJobsView);
$("#jobs-refresh")?.addEventListener("click", pollJobs);
$("#models-refresh")?.addEventListener("click", () => { state.models = []; renderModels(); });

/* ---------------- boot ---------------- */

async function boot() {
  try {
    state.specs = (await api("/api/specs")).specs;
  } catch (e) {
    document.body.innerHTML = `<div class="empty" style="padding:40px">Console API unreachable: ${esc(e.message)}</div>`;
    return;
  }
  initQuantize();
  initServe();
  initChat();
  initSpecRunner("eval", "eval");
  initSpecRunner("tool", "tools");
  renderModels();

  const v = (location.hash || "#dashboard").slice(1);
  switchView(VIEW_TITLES[v] ? v : "dashboard");

  pollSystem();
  pollJobs();
  setInterval(pollSystem, 6000);
  setInterval(pollJobs, 4000);
  // keep chat endpoint in sync with a console-started server
  setInterval(() => {
    const s = state.jobs.find((j) => j.kind === "serve" && j.status === "running");
    if (s?.meta?.port && $("#chat-base").value.endsWith("8888") && String(s.meta.port) !== "8888")
      $("#chat-base").value = `http://127.0.0.1:${s.meta.port}`;
    $("#badge-serve").textContent = s ? "●" : "–";
  }, 4000);
}

boot();
