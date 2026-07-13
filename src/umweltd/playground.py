"""The playground — one self-contained HTML page the supervisor serves at /ui.

A browser-playable surface over the daemon's own JSON API, zero build step, zero
external assets (works on an air-gapped LAN): pick a world, watch its beliefs ease
in near-real-time, push readings at it with sliders, read the shadow decisions.
The page itself holds no world data and needs no auth to load; every API call it
makes carries the X-API-Key the visitor enters (kept in localStorage).
"""
from __future__ import annotations

PLAYGROUND_HTML = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>umwelt playground</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; background:#0d1117; color:#d7dde3;
       font:15px/1.5 system-ui,sans-serif; }
header { display:flex; align-items:center; gap:1rem; flex-wrap:wrap;
         padding:.8rem 1.2rem; background:#141b23; border-bottom:1px solid #26313c;
         position:sticky; top:0; z-index:5; }
header h1 { font-size:1.05rem; margin:0; color:#f0f4f8; font-weight:600; }
header h1 span { color:#6fc3ff; }
header .spacer { flex:1; }
header a { color:#8b98a5; font-size:.85rem; text-decoration:none; }
header a:hover { color:#6fc3ff; }
input, button, select { font:inherit; color:inherit; }
input[type=password], input[type=number], input[type=text] {
  background:#0d1117; border:1px solid #2a3540; border-radius:6px;
  padding:.35rem .6rem; }
input:focus { outline:1px solid #3b82c4; }
button { background:#1f6feb; border:0; border-radius:6px; color:#fff;
         padding:.4rem .9rem; cursor:pointer; }
button:hover { background:#2f7ef7; }
button.ghost { background:#1b232c; color:#c6d0da; border:1px solid #2a3540; }
button.ghost:hover { border-color:#3b82c4; }
#status { font-size:.85rem; color:#8b98a5; }
#status.ok { color:#4cc38a; } #status.err { color:#ff7b72; }
main { display:grid; grid-template-columns:230px 1fr; min-height:calc(100vh - 57px); }
nav { border-right:1px solid #26313c; padding:1rem .8rem; background:#10161d; }
nav h2 { font-size:.75rem; text-transform:uppercase; letter-spacing:.08em;
         color:#8b98a5; margin:.2rem .4rem .6rem; }
.world { display:flex; align-items:center; gap:.5rem; width:100%; text-align:left;
         background:none; border:0; border-radius:8px; padding:.5rem .6rem;
         color:#c6d0da; cursor:pointer; }
.world:hover { background:#1b232c; }
.world.sel { background:#1d2a3a; color:#fff; }
.dot { width:.55rem; height:.55rem; border-radius:50%; background:#8b98a5;
       flex:none; }
.dot.up { background:#4cc38a; } .dot.down { background:#ff7b72; }
section { padding:1.2rem 1.6rem 4rem; max-width:70rem; }
.chips { display:flex; gap:.6rem; flex-wrap:wrap; margin:.2rem 0 1.2rem; }
.chip { background:#161d26; border:1px solid #26313c; border-radius:999px;
        padding:.25rem .8rem; font-size:.8rem; color:#a9b4bf; }
.chip b { color:#e8eef4; font-weight:600; }
h3 { margin:1.6rem 0 .7rem; font-size:1rem; color:#f0f4f8; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr));
        gap:.9rem; }
.card { background:#141b23; border:1px solid #26313c; border-radius:10px;
        padding:.8rem .95rem; }
.card h4 { margin:0 0 .55rem; font-size:.92rem; color:#e8eef4; font-weight:600; }
.card h4 small { color:#5c6a77; font-weight:400; margin-left:.4rem; }
.role { margin:.45rem 0; }
.role .lab { display:flex; justify-content:space-between; font-size:.8rem;
             color:#a9b4bf; margin-bottom:.15rem; }
.role .lab b { color:#e8eef4; font-variant-numeric:tabular-nums; }
.bar { position:relative; height:.6rem; background:#0d1117; border-radius:999px;
       border:1px solid #26313c; overflow:hidden; }
.bar .mid { position:absolute; left:50%; top:0; bottom:0; width:1px;
            background:#2a3540; }
.bar .fill { position:absolute; top:0; bottom:0; border-radius:999px;
             transition:left .6s ease, right .6s ease, background .6s; }
.axis { display:flex; justify-content:space-between; font-size:.68rem;
        color:#5c6a77; margin-top:.15rem; }
table.ing { width:100%; border-collapse:collapse; }
table.ing td { padding:.35rem .4rem; border-bottom:1px solid #1d2731;
               font-size:.87rem; }
table.ing td.sid { color:#a8d8a8; font-family:ui-monospace,monospace; }
table.ing td.tgt { color:#8b98a5; font-size:.8rem; }
table.ing input { width:7rem; }
.rec { background:#141b23; border:1px solid #26313c; border-left:3px solid #d29922;
       border-radius:8px; padding:.6rem .9rem; margin:.5rem 0; font-size:.85rem; }
.rec pre { margin:.3rem 0 0; white-space:pre-wrap; color:#a9b4bf; }
details { margin:1rem 0; }
details pre { background:#10161d; border:1px solid #26313c; border-radius:8px;
              padding:.8rem; overflow-x:auto; font-size:.78rem; max-height:26rem; }
.hint { color:#8b98a5; font-size:.85rem; }
.empty { color:#5c6a77; padding:2rem 0; }
label.poll { font-size:.8rem; color:#8b98a5; display:inline-flex; gap:.35rem;
             align-items:center; margin-left:1rem; }
@media (max-width:760px){ main{grid-template-columns:1fr}
  nav{border-right:0;border-bottom:1px solid #26313c} }
</style>
</head>
<body>
<header>
  <h1><span>umwelt</span> playground</h1>
  <input id="key" type="password" placeholder="API key (X-API-Key)" size="22">
  <button id="connect">connect</button>
  <span id="status">not connected</span>
  <div class="spacer"></div>
  <a href="/docs">docs</a>
</header>
<main>
  <nav>
    <h2>Worlds</h2>
    <div id="worlds"><div class="empty">connect first</div></div>
  </nav>
  <section id="panel">
    <div class="empty">Enter the API key (ask whoever runs this daemon), hit
    connect, and pick a world. Worlds are created on the host with
    <code>umweltctl create</code> or forged from a plain-English description with
    <code>umwelt-forge new</code>.</div>
  </section>
</main>
<script>
"use strict";
const $ = s => document.querySelector(s);
let world = null, timer = null;

const key = () => localStorage.getItem("umwelt_key") || "";
async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts,
    headers: { "Content-Type": "application/json", "X-API-Key": key(),
               ...(opts.headers || {}) } });
  if (!r.ok) {
    let msg = r.status + " " + r.statusText;
    try { msg = (await r.json()).error || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.json();
}
function setStatus(text, cls) { const el = $("#status");
  el.textContent = text; el.className = cls || ""; }

async function connect() {
  localStorage.setItem("umwelt_key", $("#key").value);
  try {
    const h = await api("/health");
    setStatus("connected — " + h.worlds.length + " world(s)", "ok");
    renderWorlds(h.worlds);
  } catch (e) { setStatus(e.message, "err"); }
}

function renderWorlds(list) {
  const box = $("#worlds"); box.innerHTML = "";
  if (!list.length) box.innerHTML =
    "<div class='empty'>no worlds yet — create one on the host</div>";
  for (const w of list) {
    const b = document.createElement("button");
    b.className = "world" + (w.name === world ? " sel" : "");
    b.innerHTML = `<span class="dot ${w.running ? "up" : "down"}"></span>` +
                  `<span>${w.name}</span>`;
    b.onclick = () => select(w.name);
    box.appendChild(b);
  }
}

async function select(name) {
  world = name;
  clearInterval(timer);
  await refresh(true);
  timer = setInterval(() => { if ($("#poll") && $("#poll").checked) refresh(false); },
                      3000);
  const h = await api("/health"); renderWorlds(h.worlds);
}

const wapi = path => api(`/worlds/${world}/${path}`);

async function refresh(full) {
  try {
    const [health, state, recs] = await Promise.all(
      [wapi("health"), wapi("state"), wapi("recommendations")]);
    if (full) {
      let bindings = [];
      try { bindings = await wapi("bindings"); } catch (e) {}
      renderPanel(health, state, recs, bindings);
    } else {
      renderChips(health); renderBeliefs(state); renderRecs(recs);
      $("#rawstate").textContent = JSON.stringify(state, null, 1);
    }
    setStatus("live — " + new Date().toLocaleTimeString(), "ok");
  } catch (e) { setStatus(e.message, "err"); }
}

function renderChips(h) {
  $("#chips").innerHTML =
    `<span class="chip">step <b>${h.step}</b></span>` +
    `<span class="chip">last event <b>${h.last_event_ts || "—"}</b></span>` +
    `<span class="chip">profile <b>${h.seed_profile}</b></span>` +
    `<span class="chip">log <b>${(h.events_db_bytes/1024).toFixed(1)}kB</b></span>`;
}

function zColor(z) {           // active pole (−1) hot, calm pole (+1) cool
  const t = (1 - z) / 2;       // 0 at z=+1 … 1 at z=−1
  return `hsl(${Math.round(210 - 180 * t)},70%,55%)`;
}
function bar(z) {
  const pct = Math.max(-1, Math.min(1, z)) * 50;
  const left = pct >= 0 ? 50 : 50 + pct, width = Math.abs(pct);
  return `<div class="bar"><div class="mid"></div>` +
    `<div class="fill" style="left:${left}%;right:${100 - left - width}%;` +
    `background:${zColor(z)}"></div></div>` +
    `<div class="axis"><span>active −1</span><span>unknown 0</span>` +
    `<span>calm +1</span></div>`;
}

function renderBeliefs(state) {
  const cards = [];
  const nodes = (state.topology && state.topology.nodes) || [];
  for (const n of nodes) {
    if (n.name.startsWith("_")) continue;
    const cl = (n.organs || []).find(o => o.type === "bloch_cluster");
    if (!cl || !(cl.roles || []).length) continue;
    const rows = cl.roles.map(r =>
      `<div class="role"><div class="lab"><span>${r.role}</span>` +
      `<b>z ${r.z >= 0 ? "+" : ""}${r.z.toFixed(3)}</b></div>${bar(r.z)}</div>`);
    cards.push(`<div class="card"><h4>${n.name}` +
      `<small>${n.kind}${cl.purity != null ? " · purity " + cl.purity : ""}</small>` +
      `</h4>${rows.join("")}</div>`);
  }
  $("#beliefs").innerHTML = cards.join("") ||
    "<div class='empty'>no belief clusters to show</div>";
}

function renderIngest(bindings) {
  if (!bindings.length) {
    $("#ingest").innerHTML = "<div class='empty'>bindings unavailable on this " +
      "world (older worker?) — push events with <code>umweltctl ingest</code></div>";
    return;
  }
  const rows = bindings.map((b, i) =>
    `<tr><td class="sid">${b.sensor_id}</td>` +
    `<td class="tgt">${b.node}.${b.role}</td>` +
    `<td><input type="number" step="any" id="in_${i}" placeholder="value"></td>` +
    `<td><button class="ghost" onclick="sendOne(${i},'${b.sensor_id}')">send` +
    `</button></td></tr>`);
  $("#ingest").innerHTML =
    `<table class="ing">${rows.join("")}</table>` +
    `<p><button onclick="sendAll()">send all filled</button> ` +
    `<span class="hint">events land timestamped now, through the same ` +
    `event-sourced path as any production feed</span></p>`;
  window._bindings = bindings;
}

async function sendEvents(events) {
  try {
    const res = await api(`/worlds/${world}/events`, { method: "POST",
      body: JSON.stringify({ events }) });
    setStatus(`ingested ${res.appended} event(s), ${res.actions} action(s)`, "ok");
    refresh(false);
  } catch (e) { setStatus(e.message, "err"); }
}
function sendOne(i, sid) {
  const v = parseFloat($(`#in_${i}`).value);
  if (isNaN(v)) { setStatus("enter a number first", "err"); return; }
  sendEvents([[new Date().toISOString(), sid, v, null]]);
}
function sendAll() {
  const now = new Date().toISOString(), events = [];
  (window._bindings || []).forEach((b, i) => {
    const v = parseFloat($(`#in_${i}`).value);
    if (!isNaN(v)) events.push([now, b.sensor_id, v, null]);
  });
  if (!events.length) { setStatus("no values filled in", "err"); return; }
  sendEvents(events);
}

function renderRecs(recs) {
  $("#recs").innerHTML = (recs || []).slice(-8).reverse().map(r =>
    `<div class="rec">shadow decision<pre>${JSON.stringify(r, null, 1)}</pre></div>`
  ).join("") || "<div class='empty'>no shadow decisions yet — outputs decide " +
                "only when their belief moves</div>";
}

function renderPanel(health, state, recs, bindings) {
  $("#panel").innerHTML = `
    <h2 style="margin:.2rem 0 .8rem">${world}
      <label class="poll"><input type="checkbox" id="poll" checked>
      auto-refresh 3s</label></h2>
    <div class="chips" id="chips"></div>
    <h3>Beliefs <span class="hint">— live, easing between observations; z is the
      belief coordinate, not a flag</span></h3>
    <div class="grid" id="beliefs"></div>
    <h3>Push readings</h3>
    <div id="ingest"></div>
    <h3>Shadow decisions <span class="hint">— decided visibly, dispatched nowhere
      until promoted</span></h3>
    <div id="recs"></div>
    <details><summary class="hint">raw state projection</summary>
      <pre id="rawstate"></pre></details>`;
  renderChips(health); renderBeliefs(state); renderIngest(bindings);
  renderRecs(recs);
  $("#rawstate").textContent = JSON.stringify(state, null, 1);
}

$("#connect").onclick = connect;
$("#key").value = key();
$("#key").addEventListener("keydown", e => { if (e.key === "Enter") connect(); });
if (key()) connect();
</script>
</body>
</html>
"""
