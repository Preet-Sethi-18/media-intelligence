"use strict";
// Browser interface over the existing API. Scraped text is untrusted: it is only ever inserted as text nodes.

const $ = (id) => document.getElementById(id);
const TYPE_LABEL = { PERSON: "person", ORG: "organisation", LOCATION: "place", TOPIC: "topic" };
const RELATION_LABEL = {
  affiliated_with: "works for / part of", criticized: "criticized", sanctioned: "sanctioned", attacked: "attacked",
  quoted_by: "quoted by", met_with: "met with", acquired: "acquired", invested_in: "invested in",
  partnered_with: "partnered with", released: "released", discussed_topic: "discussed",
  mentioned_with: "mentioned together",
};
const RELATION_HELP = {
  affiliated_with: "A holds a role in, works for, leads or founded B.",
  criticized: "A criticized, condemned, accused or blamed B.",
  sanctioned: "A imposed sanctions or an embargo on B.",
  attacked: "A carried out a military (or cyber) attack on B.",
  quoted_by: "B (a news outlet) reports A's words.",
  met_with: "A and B met or held talks.",
  acquired: "A bought or agreed to buy B.",
  invested_in: "A invested money in B.",
  partnered_with: "A and B partnered or signed a deal.",
  released: "A released or announced product B.",
  discussed_topic: "A discussed, negotiated or warned about topic B.",
  mentioned_with: "Weak link: A and B appear in the same sentence, with no clearer relation found.",
};
const state = { overview: null, cy: null, names: false };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined) node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw { status: response.status, body };
  return body;
}

function errorText(err) {
  if (err && err.body && err.body.error) return err.body.error.message;
  return "Could not reach the server. Is it still running?";
}

function fmtTime(iso) {
  if (!iso) return "—";
  return iso.replace("T", " ").replace(/:\d\d\.\d+Z$|:\d\dZ$/, "") + " UTC";
}

function message(id, text, isError = false) {
  const box = $(id);
  box.hidden = !text;
  box.className = "message" + (isError ? " error" : "");
  box.textContent = text || "";
}

function table(headers, rows) {
  return el("div", { class: "table-wrap" },
    el("table", {}, el("thead", {}, el("tr", {}, headers.map((h) => el("th", { class: h.num ? "num" : "" }, h.label || h)))),
      el("tbody", {}, rows)));
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ---------- navigation ---------- */
const loaders = { overview: loadOverview, explore: loadExplore, central: loadCentral, new: loadNew, pipeline: loadPipeline };

function show() {
  const view = (location.hash || "#overview").slice(1).split("?")[0];
  const name = loaders[view] ? view : "overview";
  document.querySelectorAll(".view").forEach((s) => { s.hidden = s.id !== "view-" + name; });
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.dataset.view === name));
  loaders[name]();
}

/* ---------- overview ---------- */
async function getOverview(force = false) {
  if (!state.overview || force) state.overview = await api("/ui-api/overview");
  return state.overview;
}

async function loadOverview() {
  let data;
  try { data = await getOverview(true); } catch (err) { $("overview-cards").replaceChildren(el("div", { class: "message error" }, errorText(err))); return; }
  const graph = data.graph;
  const cards = $("overview-cards");
  if (!graph.available) {
    cards.replaceChildren(el("div", { class: "message" }, "No graph yet (", graph.message, "). Open ",
      el("a", { href: "#pipeline" }, "Run pipeline"), " to build one."));
  } else {
    const c = graph.counts;
    const card = (value, label) => el("div", { class: "card" }, el("b", {}, value.toLocaleString()), el("span", {}, label));
    cards.replaceChildren(card(c.sources, "page versions crawled"), card(c.nodes, "entities"),
      card(graph.typed_edges, "typed connections"), card(graph.weak_edges, "weak \"mentioned together\" links"),
      card(c.evidence, "evidence sentences"));
  }
  $("config-path").textContent = data.sources.path || "config/sources.toml";
  if (data.sources.error) {
    $("overview-sources").replaceChildren(el("div", { class: "message error" }, "Config problem: ", data.sources.error));
  } else {
    $("overview-sources").replaceChildren(table(["Source", "Type", "Allowed sites", "Seed pages"],
      data.sources.items.map((s) => el("tr", {}, el("td", {}, s.name), el("td", {}, s.source_type),
        el("td", {}, s.allowed_domains.join(", ")),
        el("td", {}, el("ul", {}, s.seeds.map((u) => el("li", {}, el("a", { href: u, target: "_blank", rel: "noopener noreferrer" }, u)))))))));
  }
  const runs = graph.available ? graph.runs : [];
  $("overview-runs").replaceChildren(runs.length ? table(["Run", "Status", "Started", "Finished", { label: "New pages", num: true },
    { label: "Unchanged", num: true }, "Problems"],
  runs.map((r) => el("tr", {}, el("td", {}, "#" + r.id), el("td", {}, el("span", { class: "badge " + r.status }, r.status)),
    el("td", {}, fmtTime(r.started_at)), el("td", {}, fmtTime(r.finished_at)),
    el("td", { class: "num" }, r.stored_new ?? "—"), el("td", { class: "num" }, r.stored_unchanged ?? "—"),
    el("td", {}, problems(r)))))
    : el("p", { class: "hint" }, "No runs yet."));
}

function problems(run) {
  const parts = Object.entries(run.failed_seeds).map(([source, n]) => `${n} failed seed page(s) in ${source}`);
  if (run.missing_sources.length) parts.push("nothing stored for " + run.missing_sources.join(", "));
  return parts.length ? parts.join("; ") : "none";
}

/* ---------- explore ---------- */
async function loadExplore() {
  if (!state.names) {
    state.names = true;
    try {
      const central = await api("/entities/central?limit=100");
      $("entity-names").replaceChildren(...central.items.map((i) => el("option", { value: i.name })));
    } catch { /* suggestions are optional */ }
  }
  const params = new URLSearchParams(location.hash.split("?")[1] || "");
  if (params.get("name")) {
    $("explore-name").value = params.get("name");
    explore(params.get("name"), params.get("id"));
  }
}

function openExplore(name, id) {
  const query = new URLSearchParams({ name });
  if (id) query.set("id", id);
  const target = "#explore?" + query;
  if (location.hash === target) loadExplore();  // same address: hashchange would not fire
  else location.hash = target;
}

async function explore(name, entityId) {
  message("explore-message", "Loading…");
  const query = new URLSearchParams({ depth: $("explore-depth").value });
  if (entityId) query.set("entity_id", entityId);
  let data;
  try {
    data = await api(`/entity/${encodeURIComponent(name)}/network?${query}`);
  } catch (err) {
    if (err.status === 409 && err.body.error.candidates) {
      const box = $("explore-message");
      box.hidden = false; box.className = "message";
      box.replaceChildren(`"${name}" matches more than one entity. Pick one: `,
        ...err.body.error.candidates.map((c) => el("button", { class: "link", onclick: () => openExplore(name, c.id) },
          ` ${c.name} (${TYPE_LABEL[c.type] || c.type}, id ${c.id}) `)));
    } else {
      message("explore-message", err.status === 404 ? `No entity called "${name}" was found. Try a suggestion from the list.` : errorText(err), true);
    }
    return;
  }
  drawGraph(data);
}

const SIMPLE_WEAK_LIMIT = 10;

function drawGraph(data) {
  const simple = $("explore-mode").value === "simple";
  const isWeak = (e) => e.relation_type === "mentioned_with";
  const touchesRoot = (e) => e.source === data.root_id || e.target === data.root_id;
  const other = (e) => (e.source === data.root_id ? e.target : e.source);
  const byId = Object.fromEntries(data.nodes.map((n) => [n.id, n]));
  const root = byId[data.root_id];
  const own = data.edges.filter(touchesRoot);
  const ownTyped = own.filter((e) => !isWeak(e));
  const ownWeak = own.filter(isWeak).sort((a, b) => b.weight - a.weight || byId[other(b)].mention_count - byId[other(a)].mention_count);

  let edges;
  if (simple) {
    // Centre: the entity. Inner ring: its typed links. Outer ring: its most frequent "mentioned together" partners.
    // Links between the other entities are left out, except typed links reaching a new entity at depth 2.
    const typedPartners = new Set(ownTyped.map(other));
    const shownWeak = ownWeak.filter((e) => !typedPartners.has(other(e))).slice(0, SIMPLE_WEAK_LIMIT);
    const direct = new Set([...ownTyped, ...shownWeak].map(other));
    // Depth 2 adds typed links from a shown neighbour out to a second-ring entity, nothing in between.
    const second = data.edges.filter((e) => !isWeak(e) && !touchesRoot(e)
      && ((direct.has(e.source) && byId[e.target].distance === 2) || (direct.has(e.target) && byId[e.source].distance === 2)));
    edges = [...ownTyped, ...shownWeak, ...second];
    var positions = ringPositions(data.root_id, [...ownTyped, ...shownWeak].map(other), second, direct);
  } else {
    edges = data.edges;
  }
  const used = new Set([data.root_id]);
  edges.forEach((e) => { used.add(e.source); used.add(e.target); });
  const nodes = data.nodes.filter((n) => used.has(n.id));

  let text = `${root.name} (${TYPE_LABEL[root.type]}) is linked to ${own.length} entities: ${ownTyped.length} with a typed ` +
    `connection (blue, labelled) and ${ownWeak.length} only "mentioned together" (dashed).`;
  if (simple && ownWeak.length > SIMPLE_WEAK_LIMIT) {
    text += ` The graph shows the ${SIMPLE_WEAK_LIMIT} most frequent of those; choose "Full" to see everything.`;
  }
  if (data.meta.truncated) text += ` The result was cut to the ${data.meta.max_nodes} most-mentioned entities.`;
  message("explore-message", text);

  const colours = { PERSON: cssVar("--person"), ORG: cssVar("--org"), LOCATION: cssVar("--location"), TOPIC: cssVar("--topic") };
  const maxMentions = Math.max(...nodes.map((n) => n.mention_count), 1);
  if (state.cy) state.cy.destroy();
  state.cy = cytoscape({
    container: $("graph"),
    elements: [
      ...nodes.map((n) => ({ data: { id: "n" + n.id, nodeId: n.id, label: n.name, type: n.type, mentions: n.mention_count,
        colour: colours[n.type], size: 18 + 30 * Math.sqrt(n.mention_count / maxMentions) },
      classes: n.id === data.root_id ? "root" : "" })),
      ...edges.map((e) => ({ data: { id: "e" + e.id, edgeId: e.id, source: "n" + e.source, target: "n" + e.target,
        label: RELATION_LABEL[e.relation_type] || e.relation_type, weight: e.weight },
      classes: (isWeak(e) ? "weak" : "typed") + (e.directed ? " directed" : "") })),
    ],
    style: [
      { selector: "node", style: { "background-color": "data(colour)", width: "data(size)", height: "data(size)",
        label: "data(label)", color: cssVar("--text"), "font-size": 15, "text-valign": "bottom", "text-margin-y": 4,
        "text-wrap": "wrap", "text-max-width": 120, "text-outline-color": cssVar("--panel"), "text-outline-width": 3 } },
      { selector: "node.root", style: { "border-width": 4, "border-color": cssVar("--text"), "font-weight": "bold", "font-size": 18 } },
      { selector: "edge", style: { "curve-style": "bezier", width: "mapData(weight, 1, 8, 1.5, 6)" } },
      { selector: "edge.typed", style: { "line-color": cssVar("--accent"), "target-arrow-color": cssVar("--accent"),
        label: "data(label)", "font-size": 13, color: cssVar("--text"), "text-rotation": "autorotate",
        "text-background-color": cssVar("--panel"), "text-background-opacity": 1, "text-background-padding": 2 } },
      { selector: "edge.directed", style: { "target-arrow-shape": "triangle" } },
      { selector: "edge.weak", style: { "line-color": cssVar("--weak"), "line-style": "dashed", width: 1, "target-arrow-shape": "none" } },
      { selector: ":selected", style: { "overlay-color": cssVar("--accent"), "overlay-opacity": 0.2 } },
    ],
    layout: simple
      ? { name: "preset", positions: (node) => positions[node.data("nodeId")], padding: 40 }
      : { name: "cose", animate: false, nodeRepulsion: 12000, idealEdgeLength: 120, padding: 30 },
    wheelSensitivity: 0.3,
  });
  state.cy.on("tap", "edge", (evt) => openEdge(evt.target.data("edgeId")));
  state.cy.on("tap", "node", (evt) => openNode(evt.target.data()));
  $("legend").replaceChildren(
    ...Object.entries(TYPE_LABEL).map(([type, label]) => el("div", {}, el("span", { class: "swatch", style: `background:${colours[type]}` }), label)),
    el("div", { class: "meta" }, "Blue labelled line: typed connection (arrow = direction). Dashed grey: mentioned together. Bigger dot: mentioned on more pages."));
  drawLists(root, ownTyped, ownWeak, byId, other);
}

// Simple view geometry: the entity at the centre, its neighbours evenly on a first ring, and each second-ring
// entity on an outer ring at its neighbour's angle, so no line passes behind another dot.
function ringPositions(rootId, neighbours, second, direct) {
  const positions = { [rootId]: { x: 0, y: 0 } };
  const angle = {};
  const inner = Math.max(170, neighbours.length * 26);
  neighbours.forEach((id, i) => {
    angle[id] = -Math.PI / 2 + (2 * Math.PI * i) / neighbours.length;
    positions[id] = { x: inner * Math.cos(angle[id]), y: inner * Math.sin(angle[id]) };
  });
  const children = {};
  for (const e of second) {
    const [parent, child] = direct.has(e.source) ? [e.source, e.target] : [e.target, e.source];
    if (!(child in positions) && !Object.values(children).some((list) => list.includes(child))) {
      (children[parent] = children[parent] || []).push(child);
    }
  }
  const outer = inner + 160;
  const slot = (2 * Math.PI) / Math.max(neighbours.length, 1);
  for (const [parent, list] of Object.entries(children)) {
    const step = Math.min(0.22, (slot * 0.8) / list.length);
    list.forEach((child, j) => {
      const a = angle[parent] + (j - (list.length - 1) / 2) * step;
      positions[child] = { x: outer * Math.cos(a), y: outer * Math.sin(a) };
    });
  }
  return positions;
}

function drawLists(root, ownTyped, ownWeak, byId, other) {
  const typedItems = ownTyped.map((e) => {
    const outgoing = e.source === root.id;
    const label = RELATION_LABEL[e.relation_type] || e.relation_type;
    const partner = byId[other(e)];
    const phrase = !e.directed ? `${label} ${partner.name}` : outgoing ? `${label} → ${partner.name}` : `← ${label} by ${partner.name}`;
    return el("li", {}, el("button", { class: "link", onclick: () => openEdge(e.id) }, phrase),
      el("span", { class: "meta" }, ` · ${e.weight} page(s)`));
  });
  const weakItems = ownWeak.slice(0, 15).map((e) => {
    const partner = byId[other(e)];
    return el("li", {}, el("button", { class: "link", onclick: () => openExplore(partner.name, partner.id) }, partner.name),
      el("span", { class: "meta" }, ` · ${TYPE_LABEL[partner.type]} · together on ${e.weight} page(s)`));
  });
  $("explore-lists").replaceChildren(
    el("section", {}, el("h3", {}, `How ${root.name} is connected`),
      typedItems.length ? el("ul", {}, typedItems)
        : el("p", { class: "meta" }, "No typed connection found; only \"mentioned together\" links (right)."),
      el("p", { class: "meta" }, "Click one to read the sentence it came from.")),
    el("section", {}, el("h3", {}, "Most often mentioned together with"),
      weakItems.length ? el("ul", {}, weakItems) : el("p", { class: "meta" }, "None."),
      el("p", { class: "meta" }, "Same sentence, no clearer relation found. Click a name to explore it.")));
}

function openNode(node) {
  openDrawer(el("h3", {}, node.label), el("p", { class: "meta" }, `${TYPE_LABEL[node.type]} · mentioned on ${node.mentions} web page(s)`),
    el("button", { onclick: () => { closeDrawer(); $("explore-name").value = node.label; openExplore(node.label, node.nodeId); } },
      "Explore from here"));
}

/* ---------- evidence drawer ---------- */
function openDrawer(...content) {
  $("drawer-body").replaceChildren(...content);
  $("drawer").hidden = false;
}
function closeDrawer() { $("drawer").hidden = true; }

async function openEdge(edgeId) {
  openDrawer(el("p", {}, "Loading evidence…"));
  let data;
  try { data = await api(`/edges/${edgeId}/sources?limit=50`); } catch (err) { openDrawer(el("p", {}, errorText(err))); return; }
  const e = data.edge;
  const arrow = e.directed ? " → " : " ↔ ";
  openDrawer(
    el("h3", {}, e.source.name, arrow, RELATION_LABEL[e.relation_type] || e.relation_type, arrow, e.target.name),
    el("p", { class: "meta" }, RELATION_HELP[e.relation_type] || ""),
    el("p", {}, `Found on ${e.weight} different web page(s). First seen ${fmtTime(e.first_seen)}.`),
    ...data.items.map((item) => el("div", { class: "evidence" },
      el("blockquote", {}, item.evidence_text),
      el("div", { class: "meta" },
        el("a", { href: item.segment_url || item.source_url, target: "_blank", rel: "noopener noreferrer" }, item.title || item.source_url),
        ` · ${item.source_type}`, item.segment_author ? ` · by ${item.segment_author}` : "",
        item.published_at ? ` · published ${fmtTime(item.published_at)}` : "", ` · we saw it ${fmtTime(item.observed_at)}`),
      el("div", { class: "meta" }, `Rule: ${item.rule_id}`))));
}

/* ---------- most connected ---------- */
async function loadCentral() {
  let data;
  try { data = await api("/entities/central?limit=50"); } catch (err) { $("central-table").replaceChildren(el("div", { class: "message error" }, errorText(err))); return; }
  $("central-table").replaceChildren(table([{ label: "Rank", num: true }, "Entity", "Type", { label: "Connections", num: true },
    { label: "Kinds of connection", num: true }, { label: "Mentioned on pages", num: true }],
  data.items.map((i) => el("tr", {}, el("td", { class: "num" }, i.rank),
    el("td", {}, el("button", { class: "link", onclick: () => openExplore(i.name, i.node_id) }, i.name)),
    el("td", {}, el("span", { class: "type" }, TYPE_LABEL[i.type] || i.type)), el("td", { class: "num" }, i.degree),
    el("td", { class: "num" }, i.relation_type_count), el("td", { class: "num" }, i.mention_count)))));
}

/* ---------- new connections ---------- */
async function loadNew() {
  if (!$("new-since").value) {
    try {
      const data = await getOverview();
      const runs = data.graph.available ? data.graph.runs : [];
      if (runs.length) $("new-since").value = runs[0].started_at.replace(/\.\d+Z$/, "Z");
    } catch { /* the user can type a time */ }
  }
  if ($("new-since").value) newConnections();
}

async function newConnections() {
  const since = $("new-since").value.trim();
  let data;
  try { data = await api(`/connections/new?since=${encodeURIComponent(since)}&limit=100`); } catch (err) {
    message("new-message", err.status === 422 ? "Use a full time with a timezone, e.g. 2026-10-07T20:10:40Z (and not in the future)." : errorText(err), true);
    $("new-table").replaceChildren();
    return;
  }
  const shown = data.items.length < data.total ? ` Showing the first ${data.items.length}.` : "";
  message("new-message", `${data.total} connection(s) are new or grew significantly since ${fmtTime(data.since)}.${shown}`);
  $("new-table").replaceChildren(table(["What happened", "Entity A", "Connection", "Entity B", { label: "Pages before → now", num: true }],
    data.items.map((i) => el("tr", { class: "clickable", onclick: () => openEdge(i.edge_id) },
      el("td", {}, el("span", { class: "badge " + i.reason }, i.reason === "new" ? "new" : "grew")),
      el("td", {}, i.source.name), el("td", {}, RELATION_LABEL[i.relation_type] || i.relation_type), el("td", {}, i.target.name),
      el("td", { class: "num" }, `${i.weight_before} → ${i.weight_now}`)))));
}

/* ---------- pipeline ---------- */
let polling = null;

async function loadPipeline() {
  try { $("pipeline-db").textContent = (await getOverview()).database; } catch { /* shown on overview */ }
  refreshPipeline();
}

async function refreshPipeline() {
  let status;
  try { status = await api("/ui-api/pipeline"); } catch (err) { message("pipeline-status", errorText(err), true); return; }
  const running = status.state === "running";
  $("pipeline-start").disabled = running;
  if (status.state === "idle") { message("pipeline-status", ""); $("pipeline-log").hidden = true; return; }
  if (running) message("pipeline-status", `Running since ${fmtTime(status.started_at)}… this page updates by itself.`);
  else message("pipeline-status", `Finished at ${fmtTime(status.finished_at)}: ${status.exit_meaning} (exit code ${status.exit_code}).`,
    status.exit_code !== 0);
  $("pipeline-log").hidden = false;
  $("pipeline-log").textContent = status.log_tail.join("\n");
  clearTimeout(polling);
  if (running) polling = setTimeout(refreshPipeline, 2000);
  else state.overview = null;  // next visit to other pages shows the new data
}

$("pipeline-start").addEventListener("click", async () => {
  try { await api("/ui-api/pipeline", { method: "POST" }); } catch (err) { message("pipeline-status", errorText(err), true); }
  refreshPipeline();
});

/* ---------- wiring ---------- */
$("explore-form").addEventListener("submit", (evt) => { evt.preventDefault(); openExplore($("explore-name").value.trim()); });
$("explore-mode").addEventListener("change", () => { const n = $("explore-name").value.trim(); if (n) loadExplore(); });
$("explore-depth").addEventListener("change", () => { const n = $("explore-name").value.trim(); if (n) loadExplore(); });
$("new-form").addEventListener("submit", (evt) => { evt.preventDefault(); newConnections(); });
$("drawer-close").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (evt) => { if (evt.key === "Escape") closeDrawer(); });
window.addEventListener("hashchange", show);
show();
