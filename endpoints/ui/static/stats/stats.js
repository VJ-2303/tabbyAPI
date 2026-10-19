// TabbyAPI statistics page. Plain browser JavaScript, no build step: it polls
// /v1/stats and renders it. Everything is text-only rendering (never markup
// from the server), and the key is the one the chat client at /ui already
// keeps, so a browser that has used the client is logged in here too.

const POLL_MS = 2000;
const MAX_BACKOFF_MS = 15000;
const SPARK_SAMPLES = 150; // five minutes at one poll every two seconds
const MAX_RECENT_ROWS = 20;
const HOURLY_BARS = 24;
const STORE = "tabby-ui."; // shared with the chat client at /ui

const $ = (id) => document.getElementById(id);
const NUM = new Intl.NumberFormat();

// ---------------------------------------------------------------------------
// Settings, shared with the chat client

function readSettings() {
  try {
    return JSON.parse(localStorage.getItem(STORE + "settings")) || {};
  } catch {
    return {};
  }
}

function writeSettings(next) {
  try {
    localStorage.setItem(STORE + "settings", JSON.stringify({ ...readSettings(), ...next }));
  } catch {
    // Without storage the key still works for this page load
  }
}

const settings = readSettings();

// The server prints a statistics link with the key in the URL fragment, which
// the browser never sends to the server. Take it, then clear the address bar.
function takeKeyFromLink() {
  const key = new URLSearchParams(location.hash.slice(1)).get("key");
  if (!key) return;

  settings.apiKey = key;
  settings.baseUrl = "";
  writeSettings({ apiKey: key, baseUrl: "" });
  history.replaceState(null, "", location.pathname + location.search);
}

function apiUrl(path) {
  return (settings.baseUrl || "").replace(/\/+$/, "") + path;
}

// ---------------------------------------------------------------------------
// Formatting

function int(value) {
  return value === null || value === undefined ? "—" : NUM.format(Math.round(value));
}

function rate(value) {
  if (value === null || value === undefined) return "—";
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

function bytes(value) {
  if (value === null || value === undefined) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let index = 0;
  let size = value;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index >= 3 ? 1 : 0)} ${units[index]}`;
}

function duration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const total = Math.max(Math.round(seconds), 0);
  if (total < 60) return `${total}s`;
  const minutes = Math.floor(total / 60);
  if (minutes < 60) return `${minutes}m ${total % 60}s`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${minutes % 60}m`;
  return `${Math.floor(hours / 24)}d ${hours % 24}h`;
}

function clock(epochSeconds) {
  if (!epochSeconds) return "—";
  return new Date(epochSeconds * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function hourLabel(epochSeconds) {
  return new Date(epochSeconds * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function dayLabel(epochSeconds) {
  return new Date(epochSeconds * 1000).toLocaleDateString([], { month: "short", day: "numeric" });
}

function percent(part, whole) {
  if (!whole) return null;
  return Math.round((part / whole) * 100);
}

const FINISH_LABELS = {
  stop_token: "stopped",
  end_filter: "stopped",
  stop_string: "stop string",
  max_new_tokens: "hit max_tokens",
  loop_detected: "loop detected",
  cancelled: "cancelled",
  failed: "failed",
};

function finishLabel(reason) {
  return FINISH_LABELS[reason] || reason;
}

// ---------------------------------------------------------------------------
// Small DOM helpers

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function row(cells) {
  const tr = document.createElement("tr");
  for (const cell of cells) tr.append(cell);
  return tr;
}

function td(text, className) {
  return el("td", text, className);
}

function bar(fraction, tone) {
  const wrap = el("div", null, `bar${tone ? ` is-${tone}` : ""}`);
  const fill = el("span");
  fill.style.width = `${Math.max(Math.min(fraction * 100, 100), 0)}%`;
  wrap.append(fill);
  return wrap;
}

function readout(label, ...children) {
  const node = document.createElement("div");
  node.className = "readout";
  node.append(el("p", label, "readout-label"), ...children);
  return node;
}

// ---------------------------------------------------------------------------
// Charts

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag) {
  return document.createElementNS(SVG_NS, tag);
}

function sparkPaths(values, width, height) {
  if (values.length < 2) return { line: "", area: "" };

  const max = Math.max(...values, 1);
  const step = width / (values.length - 1);
  const points = values.map((value, index) => [
    index * step,
    height - (value / max) * (height - 6) - 3,
  ]);
  const line = points
    .map(([x, y], index) => `${index ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`)
    .join(" ");
  return { line, area: `${line} L${width} ${height} L0 ${height} Z` };
}

function renderSpark(values) {
  const { line, area } = sparkPaths(values, 640, 64);
  $("spark-line").setAttribute("d", line);
  $("spark-area").setAttribute("d", area);
}

// Bars for a series of buckets. The newest bucket is a partial hour or day, so
// it is drawn in the quieter tone
function renderBars(svg, buckets, valueKey, width, height) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const max = Math.max(...buckets.map((bucket) => bucket[valueKey] || 0), 1);
  const gap = 2;
  const barWidth = (width - gap * (buckets.length - 1)) / buckets.length;

  svg.append(
    (() => {
      const baseline = svgEl("line");
      baseline.setAttribute("class", "baseline");
      baseline.setAttribute("x1", 0);
      baseline.setAttribute("x2", width);
      baseline.setAttribute("y1", height - 0.5);
      baseline.setAttribute("y2", height - 0.5);
      return baseline;
    })()
  );

  buckets.forEach((bucket, index) => {
    const value = bucket[valueKey] || 0;
    const barHeight = (value / max) * (height - 10);
    const rect = svgEl("rect");
    rect.setAttribute("class", `barrect${index === buckets.length - 1 ? " is-quiet" : ""}`);
    rect.setAttribute("x", (index * (barWidth + gap)).toFixed(1));
    rect.setAttribute("width", barWidth.toFixed(1));
    rect.setAttribute("y", (height - barHeight).toFixed(1));
    rect.setAttribute("height", Math.max(barHeight, 0).toFixed(1));

    const when = bucket.hour ? hourLabel(bucket.hour) : dayLabel(bucket.day);
    const tooltip = svgEl("title");
    tooltip.textContent = `${when} · ${int(value)} tokens · ${int(bucket.requests)} request(s) · ${int(
      bucket.processed_tokens
    )} processed`;
    rect.append(tooltip);
    svg.append(rect);
  });
}

function renderAxis(container, labels) {
  container.replaceChildren();
  for (const label of labels) container.append(el("span", label, "mono"));
}

// ---------------------------------------------------------------------------
// Renderers

let sparkSamples = [];

function setState(tone, text) {
  $("state").className = `state${tone ? ` is-${tone}` : ""}`;
  $("state-text").textContent = text;
}

function setAlert(message) {
  const alert = $("alert");
  alert.textContent = message || "";
  alert.hidden = !message;
}

function renderHeader(data) {
  $("up-time").textContent = `${duration(data.session.uptime)} · run ${int(data.server.sessions)}`;
  $("model-name").textContent = data.model && data.model.id ? data.model.id : "no model loaded";
}

function renderHealth(data) {
  const issues = data.incidents.length;
  if (issues) {
    setState("warn", `${issues} issue${issues === 1 ? "" : "s"}`);
  } else {
    setState("", "healthy");
  }
}

function renderLive(data) {
  const live = data.live;
  const heroRate = $("hero-rate");
  const heroNote = $("hero-note");

  if (!live) {
    heroRate.textContent = "—";
    heroNote.textContent = "no model loaded";
    $("cache").replaceChildren();
    $("jobs").replaceChildren(el("p", "no generations in flight", "jobs-empty"));
    renderStateSamples(0);
    return;
  }

  heroRate.textContent = rate(live.generating_tokens_per_sec ?? 0);
  heroNote.textContent = liveNote(live);
  renderStateSamples(live.generating_tokens_per_sec ?? 0);

  renderCache(live.cache);
  renderJobs(live.jobs);
}

function liveNote(live) {
  const parts = [];
  if (live.active) parts.push(`${int(live.active)} generating`);
  const queued = live.queued || 0;
  if (queued) parts.push(`${int(queued)} queued`);
  if (!parts.length) parts.unshift("idle");
  if (live.jobs.length) parts.push(`${int(live.jobs.length)} job${live.jobs.length === 1 ? "" : "s"}`);
  return parts.join(" · ");
}

function renderCache(cache) {
  const node = $("cache");
  if (!cache || !cache.max_tokens) {
    node.replaceChildren();
    return;
  }

  const usedFraction = cache.used_tokens / cache.max_tokens;
  const details = [];
  if (cache.cached_tokens !== null && cache.cached_tokens !== undefined) {
    details.push(`${int(cache.cached_tokens)} reusable tokens`);
  }
  if (cache.hit_rate !== null && cache.hit_rate !== undefined) {
    details.push(`hit rate ${percent(cache.hit_rate, 1)}%`);
  }
  if (cache.tier_max_tokens) {
    details.push(`${int(cache.tier_cached_tokens)} in system memory`);
  }
  if (cache.disk_max_bytes) {
    details.push(`${bytes(cache.disk_used_bytes)} of ${bytes(cache.disk_max_bytes)} on disk`);
  }

  node.replaceChildren(
    readout(
      "kv cache",
      bar(usedFraction, usedFraction > 0.95 ? "bad" : usedFraction > 0.85 ? "warn" : null),
      el(
        "p",
        `${int(cache.used_tokens)} / ${int(cache.max_tokens)} tokens (${percent(
          cache.used_tokens,
          cache.max_tokens
        )}%)`,
        "readout-value mono"
      ),
      el("p", details.join(" · "), "readout-detail")
    )
  );
}

function renderGpu(data) {
  const node = $("gpu");
  if (!data.gpu.length) {
    node.replaceChildren();
    return;
  }

  const rows = data.gpu.map((gpu) => {
    const wrap = document.createElement("div");
    wrap.className = "gpu-row";
    const used = percent(gpu.used_bytes, gpu.total_bytes);
    wrap.append(
      bar(gpu.used_bytes / (gpu.total_bytes || 1), used > 95 ? "bad" : used > 85 ? "warn" : null),
      el(
        "p",
        `${bytes(gpu.used_bytes)} / ${bytes(gpu.total_bytes)} used (${used}%)`,
        "readout-value mono"
      ),
      el("p", `${gpu.name} · device ${gpu.index}`, "readout-detail")
    );
    return wrap;
  });

  node.replaceChildren(readout("vram", ...rows));
}

function renderJobs(jobs) {
  const node = $("jobs");
  if (!jobs.length) {
    node.replaceChildren(el("p", "no generations in flight", "jobs-empty"));
    return;
  }

  const table = el("table", null, "table");
  const head = document.createElement("thead");
  head.append(
    row([
      el("th", "generation"),
      el("th", "stage"),
      el("th", "tokens", "num"),
      el("th", "tokens/s", "num"),
    ])
  );
  const body = document.createElement("tbody");
  for (const job of jobs) {
    const stage = document.createElement("td");
    stage.append(el("span", job.stage, "label"), " ");
    if (job.stage === "prefill" && job.prompt_tokens) {
      stage.append(bar(job.prefill_tokens / job.prompt_tokens));
    }

    const tokens =
      job.stage === "prefill"
        ? `${int(job.prefill_tokens)} / ${int(job.prompt_tokens)}`
        : int(job.gen_tokens);

    body.append(
      row([
        td(job.label || "—", "mono label"),
        stage,
        td(tokens, "num"),
        td(rate(job.gen_tokens_per_sec), "num"),      ])
    );
  }
  table.append(head, body);
  node.replaceChildren(table);
}

function countersRow(label, sessionValue, lifetimeValue, className) {
  return row([
    el("td", label, `label ${className || ""}`.trim()),
    td(sessionValue, `num ${className || ""}`.trim()),
    td(lifetimeValue, `num ${className || ""}`.trim()),
  ]);
}

function groupRow(label) {
  const tr = document.createElement("tr");
  tr.className = "group";
  tr.append(el("td", label));
  tr.append(el("td"), el("td"));
  return tr;
}

function renderTotals(data) {
  const session = data.session;
  const lifetime = data.lifetime;
  const body = $("totals-body");
  body.replaceChildren();

  body.append(
    groupRow("tokens"),
    countersRow("generated", int(session.gen_tokens), int(lifetime.gen_tokens)),
    countersRow("prompt", int(session.prompt_tokens), int(lifetime.prompt_tokens)),
    countersRow("processed", int(session.processed_tokens), int(lifetime.processed_tokens)),
    countersRow("from cache", int(session.cached_tokens), int(lifetime.cached_tokens))
  );

  const sessionReuse = percent(session.cached_tokens, session.prompt_tokens);
  const lifetimeReuse = percent(lifetime.cached_tokens, lifetime.prompt_tokens);
  if (lifetimeReuse !== null) {
    body.append(
      countersRow(
        "cache reuse",
        sessionReuse === null ? "—" : `${sessionReuse}%`,
        `${lifetimeReuse}%`,
        "dim"
      )
    );
  }

  body.append(groupRow("speed"));
  body.append(
    countersRow(
      "generation rate",
      `${rate(session.gen_tokens_per_sec)} /s`,
      `${rate(lifetime.gen_tokens_per_sec)} /s`
    ),
    countersRow(
      "median / p95",
      `${rate(session.gen_tokens_per_sec_p50)} / ${rate(session.gen_tokens_per_sec_p95)}`,
      "—",
      "dim"
    ),
    countersRow(
      "prompt rate",
      `${rate(session.prompt_tokens_per_sec)} /s`,
      `${rate(lifetime.prompt_tokens_per_sec)} /s`
    ),
    countersRow(
      "avg first token",
      avgFirstToken(session),
      avgFirstToken(lifetime)
    ),
    countersRow("generation time", duration(session.gen_time), duration(lifetime.gen_time)),
    countersRow("prompt time", duration(session.prompt_time), duration(lifetime.prompt_time)),
    countersRow("queued time", duration(session.queue_time), duration(lifetime.queue_time))
  );

  body.append(groupRow("requests"));
  body.append(
    countersRow("requests", int(session.requests), int(lifetime.requests))
  );

  const draftSession = session.draft_accept + session.draft_reject;
  const draftLifetime = lifetime.draft_accept + lifetime.draft_reject;
  if (draftLifetime) {
    const accepted = percent(lifetime.draft_accept, draftLifetime);
    body.append(
      countersRow(
        "draft accepted",
        draftSession ? `${percent(session.draft_accept, draftSession)}%` : "—",
        `${percent(lifetime.draft_accept, draftLifetime)}% of ${int(draftLifetime)} (${accepted}%)`,
        "dim"
      )
    );
  }

  if (lifetime.embedding_requests) {
    body.append(
      countersRow(
        "embeddings",
        `${int(session.embedding_requests)} req · ${int(session.embedding_tokens)} tokens`,
        `${int(lifetime.embedding_requests)} req · ${int(lifetime.embedding_tokens)} tokens`,
        "dim"
      )
    );
  }

  if (lifetime.failed || lifetime.cancelled) {
    body.append(
      countersRow(
        "failed / cancelled",
        `${int(session.failed)} / ${int(session.cancelled)}`,
        `${int(lifetime.failed)} / ${int(lifetime.cancelled)}`
      )
    );
  }

  renderFinishNote(lifetime.finish);
}

function avgFirstToken(counters) {
  if (!counters.requests) return "—";
  return duration((counters.queue_time + counters.prompt_time) / counters.requests);
}

function renderFinishNote(finish) {
  const entries = Object.entries(finish || {}).sort((a, b) => b[1] - a[1]);
  const note = $("finish-note");
  if (!entries.length) {
    note.textContent = "";
    return;
  }

  const parts = entries.slice(0, 4).map(([reason, count]) => `${finishLabel(reason)} ${int(count)}`);
  if (entries.length > 4) parts.push(`+${entries.length - 4} more`);
  note.textContent = `all time finished: ${parts.join(" · ")}`;
}

function renderHistory(data) {
  const hourly = data.history.hourly.slice(-HOURLY_BARS);
  const daily = data.history.daily;
  const hourFigure = $("chart-hour");
  const dayFigure = $("chart-day");
  const tracking = $("tracking");

  const since = data.server.started_at
    ? `tracking since ${dayLabel(data.server.started_at)} ${hourLabel(data.server.started_at)}`
    : "tracking starts with this run";
  tracking.textContent = `${since} · ${int(data.server.sessions)} server run(s)`;

  if (!hourly.length) {
    hourFigure.hidden = true;
    dayFigure.hidden = true;
    tracking.textContent += " · no history recorded yet";
    return;
  }

  hourFigure.hidden = false;
  renderBars($("hour-svg"), hourly, "gen_tokens", 720, 110);
  renderAxis($("hour-axis"), [
    hourLabel(hourly[0].hour),
    hourLabel(hourly[Math.floor(hourly.length / 2)].hour),
    "now",
  ]);

  dayFigure.hidden = !daily.length;
  if (daily.length) {
    renderBars($("day-svg"), daily, "gen_tokens", 720, 70);
    renderAxis($("day-axis"), [
      dayLabel(daily[0].day),
      dayLabel(daily[Math.floor(daily.length / 2)].day),
      dayLabel(daily[daily.length - 1].day),
    ]);
  }
}

function renderModels(data) {
  const band = $("band-models");
  band.hidden = !data.models.length;
  if (!data.models.length) return;

  const body = $("models-body");
  body.replaceChildren();

  for (const model of data.models) {
    body.append(
      row([
        td(model.name, "mono"),
        td(int(model.loads), "num opt"),
        td(int(model.requests), "num"),
        td(int(model.gen_tokens), "num"),
        td(int(model.processed_tokens), "num opt"),
        td(model.last_loaded ? clock(model.last_loaded) : "—", "num opt mono"),
      ])
    );
  }
}

function renderRecent(data) {
  const band = $("band-recent");
  const recent = data.recent || [];
  band.hidden = !recent.length;
  if (!recent.length) return;

  // Only worth a column when more than one model has served requests
  const showModel = data.models.length > 1;
  $("recent-model-head").hidden = !showModel;

  const body = $("recent-body");
  body.replaceChildren();

  for (const entry of recent.slice(0, MAX_RECENT_ROWS)) {
    const modelCell = td(entry.model || "—", "mono dim");
    modelCell.hidden = !showModel;

    body.append(
      row([
        td(clock(entry.time), "mono dim"),
        td(entry.label || "—", "mono"),
        modelCell,
        td(`${int(entry.processed_tokens)} / ${int(entry.prompt_tokens)}`, "num opt"),
        td(int(entry.gen_tokens), "num"),
        td(rate(entry.gen_tokens_per_sec), "num"),
        td(finishLabel(entry.finish), "label")
      ])
    );
  }

  const note = $("recent-note");
  note.textContent =
    recent.length > MAX_RECENT_ROWS
      ? `showing the latest ${MAX_RECENT_ROWS} of ${int(recent.length)} kept`
      : "";
}

function renderIncidents(data) {
  const band = $("band-incidents");
  band.hidden = !data.incidents.length;
  if (!data.incidents.length) return;

  const list = $("incidents");
  list.replaceChildren();
  for (const incident of data.incidents) {
    const item = document.createElement("li");
    const time = el("time", `${clock(incident.time)}`, "mono");
    time.dateTime = new Date(incident.time * 1000).toISOString();
    item.append(time, el("span", incident.description));
    list.append(item);
  }
}

// The sparkline covers the last few minutes of this browser's readings, so a
// reload starts it over
function renderStateSamples(value) {
  sparkSamples.push(value);
  if (sparkSamples.length > SPARK_SAMPLES) sparkSamples.shift();
  renderSpark(sparkSamples);
}

function renderAll(data) {
  renderHeader(data);
  renderHealth(data);
  renderLive(data);
  renderGpu(data);
  renderTotals(data);
  renderHistory(data);
  renderModels(data);
  renderRecent(data);
  renderIncidents(data);
}

// ---------------------------------------------------------------------------
// Polling

let timer = null;
let failures = 0;
let lastReading = 0;
let needsKey = false;

async function fetchStats() {
  const headers = {};
  if (settings.apiKey) headers.Authorization = `Bearer ${settings.apiKey}`;

  const response = await fetch(apiUrl("/v1/stats"), { headers });
  if (!response.ok) {
    const error = new Error(`${response.status} ${response.statusText}`);
    error.status = response.status;
    throw error;
  }

  return response.json();
}

function schedule(delay) {
  clearTimeout(timer);
  timer = setTimeout(tick, delay);
}

async function tick() {
  if (needsKey) return;

  try {
    const data = await fetchStats();
    failures = 0;
    lastReading = Date.now();
    setAlert("");
    renderAll(data);
    schedule(POLL_MS);
  } catch (error) {
    if (error.status === 401 || error.status === 403) {
      needsKey = true;
      setState("bad", "key required");
      setAlert("This server needs an API key to read its statistics.");
      $("gate").hidden = false;
      $("gate-key").focus();
      return;
    }

    failures += 1;
    setState("warn", "disconnected");
    const seen = lastReading ? ` Last reading was at ${clock(lastReading / 1000)}.` : "";
    setAlert(`Cannot reach the server: ${error.message}.${seen}`);
    schedule(Math.min(POLL_MS * 2 ** failures, MAX_BACKOFF_MS));
  }
}

$("gate").addEventListener("submit", (event) => {
  event.preventDefault();
  const key = $("gate-key").value.trim();
  if (!key) return;

  settings.apiKey = key;
  writeSettings({ apiKey: key });
  needsKey = false;
  failures = 0;
  $("gate").hidden = true;
  setAlert("");
  tick();
});

document.addEventListener("visibilitychange", () => {
  // No polling for a page nobody is looking at
  if (document.hidden) clearTimeout(timer);
  else tick();
});

takeKeyFromLink();
tick();