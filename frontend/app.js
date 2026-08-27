const DIMENSIONS_META = [
  { key: "completeness", weightKey: "completeness", label: "Completeness", defaultWeight: 0.3 },
  { key: "correctness", weightKey: "correctness", label: "Correctness", defaultWeight: 0.4 },
  { key: "supportedContent", weightKey: "supported_content", label: "Supported Content", defaultWeight: 0.2 },
  { key: "sectionPlacement", weightKey: "section_placement", label: "Section Placement", defaultWeight: 0.1 },
];

const SEVERITY_CLASS = {
  CRITICAL: "danger",
  HIGH: "danger",
  MODERATE: "warning",
  LOW: "neutral",
  NONE: "neutral",
};

const SEVERITY_ORDER = {
  CRITICAL: 4,
  HIGH: 3,
  MODERATE: 2,
  LOW: 1,
  NONE: 0,
};

function sortBySeverity(events) {
  return [...events].sort((a, b) => (SEVERITY_ORDER[b.severity] ?? 0) - (SEVERITY_ORDER[a.severity] ?? 0));
}

const CLASSIFICATION_META = {
  CORRECT: { cls: "success", label: "Correct" },
  PARTIAL: { cls: "warning", label: "Partial" },
  MISSING: { cls: "missing", label: "Missing" },
  INCORRECT: { cls: "danger", label: "Incorrect" },
  CONTRADICTION: { cls: "danger", label: "Contradiction" },
  UNSUPPORTED: { cls: "warning", label: "Unsupported" },
};

const eventList = document.querySelector("#eventList");
const weightedRows = document.querySelector("#weightedRows");
const tfootScore = document.querySelector("#tfootScore");
const overallScore = document.querySelector("#overallScore");
const scoreStatus = document.querySelector("#scoreStatus");
const scoreMeterFill = document.querySelector("#scoreMeterFill");
const themeToggle = document.querySelector("#themeToggle");
const evaluateBtn = document.querySelector("#evaluateBtn");
const evaluateError = document.querySelector("#evaluateError");
const evaluateHint = document.querySelector("#evaluateHint");
const groundTruthInput = document.querySelector("#groundTruthInput");
const generatedInput = document.querySelector("#generatedInput");
const statGroundTruth = document.querySelector("#statGroundTruth");
const statGenerated = document.querySelector("#statGenerated");
const statEvents = document.querySelector("#statEvents");
const factCorrect = document.querySelector("#factCorrect");
const factPartial = document.querySelector("#factPartial");
const factMissing = document.querySelector("#factMissing");
const factIncorrect = document.querySelector("#factIncorrect");
const factContradictions = document.querySelector("#factContradictions");
const completenessFormula = document.querySelector("#completenessFormula");
const completenessValue = document.querySelector("#completenessValue");
const correctnessFormula = document.querySelector("#correctnessFormula");
const correctnessValue = document.querySelector("#correctnessValue");
const factsTableBody = document.querySelector("#factsTableBody");
const factsFilter = document.querySelector("#factsFilter");

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let dimensions = buildDimensions(null, null);
let currentScore = 0;
let lastFactMatches = [];

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
  ));
}

function buildDimensions(scores, weights) {
  return DIMENSIONS_META.map((meta) => ({
    label: meta.label,
    value: scores ? scores[meta.key] : 0,
    weight: weights && weights[meta.weightKey] != null ? weights[meta.weightKey] : meta.defaultWeight,
  }));
}

function animateValue(from, to, duration, onUpdate) {
  if (prefersReducedMotion || duration <= 0) {
    onUpdate(to);
    return;
  }
  const start = performance.now();
  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    onUpdate(from + (to - from) * eased);
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

function scoreTier(score) {
  if (score >= 85) return { label: "Strong", cls: "success" };
  if (score >= 65) return { label: "Moderate", cls: "warning" };
  return { label: "Needs review", cls: "danger" };
}

function setScore(score, mode = "done") {
  currentScore = score;
  overallScore.textContent = mode === "idle" ? "––" : score.toFixed(2);
  scoreMeterFill.style.setProperty("--score", score);
  tfootScore.textContent = mode === "idle" ? "–" : score.toFixed(2);

  if (mode === "idle") {
    scoreStatus.textContent = "Not run";
    scoreStatus.className = "score-status";
    return;
  }
  if (mode === "loading") {
    scoreStatus.textContent = "Evaluating…";
    scoreStatus.className = "score-status";
    return;
  }
  const tier = scoreTier(score);
  scoreStatus.textContent = tier.label;
  scoreStatus.className = `score-status ${tier.cls}`;
}

function renderWeightedRows() {
  const maxContribution = Math.max(...dimensions.map((item) => item.weight)) * 100;
  weightedRows.innerHTML = dimensions
    .map((item) => {
      const contribution = item.value * item.weight;
      const contributionPercent = maxContribution ? (contribution / maxContribution) * 100 : 0;
      return `
        <tr>
          <td>${item.label}</td>
          <td>${item.value.toFixed(1)}</td>
          <td>${Math.round(item.weight * 100)}%</td>
          <td>
            <div class="contribution-cell">
              <div class="contribution-track">
                <span style="--contribution: ${contributionPercent}"></span>
              </div>
              <strong>${contribution.toFixed(2)}</strong>
            </div>
          </td>
        </tr>
      `;
    })
    .join("");
}

function renderOverview(data) {
  statGroundTruth.textContent = data ? data.counts.groundTruthFacts : "–";
  statGenerated.textContent = data ? data.counts.generatedFacts : "–";
  statEvents.textContent = data ? data.counts.clinicalErrorEvents : "–";
}

function formatEventType(type) {
  return type.replaceAll("_", " ");
}

function renderEvents(events) {
  if (!events || events.length === 0) {
    eventList.innerHTML = `<p class="muted">No clinical error events were detected for this note pair.</p>`;
    return;
  }
  eventList.innerHTML = events
    .map((event) => {
      const cls = SEVERITY_CLASS[event.severity] || "neutral";
      return `
        <div class="event event-${cls}">
          <h3>
            <span>${escapeHtml(formatEventType(event.type))}</span>
            <span class="badge ${cls}">${escapeHtml(event.severity)}</span>
          </h3>
          <p>${escapeHtml(event.reason)}</p>
        </div>
      `;
    })
    .join("");
}

function renderCalcBreakdown(data) {
  if (!data) {
    factCorrect.textContent = "–";
    factPartial.textContent = "–";
    factMissing.textContent = "–";
    factIncorrect.textContent = "–";
    factContradictions.textContent = "–";
    completenessFormula.textContent = "–";
    completenessValue.textContent = "–";
    correctnessFormula.textContent = "–";
    correctnessValue.textContent = "–";
    return;
  }
  const c = data.counts;
  factCorrect.textContent = c.correct;
  factPartial.textContent = c.partial;
  factMissing.textContent = c.missing;
  factIncorrect.textContent = c.incorrect;
  factContradictions.textContent = c.contradictions;

  const attempted = c.groundTruthFacts - c.missing;
  completenessFormula.textContent = `(${c.correct} + 0.5 x ${c.partial}) / ${c.groundTruthFacts} x 100`;
  completenessValue.textContent = data.scores.completeness.toFixed(1);
  correctnessFormula.textContent = `(${c.correct} + 0.5 x ${c.partial}) / ${attempted} x 100`;
  correctnessValue.textContent = data.scores.correctness.toFixed(1);
}

function renderSectionCell(row) {
  const gt = row.groundTruthSection;
  const gen = row.generatedSection;

  if (gt && gen && gt !== gen) {
    const score = row.sectionScore;
    const meta = score >= 1 ? { cls: "success", label: "OK" } : { cls: "danger", label: "Wrong section" };
    return `
      <div class="section-cell">
        <span class="section-path">${escapeHtml(gt)} → ${escapeHtml(gen)}</span>
        <span class="badge ${meta.cls}" title="Section placement score: ${score}">${escapeHtml(meta.label)}</span>
      </div>
    `;
  }
  return escapeHtml(gt || gen || "—");
}

function renderFactsTable(factMatches) {
  lastFactMatches = factMatches || [];

  if (!factMatches) {
    factsTableBody.innerHTML = `
      <tr class="facts-empty-row">
        <td colspan="6" class="muted">Run an evaluation to see fact-by-fact results.</td>
      </tr>
    `;
    return;
  }

  const filter = factsFilter.value;
  const rows = filter === "all" ? factMatches : factMatches.filter((row) => row.classification === filter);

  if (rows.length === 0) {
    factsTableBody.innerHTML = `
      <tr class="facts-empty-row">
        <td colspan="6" class="muted">No facts match this filter.</td>
      </tr>
    `;
    return;
  }

  factsTableBody.innerHTML = rows
    .map((row) => {
      const meta = CLASSIFICATION_META[row.classification] || { cls: "neutral", label: row.classification };
      return `
        <tr>
          <td><span class="badge ${meta.cls}">${escapeHtml(meta.label)}</span></td>
          <td>${escapeHtml(row.concept)}</td>
          <td>${renderSectionCell(row)}</td>
          <td>${row.groundTruth ? escapeHtml(row.groundTruth) : "—"}</td>
          <td>${row.generated ? escapeHtml(row.generated) : "—"}</td>
          <td>${escapeHtml(row.reason)}</td>
        </tr>
      `;
    })
    .join("");
}

const DEFAULT_HINT_TEXT = evaluateHint.textContent.trim();
const PROGRESS_MESSAGES = [
  "Extracting clinical facts from both notes…",
  "Matching facts by clinical meaning…",
  "Scoring completeness and correctness…",
  "Checking for clinical error events…",
  "Still working — larger notes take longer…",
];

let progressTimer = null;

function startProgress() {
  const start = performance.now();
  evaluateHint.textContent = `${PROGRESS_MESSAGES[0]} (0s)`;
  progressTimer = window.setInterval(() => {
    const elapsed = Math.round((performance.now() - start) / 1000);
    const phase = PROGRESS_MESSAGES[Math.min(Math.floor(elapsed / 5), PROGRESS_MESSAGES.length - 1)];
    evaluateHint.textContent = `${phase} (${elapsed}s)`;
  }, 1000);
}

function stopProgress() {
  if (progressTimer) {
    window.clearInterval(progressTimer);
    progressTimer = null;
  }
  evaluateHint.textContent = DEFAULT_HINT_TEXT;
}

function setLoading(isLoading) {
  evaluateBtn.disabled = isLoading;
  evaluateBtn.dataset.loading = String(isLoading);
  evaluateBtn.querySelector(".btn-label").textContent = isLoading ? "Evaluating…" : "Evaluate";
  if (isLoading) {
    startProgress();
  } else {
    stopProgress();
  }
}

async function runEvaluation() {
  if (evaluateBtn.disabled) return;
  const groundTruth = groundTruthInput.value.trim();
  const generated = generatedInput.value.trim();
  evaluateError.hidden = true;

  if (!groundTruth || !generated) {
    evaluateError.textContent = "Enter both a ground-truth note and a generated note before evaluating.";
    evaluateError.hidden = false;
    return;
  }

  setLoading(true);
  dimensions = buildDimensions(null, null);
  renderWeightedRows();
  setScore(0, "loading");

  try {
    const res = await fetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ groundTruth, generated }),
    });
    const contentType = res.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
      const bodyText = await res.text();
      const looksLikeTimeout = /timeout|timed out|524|gateway/i.test(bodyText);
      throw new Error(
        looksLikeTimeout
          ? "The evaluation took too long and the connection was dropped (likely a tunnel/proxy timeout). Try again — if it keeps happening, the note may be too large for the current setup."
          : `Server returned an unexpected response (status ${res.status}). It may be temporarily unreachable.`
      );
    }
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `Request failed with status ${res.status}`);
    }
    data.events = sortBySeverity(data.events);

    dimensions = buildDimensions(data.scores, data.weights);
    renderOverview(data);
    renderCalcBreakdown(data);
    renderFactsTable(data.factMatches);
    renderEvents(data.events);
    renderWeightedRows();
    animateValue(0, data.overallScore, 700, (value) => setScore(value, "done"));
  } catch (err) {
    evaluateError.textContent = err.message || "Evaluation failed. Check the server console for details.";
    evaluateError.hidden = false;
    setScore(0, "idle");
  } finally {
    setLoading(false);
  }
}

function initTheme() {
  const STORAGE_KEY = "cnfs-theme";
  const root = document.documentElement;

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    themeToggle.setAttribute("aria-pressed", String(theme === "dark"));
    themeToggle.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  }

  const stored = localStorage.getItem(STORAGE_KEY);
  const preferred = stored === "light" || stored === "dark"
    ? stored
    : window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  applyTheme(preferred);

  themeToggle.addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    localStorage.setItem(STORAGE_KEY, next);
    applyTheme(next);
  });
}

function initInputPersistence() {
  const STORAGE_KEYS = { [groundTruthInput.id]: "cnfs-input-ground-truth", [generatedInput.id]: "cnfs-input-generated" };

  [groundTruthInput, generatedInput].forEach((field) => {
    const key = STORAGE_KEYS[field.id];
    const saved = localStorage.getItem(key);
    if (saved !== null) field.value = saved;

    let saveTimer = null;
    field.addEventListener("input", () => {
      window.clearTimeout(saveTimer);
      saveTimer = window.setTimeout(() => localStorage.setItem(key, field.value), 400);
    });
  });
}

function initFieldActions() {
  document.querySelectorAll("[data-reset-for]").forEach((button) => {
    const field = document.querySelector(`#${button.dataset.resetFor}`);
    button.addEventListener("click", () => {
      field.value = field.defaultValue;
      field.dispatchEvent(new Event("input"));
      field.focus();
    });
  });

  document.querySelectorAll("[data-clear-for]").forEach((button) => {
    const field = document.querySelector(`#${button.dataset.clearFor}`);
    button.addEventListener("click", () => {
      field.value = "";
      field.dispatchEvent(new Event("input"));
      field.focus();
    });
  });
}

function initKeyboardShortcut() {
  [groundTruthInput, generatedInput].forEach((field) => {
    field.addEventListener("keydown", (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        event.preventDefault();
        runEvaluation();
      }
    });
  });
}

evaluateBtn.addEventListener("click", runEvaluation);
factsFilter.addEventListener("change", () => renderFactsTable(lastFactMatches.length ? lastFactMatches : null));

initTheme();
initInputPersistence();
initFieldActions();
initKeyboardShortcut();
renderWeightedRows();
renderOverview(null);
renderEvents(null);
renderCalcBreakdown(null);
renderFactsTable(null);
setScore(0, "idle");
