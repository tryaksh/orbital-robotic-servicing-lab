(() => {
  "use strict";

  const API_ROOT = "/api";
  const TERMINAL_STATUSES = new Set(["succeeded", "failed", "cancelled", "canceled"]);
  const PHASES = ["preflight", "perception", "planning", "capture", "extract", "transit", "insert", "verification"];
  const DEFAULT_SEED = 4070;

  const state = {
    health: null,
    capabilities: null,
    jobs: [],
    selectedJob: null,
    events: [],
    eventAfter: 0,
    summary: null,
    summaryKey: null,
    pollTimer: null,
    healthTimer: null,
    elapsedTimer: null,
    refreshing: false,
    submitting: false,
    serviceError: null,
    capabilityError: null,
  };

  const dom = {};

  class ApiError extends Error {
    constructor(message, status, payload) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.payload = payload;
    }
  }

  function cacheDom() {
    const ids = [
      "serviceDot", "serviceStatus", "serviceLatency", "environmentBadge", "boundaryText",
      "jobForm", "presetGrid", "resetConfig", "seed", "modeHelp", "runtimeHelp",
      "preflightSummary", "preflightList", "submitJob", "formMessage", "emptyState",
      "runDashboard", "jobPickerWrap", "jobPicker", "jobModeBadge", "jobId", "copyJobId",
      "jobTitle", "jobSubtitle", "jobStatus", "jobElapsed", "progressLabel", "progressBar",
      "progressFill", "phaseTimeline", "phaseMessage", "confidenceBadge", "confidenceRing",
      "confidenceValue", "poseTarget", "poseX", "poseY", "poseZ", "poseOrientation",
      "poseSource", "rackPlan", "planGate", "bay0Score", "bay1Score", "telemetryState",
      "forceValue", "forceBar", "forceLimitMarker", "forceLimit",
      "torqueValue", "torqueBar", "torqueLimitMarker", "torqueLimit", "impulseValue",
      "impulseBar", "impulseLimitMarker", "poseErrorValue", "poseErrorBar",
      "poseErrorLimitMarker", "poseErrorLimit", "gatePanel", "gateTitle", "gateSummary",
      "gateResultPath", "successRate", "trialCount", "gateThreshold", "streamState",
      "eventList", "artifactCount", "artifactList", "cancelJob", "refreshJob",
      "limitationGrid", "buildLabel", "toastRegion",
    ];
    for (const id of ids) {
      dom[id] = document.getElementById(id);
    }
  }

  async function api(path, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), options.timeout || 10000);
    const request = {
      method: options.method || "GET",
      cache: "no-store",
      headers: { Accept: "application/json", ...(options.headers || {}) },
      signal: controller.signal,
    };
    if (options.body !== undefined) {
      request.headers["Content-Type"] = "application/json";
      request.body = typeof options.body === "string" ? options.body : JSON.stringify(options.body);
    }

    let response;
    try {
      response = await fetch(`${API_ROOT}${path}`, request);
    } catch (error) {
      if (error && error.name === "AbortError") {
        throw new ApiError("The compute service did not respond in time.", 0, null);
      }
      throw new ApiError("The compute service is unreachable.", 0, null);
    } finally {
      window.clearTimeout(timeout);
    }

    const contentType = response.headers.get("content-type") || "";
    let payload = null;
    if (response.status !== 204) {
      if (contentType.includes("application/json")) {
        payload = await response.json().catch(() => null);
      } else {
        payload = await response.text().catch(() => "");
      }
    }
    if (!response.ok) {
      const detail = payload && typeof payload === "object" ? payload.detail : null;
      const message =
        (detail && typeof detail === "object" && detail.message) ||
        (typeof detail === "string" && detail) ||
        (payload && typeof payload === "object" && payload.message) ||
        (typeof payload === "string" && payload.slice(0, 240)) ||
        `Request failed (${response.status}).`;
      throw new ApiError(message, response.status, payload);
    }
    return payload;
  }

  function text(node, value) {
    if (node) node.textContent = value == null ? "—" : String(value);
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function finite(value) {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value))) {
      return Number(value);
    }
    return null;
  }

  function normalizeFraction(value) {
    const number = finite(value);
    if (number == null) return null;
    return number > 1 && number <= 100 ? number / 100 : number;
  }

  function humanize(value) {
    return String(value || "unknown")
      .replace(/[_-]+/g, " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function compactId(value) {
    const id = String(value || "");
    return id ? id.slice(0, 8) : "—";
  }

  function formatDuration(totalSeconds) {
    const seconds = Math.max(0, Math.floor(totalSeconds || 0));
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainder = seconds % 60;
    if (hours) return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
    return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
  }

  function formatBytes(bytes) {
    const value = finite(bytes);
    if (value == null) return "size unavailable";
    if (value < 1024) return `${value} B`;
    const units = ["KB", "MB", "GB"];
    let scaled = value / 1024;
    let index = 0;
    while (scaled >= 1024 && index < units.length - 1) {
      scaled /= 1024;
      index += 1;
    }
    return `${scaled.toFixed(scaled >= 10 ? 1 : 2)} ${units[index]}`;
  }

  function formatMetric(value, decimals = 2) {
    const number = finite(value);
    if (number == null) return "—";
    return number.toFixed(decimals).replace(/\.00$/, "");
  }

  function toast(message, kind = "info") {
    if (!dom.toastRegion) return;
    const item = document.createElement("div");
    item.className = `toast${kind === "error" ? " is-error" : kind === "warning" ? " is-warning" : ""}`;
    item.textContent = message;
    dom.toastRegion.appendChild(item);
    window.setTimeout(() => item.remove(), 5000);
  }

  function selectedPresetId() {
    const selected = dom.jobForm?.querySelector('input[name="preset"]:checked');
    return selected ? selected.value : null;
  }

  function selectedCapability() {
    const id = selectedPresetId();
    const presets = state.capabilities?.presets;
    return Array.isArray(presets) ? presets.find((preset) => preset.id === id) || null : null;
  }

  function setBadge(node, mode) {
    if (!node) return;
    node.classList.remove("badge-simulation", "badge-replay", "badge-live");
    if (mode === "replay") {
      node.classList.add("badge-replay");
      text(node, "RECORDED REPLAY");
    } else {
      node.classList.add("badge-live");
      text(node, "LIVE SIMULATION");
    }
  }

  function updatePresetSelection() {
    const id = selectedPresetId();
    for (const card of dom.presetGrid.querySelectorAll(".preset-card")) {
      card.classList.toggle("is-selected", card.dataset.presetCard === id);
    }
    const capability = selectedCapability();
    const backend = capability?.backend || (id === "replay_full_chain" ? "replay" : "isaac");
    setBadge(dom.environmentBadge, backend);
    if (backend === "replay") {
      text(dom.modeHelp, "Recorded deterministic replay");
      text(dom.boundaryText, "Representative telemetry replay. It exercises the real service contract but does not launch a simulator or produce a new physical result.");
    } else {
      text(dom.modeHelp, "Live Isaac Sim subprocess");
      text(dom.boundaryText, "Live digital-twin execution, not physical hardware. Results remain simulation evidence and are not flight or safety certification.");
    }
    const seconds = finite(capability?.estimated_runtime_s);
    text(dom.runtimeHelp, seconds == null ? "Runtime estimate unavailable" : `Estimated runtime: ${seconds < 60 ? `~${seconds} seconds` : `~${Math.ceil(seconds / 60)} minutes`}`);
    validatePreflight();
  }

  function renderCapabilities() {
    const presets = Array.isArray(state.capabilities?.presets) ? state.capabilities.presets : [];
    for (const card of dom.presetGrid.querySelectorAll(".preset-card")) {
      const capability = presets.find((item) => item.id === card.dataset.presetCard);
      const input = card.querySelector('input[type="radio"]');
      const tag = card.querySelector('[data-role="tag"]');
      const titleNode = card.querySelector('[data-role="title"]');
      const description = card.querySelector('[data-role="description"]');
      const meta = card.querySelector('[data-role="meta"]');
      if (!capability) {
        if (state.capabilities) {
          input.disabled = true;
          card.classList.add("is-unavailable");
          card.setAttribute("aria-disabled", "true");
          text(tag, "NOT ADVERTISED");
          text(meta, "Unavailable in this service build");
        }
        continue;
      }
      text(titleNode, capability.title);
      text(description, capability.description);
      input.disabled = !capability.available;
      card.classList.toggle("is-unavailable", !capability.available);
      card.setAttribute("aria-disabled", String(!capability.available));
      const reasons = Array.isArray(capability.unavailable_reasons) ? capability.unavailable_reasons : [];
      card.title = capability.available ? "" : reasons.join(" · ");
      tag.classList.remove("tag-ready", "tag-investigate", "tag-system");
      if (capability.available) {
        tag.classList.add(capability.backend === "replay" ? "tag-system" : "tag-ready");
        text(tag, capability.backend === "replay" ? "REPLAY READY" : "GPU READY");
      } else {
        tag.classList.add("tag-investigate");
        text(tag, "UNAVAILABLE");
      }
      const runtime = finite(capability.estimated_runtime_s);
      const runtimeLabel = runtime == null ? "runtime unknown" : runtime < 60 ? `~${runtime} s` : `~${Math.ceil(runtime / 60)} min`;
      const boundary = capability.backend === "replay" ? "recorded replay" : "live simulation";
      text(meta, capability.available ? `${runtimeLabel} · ${boundary}${capability.produces_video ? " · video" : ""}` : reasons[0] || "Missing required runtime capability");
    }

    const selected = dom.jobForm.querySelector('input[name="preset"]:checked');
    if (!selected || selected.disabled) {
      const firstAvailable = dom.jobForm.querySelector('input[name="preset"]:not(:disabled)');
      if (firstAvailable) firstAvailable.checked = true;
    }
    if (state.capabilities?.service_version) {
      text(dom.buildLabel, `Service v${state.capabilities.service_version} · API contract v1`);
    }
    updatePresetSelection();
  }

  function setCheck(name, status, detail) {
    const item = dom.preflightList.querySelector(`[data-check="${name}"]`);
    if (!item) return;
    const icon = item.querySelector(".check-icon");
    const small = item.querySelector("small");
    icon.className = `check-icon is-${status}`;
    text(small, detail);
    item.dataset.status = status;
  }

  function validatePreflight() {
    const healthReady = state.health?.status === "ok";
    setCheck("api", healthReady ? "pass" : state.serviceError ? "fail" : "pending", healthReady ? `API v${state.health.version || "?"} responding` : state.serviceError || "Waiting for health response");

    const workerState = state.health?.worker_state;
    let workerStatus = "pending";
    let workerDetail = "Capability not loaded";
    if (workerState) {
      workerStatus = workerState === "idle" || workerState === "busy" ? "pass" : "fail";
      if (workerState === "busy") {
        const depth = finite(state.health.queue_depth) || 0;
        workerDetail = `Busy · serialized queue ${depth}`;
      } else {
        workerDetail = workerState === "idle" ? "Idle and ready" : workerState === "failed" ? "Serialized worker failed" : "Service is stopping";
      }
    }
    setCheck("worker", workerStatus, workerDetail);

    const capability = selectedCapability();
    let perceptionStatus = "pending";
    let perceptionDetail = "Capability not loaded";
    if (state.capabilities) {
      if (!capability) {
        perceptionStatus = "fail";
        perceptionDetail = "Preset not advertised";
      } else if (!capability.available) {
        perceptionStatus = "fail";
        perceptionDetail = capability.unavailable_reasons?.[0] || "Preset unavailable";
      } else if (!capability.perception) {
        perceptionStatus = "fail";
        perceptionDetail = "Preset has no perception stage";
      } else {
        perceptionStatus = "pass";
        perceptionDetail = capability.backend === "replay" ? "Schema fixture; no model inference" : "Rendered pose estimator";
      }
    } else if (state.capabilityError) {
      perceptionStatus = "fail";
      perceptionDetail = state.capabilityError;
    }
    setCheck("perception", perceptionStatus, perceptionDetail);

    const seed = finite(dom.seed.value);
    const seedValid = Number.isInteger(seed) && seed >= 0 && seed <= 2147483647;
    const configValid = Boolean(selectedPresetId()) && seedValid && (!capability || capability.available);
    setCheck("config", configValid ? "pass" : "fail", !seedValid ? "Seed must be an integer from 0 to 2147483647" : capability && !capability.available ? "Selected preset is unavailable" : "Fixed preset contract is valid");

    const checks = Array.from(dom.preflightList.querySelectorAll("li"));
    const passCount = checks.filter((item) => item.dataset.status === "pass").length;
    const allReady = passCount === checks.length;
    text(dom.preflightSummary, allReady ? "READY TO QUEUE" : `${passCount}/${checks.length} READY`);
    dom.submitJob.disabled = !allReady || state.submitting;
  }

  function renderServiceHealth(latencyMs) {
    const healthy = state.health?.status === "ok";
    dom.serviceDot.className = `status-dot ${healthy ? "is-online" : state.serviceError ? "is-offline" : "is-checking"}`;
    text(dom.serviceStatus, healthy ? `${humanize(state.health.worker_state)}${state.health.worker_state === "busy" ? " worker" : ""}` : state.serviceError ? "Offline" : "Connecting");
    text(dom.serviceLatency, healthy && latencyMs != null ? `${Math.round(latencyMs)} ms` : "— ms");
    validatePreflight();
  }

  async function checkHealth({ quiet = true } = {}) {
    const started = performance.now();
    try {
      state.health = await api("/health", { timeout: 5000 });
      state.serviceError = null;
      renderServiceHealth(performance.now() - started);
    } catch (error) {
      state.health = null;
      state.serviceError = error.message;
      renderServiceHealth(null);
      if (!quiet) toast(error.message, "error");
    }
  }

  async function loadCapabilities() {
    try {
      state.capabilities = await api("/capabilities");
      state.capabilityError = null;
    } catch (error) {
      state.capabilities = null;
      state.capabilityError = error.message;
    }
    renderCapabilities();
  }

  function mergeJob(job) {
    if (!job?.id) return;
    const index = state.jobs.findIndex((item) => item.id === job.id);
    if (index === -1) state.jobs.unshift(job);
    else state.jobs[index] = job;
    state.jobs.sort((left, right) => Date.parse(right.created_at || 0) - Date.parse(left.created_at || 0));
  }

  function renderJobPicker() {
    const hasJobs = state.jobs.length > 0;
    dom.jobPickerWrap.hidden = !hasJobs;
    dom.jobPicker.replaceChildren();
    for (const job of state.jobs.slice(0, 30)) {
      const option = document.createElement("option");
      option.value = job.id;
      option.textContent = `${String(job.status || "unknown").toUpperCase()} · ${job.preset_title || humanize(job.preset_id)} · ${compactId(job.id)}`;
      option.selected = job.id === state.selectedJob?.id;
      dom.jobPicker.appendChild(option);
    }
  }

  async function loadJobs({ selectNewest = false } = {}) {
    try {
      const response = await api("/jobs?limit=30");
      state.jobs = Array.isArray(response) ? response : Array.isArray(response?.jobs) ? response.jobs : [];
      renderJobPicker();
      if (selectNewest && !state.selectedJob && state.jobs[0]) {
        await selectJob(state.jobs[0]);
      }
    } catch (error) {
      if (!state.selectedJob) {
        text(dom.formMessage, `Could not load job history: ${error.message}`);
      }
    }
  }

  function showDashboard(show) {
    dom.emptyState.hidden = show;
    dom.runDashboard.hidden = !show;
  }

  function backendForJob(job) {
    return job?.provenance?.backend || (String(job?.preset_id || "").startsWith("replay") ? "replay" : "isaac");
  }

  function statusMessage(job) {
    const stage = humanize(job.current_stage || job.status);
    if (job.status === "queued") return `Seed ${job.seed} · Waiting for the serialized worker.`;
    if (job.status === "running") return `Seed ${job.seed} · Active stage: ${stage}.`;
    if (job.status === "succeeded") {
      return backendForJob(job) === "replay"
        ? `Seed ${job.seed} · Replay completed; no simulation or qualification evidence was produced.`
        : `Seed ${job.seed} · Execution completed; inspect the outcome and qualification status below.`;
    }
    if (job.status === "failed") return job.error || "The execution backend failed before completion.";
    if (job.status === "cancelled" || job.status === "canceled") return "Execution was cancelled before completion.";
    return `Seed ${job.seed} · ${stage}.`;
  }

  function renderElapsed() {
    const job = state.selectedJob;
    if (!job) {
      text(dom.jobElapsed, "00:00");
      return;
    }
    const start = Date.parse(job.started_at || job.created_at || "");
    const finish = Date.parse(job.finished_at || "");
    const end = Number.isFinite(finish) ? finish : Date.now();
    text(dom.jobElapsed, Number.isFinite(start) ? formatDuration((end - start) / 1000) : "—");
  }

  function stageIndex(stage) {
    const normalized = String(stage || "").toLowerCase();
    if (["queued", "starting", "startup", "preflight"].includes(normalized)) return 0;
    if (normalized.includes("perception") || normalized.includes("localiz")) return 1;
    if (normalized.includes("planning") || normalized.includes("plan")) return 2;
    if (normalized.includes("capture") || normalized.includes("seat") || normalized.includes("grasp")) return 3;
    if (normalized.includes("extract")) return 4;
    if (normalized.includes("transit") || normalized.includes("transfer") || normalized.includes("relocat")) return 5;
    if (normalized.includes("insert") || normalized.includes("install") || normalized.includes("evaluation")) return 6;
    if (["verification", "verify", "done", "succeeded", "completed"].some((name) => normalized.includes(name))) return 7;
    return 0;
  }

  function latestEvent() {
    return state.events.length ? state.events[state.events.length - 1] : null;
  }

  function renderPhases(job) {
    const status = String(job.status || "queued").toLowerCase();
    const terminal = TERMINAL_STATUSES.has(status);
    let progress = finite(job.progress);
    progress = progress == null ? 0 : progress > 1 ? progress / 100 : progress;
    if (status === "succeeded") progress = 1;
    progress = clamp(progress, 0, 1);
    text(dom.progressLabel, `${Math.round(progress * 100)}%`);
    dom.progressFill.style.width = `${progress * 100}%`;
    dom.progressBar.setAttribute("aria-valuenow", String(Math.round(progress * 100)));

    const currentIndex = stageIndex(job.current_stage);
    for (const [index, node] of Array.from(dom.phaseTimeline.children).entries()) {
      node.classList.remove("is-complete", "is-active", "is-failed", "is-cancelled");
      if (status === "succeeded" || index < currentIndex) node.classList.add("is-complete");
      else if (!terminal && index === currentIndex) node.classList.add("is-active");
      else if (terminal && status === "failed" && index === currentIndex) node.classList.add("is-failed");
      else if (terminal && ["cancelled", "canceled"].includes(status) && index === currentIndex) node.classList.add("is-cancelled");
    }

    const event = latestEvent();
    if (event?.message) text(dom.phaseMessage, event.message);
    else if (status === "queued") text(dom.phaseMessage, "Job accepted; waiting for the serialized worker.");
    else text(dom.phaseMessage, statusMessage(job));
  }

  function walkObjects(value, visit, depth = 0) {
    if (!value || typeof value !== "object" || depth > 5) return;
    visit(value);
    for (const nested of Object.values(value)) {
      if (nested && typeof nested === "object") walkObjects(nested, visit, depth + 1);
    }
  }

  function canonicalKey(key) {
    return String(key).toLowerCase().replace(/[^a-z0-9]/g, "");
  }

  function findValues(sources, keys) {
    const wanted = new Set(keys.map(canonicalKey));
    const values = [];
    for (const source of sources) {
      walkObjects(source, (object) => {
        for (const [key, value] of Object.entries(object)) {
          if (wanted.has(canonicalKey(key))) values.push(value);
        }
      });
    }
    return values;
  }

  function latestNumeric(sources, keys) {
    const values = findValues(sources, keys);
    for (let index = values.length - 1; index >= 0; index -= 1) {
      const number = finite(values[index]);
      if (number != null) return number;
    }
    return null;
  }

  function maximumNumeric(sources, keys) {
    const values = findValues(sources, keys).map(finite).filter((value) => value != null);
    return values.length ? Math.max(...values) : null;
  }

  function latestBoolean(sources, keys) {
    const values = findValues(sources, keys);
    for (let index = values.length - 1; index >= 0; index -= 1) {
      if (typeof values[index] === "boolean") return values[index];
    }
    return null;
  }

  function latestString(sources, keys) {
    const values = findValues(sources, keys);
    for (let index = values.length - 1; index >= 0; index -= 1) {
      if (typeof values[index] === "string" && values[index].trim()) return values[index].trim();
    }
    return null;
  }

  function latestObject(sources, keys) {
    const values = findValues(sources, keys);
    for (let index = values.length - 1; index >= 0; index -= 1) {
      if (values[index] && typeof values[index] === "object" && !Array.isArray(values[index])) return values[index];
    }
    return null;
  }

  function vectorFrom(value) {
    if (Array.isArray(value) && value.length >= 3) {
      const vector = value.slice(0, 3).map(finite);
      return vector.every((item) => item != null) ? vector : null;
    }
    if (value && typeof value === "object") {
      const vector = [finite(value.x ?? value.X), finite(value.y ?? value.Y), finite(value.z ?? value.Z)];
      return vector.every((item) => item != null) ? vector : null;
    }
    return null;
  }

  function pairFrom(value) {
    if (!Array.isArray(value) || value.length < 2) return null;
    const pair = value.slice(0, 2).map(finite);
    return pair.every((item) => item != null) ? pair : null;
  }

  function quaternionFrom(value) {
    if (Array.isArray(value) && value.length >= 4) {
      const quaternion = value.slice(0, 4).map(finite);
      return quaternion.every((item) => item != null) ? quaternion : null;
    }
    if (value && typeof value === "object") {
      const quaternion = [finite(value.w ?? value.W), finite(value.x ?? value.X), finite(value.y ?? value.Y), finite(value.z ?? value.Z)];
      return quaternion.every((item) => item != null) ? quaternion : null;
    }
    return null;
  }

  function extractPose(sources) {
    const positionKeys = new Set(["positionm", "position", "translationm", "translation", "xyzm", "xyz", "modulepositionm"]);
    const orientationKeys = new Set(["quaternion", "quaternionwxyz", "orientation", "orientationwxyz", "rotationquaternion"]);
    let position = null;
    let quaternion = null;
    for (let sourceIndex = sources.length - 1; sourceIndex >= 0 && (!position || !quaternion); sourceIndex -= 1) {
      walkObjects(sources[sourceIndex], (object) => {
        for (const [key, value] of Object.entries(object)) {
          const normalized = canonicalKey(key);
          if (!position && positionKeys.has(normalized)) position = vectorFrom(value);
          if (!quaternion && orientationKeys.has(normalized)) quaternion = quaternionFrom(value);
        }
        if (!position && object.x_m !== undefined && object.y_m !== undefined && object.z_m !== undefined) {
          const vector = vectorFrom({ x: object.x_m, y: object.y_m, z: object.z_m });
          if (vector) position = vector;
        }
      });
    }
    return {
      position,
      quaternion,
      confidence: normalizeFraction(latestNumeric(sources, ["pose_confidence", "detection_confidence", "confidence"])),
      source: latestString(sources, ["pose_source", "perception_source", "sensor_source", "source"]),
    };
  }

  function extractEvidence() {
    const eventData = state.events.map((event) => event.data || {});
    const result = state.selectedJob?.result || state.summary?.result || null;
    const sources = [...eventData];
    if (result) sources.push(result);
    if (state.summary) sources.push(state.summary);
    const hasPerceptionEvent = state.events.some((event) => String(event.stage || "").toLowerCase().includes("perception"));
    const qualification = result?.qualification || null;
    const hasTypedQualification = qualification !== null && typeof qualification === "object";
    const planning = result?.planning || latestObject(eventData, ["planning"]);
    const telemetry = result?.telemetry || null;
    const limits = telemetry?.applicable_limits || null;
    const eventLimits = latestObject(eventData, ["applicable_limits"]);
    const pose = extractPose(sources);
    if (typeof result?.perception?.source === "string" && result.perception.source.trim()) {
      pose.source = result.perception.source.trim();
    }
    return {
      pose,
      hasPerceptionEvent,
      planning: {
        scores: pairFrom(planning?.initial_occupancy_scores ?? planning?.initial_bay_occupancy_scores ?? planning?.occupancy_scores),
        passed: typeof planning?.gate_passed === "boolean" ? planning.gate_passed : typeof planning?.source_occupied_destination_clear === "boolean" ? planning.source_occupied_destination_clear : null,
        threshold: finite(planning?.decision_threshold),
      },
      force: finite(telemetry?.peak_force_n) ?? maximumNumeric(sources, ["peak_force_n", "peak_contact_force_n", "max_contact_force_n", "contact_force_n"]),
      forceLimit: finite(limits?.force_n) ?? finite(eventLimits?.force_n) ?? latestNumeric(sources, ["force_limit_n", "max_force_n", "contact_force_limit_n"]),
      torque: finite(telemetry?.peak_torque_nm) ?? maximumNumeric(sources, ["peak_torque_nm", "peak_joint_torque_nm", "max_torque_nm", "drive_torque_nm"]),
      torqueLimit: finite(limits?.torque_nm) ?? finite(eventLimits?.torque_nm) ?? latestNumeric(sources, ["torque_limit_nm", "max_allowed_torque_nm", "joint_torque_limit_nm"]),
      impulse: maximumNumeric(sources, ["contact_impulse_ns", "peak_contact_impulse_ns", "impulse_ns"]),
      impulseLimit: latestNumeric(sources, ["contact_impulse_limit_ns", "impulse_limit_ns"]),
      poseError: finite(result?.perception?.pose_error_mm) ?? latestNumeric(sources, ["terminal_pose_error_mm", "pose_error_mm"]),
      poseErrorLimit: latestNumeric(sources, ["pose_error_limit_mm", "max_pose_error_mm"]),
      safetyState: telemetry?.safety_state || latestString(sources, ["safety_state"]),
      // A planning/preflight gate is not a statistical qualification gate.
      // Only qualification-specific keys may color the acceptance panel.
      // Once the API emits the typed qualification object, its nulls are
      // authoritative unknowns. Do not backfill them from a planning gate,
      // episode counter, or another raw report metric.
      passed: hasTypedQualification ? (typeof qualification.passed === "boolean" ? qualification.passed : null) : latestBoolean(sources, ["qualification_passed", "acceptance_passed"]),
      successRate: hasTypedQualification ? normalizeFraction(qualification.success_rate) : normalizeFraction(latestNumeric(sources, ["run_success_rate", "qualification_success_rate", "success_rate"])),
      trials: hasTypedQualification ? finite(qualification.trials) : latestNumeric(sources, ["trials", "episodes_requested", "episode_count", "num_trials"]),
      threshold: hasTypedQualification ? normalizeFraction(qualification.threshold) : normalizeFraction(latestNumeric(sources, ["success_threshold", "gate_threshold", "required_success_rate"])),
      qualificationSummary: hasTypedQualification ? qualification.summary : latestString(sources, ["qualification_summary", "gate_summary", "acceptance_summary"]),
    };
  }

  function renderPerception(evidence) {
    const { pose } = evidence;
    const confidence = pose.confidence == null ? null : clamp(pose.confidence, 0, 1);
    dom.confidenceBadge.className = "confidence-badge";
    if (confidence == null) {
      dom.confidenceBadge.classList.add("is-pending");
      text(dom.confidenceBadge, evidence.hasPerceptionEvent ? "POSE NOT EMITTED" : "NO ESTIMATE");
      text(dom.confidenceValue, "—");
      dom.confidenceRing.style.background = "";
    } else {
      const confidenceClass = confidence >= 0.85 ? "is-good" : confidence >= 0.6 ? "is-warn" : "is-bad";
      dom.confidenceBadge.classList.add(confidenceClass);
      text(dom.confidenceBadge, confidence >= 0.85 ? "HIGH CONFIDENCE" : confidence >= 0.6 ? "REVIEW" : "LOW CONFIDENCE");
      text(dom.confidenceValue, `${Math.round(confidence * 100)}%`);
      const color = confidence >= 0.85 ? "var(--mint)" : confidence >= 0.6 ? "var(--amber)" : "var(--red)";
      dom.confidenceRing.style.background = `radial-gradient(circle, #0c1816 58%, transparent 60%), conic-gradient(${color} ${confidence * 100}%, rgba(154, 190, 177, 0.14) 0)`;
    }

    const position = pose.position;
    text(dom.poseX, position ? position[0].toFixed(4) : "—");
    text(dom.poseY, position ? position[1].toFixed(4) : "—");
    text(dom.poseZ, position ? position[2].toFixed(4) : "—");
    text(dom.poseOrientation, pose.quaternion ? pose.quaternion.map((item) => item.toFixed(4)).join(", ") : "—");
    dom.poseTarget.classList.toggle("has-pose", Boolean(position));
    if (position) {
      const horizontal = clamp(position[1] * 80, -14, 14);
      const vertical = clamp(position[2] * -45, -10, 10);
      dom.poseTarget.style.translate = `${horizontal}px ${vertical}px`;
    } else {
      dom.poseTarget.style.translate = "0 0";
    }
    text(dom.poseSource, pose.source || (evidence.hasPerceptionEvent ? "Stage reported; 6-DoF fields not emitted" : "Awaiting evidence"));

    const plan = evidence.planning;
    text(dom.bay0Score, plan.scores ? plan.scores[0].toFixed(3) : "—");
    text(dom.bay1Score, plan.scores ? plan.scores[1].toFixed(3) : "—");
    dom.rackPlan.classList.remove("is-pass", "is-fail");
    if (plan.passed === true) {
      dom.rackPlan.classList.add("is-pass");
      text(dom.planGate, "ACCEPTED");
    } else if (plan.passed === false) {
      dom.rackPlan.classList.add("is-fail");
      text(dom.planGate, "REJECTED");
    } else {
      text(dom.planGate, "NOT EVALUATED");
    }
    dom.planGate.title = plan.threshold == null ? "No decision threshold emitted" : `Decision threshold ${plan.threshold.toFixed(2)}`;
  }

  function renderMetric(name, value, limit, decimals = 2) {
    const valueNode = dom[`${name}Value`];
    const bar = dom[`${name}Bar`];
    const marker = dom[`${name}LimitMarker`];
    const limitNode = dom[`${name}Limit`];
    text(valueNode, formatMetric(value, decimals));
    if (limitNode) text(limitNode, formatMetric(limit, decimals));
    bar.classList.remove("is-warning", "is-danger");
    if (value == null || limit == null || limit <= 0) {
      bar.style.width = "0%";
      if (marker) marker.style.display = "none";
      return;
    }
    if (marker) marker.style.display = "block";
    const ratio = value / limit;
    bar.style.width = `${clamp(ratio * 100, 0, 100)}%`;
    if (ratio > 1) bar.classList.add("is-danger");
    else if (ratio >= 0.8) bar.classList.add("is-warning");
  }

  function renderTelemetry(evidence) {
    renderMetric("force", evidence.force, evidence.forceLimit, 2);
    renderMetric("torque", evidence.torque, evidence.torqueLimit, 2);
    renderMetric("impulse", evidence.impulse, evidence.impulseLimit, 2);
    renderMetric("poseError", evidence.poseError, evidence.poseErrorLimit, 2);
    const channels = [evidence.force, evidence.torque, evidence.impulse, evidence.poseError].filter((value) => value != null).length;
    const safety = String(evidence.safetyState || "").toUpperCase();
    text(dom.telemetryState, safety && safety !== "UNKNOWN" ? `${safety} · ${channels}/4` : channels ? `${channels}/4 CHANNELS` : "AWAITING DATA");
  }

  function renderGate(job, evidence) {
    let kind = "pending";
    let title = "Gate pending";
    let summary = "The service has not produced a qualification decision.";
    let path = "M13 20h14";

    if (evidence.passed === true) {
      kind = "pass";
      title = "Qualification gate passed";
      summary = evidence.qualificationSummary || "All emitted acceptance predicates passed for this run.";
      path = "m12.5 20.5 5 5L28 14";
    } else if (evidence.passed === false) {
      kind = "fail";
      title = "Qualification gate failed";
      summary = evidence.qualificationSummary || "At least one emitted acceptance predicate failed. Review telemetry and artifacts.";
      path = "M14 14l12 12m0-12L14 26";
    } else if (job.status === "failed" && job.result) {
      kind = "fail";
      title = "Workflow outcome failed · no qualification";
      summary = evidence.qualificationSummary || job.error || "The simulated workflow did not satisfy its terminal predicate.";
      path = "M14 14l12 12m0-12L14 26";
    } else if (job.status === "failed") {
      kind = "fail";
      title = "Execution failed · no gate";
      summary = job.error || "The backend stopped before it could produce an acceptance decision.";
      path = "M14 14l12 12m0-12L14 26";
    } else if (job.status === "cancelled" || job.status === "canceled") {
      title = "Run cancelled · no gate";
      summary = "Cancellation ended the evidence chain before an acceptance decision was available.";
    } else if (job.status === "succeeded") {
      kind = "neutral";
      title = backendForJob(job) === "replay" ? "Replay complete · no gate" : "Execution complete · gate not emitted";
      summary = evidence.qualificationSummary || "The backend exited successfully, but it did not emit a design acceptance result. Review artifacts before making a qualification claim.";
    }

    dom.gatePanel.className = `gate-panel panel is-${kind}`;
    text(dom.gateTitle, title);
    text(dom.gateSummary, summary);
    dom.gateResultPath.setAttribute("d", path);
    text(dom.successRate, evidence.successRate == null ? "—" : `${(evidence.successRate * 100).toFixed(1)}%`);
    text(dom.trialCount, evidence.trials == null ? "—" : String(Math.round(evidence.trials)));
    text(dom.gateThreshold, evidence.threshold == null ? "—" : `≥ ${(evidence.threshold * 100).toFixed(1)}%`);
  }

  function renderEvidence(job) {
    const evidence = extractEvidence();
    renderPerception(evidence);
    renderTelemetry(evidence);
    renderGate(job, evidence);
  }

  function renderJob(job) {
    if (!job) {
      showDashboard(false);
      return;
    }
    showDashboard(true);
    setBadge(dom.jobModeBadge, backendForJob(job));
    text(dom.jobId, `JOB ${compactId(job.id)}`);
    dom.jobId.title = job.id;
    text(dom.jobTitle, job.preset_title || humanize(job.preset_id));
    text(dom.jobSubtitle, statusMessage(job));
    const status = String(job.status || "unknown").toLowerCase();
    dom.jobStatus.className = `run-status status-${status.replace(/[^a-z-]/g, "")}`;
    text(dom.jobStatus, status.toUpperCase());
    renderElapsed();
    renderPhases(job);
    renderEvidence(job);
    dom.cancelJob.disabled = TERMINAL_STATUSES.has(status) || Boolean(job.cancel_requested);
    dom.cancelJob.textContent = job.cancel_requested ? "Cancellation requested" : "Cancel job";
    renderArtifacts(Array.isArray(job.artifacts) ? job.artifacts : []);
    renderJobPicker();
  }

  function renderEvents() {
    dom.eventList.replaceChildren();
    if (!state.events.length) {
      const placeholder = document.createElement("li");
      placeholder.className = "event-placeholder";
      placeholder.textContent = "No worker events received.";
      dom.eventList.appendChild(placeholder);
      return;
    }
    for (const event of state.events.slice(-100)) {
      const row = document.createElement("li");
      if (event.level === "error" || event.type === "error") row.classList.add("event-error");
      else if (event.level === "warning") row.classList.add("event-warning");
      const timeNode = document.createElement("span");
      timeNode.className = "event-time";
      const timestamp = Date.parse(event.timestamp || "");
      timeNode.textContent = Number.isFinite(timestamp) ? new Date(timestamp).toLocaleTimeString([], { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }) : `#${event.seq || "?"}`;
      const kind = document.createElement("span");
      kind.className = "event-kind";
      kind.textContent = event.stage || event.type || "event";
      const message = document.createElement("span");
      message.className = "event-message";
      message.textContent = event.message || "Event received";
      row.append(timeNode, kind, message);
      dom.eventList.appendChild(row);
    }
    dom.eventList.scrollTop = dom.eventList.scrollHeight;
  }

  function artifactUrl(jobId, path) {
    const encoded = String(path).split("/").map(encodeURIComponent).join("/");
    return `${API_ROOT}/jobs/${encodeURIComponent(jobId)}/artifacts/${encoded}`;
  }

  function renderArtifacts(artifacts) {
    dom.artifactList.replaceChildren();
    text(dom.artifactCount, `${artifacts.length} ${artifacts.length === 1 ? "FILE" : "FILES"}`);
    if (!artifacts.length) {
      const item = document.createElement("li");
      item.className = "artifact-placeholder";
      const icon = document.createElement("span");
      icon.className = "file-icon";
      icon.setAttribute("aria-hidden", "true");
      icon.textContent = "—";
      const copy = document.createElement("span");
      const title = document.createElement("strong");
      title.textContent = "No artifacts yet";
      const detail = document.createElement("small");
      detail.textContent = "Files appear only after the worker writes them.";
      copy.append(title, detail);
      item.append(icon, copy);
      dom.artifactList.appendChild(item);
      return;
    }

    for (const artifact of artifacts) {
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = artifactUrl(state.selectedJob.id, artifact.path);
      link.target = "_blank";
      link.rel = "noopener";
      link.title = artifact.sha256 ? `SHA-256 ${artifact.sha256}` : "Open artifact";
      const icon = document.createElement("span");
      icon.className = "file-icon";
      icon.setAttribute("aria-hidden", "true");
      const extension = String(artifact.path).split(".").pop();
      icon.textContent = extension && extension !== artifact.path ? extension.slice(0, 4) : "file";
      const copy = document.createElement("span");
      const title = document.createElement("strong");
      title.textContent = artifact.path;
      const detail = document.createElement("small");
      detail.textContent = `${formatBytes(artifact.size_bytes)} · ${artifact.media_type || "unknown media type"}`;
      copy.append(title, detail);
      link.append(icon, copy);
      item.appendChild(link);
      dom.artifactList.appendChild(item);
    }
  }

  async function loadArtifacts(job) {
    try {
      const response = await api(`/jobs/${encodeURIComponent(job.id)}/artifacts`);
      const artifacts = Array.isArray(response) ? response : Array.isArray(response?.artifacts) ? response.artifacts : [];
      if (!state.selectedJob || state.selectedJob.id !== job.id) return;
      state.selectedJob.artifacts = artifacts;
      mergeJob(state.selectedJob);
      renderArtifacts(artifacts);
      await loadSummaryArtifact(job, artifacts);
    } catch (error) {
      if (error.status !== 404) toast(`Artifact index unavailable: ${error.message}`, "warning");
    }
  }

  async function loadSummaryArtifact(job, artifacts) {
    const artifact = artifacts.find((item) => /(^|\/)summary\.json$/i.test(item.path)) || artifacts.find((item) => /(^|\/)(qualification|report)\.json$/i.test(item.path));
    if (!artifact || finite(artifact.size_bytes) > 2_000_000) return;
    const key = `${job.id}:${artifact.path}:${artifact.sha256 || artifact.size_bytes || "?"}`;
    if (state.summaryKey === key) return;
    try {
      const response = await fetch(artifactUrl(job.id, artifact.path), { cache: "no-store", headers: { Accept: "application/json" } });
      if (!response.ok) return;
      const summary = await response.json();
      if (!state.selectedJob || state.selectedJob.id !== job.id) return;
      state.summary = summary;
      state.summaryKey = key;
      renderEvidence(state.selectedJob);
    } catch (_) {
      // Artifact links remain usable even when a report is not valid JSON.
    }
  }

  async function loadEvents(jobId, reset = false, drain = false) {
    let cursor = reset ? 0 : state.eventAfter;
    let firstPage = true;
    while (true) {
      const response = await api(`/jobs/${encodeURIComponent(jobId)}/events?after=${cursor}&limit=500`);
      if (!state.selectedJob || state.selectedJob.id !== jobId) return;
      const incoming = Array.isArray(response) ? response : Array.isArray(response?.events) ? response.events : [];
      if (reset && firstPage) state.events = [];
      const known = new Set(state.events.map((event) => event.seq));
      for (const event of incoming) {
        if (!known.has(event.seq)) state.events.push(event);
      }
      state.events.sort((left, right) => (finite(left.seq) || 0) - (finite(right.seq) || 0));
      const nextAfter = finite(response?.next_after);
      const maximum = state.events.reduce((current, event) => Math.max(current, finite(event.seq) || 0), 0);
      const nextCursor = Math.max(cursor, nextAfter || 0, maximum);
      state.eventAfter = Math.max(state.eventAfter, nextCursor);
      firstPage = false;
      if (!drain || incoming.length < 500 || nextCursor <= cursor) break;
      cursor = nextCursor;
    }
    renderEvents();
  }

  function schedulePoll(job) {
    window.clearTimeout(state.pollTimer);
    if (!job || TERMINAL_STATUSES.has(String(job.status).toLowerCase())) {
      dom.streamState.className = "stream-state is-stopped";
      dom.streamState.innerHTML = "<i></i> COMPLETE";
      return;
    }
    dom.streamState.className = "stream-state is-active";
    dom.streamState.innerHTML = "<i></i> POLLING";
    const delay = document.hidden ? 3500 : job.status === "queued" ? 1250 : 850;
    state.pollTimer = window.setTimeout(() => refreshSelected(), delay);
  }

  async function refreshSelected({ resetEvents = false, notify = false } = {}) {
    const id = state.selectedJob?.id;
    if (!id || state.refreshing) return;
    window.clearTimeout(state.pollTimer);
    state.refreshing = true;
    try {
      const [job] = await Promise.all([
        api(`/jobs/${encodeURIComponent(id)}`),
        loadEvents(id, resetEvents),
      ]);
      if (!state.selectedJob || state.selectedJob.id !== id) return;
      state.selectedJob = job;
      mergeJob(job);
      renderJob(job);
      if (TERMINAL_STATUSES.has(String(job.status).toLowerCase())) {
        await loadEvents(id, false, true);
        await loadArtifacts(job);
      }
      if (notify) toast("Evidence refreshed.");
    } catch (error) {
      dom.streamState.className = "stream-state is-stopped";
      dom.streamState.innerHTML = "<i></i> DISCONNECTED";
      if (notify || error.status === 404) toast(error.message, "error");
    } finally {
      state.refreshing = false;
      schedulePoll(state.selectedJob);
    }
  }

  async function selectJob(jobOrId) {
    const job = typeof jobOrId === "string" ? state.jobs.find((item) => item.id === jobOrId) : jobOrId;
    if (!job?.id) return;
    window.clearTimeout(state.pollTimer);
    state.selectedJob = job;
    state.events = [];
    state.eventAfter = 0;
    state.summary = null;
    state.summaryKey = null;
    renderEvents();
    renderJob(job);
    await refreshSelected({ resetEvents: true });
  }

  async function submitJob(event) {
    event.preventDefault();
    validatePreflight();
    if (dom.submitJob.disabled || state.submitting) return;
    const presetId = selectedPresetId();
    const seed = Number(dom.seed.value);
    state.submitting = true;
    dom.submitJob.classList.add("is-loading");
    dom.submitJob.querySelector("span").textContent = "Submitting";
    text(dom.formMessage, "");
    validatePreflight();
    try {
      const job = await api("/jobs", { method: "POST", body: { preset_id: presetId, seed }, timeout: 15000 });
      mergeJob(job);
      renderJobPicker();
      toast(`Job ${compactId(job.id)} accepted by the serialized worker.`);
      await selectJob(job);
      void loadJobs();
    } catch (error) {
      text(dom.formMessage, error.message);
      toast(`Could not start run: ${error.message}`, "error");
      await Promise.all([checkHealth(), loadCapabilities()]);
    } finally {
      state.submitting = false;
      dom.submitJob.classList.remove("is-loading");
      dom.submitJob.querySelector("span").textContent = "Start run";
      validatePreflight();
    }
  }

  async function cancelSelectedJob() {
    const job = state.selectedJob;
    if (!job || TERMINAL_STATUSES.has(String(job.status).toLowerCase())) return;
    const confirmed = window.confirm(`Cancel job ${compactId(job.id)}? Partial artifacts, if any, will remain available.`);
    if (!confirmed) return;
    dom.cancelJob.disabled = true;
    try {
      const updated = await api(`/jobs/${encodeURIComponent(job.id)}`, { method: "DELETE" });
      state.selectedJob = updated;
      mergeJob(updated);
      renderJob(updated);
      toast("Cancellation requested.", "warning");
      schedulePoll(updated);
    } catch (error) {
      toast(`Could not cancel job: ${error.message}`, "error");
      dom.cancelJob.disabled = false;
    }
  }

  async function copyJobId() {
    const id = state.selectedJob?.id;
    if (!id) return;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(id);
      } else {
        const temporary = document.createElement("textarea");
        temporary.value = id;
        temporary.setAttribute("readonly", "");
        temporary.style.position = "fixed";
        temporary.style.opacity = "0";
        document.body.appendChild(temporary);
        temporary.select();
        document.execCommand("copy");
        temporary.remove();
      }
      toast("Job identifier copied.");
    } catch (_) {
      toast("Clipboard access is unavailable in this browser.", "warning");
    }
  }

  function resetConfiguration() {
    const replay = dom.jobForm.querySelector('input[name="preset"][value="replay_full_chain"]');
    const firstAvailable = dom.jobForm.querySelector('input[name="preset"]:not(:disabled)');
    const target = replay && !replay.disabled ? replay : firstAvailable;
    if (target) target.checked = true;
    dom.seed.value = String(DEFAULT_SEED);
    text(dom.formMessage, "");
    updatePresetSelection();
  }

  function bindEvents() {
    dom.jobForm.addEventListener("submit", submitJob);
    dom.resetConfig.addEventListener("click", resetConfiguration);
    dom.seed.addEventListener("input", validatePreflight);
    dom.presetGrid.addEventListener("change", updatePresetSelection);
    dom.jobPicker.addEventListener("change", () => selectJob(dom.jobPicker.value));
    dom.refreshJob.addEventListener("click", () => refreshSelected({ notify: true }));
    dom.cancelJob.addEventListener("click", cancelSelectedJob);
    dom.copyJobId.addEventListener("click", copyJobId);
    document.addEventListener("visibilitychange", () => {
      if (state.selectedJob && !TERMINAL_STATUSES.has(String(state.selectedJob.status).toLowerCase())) {
        schedulePoll(state.selectedJob);
      }
    });
  }

  async function initialize() {
    cacheDom();
    bindEvents();
    updatePresetSelection();
    showDashboard(false);
    renderEvents();
    renderArtifacts([]);
    await Promise.all([checkHealth({ quiet: true }), loadCapabilities()]);
    await loadJobs({ selectNewest: true });
    state.healthTimer = window.setInterval(() => checkHealth({ quiet: true }), 10000);
    state.elapsedTimer = window.setInterval(renderElapsed, 1000);
  }

  window.addEventListener("beforeunload", () => {
    window.clearTimeout(state.pollTimer);
    window.clearInterval(state.healthTimer);
    window.clearInterval(state.elapsedTimer);
  });

  initialize().catch((error) => {
    cacheDom();
    text(dom.serviceStatus, "UI error");
    if (dom.serviceDot) dom.serviceDot.className = "status-dot is-offline";
    toast(`Console initialization failed: ${error.message}`, "error");
  });
})();
