const state = {
  currentDir: ".",
  configs: [],
  configGroups: {project_configs: [], model_configs: []},
  configType: localStorage.getItem("annotationConfigType") || "project",
  selectedConfig: "",
  patients: [],
  projectState: {project_configs: [], model_configs: [], pairs: [], runs: []},
  progress: {items: []},
  stats: {runs: []},
  runMode: localStorage.getItem("annotationRunMode") || "single",
  checkpoints: {items: []},
  benchmarkRunning: false,
  server: {},
  patientDirty: false
};

const viewMeta = {
  overview: ["运行工作台", "从这里确认运行范围、启动任务、查看进度和输出。"],
  patients: ["患者选择", "选择本次要运行的患者文件。规则文件固定为沙箱内置规则。"],
  models: ["模型与并发", "配置模型任务、运行并发和模型接口压测。"],
  checkpoint: ["Checkpoint", "查看断点续跑数据库、失败患者和失败任务。"],
  config: ["配置编辑", "编辑 YAML 配置。请不要外传包含 API key 的截图。"],
  files: ["文件查看", "查看输出、日志、checkpoint 和配置文件。"]
};

const stateLabels = {
  not_started: "未开始",
  running: "运行中",
  paused_or_stopped: "已暂停/已中断",
  complete: "已完成",
  complete_with_failures: "完成但有失败"
};

const patientStatusLabels = {
  running: "运行中",
  done: "已完成",
  failed: "失败",
  partial: "部分失败"
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `${res.status} ${res.statusText}`);
  }
  const type = res.headers.get("content-type") || "";
  return type.includes("application/json") ? res.json() : res.text();
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[ch]));
}

function stat(label, value, cls = "") {
  return `<div class="stat"><div class="label">${label}</div><div class="value ${cls}">${escapeHtml(value)}</div></div>`;
}

function optionList(values, selected) {
  return values.map(value => `<option value="${escapeHtml(value)}" ${value === selected ? "selected" : ""}>${escapeHtml(value)}</option>`).join("");
}

function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", {hour12: false});
}

function formatDuration(seconds) {
  const value = Number(seconds || 0);
  if (!value) return "-";
  if (value < 60) return `${value.toFixed(1)} 秒`;
  if (value < 3600) return `${(value / 60).toFixed(1)} 分钟`;
  return `${(value / 3600).toFixed(2)} 小时`;
}

function humanSize(bytes) {
  if (bytes === null || bytes === undefined || bytes === "") return "";
  const units = ["B", "KB", "MB", "GB"];
  let n = Number(bytes);
  let i = 0;
  while (n >= 1024 && i < units.length - 1) {
    n /= 1024;
    i += 1;
  }
  return `${n.toFixed(i ? 1 : 0)}${units[i]}`;
}

function switchView(name) {
  document.querySelectorAll(".view").forEach(el => el.classList.toggle("active", el.id === `view-${name}`));
  document.querySelectorAll(".nav-item").forEach(el => el.classList.toggle("active", el.dataset.view === name));
  $("pageTitle").textContent = viewMeta[name][0];
  $("pageSubtitle").textContent = viewMeta[name][1];
}

function primaryProgressItem() {
  const currentProject = currentModePairs()[0]?.project_config;
  return (state.progress.items || []).find(item => item.project_config === currentProject) || (state.progress.items || [])[0] || {};
}

function workbenchProgressItem() {
  const items = state.progress.items || [];
  if (items.length <= 1) return items[0] || {};
  const first = items[0] || {};
  const totalTasks = items.reduce((sum, item) => sum + Number(item.total_tasks || 0), 0);
  const processedTasks = Math.min(
    totalTasks,
    items.reduce((sum, item) => sum + Number(item.processed_tasks || 0), 0)
  );
  const doneTasks = items.reduce((sum, item) => sum + Number(item.done_tasks || 0), 0);
  const failedTasks = items.reduce((sum, item) => sum + Number(item.failed_tasks || 0), 0);
  return {
    ...first,
    active: items.some(item => item.active),
    total_tasks: totalTasks,
    processed_tasks: processedTasks,
    done_tasks: doneTasks,
    failed_tasks: failedTasks,
    percent: totalTasks ? Number((processedTasks / totalTasks * 100).toFixed(2)) : 0
  };
}

function activeRuns() {
  return (state.projectState.runs || []).filter(run => run.active);
}

function currentModePairs() {
  return state.runMode === "dual"
    ? (state.projectState.dual_pairs || [])
    : (state.projectState.single_pairs || state.projectState.pairs || []);
}

function setRunMode(mode) {
  state.runMode = mode === "dual" ? "dual" : "single";
  localStorage.setItem("annotationRunMode", state.runMode);
  document.querySelectorAll('input[type="radio"][value="single"], input[type="radio"][value="dual"]').forEach(input => {
    input.checked = input.value === state.runMode;
  });
  renderProjectPairs("projectPairs");
  renderProjectPairs("overviewProjectPairs");
  renderWorkbench();
}

function renderServerStatus() {
  const el = $("serverStatus");
  if (!el) return;
  const otherCount = Number(state.server.other_dashboard_process_count || 0);
  const packageName = state.server.package || "annotation_sandbox";
  el.classList.toggle("danger", otherCount > 0);
  el.textContent = otherCount > 0
    ? `当前包：${packageName}；同端口还有 ${otherCount} 个前端服务`
    : `当前包：${packageName}；PID ${state.server.pid || "-"}`;
}

function syncActionButtons(running) {
  ["overviewStartBtn", "startBothBtn"].forEach(id => {
    if ($(id)) $(id).disabled = Boolean(running);
  });
  ["overviewStopBtn", "stopRunsBtn"].forEach(id => {
    if ($(id)) $(id).disabled = !running;
  });
}

function renderWorkbench() {
  const progress = workbenchProgressItem();
  const running = (state.progress.items || []).some(item => item.active) || activeRuns().length > 0;
  const selectedCount = state.patients.filter(item => item.selected).length;
  const selectionEnabled = Boolean(progress.selection_enabled || $("enablePatientSelection")?.checked);
  const totalPatients = Number(progress.total_patients || (selectionEnabled ? selectedCount : state.patients.length) || 0);
  const totalRules = Number(progress.total_rules || state.stats.rule_count || 0);
  const totalTasks = Number(progress.total_tasks || totalPatients * totalRules || 0);
  const processed = Number(progress.processed_tasks || 0);
  const remaining = Math.max(0, totalTasks - processed);
  const percent = Number(progress.percent || 0);
  syncActionButtons(running);
  renderServerStatus();

  $("runStatusBanner").innerHTML = `
    <div>
      <strong>${running ? "任务正在运行" : processed > 0 && processed < totalTasks ? "任务已中断或暂停" : "当前没有运行中的任务"}</strong>
      <span>${running ? "页面会自动刷新进度。" : "确认患者和模型后，可以点击“启动勾选任务”。"}</span>
    </div>
    <div class="banner-metrics">
      <span>${percent}%</span>
      <span>${processed}/${totalTasks || "-"} 个任务</span>
      <span>剩余 ${remaining}</span>
    </div>
  `;

  $("runChecklist").innerHTML = [
    checklistItem("患者文件", totalPatients > 0, selectionEnabled ? `只运行已选择的 ${totalPatients} 个患者` : `运行患者目录内全部 ${totalPatients} 个患者`),
    checklistItem("规则文件", totalRules > 0, `${totalRules || "-"} 条规则`),
    checklistItem("模型任务", currentModePairs().length > 0, `${state.runMode === "dual" ? "双模型" : "单模型"}，${currentModePairs().length} 个任务槽位`),
    checklistItem("断点续跑", true, "已启用 checkpoint，停止后再次启动会接着跑")
  ].join("");

  $("runScope").innerHTML = `
    <div><b>运行文件范围</b><span>${selectionEnabled ? "只运行勾选患者" : "运行患者目录下全部患者"}</span></div>
    <div><b>患者数量</b><span>${totalPatients || "-"}</span></div>
    <div><b>规则数量</b><span>${totalRules || "-"}</span></div>
    <div><b>预计任务数</b><span>${totalTasks || "-"}</span></div>
    <div><b>剩余任务数</b><span>${remaining}</span></div>
    <div><b>患者目录</b><code>${escapeHtml(progress.patient_dir || $("patientDir")?.textContent || "-")}</code></div>
  `;

  renderSelectedPatientsPreview(progress);
  renderRecentPatients();
}

function scheduleWorkbenchRender() {
  if (state.workbenchRenderPending) return;
  state.workbenchRenderPending = true;
  requestAnimationFrame(() => {
    state.workbenchRenderPending = false;
    renderWorkbench();
  });
}

function checklistItem(label, ok, text) {
  return `
    <div class="checklist-item ${ok ? "ok" : "danger"}">
      <span>${ok ? "✓" : "!"}</span>
      <div><b>${label}</b><small>${escapeHtml(text)}</small></div>
    </div>
  `;
}

function renderSelectedPatientsPreview(progress) {
  const selectedFiles = progress.selected_patient_files || state.patients.filter(item => item.selected).map(item => item.name).slice(0, 20);
  if (!progress.selection_enabled && !$("enablePatientSelection")?.checked) {
    const allFiles = state.patients.map(item => item.name).slice(0, 20);
    $("selectedPatientsPreview").innerHTML = allFiles.length
      ? `<p class='status full-row'>当前未启用患者筛选，会运行患者目录中的全部 Excel 文件。下面显示前 ${allFiles.length} 个：</p>${allFiles.map(name => `<span>${escapeHtml(name)}</span>`).join("")}`
      : "<p class='status'>当前未启用患者筛选，会运行患者目录中的全部 Excel 文件。</p>";
    return;
  }
  $("selectedPatientsPreview").innerHTML = selectedFiles.length
    ? selectedFiles.map(name => `<span>${escapeHtml(name)}</span>`).join("")
    : "<p class='status'>尚未勾选患者文件。请进入“患者选择”页面勾选并保存。</p>";
}

function renderRecentPatients() {
  const rows = (state.progress.items || []).flatMap(item =>
    (item.recent_patients || []).map(row => ({...row, project_config: item.project_config}))
  ).slice(0, 20);
  if (!rows.length) {
    $("recentPatientsList").innerHTML = "<p class='status'>暂无患者处理记录。任务启动后这里会显示最近处理到哪些患者。</p>";
    return;
  }
  $("recentPatientsList").innerHTML = `
    <div class="table-head">
      <span>患者</span><span>状态</span><span>规则进度</span><span>更新时间</span><span>来源文件</span>
    </div>
    ${rows.map(row => `
      <div class="table-row">
        <span>${escapeHtml(row.patient_sn)}</span>
        <span>${patientStatusLabels[row.status] || row.status}</span>
        <span>${row.done_rules}/${row.total_rules}，失败 ${row.failed_rules}</span>
        <span>${formatDateTime(row.updated_at)}</span>
        <span title="${escapeHtml(row.source_file)}">${escapeHtml(row.source_file ? row.source_file.split(/[\\/]/).pop() : "-")}</span>
      </div>
    `).join("")}
  `;
}

async function loadProgress() {
  const data = await api(`/api/progress?mode=${encodeURIComponent(state.runMode)}`);
  state.progress = data;
  $("progressUpdated").textContent = data.updated_at ? `更新时间：${formatDateTime(data.updated_at)}` : "";
  $("progressList").innerHTML = (data.items || []).map(item => progressCard(item)).join("") || "<p class='status'>暂无进度数据。</p>";
  renderWorkbench();
}

function progressCard(item) {
  const pct = Number(item.percent || 0);
  const remaining = Math.max(0, Number(item.total_tasks || 0) - Number(item.processed_tasks || 0));
  const cls = item.state === "running" ? "running" : item.state.includes("fail") ? "danger" : item.state === "complete" ? "ok" : "";
  return `
    <div class="progress-card">
      <div class="progress-head">
        <strong>${escapeHtml(item.project_config)}</strong>
        <span class="badge ${cls}">${stateLabels[item.state] || item.state}</span>
      </div>
      <div class="progress-bar"><span style="width:${Math.min(100, pct)}%"></span></div>
      <div class="progress-meta">
        <span>${pct}%</span>
        <span>${item.processed_tasks}/${item.total_tasks} 个任务</span>
        <span>剩余 ${remaining}</span>
        <span>${item.total_patients} 个患者 × ${item.total_rules} 条规则</span>
        <span>成功 ${item.done_tasks} / 失败 ${item.failed_tasks}</span>
        <span>平均每患者 ${formatDuration(item.avg_patient_elapsed_seconds)}</span>
        <span>最近更新 ${formatDateTime(item.latest_updated)}</span>
      </div>
      <div class="progress-path">${escapeHtml(item.output_dir)}</div>
    </div>
  `;
}

async function loadStats() {
  const data = await api("/api/stats");
  state.stats = data;
  $("statsUpdated") && ($("statsUpdated").textContent = data.updated_at || "");
  $("runsList").innerHTML = (data.runs || []).map(run => `
    <div class="run-item">
      <strong>${escapeHtml(run.dir)}</strong>
      <span>时间：${formatDateTime(run.summary_updated_at || run.output_updated_at || run.run_id)}</span>
      <span>模型：${escapeHtml(run.model_name || "-")}</span>
      <span>成功：${run.results_written_this_run ?? "-"}</span>
      <span>失败：${run.failures_this_run ?? "-"}</span>
      <span>耗时：${run.elapsed_seconds ?? "-"} 秒</span>
    </div>
  `).join("") || "<p class='status'>暂无运行输出。</p>";
}

async function loadConfigs() {
  const data = await api("/api/configs");
  state.configGroups = {
    project_configs: data.project_configs || data.configs || [],
    model_configs: data.model_configs || []
  };
  state.configs = currentConfigList();
  if (!state.configs.includes(state.selectedConfig)) {
    state.selectedConfig = state.configs[0] || "";
  }
  $("configTypeSelect").value = state.configType;
  $("configSelect").innerHTML = optionList(state.configs, state.selectedConfig);
  $("configSelect").value = state.selectedConfig;
  await loadConfig();
  await loadPatients();
}

function currentConfigList() {
  return state.configType === "model"
    ? state.configGroups.model_configs
    : state.configGroups.project_configs;
}

async function changeConfigType() {
  state.configType = $("configTypeSelect").value || "project";
  localStorage.setItem("annotationConfigType", state.configType);
  state.configs = currentConfigList();
  state.selectedConfig = state.configs[0] || "";
  $("configSelect").innerHTML = optionList(state.configs, state.selectedConfig);
  $("configSelect").value = state.selectedConfig;
  await loadConfig();
}

async function loadConfig() {
  const name = $("configSelect").value;
  if (!name) return;
  state.selectedConfig = name;
  const data = await api(`/api/config?name=${encodeURIComponent(name)}`);
  $("configEditor").value = data.content || "";
  $("configStatus").textContent = `已载入 ${name}`;
}

async function saveConfig() {
  const name = $("configSelect").value;
  await api(`/api/config?name=${encodeURIComponent(name)}`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({content: $("configEditor").value})
  });
  $("configStatus").textContent = `已保存 ${name}`;
  await Promise.all([loadPatients(), loadProjects(), loadConcurrency(), loadProgress()]);
}

async function loadPatients() {
  const config = currentModePairs()[0]?.project_config || "project_full_ai_only.yaml";
  const data = await api(`/api/patients?config=${encodeURIComponent(config)}`);
  state.patients = data.items || [];
  $("patientDir").textContent = data.patient_dir || "-";
  $("patientDirInput").value = data.patient_dir || "";
  $("enablePatientSelection").checked = Boolean(data.selection_enabled);
  renderPatients();
  renderWorkbench();
}

function selectedPatientNames() {
  return state.patients.filter(item => item.selected).map(item => item.name);
}

function selectedPatientCount() {
  return state.patients.reduce((count, item) => count + (item.selected ? 1 : 0), 0);
}

function updatePatientSelectionSummary() {
  $("patientSelectionSummary").textContent = `共 ${state.patients.length} 个患者文件，已选择 ${selectedPatientCount()} 个。`;
}

function syncVisiblePatientCheckboxes() {
  const selected = new Set(selectedPatientNames());
  document.querySelectorAll("#patientList input[data-patient]").forEach(input => {
    input.checked = selected.has(input.dataset.patient);
  });
}

function renderPatients() {
  const keyword = ($("patientSearch").value || "").trim().toLowerCase();
  const visible = state.patients.filter(item => !keyword || item.name.toLowerCase().includes(keyword));
  updatePatientSelectionSummary();
  $("patientList").innerHTML = visible.map(item => `
    <label class="patient-row">
      <input type="checkbox" data-patient="${escapeHtml(item.name)}" ${item.selected ? "checked" : ""}>
      <span>${escapeHtml(item.name)}</span>
      <small>${humanSize(item.size)}</small>
    </label>
  `).join("") || "<p class='status'>未找到患者文件。</p>";
}

function schedulePatientRender() {
  if (state.patientRenderTimer) {
    clearTimeout(state.patientRenderTimer);
  }
  state.patientRenderTimer = setTimeout(() => {
    state.patientRenderTimer = null;
    renderPatients();
  }, 120);
}

function markPatientDirty() {
  state.patientDirty = true;
  $("patientStatus").textContent = "患者选择已修改，启动任务时会自动保存；也可以先点击“保存选择”。";
}

async function savePatients(options = {}) {
  const files = selectedPatientNames();
  const enabled = $("enablePatientSelection").checked;
  const silent = Boolean(options.silent);
  await api("/api/patients", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({enabled, files})
  });
  $("patientStatus").textContent = enabled ? `已保存：只运行 ${files.length} 个选中患者。` : "已保存：未启用患者筛选，将运行全部患者。";
  if (silent) {
    $("patientStatus").textContent = "";
  }
  state.patientDirty = false;
  await Promise.all([loadPatients(), loadStats(), loadProgress()]);
}

async function savePatientDir() {
  const patientDir = $("patientDirInput").value.trim();
  const projectConfig = currentModePairs()[0]?.project_config || "project_full_ai_only.yaml";
  if (!patientDir) {
    $("patientStatus").textContent = "患者目录不能为空。";
    return;
  }
  const data = await api("/api/patient-dir", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({project_config: projectConfig, patient_dir: patientDir})
  });
  $("patientStatus").textContent = `已保存患者目录：${data.patient_dir}，共 ${data.total} 个 Excel 文件。`;
  await Promise.all([loadPatients(), loadStats(), loadProgress(), loadProjects()]);
}

async function loadProjects() {
  const data = await api("/api/projects");
  state.projectState = data;
  $("benchModelSelect").innerHTML = optionList(data.model_configs || [], "model_api.yaml");
  renderProjectPairs("projectPairs");
  renderProjectPairs("overviewProjectPairs");
  renderDashboardRuns(data.runs || []);
  setRunMode(state.runMode);
  renderWorkbench();
}

function renderProjectPairs(containerId) {
  const projects = state.projectState.project_configs || [];
  const models = state.projectState.model_configs || [];
  const compact = containerId === "overviewProjectPairs";
  const pairs = currentModePairs();
  const activeProjects = new Set(activeRuns().map(run => run.project_config));
  $(containerId).innerHTML = pairs.map((pair, idx) => `
    <div class="pair-row ${compact ? "compact" : ""}">
      <label class="check"><input type="checkbox" data-pair-enabled="${idx}" checked> ${escapeHtml(pair.name)}</label>
      <select data-pair-project="${idx}">${optionList(projects, pair.project_config)}</select>
      <select data-pair-model="${idx}">${optionList(models, pair.model_config)}</select>
      <span class="badge ${pair.ready ? "ok" : "danger"}">${pair.ready ? "已配置" : "未配置"}</span>
      <button data-start-one="${idx}" ${activeProjects.has(pair.project_config) ? "disabled" : ""}>${activeProjects.has(pair.project_config) ? "运行中" : "启动"}</button>
    </div>
  `).join("") || `<p class='status'>${state.runMode === "dual" ? "没有找到双模型配置，请先填写 model_api_1.yaml 和 model_api_2.yaml。" : "未找到模型项目配置。"}</p>`;
  pairs.forEach((pair, idx) => {
    const input = $(containerId).querySelector(`[data-pair-enabled="${idx}"]`);
    if (input) input.checked = pair.enabled !== false;
  });
}

function collectSelectedPairs(containerId, singleIndex = null) {
  const pairs = currentModePairs();
  const root = containerId ? $(containerId) : document;
  const selected = [];
  pairs.forEach((pair, idx) => {
    const enabledInput = root.querySelector(`[data-pair-enabled="${idx}"]`);
    const projectSelect = root.querySelector(`[data-pair-project="${idx}"]`);
    const modelSelect = root.querySelector(`[data-pair-model="${idx}"]`);
    const enabled = singleIndex === idx || enabledInput?.checked;
    if (!enabled || !projectSelect || !modelSelect) return;
    selected.push({
      name: pair.name,
      project_config: projectSelect.value,
      model_config: modelSelect.value
    });
  });
  return selected;
}

async function loadCheckpoints() {
  const data = await api("/api/checkpoints");
  state.checkpoints = data;
  $("checkpointUpdated").textContent = data.updated_at ? `更新时间：${formatDateTime(data.updated_at)}` : "";
  $("checkpointList").innerHTML = (data.items || []).map(renderCheckpointCard).join("") || "<p class='status'>暂无 checkpoint 数据。</p>";
}

function renderCheckpointCard(item) {
  return `
    <div class="checkpoint-card">
      <div class="progress-head">
        <strong>${escapeHtml(item.project_config)}</strong>
        <span class="badge">${stateLabels[item.state] || item.state}</span>
      </div>
      <div class="scope-box">
        <div><b>数据库</b><code>${escapeHtml(item.checkpoint_db)}</code></div>
        <div><b>输出目录</b><code>${escapeHtml(item.output_dir)}</code></div>
        <div><b>任务统计</b><span>成功 ${item.done_tasks}，失败 ${item.failed_tasks}，总计 ${item.total_tasks}</span></div>
        <div><b>患者统计</b><span>${escapeHtml(JSON.stringify(item.patient_counts || {}))}</span></div>
      </div>
      <h3>失败患者</h3>
      ${renderSmallRows(item.failed_patient_rows || [], ["patient_sn", "status", "done_rules", "failed_rules", "updated_at"])}
      <h3>失败任务</h3>
      ${renderSmallRows(item.failed_task_rows || [], ["patient_sn", "trial_id", "standard_no", "updated_at"])}
    </div>
  `;
}

function renderSmallRows(rows, keys) {
  if (!rows.length) return "<p class='status'>暂无记录。</p>";
  return `
    <div class="small-table">
      <div class="small-row head">${keys.map(key => `<span>${escapeHtml(key)}</span>`).join("")}</div>
      ${rows.slice(0, 20).map(row => `<div class="small-row">${keys.map(key => `<span title="${escapeHtml(row[key])}">${escapeHtml(row[key] ?? "-")}</span>`).join("")}</div>`).join("")}
    </div>
  `;
}

async function startRuns(containerId, singleIndex = null) {
  const pairs = collectSelectedPairs(containerId, singleIndex);
  const statusEl = containerId === "overviewProjectPairs" ? $("overviewRunStatus") : $("multiStatus");
  if (!pairs.length) {
    statusEl.textContent = "未选择运行任务。";
    return;
  }
  statusEl.textContent = "正在保存患者选择...";
  await savePatients({silent: true});
  statusEl.textContent = "正在进行启动前检查...";
  const preflight = await api("/api/preflight", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({pairs})
  });
  if (!preflight.ok) {
    statusEl.innerHTML = `<span class="danger">启动前检查未通过：</span><br>${preflight.errors.map(escapeHtml).join("<br>")}`;
    return;
  }
  if (preflight.warnings?.length) {
    statusEl.innerHTML = `启动前检查通过，提示：<br>${preflight.warnings.map(escapeHtml).join("<br>")}`;
  }
  statusEl.textContent = "正在启动任务...";
  const data = await api("/api/runs", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({pairs})
  });
  statusEl.textContent = `已启动 ${data.started.length} 个运行任务，跳过 ${data.skipped?.length || 0} 个已在运行的任务。`;
  await Promise.all([loadProjects(), loadProgress(), loadStats()]);
}

async function stopRuns(payload = {}) {
  if (!confirm("确认中断运行中的任务吗？中断不会删除 checkpoint，下次启动会继续续跑。")) {
    return;
  }
  const data = await api("/api/stop", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({mode: "active", ...payload})
  });
  const message = `已中断 ${data.stopped.length} 个任务，跳过 ${data.skipped.length} 个任务。`;
  $("overviewRunStatus").textContent = message;
  $("multiStatus").textContent = message;
  await Promise.all([loadProjects(), loadProgress(), loadStats(), loadCheckpoints()]);
}

async function clearCheckpoints() {
  if (activeRuns().length) {
    $("checkpointActionStatus").textContent = "当前仍有任务运行，请先中断任务，再清除 Checkpoint。";
    return;
  }
  const firstConfirm = confirm("确认清除所有 Checkpoint 吗？清除后，已跑过的患者和规则会从头重新执行；历史输出文件不会删除。");
  if (!firstConfirm) return;
  const typed = prompt("为了防止误触，请输入 CLEAR 后再确认清除：");
  if (typed !== "CLEAR") {
    $("checkpointActionStatus").textContent = "已取消：确认词不匹配。";
    return;
  }
  $("clearCheckpointsBtn").disabled = true;
  $("checkpointActionStatus").textContent = "正在清除 Checkpoint...";
  try {
    const data = await api("/api/clear-checkpoints", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({confirm: "CLEAR"})
    });
    $("checkpointActionStatus").textContent = `已清除 ${data.deleted.length} 个 Checkpoint 文件。下次启动会从头运行，不会跳过旧患者。`;
    await Promise.all([loadProjects(), loadProgress(), loadCheckpoints()]);
    renderWorkbench();
  } finally {
    $("clearCheckpointsBtn").disabled = false;
  }
}

function renderDashboardRuns(runs) {
  const html = runs.map(run => `
    <div class="run-item">
      <strong>${escapeHtml(run.name)}</strong>
      <span>启动时间：${formatDateTime(run.started_at)}</span>
      <span>日志更新时间：${formatDateTime(run.log_updated_at)}</span>
      <span>状态：${run.active ? "运行中" : "已停止"}</span>
      <span>进程 ID：${escapeHtml(run.pid)}</span>
      <span>项目配置：${escapeHtml(run.project_config)}</span>
      <span>模型配置：${escapeHtml(run.model_config)}</span>
      <span>日志：${escapeHtml(run.log)}</span>
      <button data-open-log="${escapeHtml(run.log)}" class="small">查看日志</button>
      ${run.stopped_at ? `<span>中断时间：${formatDateTime(run.stopped_at)}</span>` : ""}
      ${run.active ? `<button data-stop-pid="${escapeHtml(run.pid)}" class="danger-btn small">中断这个任务</button>` : ""}
    </div>
  `).join("") || "<p class='status'>暂无启动记录。</p>";
  $("dashboardRunsOverview").innerHTML = html;
}

async function loadConcurrency() {
  const data = await api("/api/concurrency");
  $("runConcurrency").value = data.run_concurrency || 1;
  $("ruleConcurrency").value = data.rule_concurrency || 1;
  $("llmBatchSize").value = data.llm_batch_size || 1;
  $("maxFailures").value = data.max_consecutive_llm_failures ?? 5;
  $("maxTokens").value = data.model_max_tokens || 1024;
  $("groupRulesForLlm").checked = Boolean(data.group_rules_for_llm);
  $("writeXlsx").checked = Boolean(data.write_xlsx);
  $("benchConcurrency").value = data.run_concurrency || data.model_concurrency || 1;
  if (state.projectState.model_configs.length) {
    $("benchModelSelect").innerHTML = optionList(state.projectState.model_configs, data.model_config || "model_api.yaml");
  }
  renderBenchmark(data.latest_benchmark);
}

async function loadServer() {
  const data = await api("/api/server");
  state.server = data;
  renderServerStatus();
}

async function saveConcurrency() {
  const concurrency = Number($("runConcurrency").value || 1);
  const ruleConcurrency = Number($("ruleConcurrency").value || 1);
  const llmBatchSize = Number($("llmBatchSize").value || 1);
  const maxFailures = Number($("maxFailures").value || 0);
  const maxTokens = Number($("maxTokens").value || 1024);
  const data = await api("/api/concurrency", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      action: "set",
      concurrency,
      rule_concurrency: ruleConcurrency,
      group_rules_for_llm: $("groupRulesForLlm").checked,
      llm_batch_size: llmBatchSize,
      max_consecutive_llm_failures: maxFailures,
      max_tokens: maxTokens,
      write_xlsx: $("writeXlsx").checked
    })
  });
  $("concurrencyStatus").textContent = `已保存：患者并发 ${data.concurrency}，规则并发 ${data.rule_concurrency}，批量 ${data.llm_batch_size}，max_tokens ${data.max_tokens}。`;
  await loadConcurrency();
}

async function runBenchmark() {
  if (state.benchmarkRunning) return;
  state.benchmarkRunning = true;
  $("runBenchmarkBtn").disabled = true;
  $("benchmarkOutput").textContent = "压测运行中...";
  $("concurrencyStatus").textContent = "";
  try {
    const payload = {
      action: "benchmark",
      concurrency: Number($("benchConcurrency").value || 1),
      requests: Number($("benchRequests").value || 4),
      prompt_mode: $("benchPromptMode").value,
      model_config: $("benchModelSelect").value || "model_api.yaml"
    };
    const data = await api("/api/concurrency", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    renderBenchmark(data.latest_benchmark);
    $("benchmarkOutput").textContent = [data.stdout || "", data.stderr || ""].filter(Boolean).join("\n\n") || "无输出。";
    $("concurrencyStatus").textContent = data.ok ? "压测已完成。" : "压测完成，但存在失败。";
  } finally {
    state.benchmarkRunning = false;
    $("runBenchmarkBtn").disabled = false;
  }
}

function renderBenchmark(summary) {
  if (!summary) {
    $("benchmarkSummary").innerHTML = "<p class='status'>暂无压测数据。</p>";
    return;
  }
  $("benchmarkSummary").innerHTML = [
    stat("并发数", summary.concurrency ?? "-"),
    stat("请求数", summary.requests ?? "-"),
    stat("成功", summary.success ?? "-", Number(summary.failed || 0) ? "" : "ok"),
    stat("失败", summary.failed ?? "-", Number(summary.failed || 0) ? "danger" : "ok"),
    stat("请求/分钟", summary.requests_per_minute ?? "-"),
    stat("P95 秒", summary.p95_latency_seconds ?? "-")
  ].join("");
}

async function loadFiles(dir = state.currentDir) {
  const data = await api(`/api/files?dir=${encodeURIComponent(dir)}`);
  state.currentDir = data.dir;
  $("currentPath").textContent = data.dir;
  $("fileList").innerHTML = data.items.map(item => `
    <div class="file-row" data-path="${escapeHtml(item.path)}" data-type="${escapeHtml(item.type)}">
      <span class="file-name">${item.type === "dir" ? "[目录]" : "[文件]"} ${escapeHtml(item.name)}</span>
      <span class="file-meta">${item.type === "file" ? humanSize(item.size) : ""}</span>
    </div>
  `).join("");
}

async function openFile(path, type) {
  if (type === "dir") {
    await loadFiles(path);
    return;
  }
  const data = await api(`/api/file?path=${encodeURIComponent(path)}`);
  $("viewerTitle").textContent = path;
  $("filePreview").textContent = data.preview || "";
  $("downloadLink").href = `/download?path=${encodeURIComponent(path)}`;
}

async function goUp() {
  const parts = state.currentDir.split(/[\\/]/).filter(Boolean);
  parts.pop();
  await loadFiles(parts.join("/") || ".");
}

function bind() {
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });
  $("refreshBtn").addEventListener("click", refreshAll);
  $("overviewPatientsBtn").addEventListener("click", () => switchView("patients"));
  $("overviewCheckpointBtn").addEventListener("click", () => switchView("checkpoint"));
  $("overviewStartBtn").addEventListener("click", () => startRuns("overviewProjectPairs"));
  $("overviewStopBtn").addEventListener("click", () => stopRuns());
  document.querySelectorAll('input[type="radio"][value="single"], input[type="radio"][value="dual"]').forEach(input => {
    input.addEventListener("change", async () => {
      setRunMode(input.value);
      await Promise.all([loadProgress(), loadProjects()]);
    });
  });
  $("configSelect").addEventListener("change", loadConfig);
  $("configTypeSelect").addEventListener("change", changeConfigType);
  $("reloadConfigBtn").addEventListener("click", loadConfig);
  $("saveConfigBtn").addEventListener("click", saveConfig);
  $("savePatientsBtn").addEventListener("click", savePatients);
  $("savePatientDirBtn").addEventListener("click", savePatientDir);
  $("saveConcurrencyBtn").addEventListener("click", saveConcurrency);
  $("runBenchmarkBtn").addEventListener("click", runBenchmark);
  $("startBothBtn").addEventListener("click", () => startRuns("projectPairs"));
  $("stopRunsBtn").addEventListener("click", () => stopRuns());
  $("clearCheckpointsBtn").addEventListener("click", clearCheckpoints);

  for (const containerId of ["projectPairs", "overviewProjectPairs"]) {
    $(containerId).addEventListener("click", event => {
      const button = event.target.closest("button[data-start-one]");
      if (!button) return;
      startRuns(containerId, Number(button.dataset.startOne));
    });
  }
  $("dashboardRunsOverview").addEventListener("click", event => {
    const button = event.target.closest("button[data-stop-pid]");
    if (button) {
      stopRuns({pid: Number(button.dataset.stopPid || 0)});
      return;
    }
    const logButton = event.target.closest("button[data-open-log]");
    if (!logButton) return;
    switchView("files");
    openFile(logButton.dataset.openLog, "file");
  });

  $("selectAllPatientsBtn").addEventListener("click", () => {
    state.patients.forEach(item => { item.selected = true; });
    $("enablePatientSelection").checked = true;
    syncVisiblePatientCheckboxes();
    updatePatientSelectionSummary();
    scheduleWorkbenchRender();
    markPatientDirty();
  });
  $("clearPatientsBtn").addEventListener("click", () => {
    state.patients.forEach(item => { item.selected = false; });
    syncVisiblePatientCheckboxes();
    updatePatientSelectionSummary();
    scheduleWorkbenchRender();
    markPatientDirty();
  });
  $("patientSearch").addEventListener("input", schedulePatientRender);
  $("patientList").addEventListener("change", event => {
    const input = event.target.closest("input[data-patient]");
    if (!input) return;
    const item = state.patients.find(row => row.name === input.dataset.patient);
    if (item) item.selected = input.checked;
    if (input.checked) $("enablePatientSelection").checked = true;
    updatePatientSelectionSummary();
    scheduleWorkbenchRender();
    markPatientDirty();
  });
  $("enablePatientSelection").addEventListener("change", () => {
    scheduleWorkbenchRender();
    markPatientDirty();
  });
  $("upBtn").addEventListener("click", goUp);
  $("fileList").addEventListener("click", event => {
    const row = event.target.closest(".file-row");
    if (!row) return;
    openFile(row.dataset.path, row.dataset.type);
  });
}

async function refreshAll() {
  await Promise.all([loadServer(), loadProjects()]);
  await Promise.all([loadProgress(), loadStats(), loadConfigs(), loadFiles("."), loadConcurrency(), loadCheckpoints()]);
  renderWorkbench();
}

async function init() {
  bind();
  await refreshAll();
  setInterval(async () => {
    await Promise.all([loadServer(), loadProgress(), loadProjects(), loadStats(), loadCheckpoints()]);
  }, 5000);
}

init().catch(err => {
  console.error(err);
  alert(`错误：${err.message || err}`);
});
