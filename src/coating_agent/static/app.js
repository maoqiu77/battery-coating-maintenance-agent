'use strict';

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const icon = (name) => `<svg class="icon" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const fmt = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const scenarioNames = {blockage:'供浆阻力上升',sensor_bias:'压力传感器偏差',pump_loss:'泵效率下降',missing:'数据缺失',mixed:'混合异常',normal:'正常运行'};
const stateNames = {OBSERVING:'观测中',ALERTED:'发现异常',DIAGNOSED:'已完成诊断',AWAITING_CONFIRMATION:'等待确认',ACTION_EXECUTED:'待独立复核',VERIFYING:'复核中',RESOLVED:'已验证恢复',UNRESOLVED:'待进一步分析'};
const eventNames = {created:'创建模拟会话',advance:'推进模拟观测',diagnosis:'完成证据诊断',confirmation_requested:'申请模拟清理',confirmation_refused:'取消模拟处置',confirmation_denied:'确认未通过校验',action_denied:'处置条件未满足',action_executed:'已执行模拟清理',verification:'完成独立窗口复核',prediction:'生成模拟事件评分',optimization:'完成参数对照实验'};
const views = {overview:['运维总览','让设备状态，一目了然','从异常发现到证据诊断，完成一次可追溯的运维闭环。'],diagnosis:['诊断与证据','让每一个判断，都有依据','关联观测、候选原因与知识来源，让诊断过程透明可查。'],knowledge:['设备知识库','知识有来源，推断有边界','为模拟供浆诊断整理的公开资料、项目假设与模拟规则。'],reports:['会话与报告','把运维过程，完整留存','继续历史会话，或导出包含证据和审计轨迹的模拟报告。']};
views.agent = ['Agent 工作流','从发现异常，到验证恢复','给出目标，让 Agent 取证、选择下一步，并在你的确认下完成模拟处置。'];
const state = {id:null,status:null,report:null,config:{},sessions:[],cards:[],rows:[],range:180,series:new Set(['pressure','flow']),busy:false,connected:false,view:'overview',prediction:null};
let toastTimer;

async function request(path, method = 'GET', data) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), /\/agent\/(start|resume)$/.test(path) ? 300000 : 90000);
  try {
    const response = await fetch(path, {method,signal:controller.signal,headers:data === undefined ? {} : {'Content-Type':'application/json'},body:data === undefined ? undefined : JSON.stringify(data)});
    state.connected = true;
    const value = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(value.detail) ? value.detail.map((x) => x.msg).join('；') : value.detail;
      const error = new Error(detail || `请求失败（${response.status}）`);
      error.status = response.status;
      throw error;
    }
    return value;
  } catch (error) {
    if (error.name === 'AbortError') { state.connected=false; throw new Error('请求超时，请刷新会话确认最新状态后重试。'); }
    if (error instanceof TypeError) { state.connected=false; throw new Error('无法连接本地服务，请确认服务仍在运行后重试。'); }
    throw error;
  } finally { clearTimeout(timeout); }
}
const endpoint = (suffix = '') => `/api/sessions/${encodeURIComponent(state.id)}${suffix}`;
function remember(id) { try { localStorage.setItem('coating-session', id); } catch {} }
function showError(error) { $('error-banner').hidden = false; $('error-banner').innerHTML = `<span>${esc(error.message)}</span><button aria-label="关闭提示" data-action="dismiss-error">${icon('close')}</button>`; }
function toast(message) { clearTimeout(toastTimer); $('toast').textContent = message; $('toast').hidden = false; toastTimer = setTimeout(() => { $('toast').hidden = true; }, 4500); }
function setBusy(value, label = '正在处理') {
  state.busy = value;
  document.body.classList.toggle('busy', value);
  $('main').setAttribute('aria-busy', String(value));
  $('connection').innerHTML = `<i class="live-dot"></i>${value ? esc(label) : state.connected ? '本地服务已连接' : '服务连接异常'}`;
  $('connection').classList.toggle('offline',!value && !state.connected);
  syncButtons();
}
function syncButtons() {
  document.querySelectorAll('[data-action]').forEach((button) => {
    if (button.dataset.action === 'dismiss-error') return;
    const capability = button.dataset.capability;
    const allowed = !capability || !!state.status?.capabilities?.[capability];
    const configured = capability !== 'predict' || state.config.prediction_available;
    button.disabled = state.busy || !allowed || !configured || (button.hasAttribute('data-session') && !state.id);
    button.title = !configured ? '尚未生成预测模型，请先运行训练脚本' : capability && !allowed ? state.status?.capability_reasons?.[capability] || '请先创建会话并采集足够观测' : '';
  });
  document.querySelectorAll('[data-session-id], [data-range], #scenario, #advance-steps, #diagnosis-mode').forEach((el) => { el.disabled = state.busy; });
  window.AgentUI?.sync();
}
async function task(fn, label) {
  if (state.busy) return;
  $('error-banner').hidden = true;
  setBusy(true, label);
  try { await fn(); }
  catch (error) {
    showError(error);
    if (state.id) { try { await refresh(); } catch {} }
  } finally { setBusy(false); }
}
async function refresh(rid = state.id, stillCurrent = null) {
  if (!rid) return;
  const path = `/api/sessions/${encodeURIComponent(rid)}`;
  const [report, sessions] = await Promise.all([request(`${path}/report`), request('/api/sessions')]);
  const status = report.status;
  const rows = status.clock ? await request(`${path}/window?end=${status.clock}&size=${state.range}`) : [];
  if (stillCurrent && !stillCurrent()) return;
  // Commit a complete snapshot together so failed switches cannot mix sessions.
  state.id = rid; state.status = status; state.report = report; state.sessions = sessions.sessions;
  state.prediction = status.prediction || null; state.rows = rows;
  remember(rid);
  render();
}
async function createSession() {
  const value = await request('/api/sessions', 'POST', {scenario:$('scenario').value,seed:42});
  state.id = value.run_id; state.status = null; state.report = null; state.prediction = null; state.rows = [];
  remember(state.id);
}
async function demo() {
  await createSession();
  setBusy(true, '正在生成模拟观测');
  await request(endpoint('/advance'), 'POST', {steps:180});
  setBusy(true, '正在分析模拟工况');
  await request(endpoint('/diagnose'), 'POST', {mode:'offline'});
  await refresh();
  navigate('overview');
  toast('演示已就绪：查看诊断，再按需申请模拟处置。');
}
function navigate(view) {
  state.view = views[view] ? view : 'agent';
  document.querySelectorAll('.view').forEach((el) => { el.hidden = el.id !== `view-${state.view}`; });
  document.querySelectorAll('.nav-item').forEach((el) => { el.classList.toggle('active', el.dataset.view === state.view); el.setAttribute('aria-current', el.dataset.view === state.view ? 'page' : 'false'); });
  const [name,title,description] = views[state.view];
  $('breadcrumb-title').textContent = name;
  $('page-title').innerHTML = `${esc(title)}<span class="title-dot">.</span>`;
  $('page-description').textContent = description;
  $('demo-button').dataset.action = state.view === 'agent' ? 'agent-demo' : 'demo';
  $('demo-button').innerHTML = `${icon('play')}${state.view === 'agent' ? '新建闭环演示' : '运行诊断演示'}`;
  if (location.hash !== `#${state.view}`) history.replaceState(null, '', `#${state.view}`);
  closeSidebar();
}
const mobileLayout = window.matchMedia('(max-width: 720px)');
function syncSidebarAccessibility() {
  const open = mobileLayout.matches && $('sidebar').classList.contains('open');
  $('sidebar').inert = mobileLayout.matches && !open;
  document.querySelector('.app-shell').inert = open;
  $('sidebar-backdrop').hidden = !open;
}
function closeSidebar() {
  const wasOpen=$('sidebar').classList.contains('open');
  $('sidebar').classList.remove('open'); $('menu-toggle').setAttribute('aria-expanded','false');
  syncSidebarAccessibility();
  if(wasOpen && mobileLayout.matches) $('menu-toggle').focus();
}
function render() {
  const s = state.status;
  $('session-meta').innerHTML = s ? `<span class="status-dot ${s.detection.alert ? 'warning' : ''}"></span>${esc(scenarioNames[s.scenario] || s.scenario)} · ${esc(s.run_id.slice(0,8))} · ${esc(stateNames[s.state] || s.state)}` : '<span class="status-dot"></span>等待开始模拟';
  renderWorkflow(); renderMetrics(); renderChart(); renderAssistant(); renderActions(); renderTimeline(); renderDiagnosis(); renderSessions(); renderReport(); renderOptimization(); window.AgentUI?.render(); syncButtons();
}
function renderWorkflow() {
  const s = state.status;
  const step = !s?.clock ? 0 : s.state === 'RESOLVED' ? 5 : ['ACTION_EXECUTED','VERIFYING'].includes(s.state) ? 4 : s.state === 'AWAITING_CONFIRMATION' || s.capabilities.request_action ? 3 : s.diagnosis && !s.diagnosis_stale ? 2 : s.detection.alert ? 2 : 1;
  $('workflow').innerHTML = ['采集观测','检测异常','证据诊断','人工确认','独立复核'].map((name,index) => `<div class="workflow-step ${index < step ? 'done' : index === step ? 'current' : ''}" ${index === step ? 'aria-current="step"' : ''}><span class="step-index">${index < step ? icon('check') : `0${index + 1}`}</span><span>${name}</span></div>`).join('');
}
function sparkline(key, color) {
  const rows = state.status?.observations || [];
  const values = rows.map((r) => r[key]).filter(Number.isFinite);
  if (values.length < 2) return '';
  const min = Math.min(...values), span = Math.max(...values) - min || .01;
  let path = '', active = false;
  rows.forEach((r,i) => { if (!Number.isFinite(r[key])) { active = false; return; } path += `${active ? 'L':'M'}${(i/Math.max(rows.length-1,1)*72).toFixed(1)},${(26-(r[key]-min)/span*22).toFixed(1)} `; active = true; });
  return `<svg class="metric-spark" viewBox="0 0 74 30" aria-hidden="true"><path d="${path}" fill="none" stroke="${color}" stroke-width="1.4"/></svg>`;
}
function renderMetrics() {
  const s = state.status, d = s?.detection || {}, has = !!s?.clock;
  const complete = has ? d.completeness ?? (s.observations.filter((r) => r.pressure != null && r.flow != null).length / s.observations.length) : null;
  const metrics = [
    {label:'供浆压力',icon:'gauge',value:fmt(d.pressure_mean),unit:'均值',tag:!has || d.pressure_mean == null ? '待观测' : d.pressure_mean > 1.17 ? '高于阈值' : '阈值内',tone:!has || d.pressure_mean == null ? 'unknown' : d.pressure_mean > 1.17 ? 'warning' : '',note:'模拟上限 1.170',spark:sparkline('pressure','#0d927c')},
    {label:'供浆流量',icon:'drop',value:fmt(d.flow_mean),unit:'均值',tag:!has || d.flow_mean == null ? '待观测' : d.flow_mean < .84 ? '低于阈值' : '阈值内',tone:!has || d.flow_mean == null ? 'unknown' : d.flow_mean < .84 ? 'warning' : '',note:'模拟下限 0.840',spark:sparkline('flow','#6687cc')},
    {label:'数据完整率',icon:'shield',value:complete == null ? '—' : fmt(complete*100,0),unit:'%',tag:!has ? '待采集' : complete >= .8 ? '数据可用' : '数据不足',tone:!has ? 'unknown' : complete < .8 ? 'warning' : '',note:has ? `${d.valid_count ?? '—'} / ${d.sample_count ?? '—'} 个有效观测` : '最近 30 分钟窗口'},
    {label:'模拟运行时长',icon:'clock',value:has ? String(s.clock) : '—',unit:'min',tag:has ? '模拟时钟' : '未开始',tone:has ? '' : 'unknown',note:s ? `固定种子 ${s.seed ?? 42} · 可复现` : '选择工况即可开始'}
  ];
  $('metrics').innerHTML = metrics.map((m) => `<article class="metric"><div class="metric-top"><span>${m.label}</span>${icon(m.icon)}</div><div class="metric-value">${m.value}<span class="metric-unit">${m.unit}</span></div><div class="metric-bottom"><span class="metric-tag ${m.tone}">${m.tag}</span><span>${m.note}</span></div>${m.spark || ''}</article>`).join('');
}
function renderChart() {
  const rows = state.rows, width = Math.max(300,$('chart').clientWidth-20), height = Math.max(210,$('chart').clientHeight-16), left = 43, right = 16, top = 18, bottom = 31;
  const colors = {pressure:'#0d927c',flow:'#6687cc',coating_proxy:'#c79a53'};
  const allValues = rows.flatMap((r) => [...state.series].map((k) => r[k])).filter(Number.isFinite);
  const min = Math.floor(Math.min(.6,...allValues)*5)/5, max = Math.ceil(Math.max(1.4,...allValues)*5)/5;
  const x = (i) => left + i/Math.max(rows.length-1,1)*(width-left-right);
  const y = (v) => top+(max-v)/(max-min)*(height-top-bottom);
  let svg = `<svg id="trend-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="最近 ${state.range} 模拟分钟趋势，压力阈值1.17、流量阈值0.84"><defs><linearGradient id="pressure-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#138d76" stop-opacity=".10"/><stop offset="100%" stop-color="#138d76" stop-opacity="0"/></linearGradient><clipPath id="plot-clip"><rect x="${left}" y="${top}" width="${width-left-right}" height="${height-top-bottom}"/></clipPath></defs>`;
  for (let i=0;i<5;i++) { const v = min+(max-min)*i/4; svg += `<line x1="${left}" x2="${width-right}" y1="${y(v)}" y2="${y(v)}" stroke="#e9eee9" stroke-dasharray="3 5"/><text x="${left-10}" y="${y(v)+3}" text-anchor="end" fill="#9ba79c" font-size="9">${v.toFixed(2)}</text>`; }
  if (rows.length) {
    if (state.series.has('pressure')) svg += `<line x1="${left}" x2="${width-right}" y1="${y(1.17)}" y2="${y(1.17)}" stroke="#b8cdbf" stroke-dasharray="5 5"/><text x="${width-right-3}" y="${y(1.17)-5}" text-anchor="end" fill="#9eae9b" font-size="8">压力上限 1.17</text>`;
    if (state.series.has('flow')) svg += `<line x1="${left}" x2="${width-right}" y1="${y(.84)}" y2="${y(.84)}" stroke="#c4cde0" stroke-dasharray="5 5"/><text x="${width-right-3}" y="${y(.84)-5}" text-anchor="end" fill="#9ca9bf" font-size="8">流量下限 0.84</text>`;
    for (const key of state.series) {
      let path = '', active = false;
      rows.forEach((row,i) => { if (!Number.isFinite(row[key])) { active=false; return; } path += `${active ? 'L' : 'M'}${x(i).toFixed(2)},${y(row[key]).toFixed(2)} `; active=true; });
      if (key === 'pressure' && rows.length > 1 && rows.every((r) => Number.isFinite(r[key]))) svg += `<path d="${path}L${x(rows.length-1)},${height-bottom}L${left},${height-bottom}Z" fill="url(#pressure-fill)" clip-path="url(#plot-clip)"/>`;
      svg += `<path d="${path}" fill="none" stroke="${colors[key]}" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round" clip-path="url(#plot-clip)"/>`;
      // Isolated valid readings in a missing-data window remain visible; gaps are never joined.
      if (rows.some((r) => !Number.isFinite(r[key])) || rows.length === 1) rows.forEach((r,i) => { if (Number.isFinite(r[key])) svg += `<circle cx="${x(i)}" cy="${y(r[key])}" r="2" fill="${colors[key]}"/>`; });
    }
    const marks = Math.min(width < 450 ? 4 : 6,rows.length);
    for (let i=0;i<marks;i++) { const index = Math.round(i/Math.max(marks-1,1)*(rows.length-1)); svg += `<text x="${x(index)}" y="${height-9}" fill="#9ba79c" text-anchor="middle" font-size="9">${rows[index].minute} min</text>`; }
    const actions = (state.report?.audit || []).filter((e) => e.event === 'action_executed');
    for (const action of actions) { const minute = action.detail?.before?.clock; const index=rows.findIndex((r) => r.minute >= minute); if (index >= 0 && minute >= rows[0].minute) svg += `<line x1="${x(index)}" x2="${x(index)}" y1="${top}" y2="${height-bottom}" stroke="#b89563" stroke-dasharray="3 4"/><text x="${Math.min(x(index)+5,width-73)}" y="${top+8}" fill="#a88b65" font-size="8">模拟清理</text>`; }
  }
  svg += '</svg><div id="chart-tooltip" class="chart-tooltip" hidden></div>';
  if (!rows.length) svg += `<div class="chart-empty">${icon('activity')}<span>从一段模拟观测开始</span><small>运行诊断演示，或创建会话后推进模拟</small></div>`;
  $('chart').innerHTML = svg;
  $('chart-caption').textContent = rows.length ? `${rows[0].minute}–${rows.at(-1).minute} min · ${rows.length} 个观测点 · 缺失值保留断点` : '等待模拟观测';
  document.querySelectorAll('[data-range]').forEach((el) => { el.classList.toggle('active', Number(el.dataset.range) === state.range); el.setAttribute('aria-pressed', String(Number(el.dataset.range) === state.range)); });
  document.querySelectorAll('[data-series]').forEach((el) => { el.classList.toggle('active', state.series.has(el.dataset.series)); el.setAttribute('aria-pressed', String(state.series.has(el.dataset.series))); });
  const chart = $('trend-svg');
  chart.addEventListener('pointermove',(event) => {
    if (!rows.length) return;
    const rect=chart.getBoundingClientRect(), fraction=((event.clientX-rect.left)/rect.width*width-left)/(width-left-right);
    const row=rows[Math.max(0,Math.min(rows.length-1,Math.round(fraction*(rows.length-1))))];
    const tooltip=$('chart-tooltip'); tooltip.hidden=false;
    tooltip.innerHTML=`<strong>模拟第 ${row.minute} 分钟</strong><br>压力 ${fmt(row.pressure)}<br>流量 ${fmt(row.flow)}<br>涂布量 ${fmt(row.coating_proxy)}`;
    const container=$('chart').getBoundingClientRect();
    tooltip.style.left=`${Math.max(5,Math.min(event.clientX-container.left+14,container.width-145))}px`; tooltip.style.top='12px';
  });
  chart.addEventListener('pointerleave',() => { $('chart-tooltip').hidden=true; });
}
function predictionMarkup() {
  const result = state.prediction, s=state.status;
  if (!result) return `<div class="prediction-box"><div class="prediction-heading"><span>未来 30 分钟 · 模拟事件评分</span><button class="text-button" data-action="predict" data-capability="predict">评估 ${icon('arrow')}</button></div><p class="prediction-note">仅在模拟训练域内参考，不代表真实故障概率。</p></div>`;
  if (!result.available) return `<div class="prediction-box"><div class="prediction-heading"><span>暂不提供预测</span></div><p class="prediction-note">${esc(result.reason)}</p></div>`;
  const stale=result.clock !== s.clock || !s.capabilities.predict;
  return `<div class="prediction-box"><div class="prediction-heading"><span>${stale ? '历史' : '模拟事件'}评分 · ${result.clock} min</span><strong class="prediction-value">${fmt(result.score,3)}</strong></div><div class="prediction-track"><span style="width:${Math.max(0,Math.min(100,result.score*100))}%"></span></div><p class="prediction-note">${stale ? '工况已变化，此评分不代表当前状态。' : '分值 0–1，仅模拟训练域参考，非真实故障概率。'}</p><button class="text-button" data-action="predict" data-capability="predict">重新评估 ${icon('arrow')}</button></div>`;
}
function renderAssistant() {
  const s=state.status, d=s?.diagnosis;
  $('mode-badge').textContent = d?.mode === 'LLM' ? 'LLM + 证据' : '规则引擎';
  $('evidence-count').textContent = d?.evidence?.length || 0;
  let html;
  if (!d) html=`<div class="assistant-intro">你好，我是供浆诊断助手。</div><p class="assistant-description">我会结合压力、流量与知识证据，帮助你理解异常，并给出下一步检查建议。</p><div class="assistant-hint">${icon('info')}<span>${s?.clock >= 30 ? '观测已就绪，点击下方按钮开始分析。' : '至少需要 30 分钟观测。也可以运行演示，一次完成采集与诊断。'}</span></div>`;
  else html=`${s.diagnosis_stale ? '<div class="stale-note">观测已更新，以下为历史诊断，请重新分析。</div>' : ''}<span class="finding-status ${esc(d.severity || '')}"><span class="status-dot"></span>${d.severity === 'normal' ? '未见阈值异常' : d.severity === 'unknown' ? '先检查数据质量' : '发现需要关注的信号'}</span><p class="finding-summary">${esc(d.summary || s.detection.reason)}</p>${d.candidates.slice(0,2).map((c,i) => `<div class="mini-candidate"><span class="candidate-number">${i+1}</span><span>${esc(c.cause)}</span></div>`).join('')}<p class="assistant-description">${esc(d.next_step)}</p><button class="text-button diagnosis-detail-link" data-view="diagnosis">查看诊断与 ${d.evidence.length} 条证据 ${icon('arrow')}</button>`;
  $('assistant-content').innerHTML = html + (s?.clock ? predictionMarkup() : '');
}
function renderActions() {
  const s=state.status;
  const items=[['shield','01 · 人工确认','满足异常条件后，确认清理当前模拟流路。','申请模拟清理','request-action','request_action'],['check','02 · 独立复核','续跑 30 分钟，用新观测确认处置效果。','执行后复核','verify','verify'],['sliders','03 · 参数实验','恢复后对比 5 组泵速，结果不自动应用。','运行参数实验','optimize','optimize']];
  $('action-content').innerHTML = `<div class="action-steps">${items.map(([symbol,title,description,label,action,cap]) => `<div class="action-step"><span class="action-icon">${icon(symbol)}</span><h3>${title}</h3><p>${esc(s?.capability_reasons?.[cap] || description)}</p><button class="button small" data-action="${action}" data-capability="${cap}">${label}</button></div>`).join('')}</div>`;
}
function eventDetail(event) {
  const d=event.detail || {};
  if (event.event === 'created') return `${scenarioNames[d.scenario] || '模拟工况'} · 种子 ${d.seed ?? 42}`;
  if (event.event === 'advance') return `推进 ${d.steps} 分钟 · 模拟时钟 ${d.clock} min`;
  if (event.event === 'diagnosis') return `${d.mode || '规则模板'} · ${d.clock} min`;
  if (event.event === 'verification') return `${stateNames[d.state] || d.state} · ${d.before_clock} → ${d.after_clock} min`;
  if (d.clock != null) return `模拟时钟 ${d.clock} min`;
  if (event.event === 'action_executed') return `作用于模拟器 · 待独立窗口复核`;
  return '已写入本地审计记录';
}
function renderTimeline() {
  const events = state.report?.audit || [];
  $('event-count').textContent = `${events.length} 条记录`;
  $('timeline').innerHTML = !events.length ? '<p class="empty-small">尚无运行记录。<br>创建会话后，关键步骤会自动记录在这里。</p>' : `<div class="timeline">${events.slice(-8).reverse().map((event) => `<div class="timeline-item"><span class="timeline-dot"></span><div class="timeline-event">${esc(eventNames[event.event] || event.event)}<small>${esc(eventDetail(event))}</small></div><time class="timeline-time">${new Date(event.timestamp*1000).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',hour12:false})}</time></div>`).join('')}</div>`;
}
function renderDiagnosis() {
  const s=state.status,d=s?.diagnosis;
  if (!d) { $('diagnosis-detail').innerHTML=`<section class="panel empty-panel">${icon('spark')}<h2>第一份诊断，从观测开始</h2><p>采集至少 30 分钟观测，然后使用规则引擎或已配置的大模型分析。</p><button class="button primary" data-action="diagnose" data-capability="diagnose">分析当前工况</button></section>`; return; }
  $('diagnosis-detail').innerHTML=`<section class="panel diagnosis-header">${s.diagnosis_stale ? '<div class="stale-note">历史诊断：模拟观测已变化，请重新分析后再决定处置。</div>' : ''}<div class="diagnosis-meta"><span class="finding-status ${esc(d.severity || '')}">${esc(d.mode)}</span><span>${d.evidence.length} 条可追溯证据</span><span>${d.model_calls ?? 0} 次模型调用</span><button class="text-button" data-action="diagnose" data-capability="diagnose">重新分析 ${icon('arrow')}</button></div><h2>${esc(d.summary || s.detection.reason)}</h2><p>${esc(d.uncertainty)}</p>${d.warning ? `<p class="stale-note">${esc(d.warning)}</p>` : ''}</section><div class="diagnosis-columns"><section class="panel diagnosis-candidates"><h2>候选原因与待补证据</h2>${d.candidates.length ? d.candidates.map((c,i) => `<article class="candidate-card"><h3>${i+1}. ${esc(c.cause)}</h3><p>${esc(c.missing)}</p><div class="evidence-chips">${c.evidence_ids.map((id) => `<a class="evidence-chip" href="#evidence-${esc(id)}">${esc(id)}</a>`).join('')}</div></article>`).join('') : '<p class="empty-small">当前没有足以支持故障原因的候选证据。</p>'}<article class="candidate-card"><h3>下一步建议</h3><p>${esc(d.next_step)}</p></article></section><section class="panel evidence-list"><h2>证据链</h2>${d.evidence.map((e) => `<article class="evidence-item" id="evidence-${esc(e.id)}"><strong>${esc(e.id)} · ${esc(e.title || e.kind)}</strong><p>${e.data ? esc(`窗口 ${e.data.sample_count ?? '—'} 点，有效 ${e.data.valid_count ?? '—'} 点；压力均值 ${fmt(e.data.pressure_mean)}，流量均值 ${fmt(e.data.flow_mean)}。`) : esc(e.text)}</p>${sourceMarkup(e.source)}${e.limitation ? `<small>${esc(e.limitation)}</small>` : ''}</article>`).join('')}</section></div><section class="panel trace-panel"><h2>结构化诊断结果</h2><details class="raw-details"><summary>展开原始结果与引用 ID</summary><pre>${esc(JSON.stringify(d,null,2))}</pre></details></section>`;
}
function sourceMarkup(source) {
  if (!source) return '';
  try { const url = new URL(source); if (['https:','http:'].includes(url.protocol)) return `<a href="${esc(url.href)}" target="_blank" rel="noopener noreferrer">查看公开来源 ${icon('link')}</a>`; } catch {}
  return `<div class="source-path">项目来源：${esc(source)}</div>`;
}
function renderKnowledge() {
  const query=$('knowledge-search').value.trim().toLocaleLowerCase();
  const cards=state.cards.filter((card) => !query || `${card.id} ${card.title} ${card.text} ${card.kind}`.toLocaleLowerCase().includes(query));
  $('knowledge-grid').innerHTML=cards.length ? cards.map((card) => `<article class="panel knowledge-card"><div class="knowledge-card-header"><span class="knowledge-id">${esc(card.id)}</span><span class="knowledge-kind">${esc(card.kind)}</span></div><h2>${esc(card.title)}</h2><p>${esc(card.text)}</p><div class="knowledge-card-footer"><span>v${esc(card.version || '1.0')} · ${esc(card.subsystem || '供浆')}</span><span>${esc(card.limitation || '不可作为现场操作依据')}</span></div>${sourceMarkup(card.source)}</article>`).join('') : '<div class="panel empty-panel"><h2>没有匹配的知识卡片</h2><p>试试“压力”“泵”或“数据”。</p></div>';
}
function renderSessions() {
  $('recent-sessions').innerHTML=state.sessions.length ? state.sessions.slice(0,5).map((s) => `<button class="recent-session ${s.run_id === state.id ? 'selected' : ''}" data-session-id="${esc(s.run_id)}" title="${esc(s.run_id)}"><span></span>${esc(scenarioNames[s.scenario] || s.scenario)}<small>${s.clock} m</small></button>`).join('') : '<p class="sidebar-empty">创建会话后在这里继续</p>';
  $('session-table').innerHTML=state.sessions.length ? `<div class="table-wrap"><table><thead><tr><th>会话 / 工况</th><th>模拟时钟</th><th>流程状态</th><th>操作</th></tr></thead><tbody>${state.sessions.map((s) => `<tr class="${s.run_id === state.id ? 'selected-row' : ''}"><td>${esc(scenarioNames[s.scenario] || s.scenario)}<br><small class="muted">${esc(s.run_id.slice(0,8))}</small></td><td>${s.clock} min</td><td>${esc(stateNames[s.state] || s.state)}</td><td><button class="button small" data-session-id="${esc(s.run_id)}">${s.run_id === state.id ? '当前会话' : '继续会话'}</button></td></tr>`).join('')}</tbody></table></div>` : '<p class="empty-small">还没有模拟会话。</p>';
}
function renderReport() {
  const s=state.status;
  $('report-summary').innerHTML=s ? `<h2>当前会话概览</h2><div class="report-stats"><div class="report-stat">模拟工况<strong>${esc(scenarioNames[s.scenario])}</strong></div><div class="report-stat">当前流程状态<strong>${esc(stateNames[s.state])}</strong></div><div class="report-stat">证据条数<strong>${s.diagnosis?.evidence?.length || 0} 条</strong></div><div class="report-stat">审计记录<strong>${state.report?.audit?.length || 0} 条</strong></div></div><details class="raw-details"><summary>预览完整 JSON 报告</summary><pre>${esc(JSON.stringify(state.report,null,2))}</pre></details>` : '<p class="empty-small">创建或选择会话后，即可查看和导出报告。</p>';
}
function renderOptimization() {
  const value=state.status?.optimization, panel=$('optimization-panel');
  panel.hidden=!value;
  if (!value) return;
  panel.innerHTML=`<h2>模拟参数实验结果 <span class="subtle-badge">仅比较 · 未应用</span></h2><p>${value.clock != null ? `实验时钟 ${value.clock} min · ` : ''}目标涂布量代理 ${fmt(value.target)}；每个候选使用 3 组相同扰动比较，目标值越低越好。</p><div class="table-wrap"><table><thead><tr><th>模拟泵速</th><th>目标值均值</th><th>目标值波动</th><th>约束检查</th><th>对照结果</th></tr></thead><tbody>${value.candidates.map((c) => `<tr class="${c.pump === value.recommendation?.pump ? 'selected-row' : ''}"><td>${fmt(c.pump)}</td><td>${fmt(c.objective_mean,4)}</td><td>${fmt(c.objective_std,4)}</td><td>${c.feasible ? '满足模拟约束' : '未满足约束'}</td><td>${c.pump === value.recommendation?.pump ? '推荐候选' : c.pump === value.baseline?.pump ? '当前基线' : '对照候选'}</td></tr>`).join('')}</tbody></table></div><p>${esc(value.limitation)}。${value.recommendation ? '结果不会自动修改当前会话参数。' : '本次没有满足约束的候选。'}</p>`;
}
async function confirmAction() {
  const permit=await request(endpoint('/request-action'),'POST');
  await refresh();
  const dialog=$('confirm-dialog'), expires=Date.now()+(permit.expires_seconds || 120)*1000;
  $('confirmation-error').hidden=true;
  $('confirmation-summary').innerHTML=`工况：${esc(scenarioNames[state.status.scenario])}<br>会话：${esc(state.id.slice(0,8))} · ${state.status.clock} min<br>动作：模拟流路阻力清理`;
  const approved=await new Promise((resolve) => {
    let done=false;
    const finish=(answer) => { if(done)return; done=true; clearInterval(timer); dialog.close(); dialog.removeEventListener('cancel',cancel); $('cancel-action').onclick=null; $('approve-action').onclick=null; resolve(answer); };
    const cancel=(event) => {event.preventDefault();finish(false);};
    const tick=() => {const seconds=Math.max(0,Math.ceil((expires-Date.now())/1000));$('confirm-expiry').textContent=`确认将在 ${seconds} 秒后失效`;$('approve-action').disabled=seconds===0;if(seconds===0)finish(false);};
    const timer=setInterval(tick,1000);
    $('cancel-action').onclick=() => finish(false); $('approve-action').onclick=() => finish(true); dialog.addEventListener('cancel',cancel); dialog.showModal(); tick();
  });
  await request(endpoint('/confirm'),'POST',{token:permit.token,approve:approved});
  await refresh();
  toast(approved ? '模拟清理已执行，请点击“执行后复核”确认效果。' : '已取消本次模拟处置。');
}
async function download(format) {
  let content,type;
  if (format === 'json') {content=JSON.stringify(await request(endpoint('/report')),null,2);type='application/json';}
  else {const response=await fetch(endpoint('/report.md'));if(!response.ok)throw new Error('报告下载失败，请重试。');content=await response.text();type='text/markdown;charset=utf-8';}
  const url=URL.createObjectURL(new Blob([content],{type}));
  const link=document.createElement('a');link.href=url;link.download=`coating-report-${state.id.slice(0,8)}.${format}`;document.body.append(link);link.click();link.remove();setTimeout(() => URL.revokeObjectURL(url),1000);
  toast('报告已生成并下载。');
}
const actions = {
  demo,
  create:async () => {await createSession();await refresh();navigate('overview');toast('已创建会话，可推进模拟或运行诊断演示。');},
  advance:async () => {await request(endpoint('/advance'),'POST',{steps:Number($('advance-steps').value)});await refresh();toast('观测已更新，请依据当前窗口重新分析。');},
  diagnose:async () => {await request(endpoint('/diagnose'),'POST',{mode:$('diagnosis-mode').value});await refresh();toast('诊断完成，证据与建议已更新。');},
  predict:async () => {const value=await request(endpoint('/prediction'));await refresh();state.prediction=value;renderAssistant();syncButtons();},
  'request-action':confirmAction,
  verify:async () => {const value=await request(endpoint('/verify'),'POST');await refresh();toast(value.state === 'RESOLVED' ? '独立窗口复核通过，模拟工况已恢复。' : '复核尚未确认恢复，请查看最新诊断。');},
  optimize:async () => {await request(endpoint('/optimize'),'POST',{target:1,seed:42});await refresh();$('optimization-panel').scrollIntoView({behavior:'smooth',block:'center'});toast('参数对照实验完成，未应用任何参数。');},
  'download-md':() => download('md'),'download-json':() => download('json'),
  refresh:async () => {state.sessions=(await request('/api/sessions')).sessions;if(state.id)await refresh();else renderSessions();},
};
document.addEventListener('click',(event) => {
  const button=event.target.closest('button');
  if (!button || button.disabled) return;
  if (button.dataset.view) {navigate(button.dataset.view);return;}
  if (button.dataset.action === 'dismiss-error') {$('error-banner').hidden=true;return;}
  if (button.dataset.action === 'agent-stop') {window.AgentUI?.stop();return;}
  if (button.dataset.action && actions[button.dataset.action]) {task(actions[button.dataset.action],button.dataset.action==='diagnose' ? '正在诊断取证' : '正在处理');return;}
  if (button.dataset.sessionId) {task(async () => {await refresh(button.dataset.sessionId);$('scenario').value=state.status.scenario;navigate('overview');},'恢复会话中');return;}
  if (button.dataset.range) {task(async () => {state.range=Number(button.dataset.range);if(state.status?.clock)state.rows=await request(`${endpoint('/window')}?end=${state.status.clock}&size=${state.range}`);renderChart();},'加载观测窗口');return;}
  if (button.dataset.series) {const key=button.dataset.series;if(state.series.has(key) && state.series.size>1)state.series.delete(key);else state.series.add(key);renderChart();}
});
$('knowledge-search').addEventListener('input',renderKnowledge);
$('menu-toggle').addEventListener('click',() => {const open=$('sidebar').classList.toggle('open');$('menu-toggle').setAttribute('aria-expanded',String(open));syncSidebarAccessibility();if(open)$('sidebar').querySelector('.nav-item.active').focus();});
mobileLayout.addEventListener('change',closeSidebar);
$('sidebar-backdrop').addEventListener('click',closeSidebar);
document.addEventListener('keydown',(event) => {if(event.key==='Escape' && !$('confirm-dialog').open)closeSidebar();});
window.addEventListener('hashchange',() => {const view=location.hash.slice(1);if(views[view])navigate(view);});
let chartSize = '';
new ResizeObserver(() => {
  const size = `${$('chart').clientWidth}:${$('chart').clientHeight}`;
  if ($('chart').clientWidth && size !== chartSize) {chartSize=size;requestAnimationFrame(renderChart);}
}).observe($('chart'));
async function initialize() {
  render(); navigate(location.hash.slice(1));
  setBusy(true,'正在连接本地服务');
  try {
    const [config,sessions,knowledge]=await Promise.all([request('/api/config'),request('/api/sessions'),request('/api/knowledge')]);
    state.config=config;state.sessions=sessions.sessions;state.cards=knowledge.cards;
    const option=$('diagnosis-mode').querySelector('[value="llm"]');option.disabled=!config.llm_available;option.textContent=config.llm_available ? 'LLM 诊断 · 已配置' : 'LLM 诊断 · 未配置';
    let saved;try{saved=localStorage.getItem('coating-session');}catch{}
    if(saved){state.id=saved;try{await refresh();$('scenario').value=state.status.scenario;}catch(error){if(error.status===404){state.id=null;state.status=null;state.report=null;state.rows=[];try{localStorage.removeItem('coating-session');}catch{}}else throw error;}}
    render();renderKnowledge();
  } catch(error){showError(error);}
  finally{setBusy(false);if(!state.cards.length)$('connection').textContent='服务连接异常';}
}
initialize();
