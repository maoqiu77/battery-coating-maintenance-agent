'use strict';

// The orchestration lives on the server; this view never decides or auto-approves an action.
window.AgentUI = (() => {
  const statuses = {IDLE:'待启动',RUNNING:'正在执行',WAITING_APPROVAL:'等待你的确认',RESOLVED:'已验证恢复',MONITORING:'本轮观察完成',ESCALATED:'已交接人工',STOPPED:'已停止'};
  const intents = {continue_monitoring:'继续观察',collect_more_data:'补充观测',request_approval:'申请处置',notify_operator:'通知用户',escalate:'交接人工',finish:'完成本轮'};
  const labels = {clean_blockage:'清理供浆阻力',calibrate_pressure:'校准压力传感器',service_pump:'维护供浆泵',restore_acquisition:'恢复采集链路'};
  const toolLabels = {get_statistics:'读取观测统计',search_knowledge:'检索设备知识',inspect_pressure:'独立压力检查',inspect_pump:'泵性能检查',inspect_acquisition:'采集链路检查',submit_decision:'提交下一步计划'};
  const sourceNames = {rules:'规则编排',llm:'LLM 决策',llm_unavailable:'LLM 未完成决策',runtime:'编排程序',simulator:'模拟采集',observation_tool:'观测工具',detection_rules:'阈值检测',knowledge_tool:'知识检索',inspection_tool:'独立检查',approval_gate:'人工审批门禁',simulation_action:'模拟执行',verification:'独立复核',verification_tool:'独立复核',user:'用户操作'};
  let modeInitialized = false, pollGeneration = 0, stopping = false, displayedRunId = null;
  let polling = false, lastPollAt = 0;
  const agent = () => state.status?.agent || state.report?.agent || null;
  const expired = (a) => !!a?.pending_approval && Date.now() >= a.pending_approval.expires_at * 1000;
  const active = (a) => a && ['RUNNING','WAITING_APPROVAL'].includes(a.status) && !expired(a);
  const empty = (symbol,title,body) => `<div class="agent-empty">${icon(symbol)}<strong>${esc(title)}</strong><p>${esc(body)}</p></div>`;
  const sourceName = (name) => sourceNames[name] || name || '待运行';

  function sync() {
    const a=agent(), available=!!state.config.llm_available;
    const option=$('agent-mode').querySelector('[value="llm"]');
    option.disabled=!available;
    option.textContent=available ? `LLM · ${state.config.llm_model}` : 'LLM 决策 · 未配置';
    if (!modeInitialized && Object.hasOwn(state.config,'llm_available')) { $('agent-mode').value=available?'llm':'rules'; modeInitialized=true; }
    if(a?.run_id && (displayedRunId!==a.run_id || active(a))) {
      $('agent-mode').value=a.mode;
      $('agent-objective').value=a.objective;
      displayedRunId=a.run_id;
    }
    $('agent-start').disabled=state.busy || active(a);
    $('agent-start').innerHTML=`${icon('play')}${a?.run_id && !active(a) ? '再次启动 Agent' : '启动 Agent 闭环'}`;
    $('agent-stop').disabled=stopping || !active(a);
    $('agent-mode').disabled=state.busy || active(a);
    $('agent-objective').disabled=state.busy || active(a);
    document.querySelectorAll('[data-agent-preset]').forEach((el)=>{el.disabled=state.busy || active(a);});
    document.querySelectorAll('[data-action="agent-approve"],[data-action="agent-reject"]').forEach((el)=>{el.disabled=state.busy || !a?.pending_approval || expired(a);});
    if($('demo-button').dataset.action==='agent-demo') $('demo-button').disabled=state.busy || active(a);
    if(active(a)) document.querySelectorAll('[data-action="create"],[data-action="demo"]').forEach((el)=>{el.disabled=true;el.title='请先结束或停止当前 Agent 任务';});
    $('agent-launch-note').textContent=available ? '模型可自主调用只读工具；模拟设备动作仍需逐项确认。' : '尚未配置模型，离线规则可演示编排闭环，但不代表大模型决策。';
  }

  function render() {
    const a=agent(), source=a?.planner_source || 'rules';
    const sourceText=a?.status==='RUNNING' && a.mode==='llm' ? 'LLM · 运行中' : a?.run_id ? sourceName(source) : state.config.llm_available ? `LLM · ${state.config.llm_model}` : '规则编排 · 离线可用';
    $('agent-source-badge').textContent=sourceText;
    $('agent-source-badge').className=`agent-mode-badge ${source==='llm' ? 'llm' : source==='llm_unavailable' ? 'unavailable' : ''}`;
    const status=a?.status || 'IDLE';
    $('agent-status-strip').className=`agent-status-strip ${status==='WAITING_APPROVAL'?'waiting':status==='ESCALATED'?'error':''}`;
    $('agent-status-strip').innerHTML=`<span class="status-dot"></span><strong>${esc(statuses[status] || status)}</strong><span>${a?.run_id ? `任务 ${esc(a.run_id.slice(0,8))} · 模拟时钟 ${state.status.clock} min` : '选择工况，输入目标，即可开始一次有界运维任务。'}</span>${a?.run_id ? '<button class="text-button" data-action="agent-refresh">刷新任务状态</button>' : ''}`;
    renderFlow(a);renderStats(a);renderDecision(a);renderTrace(a);renderNotifications(a);renderEffects(a);
    sync();
  }

  function renderFlow(a) {
    const nodes=[['observe','观测','OBSERVE'],['detect','检测','DETECT'],['gather','取证','INSPECT'],['plan','规划','PLAN'],['approval','审批','APPROVE'],['execute','执行','ACT'],['verify','复核','VERIFY']];
    const seen=new Set((a?.trace || []).map((x)=>x.node));
    if(seen.has('inspect') || seen.has('tool'))seen.add('gather');
    const current=a?.status==='WAITING_APPROVAL'?'approval':a?.status==='RUNNING'?(a.trace?.at(-1)?.node || 'observe'):a?.status==='RESOLVED'?'verify':a?.status==='ESCALATED'?'plan':null;
    $('agent-flow').innerHTML=nodes.map(([key,label,english])=>`<div class="agent-flow-node ${key===current ? 'current' : seen.has(key) ? 'done' : ''}">${label}<small>${english}</small></div>`).join('');
  }

  function renderStats(a) {
    const stats=[['规划轮次',a?.cycle || 0,'轮',`最多 ${a?.max_cycles || 4} 轮，未解决则交接人工`],['模型请求次数',a?.model_attempts ?? a?.model_calls ?? 0,'次',a?.mode==='llm'?`收到 ${a?.model_completed_calls ?? 0} 次响应，含失败请求尝试`:'离线规则，不调用大模型'],['工具调用',a?.tool_calls || 0,'次','包含观测、检索、检查和执行'],['已执行动作',a?.effects?.length || 0,'项','每项均经确认并独立复核']];
    $('agent-summary-stats').innerHTML=stats.map(([name,value,unit,note])=>`<article class="agent-stat"><span>${name}</span><strong>${value}<small>${unit}</small></strong><em>${esc(note)}</em></article>`).join('');
    $('agent-round').textContent=a?.cycle ? `第 ${a.cycle} 轮规划` : '尚未规划';
  }

  function renderDecision(a) {
    const d=a?.decision, pending=a?.pending_approval;
    if(!d){$('agent-decision').innerHTML=empty('spark',a?.status==='RUNNING'?'正在读取观测与证据':'等待一个明确的目标',a?.status==='RUNNING'?'实际工具调用和决策会逐步记录在下方。':'例如：处理当前供浆异常；或者只分析原因并通知我。');return;}
    let title=labels[d.action] || intents[d.intent] || '分析当前证据';
    if(a.status==='RESOLVED')title='新观测已确认恢复';
    if(a.status==='MONITORING')title='当前未见异常，本轮观察结束';
    if(a.status==='STOPPED')title='任务已停止，未批准动作不再执行';
    const reason=a.status==='RESOLVED' ? `已完成 ${(a.effects || []).map((effect)=>labels[effect.action] || effect.label).join('、')}。处置后末 20 分钟的新观测窗口数据质量达标，压力与流量均通过模拟阈值复核。` : d.reason;
    let html=`<div class="agent-decision-body"><div class="agent-decision-title">${esc(title)}<span class="agent-intent-pill">${esc(sourceName(d.source))}</span></div><p class="agent-decision-reason">${esc(reason)}</p>${a.status==='RESOLVED'?`<details class="agent-tool-result"><summary>最后一轮执行前的决策理由</summary><p>${esc(d.reason)}</p></details>`:''}<div class="agent-evidence-chips">${(d.evidence_ids || []).map((id)=>`<span>${esc(id)}</span>`).join('')}</div>`;
    if(pending)html+=`<div class="agent-approval-box"><h3>${expired(a)?'本次审批已过期':'需要你的确认'}</h3><p>${esc(pending.label)}。批准后自动执行并复核；拒绝后将记录交接待办。</p><div class="agent-approval-actions"><button class="button primary" data-action="agent-approve">查看并批准动作</button><button class="button" data-action="agent-reject">拒绝并交接人工</button><small id="agent-approval-countdown"></small></div></div>`;
    else if(a.status==='RESOLVED')html+='<div class="finding-status normal">独立窗口复核通过 · 效果对照见下方</div>';
    else if(a.status==='ESCALATED')html+='<div class="agent-warning">任务已交接：请查看右侧通知中的原因和待办。系统不会继续循环尝试。</div>';
    else if(a.status==='RUNNING')html+='<span class="agent-running-indicator"><i class="live-dot"></i>正在执行下一节点</span>';
    if(a.evidence?.length)html+=`<details class="agent-tool-result"><summary>查看本轮实际取得的 ${a.evidence.length} 条证据</summary><pre>${esc(JSON.stringify(a.evidence,null,2))}</pre></details>`;
    html+=`<p class="agent-source-note">${a.mode==='llm'?'LLM 提交下一步计划，执行门禁独立校验观测与工具证据。':'本次为规则编排演示，决策由明确规则生成。'} 原因、动作和复核结果均保留在报告中。</p></div>`;
    $('agent-decision').innerHTML=html;
    updateCountdown();
  }

  function detailSummary(entry) {
    const d=entry.detail;
    if(typeof d==='string')return d;
    if(!d)return '';
    if(d.reason)return d.reason;
    if(d.summary)return d.summary;
    if(d.tool)return toolLabels[d.tool] || d.tool;
    if(d.action)return labels[d.action] || d.label || d.action;
    if(d.detection)return d.detection.reason;
    if(d.signal_pattern || d.quality)return d.reason || '已读取当前窗口的统计信息';
    if(d.steps)return `追加 ${d.steps} 分钟模拟观测`;
    if(d.samples != null)return `读取 ${d.samples} 个模拟观测点`;
    if(d.cycle)return `第 ${d.cycle} 轮 · ${sourceName(d.mode)}`;
    if(d.state)return stateNames[d.state] || d.state;
    return '已记录本次节点的执行结果';
  }
  function renderTrace(a) {
    const entries=a?.trace || [];
    $('agent-trace-count').textContent=`${entries.length} 个节点`;
    $('agent-trace').innerHTML=entries.length ? `<div class="agent-trace">${entries.map((entry,i)=>`<article class="agent-trace-item ${esc(entry.node)}"><span class="agent-trace-index">${i+1}</span><div class="agent-trace-content"><div class="agent-trace-heading"><span>${esc(entry.label || entry.node)}</span><small>${entry.clock} min · ${esc(sourceName(entry.source))}</small></div><div class="agent-trace-detail">${esc(detailSummary(entry))}</div>${entry.detail && typeof entry.detail==='object' ? `<details class="agent-tool-result"><summary>节点数据</summary><pre>${esc(JSON.stringify(entry.detail,null,2))}</pre></details>` : ''}</div></article>`).join('')}</div>` : empty('activity','流程尚未启动','开始后可看到 Agent 调用了什么、依据什么作出决策。');
  }

  function renderNotifications(a) {
    const notifications=a?.notifications || [];
    $('agent-notice-count').textContent=notifications.length;
    $('agent-notifications').innerHTML=notifications.length ? `<div class="agent-notifications">${notifications.slice().reverse().map((n)=>`<article class="agent-notice ${esc(n.level)}"><div class="agent-notice-heading"><span>${esc(n.title)}</span><small>${n.clock} min</small></div><p>${esc(n.message)}</p></article>`).join('')}</div>` : empty('info','暂无通知','需要审批、复核完成或需要人工接手时，Agent 会在这里通知你。');
  }
  function renderEffects(a) {
    const effects=a?.effects || [];
    if(!effects.length){$('agent-effects').innerHTML=empty('gauge','执行后才评估效果','动作完成不等于故障消失。系统会续跑 30 分钟，用末 20 分钟的独立窗口验证。');return;}
    const metrics=[['压力均值','pressure_mean',(v)=>fmt(v)],['流量均值','flow_mean',(v)=>fmt(v)],['数据完整率','completeness',(v)=>Number.isFinite(v)?`${fmt(v*100,0)}%`:'—']];
    $('agent-effects').innerHTML=`<div class="agent-effects-content">${effects.map((e,i)=>`<article class="agent-effect"><div class="agent-effect-heading"><strong>${i+1}. ${esc(e.label || labels[e.action] || e.action)}</strong><small>${e.before_clock} → ${e.clock} min · ${e.recovered?'复核恢复':'仍有异常，需继续判断'}</small></div><div class="table-wrap"><table><thead><tr><th>观测指标</th><th>处置前</th><th>处置后</th><th>本轮不处置</th></tr></thead><tbody>${metrics.map(([name,key,format])=>`<tr><td>${name}</td><td>${format(e.before?.[key])}</td><td><strong>${format(e.after?.[key])}</strong></td><td>${format(e.control?.[key])}</td></tr>`).join('')}<tr><td>窗口判定</td><td>${detectionLabel(e.before)}</td><td>${detectionLabel(e.after)}</td><td>${detectionLabel(e.control)}</td></tr></tbody></table></div><div class="agent-effect-conclusion">${e.recovered?'新的观测窗口达到模拟恢复条件。':'处置后仍有未消除的信号，Agent 将依据新证据重新规划，或交接人工。'} 对照保留此前已完成的动作，仅不执行本轮动作。此结果仅解释模拟效果，不代表实际产线收益。</div></article>`).join('')}</div>`;
  }
  function detectionLabel(d) {return !d?'—':d.quality!=='ok'?'数据不足':d.alert?'仍有异常':'阈值内';}
  function updateCountdown() {
    const pending=agent()?.pending_approval, element=$('agent-approval-countdown');
    if(!pending || !element)return;
    const remaining=Math.max(0,Math.ceil(pending.expires_at-Date.now()/1000));
    element.textContent=remaining ? `${remaining} 秒内有效` : '已过期，请重新启动取证';
    if(!remaining) {sync();void pollCurrent(true);}
  }

  async function pollCurrent(force=false) {
    const a=agent(), rid=state.id;
    if(polling || !rid || !a || !['RUNNING','WAITING_APPROVAL'].includes(a.status))return;
    if(!force && Date.now()-lastPollAt < (a.status==='RUNNING'?1100:5000))return;
    const generation=pollGeneration, runId=a.run_id;
    polling=true;lastPollAt=Date.now();
    try {
      const result=await request(`/api/sessions/${encodeURIComponent(rid)}/agent`);
      if(generation!==pollGeneration || rid!==state.id || agent()?.run_id!==runId)return;
      if(result.status!=='RUNNING' && result.status!==a.status) await refresh(rid,()=>generation===pollGeneration && rid===state.id && agent()?.run_id===runId);
      else if(state.status){state.status.agent=result;render();}
    } catch {
      if(expired(a) && rid===state.id) $('agent-status-strip').innerHTML='<span class="status-dot"></span><strong>审批已过期</strong><span>暂时无法同步服务端状态，请恢复连接后刷新。</span>';
    } finally {polling=false;}
  }
  async function withProgress(operation) {
    const generation=++pollGeneration;
    // Poll independently of this POST so reloading a running task also resumes updates.
    if(state.status)state.status.agent={...(agent() || {}),status:'RUNNING'};
    render();
    try{return await operation();}finally{if(generation===pollGeneration)pollGeneration++;}
  }
  async function start(fresh=false) {
    const objective=$('agent-objective').value.trim();
    if(!objective)throw new Error('请先输入这次 Agent 的运维目标。');
    const mode=$('agent-mode').value;
    if(fresh || !state.id)await createSession();
    await refresh(); navigate('agent');
    setBusy(true,mode==='llm'?'大模型正在取证规划':'规则编排正在运行');
    await withProgress(()=>request(endpoint('/agent/start'),'POST',{mode,objective,max_cycles:4,observation_steps:180}));
    await refresh();
    toast(agent()?.status==='WAITING_APPROVAL'?'Agent 已完成取证，正在等待你的确认。':`Agent：${statuses[agent()?.status] || '本轮已完成'}`);
  }
  async function approve() {
    const a=agent(), pending=a?.pending_approval;
    if(!pending || expired(a))throw new Error('审批已失效，请刷新状态并重新启动取证。');
    const rid=state.id, dialog=$('agent-confirm-dialog');
    $('agent-confirm-summary').innerHTML=`<strong>${esc(pending.label)}</strong><br>会话 ${esc(rid.slice(0,8))} · 第 ${a.cycle} 轮<br>${esc(a.decision?.reason)}`;
    let accepted=await new Promise((resolve)=>{
      let finished=false;
      const finish=(value)=>{if(finished)return;finished=true;clearInterval(timer);dialog.close();dialog.removeEventListener('cancel',cancel);$('agent-confirm-cancel').onclick=null;$('agent-confirm-approve').onclick=null;resolve(value);};
      const cancel=(event)=>{event.preventDefault();finish(false);};
      const tick=()=>{const remaining=Math.max(0,Math.ceil(pending.expires_at-Date.now()/1000));$('agent-confirm-expiry').textContent=`本次审批 ${remaining} 秒内有效，仅作用于模拟器。`;$('agent-confirm-approve').disabled=remaining===0;if(!remaining)finish(false);};
      const timer=setInterval(tick,1000);
      $('agent-confirm-cancel').onclick=()=>finish(false);$('agent-confirm-approve').onclick=()=>finish(true);dialog.addEventListener('cancel',cancel);dialog.showModal();tick();
    });
    if(!accepted){await refresh();return;}
    setBusy(true,'执行模拟动作并自动复核');
    await withProgress(()=>request(`/api/sessions/${encodeURIComponent(rid)}/agent/resume`,'POST',{approval_id:pending.id,approve:true}));
    await refresh();
    toast(agent()?.status==='RESOLVED'?'新窗口验证恢复，效果对照已生成。':agent()?.status==='WAITING_APPROVAL'?'尚未完全恢复，Agent 已提出下一项处置。':'本轮处理结束，请查看通知。');
  }
  async function reject() {
    const pending=agent()?.pending_approval;
    if(!pending)throw new Error('当前没有等待确认的动作。');
    await request(endpoint('/agent/resume'),'POST',{approval_id:pending.id,approve:false});
    await refresh();toast('动作已拒绝，已在工作台内记录人工交接待办。');
  }
  async function stop() {
    if(!state.id || stopping)return;
    stopping=true;sync();
    try {await request(endpoint('/agent/stop'),'POST');await refresh();toast('Agent 已停止，未批准动作不再执行。');}
    catch(error){showError(error);}finally{stopping=false;sync();}
  }

  actions['agent-start']=()=>start(false);
  actions['agent-demo']=()=>start(true);
  actions['agent-approve']=approve;
  actions['agent-reject']=reject;
  actions['agent-refresh']=refresh;
  document.addEventListener('click',(event)=>{
    const button=event.target.closest('[data-agent-preset]');if(!button || button.disabled)return;
    $('agent-objective').value=button.dataset.agentPreset==='diagnose'?'只诊断当前异常并通知我，不要申请或执行设备处置。':'发现供浆异常后检查证据，提出对应处置方案；等待我确认后执行并复核，未恢复则继续分析，无法处理时通知我。';
  });
  setInterval(updateCountdown,1000);
  setInterval(pollCurrent,1200);
  return {render,sync,stop};
})();
window.AgentUI.render();
