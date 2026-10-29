import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from coating_agent.service import Service


@pytest.mark.parametrize('scenario,actions', [
    ('normal', []), ('blockage', ['clean_blockage']),
    ('sensor_bias', ['calibrate_pressure']), ('pump_loss', ['service_pump']),
    ('missing', ['restore_acquisition']), ('mixed', ['clean_blockage', 'calibrate_pressure']),
])
def test_six_scenarios_follow_observed_evidence_and_independent_verification(scenario, actions):
    s = Service(':memory:')
    rid = s.create(42, scenario)
    result = s.agent_start(rid)
    actual = []
    while result['agent']['status'] == 'WAITING_APPROVAL':
        assert len(actual) < 4
        approval = result['agent']['pending_approval']
        actual.append(approval['action'])
        assert approval['version'] == result['version']
        assert result['agent']['decision']['source'] == 'rules'
        result = s.agent_resume(rid, approval['id'], True)
    assert actual == actions
    assert result['agent']['status'] == ('RESOLVED' if actions else 'MONITORING')
    assert result['agent']['model_calls'] == 0
    assert result['agent']['notifications']
    tools = [item['detail'].get('tool') for item in result['agent']['trace'] if item['node'] == 'tool']
    assert 'get_statistics' in tools and 'search_knowledge' in tools
    for effect in result['agent']['effects']:
        assert effect['after']['sample_count'] == 20
        assert effect['control']['sample_count'] == 20
        assert effect['clock'] - effect['before_clock'] == 30
    if actions:
        assert result['agent']['effects'][-1]['recovered']
        assert not result['agent']['effects'][-1]['after']['alert']
        assert result['agent']['effects'][-1]['control']['alert']
        assert not result['capabilities']['predict']
    if scenario == 'mixed':
        assert not result['agent']['effects'][0]['recovered']
        assert any(item['node'] == 'replan' for item in result['agent']['trace'])
    assert s.report(rid)['agent'] == result['agent']


def test_failed_action_is_not_claimed_recovered_and_is_not_repeated():
    s = Service(':memory:')
    rid = s.create(42, 'blockage')
    start = s.agent_start(rid)
    result = s.agent_resume(rid, start['agent']['pending_approval']['id'], True, success=False)
    assert result['agent']['status'] == 'ESCALATED'
    assert len(result['agent']['effects']) == 1
    effect = result['agent']['effects'][0]
    assert effect['recovered'] is False
    assert effect['after'] == effect['control']
    assert result['agent']['pending_approval'] is None
    assert '同一' in result['agent']['notifications'][-1]['message']


def test_max_cycles_bounds_mixed_fault_replanning():
    s = Service(':memory:')
    rid = s.create(42, 'mixed')
    start = s.agent_start(rid, max_cycles=1)
    result = s.agent_resume(rid, start['agent']['pending_approval']['id'], True)
    assert result['agent']['status'] == 'ESCALATED'
    assert result['agent']['cycle'] == 1
    assert len(result['agent']['effects']) == 1
    assert result['agent']['pending_approval'] is None


def test_read_only_objective_notifies_without_requesting_actions():
    s = Service(':memory:')
    rid = s.create(42, 'blockage')
    result = s.agent_start(rid, objective='只诊断，不执行任何设备动作')
    assert result['agent']['status'] == 'ESCALATED'
    assert result['agent']['decision']['intent'] == 'notify_operator'
    assert result['agent']['pending_approval'] is None
    assert not result['agent']['effects']
    assert '只诊断' in result['agent']['notifications'][-1]['message']


def test_rejection_stops_without_alternative_approval():
    s = Service(':memory:')
    rid = s.create(42, 'blockage')
    start = s.agent_start(rid)
    result = s.agent_resume(rid, start['agent']['pending_approval']['id'], False)
    assert result['clock'] == start['clock']
    assert result['agent']['status'] == 'ESCALATED'
    assert result['agent']['pending_approval'] is None
    assert not result['agent']['effects']
    assert '拒绝' in result['agent']['notifications'][-1]['message']


def test_manual_actions_cannot_bypass_agent_approval_or_reuse_old_manual_token():
    s = Service(':memory:')
    rid = s.create(42, 'blockage')
    s.advance(rid, 180)
    s.diagnose(rid)
    old_token = s.request_action(rid)
    start = s.agent_start(rid)
    assert start['agent']['status'] == 'WAITING_APPROVAL'
    for action in [lambda: s.advance(rid, 1), lambda: s.diagnose(rid), lambda: s.request_action(rid),
                   lambda: s.confirm_action(rid, old_token, True), lambda: s.verify(rid)]:
        with pytest.raises(ValueError):
            action()
    assert s.status(rid)['clock'] == start['clock']
    assert not any(event['event'] == 'action_executed' for event in s.report(rid)['audit'])


def test_concurrent_agent_approval_executes_once():
    s = Service(':memory:')
    rid = s.create(42, 'blockage')
    start = s.agent_start(rid)
    approval = start['agent']['pending_approval']['id']

    def attempt():
        try:
            s.agent_resume(rid, approval, True)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(lambda _: attempt(), range(2))) == 1
    assert len(s.agent_status(rid)['effects']) == 1
    assert sum(event['event'] == 'action_executed' for event in s.report(rid)['audit']) == 1


def test_expired_agent_approval_is_visible_without_writes_and_never_executes():
    s = Service(':memory:')
    rid = s.create(42, 'blockage')
    start = s.agent_start(rid)
    approval = start['agent']['pending_approval']['id']
    s.sessions[rid]['agent']['pending_approval']['expires_at'] = 0
    writes = []
    s.db.set_trace_callback(writes.append)
    state = s.status(rid)
    assert state['agent']['status'] == 'ESCALATED'
    assert state['agent']['approval_expired'] is True
    assert state['agent']['pending_approval'] is None
    assert not any(sql.startswith(('INSERT', 'UPDATE', 'REPLACE')) for sql in writes)
    with pytest.raises(ValueError):
        s.agent_resume(rid, approval, True)
    assert not s.agent_status(rid)['effects']
    s.advance(rid, 1)
    assert s.status(rid)['state'] == 'ALERTED'


def test_restart_invalidates_agent_approval_and_keeps_trace(tmp_path):
    path = str(tmp_path / 'audit.sqlite3')
    s = Service(path)
    rid = s.create(42, 'blockage')
    start = s.agent_start(rid)
    approval = start['agent']['pending_approval']['id']
    restored = Service(path)
    result = restored.agent_status(rid)
    assert result['status'] == 'ESCALATED'
    assert result['pending_approval'] is None
    assert result['trace'] == start['agent']['trace']
    with pytest.raises(ValueError):
        restored.agent_resume(rid, approval, True)
    assert not restored.agent_status(rid)['effects']


def test_stop_revokes_approval_and_slow_planner_cannot_overwrite_new_run(monkeypatch):
    import coating_agent.planner as planner
    entered, release = Event(), Event()
    original = planner.plan

    def slow_plan(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        return original(*args, **kwargs)

    monkeypatch.setattr(planner, 'plan', slow_plan)
    s = Service(':memory:')
    rid = s.create(42, 'blockage')
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(s.agent_start, rid)
        assert entered.wait(2)
        try:
            assert pool.submit(s.agent_status, rid).result(timeout=1)['status'] == 'RUNNING'
            stopped = pool.submit(s.agent_stop, rid).result(timeout=1)
            assert stopped['agent']['status'] == 'STOPPED'
            monkeypatch.setattr(planner, 'plan', original)
            restarted = s.agent_start(rid)
            new_run = restarted['agent']['run_id']
        finally:
            release.set()
        pending.result(timeout=2)
    assert s.agent_status(rid)['run_id'] == new_run
    assert s.agent_status(rid)['status'] == 'WAITING_APPROVAL'
    approval = s.agent_status(rid)['pending_approval']['id']
    s.agent_stop(rid)
    with pytest.raises(ValueError):
        s.agent_resume(rid, approval, True)


def test_no_hidden_simulator_state_in_agent_or_report():
    s = Service(':memory:')
    rid = s.create(42, 'mixed')
    start = s.agent_start(rid)
    s.agent_resume(rid, start['agent']['pending_approval']['id'], True)
    payload = s.report(rid)
    forbidden = {'z', 'eta', 'random_state', 'failure_z', 'event_time', '_cleaned', '_observations', '_execution'}

    def inspect(value):
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(payload)
    assert '_observation_target' not in json.dumps(payload)


def test_llm_configuration_failure_is_explicit_not_rule_fallback(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('COATING_LLM_MODEL', raising=False)
    s = Service(':memory:')
    rid = s.create(42, 'blockage')
    result = s.agent_start(rid, mode='llm')
    assert result['agent']['status'] == 'ESCALATED'
    assert result['agent']['planner_source'] == 'llm_unavailable'
    assert result['agent']['pending_approval'] is None
    assert not result['agent']['effects']
    decision = result['agent']['decision']
    assert decision['warning'] == result['agent']['notifications'][-1]['message']
    assert s.report(rid)['agent']['decision'] == decision
    assert next(item['detail'] for item in result['agent']['trace']
                if item['node'] == 'decision') == decision


def test_executor_rejects_plan_without_independent_action_evidence(monkeypatch):
    import coating_agent.planner as planner
    monkeypatch.setattr(planner, 'plan', lambda *args, **kwargs: {
        'intent': 'request_approval', 'action': 'clean_blockage', 'reason': 'untrusted action',
        'evidence_ids': ['O1'], 'source': 'llm', 'model_calls': 1, 'tool_calls': 0, 'evidence': [],
    })
    s = Service(':memory:')
    rid = s.create(42, 'blockage')
    result = s.agent_start(rid, mode='llm')
    assert result['agent']['status'] == 'ESCALATED'
    assert result['agent']['pending_approval'] is None
    assert not result['agent']['effects']


def test_repeated_monitor_runs_advance_time_until_emerging_fault_is_detected():
    s = Service(':memory:')
    rid = s.create(42, 'blockage')
    result = s.agent_start(rid, observation_steps=30)
    assert result['clock'] == 30 and result['agent']['status'] == 'MONITORING'
    result = s.agent_start(rid, observation_steps=30)
    assert result['clock'] == 60
    for _ in range(4):
        if result['agent']['status'] == 'WAITING_APPROVAL':
            break
        before = result['clock']
        result = s.agent_start(rid, observation_steps=30)
        assert result['clock'] == before + 30
    assert result['agent']['status'] == 'WAITING_APPROVAL'
    assert result['agent']['pending_approval']['action'] == 'clean_blockage'


def test_first_manual_takeover_reuses_window_but_next_run_gets_new_observations():
    s = Service(':memory:')
    rid = s.create(42, 'normal')
    s.advance(rid, 180)
    result = s.agent_start(rid, observation_steps=30)
    assert result['clock'] == 180
    result = s.agent_start(rid, observation_steps=30)
    assert result['clock'] == 210


def test_monitoring_after_resolution_collects_fresh_window():
    s = Service(':memory:')
    rid = s.create(42, 'blockage')
    start = s.agent_start(rid)
    resolved = s.agent_resume(rid, start['agent']['pending_approval']['id'], True)
    assert resolved['agent']['status'] == 'RESOLVED'
    monitored = s.agent_start(rid, observation_steps=30)
    assert monitored['clock'] == resolved['clock'] + 30
    assert monitored['agent']['status'] == 'MONITORING'


@pytest.mark.parametrize('restart', [False, True])
def test_stop_keeps_request_audit_and_blocks_late_model_response(monkeypatch, restart):
    from langchain_core.messages import AIMessage
    import langchain_openai

    entered, release = Event(), Event()
    requests = []

    class DelayedModel:
        def __init__(self, **kwargs):
            pass

        def bind_tools(self, *args, **kwargs):
            return self

        def invoke(self, messages, **kwargs):
            requests.append(1)
            entered.set()
            assert release.wait(4)
            return AIMessage(content='', tool_calls=[{
                'name': 'get_statistics', 'args': {}, 'id': 'late-call', 'type': 'tool_call',
            }])

    monkeypatch.setattr(langchain_openai, 'ChatOpenAI', DelayedModel)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-not-a-real-key')
    monkeypatch.setenv('COATING_LLM_MODEL', 'test-model')
    s = Service(':memory:')
    rid = s.create(42, 'blockage')
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(s.agent_start, rid, 'llm')
        assert entered.wait(2)
        try:
            running = s.agent_status(rid)
            old_run = running['run_id']
            assert running['model_calls'] == running['model_attempts'] == 1
            assert running['model_completed_calls'] == 0
            s.agent_stop(rid)
            if restart:
                restarted = s.agent_start(rid, mode='rules', observation_steps=30)
                new_run = restarted['agent']['run_id']
        finally:
            release.set()
        pending.result(timeout=2)
    assert len(requests) == 1
    agent = s.agent_status(rid)
    if restart:
        assert agent['run_id'] == new_run and agent['model_calls'] == 0
        assert any(event['event'] == 'agent_late_event' and event['detail']['agent_run_id'] == old_run
                   for event in s.report(rid)['audit'])
    else:
        assert agent['status'] == 'STOPPED'
        assert agent['model_attempts'] == agent['model_completed_calls'] == 1
        assert not any(item['node'] == 'tool' for item in agent['trace'])
    assert agent['pending_approval'] is None if not restart else agent['pending_approval'] is not None
