from fastapi.testclient import TestClient
import pytest
from coating_agent import api
from coating_agent.api import app
from coating_agent.service import Service


@pytest.fixture(autouse=True)
def isolated_service(monkeypatch):
    local_service = Service(":memory:")
    monkeypatch.setattr(api, "service", local_service)
    yield local_service
    local_service.db.close()

def test_api_demo_and_clock_guard():
    c = TestClient(app)
    rid = c.post('/api/sessions', json={'seed': 42, 'scenario': 'blockage'}).json()['run_id']
    assert c.post(f'/api/sessions/{rid}/advance', json={'steps': 180}).status_code == 200
    assert c.get(f'/api/sessions/{rid}/window', params={'end': 181, 'size': 10}).status_code == 400
    assert c.post(f'/api/sessions/{rid}/diagnose', json={'mode': 'offline'}).status_code == 200
    prediction = c.get(f'/api/sessions/{rid}/prediction').json()
    if prediction['available']:
        assert c.get(f'/api/sessions/{rid}/report').json()['prediction']['target']
    token = c.post(f'/api/sessions/{rid}/request-action').json()['token']
    assert c.post(f'/api/sessions/{rid}/confirm', json={'token': token, 'approve': True}).status_code == 200
    assert c.post(f'/api/sessions/{rid}/verify').json()['state'] == 'RESOLVED'
    assert c.post(f'/api/sessions/{rid}/optimize', json={'target': 1, 'seed': 42}).status_code == 200
    assert c.get(f'/api/sessions/{rid}/report').json()['optimization']['baseline']
    assert c.get(f'/api/sessions/{rid}/prediction').json()['available'] is False
    assert c.get(f'/api/sessions/{rid}/report').status_code == 200
    markdown = c.get(f'/api/sessions/{rid}/report.md')
    assert markdown.status_code == 200 and '模拟' in markdown.text and '诊断' in markdown.text


def test_session_discovery_config_and_knowledge(monkeypatch):
    c = TestClient(app)
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-key")
    monkeypatch.setenv("COATING_LLM_MODEL", "test-model")
    config = c.get('/api/config')
    assert config.json()['llm_available'] is True
    assert 'private-test-key' not in config.text
    assert c.get('/api/health').json()['status'] == 'ok'
    assert c.get('/api/sessions').json() == {'sessions': []}
    rid = c.post('/api/sessions', json={'scenario': 'normal'}).json()['run_id']
    assert c.get('/api/sessions').json()['sessions'][0]['run_id'] == rid
    state = c.get(f'/api/sessions/{rid}').json()
    assert state['scenario'] == 'normal' and state['seed'] == 42
    assert state['capabilities']['advance'] and not state['capabilities']['diagnose']
    assert {card['id'] for card in c.get('/api/knowledge').json()['cards']} >= {'K01', 'K10', 'K11', 'K12', 'K13', 'K14'}
    assert c.get('/api/knowledge', params={'q': '手机屏幕'}).json()['cards'] == []


@pytest.mark.parametrize('payload', [{'scenario': 'unknown'}, {'seed': -1}])
def test_invalid_sessions_return_validation_error(payload):
    assert TestClient(app).post('/api/sessions', json=payload).status_code == 422


def test_api_validation_and_missing_session():
    c = TestClient(app)
    assert c.get('/api/sessions/not-present').status_code == 404
    rid = c.post('/api/sessions', json={}).json()['run_id']
    assert c.post(f'/api/sessions/{rid}/diagnose', json={'mode': 'unknown'}).status_code == 422
    assert c.get(f'/api/sessions/{rid}/window', params={'end': 1, 'size': 0}).status_code == 422
    assert c.post(f'/api/sessions/{rid}/optimize', json={'target': 10}).status_code == 422
    assert c.post(f'/api/sessions/{rid}/advance', json={'steps': 0}).status_code == 422
    assert c.post(f'/api/sessions/{rid}/confirm', json={'token': '无效', 'approve': True}).status_code == 422


def test_missing_observations_do_not_emit_prediction():
    c = TestClient(app)
    rid = c.post('/api/sessions', json={'scenario': 'missing'}).json()['run_id']
    c.post(f'/api/sessions/{rid}/advance', json={'steps': 90})
    response = c.get(f'/api/sessions/{rid}/prediction')
    assert response.status_code == 200
    assert response.json()['available'] is False
    assert 'score' not in response.json()


def test_agent_api_approval_replanning_report_and_stop():
    c = TestClient(app)
    rid = c.post('/api/sessions', json={'scenario': 'mixed'}).json()['run_id']
    base = f'/api/sessions/{rid}'
    assert c.get(base + '/agent').json()['status'] == 'IDLE'
    start = c.post(base + '/agent/start', json={'mode': 'rules', 'max_cycles': 4}).json()
    assert start['agent']['status'] == 'WAITING_APPROVAL'
    assert c.post(base + '/advance', json={'steps': 30}).status_code == 400
    approval = start['agent']['pending_approval']['id']
    next_step = c.post(base + '/agent/resume', json={'approval_id': approval, 'approve': True})
    assert next_step.status_code == 200
    next_agent = next_step.json()['agent']
    assert next_agent['pending_approval']['action'] == 'calibrate_pressure'
    assert c.post(base + '/agent/resume', json={'approval_id': approval, 'approve': True}).status_code == 400
    assert c.get(base + '/report').json()['agent']['effects'][0]['control']
    assert 'Agent 决策' in c.get(base + '/report.md').text
    assert c.post(base + '/agent/stop').json()['agent']['status'] == 'STOPPED'
    assert c.post(base + '/agent/resume', json={'approval_id': next_agent['pending_approval']['id'], 'approve': True}).status_code == 400


@pytest.mark.parametrize('payload', [{'mode': 'unknown'}, {'max_cycles': 7}, {'observation_steps': 10}, {'objective': ''}])
def test_agent_api_validates_budget_and_mode(payload):
    c = TestClient(app)
    rid = c.post('/api/sessions', json={}).json()['run_id']
    assert c.post(f'/api/sessions/{rid}/agent/start', json=payload).status_code == 422
