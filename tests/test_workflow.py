import pytest
from coating_agent.service import Service


def test_confirmed_action_branches_and_verifies():
    s = Service(":memory:"); rid = s.create(42, "blockage")
    s.advance(rid, 180); s.diagnose(rid)
    token = s.request_action(rid)
    before = s.status(rid)["clock"]
    s.confirm_action(rid, token, True)
    with pytest.raises(ValueError): s.confirm_action(rid, token, True)
    assert s.verify(rid)["state"] == "RESOLVED"
    assert s.status(rid)["clock"] > before
    assert any(a["event"] == "action_executed" for a in s.report(rid)["audit"])


def test_refusal_and_state_version_conflict():
    s = Service(":memory:"); rid = s.create(42, "blockage")
    s.advance(rid, 180); s.diagnose(rid); token = s.request_action(rid)
    before = s.status(rid)["clock"]
    s.confirm_action(rid, token, False)
    assert s.status(rid)["clock"] == before
    assert s.status(rid)["state"] == "UNRESOLVED"
    rid2 = s.create(43, "blockage"); s.advance(rid2, 180); s.diagnose(rid2)
    token2 = s.request_action(rid2); s.advance(rid2, 1)
    with pytest.raises(ValueError): s.confirm_action(rid2, token2, True)


def test_insufficient_data_cannot_request_action():
    s = Service(":memory:"); rid = s.create(42, "missing"); s.advance(rid, 90); s.diagnose(rid)
    with pytest.raises(ValueError): s.request_action(rid)

def test_failed_clean_keeps_alarm_and_expired_permit_is_rejected():
    s = Service(":memory:"); rid = s.create(42, "blockage")
    s.advance(rid, 180); s.diagnose(rid); token = s.request_action(rid)
    s.confirm_action(rid, token, True, False)
    assert s.verify(rid)["state"] == "UNRESOLVED"
    rid2 = s.create(42, "blockage"); s.advance(rid2, 180); s.diagnose(rid2)
    token2 = s.request_action(rid2); s.sessions[rid2]["token"]["expires"] = 0
    with pytest.raises(ValueError): s.confirm_action(rid2, token2, True)

def test_concurrent_confirm_only_executes_once():
    from concurrent.futures import ThreadPoolExecutor
    s = Service(":memory:"); rid = s.create(42, "blockage")
    s.advance(rid, 180); s.diagnose(rid); token = s.request_action(rid)
    def attempt():
        try:
            s.confirm_action(rid, token, True)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sum(pool.map(lambda _: attempt(), range(2))) == 1
    assert sum(x["event"] == "action_executed" for x in s.report(rid)["audit"]) == 1

def test_session_and_report_survive_restart(tmp_path):
    db = str(tmp_path / "audit.sqlite3")
    s = Service(db); rid = s.create(42, "blockage"); s.advance(rid, 180); s.diagnose(rid)
    before = s.status(rid)
    restored = Service(db)
    assert restored.status(rid) == before
    assert restored.report(rid)["diagnosis"] is not None


def test_reads_do_not_write_and_mutations_only_persist_their_session():
    s = Service(":memory:")
    rid = s.create(42, "blockage")
    other = s.create(43, "normal")
    statements = []
    s.db.set_trace_callback(statements.append)
    s.status(rid)
    s.report(rid)
    s.list_sessions()
    assert not any(sql.startswith(("INSERT", "UPDATE", "REPLACE")) for sql in statements)
    statements.clear()
    s.advance(rid, 60)
    writes = [sql for sql in statements if sql.startswith("INSERT OR REPLACE INTO sessions")]
    assert len(writes) == 1 and rid in writes[0] and other not in writes[0]


def test_advance_invalidates_diagnosis_and_confirmation():
    s = Service(":memory:")
    rid = s.create(42, "blockage")
    s.advance(rid, 180)
    s.diagnose(rid)
    assert s.status(rid)['capabilities']['request_action']
    s.advance(rid, 1)
    status = s.status(rid)
    assert status['state'] == 'ALERTED'
    assert status['diagnosis_stale']
    assert not status['capabilities']['request_action']
    with pytest.raises(ValueError):
        s.request_action(rid)
    s.diagnose(rid)
    assert not s.status(rid)['diagnosis_stale']
    assert s.status(rid)['capabilities']['request_action']


def test_action_requires_verification_and_prediction_exclusion_survives_restart(tmp_path):
    path = str(tmp_path / 'audit.sqlite3')
    s = Service(path)
    rid = s.create(42, 'blockage')
    s.advance(rid, 180)
    s.diagnose(rid)
    token = s.request_action(rid)
    s.confirm_action(rid, token, True)
    with pytest.raises(ValueError):
        s.diagnose(rid)
    with pytest.raises(ValueError):
        s.advance(rid, 1)
    assert s.status(rid)['capabilities']['verify']
    s.verify(rid)
    s.diagnose(rid)
    assert s.status(rid)['state'] == 'RESOLVED'
    assert not s.status(rid)['capabilities']['predict']
    restored = Service(path)
    assert not restored.status(rid)['capabilities']['predict']


def test_diagnosis_does_not_lock_reads_and_rejects_concurrent_state_change(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    import coating_agent.service as module
    entered, release = Event(), Event()
    original = module.diagnose

    def slow_diagnosis(window, mode):
        entered.set()
        assert release.wait(3)
        return original(window, mode)

    monkeypatch.setattr(module, 'diagnose', slow_diagnosis)
    s = Service(":memory:")
    rid = s.create(42, 'blockage')
    s.advance(rid, 180)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = pool.submit(s.diagnose, rid)
        assert entered.wait(2)
        try:
            assert pool.submit(s.status, rid).result(timeout=1)['clock'] == 180
            pool.submit(s.advance, rid, 1).result(timeout=1)
        finally:
            release.set()
        with pytest.raises(ValueError, match='状态已变化'):
            pending.result(timeout=2)
    assert s.status(rid)['diagnosis'] is None


def test_normal_diagnosis_freshness_and_empty_session():
    s = Service(":memory:")
    rid = s.create(42, 'normal')
    assert not s.status(rid)['diagnosis_stale']
    s.advance(rid, 90)
    s.diagnose(rid)
    assert not s.status(rid)['diagnosis_stale']
    assert not s.status(rid)['capabilities']['request_action']
    s.advance(rid, 30)
    assert s.status(rid)['state'] == 'OBSERVING'
    assert s.status(rid)['diagnosis_stale']
    s.diagnose(rid)
    assert not s.status(rid)['diagnosis_stale']


def test_expired_confirmation_is_visible_without_read_writes():
    import json
    s = Service(":memory:")
    rid = s.create(42, 'blockage')
    s.advance(rid, 180)
    s.diagnose(rid)
    token = s.request_action(rid)
    s.sessions[rid]['token']['expires'] = 0
    statements = []
    s.db.set_trace_callback(statements.append)
    status = s.status(rid)
    assert status['state'] == 'UNRESOLVED' and status['confirmation_expired']
    assert s.list_sessions()[0]['state'] == 'UNRESOLVED'
    assert s.report(rid)['status']['state'] == 'UNRESOLVED'
    assert not any(sql.startswith(('INSERT', 'UPDATE', 'REPLACE')) for sql in statements)
    assert token not in json.dumps(status)
    with pytest.raises(ValueError):
        s.confirm_action(rid, token, True)
    s.diagnose(rid)
    assert s.status(rid)['capabilities']['request_action']
    assert not s.status(rid)['confirmation_expired']


def test_restart_invalidates_pending_confirmation(tmp_path):
    path = str(tmp_path / 'audit.sqlite3')
    s = Service(path)
    rid = s.create(42, 'blockage')
    s.advance(rid, 180)
    s.diagnose(rid)
    token = s.request_action(rid)
    restored = Service(path)
    assert restored.status(rid)['state'] == 'UNRESOLVED'
    with pytest.raises(ValueError):
        restored.confirm_action(rid, token, True)
    assert not any(event['event'] == 'action_executed' for event in restored.report(rid)['audit'])
    restored.diagnose(rid)
    assert restored.status(rid)['capabilities']['request_action']


@pytest.mark.parametrize('scenario,expected_state', [('normal', 'OBSERVING'), ('blockage', 'ALERTED')])
def test_legacy_stale_diagnosis_migrates_on_restart(tmp_path, scenario, expected_state):
    import json
    path = str(tmp_path / 'audit.sqlite3')
    s = Service(path)
    rid = s.create(42, scenario)
    s.advance(rid, 180)
    s.diagnose(rid)
    s.advance(rid, 30)
    saved = json.loads(s.db.execute('SELECT payload FROM sessions WHERE run_id=?', (rid,)).fetchone()[0])
    saved['state'] = 'DIAGNOSED'
    for key in ('diagnosis_clock', 'has_execution', 'created_at'):
        saved.pop(key)
    s.db.execute('UPDATE sessions SET payload=? WHERE run_id=?', (json.dumps(saved), rid))
    s.db.commit()
    restored = Service(path)
    state = restored.status(rid)
    assert state['state'] == expected_state
    assert state['diagnosis_stale']
    assert not state['capabilities']['request_action']


def test_legacy_execution_audit_still_excludes_prediction(tmp_path):
    import json
    path = str(tmp_path / 'audit.sqlite3')
    s = Service(path)
    rid = s.create(42, 'blockage')
    s.advance(rid, 180)
    s.diagnose(rid)
    token = s.request_action(rid)
    s.confirm_action(rid, token, True)
    s.verify(rid)
    s.diagnose(rid)
    saved = json.loads(s.db.execute('SELECT payload FROM sessions WHERE run_id=?', (rid,)).fetchone()[0])
    saved.pop('has_execution')
    s.db.execute('UPDATE sessions SET payload=? WHERE run_id=?', (json.dumps(saved), rid))
    s.db.commit()
    restored = Service(path)
    assert not restored.status(rid)['capabilities']['predict']
