import pytest

from coating_agent.simulator import Simulator
from coating_agent.detection import detect


def test_seed_and_no_truth_in_observations():
    a, b = Simulator(42, "blockage"), Simulator(42, "blockage")
    a.advance(120); b.advance(120)
    assert a.window(120, 30) == b.window(120, 30)
    assert "z" not in a.window(120, 1)[0]
    assert "scenario" not in a.window(120, 1)[0]


def test_future_access_and_bad_windows_rejected():
    sim = Simulator(42, "normal")
    sim.advance(10)
    with pytest.raises(ValueError): sim.window(11, 1)
    with pytest.raises(ValueError): sim.window(10, 0)


def test_blockage_detected_and_clean_changes_future():
    sim = Simulator(42, "blockage")
    sim.advance(180)
    assert detect(sim.window(180, 30))["alert"]
    original = sim.clone(); sim.clean(True)
    sim.advance(30); original.advance(30)
    assert sim.window(210, 1)[0]["flow"] > original.window(210, 1)[0]["flow"]

def test_missing_flow_cannot_be_recovered_from_quality_proxy():
    sim = Simulator(42, "missing"); sim.advance(30)
    assert all(row["coating_proxy"] is None for row in sim.window(30, 30) if row["flow"] is None)


@pytest.mark.parametrize("scenario,action", [
    ("blockage", "clean_blockage"), ("sensor_bias", "calibrate_pressure"),
    ("pump_loss", "service_pump"), ("missing", "restore_acquisition"),
])
def test_targeted_action_restores_future_window_without_rewriting_history(scenario, action):
    sim = Simulator(42, scenario)
    sim.advance(180)
    history = sim.window(180, 180)
    assert detect(history[-30:])["alert"]
    sim.perform_action(action)
    assert sim.clock == 180 and sim.window(180, 180) == history
    sim.advance(30)
    after = detect(sim.window(210, 20))
    assert after["quality"] == "ok" and not after["alert"]
    # Repair remains effective after a round trip and continued observation.
    restored = Simulator.from_state(sim.export_state())
    restored.advance(180)
    assert not detect(restored.window(restored.clock, 30))["alert"]


def test_mixed_fault_requires_cleaning_and_pressure_calibration():
    sim = Simulator(42, "mixed")
    sim.advance(180)
    assert sim.inspection("inspect_pressure")["data"]["deviation"] > .15
    assert detect(sim.window(180, 30))["signal_pattern"] == "pressure_high_flow_low"
    sim.perform_action("clean_blockage")
    sim.advance(30)
    remaining = detect(sim.window(210, 20))
    assert remaining["alert"] and remaining["signal_pattern"] == "pressure_high"
    assert sim.inspection("inspect_pressure")["data"]["deviation"] > .15
    sim.perform_action("calibrate_pressure")
    sim.advance(30)
    assert not detect(sim.window(240, 20))["alert"]
    assert abs(sim.inspection("inspect_pressure")["data"]["deviation"]) < .15


@pytest.mark.parametrize("scenario,action", [
    ("blockage", "clean_blockage"), ("sensor_bias", "calibrate_pressure"),
    ("pump_loss", "service_pump"), ("missing", "restore_acquisition"),
])
def test_failed_actions_leave_state_and_following_trajectory_unchanged(scenario, action):
    sim = Simulator(42, scenario)
    sim.advance(180)
    original = sim.export_state()
    control = sim.clone()
    sim.perform_action(action, success=False)
    assert sim.export_state() == original
    sim.advance(30)
    control.advance(30)
    assert sim.window(210, 30) == control.window(210, 30)
    assert detect(sim.window(210, 20))["alert"]


def test_inspections_are_independent_measurements_and_do_not_consume_process_randomness():
    for scenario in ("normal", "blockage", "sensor_bias", "pump_loss", "missing", "mixed"):
        sim = Simulator(42, scenario)
        sim.advance(180)
        before = sim.export_state()
        checks = {name: sim.inspection(name) for name in
                  ("inspect_pressure", "inspect_pump", "inspect_acquisition")}
        assert sim.export_state() == before
        for name, result in checks.items():
            assert result == sim.inspection(name)
            assert result["clock"] == 180 and result["tool"] == name
            assert result["kind"] == "模拟检查" and result["limitation"]
            assert not {"scenario", "z", "eta", "event_time"}.intersection(result)
            assert not {"scenario", "z", "eta", "event_time"}.intersection(result["data"])
        if scenario in {"sensor_bias", "mixed"}:
            assert checks["inspect_pressure"]["data"]["deviation"] > .15
        if scenario == "pump_loss":
            assert checks["inspect_pump"]["data"]["efficiency_ratio"] < .9
        else:
            assert checks["inspect_pump"]["data"]["efficiency_ratio"] >= .9
        if scenario == "missing":
            assert checks["inspect_acquisition"]["data"]["completeness"] < .8
        else:
            assert checks["inspect_acquisition"]["data"]["completeness"] == 1


def test_no_observations_or_missing_sensor_do_not_fabricate_inspection_readings():
    sim = Simulator(42, "missing")
    assert sim.inspection("inspect_pressure")["data"]["deviation"] is None
    assert sim.inspection("inspect_pump")["data"]["efficiency_ratio"] is None
    assert sim.inspection("inspect_acquisition")["data"]["sample_count"] == 0
    sim.advance(181)
    assert sim.inspection("inspect_pressure")["data"]["measured_pressure"] is None
    assert sim.inspection("inspect_pressure")["data"]["deviation"] is None


def test_legacy_state_defaults_to_unrepaired_flags():
    sim = Simulator(42, "mixed")
    sim.advance(180)
    saved = sim.export_state()
    for key in ("_pressure_calibrated", "_pump_serviced", "_acquisition_restored"):
        saved.pop(key)
    restored = Simulator.from_state(saved)
    restored.advance(30)
    sim.advance(30)
    assert restored.window(210, 30) == sim.window(210, 30)


@pytest.mark.parametrize("scenario,digest", [
    ("normal", "e13e331bcf0de80cf4a5d5998ea2d1baf2ddb920d916048b6a0c2ef9c0551ff3"),
    ("blockage", "74a9332775de4253085012c730e7e57a02dd863cedd4b6d5f1e9c54b94f7c3ed"),
    ("sensor_bias", "6a04c71ab9c944385f0ce52f161186255b5de86714732346809334d9dd538ade"),
    ("pump_loss", "cd81ec45baf5b086184075db89921ad264b4ebfa0741ab26833fbc6e5cbc1532"),
    ("missing", "9cd6d5db3e26b77db3d6c1a81b98c036e7fb4f68c62c8603790ee09b22c8af4e"),
    ("mixed", "d14f5f35b20b85d847f9b6c07f00d760bbb68bdaff9d19d15fd7d63432f880b7"),
])
def test_unrepaired_trajectories_match_original_training_simulator(scenario, digest):
    import hashlib
    import json
    sim = Simulator(42, scenario)
    sim.advance(180)
    actual = hashlib.sha256(json.dumps(sim.window(180, 180), sort_keys=True).encode()).hexdigest()
    assert actual == digest


def test_unknown_actions_and_tools_are_rejected_without_mutation():
    sim = Simulator(42, "normal")
    before = sim.export_state()
    with pytest.raises(ValueError):
        sim.perform_action("write_plc")
    with pytest.raises(ValueError):
        sim.perform_action("clean_blockage", "false")
    with pytest.raises(ValueError):
        sim.inspection("read_hidden_truth")
    assert sim.export_state() == before


@pytest.mark.parametrize("scenario,action", [
    ("blockage", "clean_blockage"), ("sensor_bias", "calibrate_pressure"),
    ("pump_loss", "service_pump"), ("missing", "restore_acquisition"),
])
def test_action_and_counterfactual_share_process_disturbances(scenario, action):
    sim = Simulator(42, scenario)
    sim.advance(180)
    control = sim.clone()
    sim.perform_action(action)
    sim.advance(30)
    control.advance(30)
    assert sim.rng.getstate() == control.rng.getstate()
    treated = detect(sim.window(210, 20))
    untreated = detect(control.window(210, 20))
    assert not treated["alert"] and untreated["alert"]
    if action == "clean_blockage":
        assert treated["pressure_mean"] < untreated["pressure_mean"]
        assert treated["flow_mean"] > untreated["flow_mean"]
    elif action == "calibrate_pressure":
        assert untreated["pressure_mean"] - treated["pressure_mean"] == pytest.approx(.4, abs=.0001)
        assert treated["flow_mean"] == untreated["flow_mean"]
    elif action == "service_pump":
        assert treated["flow_mean"] > untreated["flow_mean"] + .1
    else:
        assert treated["completeness"] == 1 and untreated["completeness"] < .8
        # Existing shared sample times retain exactly the same sensor noise.
        shared = [(a, b) for a, b in zip(sim.window(210, 30), control.window(210, 30))
                  if b["flow"] is not None]
        assert all(a == b for a, b in shared)


def test_sequential_mixed_repairs_keep_each_counterfactual_aligned():
    sim = Simulator(42, "mixed")
    sim.advance(180)
    before_cleaning = sim.clone()
    sim.perform_action("clean_blockage")
    sim.advance(30)
    before_cleaning.advance(30)
    assert sim.rng.getstate() == before_cleaning.rng.getstate()
    treated = detect(sim.window(210, 20))
    untreated = detect(before_cleaning.window(210, 20))
    assert treated["signal_pattern"] == "pressure_high"
    assert untreated["signal_pattern"] == "pressure_high_flow_low"
    assert treated["flow_mean"] > untreated["flow_mean"]
    before_calibration = sim.clone()
    sim.perform_action("calibrate_pressure")
    sim.advance(30)
    before_calibration.advance(30)
    assert sim.rng.getstate() == before_calibration.rng.getstate()
    treated = detect(sim.window(240, 20))
    untreated = detect(before_calibration.window(240, 20))
    assert not treated["alert"] and untreated["signal_pattern"] == "pressure_high"
    assert treated["flow_mean"] == untreated["flow_mean"]
    assert untreated["pressure_mean"] - treated["pressure_mean"] == pytest.approx(.4, abs=.0001)
