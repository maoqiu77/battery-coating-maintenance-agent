from coating_agent.predict import label, features, split_assets

def test_future_labels_and_censoring():
    assert label(70, 90, 100) == 1
    assert label(50, 90, 100) == 0
    assert label(90, 90, 100) is None
    assert label(80, None, 100) is None
    assert label(70, None, 100) == 0

def test_features_only_use_supplied_window():
    rows = [{"pressure": 1., "flow": 1., "pump": 1.} for _ in range(30)]
    assert features(rows) == features(rows[:])
    assert len(features(rows)) == 8

def test_asset_splits_disjoint():
    train, val, test = split_assets(list(range(40)), seed=42)
    assert not (set(train) & set(val) or set(val) & set(test) or set(train) & set(test))
    assert len(train) + len(val) + len(test) == 40
