from coating_agent.optimize import optimize
from coating_agent.simulator import Simulator

def test_optimization_does_not_mutate_session():
    s = Simulator(42, "normal"); s.advance(60)
    before = s.snapshot()
    result = optimize(s, 1.0, seed=5)
    assert s.snapshot() == before
    assert all(.8 <= c["pump"] <= 1.2 for c in result["candidates"])
    assert result["baseline"]

def test_no_feasible_candidate():
    s = Simulator(42, "normal"); s.advance(60)
    assert optimize(s, 1, seed=5, max_pressure=.01)["recommendation"] is None
