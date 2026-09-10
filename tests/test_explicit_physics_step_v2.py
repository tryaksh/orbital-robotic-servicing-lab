import json
from pathlib import Path

import pytest

from assembly_recovery.explicit_physics_step_v2 import explicit_physics_step_v2
from assembly_recovery.protocol import sha256


@pytest.mark.parametrize("dt", [1 / 120, 1 / 240])
def test_explicit_native_integration_and_fetch_without_scene_mutation(dt):
    calls = []

    class Interface:
        def simulate(self, native_dt, current_time):
            calls.append(("simulate", native_dt, current_time))

        def fetch_results(self):
            calls.append(("fetch_results",))

    explicit_physics_step_v2(Interface(), dt=dt, current_time=4.)
    assert calls == [("simulate", dt, 4.), ("fetch_results",)]


@pytest.mark.parametrize("dt,render", [(1 / 60, False), (1 / 240, True)])
def test_unregistered_or_rendering_step_rejected_before_backend_call(dt, render):
    with pytest.raises(ValueError):
        explicit_physics_step_v2(object(), dt=dt, current_time=0., update_fabric=render)


def test_revised_physics_registration_reruns_both_resolutions_and_preserves_failures():
    root = Path(__file__).resolve().parents[1]
    r = json.loads((root / "configs/learned_physics_validation_v3.json").read_text())
    assert len(r["run_order"]) == 6
    assert r["run_order"][0]["compatibility_required"] and r["run_order"][1]["compatibility_required"]
    assert r["prior_probe_cost"] == 38738.
    assert r["expected_charged_reference_transitions"] == 114807.
    assert r["total_session_expected_charged_reference_transitions"] == 153545.
    for p, h in r["source_sha256"].items():
        assert sha256(root / p) == h, p
