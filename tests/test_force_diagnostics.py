import math

import pytest

from assembly_recovery.force_diagnostics import TemporalLoadMonitor


def test_filter_does_not_erase_raw_violation_from_a_short_pulse():
    monitor = TemporalLoadMonitor()
    monitor.observe(10, 1 / 120)
    result = monitor.observe(30, 1 / 120)
    assert result["raw_violation_seen"]
    assert not result["sustained_overload_seen"]
    for _ in range(20):
        result = monitor.observe(10, 1 / 120)
    assert result["raw_violation_seen"] and result["peak_raw_force_n"] == 30
    assert result["excess_impulse_n_s"] == pytest.approx(10 / 120)
    assert not result["sustained_overload_seen"]


def test_sustained_load_detection_uses_physical_time_across_step_sizes():
    alarm_times = []
    for dt in (1 / 120, 1 / 240):
        monitor = TemporalLoadMonitor()
        monitor.observe(10, dt)
        for step in range(1, math.ceil(0.2 / dt) + 1):
            if monitor.observe(30, dt)["sustained_overload_seen"]:
                alarm_times.append(step * dt)
                break
    assert len(alarm_times) == 2
    assert max(alarm_times) < 0.06
    assert abs(alarm_times[0] - alarm_times[1]) <= 1 / 120


@pytest.mark.parametrize("force,dt", [(float("nan"), 0.01), (-1, 0.01), (1, 0), (1, float("inf"))])
def test_invalid_diagnostic_samples_are_rejected(force, dt):
    with pytest.raises(ValueError):
        TemporalLoadMonitor().observe(force, dt)
