import ctypes

import pytest

from assembly_recovery import power_guard


def test_windows_power_structure_has_fixed_abi_size():
    assert ctypes.sizeof(power_guard.PowerStatus) == 12


def test_idle_sleep_request_is_released_even_when_child_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(power_guard, "system_power_status", lambda: {"platform": "windows", "ac_line_status": 1})
    monkeypatch.setattr(power_guard, "_execution_state", lambda flags: calls.append(flags) or power_guard.CONTINUOUS)
    with pytest.raises(RuntimeError, match="child failure"), power_guard.system_awake() as record:
        assert record["idle_sleep_prevented"]
        raise RuntimeError("child failure")
    assert calls == [power_guard.CONTINUOUS | power_guard.SYSTEM_REQUIRED, power_guard.CONTINUOUS]
    assert record["released"]


def test_battery_refuses_before_setting_power_request(monkeypatch):
    calls = []
    monkeypatch.setattr(power_guard, "system_power_status", lambda: {"platform": "windows", "ac_line_status": 0})
    monkeypatch.setattr(power_guard, "_execution_state", lambda flags: calls.append(flags))
    with pytest.raises(RuntimeError, match="AC power"), power_guard.system_awake():
        pytest.fail("Battery-powered run must not start")
    assert calls == []
