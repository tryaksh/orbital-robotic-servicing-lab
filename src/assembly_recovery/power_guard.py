"""Scoped Windows idle-sleep prevention for a bounded GPU process.

Microsoft documents that this does not override a lid close or power button:
https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate
"""
from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager

CONTINUOUS = 0x80000000
SYSTEM_REQUIRED = 0x00000001


class PowerStatus(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ubyte) for name in ("ac_line", "battery_flag", "battery_percent", "system_status")] + [
        ("battery_seconds", ctypes.c_uint32), ("full_battery_seconds", ctypes.c_uint32)]


def system_power_status():
    if os.name != "nt":
        return {"platform": "other", "ac_line_status": None, "battery_percent": None}
    status = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        raise OSError("Cannot verify system power status")
    return {"platform": "windows", "ac_line_status": status.ac_line, "battery_percent": status.battery_percent}


def _execution_state(flags):
    function = ctypes.windll.kernel32.SetThreadExecutionState
    function.argtypes = [ctypes.c_uint32]
    function.restype = ctypes.c_uint32
    return function(flags)


@contextmanager
def system_awake(*, require_ac=True):
    record = {"power_at_start": system_power_status(), "idle_sleep_prevented": False, "released": False,
              "scope": "Scoped idle-sleep request; no display requirement, persistent power-policy change, or lid/button override."}
    windows = record["power_at_start"]["platform"] == "windows"
    if windows and require_ac and record["power_at_start"]["ac_line_status"] != 1:
        raise RuntimeError("AC power is required for the bounded GPU restart")
    if windows:
        if not _execution_state(CONTINUOUS | SYSTEM_REQUIRED):
            raise OSError("Cannot prevent idle sleep for the bounded run")
        record["idle_sleep_prevented"] = True
    try:
        yield record
    finally:
        record["released"] = bool(_execution_state(CONTINUOUS)) if windows else True
