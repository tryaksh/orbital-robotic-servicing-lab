"""Read what an extraction actually died of, from the rows a play run recorded.

``play.py --grip_axis_metrics`` splits the grip error into the gripper's own
axes. The magnitude columns every certification already carries cannot say
whether a lost grip was the pads walking along the pin or the module swinging
off its axis, and those two have different fixes: the first is a feature on the
pin, the second is authority in the wrist.

Run over one or more ``.npz`` files written by ``play.py --episode_metrics``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> tuple[np.ndarray, list[str], dict]:
    data = np.load(path, allow_pickle=True)
    return data["rows"], [str(name) for name in data["fields"]], json.loads(str(data["metadata"]))


def _stats(values: np.ndarray) -> str:
    if values.size == 0:
        return "        --        --        --"
    return f"{np.median(values):10.4f}{np.percentile(values, 95):10.4f}{values.max():10.4f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", type=Path, nargs="+")
    parser.add_argument("--report", type=Path, default=None)
    arguments = parser.parse_args()

    payload: dict[str, object] = {}
    for path in arguments.rows:
        rows, fields, metadata = _load(path)
        index = {name: position for position, name in enumerate(fields)}
        success = rows[:, index["success"]] > 0.5
        print("=" * 78)
        section = metadata.get("stress", {}).get("blade_cross_section_m")
        print(f"{path.name}: {len(rows)} episodes, {success.mean() * 100:.2f}% success", end="")
        print(f", section {section}" if section else "")

        entry: dict[str, object] = {
            "episodes": int(len(rows)),
            "success_rate": float(success.mean()),
            "blade_cross_section_m": section,
        }
        columns = [name for name in fields if name.startswith(("grip_offset", "grip_attitude"))]
        columns += ["tool_to_handle_error_m", "tool_to_handle_orientation_rad"]
        columns += ["blade_linear_velocity_mps", "blade_angular_velocity_radps", "axial_error_m"]
        print(f"{'column':<34}{'succ p50':>10}{'p95':>10}{'max':>10}{'fail p50':>12}{'p95':>10}{'max':>10}")
        for name in columns:
            if name not in index:
                continue
            values = rows[:, index[name]]
            print(f"{name:<34}{_stats(values[success])}  {_stats(values[~success])}")
            entry[name] = {
                "success_p50": float(np.median(values[success])) if success.any() else None,
                "success_p95": float(np.percentile(values[success], 95)) if success.any() else None,
                "failure_p50": float(np.median(values[~success])) if (~success).any() else None,
                "failure_p95": float(np.percentile(values[~success], 95)) if (~success).any() else None,
                "failure_max": float(values[~success].max()) if (~success).any() else None,
            }
        payload[path.stem] = entry

    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {arguments.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
