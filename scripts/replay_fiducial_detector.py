"""Compare the deployed fiducial fallback on preserved workflow RGB frames.

This is a detector-only replay: it does not run physics and it makes no pose or
workflow-success claim.  Each input image is hashed so a before/after result is
bound to exactly the same rendered observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

from zero_g_blade_swap.fiducial import (
    FIDUCIAL_DICTIONARY,
    FIDUCIAL_MARKER_ID,
    _marker_corners,
)


def _baseline_detects(image_rgb: np.ndarray) -> bool:
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.minMarkerPerimeterRate = 0.02
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(FIDUCIAL_DICTIONARY), parameters
    )
    gray = cv2.cvtColor(image_rgb[..., :3], cv2.COLOR_RGB2GRAY)
    _, identifiers, _ = detector.detectMarkers(gray)
    return identifiers is not None and int(FIDUCIAL_MARKER_ID) in identifiers.reshape(-1)


def _step(path: Path) -> int:
    match = re.search(r"step-(\d+)", path.name)
    if match is None:
        raise ValueError(f"frame name has no step number: {path.name}")
    return int(match.group(1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path)
    parser.add_argument("--late-step", type=int, default=1440)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    paths = sorted(args.frames.glob("step-*.png"), key=_step)
    if not paths:
        raise ValueError(f"no step-*.png frames under {args.frames}")
    rows = []
    for path in paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        baseline = _baseline_detects(rgb)
        try:
            _marker_corners(rgb)
            deployed = True
        except RuntimeError:
            deployed = False
        rows.append(
            {
                "file": path.name,
                "step": _step(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "baseline_detected": baseline,
                "fallback_detected": deployed,
            }
        )

    late = [row for row in rows if row["step"] >= args.late_step]
    if not late:
        raise ValueError(f"no frames at or after step {args.late_step}")
    result = {
        "status": "passed" if all(row["fallback_detected"] for row in rows) else "failed",
        "title": "Fiducial decoder replay on continuous late-approach RGB frames",
        "evidence_type": "offline_same_frame_detector_comparison",
        "scope": "Detector availability only; no pose, dynamics, or workflow-success claim.",
        "frames": len(rows),
        "all_frames": {
            "baseline_detections": sum(row["baseline_detected"] for row in rows),
            "fallback_detections": sum(row["fallback_detected"] for row in rows),
        },
        "late_approach": {
            "first_step": args.late_step,
            "frames": len(late),
            "baseline_detections": sum(row["baseline_detected"] for row in late),
            "fallback_detections": sum(row["fallback_detected"] for row in late),
        },
        "unchanged": [
            "rendered RGB inputs",
            "flush marker geometry",
            "camera calibration and resolution",
            "pose and workflow success limits",
        ],
        "inputs": rows,
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
