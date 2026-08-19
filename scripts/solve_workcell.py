"""Find a workcell in which the arm can hold the capture attitude everywhere the relocation needs it.

``evidence/relocation_reach_boundary.json`` measured the failure and named the
suspect. Driving position *and* the head-on capture attitude at full authority,
the tool parks at local x = -0.0258 -- 0.4242 m forward of the base at -0.45 --
and the relocation needs 88.7 mm past that to finish an extraction and 167 mm
past it to retreat behind the lead-in flares. Two controls said the boundary is
made of attitude rather than reach: with the orientation command removed every
depth converges to 0.00 mm, and lifting over the flares does not move the
parking point at any height from 0 to 200 mm.

Near the folded configuration the retreat demands, the Jacobian is
ill-conditioned and attitude trades against reach at about 7.5 m/rad. Moving the
base back unfolds the arm at those depths. That is a hypothesis with a number in
it -- if the parking point is genuinely base-relative at 0.4242 m, a base at or
behind x = -0.617 reaches the retreat -- and this script is the measurement.

**What it sweeps.** One app launch per candidate base, because the base is an
articulation spawn pose and every environment in a scene shares it. Inside each
launch, every pose the relocation needs solves at once, one environment per
(depth, bay, wrist seed).

**What it gates on.** Not "the arm gets there" -- the attitude-free control
already proved that. Every required pose has to converge to within the tolerance
on *both* position and attitude, with orientation driven at full authority, on
at least one wrist seed. A base that reaches the retreat but loses the capture
pose has not solved anything: the relocation is one continuous episode.

Nothing here trains and nothing here is a task change. It writes
``evidence/workcell_reach_solution.json`` and prints the base to adopt.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

#: The relocation's poses, as offsets from the module's INSTALLED pose.
#:
#: The calibrator servos the tool onto the blade's grip point at the curriculum
#: stage it is given, and every swept offset moves that target while the module
#: stays pinned where the stage put it. Which stage the sweep runs from is
#: therefore not a free choice, and the first version of this script got it
#: wrong: run from stage 2, with the module parked at the rack mouth, the
#: "installed" target sits 136 mm FORWARD of the pinned module, so solving it
#: drives the gripper into the module's own body. That cell reported 91.6 mm and
#: 1.02 rad -- a miss produced by the probe, not by the arm.
#:
#: Stage 0 is the one stage from which every pose is behind the pinned module's
#: pin, so no target is inside anything:
#:
#:   installed  blade centre 0.7194748  (INSERTION_STAGE_BLADE_POSE[0])   tool  0.3800
#:   staging    blade centre 0.5829     (INSERTION_STAGING_BLADE_POS)     tool  0.2434
#:   extracted  blade centre 0.2250     (EXTRACTED_BLADE_CENTRE_X)        tool -0.1145
#:   retreated  blade centre 0.1468     (TRANSIT_CLEAR_BLADE_CENTRE_X)    tool -0.1928
#:
#: computed by ``_required_poses`` from ``zero_g_blade_swap.grapple_geometry`` and
#: ``assets.py`` so an edit to either moves this sweep with it.
POSE_NAMES = ("installed", "staging", "extracted", "retreated")


def _read_literal(path: Path, name: str, *index: int):
    """Return a module-level literal assignment without importing the module.

    ``assets.py`` imports Isaac Lab, which needs the Kit runtime, and this driver
    runs before any app is launched. Parsing the assignment is not a shortcut
    around rule 2 -- it is the rule: the number is read from the one file that
    owns it, so an edit there moves this sweep with it, and a name that stops
    being a literal raises instead of silently going stale.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
        for target in targets:
            if isinstance(target, ast.Name) and target.id == name and node.value is not None:
                value = node.value
                # Descend the literal tuple by index before evaluating, so a
                # sibling entry that is not a literal -- these tuples splice
                # other constants in with ``*`` -- does not make the one that is
                # unreadable.
                for position in index:
                    if not isinstance(value, (ast.Tuple, ast.List)):
                        raise TypeError(f"{name} is not indexable at {position}")
                    value = value.elts[position]
                return ast.literal_eval(value)
    raise KeyError(f"{name} is not a literal assignment in {path}")


def _required_poses() -> tuple[dict[str, float], float, float]:
    """Return the swept x offsets by name, the stage-0 tool x, and the bay pitch."""

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from zero_g_blade_swap.grapple_geometry import (
        EXTRACTED_BLADE_CENTRE_X,
        GRAPPLE_PIN_GRIP_OFFSET,
        TRANSIT_CLEAR_BLADE_CENTRE_X,
    )

    assets = root / "src" / "zero_g_blade_swap" / "tasks" / "blade_swap" / "assets.py"
    # CONTACT_INSERTION_STAGE_BLADE_POSE[0] takes its x straight from
    # INSERTION_STAGE_BLADE_POSE[0], which is the literal one.
    installed_centre = _read_literal(assets, "INSERTION_STAGE_BLADE_POSE", 0, 0)
    staging_centre = _read_literal(assets, "INSERTION_STAGING_BLADE_POS", 0)
    second_bay_y = _read_literal(assets, "SECOND_SLOT_CENTER_Y")

    grip_behind_centre = -GRAPPLE_PIN_GRIP_OFFSET[0]
    centres = {
        "installed": installed_centre,
        "staging": staging_centre,
        "extracted": EXTRACTED_BLADE_CENTRE_X,
        "retreated": TRANSIT_CLEAR_BLADE_CENTRE_X,
    }
    offsets = {name: round(centre - installed_centre, 6) for name, centre in centres.items()}
    installed_tool_x = round(installed_centre - grip_behind_centre, 6)
    return offsets, installed_tool_x, second_bay_y


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base_x",
        type=float,
        nargs="+",
        default=[-0.45, -0.55, -0.65, -0.75],
        help="Candidate base x positions, one app launch each. -0.45 is the certified cell and the control.",
    )
    parser.add_argument(
        "--base_y",
        type=float,
        nargs="+",
        default=[0.0, -0.11],
        help=(
            "Candidate base y positions, crossed with --base_x. The two bays are 220 mm apart in y, and the "
            "control run says this axis matters more than x: with the base at -0.45, 0, every one of the four "
            "poses converges in the SECOND bay at y = -0.22 and three of four miss on the first bay's centre "
            "line. -0.11 is the midpoint, which puts both bays the same distance off the base's own plane."
        ),
    )
    parser.add_argument(
        "--base_z",
        type=float,
        nargs="+",
        default=[0.15],
        help="Candidate base heights, crossed with the others. 0.15 is the certified cell.",
    )
    parser.add_argument(
        "--base_xyz",
        type=float,
        nargs="+",
        default=None,
        metavar="V",
        help=(
            "Explicit base positions as flattened x y z triples, instead of the cross product of --base_x, "
            "--base_y and --base_z. The cross product wastes launches once the sweep has a direction: "
            "y = -0.11 puts BOTH bays about 110 mm off the base's plane and neither converges, while a bay "
            "220 mm off it converges at every depth, so the useful lateral candidates are on one side of the "
            "rack rather than between the bays."
        ),
    )
    parser.add_argument(
        "--alt_start",
        type=float,
        nargs=6,
        default=[0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0],
        help=(
            "A second start pose for every solve: shoulder lifted to vertical with the elbow straight, which "
            "parks the tool about 1.3 m above the base and is clear of the rack at every candidate position."
        ),
    )
    parser.add_argument("--steps", type=int, default=3000, help="Servo iterations. Rule 7: converge the solver.")
    parser.add_argument("--tolerance_m", type=float, default=0.002)
    parser.add_argument("--tolerance_rad", type=float, default=0.010)
    parser.add_argument("--task", default="Isaac-ZeroG-Blade-GrapplePin-Capture-v0")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--python", default="C:/isaac-sim/python.bat")
    parser.add_argument("--out", type=Path, default=Path("artifacts/workcell"))
    parser.add_argument("--report", type=Path, default=Path("evidence/workcell_reach_solution.json"))
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse a per-base report already on disk. For re-aggregating without re-solving.",
    )
    return parser


def _run_one(args, base: tuple[float, float, float], offsets: dict[str, float], bays: list[float]) -> dict:
    tag = f"x{base[0]:+.3f}_y{base[1]:+.3f}_z{base[2]:+.3f}".replace(".", "p")
    report = args.out / f"base_{tag}.json"
    log = args.out / f"base_{tag}.log"
    if args.skip_existing and report.is_file():
        print(f"[WORKCELL] reusing {report}")
        return json.loads(report.read_text(encoding="utf-8"))

    command = [
        args.python,
        "scripts/calibrate_grasp_pose.py",
        "--headless",
        "--task",
        args.task,
        "--steps",
        str(args.steps),
        "--pin_blade",
        "--finger_joint",
        "0.02",
        "--stages",
        str(args.stage),
        "--tolerance_m",
        str(args.tolerance_m),
        "--tolerance_rad",
        str(args.tolerance_rad),
        "--sweep_offset_x",
        *[f"{offsets[name]:.6f}" for name in POSE_NAMES],
        "--sweep_offset_y",
        *[f"{bay:.6f}" for bay in bays],
        # The task's own reset pose is solved for ONE base, so a moved base
        # carries the arm bodily with it and can spawn the gripper inside the
        # rack. Every target is therefore also solved from an upright pose that
        # is clear of the scene at any base, and the better start wins.
        "--alt_start_joint_pos",
        *[f"{value:.6f}" for value in args.alt_start],
        "--robot_base_x",
        f"{base[0]:.6f}",
        "--robot_base_y",
        f"{base[1]:.6f}",
        "--robot_base_z",
        f"{base[2]:.6f}",
        "--report",
        str(report),
    ]
    started = time.time()
    print(f"[WORKCELL] base {base} -> {report}")
    with log.open("w", encoding="utf-8") as handle:
        status = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT)
    print(f"[WORKCELL]   exit={status} in {time.time() - started:.0f}s")
    if not report.is_file():
        return {"status": "failed", "error": f"no report written, exit={status}", "stages": []}
    return json.loads(report.read_text(encoding="utf-8"))


def _summarise(raw: dict, offsets: dict[str, float], bays: list[float], args) -> dict:
    """Reduce one base's rows to the best wrist seed per (pose, bay)."""

    by_name = {round(value, 6): name for name, value in offsets.items()}
    best: dict[tuple[str, float], dict] = {}
    for row in raw.get("stages", []):
        name = by_name.get(round(row["sweep_offset_x_m"], 6))
        if name is None:
            continue
        key = (name, round(row["sweep_offset_y_m"], 6))
        # "Best" is the seed that satisfies both tolerances if any does, and
        # otherwise the smallest position residual. A seed that converges the
        # position by giving the attitude away is exactly the failure this whole
        # sweep exists to detect, so converged sorts ahead of close.
        rank = (not row["converged"], row["residual_tool_to_handle_m"])
        if key not in best or rank < best[key]["_rank"]:
            best[key] = {**row, "_rank": rank}
    poses = []
    for name in POSE_NAMES:
        for bay in bays:
            row = best.get((name, round(bay, 6)))
            if row is None:
                poses.append({"pose": name, "bay_y_m": bay, "solved": False, "note": "no row"})
                continue
            poses.append(
                {
                    "pose": name,
                    "bay_y_m": round(bay, 6),
                    "target_tool_x_local_m": row["target_tool_x_local_m"],
                    "reached_tool_x_local_m": row["reached_tool_x_local_m"],
                    "residual_m": round(row["residual_tool_to_handle_m"], 6),
                    "residual_rad": round(row["residual_orientation_rad"], 6),
                    "solved": bool(row["converged"]),
                    "arm_joint_pos_rad": row["arm_joint_pos_rad"],
                }
            )
    return {
        "robot_root_local_m": raw.get("robot_root_local_m"),
        "all_required_poses_solved": all(pose["solved"] for pose in poses),
        "worst_residual_m": round(max((p.get("residual_m", 9.9) for p in poses), default=9.9), 6),
        "worst_residual_rad": round(max((p.get("residual_rad", 9.9) for p in poses), default=9.9), 6),
        "poses": poses,
    }


def main() -> int:
    args = _parser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    offsets, installed_tool_x, second_bay_y = _required_poses()
    bays = [0.0, second_bay_y]
    print(f"[WORKCELL] installed tool x = {installed_tool_x:+.4f}, offsets {offsets}, bays {bays}")

    candidates = []
    if args.base_xyz:
        if len(args.base_xyz) % 3:
            raise SystemExit("--base_xyz takes flattened x y z triples")
        values = list(args.base_xyz)
        candidates = [tuple(values[index : index + 3]) for index in range(0, len(values), 3)]
    else:
        for base_x in args.base_x:
            for base_y in args.base_y:
                for base_z in args.base_z:
                    candidates.append((base_x, base_y, base_z))

    results = []
    for base in candidates:
        raw = _run_one(args, base, offsets, bays)
        summary = _summarise(raw, offsets, bays, args)
        summary["requested_base_m"] = list(base)
        results.append(summary)
        placed = summary["robot_root_local_m"]
        state = "ALL SOLVED" if summary["all_required_poses_solved"] else "incomplete"
        print(
            f"[WORKCELL]   base requested {base} placed {placed}: {state}, "
            f"worst {summary['worst_residual_m'] * 1000:.2f} mm / {summary['worst_residual_rad']:.4f} rad"
        )
        for pose in summary["poses"]:
            mark = "ok " if pose["solved"] else "MISS"
            print(
                f"[WORKCELL]     {mark} {pose['pose']:<10} bay y={pose['bay_y_m']:+.3f} "
                f"target x={pose.get('target_tool_x_local_m', float('nan')):+.4f} "
                f"reached {pose.get('reached_tool_x_local_m', float('nan')):+.4f} "
                f"{pose.get('residual_m', 9.9) * 1000:8.2f} mm {pose.get('residual_rad', 9.9):.4f} rad"
            )

    solved = [row for row in results if row["all_required_poses_solved"]]
    # Prefer the base closest to the certified -0.45, so the smallest workcell
    # change that works is the one adopted. A bigger move is not more correct,
    # and every extra millimetre is reach spent at the capture end.
    def _move_from_certified(row: dict) -> float:
        base = row["requested_base_m"]
        return abs(base[0] + 0.45) + abs(base[1]) + abs(base[2] - 0.15)

    chosen = min(solved, key=_move_from_certified) if solved else None

    report = {
        "title": "A workcell in which the arm holds the capture attitude at every pose the relocation needs",
        "evidence_type": "simulation_kinematic_calibration",
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "question": (
            "evidence/relocation_reach_boundary.json measured that the arm cannot hold the head-on capture "
            "attitude at the extraction end (88.7 mm short) or the transit retreat (167 mm short) with the "
            "base at x = -0.45, and that neither leg order nor lifting over the flares moves that boundary. "
            "It is a folded-configuration problem, so the base is the variable. Which base position, if any, "
            "puts all four poses -- installed, staging, extracted, retreated -- inside the attitude-holding "
            "region, in BOTH bays?"
        ),
        "method": (
            "scripts/solve_workcell.py, one scripts/calibrate_grasp_pose.py launch per candidate base because "
            "the base is an articulation spawn pose shared by every environment in a scene. Inside each launch "
            "every (depth, bay, wrist-seed) combination solves in its own environment: the Capture task, blade "
            f"pinned at its stage-{args.stage} pose, fingers held at 0.02 rad, {args.steps} servo steps, four wrist_1 "
            "seed quadrants, orientation driven at FULL authority. A pose counts as solved only when one seed "
            f"satisfies both {args.tolerance_m * 1000:.0f} mm and {args.tolerance_rad:.3f} rad; reaching the depth by "
            "giving the attitude away is the failure being measured, not a pass."
        ),
        "tolerance_m": args.tolerance_m,
        "tolerance_rad": args.tolerance_rad,
        "servo_steps": args.steps,
        "task": args.task,
        "curriculum_stage": args.stage,
        "alt_start_joint_pos_rad": list(args.alt_start),
        "required_pose_offsets_m": offsets,
        "bays_y_m": bays,
        "installed_tool_x_local_m": installed_tool_x,
        "candidates": results,
        "solution": chosen,
        "passed": chosen is not None,
        "scope_and_limitations": [
            "Kinematics only, with the module pinned. It says where the tool can be put, not what a policy does when it gets there.",
            "Every pose is swept from the INSTALLED module pose. An earlier version of this script swept from the "
            "rack-mouth pose, which puts the installed target 136 mm in front of the pinned module and solves it by "
            "driving the gripper into the module; that cell reported 91.6 mm and 1.02 rad and was a probe artefact.",
            "This is a differential-IK servo, not an optimiser. Where position and orientation trade steeply, what it "
            "converges to depends on the authority each is driven with; orientation is at full authority throughout, "
            "which is the strict side of that trade.",
            "The lateral sweep moves the TARGET to the second bay while the pinned module stays on the centre line. It "
            "measures reach into that bay, not contact with anything standing in it.",
            "One curriculum stage, four wrist_1 seed quadrants, and two start poses. A different arm or a seventh "
            "axis would have a different boundary.",
            "The mount anchor moves with the base. The compliant mount is a D6 joint between the robot root and a "
            "mount_anchor body and robot_mount_unstable terminates above 16.5 mm of offset, so a base moved without "
            "its anchor is reset to its spawn pose every step and the whole sweep reports that pose as the "
            "reachable set. That is the third layer of the --robot_base_x defect and the only one that produces a "
            "plausible wrong answer rather than no answer.",
            "Simulation only. No result here was produced on real hardware.",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[WORKCELL] wrote {args.report}")
    if chosen is None:
        print("[WORKCELL] NO BASE SOLVES EVERY POSE -- a rail or a rack move is the next candidate")
        return 1
    print(f"[WORKCELL] ADOPT base {chosen['robot_root_local_m']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
