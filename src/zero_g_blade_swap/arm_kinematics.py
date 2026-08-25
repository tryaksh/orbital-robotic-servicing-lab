"""UR10e forward kinematics, Jacobian, and damped-least-squares IK.

One set of Denavit-Hartenberg parameters, used twice.

``scripts/check_workcell_geometry.py`` uses the NumPy half to answer where the
arm can stand before anything launches a simulator, and validates it against the
simulator's own recorded configurations
(``evidence/workcell_reach_solution.json``) before it reports a number.

``scripts/run_workflow_demo.py`` uses the Torch half to command the scripted
transit legs. That is the same solver, batched over environments and seeded from
the joint positions the arm is actually at, and it exists because the
alternative does not converge: IsaacLab's differential IK runs in *relative*
mode, re-anchoring on the tool's current pose every control step and driving to
current-plus-delta across the decimation, so while the joints lag the deltas
accumulate ahead of the arm. A squaring leg on that controller limit-cycles at
about one action scale -- 4.5, 11.4, 13.9, 15.1 mrad on successive samples --
against a channel that admits 2.22. Solving the pose and commanding joint
targets removes the accumulation: the setpoint converges to the leg's target and
stops there, and the arm converges to the setpoint.

Both halves command actuator targets. Nothing here writes a joint or body state.

The Torch half is validated against the NumPy half by
``tests/test_arm_kinematics.py``, and against the running simulator by the
driver, which refuses to command a solved leg while its forward kinematics
disagree with the simulator's own tool frame. That check is what makes a
joint-order mistake loud instead of subtle.
"""

from __future__ import annotations

import numpy as np

from .grapple_geometry import PAD_SPAN_FROM_FLANGE_M

# ---------------------------------------------------------------------------
# Standard Denavit-Hartenberg parameters for the UR10e. The USD's ``base_link``
# frame is the DH base frame turned 180 degrees about z, which is what
# ``BASE_FROM_DH`` is.
DH_A = (0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0)
DH_D = (0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655)
DH_ALPHA = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)
BASE_FROM_DH = np.diag([-1.0, -1.0, 1.0])

#: The joint order these parameters are written in. ``ARM_JOINTS`` in
#: ``zero_g_blade_swap.tasks.blade_swap.env_cfg`` is the same order; anything
#: reading joints out of the simulator has to resolve names with
#: ``preserve_order=True`` or permute into this order, and the driver's own
#: check catches it when it does not.
JOINT_ORDER = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

#: The grapple tool point: the middle of the measured pad span, on the flange axis.
TOOL_OFFSET_Z = 0.5 * (PAD_SPAN_FROM_FLANGE_M[0] + PAD_SPAN_FROM_FLANGE_M[1])

#: ``DifferentialIKControllerCfg(ik_method="dls")``'s default damping.
DLS_LAMBDA = 0.01

#: UR10e joint travel. Every joint on this arm is plus or minus 2 pi.
JOINT_LIMIT_RAD = 2.0 * np.pi


# ---------------------------------------------------------------------------
# NumPy: one configuration at a time, for the geometry check.


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _chain(joints: np.ndarray) -> list[np.ndarray]:
    transform = np.eye(4)
    frames = [transform.copy()]
    for index in range(6):
        transform = transform @ _dh(joints[index], DH_D[index], DH_A[index], DH_ALPHA[index])
        frames.append(transform.copy())
    return frames


def tool_pose(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the tool point and orientation in the robot base_link frame."""

    flange = _chain(joints)[-1]
    position = flange[:3, 3] + flange[:3, 2] * TOOL_OFFSET_Z
    return BASE_FROM_DH @ position, BASE_FROM_DH @ flange[:3, :3]


def tool_jacobian(joints: np.ndarray) -> np.ndarray:
    """Return the 6x6 geometric Jacobian of the tool point in base_link."""

    frames = _chain(joints)
    flange = frames[-1]
    tool = flange[:3, 3] + flange[:3, 2] * TOOL_OFFSET_Z
    jacobian = np.zeros((6, 6))
    for index in range(6):
        axis = frames[index][:3, 2]
        origin = frames[index][:3, 3]
        jacobian[:3, index] = np.cross(axis, tool - origin)
        jacobian[3:, index] = axis
    jacobian[:3, :] = BASE_FROM_DH @ jacobian[:3, :]
    jacobian[3:, :] = BASE_FROM_DH @ jacobian[3:, :]
    return jacobian


def rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3)
    return (angle / (2.0 * np.sin(angle))) * np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )


def quaternion_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _dls_step(jacobian: np.ndarray, twist: np.ndarray) -> np.ndarray:
    gram = jacobian @ jacobian.T
    return jacobian.T @ np.linalg.solve(gram + DLS_LAMBDA**2 * np.eye(6), twist)


def solve_ik(
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    seeds: list[np.ndarray],
    iterations: int = 600,
) -> tuple[np.ndarray, float, float]:
    """Return the best of ``seeds`` and its position and attitude residuals."""

    best: tuple[np.ndarray, float, float] | None = None
    for seed in seeds:
        joints = np.array(seed, dtype=float)
        for _ in range(iterations):
            position, rotation = tool_pose(joints)
            error = np.concatenate(
                [target_position - position, rotation_vector(target_rotation @ rotation.T)]
            )
            if np.linalg.norm(error[:3]) < 1e-7 and np.linalg.norm(error[3:]) < 1e-7:
                break
            joints = np.clip(
                joints + 0.5 * _dls_step(tool_jacobian(joints), error),
                -JOINT_LIMIT_RAD,
                JOINT_LIMIT_RAD,
            )
        position, rotation = tool_pose(joints)
        error = np.concatenate(
            [target_position - position, rotation_vector(target_rotation @ rotation.T)]
        )
        candidate = (joints, float(np.linalg.norm(error[:3])), float(np.linalg.norm(error[3:])))
        if best is None or (candidate[1] + candidate[2]) < (best[1] + best[2]):
            best = candidate
    assert best is not None
    return best


def realised_authority(joints: np.ndarray) -> dict[str, float]:
    """Return what fraction of a commanded twist the DLS controller delivers."""

    jacobian = tool_jacobian(joints)
    gram = jacobian @ jacobian.T
    gain = gram @ np.linalg.inv(gram + DLS_LAMBDA**2 * np.eye(6))
    rotational = 0.5 * (gain[3:, 3:] + gain[3:, 3:].T)
    return {
        "authority_pitch": float(gain[4, 4]),
        "authority_yaw": float(gain[5, 5]),
        "authority_worst_rotation_axis": float(np.linalg.eigvalsh(rotational).min()),
        "authority_worst_any_axis": float(np.linalg.eigvalsh(0.5 * (gain + gain.T)).min()),
        "jacobian_min_singular_value": float(np.linalg.svd(jacobian, compute_uv=False).min()),
        "joint_travel_used_fraction": float(np.abs(joints).max() / JOINT_LIMIT_RAD),
    }


# ---------------------------------------------------------------------------
# Torch: the whole batch at once, for the control loop.
#
# Imported lazily so the geometry check, which is the thing that has to run in
# CI without a GPU or a simulator, keeps a NumPy-only dependency.


def _torch():
    import torch

    return torch


def batched_chain(joints):
    """Return the seven cumulative DH frames for ``joints`` of shape ``(B, 6)``."""

    torch = _torch()
    batch = joints.shape[0]
    transform = torch.eye(4, dtype=joints.dtype, device=joints.device).expand(batch, 4, 4)
    frames = [transform]
    zero = torch.zeros(batch, dtype=joints.dtype, device=joints.device)
    one = torch.ones(batch, dtype=joints.dtype, device=joints.device)
    for index in range(6):
        theta = joints[:, index]
        ct, st = torch.cos(theta), torch.sin(theta)
        ca = float(np.cos(DH_ALPHA[index]))
        sa = float(np.sin(DH_ALPHA[index]))
        a = float(DH_A[index])
        d = float(DH_D[index])
        step = torch.stack(
            (
                torch.stack((ct, -st * ca, st * sa, a * ct), dim=-1),
                torch.stack((st, ct * ca, -ct * sa, a * st), dim=-1),
                torch.stack((zero, zero + sa, zero + ca, zero + d), dim=-1),
                torch.stack((zero, zero, zero, one), dim=-1),
            ),
            dim=-2,
        )
        transform = transform @ step
        frames.append(transform)
    return frames


def batched_tool_pose(joints):
    """Return tool position ``(B, 3)`` and rotation ``(B, 3, 3)`` in base_link."""

    torch = _torch()
    frames = batched_chain(joints)
    flange = frames[-1]
    position = flange[:, :3, 3] + flange[:, :3, 2] * TOOL_OFFSET_Z
    basis = torch.as_tensor(BASE_FROM_DH, dtype=joints.dtype, device=joints.device)
    return position @ basis.transpose(-1, -2), basis @ flange[:, :3, :3]


def batched_tool_jacobian(joints):
    """Return the geometric Jacobian ``(B, 6, 6)`` of the tool point in base_link."""

    torch = _torch()
    frames = batched_chain(joints)
    flange = frames[-1]
    tool = flange[:, :3, 3] + flange[:, :3, 2] * TOOL_OFFSET_Z
    axes = torch.stack([frame[:, :3, 2] for frame in frames[:6]], dim=-1)
    origins = torch.stack([frame[:, :3, 3] for frame in frames[:6]], dim=-1)
    linear = torch.cross(axes, tool.unsqueeze(-1) - origins, dim=1)
    basis = torch.as_tensor(BASE_FROM_DH, dtype=joints.dtype, device=joints.device)
    return torch.cat((basis @ linear, basis @ axes), dim=1)


def batched_rotation_vector(rotation):
    """Return the axis-angle vector of ``rotation`` of shape ``(B, 3, 3)``.

    First order near zero, exact elsewhere, and not usable within a few
    milliradians of a half turn -- which this controller never sees, because
    every solve is seeded from the configuration the arm is already in and the
    setpoint it chases moves by at most one action scale per control step.
    """

    torch = _torch()
    trace = rotation[:, 0, 0] + rotation[:, 1, 1] + rotation[:, 2, 2]
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.arccos(cosine)
    sine = torch.sin(angle)
    scale = torch.where(sine.abs() < 1.0e-6, torch.full_like(angle, 0.5), angle / (2.0 * sine))
    vector = torch.stack(
        (
            rotation[:, 2, 1] - rotation[:, 1, 2],
            rotation[:, 0, 2] - rotation[:, 2, 0],
            rotation[:, 1, 0] - rotation[:, 0, 1],
        ),
        dim=-1,
    )
    return scale.unsqueeze(-1) * vector


def batched_solve_ik(
    target_position,
    target_rotation,
    seed,
    iterations: int = 60,
    gain: float = 0.5,
    position_tolerance: float = 1.0e-6,
    attitude_tolerance: float = 1.0e-6,
):
    """Solve for joint angles reaching a tool pose, seeded from ``seed``.

    ``target_position`` is ``(B, 3)`` and ``target_rotation`` ``(B, 3, 3)``, both
    in the robot's base_link frame; ``seed`` is ``(B, 6)`` in ``JOINT_ORDER``.

    Returns the joints and the per-environment residuals, so a caller can refuse
    to command a solve that did not converge rather than commanding a pose that
    is not the one it asked for.
    """

    torch = _torch()
    joints = seed.clone()
    identity = torch.eye(6, dtype=joints.dtype, device=joints.device) * (DLS_LAMBDA**2)
    position_residual = torch.zeros(joints.shape[0], dtype=joints.dtype, device=joints.device)
    attitude_residual = torch.zeros_like(position_residual)
    for _ in range(iterations):
        position, rotation = batched_tool_pose(joints)
        linear = target_position - position
        angular = batched_rotation_vector(target_rotation @ rotation.transpose(-1, -2))
        position_residual = torch.linalg.vector_norm(linear, dim=-1)
        attitude_residual = torch.linalg.vector_norm(angular, dim=-1)
        if bool(position_residual.max() < position_tolerance) and bool(
            attitude_residual.max() < attitude_tolerance
        ):
            break
        jacobian = batched_tool_jacobian(joints)
        twist = torch.cat((linear, angular), dim=-1).unsqueeze(-1)
        gram = jacobian @ jacobian.transpose(-1, -2) + identity
        delta = jacobian.transpose(-1, -2) @ torch.linalg.solve(gram, twist)
        joints = (joints + gain * delta.squeeze(-1)).clamp(-JOINT_LIMIT_RAD, JOINT_LIMIT_RAD)
    return joints, position_residual, attitude_residual


__all__ = [
    "BASE_FROM_DH",
    "DH_A",
    "DH_ALPHA",
    "DH_D",
    "DLS_LAMBDA",
    "JOINT_LIMIT_RAD",
    "JOINT_ORDER",
    "TOOL_OFFSET_Z",
    "batched_chain",
    "batched_rotation_vector",
    "batched_solve_ik",
    "batched_tool_jacobian",
    "batched_tool_pose",
    "quaternion_to_matrix",
    "realised_authority",
    "rotation_vector",
    "solve_ik",
    "tool_jacobian",
    "tool_pose",
]
