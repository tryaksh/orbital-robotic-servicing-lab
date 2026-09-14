"""Physical-response checks on a calibrated damped joint, plus backend guards."""

import numpy as np
import pytest

from zero_g_blade_swap.motion_profile import (
    JOINT_TRIM_BIAS_LIMIT_RAD,
    JOINT_TRIM_RATE_LIMIT_RAD_PER_S,
    update_joint_trim,
)


def _simulate(loads, *, trim=True, target=0.0):
    """Unit-inertia actuator: qdd = kp*(setpoint-q) - kd*qdot + load.

    This independent plant has the shipped wrist's 216 Nm/rad stiffness and
    critical damping. Physics runs at 600 Hz; encoder-based trim updates at
    30 Hz. External load is held constant during each control interval.
    """
    dt = 1 / 30
    physics_dt = dt / 20
    kp = 216.0
    kd = 2 * np.sqrt(kp)
    loads = np.asarray(loads, dtype=float)
    q, velocity, bias = np.zeros(6), np.zeros(6), np.zeros(6)
    positions, velocities, biases = [q.copy()], [velocity.copy()], [bias.copy()]
    for load in loads:
        if trim:
            bias = update_joint_trim(target - q, bias, dt, True)
        command = target + bias
        for _ in range(20):
            acceleration = kp * (command - q) - kd * velocity + load
            velocity += acceleration * physics_dt
            q += velocity * physics_dt
        positions.append(q.copy())
        velocities.append(velocity.copy())
        biases.append(bias.copy())
    return np.array(positions), np.array(velocities), np.array(biases)


def test_loaded_joint_recovers_alignment_without_changing_stiffness():
    loads = np.tile([3.3, -3.3, 3.3, -3.3, 3.3, -3.3], (8 * 30, 1))
    untrimmed, _, _ = _simulate(loads, trim=False)
    trimmed, velocity, biases = _simulate(loads)
    # Independent equilibrium: kp*q = external torque for an unbiased target.
    np.testing.assert_allclose(untrimmed[-1], loads[-1] / 216, atol=1e-10)
    assert abs(untrimmed[-1, 0]) == pytest.approx(0.0152777778)
    assert np.max(np.abs(trimmed[-30:])) < 1e-5
    assert np.max(np.abs(velocity[-30:])) < 1e-5
    np.testing.assert_allclose(biases[-1], -loads[-1] / 216, atol=1e-5)


def test_loaded_joint_unwinds_promptly_when_torque_reverses():
    loads = np.full((16 * 30, 6), 3.3)
    loads[8 * 30:] *= -1
    position, velocity, biases = _simulate(loads)
    assert np.max(biases[8 * 30]) < -0.015
    assert np.min(biases[9 * 30]) > 0
    assert np.max(np.abs(position[-30:])) < 1e-5
    assert np.max(np.abs(velocity[-30:])) < 1e-5
    np.testing.assert_allclose(biases[-1], 3.3 / 216, atol=1e-5)


def test_overloaded_joint_respects_rate_and_bias_caps_without_hidden_windup():
    loads = np.full((18 * 30, 6), 30.0)
    loads[9 * 30:] *= -1
    position, _, biases = _simulate(loads)
    assert np.max(np.abs(biases)) <= JOINT_TRIM_BIAS_LIMIT_RAD
    assert np.max(np.abs(np.diff(biases, axis=0))) <= JOINT_TRIM_RATE_LIMIT_RAD_PER_S / 30 + 1e-15
    assert biases[9 * 30, 0] == -JOINT_TRIM_BIAS_LIMIT_RAD
    assert biases[-1, 0] == JOINT_TRIM_BIAS_LIMIT_RAD
    # The finite actuator correction cannot hide an excessive load: even at
    # saturated bias, a residual position error must remain in the plant.
    assert position[9 * 30, 0] > 0.07
    assert position[-1, 0] < -0.07
    # Returning from -cap to zero takes about cap/rate = 2 seconds, rather than
    # waiting to drain an unbounded integrator accumulated over nine seconds.
    assert biases[12 * 30, 0] > 0


def test_per_environment_mask_preserves_inactive_biases_and_numpy_dtype():
    error = np.full((2, 3, 6), 0.2, dtype=np.float32)
    initial = np.full_like(error, 0.012)
    mask = np.array([[True, False, True], [False, True, False]])
    updated = update_joint_trim(error, initial, 1 / 30, mask)
    assert updated.dtype == np.float32
    np.testing.assert_array_equal(updated[~mask], initial[~mask])
    assert np.all(updated[mask] > initial[mask])
    np.testing.assert_array_equal(initial, np.full_like(initial, 0.012))
    np.testing.assert_array_equal(update_joint_trim(error, initial, 1 / 30, False), initial)


@pytest.mark.parametrize("dt", [0, -0.01, float("nan"), float("inf")])
def test_invalid_dt_is_rejected(dt):
    with pytest.raises(ValueError, match="positive and finite"):
        update_joint_trim(np.zeros(6), np.zeros(6), dt, True)


def test_torch_operator_path_preserves_dtype_and_non_cpu_device():
    torch = pytest.importorskip("torch")
    error = torch.full((2, 6), 0.2, dtype=torch.float64)
    bias = torch.zeros_like(error)
    active = torch.tensor([True, False])
    result = update_joint_trim(error, bias, 1 / 30, active)
    assert result.dtype == error.dtype and result.device == error.device
    torch.testing.assert_close(result[0], torch.full((6,), 0.001, dtype=torch.float64))
    torch.testing.assert_close(result[1], bias[1], rtol=0, atol=0)
    # Meta tensors exercise device-preserving operators without requiring a
    # GPU. A hidden .numpy() or .cpu() conversion would fail this call.
    meta = torch.empty((2, 6), dtype=torch.float32, device="meta")
    meta_mask = torch.empty(2, dtype=torch.bool, device="meta")
    result = update_joint_trim(meta, meta, 1 / 30, meta_mask)
    assert result.device.type == "meta" and result.dtype == torch.float32
