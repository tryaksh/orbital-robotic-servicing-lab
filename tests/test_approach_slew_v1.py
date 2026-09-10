import pytest
import torch

from assembly_recovery.approach_slew_v1 import clamp_impedance_target, slew_reference


def test_euclidean_slew_does_not_allow_sqrt_three_diagonal_speed():
    previous = torch.zeros((2, 3), dtype=torch.float64)
    desired = torch.tensor([[1., 1., 1.], [0., 0., 0.]], dtype=torch.float64)
    actual = slew_reference(previous, desired)
    assert float(actual[0].norm()) == pytest.approx(.020 / 480)
    assert torch.equal(actual[1], previous[1])
    assert torch.equal(actual[0], actual[0, 0].expand(3))


def test_slew_reaches_nearby_target_without_overshoot_and_reverses():
    previous = torch.tensor([[.6, .1, .05]], dtype=torch.float64)
    desired = previous + torch.tensor([[1e-6, -2e-6, 3e-6]], dtype=torch.float64)
    assert torch.equal(slew_reference(previous, desired), desired)
    first = slew_reference(previous, previous + .02)
    second = slew_reference(first, previous - .02)
    assert float((second - first).norm()) == pytest.approx(.020 / 480)


def test_same_elapsed_servo_time_has_same_reference_at_both_native_rates():
    target = torch.tensor([[.01, -.02, .03]], dtype=torch.float64)
    results = []
    for native_stride in (1, 2):
        reference = torch.zeros_like(target)
        for native_step in range(960 * native_stride):
            if native_step % native_stride == 0:
                reference = slew_reference(reference, target)
        results.append(reference)
    assert torch.equal(*results)
    assert torch.equal(results[0], target)


def test_original_error_clamp_remains_after_reference_interpolation():
    previous = torch.tensor([[.1, -.1, .1]], dtype=torch.float64)
    desired = previous + 1.
    interpolated = slew_reference(previous, desired)
    tool = torch.zeros_like(previous)
    thresholds = torch.tensor([[.02, .01, .025]], dtype=torch.float64)
    actual = clamp_impedance_target(interpolated, tool, thresholds)
    assert torch.equal(actual, torch.tensor([[.02, -.01, .025]], dtype=torch.float64))
    assert bool(((actual - tool).abs() <= thresholds).all())


@pytest.mark.parametrize("speed,dt", [(0., 1 / 480), (-1., 1 / 480), (.02, 0.), (float("nan"), 1 / 480)])
def test_invalid_reference_contract_rejected(speed, dt):
    with pytest.raises(ValueError):
        slew_reference(torch.zeros((1, 3)), torch.ones((1, 3)), speed_m_s=speed, servo_dt=dt)

