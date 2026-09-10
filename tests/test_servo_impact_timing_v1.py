import pytest

from assembly_recovery.servo_impact_timing_v1 import ServoImpactTimingV1


@pytest.mark.parametrize("refinement", [4, 8])
def test_fixed_servo_and_sensor_periods_share_the_registered_four_second_prefix(refinement):
    timing = ServoImpactTimingV1(refinement)
    steps = timing.handoff_step
    commands = [step for step in range(steps) if timing.servo_tick(step)]
    sensors = [step for step in range(1, steps + 1) if timing.sensor_tick(step)]
    assert len(commands) == 1920
    assert len(sensors) == 480
    assert commands[0] == 0
    assert commands[-1] * timing.physics_dt == pytest.approx(4 - 1 / 480)
    assert sensors[0] * timing.physics_dt == pytest.approx(1 / 120)
    assert sensors[-1] * timing.physics_dt == pytest.approx(4)
    assert timing.decimation * timing.physics_dt == pytest.approx(1 / 15)
    assert timing.deadline_steps * timing.physics_dt == pytest.approx(30)
    # Every original120Hz command instant remains a command instant; the three
    # added updates lie inside that same physical interval, independent of dt.
    assert all(timing.servo_tick(step) for step in range(0, steps, refinement))
    assert all((b - a) * timing.physics_dt == pytest.approx(1 / 480)
               for a, b in zip(commands[:-1], commands[1:], strict=True))


@pytest.mark.parametrize("refinement", [1, 2, 3, 16, True, 4.])
def test_servo_condition_rejects_unregistered_native_rates(refinement):
    with pytest.raises(ValueError):
        ServoImpactTimingV1(refinement)

