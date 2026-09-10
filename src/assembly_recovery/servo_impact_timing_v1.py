"""CPU-only clock contract for a fixed480Hz external servo sensitivity check."""
from assembly_recovery.physics_validation_v1 import PhysicsTimingV1


class ServoImpactTimingV1(PhysicsTimingV1):
    external_servo_hz = 480
    external_sensor_hz = 120
    policy_hz = 15

    def __post_init__(self):
        if type(self.refinement) is not int or self.refinement not in (4, 8):
            raise ValueError("Fixed480Hz external servo requires native480 or960Hz")

    def servo_tick(self, completed_native_steps):
        return completed_native_steps % (self.refinement // 4) == 0

