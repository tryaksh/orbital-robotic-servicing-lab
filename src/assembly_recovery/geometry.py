"""Small geometric checks; inputs come from the simulator's resolved task config."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PegGeometry:
    peg_diameter_m: float
    hole_diameter_m: float
    peg_height_m: float
    hole_height_m: float
    fingerpad_length_m: float
    position_action_bound_m: float
    success_height_fraction: float

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Geometry must be finite and positive")
        if self.peg_diameter_m >= self.hole_diameter_m:
            raise ValueError("Peg cannot fit through the configured hole")
        if self.fingerpad_length_m >= self.peg_height_m:
            raise ValueError("This derivation requires a peg longer than the finger pad")
        if self.maximum_withdrawn_base_clearance_m <= 0:
            raise ValueError("The position action bound cannot withdraw the peg above the hole")

    @property
    def radial_clearance_m(self):
        return (self.hole_diameter_m - self.peg_diameter_m) / 2

    @property
    def seated_fingertip_above_hole_top_m(self):
        return self.peg_height_m - self.fingerpad_length_m - self.hole_height_m

    @property
    def maximum_withdrawn_base_clearance_m(self):
        return self.position_action_bound_m - (self.peg_height_m - self.fingerpad_length_m)

    @property
    def success_height_tolerance_m(self):
        return self.hole_height_m * self.success_height_fraction

    @property
    def gross_separation_distance_m(self):
        # Conservative bounding spheres for the peg and finger-pad region.
        return math.hypot(self.peg_height_m / 2, self.peg_diameter_m / 2) + self.fingerpad_length_m

    def report(self):
        return {**asdict(self), "radial_clearance_m": self.radial_clearance_m,
                "seated_fingertip_above_hole_top_m": self.seated_fingertip_above_hole_top_m,
                "maximum_withdrawn_base_clearance_m": self.maximum_withdrawn_base_clearance_m,
                "success_height_tolerance_m": self.success_height_tolerance_m,
                "gross_separation_distance_m": self.gross_separation_distance_m,
                "scope": "Configured dimensions, not a USD mesh or contact-patch certification. Separation detects gross loss, not every loss of contact."}
