"""Explicit PhysX integration matching PhysicsContext._step without USD edits."""


def explicit_physics_step_v2(interface, *, dt, current_time, update_fabric=False):
    if dt not in (1 / 120, 1 / 240) or update_fabric:
        raise ValueError("Only registered headless native physics steps are supported")
    interface.simulate(dt, current_time)
    interface.fetch_results()
