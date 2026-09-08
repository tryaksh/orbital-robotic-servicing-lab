"""Bounded scripted development retry using only the pinned FORGE actor vector."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass
from statistics import median


@dataclass(frozen=True)
class RetrySettings:
    first_attempt_s: float = 4.0
    hold_s: float = 1.0
    stall_window_s: float = 1.0
    stall_progress_m: float = 0.001
    stall_load_n: float = 5.0
    stall_load_occupancy: float = 0.8
    withdrawal_s: float = 3.0
    withdrawal_height_m: float = 0.045
    withdrawal_rise_m: float = 0.005
    spiral_radial_speed_m_s: float = 0.00018
    spiral_angular_speed_rad_s: float = 1.5
    spiral_max_radius_m: float = 0.004
    lateral_integral_gain: float = 0.8
    lateral_integral_limit_m: float = 0.012
    insertion_load_n: float = 8.0
    force_integral_gain_m_n_s: float = 0.0004
    initial_pressure_offset_m: float = 0.012
    capture_depth_m: float = 0.004


class ActorRetryController:
    """No simulator/evaluator reference is accepted or stored.

    Layout: noisy relative XYZ, quaternion, linear/angular velocity, noisy
    wrist force, force threshold, previous applied actions (24 scalars).
    The same controller runs in continued-insertion mode as its paired control.
    """

    observation_order = ["fingertip_pos_rel_fixed", "fingertip_quat", "ee_linvel",
                         "ee_angvel", "ft_force", "force_threshold"]

    def __init__(self, initial_observation, *, step_dt, position_bounds, seated_height_m,
                 retry, settings=None):
        self.settings = settings or RetrySettings()
        self.dt = step_dt
        self.bounds = tuple(position_bounds)
        self.seated_height = seated_height_m
        self.retry = retry
        self.initial_action = self._observation(initial_observation)[-7:]
        self.initial_action = [max(-1.0, min(1.0, x)) for x in self.initial_action]
        self.phase = "hold"
        self.phase_started = 0.0
        self.history = deque(maxlen=math.ceil(self.settings.stall_window_s / self.dt))
        self.short_history = deque(maxlen=max(3, math.ceil(0.2 / self.dt)))
        self.best_height = None
        self.last_progress = 0.0
        self.trigger = None
        self.events = []
        self.integral_xy = [0.0, 0.0]
        self.pressure_offset = self.settings.initial_pressure_offset_m
        self.search_contact_height = None
        self.captured_xy = None

    @staticmethod
    def _observation(observation):
        if len(observation) != 24 or not all(math.isfinite(x) for x in observation):
            raise ValueError("Expected 24 finite actor measurements")
        return list(observation)

    def _transition(self, phase, elapsed, **measurements):
        self.events.append({"time_s": elapsed, "from": self.phase, "to": phase, **measurements})
        self.phase, self.phase_started = phase, elapsed

    def act(self, observation, elapsed):
        obs = self._observation(observation)
        s = self.settings
        xyz, force = obs[:3], math.sqrt(sum(x * x for x in obs[13:16]))
        self.short_history.append((xyz, force))
        height = median(row[0][2] for row in self.short_history)
        load = sum(row[1] for row in self.short_history) / len(self.short_history)
        action = self.initial_action.copy()
        if elapsed >= s.hold_s and self.phase == "hold":
            self._transition("insert", elapsed)
        if self.phase == "insert":
            self.history.append(load >= s.stall_load_n)
            if self.best_height is None or height < self.best_height - s.stall_progress_m:
                self.best_height, self.last_progress = height, elapsed
            stalled = (len(self.history) == self.history.maxlen
                       and elapsed - self.last_progress >= s.stall_window_s
                       and sum(self.history) / len(self.history) >= s.stall_load_occupancy)
            if elapsed >= s.first_attempt_s and stalled and self.trigger is None:
                self.trigger = {"time_s": elapsed, "observed_height_m": height,
                                "noisy_load_mean_n": load,
                                "load_occupancy": sum(self.history) / len(self.history)}
                if self.retry:
                    self._transition("withdraw", elapsed, **{k: v for k, v in self.trigger.items() if k != "time_s"})
        if (self.phase == "withdraw" and elapsed - self.phase_started >= s.withdrawal_s
                and height - self.trigger["observed_height_m"] >= s.withdrawal_rise_m):
            self._transition("search", elapsed, observed_rise_m=height - self.trigger["observed_height_m"])
            self.search_contact_height = self.trigger["observed_height_m"]
        if self.phase == "insert":
            action[:3] = [0.0, 0.0, self.seated_height / self.bounds[2]]
        elif self.phase == "withdraw":
            action[:3] = [0.0, 0.0, s.withdrawal_height_m / self.bounds[2]]
        elif self.phase in {"search", "seat"}:
            time_in_search = elapsed - self.phase_started
            if self.phase == "search":
                radius = min(s.spiral_max_radius_m, s.spiral_radial_speed_m_s * time_in_search)
                angle = s.spiral_angular_speed_rad_s * time_in_search
                desired_xy = [radius * math.cos(angle), radius * math.sin(angle)]
                if height < self.search_contact_height - s.capture_depth_m:
                    self.captured_xy = [median(row[0][axis] for row in self.short_history) for axis in range(2)]
                    self._transition("seat", elapsed, observed_depth_gain_m=self.search_contact_height - height)
                    desired_xy = self.captured_xy
            else:
                desired_xy = self.captured_xy
            for axis in range(2):
                self.integral_xy[axis] += s.lateral_integral_gain * self.dt * (desired_xy[axis] - xyz[axis])
                self.integral_xy[axis] = max(-s.lateral_integral_limit_m, min(s.lateral_integral_limit_m, self.integral_xy[axis]))
                action[axis] = (desired_xy[axis] + self.integral_xy[axis]) / self.bounds[axis]
            self.pressure_offset += s.force_integral_gain_m_n_s * self.dt * (s.insertion_load_n - load)
            self.pressure_offset = max(0.002, min(0.025, self.pressure_offset))
            action[2] = max(self.seated_height - 0.004, height - self.pressure_offset) / self.bounds[2]
        action = [max(-1.0, min(1.0, x)) for x in action]
        return action, {"phase": self.phase, "noisy_load_mean_n": load, "observed_height_m": height,
                        "trigger": self.trigger, "pressure_offset_m": self.pressure_offset,
                        "integral_xy_m": self.integral_xy.copy()}

    def report(self):
        return {"controller": "scripted_actor_retry" if self.retry else "scripted_continued_insertion",
                "settings": asdict(self.settings), "trigger": self.trigger, "events": self.events,
                "scope": "Development feasibility controller; no learned actions or evaluator contact input."}
