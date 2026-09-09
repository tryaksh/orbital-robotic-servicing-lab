"""Initialization-only numeric faults around the unchanged pinned peg geometry."""
from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply

from assembly_recovery.peg_env import PegStudyEnv, WithinJobMutationError
from assembly_recovery.training_inputs import bounded_initial_actions


class PegFaultEnv(PegStudyEnv):
    def __init__(self, cfg, *, cases, nominal, **kwargs):
        if len(cases) != cfg.scene.num_envs:
            raise ValueError("One declared fault case is required per requested environment")
        self.fault_cases = cases
        cfg.obs_rand.fixed_asset_pos[:2] = [0.0, 0.0]
        cfg.task.hand_init_pos[2] = nominal["hand_height_above_fixture_m"]
        cfg.task.hand_init_pos_noise = [0.0, 0.0, nominal["hand_height_noise_m"]]
        super().__init__(cfg, **kwargs)

    def set_pos_inverse_kinematics(self, ctrl_target_fingertip_midpoint_pos,
                                   ctrl_target_fingertip_midpoint_quat, env_ids):
        if self.jobs_live:
            for job in self.jobs:
                job.forbidden_event("within_job_state_write")
            raise WithinJobMutationError("Fault initialization IK is forbidden within a job")
        target = ctrl_target_fingertip_midpoint_pos.clone()
        target[:, :2] += torch.tensor([c["approach_xy_m"] for c in self.fault_cases], device=self.device)
        return super().set_pos_inverse_kinematics(target, ctrl_target_fingertip_midpoint_quat, env_ids)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        # Replace only controlled XY bias. Keep the sampled shared Z observation error.
        bias = torch.tensor([c["fixture_bias_xy_m"] for c in self.fault_cases], device=self.device)
        delta = bias - self.init_fixed_pos_obs_noise[:, :2]
        self.init_fixed_pos_obs_noise[:, :2] = bias
        bounds = torch.tensor(self.cfg.ctrl.pos_action_bounds[:2], device=self.device)
        self.actions[:, :2] -= delta / bounds
        self.prev_actions[:, :2] = self.actions[:, :2]
        materials = self._fixed_asset.root_physx_view.get_material_properties()
        for i, case in enumerate(self.fault_cases):
            materials[i, :, 0] = case["fixed_static_friction"]
            materials[i, :, 1] = case["fixed_dynamic_friction"]
        self._fixed_asset.root_physx_view.set_material_properties(materials, env_ids.cpu())
        # The initial physical pose can lie outside the noisy action frame's box.
        # Bound EMA history exactly as future requested actions are bounded.
        self.initial_actions_before_projection = self.actions.clone()
        self.actions.copy_(bounded_initial_actions(self.actions))
        self.prev_actions.copy_(self.actions)

    def initialization_report(self):
        axis = torch.zeros_like(self.held_pos)
        axis[:, 2] = 1.0
        axis = quat_apply(self.held_quat, axis)
        bottom = torch.minimum(self.held_pos[:, 2], self.held_pos[:, 2] + self.geometry.peg_height_m * axis[:, 2])
        bottom -= self.geometry.peg_diameter_m / 2 * torch.linalg.vector_norm(axis[:, :2], dim=-1)
        clearance = bottom - self.fixed_pos[:, 2] - self.geometry.hole_height_m
        forces = torch.linalg.vector_norm(self.fixture_contact.data.force_matrix_w, dim=-1).sum(dim=1)
        materials = self._fixed_asset.root_physx_view.get_material_properties()
        result = []
        for i, case in enumerate(self.fault_cases):
            actual_material = materials[i].cpu().tolist()
            checks = {
                "peg_cylinder_above_fixture_top": float(clearance[i]) > 0.001,
                "fixture_contact_clear": float(forces[i, 0]) <= 0.1,
                "bilateral_grasp_contact": float(forces[i, 1:].min()) > self.criteria.finger_contact_threshold_n,
                "initial_action_within_bounds": bool((self.actions[i].abs() <= 1).all()),
                "finite_state": bool(torch.isfinite(self.actions[i]).all() & torch.isfinite(self.held_pos[i]).all()),
                "requested_approach_applied": bool(torch.linalg.vector_norm(
                    self.fingertip_midpoint_pos[i, :2] - self.fixed_pos[i, :2]
                    - torch.tensor(case["approach_xy_m"], device=self.device)) <= 0.0015),
                "requested_bias_applied": bool(torch.allclose(self.init_fixed_pos_obs_noise[i, :2],
                                                              torch.tensor(case["fixture_bias_xy_m"], device=self.device), atol=1e-8, rtol=0)),
                "requested_friction_applied": all(abs(m[0] - case["fixed_static_friction"]) < 1e-6
                                                   and abs(m[1] - case["fixed_dynamic_friction"]) < 1e-6 for m in actual_material),
            }
            result.append({"case_id": case["case_id"], "valid": all(checks.values()), "checks": checks,
                           "initial_action_before_projection": self.initial_actions_before_projection[i].cpu().tolist(),
                           "initial_action": self.actions[i].cpu().tolist(),
                           "initial_action_projection_applied": bool((self.initial_actions_before_projection[i] != self.actions[i]).any()),
                           "conservative_peg_cylinder_clearance_m": float(clearance[i]),
                           "fixture_contact_n": float(forces[i, 0]), "finger_contact_n": forces[i, 1:].cpu().tolist(),
                           "actual_fixed_material": actual_material,
                           "actual_approach_xy_m": (self.fingertip_midpoint_pos[i, :2] - self.fixed_pos[i, :2]).cpu().tolist(),
                           "scope": "Positive cylinder clearance is a sufficient configured-geometry nonintersection check at job start; not a mesh certification."})
        return result
