# Why the chain uses guarded insertion

Guarded insertion remains the deployed controller because the learned alternatives
have not shown a consistent advantage on the states and rack geometry the chain
actually supplies. A high isolated skill score does not settle that comparison.

## Current comparison

The force-aware v33force policy passed its isolated test: **2,977/3,001 = 99.20%**,
with three held-out evaluation seeds and one training seed. In the chain, on the
prescribed 11.065 mm channel, it scored **24/96**, compared with **23/96** for
guarded insertion. These paired chain arms release the robot's supports and
recheck the free module; they do not include the destination rack pawls.

[Isolated certificate](../evidence/grapple_insert_v33force_c11065_certification.json),
[learned chain](../evidence/workflow_robot_carried_insert_v33force_c11065_chain_policy_n96_certification.json),
[guarded chain](../evidence/workflow_robot_carried_insert_v33force_c11065_chain_guarded_n96_certification.json)

Geometry changes the comparison:

| Clearance per side | Guarded insertion | Learned force-aware insertion |
| --- | ---: | ---: |
| 11.065 mm, prescribed bay | 7/24 | 8/24 |
| 15.678 mm, relieved bay | 17/24 | 4/24 |

The same seeds and checkpoint set were used across these four cells, without
rack pawls. Widening the bay helps guarded control and moves the learned arm in
the opposite direction; the learned arm's change alone is not statistically
resolved at this sample size. The larger paired 96-episode comparison at the
prescribed bay differs by only one success.
[Bay/controller factorial](../evidence/seating_bay_factorial_v1.json)

## What the guard does

The controller advances the robot only when the current estimator detects a
marker and places the module inside the geometric entry envelope. It otherwise
holds. The robot-side form lock becomes compliant during mating; rack retention
engages after measured seating, then both robot supports release and a separate
0.70 s rack-only check must pass.

The current camera application uses the previously derived lead-in catch for
admission. This differs from the original estimator-noise bounds and is recorded
as a separate arm. The final seating tolerances have not changed. Module pose
comes from RGB-D; robot kinematics supply velocity after capture.

## Earlier negative results

v24 scored 36.77% in isolation and 0/96 on recorded chain handoffs. Earlier reward
variants converged to 84.26, 84.61 and 84.58 mrad against a 52.4 mrad acceptance
tolerance. Those experiments show that the tested objectives and reset conditions
did not solve the integration problem; they do not prove a universal limitation
of learned control. Training reward is not a success rate.

[Handoff comparison](../evidence/seating_controller_head_to_head.json),
[attitude diagnosis](../evidence/insert_attitude_diagnosis.json)

A learned replacement must beat the guarded controller pooled and on every
shared seed, with matched geometry, sensing, handoffs and release criteria.
[Next experiment](NEXT_WORK.md#t13-learned-seating-transfer)
