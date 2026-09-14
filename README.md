# Orbital Robotic Servicing Lab

Design and simulation of a robot that moves a compute module between two rack
bays in zero gravity. The goal is to identify what modular hardware needs for
autonomous servicing: a graspable interface, sufficient clearance, reliable
alignment, and a rack that holds the module after the robot releases it.

A UR10e arm performs the transfer in one continuous Isaac Lab simulation.
PPO policies capture and extract the module; solved inverse kinematics carries
it; guarded control aligns and inserts it. RGB-D cameras track flush markers
on the module. This models the mechanical transfer portion of a changeout,
not electrical reconnection or a complete failed-unit/spare inventory cycle.

## Run a verified mission

The mission application checks its checkpoints and validation evidence before
starting. It records the robot, verifies seating and release, then hashes the
report, video, and motion traces. Its three frozen checkpoints are included
in [policies/servicing_v2](policies/servicing_v2), about 4 MB in total.

```powershell
python -m pip install -e .
zero-g-mission preflight
zero-g-mission run --seed 6070
zero-g-mission verify <path-to-job.json-printed-by-the-run>
```

The live run needs the local Isaac Sim installation described in
[Installation](docs/INSTALL.md). Preflight reports missing requirements.
Verification runs on an ordinary CPU without Isaac or the original checkpoints.
[Mission commands and API](docs/compute_service_demo.md) covers outputs, cancellation,
and reproducing the validation.

The recorded application run passed all nine checks: **1,320/1,320 camera detections**,
**1.370 mm maximum transit drift**, and **0.733 seconds held by the rack alone**
after both robot supports released. This is one simulation episode with stable
lighting, not a reliability rate. [Validation report](evidence/live_service_application_seed6070_v1.json)

## What the experiments show

| Experiment | Result | Scope |
| --- | ---: | --- |
| Full transfer using simulator state | **22/24 (91.67%)**, 95% interval **[74.2%, 97.7%]** | Three held-out seeds; rack-only hold required |
| Original camera-driven transfer | **4/24 (16.67%)** | Same camera pipeline as its 20/24 oracle-pose control |
| Camera pipeline used by the mission application | **17/24 (70.83%)** | Noise-trained extraction, robot-kinematic velocity, and the derived lead-in guard |

These are recorded research cohorts. Both full-transfer results remain below
the unchanged **95%** gate. The application validation above is separate from
these pooled measurements. [Results, evidence, and reproduction details](docs/NOW.md)

A strong isolated skill did not guarantee a strong system: learned force-aware
seating scored **99.20%** alone, but **24/96** inside the chain versus **23/96**
for guarded insertion on the same prescribed rack. The controller's performance
also depended on the bay geometry. [Seating comparison](docs/seating_controller.md)

The mechanical design matters as much as the policies. A wider channel admits
more misalignment but also allows a released module to remain off-centre.
The CPU design tools calculate those competing requirements before another
training run is needed:

```powershell
python scripts/derive_rack_requirement.py
python scripts/check_channel_holds_its_tolerance.py
```

The second command reports the shipped bay as incompatible with a guarantee of
passive seating. It does not say the robot cannot seat a module there; active
alignment and rack retention are part of the solution.

## Limits and evidence

Everything is simulated; nothing has run on hardware. Robot-side and rack-side
locks use **idealized** joint load paths, and the robot base is fixed to the world.
The broader serviceability envelope is **not qualified**.
[Simulation limits and proposed bench tests](docs/sim_to_real.md)

The [evidence manifest](evidence/MANIFEST.json) distinguishes current, historical,
and retracted reports. Some older experiments used uncommitted code and remain
unreproducible; [T0](docs/NEXT_WORK.md#t0) tracks that gap. The mission checkpoints
are now included, while the full training archives and rendered datasets remain
local. Failed results are retained.

Built with Python, NVIDIA Isaac Sim / Isaac Lab, PhysX, PyTorch / RL-Games,
OpenCV ArUco and NumPy. The optional local API uses FastAPI.

[Current state](docs/NOW.md) - [Roadmap](ROADMAP.md)
[Demo provenance](docs/DEMOS.md) - [Script index](scripts/README.md)
[Mechanical specification](docs/service_interface_spec.md) - [License](LICENSE)
