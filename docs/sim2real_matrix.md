# Sim2Real Randomization Matrix

| Gap | Implemented simulation distribution | Cadence/scope | Deployment interpretation | Actor exposure | Current evidence |
| --- | --- | --- | --- | --- | --- |
| Gravity | Fixed `(0,0,0)` m/s2 | Every profile | Microgravity baseline; omits residual acceleration | Indirect | Config contract; simulator smoke does not measure acceleration |
| Blade mass | Uniform 5-15 kg with recomputed inertia, independently for failed and spare blades | Startup, per environment | Payload, packaging, and compute configuration | Critic only | Level-2 secured-grasp physics smoke passed; no trained randomized policy |
| Rail Coulomb friction | Dynamic 0.2-1.5, static 0.25-2.0, restitution 0-0.05, 32 buckets | Startup, slot and side guides | Oxidation, thermal cycling, dry lubricant, surface finish | Critic only | Bounds encoded; not calibrated against a real rail |
| Rail stiction | Smooth breakaway 10-120 N plus viscous 2-25 N s/m, clamped to 160 N | Per episode, failed and spare rails | Thermal welding and stuck blades | Critic only | Level 3 reaches geometry but fails low-velocity settling; no physical identification |
| Reset pose/velocity | Millimetre-scale blade position jitter, small orientation and velocity jitter | Per episode, per environment | Assembly tolerance and residual motion | Teacher/critic through state; student through RGB | Runtime vision smoke passed |
| Mount compliance | D6 limits +/-15 mm and +/-2 degrees; translational gains 12,000/220, rotational gains 600/50 | Fixed mechanical model | Flexible satellite panel or imperfect arm fixture | Proprioception/critic | Runtime construction exercised; gain identification pending |
| Mount disturbance | Wrench pulses lasting 0.2-0.8 s, up to +/-30 N and +/-6 N m, with 0.75-2.0 s quiet intervals | Interval, per environment | Structural vibration, reaction wheels, crew contact | Indirect | Encoded; frequency spectrum is not flight-data-derived |
| Sun pose | One DistantLight, yaw 0-360 degrees, pitch 20-80 degrees, angular size 0.10-0.45 | Reset, global to vision scene | Orbital attitude and hard moving shadows | RGB only | Dark background and cross-environment visual variation measured in 8-env smoke |
| Sun radiometry | Intensity 2,500-8,000 and color temperature 5,500-7,500 K | Reset, global to vision scene | Direct sun and partial occlusion proxy | RGB only | Range encoded; values are artistic units, not calibrated irradiance |
| Rack appearance | Steel-gray or gold ranges; metallic 0.65-1.0; roughness 0.08-0.55 | Pre-startup, per environment | Bare chassis, MLI, and specular reflection | RGB only | Overall cross-environment image variation measured; rack contribution and material labels were not isolated |
| Camera degradation | Independent additive Gaussian sigma 0.025, clipped to `[0,1]` | Every RGB observation/pixel | First-order radiation/thermal read-noise proxy | RGB only | Repeated-frame delta sigma 0.02469 in vision smoke |

The matrix deliberately separates actor-visible measurements from privileged
parameters. The vision actor mapping contains only `proprio` and `rgb`; the
diagnostic blade-pose group is excluded. This prevents a simulation-only object
pose or randomized coefficient from leaking into the deployment policy.

## Coverage gaps before a physical transfer claim

| Missing factor | Why it matters | Proposed next measurement/model |
| --- | --- | --- |
| Robot actuator delay, backlash, torque saturation, and thermal drift | Changes contact timing and insertion stability | Identify from UR10e joint logs; randomize delay/gain/friction and enforce hardware limits |
| Camera intrinsics/extrinsics, distortion, exposure, blur, hot pixels, and temporal persistence | Gaussian white noise alone does not match radiation-damaged imagery | Calibrate with the target camera; fit spatial/temporal noise and exposure distributions |
| Rail and connector force-displacement curves | Success based only on pose can reward damaging insertion | Instrument a representative slot; add force/torque limits and connector engagement model |
| Blade/rack CAD tolerances and cable/connector geometry | Collision proxies may overestimate clearance | Import simplified measured CAD with conservative collision margins |
| Residual acceleration and satellite attitude rates | Perfect zero gravity omits drag, jitter, and rotation-frame effects | Replay bounded 6-DoF acceleration profiles from mission assumptions |
| Vacuum thermal effects, outgassing, electrostatics, and lubrication aging | Can dominate friction and adhesion in orbit | Convert test/mission envelopes into held-out parameter sets |
| Real detector and pose-estimator errors | End-to-end student robustness depends on more than image appearance | Evaluate the deployed perception/policy stack on recorded and HIL sequences |

Randomization ranges are engineering priors, not identified probability
distributions. Report performance both inside the training ranges and on
held-out edge cases. A physical Sim2Real claim requires real or HIL results with
success rate, insertion force, pose error, cycle time, and categorized failures;
the current repository does not yet contain those results.
