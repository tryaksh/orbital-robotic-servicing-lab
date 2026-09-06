# Pre-registration: predicting two module geometries never run here

**Written 2026-09-05 before any episode of either configuration existed.** The
runs are queued behind `supervise_jam_mechanism.sh` and have not started. If
this file's git timestamp is not earlier than the first
`artifacts/jam_prediction/**/nominal.npz`, the pre-registration is void and must
be read as a diagnosis instead.

## What is being predicted

Two module cross-sections, at the prescribed channel relief (0.00 mm) with the
retention pawls fitted, three seeds, 64 environments, everything else the
reference workcell:

| tag | width | thickness | what it changes against the nominal 130 x 20 |
| --- | ---: | ---: | --- |
| `wider_140x20` | 140 mm | 20 mm | lateral clearance only: 11.065 -> 6.065 mm per side |
| `thicker_130x26` | 130 mm | 26 mm | vertical clearance only; lateral unchanged |

Neither has ever been run in this repository. The two section points that exist
vary width and thickness together, so they cannot separate the axes.

## What the evidence says so far

The twelve jams the prescribed relief causes are a **yaw** — rotation about the
vertical axis, wedging between the side guides. Measured on the instrumented
prescribed arm, seed 4070: jammed episodes rest at 86.6 to 90.4 mrad of yaw with
6.5 mrad of pitch, and comparing peaks across the stroke, roll and pitch are
indistinguishable between jammed and seated episodes (56.25 against 56.76 mrad;
14.80 against 15.64) while yaw alone separates them, 89.85 against 56.25.

So lateral clearance governs the jam. Vertical clearance does not.

## The simple alternative, given its best shot

The design library already carries a clearance rule, and it is **not silent
here**. `engagement_depth_limit_m` is Whitney's `2c/theta`:

| config | lateral clearance per side | `2c/theta` at 90 mrad | verdict |
| --- | ---: | ---: | --- |
| 130 x 20 (base) | 11.065 mm | 245.9 mm | needs a correcting lead-in |
| 140 x 20 | 6.065 mm | 134.8 mm | needs a correcting lead-in |
| 130 x 26 | 11.065 mm | 245.9 mm | needs a correcting lead-in |

**The existing rule already predicts the direction correctly**: `140x20` is
worse, `130x26` is unchanged. Any claim that the method beats it must therefore
rest on something the rule structurally cannot produce — not on getting the same
ranking. It produces no rate, no phase, and no delivery/precision split, and its
depth limit is wrong by a factor of 2.8 against the observed 88 mm jam.

## Predictions

### `wider_140x20`

| | existing clearance rule | this method |
| --- | --- | --- |
| direction | worse | worse |
| failing phase | not produced | **insert** |
| failure kind | not produced | **precision**: the module is delivered and fails to seat |
| jam rate among delivered | not produced | **>= 25%**, point estimate 30-50% (against 6.4% at 130 x 20) |
| delivery rate | not produced | **>= 0.85**, roughly unchanged from 0.974 |

Reasoning: the jam is a yaw and this configuration removes 45% of the lateral
clearance the yaw has to fit into. Width does not obviously change the grasp, so
delivery should hold up.

### `thicker_130x26` — the discriminating case

| | existing clearance rule | this method |
| --- | --- | --- |
| direction | **no change** | **worse, in a different phase** |
| failing phase | not produced | **extract** |
| failure kind | not produced | **delivery**: modules never arrive |
| delivery rate | not produced | **<= 0.92**, point estimate 0.85-0.92 (against 0.974) |
| jam rate among delivered | not produced | **<= 8%**, near the 6.4% baseline |

Reasoning: lateral clearance is untouched, so by the yaw mechanism the jam rate
should not move. But thickness drove *extraction* losses in the one existing
cohort that varied it — `section_140x26` lost 16 episodes to extract against 1
to capture, inverting the pattern of the thinner `section_120x16` at 33 capture
to 7 extract. A thicker module is harder to pull clear of the source bay.

**This is where the two disagree.** The clearance rule says `130x26` is
harmless. This method says it is harmful in a phase the rule cannot see.

## What would falsify this

* `130x26` comes back clean on delivery (> 0.95): the clearance rule was right,
  this method over-predicted, and the phase-attribution claim loses its only
  discriminating case here.
* `130x26` jams more at insert: the yaw reading is wrong, because lateral
  clearance is unchanged.
* `140x20` does not jam more: the yaw mechanism is wrong.
* Both configurations move together, or neither moves: the framing is wrong
  rather than either branch.

## Scope

Simulation only. One point of the sweep, three seeds, one frozen checkpoint set,
one workcell. A correct prediction here is the second this method has made, not
a validated capability, and the manuscript may not describe it as one.
