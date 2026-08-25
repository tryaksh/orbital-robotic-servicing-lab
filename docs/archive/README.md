# Archive

Session handoffs and retired state files, in the order they were written. Every
one has been superseded by [`../NOW.md`](../NOW.md), which is the file to read
for current state, and [`../NEXT_WORK.md`](../NEXT_WORK.md), which is the file to
read for what to do next.

These are kept for two reasons: the reasoning that produced a decision is often
more useful than the decision, and the negative results in them stop a later
session spending a night rediscovering that something does not work. Do not quote
a rate or a constant from any of them.

| File | Written | What is in it that is not elsewhere |
| --- | --- | --- |
| `claude_opus_5_handoff.md` | Start of the branch | The owner's original task statement: carry the module with the robot, no hidden carrier, prove an interface limitation with measurements rather than working around it. Everything since is an answer to this. |
| `robot_carried_handoff.md` | After the first robot-carried attempt | The measurement that killed the passive parallel-jaw carry, and the reasoning that led to a robot-side form lock instead of a module-side feature. Four failed mechanical attempts at constraining attitude. |
| `next_session_handoff.md` | After the workcell move | Why the robot base sits where it does, why widening the channel and shortening the module are both dead ends, and the reach and authority sweeps behind the lateral rail. |
| `final_session_handoff.md` | After the chain reached 96.88% | Six defects that had each been reported as a physical result, the depth-dependent attitude envelope that was built and refuted before it ran, and the first evidence that the published skill certifications described a module that no longer existed. Its closing suggestion — that extract needs more epochs — is refuted in the next file. |
| `grip_criterion_handoff.md` | Before the 97.92% chain | The full working of why extract was stuck: the module cross-section experiment, the grip criterion that charged a tapered pin's own load path as a dropped module, the reset that was producing unwinnable episodes, and the lead-in placement bug that took the chain to 0.00% for one run. `../NOW.md` carries the conclusions; this carries the derivations. |
| `status_2026-08-24.md` | The state file, as of 2026-08-24 | The long-form session narrative behind the current numbers: the extract attribution ladder step by step, the 12.0 mm pin-feed measurement over 433 extractions, the audit that found the insert skill and the chain describing different problems, and the reasoning for each. Superseded as a *status* file by `../NOW.md` on 2026-08-25; kept for the derivations, which are not repeated there. |

## A note on `docs/status.md`

Comments in several scripts cite `docs/status.md`, a rolling status file pruned
in commit `0b28191`. Its content was split: the current state went into
[`../NOW.md`](../NOW.md) and the reasoning behind superseded decisions is
in the handoffs above. Read those citations as "the project's status record"
rather than as a path.
