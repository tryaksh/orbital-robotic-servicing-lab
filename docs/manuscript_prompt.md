# Starting the manuscript: setup, and a prompt for a fresh session

## Where the manuscript should live, and why not in this repository

Put it in its own folder and its own git repository — `D:\orbital-servicing-paper\`
is fine. Two reasons. This repository's history is evidence, and mixing eighty
drafts of a paragraph into it makes `git log` useless for the thing it is for.
And a manuscript that lives beside the code invites quoting numbers from memory;
kept separate, every number has to be fetched deliberately.

```
D:\orbital-servicing-paper\
  draft.md              the manuscript, in Markdown, one section per heading
  numbers.md            generated -- every figure the draft may quote, with its source file
  figures/              plots and diagrams, each with the script that made it
  refs.bib              citations
  README.md             how to regenerate numbers.md, and the rule below
```

**The rule:** no number appears in `draft.md` that is not in `numbers.md`, and
`numbers.md` is generated from `evidence/*.json` in this repository by a script.
That is the same discipline the code already runs on, applied to the prose. It is
also the single cheapest defence against the failure mode that would actually
sink this paper — a figure that was true in September and is not in November.

## Markdown first, Overleaf last

Draft in Markdown, not LaTeX. Assistants write markedly better prose when they
are not also generating `\begin{itemize}`, and you will read the drafts more
carefully when they are readable. Frontiers accepts Word and LaTeX; converting a
finished Markdown draft with `pandoc` into their template is an afternoon, and it
happens once, at the end, when the words have stopped moving.

Use Overleaf for the final pass — their template, the reference list, the figure
placement, the submission PDF. Not for the writing.

## The prompt for a fresh session

Copy everything below into a new session started in `D:\6axis-space-robotics`.

---

> I want a first draft of the paper this repository is the evidence for. Work in
> `D:\orbital-servicing-paper\` — create it if it does not exist, and make it a
> git repository of its own. Do not add manuscript files to the code repository.
>
> **Read these first, in this order, and do not start writing until you have:**
> `docs/paper_position.md` — what the paper may and may not claim, and what is
> prior art. It is the most important file and several of its decisions would be
> expensive for you to re-derive. Then `docs/NOW.md` for the measured state,
> `docs/seating_controller.md` and `docs/sim_to_real.md` for two arguments that
> are already written out, and `evidence/MANIFEST.json` for which reports are
> canonical. Never quote a number from a report the manifest calls retracted or
> historical.
>
> **Before writing prose, build `numbers.md`.** Write a small script that reads
> the canonical reports and emits every figure the draft might use, each with the
> file it came from. Then write from that file only. If you find yourself wanting
> a number that is not in it, add it to the script rather than typing it.
>
> **Then draft, section by section, and stop after each for review.** Follow the
> claim structure in `docs/paper_position.md` — it is ordered deliberately.
> Introduction, related work with the separations that document names, the design
> derivation and its library, the simulator's verdict on each derived bound, the
> perception result, methods, limitations, conclusion.
>
> **On the writing itself.** Write it the way a careful engineer writes for
> colleagues: plain declarative sentences, the number and then what it means, no
> throat-clearing. Do not open paragraphs with "Moreover", "Furthermore",
> "Notably" or "It is worth noting that". Do not write "leverage", "utilize",
> "robust" as a filler adjective, "novel framework", "paradigm", or "we posit".
> Never use three adjectives where one is true. Prefer "the module wedges" to
> "a wedging phenomenon is observed to occur". If a sentence would survive being
> cut, cut it.
>
> Say what failed. This project's strongest material is negative — a criterion
> attached to the wrong decision, found twice; a published claim overturned by a
> defect in our own sweep; a policy that could not feel the contact it was making.
> A paper that hides those is weaker and a reviewer will smell the gap.
>
> **What you must not do.** Do not invent a number, a citation, or a result. Do
> not describe an experiment that has not run — check `evidence/` and say
> "pending" if it is pending. Do not claim hardware validation; nothing here has
> run on hardware. Do not use the framing `docs/paper_position.md` forbids.
>
> Start by reading, then show me your outline and the `numbers.md` script before
> you write a single paragraph of prose.

---

## What the draft cannot claim yet, as of 2026-09-03

Check these against `docs/NOW.md` before drafting, because they are the ones most
likely to have moved:

- the pooled camera-driven chain rate with the retrained extraction (running);
- the training-seed spreads for capture and extraction (running);
- whether the seating phase is learned or scripted (one experiment running —
  `Isaac-ZeroG-Blade-GrapplePin-InsertForce-v0`, the first version of that policy
  that can feel contact);
- a perception certificate for the datum layout actually deployed ([T20](NEXT_WORK.md#t20)).
