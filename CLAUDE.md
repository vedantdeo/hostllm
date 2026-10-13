# hostllm — notes for Claude

Project rules live here rather than in a session's private memory, so they travel with the repo and
reach anyone who clones it. Six sections — constants, tests, tests for new features, reference data,
comments and paid calls — are verbatim mirrors of global preferences in `~/.claude/CLAUDE.md`, which
is machine-local and backed up by nothing; these copies are the durable ones. Edit both, or neither.

## Standalone

hostllm serves any causal LM with LoRA adapters behind an OpenAI-compatible API. **It imports
nothing from underhood or aigent**: underhood publishes adapters and tells hostllm what to host;
aigent calls the endpoint over HTTP. Anything it needs, such as a sampler, it writes for itself.

The chain is aigent → underhood → hostllm: each depends on the next and never the other way. In CI
a green hostllm `main` starts underhood's mains CI, which starts aigent's, and hostllm joins the
joint release.

**In this repo**, the tests are the spec and come first: written red, then the bodies against them.
Paid steps — a Modal deploy, a bench run against a live endpoint — each need their own go-ahead,
with the dry-run estimate first.

## Jobs that finish live in underhood; servers that stay up live in hostllm

**The rule (2026-10-13):** underhood runs work that ends — training, evals, censuses — and the
compute for it (`underhood.gpu`: Kaggle kernels, the Hub, run pricing). hostllm runs servers that
stay up: the engine, its deployments (Modal, JarvisLabs Serverless) and their benchmarks. underhood
publishes adapters and names what to host; hostllm imports nothing from it.

**Why:** a batch kernel that runs once and exits shares little with an endpoint that starts on
demand, keeps weights warm and is measured under load. If hostllm later needs part of
`underhood.gpu`, lift out that generic part then, shaped by a real second user.

## `main` only takes pull requests

Public, like aigent and underhood, and `main` is guarded by the same ruleset, "main via PR": no
direct push, no force push, no deletion, and `mains / lint · types · tests` passes before a merge.
Vedant's merge is the approval. **How to apply:** "push" means push the `vedant` branch and open a PR
with `gh pr create`, each with its own confirmation. Never merge a PR.

## Constants and configuration

**The rule: tunable constants live in one config module per project** — `config.py`, `settings.py`,
or whatever the project already calls it — rather than scattered beside the code that uses them.
Qualify names by call site when they would otherwise collide: `MAX_TOKENS_CHAT`, not a local
`MAX_TOKENS` in every module.

- Move the settings: limits, caps, timeouts, retries, defaults, model names, spending ceilings.
- Leave behind what is not a setting: values derived at import from `__file__` (those are locations,
  not knobs), and private algorithm guards nobody would tune from outside.

**Why:** a number sitting alone in its own module is invisible. Side by side the set reads as a set —
you can see which call is the expensive one, notice a value that drifted from its neighbours, and
change a policy in one edit instead of six.

**The fallback, when one config module stops working: split the config — never scatter the constants
back into the code.** Two seams are legitimate:

- **By concern**, once the file is genuinely too big to read: settings in one module, the arithmetic
  that consumes them in another.
- **By isolation**, when a module is deliberately free of internal imports so it can be lifted into
  another project or framework unchanged. It gets its own config module travelling beside it, not a
  dependency on the package-wide one. Both properties then survive — one place per concern, and no
  import. Say so in that module's docstring.

Don't split pre-emptively, don't split alphabetically, and never split into one config per consumer.
The fallback exists so the rule survives a hard case, not so it can be worked around.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss of
this machine. Edit both, or neither.

## Tests

**The rule: when several tests differ only in their inputs and their expected outputs, make them
one parametrized test over a table of rows** — not one function per case. A table is the shape the
knowledge already has: a column for each thing that varies, a row per case.

- Club together what differs only in data: the same call, the same assertions, different arguments.
- Give every row an id that reads as a sentence about the case (`id="a symlink pointing out"`), so
  the output names the broken case without anyone opening the file. Plain tuples are fine where the
  values speak for themselves; reach for `pytest.param(..., id=...)` where they do not.
- Keep apart what differs in *shape*: different setup, a sequence of calls, several objects
  interacting. Needing a column of optional expectations that most rows leave empty is the signal
  that those cases are not one test.
- Share fakes and fixtures through `conftest.py` rather than copying a stub into each module that
  needs it. Two copies of a fake client drift, and the drift stays invisible until one is wrong.
- Assert against the row's own data, and pass the interesting value as the assertion message
  (`assert score.passed is passed, score.detail`), so a failing row explains itself.

**Why:** ten near-identical functions hide the one line that differs between them, and adding the
eleventh case means copying a function instead of adding a row. A table makes the coverage readable
as a set — you can see at a glance which case is missing — and it is the difference between a suite
that grows by editing and one that grows by cloning.

**Don't table for its own sake.** A row that needs its own assertions is a test, not a row, and a
table that grows more columns than rows has stopped being one. Fewer test *functions* is the goal;
fewer lines is a frequent side effect, not the point.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss
of this machine. Edit both, or neither.

## New features ship with their tests

**The rule: a push that adds a feature carries the tests that pin it down, in the same push.** Not a
follow-up commit, not "tested by hand", not covered by a live run alone. Comprehensive means every
behaviour the feature promises has a test that fails when that behaviour breaks:

- The ordinary path, each branch the code takes on its input, every refusal, and every error it
  turns into data rather than raising.
- Each guard — a lock, a ceiling, a validation, a cancel — has a test that fails with the guard
  removed. Check by deleting it and running the suite, not by reading the test.
- A test that cannot fail is not cover. Where a property cannot be shown (a race the GIL hides),
  drop the test and say so in the decision record, rather than keep one that passes either way.
- Paid or slow behaviour is tested against fakes. A live test is extra cover, never the only cover.
- Before proposing a push, name the feature's behaviours and the test that holds each one; a
  behaviour with no test is a gap to close or to state, never to leave silent.

**Why:** an untested feature is a claim. The first refactor breaks it without a sound, and the next
reader cannot tell what it was meant to do. Written alongside the code, the tests are the cheapest
spec there is; written later, they describe whatever the code happens to do by then.

**What is not a feature:** a docs or data change, a config value moved, an experiment run, a
rename. A bug fix is not a feature either, but it brings the test that would have caught it.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss
of this machine. Edit both, or neither.

## Reference data stays in one canonical order

**The rule: a file that people hand-edit and code reads back — a lookup table, an allowlist, a
mapping, a set of fixtures — is written in one canonical order, sorted by the key it is looked up
by.** Sort on write, in the code that dumps it, not as a tidy-up pass afterwards.

- Sort by the key the data is addressed by: the ticker in a ticker table, the package in a
  dependency list, the id in a fixture file. Where there is no natural key, sort by whatever a
  reader would scan for.
- Keep the shape inside each entry stable too — the same fields in the same sequence — so a changed
  value is a one-line diff and not a rewritten block.
- Guard it with a test rather than a convention. A canonical order that nothing checks survives
  exactly until the first tool writes the file back in its own order.

**Why:** a canonical order gives a new entry exactly one place it can go, so its diff shows that
entry and nothing else. Append-anywhere files drift into a private order that only the last editor
knows, and then the first process to re-serialise one produces a diff touching every line — which
buries the single line that actually changed and makes reviewing a data file worthless. Sorted data
is also scannable: a human looks a key up instead of searching for it.

**The exception is when the order is itself the data.** A migration list, a pipeline's stages, a
priority ranking, a changelog, a fallback chain — these are sequences, and sorting them destroys
meaning rather than revealing it. Say so in the file, so the next reader knows the disorder is
deliberate and the next tool does not helpfully fix it.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss
of this machine. Edit both, or neither.

## Comments and docstrings both stay short

**The rule: a comment is one line, and a docstring is a brief description of the module, class or
function — not an argument for it.** A sentence or two saying what the thing is, plus anything a
caller cannot read off the signature. That is the whole budget.

- Say only what the code does not. `# increment i` is noise; `# the API 1-indexes pages` is not.
- Put it at the line that surprises, not at the top of the block.
- **Rationale is not documentation.** Why a default was chosen, what a run measured, which bug a
  guard was written for, what was tried and rejected — none of that belongs beside the code. It
  goes wherever the project keeps its decision record, which is read once on arrival.
- When a comment grows past its line, or a docstring past its couple of sentences, that is the
  signal to move it, not to wrap it.

**Why:** ten lines above a one-line constant do not get read, they get scrolled past — and then they
rot unnoticed, because nobody was reading closely enough to catch the drift. A docstring that argues
its own case is the same failure at larger scale: it buries the one sentence the reader needed, and
it goes stale the first time the reasoning changes without the code.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss
of this machine. Edit both, or neither.

## Paid calls

**The rule: never send a paid API call without explicit per-action confirmation.** Each run needs
its own — previous approval is single-use, exactly as it is for `git commit` and `git push`. A
one-off diagnostic call is not exempt: a two-cent probe is still someone else's money, spent on
your initiative.

- Say what the call is for and what it will cost, then wait. Never report a paid call as already
  running.
- Where a script has a dry-run flag, run *that* first and surface the number it prints, rather than
  estimating the cost in prose.
- Token counting is free. Use it freely, and say it is free when a plan rests on it, so the
  unpriced half of the work is never mistaken for the priced half.
- "It was only a few cents" is the reasoning that erodes this. The amount is not the point; the
  decision belongs to whoever is paying.

**Why:** cost ceilings, pre-flight counts and dry-run flags all exist to put the spend in front of
a person before it happens. Spending unasked routes around machinery someone deliberately built,
and the habit scales badly — the instinct that fires off an unapproved probe is the one that later
fires off an unapproved hundred-row run.

This section is mirrored verbatim into each project's tracked `CLAUDE.md`, so it survives the loss
of this machine. Edit both, or neither.
