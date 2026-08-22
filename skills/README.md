# Qwen-Image-Edit Fine-Tune on Intel AI PC — Step-by-Step Skills

A set of **8 self-contained skills** for fine-tuning **Qwen-Image-Edit** with **NF4 QLoRA**
on an Intel AI PC (Core Ultra processor) Windows laptop. Each skill covers exactly one phase
of the workflow, and the skills run in order (1 → 8).

## The 8 skills

| Order | Skill (folder name = skill name) | Phase |
|---|---|---|
| 1 | `qwen-image-edit-aipc-finetune-01-preflight` | Human prerequisites: driver, registry, conda, oneAPI, model download |
| 2 | `qwen-image-edit-aipc-finetune-02-dataset` | Dataset preparation, validation, held-out test set |
| 3 | `qwen-image-edit-aipc-finetune-03-env-setup` | Python env: conda, torch+xpu, project deps, hardware probe |
| 4 | `qwen-image-edit-aipc-finetune-04-config` | Trainer detection + recommended training config + config lint |
| 5 | `qwen-image-edit-aipc-finetune-05-adaptation` | XPU/NF4 source patches to the qwen-image-finetune clone |
| 6 | `qwen-image-edit-aipc-finetune-06-quantize` | One-time NF4 pre-quantization (conditional; transformer + text_encoder) |
| 7 | `qwen-image-edit-aipc-finetune-07-training` | Cache phase + fit phase (the actual training) |
| 8 | `qwen-image-edit-aipc-finetune-08-validation` | Post-training visual comparison + evaluation |

Run them **strictly in order, 1 → 8**. Each skill ends with a **PASS signal** (a concrete,
checkable outcome) and names the next skill. The next step is only meaningful once the current
step's PASS signal is genuinely met — whether the agent pauses between steps or runs through
them continuously (see "Step C: run the skills").

---

## Step A: add the skills to a skills root

Copy the 8 folders below as **direct children** of your agent's skills root — for Claude Code,
that's `.claude/skills/` in a project, or `~/.claude/skills/` for your user account.

**Result** (`<skills-root>` is your agent's skills directory):

```
<skills-root>/
├── qwen-image-edit-aipc-finetune-01-preflight/
│   └── SKILL.md
├── qwen-image-edit-aipc-finetune-02-dataset/
│   └── SKILL.md
├── qwen-image-edit-aipc-finetune-03-env-setup/
│   └── SKILL.md
├── qwen-image-edit-aipc-finetune-04-config/
│   └── SKILL.md
├── qwen-image-edit-aipc-finetune-05-adaptation/
│   └── SKILL.md
├── qwen-image-edit-aipc-finetune-06-quantize/
│   └── SKILL.md
├── qwen-image-edit-aipc-finetune-07-training/
│   └── SKILL.md
└── qwen-image-edit-aipc-finetune-08-validation/
    └── SKILL.md
```

## Step B: clone the training framework

Clone the upstream framework to a local directory. You will run your agent from
inside this directory.

```bat
git clone --no-checkout https://github.com/tsiendragon/qwen-image-finetune <local-path>
cd <local-path>
git checkout main -- . ":(exclude)docs/plan/v1.6.0:sampling_during_train.md" ":(exclude)tests/src/trainer/ /"
```

> **Why not a plain `git clone`?** The upstream repository contains two files with
> Windows-incompatible names (a colon and a space-only directory). A plain `git clone` fails
> the checkout entirely and leaves the working tree empty. The commands above use git's
> `:(exclude)` pathspec to skip those two paths. After cloning, `git status` shows those two
> paths as `deleted` — expected, and it does not affect the training workflow. If upstream
> removes those files, the `:(exclude)` patterns match nothing and all files check out normally.

The agent applies all XPU adaptations to this clone during Step 5 — they are not auto-applied.

## Step C: run the skills

Agents differ in how they move through a multi-step series, and both behaviors are normal:

- **Paced** — the agent stops after each step and waits for you to say "continue" before loading
  the next skill.
- **Continuous** — the agent runs straight through several steps in one go, loading the next
  skill on its own without pausing.

You do not need to force either behavior. What matters is not *whether* the agent pauses, but
that the **hands-on prerequisites actually get done** — see the callout below.

1. **Open your agent inside the cloned directory** (Step B), with the 8 skills added (Step A).

2. **Start Step 1.** Say, for example:

   > "I want to fine-tune Qwen-Image-Edit on my Intel AI PC. Start with step 1, and go through the steps in order."

   The agent loads `qwen-image-edit-aipc-finetune-01-preflight` and begins.

3. **Let the agent work through the steps in order.** Each skill ends with a PASS signal and
   names the skill that follows it, so the agent can advance on its own:
   - If your agent **pauses** between steps, tell it to continue when you're ready (e.g. *"That
     passed, do the next step"*).
   - If your agent **runs continuously**, just let it proceed — it will load each next skill
     itself. You can interrupt at any time to inspect results or ask it to slow down.

4. **Supply the hands-on items when they come up.** Steps 1 and 2 (and part of Step 6) depend on
   physical actions only you can do — installing software, downloading the model, preparing the
   dataset. Provide these when the agent reaches them.

5. **Finish at Step 8.** After Step 8's validation, the walkthrough is complete.

> **Before each step, its PASS signal must be genuinely met** — later steps build on the state
> and files earlier steps produce (see "How the steps connect"). A continuous agent may race past
> a human-gated item (e.g. oneAPI in Step 1, or the dataset in Step 2) before you've done it,
> breaking later steps. If that happens, stop it, finish the item, and re-run that step's check.

> **If the agent doesn't seem to see a skill**, re-check Step A — the most common cause is
> nesting a skill folder under an extra wrapper directory instead of making it a direct child
> of your skills root.

### What you do vs. what the agent does

- **You (one-time, hands-on)**: clone the training framework repo (Step B) · Intel Arc driver
  install · Miniforge / conda install · Visual Studio Community + C++ workload · Intel oneAPI
  Base Toolkit · Windows long-path registry tweak · GPU shared-memory registry tweak · model +
  dataset on disk · visual inspection of training results.
- **The agent**: hardware probe · training config recommendation · conda env creation ·
  `torch+xpu` install · framework XPU adaptations · NF4 pre-quantization · cache build ·
  training launch · inference image generation (base + LoRA) for your review.

---

## How the steps connect

Each skill produces something the next one needs. Key hand-offs:

- **Step 1 → Step 3**: oneAPI version confirmed → determines the `torch+xpu` version to install
- **Step 1 → Step 4**: model path + confirmed RAM tier → needed for config recommendation
- **Step 2 → Step 4**: validated dataset path → needed for config recommendation
- **Step 3 → Step 4**: conda env active with torch+xpu → `probe_hw.py` can run
- **Step 4 → Step 5**: `probe.json` + linted `config.yaml` (trainer set) → framework adaptation can proceed
- **Step 5 → Step 6**: `adaptation_check.py` PASS → safe to quantize
- **Step 6 → Step 7**: NF4 files on disk + `transformer_path`/`text_encoder_path` in config → training can start
- **Step 7 → Step 8**: checkpoints written → validation can run

**Step 6 is conditional.** Whether you pre-quantize — and which components — depends on your
machine's RAM tier.

## When something goes wrong

Each skill has a Troubleshooting section scoped to its own phase. If an error isn't covered
there, its root cause could lie in an earlier step — for example, a partial Step 5 framework
adaptation surfacing as a failure in Step 7 training. Re-running Step 5's
`adaptation_check.py` is a good way to rule that out.
