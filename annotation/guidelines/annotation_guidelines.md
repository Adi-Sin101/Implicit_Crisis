# Annotation Guidelines — Implicit Crisis Language Detection

Version 0.1 (pre-calibration draft). Revise after the calibration round and
increment the version; record what changed in `annotation/agreement/`.

## 0. What you are labelling

You are labelling **language**, not people. The task is to judge what a piece
of text expresses, not to assess whether its author is at risk. Nothing you
label is a clinical judgement, and no output of this project is used to
identify, contact or assess anyone.

Each item gets **exactly one** of four labels.

You will see only the post text. You will not see which dataset it came from,
what label that dataset gave it, or whether it contains crisis vocabulary.
That is deliberate — those signals are exactly what the project is testing,
and knowing them would bias your judgement.

## 1. The four labels

### `explicit_crisis`

The author conveys their **own** crisis, and the crisis meaning is carried by
overt crisis terminology on the surface.

Markers: *suicide*, *kill myself*, *end my life*, *take my own life*,
*end it all*, *kms*, *want to die*, and direct statements of self-harm intent
or planning.

> "I've been thinking about killing myself all week."
> "I have a plan and I'm going to do it tonight."

The key requirement is *first-person* and *own crisis*. Crisis vocabulary
about somebody else is not this label (see `hard_negative`).

### `implicit_crisis`

The author conveys their **own** crisis or crisis-level distress **without**
crisis terminology. The meaning has to be assembled from context.

Typical forms:

- Perceived burdensomeness — "everyone would be better off without me"
- Thwarted belonging with finality — "nobody would even notice I was gone"
- Hopelessness with no future — "there's no version of this that gets better"
- Farewell / closure framing — giving things away, apologising to everyone,
  "this is my last post"
- Escape-from-existence phrasing — "I just want it to stop permanently",
  "I want to go to sleep and not wake up"
- Finality of exhaustion — "I can't keep doing this, I'm done, I've decided"

**This is the hardest boundary in the project and the most important one to
get right.** See §2.

### `hard_negative`

Crisis vocabulary **is** present, but the text does **not** express the
author's own crisis. Typical cases:

- News, statistics, research, or policy discussion about suicide
- Prevention and advocacy messaging, helpline promotion
- Concern about a third party — "my brother said he wants to kill himself"
- Fiction, lyrics, quotes, or media discussion
- Casual hyperbole — "this homework is making me want to kms",
  "I'd rather die than sit through another meeting"
- Recovery narrative in the past tense with no current crisis — "two years
  ago I wanted to end it, and I'm glad I'm still here"

These items exist to test whether a model has learned meaning or just a
keyword list. Label them carefully; they carry a lot of the study's weight.

### `non_crisis`

Neither crisis meaning nor crisis vocabulary. This includes ordinary negative
emotion: sadness, stress, loneliness, frustration, grief, anxiety, a bad day,
a breakup, exam pressure, burnout complaints.

> "I'm so tired of this semester, nothing is going right."
> "Broke up with my girlfriend, feeling pretty low."

## 2. The critical distinction: implicit crisis vs. ordinary distress

**Ordinary negative emotion is NOT implicit crisis.** Sadness, loneliness,
stress, exhaustion, hopelessness about a *specific situation*, and venting are
`non_crisis` by default.

Do not label something `implicit_crisis` because it *feels* sad or because the
author sounds like they are struggling. Most people who write about being
miserable are not in crisis, and a model trained on the opposite assumption
learns nothing useful.

Apply `implicit_crisis` **only** when the text carries crisis-relevant
contextual evidence. At least one of:

1. **Self-directed finality** — the distress points at the author's continued
   existence, not merely at their circumstances.
2. **Burdensomeness** — the author frames themselves as a net harm to others,
   or others as better off without them.
3. **Escape framing that exceeds the situation** — wanting to permanently stop
   existing, disappear entirely, not wake up.
4. **Closure behaviour** — farewells, giving belongings away, "last" framing,
   settling affairs, apologies addressed broadly.
5. **Total hopelessness** — no imaginable future, all options exhausted, no
   conditions under which things could improve.

Contrast:

| Text | Label | Why |
|---|---|---|
| "I'm exhausted and I hate my job." | `non_crisis` | Situational distress. |
| "I'm exhausted and I don't want to be here any more." | `implicit_crisis` | Escape framing directed at existence. |
| "Nobody texted me on my birthday, I feel invisible." | `non_crisis` | Loneliness, no finality. |
| "Nobody would notice if I stopped existing. I'm just taking up space." | `implicit_crisis` | Burdensomeness + finality. |
| "I've failed every exam, my life is over." | `non_crisis` | Hyperbole about a situation. |
| "I've made my peace with things. Thank you all for everything. Take care of my dog." | `implicit_crisis` | Closure behaviour. |

### When you are unsure

Choose the **lower-intensity** label and record `confidence: low` with a note
saying what made it borderline. Disagreement is expected on this boundary and
is reported honestly rather than smoothed over — do not guess in order to
avoid a low-confidence mark.

Never upgrade a label because a post is upsetting to read.

## 3. Decision procedure

1. Read the whole post.
2. Does it express crisis about the **author's own** situation?
   - No, but crisis words are present → `hard_negative`
   - No, and no crisis words → `non_crisis`
3. If yes: is the crisis meaning carried by overt crisis terminology?
   - Yes → `explicit_crisis`
   - No → check §2 for crisis-relevant contextual evidence.
     - Evidence present → `implicit_crisis`
     - Only ordinary negative emotion → `non_crisis`

Edge conventions:

- **Mixed post** (recovery narrative that ends in current crisis): label by
  the author's *current* state.
- **Both explicit terms and implicit content**: `explicit_crisis` wins — the
  slice is defined by whether the surface form carries the meaning.
- **Too short / truncated / unintelligible**: label `non_crisis` with
  `confidence: low` and note "uninterpretable"; these are reviewed later.
- **Quoting somebody else's crisis words about themselves**: `hard_negative`.

## 4. Filling in the sheet

| Column | What to put |
|---|---|
| `id` | Do not change. |
| `text` | Do not change. |
| `label` | Exactly one of: `explicit_crisis`, `implicit_crisis`, `hard_negative`, `non_crisis` |
| `confidence` | `high`, `medium`, or `low` |
| `notes` | Free text. Required when `confidence` is `low`, or when the item is a boundary case worth discussing. |

Save as CSV with the same columns. Do not reorder or delete rows.

## 5. Process

1. Both annotators independently label the 150-item calibration subset.
2. Agreement is computed (`scripts/annotation/calculate_agreement.py`).
3. Disagreements are discussed; these guidelines are revised where they were
   ambiguous, and the revision is recorded.
4. Full annotation proceeds under the revised guidelines.
5. A portion of the full batch stays double-annotated so a final agreement
   figure can be reported.

Adjudicate remaining disagreements by discussion. If no agreement is reached,
drop the item and record why — do not coin-flip it into the gold set.

## 6. Wellbeing

This material is heavy. Work in short sessions, stop when you need to, and do
not annotate alone for long stretches. If any item suggests an immediate,
identifiable risk to a real person, stop and raise it with the project
supervisor rather than handling it yourself.
