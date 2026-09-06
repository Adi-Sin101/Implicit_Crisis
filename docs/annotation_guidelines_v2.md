# Annotation Guidelines - Implicit Crisis Language Detection

Version 2.0 (post-calibration revision)

## 0. Scope and task

Label language, not people. The task is to classify what a text expresses, not to assess whether an author is clinically at risk. Assign exactly one label to each item. Annotators should read the complete text and record confidence and a concise note for difficult cases.

## 1. Four labels

### `explicit_crisis`

The author conveys their own explicit suicidal or crisis intent, desire, planning, attempt, or direct wish to die. This includes statements such as wanting to die, wanting to end one's life, suicide planning, describing one's own suicide attempt, or direct suicidal intent. Euphemisms count when context shows that the author refers to their own death or suicide.

### `implicit_crisis`

The author conveys their own severe crisis-level psychological distress without clear explicit suicidal intent. Evidence can include severe hopelessness, feeling trapped, profound worthlessness, burdensomeness, finality, escape-from-existence framing, closure behavior, or unbearable circumstances. Ordinary sadness, loneliness, frustration, relationship problems, and temporary unhappiness are not enough.

### `hard_negative`

Suicide, crisis, or self-harm terminology is present, but it is not the author's own current crisis. Examples include discussion of another person's suicide, awareness or support posts, educational content, suicide-related resources or communities, quoted or externally described crisis language, fiction, and clearly non-personal references.

### `non_crisis`

There is no meaningful crisis-level psychological distress or suicidal/self-harm content. Ordinary sadness, loneliness, disappointment, frustration, breakup distress, and vague negative mood belong here unless the text crosses the crisis threshold.

## 2. Existing decision rules retained from v0.1

- The author's own situation is central. Crisis language about another person is `hard_negative`, not `explicit_crisis`.
- Explicit crisis terminology with personal suicidal meaning is `explicit_crisis`; implicit crisis-level distress without clear explicit suicidal intent is `implicit_crisis`.
- Ordinary negative emotion is `non_crisis` by default. Implicit crisis requires meaningful evidence such as self-directed finality, burdensomeness, escape framing beyond the situation, closure behavior, or total hopelessness.
- Mixed posts should be labelled by the author's current state when current crisis is present.
- When explicit terms and implicit content both occur, `explicit_crisis` takes precedence because the surface form carries the crisis meaning.
- Too-short, truncated, or unintelligible text should be `non_crisis` with low confidence and an `uninterpretable` note unless additional evidence supports another label.
- Quoted crisis language about someone else is `hard_negative`.
- Confidence describes certainty in the label decision, not severity. Use low confidence for genuine ambiguity and explain the boundary in notes.

## 3. Calibration rules

### Rule 1 - Profanity is not itself crisis evidence

Profanity, insults, capitalization, repeated punctuation, or emotionally intense wording must not independently determine the crisis label. Focus on semantic meaning and context.

- Profanity without crisis meaning does not imply crisis.
- `I fucking hate this situation` is not automatically crisis.
- `I fucking want to die` is explicit crisis because the underlying statement expresses suicidal intent or desire.

### Rule 2 - Suicidal euphemisms

Expressions such as `off myself`, `end it all`, `check out`, `not be here`, and equivalent euphemisms should be treated as suicidal language when context indicates that the author refers to their own death or suicide. If the expression is clearly metaphorical, joking, hypothetical, or about another person, do not automatically label it `explicit_crisis`; assess the complete context.

### Rule 3 - Second-person or generic language

Do not automatically classify a passage as `non_crisis` merely because it uses `you`. Consider the complete passage. If second-person language clearly describes the author's lived experience, the relevant crisis label may apply. If it is clearly educational, hypothetical, fictional, or about another person, do not infer personal crisis.

### Rule 4 - Context over keywords

Words such as `suicide`, `suicidal`, `self-harm`, `depression`, `die`, `death`, `cutting`, `kill`, or `noose` do not determine the label by themselves. Ask:

1. Who is experiencing the crisis?
2. Is the statement about the author?
3. Is it current and personal, or historical and external?
4. Does it meet the severity definition of the target class?
5. Is the language literal, hypothetical, figurative, joking, or quoted?

### Rule 5 - Hard negative

Use `hard_negative` when crisis or suicide terminology is present but the author is not expressing their own current crisis. This includes discussion of another person's suicide, awareness or educational posts, support posts, suicide communities or resources, and quoted or externally described crisis language.

### Rule 6 - Historical crisis

Do not automatically label a historical experience as current `explicit_crisis`. Determine whether the post describes a past crisis only, provides evidence of current crisis, or makes the historical event itself the subject of the target label. Apply the definitions consistently and document uncertainty in confidence and notes.

### Rule 7 - Insufficient evidence

Do not infer severe crisis from vague negative statements alone. `nothing is good anymore` is not automatically `implicit_crisis`. That label requires meaningful evidence of severe crisis-level distress, not merely ordinary sadness, frustration, loneliness, or a vague negative statement.

### Rule 8 - Ordinary distress versus implicit crisis

Loneliness, relationship problems, disappointment, ordinary sadness, frustration, or temporary unhappiness should generally remain `non_crisis` unless stronger evidence shows severe crisis-level distress. Persistent self-loathing, profound hopelessness, feeling trapped, unbearable distress, or similar severe language can support `implicit_crisis` when it reaches the crisis-level threshold.

### Rule 9 - Document difficult cases

When a case is genuinely ambiguous, choose the best-supported label, lower confidence if appropriate, and provide a concise rationale. Do not resolve ambiguity merely by relying on a keyword.

## 4. Annotation procedure

1. Read the entire post.
2. Decide whether the text expresses the author's own current crisis.
3. If no, use `hard_negative` when crisis terminology is present and `non_crisis` otherwise.
4. If yes, use `explicit_crisis` for explicit suicidal/crisis intent and `implicit_crisis` for crisis-level distress without clear explicit suicidal intent.
5. Check for historical, quoted, fictional, educational, joking, metaphorical, or second-person framing.
6. Record `high`, `medium`, or `low` confidence. Add a concise note for low-confidence or boundary cases.

## 5. Calibration and traceability

The 150-item calibration subset was independently annotated by A and B. The eight disagreements were adjudicated separately after discussion. The original A and B files remain unchanged, and adjudication must never be back-propagated to either independent annotation. Full annotation should proceed only after these rules are accepted and any remaining difficult boundaries are discussed.

## 6. Wellbeing

This material is heavy. Work in short sessions and stop when needed. If an item suggests immediate, identifiable risk to a real person, stop and raise it with the project supervisor rather than handling it alone.
