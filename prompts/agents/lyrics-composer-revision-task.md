## Cantonese Lyrics {attempt_label}

The previous draft was evaluated and did **not** meet the quality threshold
(score = {score:.2f}/1.00). You must revise it addressing **all** corrections listed.

---

### Previous Draft
Lyrics:
{prev_lyrics}

Jyutping:
{prev_jyutping}

---

### Validator Feedback
{feedback}

### Required Corrections
{corrections_str}

---

### Constraints (unchanged)
- Total syllable count: **{syllable_count}**
- Strong-beat positions: {strong_str}
- Rhyme positions: {rhyme_str}
- Reference text / artistic concept: {reference_text}
- Lean 0243 target sequence: {tone_seq_str}

---

### Instructions
1. Fix every item in the Required Corrections list.
2. Keep improvements from the previous draft where they already satisfy constraints.
3. Do **not** simply paraphrase the previous draft – make genuine corrections.
4. Ensure the total syllable count equals **{syllable_count}** exactly.

### Output Format
Return a **JSON object only**:

```json
{{
  "lyrics":   "<revised Chinese lyrics, newlines between lines>",
  "jyutping": "<full Jyutping romanisation, space between syllables>",
  "lines": [
    {{"text": "<line1>", "jyutping": "<line1 jyutping>", "syllable_count": 0}},
    {{"text": "<line2>", "jyutping": "<line2 jyutping>", "syllable_count": 0}}
  ],
  "rhyme_scheme": "<e.g. AABB>",
  "changes_made": "<summary of what you changed and why>",
  "notes": "<any additional notes>"
}}
```
