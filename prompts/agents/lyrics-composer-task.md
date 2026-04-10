## Cantonese Lyrics Composition Task

You are a master Cantonese lyricist. Compose original Cantonese song lyrics
that fit the given melody while honouring the artistic spirit of the reference text.

---

### 1. Melody Constraints
| Parameter             | Value                        |
|-----------------------|------------------------------|
| Total syllables       | **{syllable_count}**         |
| Tempo                 | {tempo} BPM                  |
| Key signature         | {key}                        |
| Strong-beat positions | syllable indices {strong_str}|
| Rhyme positions       | syllable indices {rhyme_str} |

---

### 2. Reference Text & Artistic Concept
{reference_text}

Embedded lyric hints from MIDI:
  source: {embedded_source}
  unit count: {embedded_count}
  syllable-count basis: {effective_count_source}
  preview: {embedded_str}

Jyutping reference (selected reading):
  {selected_jp}

Per-character breakdown (up to 20 chars):
{breakdown_str}

---

### 3. Melody Guidance
Lean 0243 target sequence (one per syllable):
  {tone_seq_str}

Rules:
- Follow 0243.hk default lean 0243 behavior as the primary melody constraint
- Strong beats should prefer stable, singable syllables
- Phrase-final positions should cadence naturally
- Rhyming syllables must share the same Jyutping final vowel/coda

返回前的最终检查：

- 先自行统计草稿音节数。
- 如果总数不等于 **{syllable_count}**，先修改草稿再输出 JSON。
- 不要返回仍然存在音节数不一致的草稿。

---

### 4. Output Format
Return a **JSON object only** – no prose outside the JSON:

```json
{{
  "lyrics":   "<Chinese lyrics text, newlines between lines>",
  "jyutping": "<full Jyutping romanisation, space between syllables>",
  "lines": [
    {{"text": "<line1>", "jyutping": "<line1 jyutping>", "syllable_count": 0}},
    {{"text": "<line2>", "jyutping": "<line2 jyutping>", "syllable_count": 0}}
  ],
  "rhyme_scheme": "<e.g. AABB or ABAB>",
  "notes": "<brief explanation of tonal and rhyme choices>"
}}
```

The total syllable count across all lines **must equal {syllable_count}**.
