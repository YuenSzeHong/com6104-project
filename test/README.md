# Test Fixtures

This directory stores fixture material for manual and automated evaluation.

## Layout

- `midi/`: source MIDI files used as melody constraints
- `lyrics/raw/`: raw lyric page exports or scraped source material
- `lyrics/*.clean.txt`: HTML noise removed, lyric lines only
- `lyrics/*.kanji-yomi.txt`: cleaned Japanese lyric lines in `kanji[yomi]` format

`clean` and `kanji-yomi` do not replace MIDI analysis.

- `clean` is for stripping the original lyric page down to plain lyric lines.
- `kanji-yomi` is for keeping lyric lines together with ruby-derived reading hints.
- Melody, timing, and syllable constraints still come from the MIDI file.

## Case Categories

The project should not evaluate all songs in the same way. Some songs are highly memorable and may already exist in model training data together with well-known Cantonese adaptations.

### High-Memorization Cases

These are useful for demos, regression checks, and comparison against known adaptation strategies, but not as the only proof of true adaptation ability.

#### `X54896G2` — `それが大事`

- `midi/X54896G2.MID`
- `lyrics/raw/X54896G2.txt`
- `lyrics/X54896G2.clean.txt`
- `lyrics/X54896G2.kanji-yomi.txt`

This case is strongly associated with the Cantonese adaptation `紅日`, so it should be treated as a high-memorization comparison case.

#### `ドラえもんのうた`

- `midi/ドラえもんのうた.mid`
- `lyrics/raw/ドラえもんのうた.txt`
- `lyrics/ドラえもんのうた.clean.txt`
- `lyrics/ドラえもんのうた.kanji-yomi.txt`

This is also a high-memorization case because the source song is extremely well known and easy for a model to recall or overfit.

### General Adaptation Fixtures

These can still be useful adaptation samples, but they are not currently annotated as known high-risk comparison cases.

#### `R00317G2`

- `midi/R00317G2.MID`
- `lyrics/raw/R00317G2.txt`
- `lyrics/R00317G2.clean.txt`
- `lyrics/R00317G2.kanji-yomi.txt`

## Evaluation Use

These fixtures are for a Cantonese lyric adaptation workflow, not for exact-answer lyric matching.

- Use the source-language lyric as semantic and structural input.
- Use `clean` when you only need lyric text with HTML noise removed.
- Use `kanji-yomi` when you also want the source-language reading hints preserved.
- Use the MIDI as the singing constraint.
- Use known Cantonese adaptations as a reference for comparison, not as the only acceptable output.

For high-memorization cases, evaluate:

- melodic fit
- line structure
- rhyme handling
- adaptation strategy
- whether the output collapses into a memorized published version

Do not score those cases only by literal similarity to an existing Cantonese release.
