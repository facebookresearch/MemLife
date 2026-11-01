# EgoLife-EQA

**E**gocentric **Q**uestions about **R**ecurring **E**vents: 100 multiple-choice questions about
activities that repeat across the seven days of the [EgoLife](https://egolife-ai.github.io/)
recordings. Answering one means aggregating evidence spread over the whole history: on average
3.9 evidence segments across 3.3 distinct days.

This folder holds the **questions and evidence annotations only** (`test.jsonl`). The recordings
come from the EgoLife release (subjects `A1_JAKE` and `A3_TASHA`) and are not redistributed.

| Type | Questions | Options | Avg. correct options | Example |
|---|---|---|---|---|
| Frequency counting | 45 | 8 | 2.33 | *On which days did I go grocery shopping? Choose all that apply.* |
| Routine | 31 | 2–4 | 1 | *What do I usually check on my phone in the morning?* |
| Comparison | 24 | 3 | 1 | *Which activity did I do in the bedroom on more days, X or Y?* |

`A3_TASHA` has 75 questions and `A1_JAKE` 25. Every question's cutoff is the end of the
recording.

## Fields

```jsonc
{
  "question_id":    "A3_TASHA_DAY1-DAY7_compose_alone_20",
  "question":       "On which days did I copy reference material into a notebook by myself? ...",
  "choices":        ["DAY 1", "DAY 2", "DAY 3", "DAY 4", "DAY 5", "DAY 6", "DAY 7", "NONE"],
  "correct_letter": "A,D",        // sorted, comma-joined: a set
  "multi_select":   true,
  "n_gold":         2,
  "type":           "frequency_counting",   // frequency_counting | routine | comparison
  "subject":        "A3_TASHA",
  "activity":       "compose",
  "source":         "compose",
  "answer_value":   "DAY 1, DAY 4",
  "query_abs_s":    585356.873,             // cutoff, seconds on the stream clock
  "evidence": [
    {"day": 4, "time": "17:25:03", "abs_s": 321903.0, "video_id": null,
     "text": "I seem to be copying a tutorial from my phone."}
  ]
}
```

- Times are on the EgoLife stream clock, `(day - 1) * 86400 + seconds_of_day`.
- Evidence is an annotated **instant**; the segment containing it counts as gold. One of the 390
  entries has `abs_s: null` (the day was confirmed but not the time) and is skipped by the loader.

## Evaluation

Accuracy is **exact set match**: a multi-select prediction is correct only if it equals the full
annotated set. Unparseable answers count as wrong and are reported separately.

## Annotation

Candidate recurring events were drawn from the official EgoLife captions and turned into
questions. Two annotators independently verified every candidate evidence segment against its
question's answer (92.1% agreement, disagreements resolved by discussion). Questions without
verified evidence or an unambiguous answer were discarded.

The annotations are released under this repository's [LICENSE](../../LICENSE). The EgoLife
recordings remain subject to the EgoLife project's own terms.
