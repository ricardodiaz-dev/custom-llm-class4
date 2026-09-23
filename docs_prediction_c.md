### My choices and prediction — Experiment C (the hypothesis test)

Same 3,000 steps and same 0.001 learning rate as A and B. The corpus is still the only
variable across all three runs.

**Why this run exists.** Experiment B produced a specific claim, and I want to test it
rather than assert it. Probing B's trained model showed that on the `did not buy` frame
its output is *identical to three decimal places* no matter which buyer or which goods
the prompt names — it ignores the context entirely and recites one fixed ranking of
groceries. Meanwhile the `contains` frame was learned correctly from only 30 passages,
half the data the buy frame got.

So the difference is not volume and not correlation. It is what the task asks for:

- **Associative / positional** — `X is inside Y . Y contains the ___`. Two nouns in
  context, one of them sitting immediately before the blank. "Use the other noun" is
  structurally unambiguous. **Learned.**
- **Discrimination** — `did not buy A . she bought B . bought ___`. Both candidates are
  groceries, both sit in context at similar distance, and the model must work out which
  one followed *"she bought"* rather than *"did not buy"*. **Not learned.**

**My claim: this architecture learns the first kind and fails at the second, and no
reasonable amount of clean data fixes the second.**

**How Experiment C tests it.** I teach both kinds at once and see which take:

| Role | Categories | Teaching material |
|---|---|---|
| HARD — discrimination | `negation` | decorrelated and ~4x larger (59 → 649 lines) |
| EASY — associative | `opposites`, `grammar`, `everyday_knowledge` | newly added |
| Carried over | `spatial_relations`, `categories_and_analogies` | unchanged from B |
| CONTROL — untaught | `reference`, `sequence` | nothing, and both are discrimination tasks |

**What I predict.**

1. Scorable cases rise from 33 to **42**. The six control cases stay
   `out_of_vocabulary` at zero.
2. The three EASY categories score **well — 7 to 9 of 9**. They are bigram-shaped: the
   answer follows from the words immediately before it.
3. **Negation still fails, at roughly 0–1 of 3**, despite four times the data and full
   decorrelation. This is the prediction that matters. If negation jumps to 3/3, my claim
   is wrong and the real problem was my Experiment B data all along.
4. Negation's *probabilities* change shape even if its score does not. Decorrelation
   removes the per-subject prior, so `lang_31` should move from green 0.414 / blue 0.062
   toward roughly uniform — about 0.25 on each colour. That would mean the model went
   from confidently wrong to honestly guessing, which is a real change even at the same
   score.
5. `lang_46` (`a salmon is a ___`) **still fails**. I did not touch the category
   frequencies, so `bird` remains the most common class in the teaching data and should
   still win the slot.
6. Training and validation loss both rise again, because the vocabulary grows from 320 to
   about 407. Losses across different corpora are not comparable as a ranking.

**The honest caveat.** I designed this run after inspecting eval results from the previous
run. That makes this a development benchmark, not a test of unseen generalization. I am
running it once and reporting what happens either way, rather than iterating until the
number looks good.
