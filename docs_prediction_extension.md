### My choices and prediction — Experiment B (corpus extension)

Same 3,000 steps and same 0.001 learning rate as Experiment A, so the **only** thing
that changes between the two runs is the training data. Holding the settings fixed is
what makes the comparison interpretable.

**Corpus:** the classroom sentences plus teaching material I generated with
`build_extension_corpus.py`, targeting three of the eight eval extension categories:
**negation**, **spatial relations**, and **categories and analogies**. The examples are
written in land-use language where the pattern allows it — parcels, permits, setbacks,
districts — paired with everyday instantiations so the fixed eval vocabulary is actually
covered. I deliberately left **grammar, opposites, reference, sequence and everyday
knowledge** untaught so they act as a control group.

**What I expect.** Scorable cases should rise from 24 to 33: the nine cases in my three
taught categories gain vocabulary coverage, and the fifteen untaught cases stay
`out_of_vocabulary` at zero. If the untaught categories move at all, my attribution is
wrong and I should say so.

Within the taught categories I expect the pattern cases to do better than the knowledge
cases. Negation and spatial relations are copy-from-context patterns — the answer is a
word that already appeared a few tokens earlier, which is something attention can
learn. Categories and analogies requires recalling a specific thing-to-class fact, which
is memorization from a smaller number of examples.

I expect a **higher** training and validation loss than Experiment A, not a lower one,
because the corpus is now more varied: the vocabulary roughly doubles from 133 to about
281 types and the passages are less uniform. Losses across different corpora are not
comparable as a quality ranking, and I should not read a higher number as a worse model.

The risk I see: the teaching material is only about 7% of the training passages, so the
patterns may be too rare to overcome the classroom frames that dominate the data.
