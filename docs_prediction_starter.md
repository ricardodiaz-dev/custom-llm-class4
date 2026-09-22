### My choices and prediction — Experiment A (starter corpus)

**Corpus:** the supplied synthetic classroom sentences only, with an empty `corpus/`.
This is the control run. I want a clean baseline before I change anything, so that any
later movement can be attributed to the data I add rather than to a settings change.

**Training steps: 3,000.** The assignment's suggested budget. One step updates weights
from 32 passages, so 3,000 steps is about 96,000 passage samples over roughly 4,900
unique documents — each document is seen on the order of twenty times.

**Learning rate: 0.001**, with the notebook's warmup and cosine decay. Too large an
update overshoots the minimum and the loss diverges or oscillates; too small a one
crawls and 3,000 steps would end far from anything useful. 0.001 is the standard
starting point for AdamW on a model this size.

**What I expect.** Training loss should start near 6.2 — the natural log of the
vocabulary size, which is what random guessing costs — and fall steeply, because the
classroom corpus is eight domains crossed with eight sentence frames and is therefore
extremely repetitive. Validation loss should track training loss closely rather than
diverging, because held-out passages come from the same templates; that closeness will
measure template memorization, not generalization.

On the evals I predict exactly 24 of 48 cases are scorable and 24 are
`out_of_vocabulary`. The 24 extension cases use words the classroom corpus never
contains, so the model scores zero on them for vocabulary reasons, not reasoning
reasons. I expect the untrained model near chance (about 25% of scorable cases) and
the trained model clearly above chance on `starter_patterns`.

Samples should move from a random word salad to template-shaped sentences that copy the
corpus frames. For the embedding inspection I will look at `customer`; I expect its
nearest neighbors after training to be the other five nouns that share its contexts.
