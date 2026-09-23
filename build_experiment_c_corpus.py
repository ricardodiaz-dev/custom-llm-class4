"""Generate the Experiment C teaching material.

Experiment B produced a specific, testable claim: this model learns patterns whose
answer is fixed by position or by association, and fails at patterns that require
deciding which of two same-category words in context was the corrected one.

Evidence for that claim, measured on B's trained model:
  - `X is inside Y . Y contains the ___`  learned, from only 30 passages
  - `did not buy A . she bought B . bought ___`  not learned, from 59 passages,
    and its output is byte-identical whatever the prompt says -- it ignores context

Experiment C tests the claim by teaching both kinds at once:

  HARD  (discrimination)   negation -- now decorrelated and roughly 4x larger
  EASY  (associative)      opposites, grammar, everyday knowledge -- newly added
  CARRIED OVER             spatial relations, categories and kinds
  CONTROL (untaught)       reference, sequence -- both discrimination tasks

If the easy three land while negation still fails on clean, plentiful data, the
claim holds and the limit is the task, not the corpus.

Two mechanics carried over from the Experiment B generator:
  - multi-clause examples write the period with no following space ("blue.it"),
    because chunk_text splits passages at sentence boundaries
  - no '#' heading line, because heading tokens become training data

Usage: python build_experiment_c_corpus.py
"""
import sys

sys.path.insert(0, ".")
from build_extension_corpus import (OUT, article, spatial_lines, category_lines,
                                    write)
from run_evals import load_suite

# --- HARD: negation, decorrelated -------------------------------------------
# Every subject is crossed with every ordered pair inside its value set, so
# "the box is ___" is followed by each colour equally often. A per-subject prior
# therefore earns nothing, and copying from context is the only way to predict.
VALUE_SETS = [
    (["box", "fence", "sign", "wall", "door", "gate", "window", "roof", "awning", "crate"],
     ["red", "blue", "green", "yellow"]),
    (["door", "gate", "window", "shop", "office"],
     ["open", "closed"]),
    (["parcel", "lot", "unit", "site", "building"],
     ["vacant", "developed", "occupied"]),
    (["permit", "plan", "application", "record", "drawing"],
     ["approved", "denied", "filed", "missing"]),
    (["lot", "alley", "street", "driveway", "corridor"],
     ["wide", "narrow"]),
]
BUYERS = ["ava", "nora", "iris", "june", "maya", "rosa", "elena", "paz"]
GOODS = ["tea", "milk", "bread", "rice"]


def negation_lines():
    lines = []
    for subjects, values in VALUE_SETS:
        for subject in subjects:
            for wrong in values:
                for right in values:
                    if wrong == right:
                        continue
                    lines.append(f"the {subject} is not {wrong}.it is {right}.the {subject} is {right} .")
                    lines.append(f"the {subject} is not {wrong}.it is {right} .")
    for buyer in BUYERS:
        for wrong in GOODS:
            for right in GOODS:
                if wrong == right:
                    continue
                lines.append(f"{buyer} did not buy {wrong}.she bought {right}.{buyer} bought {right} .")
                lines.append(f"{buyer} did not take {wrong}.she took {right}.{buyer} took {right} .")
    return lines


# --- EASY: opposites ---------------------------------------------------------
# The three tested pairs are taught in the REVERSE direction only. "the opposite
# of hot is" is lang_28's prompt verbatim, so the corpus teaches "the opposite of
# cold is hot" and leaves the model to apply the relation symmetrically.
OPPOSITE_PAIRS = [
    ("cold", "hot"), ("full", "empty"), ("quiet", "noisy"),
    ("slow", "fast"), ("light", "heavy"), ("cool", "warm"),
    ("late", "early"), ("hard", "soft"), ("soft", "hard"),
    ("early", "late"), ("fast", "slow"), ("heavy", "light"),
    ("warm", "cool"), ("narrow", "wide"), ("wide", "narrow"),
    ("open", "closed"), ("closed", "open"), ("new", "old"), ("old", "new"),
    ("tall", "short"), ("short", "tall"), ("round", "square"), ("square", "round"),
    ("loud", "faint"), ("faint", "loud"), ("dry", "wet"), ("wet", "dry"),
]


def opposite_lines():
    lines = []
    for a, b in OPPOSITE_PAIRS:
        lines.append(f"the opposite of {a} is {b} .")
        lines.append(f"{a} and {b} are opposites .")
        lines.append(f"when a thing is not {a} it is {b} .")
    return lines


# --- EASY: grammar -----------------------------------------------------------
# "one bird", "the dogs" and "yesterday she" are eval prompts in full, so none of
# them may appear. Singular/plural and past tense are taught on other words.
SINGULARS = ["cat", "dog", "duck", "goat", "horse", "robin", "tree", "lamp", "desk", "parcel"]
PLURALS = [("cats", "cat"), ("dogs", "dog"), ("ducks", "duck"), ("goats", "goat"),
           ("horses", "horse"), ("robins", "robin"), ("trees", "tree"), ("lamps", "lamp"),
           ("desks", "desk"), ("parcels", "parcel")]
VERBS = [("walk", "walks", "walked", "walking"), ("work", "works", "worked", "working"),
         ("wait", "waits", "waited", "waiting"), ("park", "parks", "parked", "parking")]
ACTORS = ["he", "maya", "nora", "the owner", "the tenant"]


def grammar_lines():
    lines = []
    for s in SINGULARS:
        lines.append(f"one {s} is here .")
        lines.append(f"the {s} is here .")
        lines.append(f"a {s} is small .")
    for plural, _ in PLURALS:
        lines.append(f"two {plural} are here .")
        lines.append(f"many {plural} are small .")
        lines.append(f"the {plural} were here .")
    for base, third, past, cont in VERBS:
        for actor in ACTORS:
            lines.append(f"yesterday {actor} {past} .")
            lines.append(f"today {actor} {third} .")
        lines.append(f"they {base} every day .")
        # A word carried by only one or two passages can land entirely in the
        # held-out 10% and never enter the vocabulary, which is built from
        # training text alone. Repeat each form across several actors.
        for actor in ACTORS:
            lines.append(f"{actor} is {cont} now .")
            lines.append(f"{actor} will {base} later .")
        lines.append(f"i am {cont} now .")
    lines.append("i am the owner .")
    lines.append("they were here yesterday .")
    return lines


# --- EASY: everyday knowledge ------------------------------------------------
# Each tested fact is taught with a different subject or setting so that no eval
# prompt is reproduced: "milk freezes into ice", not "water freezes into ice".
def everyday_lines():
    lines = []
    for liquid in ["milk", "juice", "rain", "cream"]:
        lines.append(f"{liquid} freezes into ice .")
        lines.append(f"{liquid} becomes ice when it is cold .")
    lines.append("ice melts into water .")
    lines.append("water becomes ice in winter .")
    lines.append("steam rises from hot water .")
    lines.append("sand is dry and wood is dry .")
    for cover in ["a coat", "a canopy", "an awning", "a roof"]:
        lines.append(f"a person uses {cover} to stay dry .")
        lines.append(f"{cover} keeps a person dry in the rain .")
    lines.append("an umbrella keeps a person dry .")
    lines.append("an umbrella is open in the rain .")
    lines.append("a person is wet without an umbrella .")
    for who in ["a person", "the tenant", "the owner", "a child", "he"]:
        lines.append(f"{who} is hungry before a meal ." if who != "he"
                     else "he is hungry before a meal .")
        lines.append(f"{who} eats bread when hungry .")
    lines.append("a hungry person eats bread .")
    lines.append("a hungry person is not asleep .")
    lines.append("an asleep person rests .")
    lines.append("an asleep person is not hungry .")
    for place in ["hall", "office", "kitchen", "corridor"]:
        lines.append(f"to see in a dark {place} we turn on a light .")
        lines.append(f"the {place} is dark until we turn on a light .")
    lines.append("we turn on a light to see .")
    lines.append("a dark room needs a light .")
    lines.append("a pillow is soft and a spoon is hard .")
    lines.append("a shoe is not a light .")
    return lines


if __name__ == "__main__":
    suite = load_suite()
    OUT.mkdir(exist_ok=True)
    total = 0
    for name, builder in [
        ("negation_and_correction.md", negation_lines),     # HARD
        ("spatial_relations.md", spatial_lines),            # carried over
        ("categories_and_kinds.md", category_lines),        # carried over
        ("opposites.md", opposite_lines),                   # EASY
        ("grammar_number_and_tense.md", grammar_lines),     # EASY
        ("everyday_knowledge.md", everyday_lines),          # EASY
    ]:
        total += len(write(name, builder(), suite, ""))
    print(f"total teaching lines: {total}")
