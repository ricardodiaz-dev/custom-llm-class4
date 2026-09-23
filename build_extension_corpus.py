"""Generate the corpus-extension teaching material.

Targets exactly three of the eight eval extension categories -- negation,
spatial_relations and categories_and_analogies -- so that the five untaught
categories act as a control group. Examples are written in land-use language
where the pattern allows it, paired with everyday instantiations so the fixed
eval vocabulary is actually covered.

Two mechanics matter and are easy to get wrong:

1. chunk_text() splits passages at sentence boundaries, so "a . b" becomes two
   passages. Multi-clause patterns therefore write the period without a
   following space ("blue.it"), which keeps one passage and tokenizes the same.
2. reject_eval_leakage() runs on whole-file text, so two individually safe lines
   can spell out an eval prompt when adjacent. Lines are shuffled and the whole
   file is verified before writing.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")
from run_evals import load_suite, matching_cases

OUT = Path("corpus")
SEED = 7

# (subject, wrong, right) -- land-use first, everyday second so the eval words appear.
NEGATION_PAIRS = [
    ("parcel", "vacant", "developed"), ("parcel", "developed", "vacant"),
    ("lot", "narrow", "wide"), ("lot", "wide", "narrow"),
    ("permit", "missing", "filed"), ("permit", "denied", "approved"),
    ("permit", "approved", "denied"), ("plan", "missing", "filed"),
    ("gate", "open", "closed"), ("gate", "closed", "open"),
    ("door", "closed", "open"), ("window", "open", "closed"),
    ("fence", "red", "green"), ("fence", "green", "red"),
    ("sign", "yellow", "blue"), ("sign", "blue", "yellow"),
    ("box", "blue", "red"), ("box", "green", "yellow"),
    ("box", "yellow", "green"), ("roof", "red", "yellow"),
    ("wall", "blue", "green"), ("unit", "vacant", "occupied"),
    ("unit", "occupied", "vacant"), ("building", "new", "old"),
]
BUYERS = ["ava", "nora", "iris", "june", "maya"]
GOODS = ["tea", "milk", "bread", "rice"]

SPATIAL_CONTAINS = [
    ("unit", "building"), ("parcel", "district"), ("garage", "lot"),
    ("book", "drawer"), ("file", "bag"), ("lamp", "carton"),
    ("map", "folder"), ("plan", "binder"), ("ball", "crate"),
    ("desk", "office"), ("lamp", "box"), ("book", "shelf"),
    ("permit", "record"), ("tenant", "unit"), ("shop", "arcade"),
]
SPATIAL_ABOVE = [
    ("roof", "floor"), ("balcony", "sidewalk"), ("sign", "entrance"),
    ("lamp", "shelf"), ("shelf", "desk"), ("ceiling", "room"),
    ("cornice", "window"), ("canopy", "door"), ("floor", "garage"),
    ("awning", "shopfront"), ("parapet", "roof"),
]
SPATIAL_SIDE = [
    ("garage", "entrance"), ("ball", "door"), ("desk", "window"),
    ("driveway", "porch"), ("bag", "shelf"),
    ("box", "crate"), ("lamp", "desk"), ("stair", "elevator"),
    ("kitchen", "dining"),
]
SPATIAL_COMPASS = [
    ("parcel", "street"), ("lot", "alley"), ("building", "park"),
    ("block", "avenue"), ("site", "creek"), ("tower", "plaza"),
]

KINDS = [
    ("robin", "bird"), ("duck", "bird"), ("sparrow", "bird"),
    ("salmon", "fish"), ("trout", "fish"),
    ("goat", "animal"), ("horse", "animal"), ("cat", "animal"), ("dog", "animal"),
    ("maple", "tree"), ("willow", "tree"),
    ("hammer", "tool"), ("wrench", "tool"),
    ("carrot", "vegetable"), ("onion", "vegetable"),
    ("bus", "vehicle"), ("truck", "vehicle"),
    ("duplex", "building"), ("cottage", "building"),
    ("finch", "bird"), ("heron", "bird"), ("cod", "fish"), ("bass", "fish"),
    ("sheep", "animal"), ("mare", "animal"), ("cedar", "tree"), ("elm", "tree"),
    ("chisel", "tool"), ("drill", "tool"), ("turnip", "vegetable"), ("pea", "vegetable"),
    ("tram", "vehicle"), ("van", "vehicle"), ("tower", "building"), ("shed", "building"),
]
KINDS_AN = [("apple", "fruit"), ("onion", "vegetable"), ("acre", "measure")]
GROWS = [("puppy", "dog"), ("kitten", "cat"), ("seedling", "tree"), ("calf", "cow"),
         ("chick", "bird"), ("lamb", "sheep"), ("foal", "horse"), ("sapling", "tree")]
MATERIALS = [("steel", "metal"), ("copper", "metal"), ("cotton", "fabric"), ("linen", "fabric")]


def negation_lines():
    lines = []
    for subject, wrong, right in NEGATION_PAIRS:
        lines.append(f"the {subject} is not {wrong}.it is {right}.the {subject} is {right} .")
        lines.append(f"the {subject} is not {wrong}.it is {right} .")
        lines.append(f"the {subject} is {right}.the {subject} is not {wrong} .")
    for buyer in BUYERS:
        for wrong in GOODS:
            for right in GOODS:
                if wrong == right:
                    continue
                lines.append(f"{buyer} did not buy {wrong}.she bought {right}.{buyer} bought {right} .")
                lines.append(f"{buyer} did not buy {wrong}.she bought {right} .")
    return lines


def spatial_lines():
    lines = []
    for inner, outer in SPATIAL_CONTAINS:
        lines.append(f"the {inner} is inside the {outer}.the {outer} contains the {inner} .")
        lines.append(f"the {outer} contains the {inner}.the {inner} is inside the {outer} .")
        lines.append(f"the {inner} is inside the {outer} .")
    for high, low in SPATIAL_ABOVE:
        lines.append(f"the {high} is above the {low}.the {low} is below the {high} .")
        lines.append(f"the {low} is below the {high}.the {high} is above the {low} .")
    for first, second in SPATIAL_SIDE:
        lines.append(f"the {first} is left of the {second}.the {second} is to the right of the {first} .")
        lines.append(f"the {second} is right of the {first}.the {first} is to the left of the {second} .")
        lines.append(f"the {first} is beside the {second} .")
    for near, far in SPATIAL_COMPASS:
        lines.append(f"the {near} is north of the {far}.the {far} is to the south of the {near} .")
        lines.append(f"the {far} is south of the {near}.the {near} is to the north of the {far} .")
    return lines


def article(word):
    """'a' or 'an' -- the generated corpus is the model's only grammar lesson."""
    return "an" if word[0] in "aeiou" else "a"


def category_lines():
    lines = []
    for thing, kind in KINDS + KINDS_AN:
        lines.append(f"{article(thing)} {thing} is {article(kind)} {kind} .")
        lines.append(f"the {thing} is {article(kind)} {kind} .")
        lines.append(f"every {thing} is {article(kind)} {kind} .")
    for young, grown in GROWS:
        lines.append(f"{article(young)} {young} grows into {article(grown)} {grown} .")
        lines.append(f"the {young} grows into the {grown} .")
    for material, kind in MATERIALS:
        lines.append(f"{material} is a {kind} .")
        lines.append(f"the {material} beam is a {kind} part .")
    return lines


def write(name, lines, suite, header):
    """Drop any line that reproduces a test item, then shuffle until the whole
    assembled file is clean too: adjacent safe lines can still spell out a prompt."""
    lines = sorted(set(lines))
    kept, dropped = [], []
    for line in lines:
        (dropped if matching_cases(line, suite) else kept).append(line)
    if dropped:
        print(f"{name}: dropped {len(dropped)} line(s) that reproduced a test item:")
        for line in dropped:
            print(f"    {line}  -> {matching_cases(line, suite)}")
    lines = kept
    rng = random.Random(SEED)
    for attempt in range(200):
        rng.shuffle(lines)
        text = (header + "\n" if header else "") + "\n".join(lines) + "\n"
        hits = matching_cases(text, suite)
        if not hits:
            (OUT / name).write_text(text, encoding="utf-8")
            print(f"{name}: {len(lines)} lines, clean after {attempt + 1} shuffle(s)")
            return lines
    raise SystemExit(f"{name}: could not avoid eval prompts {hits}")


if __name__ == "__main__":
    suite = load_suite()
    OUT.mkdir(exist_ok=True)
    total = 0
    for name, builder, header in [
        ("negation_and_correction.md", negation_lines,
         "# Negation and correction: a wrong value is stated, then corrected."),
        ("spatial_relations.md", spatial_lines,
         "# Spatial relations: containment, above/below, left/right and compass directions."),
        ("categories_and_kinds.md", category_lines,
         "# Categories and kinds: a specific thing belongs to a general class."),
    ]:
        total += len(write(name, builder(), suite, header))
    print(f"total teaching lines: {total}")
