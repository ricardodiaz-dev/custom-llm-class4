"""Cosine nearest neighbours in the full 64-dimension embedding space.

The viewer's map is a PCA projection down to 3 dimensions; these numbers use the
whole vector, which is why the two can disagree.
"""
import json
import math
import sys

path, words = sys.argv[1], sys.argv[2:]
checkpoint = json.loads(open(path).read())
vocabulary = checkpoint["vocabulary"]
final = checkpoint["weights"][next(iter(checkpoint["weights"]))]
initial = checkpoint["initial_embeddings"]


def neighbours(table, index, k=6):
    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1e-9
        nb = math.sqrt(sum(y * y for y in b)) or 1e-9
        return dot / (na * nb)
    scores = [(cosine(table[index], table[j]), vocabulary[j])
              for j in range(len(vocabulary)) if j != index]
    return sorted(scores, reverse=True)[:k]


for word in words:
    if word not in vocabulary:
        print(f"{word}: not in this run's vocabulary")
        continue
    i = vocabulary.index(word)
    fmt = lambda pairs: ", ".join(f"{w} ({s:.3f})" for s, w in pairs)
    print(f"\n`{word}` (ID {i})")
    print(f"  before training: {fmt(neighbours(initial, i))}")
    print(f"  after training : {fmt(neighbours(final, i))}")
