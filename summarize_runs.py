"""Emit the README tables from the saved run artifacts.

Usage: python summarize_runs.py <starter_run_dir> <extension_run_dir>

Everything printed here is read from the saved JSON. Nothing is recomputed or
rounded by hand, so the README and the evidence files cannot drift apart.
"""
import json
import sys
from pathlib import Path

A, B = Path(sys.argv[1]), Path(sys.argv[2])
load = lambda d, name: json.loads((d / name).read_text())


def facts():
    print("## Run facts\n")
    ca, cb = load(A, "config.json"), load(B, "config.json")
    sa, sb = load(A, "training_summary.json"), load(B, "training_summary.json")
    va, vb = load(A, "vocabulary_report.json"), load(B, "vocabulary_report.json")
    rows = [
        ("Corpus mode", ca["corpus_mode"], cb["corpus_mode"]),
        ("Corpus files", ca["corpus_files"], cb["corpus_files"]),
        ("Unique passages", f'{ca["train_documents"] + ca["validation_documents"]:,}',
         f'{cb["train_documents"] + cb["validation_documents"]:,}'),
        ("Train / validation", f'{ca["train_documents"]:,} / {ca["validation_documents"]:,}',
         f'{cb["train_documents"]:,} / {cb["validation_documents"]:,}'),
        ("Reserved eval passages", ca["reserved_eval_passages"], cb["reserved_eval_passages"]),
        ("Distinct training types", va["training_types"], vb["training_types"]),
        ("Vocabulary size", ca["vocabulary_size"], cb["vocabulary_size"]),
        ("Training UNK rate", f'{ca["training_unknown_rate"]:.2%}', f'{cb["training_unknown_rate"]:.2%}'),
        ("Held-out UNK rate", f'{ca["validation_unknown_rate"]:.2%}', f'{cb["validation_unknown_rate"]:.2%}'),
        ("Parameters", f'{ca["parameters"]:,}', f'{cb["parameters"]:,}'),
        ("Training steps", ca["training_steps"], cb["training_steps"]),
        ("Learning rate", ca["learning_rate"], cb["learning_rate"]),
        ("Completed steps", sa["completed_steps"], sb["completed_steps"]),
        ("Interrupted", sa["interrupted"], sb["interrupted"]),
        ("Elapsed (s)", f'{sa["elapsed_seconds"]:.1f}', f'{sb["elapsed_seconds"]:.1f}'),
        ("Device", ca["device"], cb["device"]),
        ("Hardware", ca["hardware"], cb["hardware"]),
        ("Eval panel size", ca["evaluation_panel_size"], cb["evaluation_panel_size"]),
    ]
    print("| | A — starter | B — extension |\n|---|---|---|")
    for label, a, b in rows:
        print(f"| {label} | {a} | {b} |")
    print()


def losses():
    for label, d in [("A — starter", A), ("B — extension", B)]:
        print(f"## Loss panels — {label}\n")
        print("| Step | Training loss | Validation loss |\n|---:|---:|---:|")
        for row in load(d, "history.json"):
            print(f'| {row["step"]} | {row["training_loss"]:.4f} | {row["validation_loss"]:.4f} |')
        print()


def evals():
    print("## Eval comparison — all four result sets\n")
    print("| Experiment | Stage | Correct /48 | All-case success | Scorable | Accuracy on scorable | Coverage |")
    print("|---|---|---:|---:|---:|---:|---:|")
    sets = {}
    for label, d in [("A — starter", A), ("B — extension", B)]:
        comparison = load(d, "language_eval_comparison.json")
        for stage in ("untrained", "final"):
            o = comparison[stage]["overall"]
            sets[(label, stage)] = comparison[stage]
            acc = "n/a" if o["accuracy_scorable_cases"] is None else f'{o["accuracy_scorable_cases"]:.1%}'
            print(f'| {label} | {stage} | {o["correct"]} | {o["success_rate_all_cases"]:.1%} | '
                  f'{o["scorable"]} | {acc} | {o["coverage"]:.0%} |')
    print()

    taught = {"negation", "spatial_relations", "categories_and_analogies"}
    print("## Eval by category — trained models, taught vs. control\n")
    print("| Category | Taught? | A correct | A scorable | B correct | B scorable |")
    print("|---|---|---:|---:|---:|---:|")
    a_cat = sets[("A — starter", "final")]["by_category"]
    b_cat = sets[("B — extension", "final")]["by_category"]
    for category in sorted(set(a_cat) | set(b_cat)):
        a, b = a_cat.get(category, {}), b_cat.get(category, {})
        mark = "**yes**" if category in taught else "control"
        print(f'| `{category}` | {mark} | {a.get("correct", 0)}/{a.get("total", 0)} | {a.get("scorable", 0)} '
              f'| {b.get("correct", 0)}/{b.get("total", 0)} | {b.get("scorable", 0)} |')
    print()


def inspection():
    for label, d in [("A — starter", A), ("B — extension", B)]:
        i = load(d, "inspection.json")
        vocabulary = load(d, "tokenization.json")["vocabulary"]
        print(f"## Token inspection — {label}\n")
        print(f'- Token `{i["token"]}` has ID **{i["token_id"]}** (row {i["token_id"]} of the embedding table)')
        print(f'- Embedding before (first 8 of 64): `{[round(v, 4) for v in i["embedding_before"][:8]]}`')
        print(f'- Embedding after  (first 8 of 64): `{[round(v, 4) for v in i["embedding_after"][:8]]}`')
        moved = max(abs(a - b) for a, b in zip(i["embedding_after"], i["embedding_before"]))
        print(f'- Largest single-coordinate change across the 64 numbers: **{moved:.4f}**')
        u = i["first_update"]
        print(f'- First saved update — token `{u["token"]}`, coordinate {u["coordinate"]}: '
              f'value `{u["before"]:.8f}`, gradient `{u["gradient"]:.8f}`, '
              f'learning rate `{u["learning_rate"]}` → new value `{u["after"]:.8f}` '
              f'(moved {u["after"] - u["before"]:+.2e})')
        before, after = i["probabilities_before"], i["probabilities_after"]
        top = sorted(range(len(after)), key=lambda k: after[k], reverse=True)[:6]
        print(f'\n  Next-token probabilities after the prefix `{i["prefix"]}` '
              f'(top 6 after training):\n')
        print("| Next token | Before training | After training |\n|---|---:|---:|")
        for k in top:
            print(f"| `{vocabulary[k]}` | {before[k]:.5f} | {after[k]:.5f} |")
        rows = i["attention_rows"]
        print(f'\n  Attention (first head, first block) over the first {len(rows)} positions — '
              f'each row sums to 1 and has zeros to the right, which is the causal mask:\n')
        print("```")
        for r in rows:
            print("  [" + ", ".join(f"{v:.3f}" for v in r) + "]")
        print("```\n")


for section in (facts, losses, evals, inspection):
    section()
