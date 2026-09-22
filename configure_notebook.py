"""Produce a configured copy of custom_llm.ipynb for one experiment.

Patches the three graded choices in section 1, replaces the prediction markdown
with the prediction written before that run, and expands section 10 into several
chat turns so the executed notebook shows more than one interaction.

Usage: python configure_notebook.py <output.ipynb> <steps> <lr> <prediction.md> <prompt> [prompt ...]
"""
import re
import sys
import nbformat

out, steps, lr, prediction_path, *prompts = sys.argv[1:]
nb = nbformat.read("custom_llm.ipynb", as_version=4)
prediction = open(prediction_path).read().strip()

patched_config = patched_prediction = chat_index = None
for index, cell in enumerate(nb.cells):
    source = cell.source
    if cell.cell_type == "code" and source.lstrip().startswith("CORPUS = "):
        source = re.sub(r"^TRAINING_STEPS = \d+", f"TRAINING_STEPS = {steps}", source, flags=re.M)
        source = re.sub(r"^LEARNING_RATE = [\d.e-]+", f"LEARNING_RATE = {lr}", source, flags=re.M)
        cell.source, patched_config = source, index
    elif cell.cell_type == "markdown" and "Replace this text with your choices" in source:
        cell.source, patched_prediction = prediction, index
    elif cell.cell_type == "code" and source.lstrip().startswith("CHAT_PROMPT = "):
        chat_index = index

if patched_config is None or patched_prediction is None or chat_index is None:
    raise SystemExit(f"could not locate cells: config={patched_config} "
                     f"prediction={patched_prediction} chat={chat_index}")

template = nb.cells[chat_index].source
nb.cells[chat_index].source = re.sub(r'^CHAT_PROMPT = ".*"',
                                     f'CHAT_PROMPT = "{prompts[0]}"', template, flags=re.M)
for offset, prompt in enumerate(prompts[1:], start=1):
    turn = nbformat.v4.new_code_cell(
        re.sub(r'^CHAT_PROMPT = ".*"', f'CHAT_PROMPT = "{prompt}"', template, flags=re.M))
    nb.cells.insert(chat_index + offset, turn)

nbformat.write(nb, out)
print(f"wrote {out}: steps={steps} lr={lr} chat_turns={len(prompts)} cells={len(nb.cells)}")
