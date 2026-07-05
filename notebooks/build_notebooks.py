"""Convert the percent-format .py sources to executed .ipynb notebooks.

Usage: python3 build_notebooks.py [name ...]
Run from the notebooks/ directory. Not part of the assignment deliverables,
just tooling so the committed notebooks always carry fresh outputs.
"""

import sys
import jupytext
from nbclient import NotebookClient

NAMES = ["part1_em_gaussian_mixture", "part2_bayesian_sentiment",
         "part4_gradient_descent"]

for name in (sys.argv[1:] or NAMES):
    nb = jupytext.read(f"{name}.py")
    print(f"executing {name} ...", flush=True)
    NotebookClient(nb, timeout=1800, kernel_name="python3").execute()
    jupytext.write(nb, f"{name}.ipynb")
    print(f"wrote {name}.ipynb", flush=True)
