"""Replay the construction cells of ``sgwb_data_generation.ipynb`` headlessly.

The notebook stays the source of truth for the archive; this runner only
executes its code cells up to and including the build cell, so a rebuild can
run on a cluster or under pytest without a Jupyter kernel.  The diagnostic
cells that follow the build are skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent / "sgwb_data_generation.ipynb"
BUILD_CELL_MARKER = "if RUN_FULL_GENERATION:"


def load_generation_namespace(run_full_generation: bool = False) -> dict:
    """Execute the notebook's construction cells and return their namespace.

    With ``run_full_generation`` the build cell writes the archive; otherwise
    the cells only define the model and synthesis functions, which is what the
    tests need.
    """
    os.environ.setdefault("MPLBACKEND", "Agg")
    cells = json.loads(NOTEBOOK_PATH.read_text())["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    last = next(
        index
        for index, cell in enumerate(code_cells)
        if BUILD_CELL_MARKER in "".join(cell["source"])
    )
    namespace: dict = {"__name__": "sgwb_data_generation"}
    for index, cell in enumerate(code_cells[: last + 1]):
        source = "".join(cell["source"])
        if run_full_generation:
            source = source.replace("RUN_FULL_GENERATION = False", "RUN_FULL_GENERATION = True")
        exec(compile(source, f"{NOTEBOOK_PATH.name}:code[{index}]", "exec"), namespace)
    return namespace


if __name__ == "__main__":
    load_generation_namespace(run_full_generation=True)
