"""Export collaborator-facing PDFs from the three analysis notebooks.

The source notebooks remain executable.  Code inputs are removed only from a
temporary export copy; existing cell outputs and reader-facing markdown are
retained.  Pandoc plus XeLaTeX are used because they are available in the
workspace even when nbconvert is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf"
NOTEBOOKS = {
    "sgwb_data_generation.ipynb": "lisa_data_generation.pdf",
    "pspline_univar_surface_fit.ipynb": "scalar_x2_m0_m1_inference.pdf",
    "pspline_aet_diagonal_fit.ipynb": "diagonal_aet_component_inference.pdf",
}


def report_copy(notebook: dict) -> dict:
    """Return a code-hidden notebook copy for PDF conversion."""
    output = dict(notebook)
    report_cells = []
    for original in notebook["cells"]:
        cell = dict(original)
        if cell["cell_type"] == "code":
            # Keep figures but omit source code and raw diagnostic streams.
            # Key scalar diagnostics are stated explicitly in notebook prose.
            image_outputs = []
            for raw_output in cell.get("outputs", []):
                data = raw_output.get("data", {})
                image_types = {
                    key: value
                    for key, value in data.items()
                    if key in {"image/png", "image/svg+xml", "application/pdf"}
                }
                if not image_types:
                    continue
                cleaned_output = dict(raw_output)
                cleaned_output["data"] = image_types
                image_outputs.append(cleaned_output)
            if not image_outputs:
                continue
            cell["source"] = []
            cell["execution_count"] = None
            cell["outputs"] = image_outputs
        report_cells.append(cell)
    output["cells"] = report_cells
    return output


def run() -> list[Path]:
    pandoc = shutil.which("pandoc")
    xelatex = shutil.which("xelatex")
    if pandoc is None or xelatex is None:
        raise RuntimeError("PDF export requires pandoc and xelatex on PATH")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generated = []
    with tempfile.TemporaryDirectory(prefix="collaborator-notebooks-") as directory:
        temporary = Path(directory)
        header = temporary / "notebook_header.tex"
        header.write_text(
            r"""
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{caption}
\usepackage{graphicx}
\setkeys{Gin}{width=\linewidth,height=0.68\textheight,keepaspectratio}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em}
\renewcommand{\arraystretch}{1.12}
""".strip()
            + "\n"
        )
        for source_name, output_name in NOTEBOOKS.items():
            source = ROOT / source_name
            notebook = json.loads(source.read_text())
            stripped = temporary / source_name
            stripped.write_text(json.dumps(report_copy(notebook), indent=1) + "\n")
            destination = OUTPUT_DIR / output_name
            command = [
                pandoc,
                str(stripped),
                "--from=ipynb",
                "--to=pdf",
                f"--pdf-engine={xelatex}",
                f"--include-in-header={header}",
                f"--resource-path={ROOT}:{PROJECT_ROOT}",
                "--toc",
                "--toc-depth=2",
                "--variable=geometry:margin=0.7in",
                "--variable=papersize=a4",
                "--variable=fontsize=10pt",
                "--variable=colorlinks=true",
                "--output",
                str(destination),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            generated.append(destination)
    return generated


if __name__ == "__main__":
    for path in run():
        print(path)
