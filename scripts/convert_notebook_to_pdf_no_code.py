"""
Convert a Jupyter notebook to PDF with no code input (only markdown and outputs).
Run from project root: python scripts/convert_notebook_to_pdf_no_code.py

Requires: nbformat, nbconvert, pandoc (for PDF), and a LaTeX distribution (e.g. MiKTeX).
If PDF fails, the script also creates an HTML version (--no-input equivalent).
"""
import json
import subprocess
import sys
from pathlib import Path

NOTEBOOK = Path("reports/Churn_Prediction_Final_Reportv2 (2).ipynb")
OUTPUT_PDF = Path("reports/Churn_Prediction_Final_Reportv2.pdf")
OUTPUT_HTML = Path("reports/Churn_Prediction_Final_Reportv2.html")


def main():
    root = Path(__file__).resolve().parent.parent
    nb_path = root / NOTEBOOK
    if not nb_path.exists():
        print(f"Notebook not found: {nb_path}")
        sys.exit(1)

    # Load notebook
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # Remove code input from all code cells (keep outputs)
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["source"] = []

    # Save no-code version
    no_code_path = nb_path.parent / (nb_path.stem + "_no_code.ipynb")
    with open(no_code_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Saved no-code notebook: {no_code_path}")

    # Try nbconvert to PDF
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "pdf",
                "--no-input",
                str(no_code_path),
                "--output", str(OUTPUT_PDF.stem),
                "--output-dir", str(OUTPUT_PDF.parent),
            ],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=120,
        )
        if result.returncode == 0:
            print(f"PDF saved: {OUTPUT_PDF.parent / (OUTPUT_PDF.stem + '.pdf')}")
        else:
            print("PDF conversion failed. Trying HTML instead...")
            print(result.stderr or result.stdout)
            _convert_to_html(no_code_path, root)
    except FileNotFoundError:
        print("jupyter nbconvert not found. Creating HTML instead...")
        _convert_to_html(no_code_path, root)
    except subprocess.TimeoutExpired:
        print("Conversion timed out.")
        sys.exit(1)


def _convert_to_html(no_code_path: Path, root: Path):
    """Fallback: convert to HTML (no LaTeX needed)."""
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "html",
                "--no-input",
                str(no_code_path),
                "--output", str(OUTPUT_HTML.stem),
                "--output-dir", str(OUTPUT_HTML.parent),
            ],
            capture_output=True,
            text=True,
            cwd=str(root),
            timeout=60,
        )
        if result.returncode == 0:
            out = OUTPUT_HTML.parent / (OUTPUT_HTML.stem + ".html")
            print(f"HTML saved: {out}")
            print("Open in browser and use Print > Save as PDF for a PDF.")
        else:
            print(result.stderr or result.stdout)
    except Exception as e:
        print(f"Error: {e}")
        print("\nManual steps:")
        print(f"1. Open {no_code_path} in Jupyter")
        print("2. File > Download as > PDF via LaTeX (or HTML, then Print > Save as PDF)")


if __name__ == "__main__":
    main()
