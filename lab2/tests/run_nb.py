"""Execute a notebook headlessly and print its stream output and errors.

The notebook reads its configuration from LAB2_* environment variables, so set them before
calling this script. The executed copy (with outputs) is written to <out.ipynb>, even on failure.

Usage: python run_nb.py <in.ipynb> <out.ipynb> [kernel_name]
"""
import sys

import nbformat
from nbclient import NotebookClient

src, dst = sys.argv[1], sys.argv[2]
kernel = sys.argv[3] if len(sys.argv) > 3 else "lab2test"
nb = nbformat.read(src, as_version=4)
failed = False
try:
    NotebookClient(nb, kernel_name=kernel, timeout=1800).execute()
except Exception:
    failed = True
finally:
    nbformat.write(nb, dst)
for c in nb.cells:
    if c.cell_type == "code":
        for o in c.get("outputs", []):
            if o.output_type == "stream":
                print(o.text, end="")
            elif o.output_type == "error":
                print("ERROR", o.ename, o.evalue)
sys.exit(1 if failed else 0)
