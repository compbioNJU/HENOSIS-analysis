#!/usr/bin/env python3
"""Execute code cells in order without modifying the notebook or saving outputs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    path = args.notebook.resolve()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    scope = {"__name__": "__main__", "__file__": str(path)}
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if source.strip():
            print(f"{path.name}: cell {index + 1}", flush=True)
            exec(compile(source, f"{path}:cell{index + 1}", "exec"), scope)


if __name__ == "__main__":
    main()
