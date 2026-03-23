"""Auto-generate problem_set.ipynb from all problem-* directories."""
import glob, os
import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

nb.cells.append(nbformat.v4.new_markdown_cell("# Limestone Data Challenge 2026"))

for problem_dir in sorted(glob.glob("problem-*")):
    if not os.path.isdir(problem_dir):
        continue
    dir_name = os.path.basename(problem_dir)
    nb.cells.append(nbformat.v4.new_markdown_cell(f"---\n# {dir_name}"))

    py_files = sorted(glob.glob(f"{problem_dir}/*.py"))
    if not py_files:
        continue

    for path in py_files:
        fname = os.path.basename(path)
        with open(path) as f:
            code = f.read()
        nb.cells.append(nbformat.v4.new_markdown_cell(f"### `{fname}`"))
        nb.cells.append(nbformat.v4.new_code_cell(code))

with open("problem_set.ipynb", "w") as f:
    nbformat.write(nb, f)

print(f"Created problem_set.ipynb")
