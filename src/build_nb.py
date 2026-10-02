import nbformat, re, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
def build(src_py, out_ipynb):
    src = open(src_py).read().replace("#@@PREP_COMMON@@", open(os.path.join(HERE, "prep_common.py")).read()).replace("#@@MODEL_COMMON@@", open(os.path.join(HERE, "model_common.py")).read())
    cells = []
    for chunk in re.split(r"^# %%", src, flags=re.M)[1:]:
        if chunk.startswith(" [markdown]"):
            body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in chunk.split("\n", 1)[1].strip().splitlines())
            cells.append(nbformat.v4.new_markdown_cell(body))
        else:
            cells.append(nbformat.v4.new_code_cell(chunk.strip("\n")))
    nbformat.write(nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}), out_ipynb)
    open(out_ipynb.replace(".ipynb", "_flat.py"), "w").write(src)
    print(out_ipynb, len(cells), "cells")
build(sys.argv[1], sys.argv[2])
