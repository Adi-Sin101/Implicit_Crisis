# Notebooks

Exploratory analysis only. Anything that becomes part of the pipeline moves
into `src/` and gets a script in `scripts/`.

Conventions:

- Import from `src/` rather than redefining logic here; add the project root to
  `sys.path` at the top of the notebook.
- Use paths from `configs/paths.yaml` — no absolute paths.
- Clear outputs before committing. Cell outputs over corpus text would commit
  verbatim posts that `data/` deliberately keeps out of Git.
