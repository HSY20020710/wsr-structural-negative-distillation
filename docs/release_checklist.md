# Release Checklist

Use this checklist before uploading the repository to GitHub.

- [ ] Confirm no private raw reports are present.
- [ ] Confirm no model weights or checkpoints are present.
- [ ] Confirm `outputs/`, `reports/`, `work/`, and `models/` are not staged.
- [ ] Confirm internal API addresses have been replaced by localhost or environment variables.
- [ ] Run `python -m py_compile` on public Python files.
- [ ] Run unit tests if dependencies are available.
- [ ] Update `CITATION.cff` with the final GitHub URL and paper DOI if available.
- [ ] Update README paper citation after acceptance or preprint release.
