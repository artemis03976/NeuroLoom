# Import corpus

Two sets of model files for NeuroLoom's import tests.

## handwritten/

Model files in the styles found in real repositories, written for these tests
(not copies of third-party code). Each file's header gives the input shapes
(`# inputs:`) and, when the entry is not the last unreferenced class, the entry
class (`# class:`).

`packages/core/test/corpus.test.ts` imports every file statically and
dynamically, checks that the generated code is equivalent to the original
(`state_dict` loads with `strict=True`, same outputs), and prints how much of
each file was structured.

## third_party/

Model files from common GitHub repositories, unmodified, for the PRD success
metric "static import success ≥ 80% on 50 repository model files": 50 measured
files and 12 held out. How they were chosen, and the results, are in
`docs/import-corpus-report.md`.

- `manifest.json`: for each file
  - where it comes from: repository, pinned commit, path, sha256, license;
  - how it is built: entry class, `setup` statements and the `construct` expression,
    an `example` input expression and the input shapes given to the importer;
  - the repository files Python runs to import it (`files`), and whether it is a
    script whose top-level code only its definitions may run (`script`);
  - `measure` or `heldout`.
- `fetch.py` downloads the files at their commits and checks the hashes. The files
  are committed, so tests never download; `--update` records new hashes and
  rewrites `THIRD_PARTY_NOTICES.md`.
- `realworld.py` builds each original as its repository does and compares code
  NeuroLoom generated from it, offline (`originals` builds them all once).
- `packages/core/test/realworld.test.ts` computes the metric.

Library files (timm, transformers, diffusers) are pinned to the release whose
version `requirements-dev.txt` installs, and their originals run from the installed
package.
