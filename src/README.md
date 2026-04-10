# src — Canonical Python Source Tree

The canonical Python source tree for the project. The only thing
inside is the `dwipada/` package; the project is configured as
editable via `pyproject.toml` so that `pip install -e .` from the
repo root makes the `dwipada` CLI and the `dwipada.*` import paths
available globally.

## Layout

```
src/
├── dwipada/         # The main Python package (see dwipada/README.md)
└── dwipada.egg-info/ # pip install -e . metadata (auto-generated)
```

## Relationship with `chandomitra/src/`

There is a second copy of the same package at
`../chandomitra/src/dwipada/`. The two are **forks**, not symlinks:
the chandomitra copy is a snapshot that adds extra files used only
by the standalone Chandomitra benchmark in `../chandomitra/`
(constrained-generation utilities, an extra `vertex.py` in `batch/`,
extra `validate.py`/`validation_utils.py`/`word_ttr.py` in `dataset/`,
the `crawl_base.py` + `crawlers/` subtree, and the
`training/constrained/` subdirectory).

The canonical version is the one in **this** folder (`src/dwipada/`);
this is the copy that `tests/` and the top-level CLI consume. See
`dwipada/README.md` for the package overview and the per-subpackage
READMEs (`dwipada/core/README.md`, `dwipada/batch/README.md`,
`dwipada/data/README.md`, `dwipada/dataset/README.md`) for details on
each component.

## Installation

From the repository root:

```bash
pip install -e ".[dev]"
```

This makes the `dwipada` console script available system-wide and
adds `src/` to `sys.path` for development.
