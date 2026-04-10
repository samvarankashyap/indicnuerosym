# chandomitra/src — Chandomitra-Side Python Source Tree

The Python source tree as snapshotted into the Chandomitra
benchmark folder. Mirrors the canonical `../../src/` tree but
includes extra files needed by the Chandomitra constrained-decoding
benchmark scripts in `../`.

## Layout

```
chandomitra/src/
├── dwipada/         # The dwipada package (snapshot fork — see dwipada/README.md)
└── dwipada.egg-info/ # pip install -e . metadata (auto-generated)
```

## Relationship with `../../src/`

The canonical version of this package lives at `../../src/dwipada/`.
This folder is a **fork-snapshot** that adds files needed only by the
Chandomitra benchmarks:

| Sub-package | Extra files in this fork (vs canonical) |
| --- | --- |
| `dwipada/batch/` | `vertex.py` (Vertex AI client) |
| `dwipada/data/` | `crawl_base.py`, `crawlers/` (5 source crawlers) |
| `dwipada/dataset/` | `validate.py`, `validation_utils.py`, `word_ttr.py` |
| `dwipada/training/` | `constrained/` subtree (`logits_processor.py`, `pattern_trie.py`, `generation_state.py`, `syllable_utils.py`) and `generate_constrained.py` |

These extras implement the Chandomitra port of Algorithm 1 from
Jagadeeshan et al. 2026 (the published Chandomitra paper, included
as `../chandomitra.pdf`). They are intentionally kept here so the
Chandomitra benchmark scripts in `../` are self-contained and can be
re-run independently of the canonical `src/` tree.

## Installation

The Chandomitra benchmarks are typically run from the chandomitra/
folder with its own `pyproject.toml`:

```bash
cd ../  # i.e. chandomitra/
pip install -e .
```

That makes the `dwipada` console script and the `dwipada.training.constrained.*`
import paths available for the chandomitra benchmark scripts.

## See also

- `../README.md` — full Chandomitra benchmark documentation
- `../../src/README.md` — canonical package source tree
- `dwipada/README.md` — package overview (per-subpackage READMEs in
  the subdirectories)
