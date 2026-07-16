# SkillSieve

## Paper

The current manuscript is [`paper/main_v2.pdf`](paper/main_v2.pdf), compiled
from [`paper/main_v2.tex`](paper/main_v2.tex).

## Installation

```bash
pip install -e ".[dev]"
```

Layer 1 runs locally. Layers 2 and 3 use remote LLM APIs through LiteLLM.
Provider credentials must be supplied through local environment or LiteLLM
configuration and must not be committed to this repository.

## Basic usage

```bash
skillsieve scan ./path/to/skill/
skillsieve scan ./path/to/skill/ --max-layer 1
skillsieve scan-batch ./skills-dir/ --output results.json
```

## Public data

[`results/labeled_dataset_final.json`](results/labeled_dataset_final.json)
contains the benchmark index and labels used by the paper. It includes the
source `author/skill` path, label provenance, and the flag indicating whether a
record belongs to the 390-skill evaluation subset.

Copies of third-party skill packages are not redistributed. This avoids
republishing upstream code, credentials embedded in public samples, and files
with incompatible or unclear licenses. The labels apply to the archived
versions used in the study and are not claims about an author or a package's
current version.

Raw LLM responses, intermediate checkpoints, internal audits, superseded
result files, and draft manuscripts are intentionally excluded. Aggregate
results are reported in the paper.

## Repository layout

```text
skillsieve/   Core detection pipeline
experiments/  Paper experiment source code
results/      Minimal public benchmark labels
paper/        LaTeX source, required figures, and current PDF
tests/        Unit tests
bot/          Feishu deployment described in the paper
```

## Tests

```bash
pytest -q tests
```

## License

MIT
