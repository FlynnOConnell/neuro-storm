# neuro-storm

[![Build status](https://img.shields.io/github/actions/workflow/status/FlynnOConnell/neuro-storm/main.yml?branch=main)](https://github.com/FlynnOConnell/neuro-storm/actions/workflows/main.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/FlynnOConnell/neuro-storm)](https://img.shields.io/github/license/FlynnOConnell/neuro-storm)

Brainstorming utilities for optical neuroscience data processing and visualization.

- **Repository**: <https://github.com/FlynnOConnell/neuro-storm/>
- **Documentation**: <https://FlynnOConnell.github.io/neuro-storm/>

## Install

```bash
make install
```

## Usage

Load and summarize raw LBM TIFF data:

```bash
load-raw ~/lbm_data/raw
```

Or run the script standalone (uv reads the inline PEP 723 metadata):

```bash
uv run neuro_storm/cli/load_raw.py ~/lbm_data/raw
```

## Development

```bash
make install          # set up the environment and pre-commit hooks
uv run pytest         # run the tests
uv run pre-commit run -a
```
