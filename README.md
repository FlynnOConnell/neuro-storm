<h1 align="center">neuro-storm</h1>

<samp>
<p align="center">
Brainstorming utilities for optical neuroscience data processing and visualization.
<br>
Like a notebook repo, but every script is a standalone <b>uv</b> program that carries its own dependencies.
<br>
<br>
<a href="#about">about</a> ·
<a href="#install">install</a> ·
<a href="#usage">usage</a> ·
<a href="#write-a-script">write a script</a>
</p>
</samp>

## About

Each script in `neuro_storm/cli/` is a self-contained [PEP 723](https://peps.python.org/pep-0723/) program:

- **declares its dependencies inline** — no shared environment to install or keep in sync
- **runs with `uv run <script.py>`** — uv builds a per-script virtualenv on the fly, cached after the first run
- **quick to spin up** — copy the header, write a `main()`, and you have a throwaway tool to test or develop an idea
- **installable when it matters** — a script wired into `pyproject.toml` also becomes a console command (e.g. `load-raw`)

## Install

```bash
make install
```

## Usage

Load and summarize raw LBM TIFF data:

```bash
load-raw ~/lbm_data/raw
```

Or run any script standalone — uv reads the inline metadata and builds the venv for you:

```bash
uv run neuro_storm/cli/load_raw.py ~/lbm_data/raw
```

## Write a script

Add a `# /// script` block at the top and the file is runnable on its own:

```python
#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = ["tifffile", "numpy"]
# ///
"""Do one thing."""

def main() -> int:
    ...

if __name__ == "__main__":
    raise SystemExit(main())
```

Then `uv run neuro_storm/cli/your_script.py`, or `chmod +x` it and run `./your_script.py` directly.
