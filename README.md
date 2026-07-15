<h1 align="center">neuro-storm</h1>

<samp>
<p align="center">
A hub for self-contained [PEP 723-complaint](https://peps.python.org/pep-0723/) optical neuroscience data processing and visualization scripts that carry their own dependencies.
<br>

## Usage

It's recommended to run scripts with [uv](https://docs.astral.sh/uv/) to take advantage of dependency caching and making new scripts faster to spin up.

You start with a plain `.py` file, and the dependency block at the **top** is the whole environment. 

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

Run the script 

```bash
uv run my_script.py
```

Published scripts can be called without without needing to worry about the environment at all: 

```bash
uvx --from neuro-storm my_command
```
