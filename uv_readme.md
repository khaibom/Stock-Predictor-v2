# uv_readme.md — Setup Guide Using **uv**

This guide explains how to install and manage all dependencies for the **Stock-Predictor (Dagster + LSTM)** project using **[uv](https://github.com/astral-sh/uv)** — a fast, modern Python package manager that replaces `pip` and `venv`.

---

## Requirements

- **Python 3.10.x**  
  > TensorFlow does **not** yet support Python 3.13 or newer.

- **uv ≥ 0.4**

Check your versions:

```bash
python --version
uv --version
```

If you don’t have **uv**, install it:

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

---

## Create a uv Environment

From the project root:

```bash
# create a virtual environment with Python 3.10
uv venv -p 3.10

# activate it
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

---

## Install Dependencies

All dependencies are defined in **`pyproject.toml`** and **`uv.lock`**.

To install everything:

```bash
uv sync
```

This will:
- create the `.venv/` directory if missing  
- install all runtime and dev dependencies  
- reproduce exact versions from `uv.lock` (if present)

To upgrade everything:

```bash
uv lock --upgrade
uv sync
```

---

## Add or Remove Packages

Add new runtime dependencies (instead of `pip install`):

```bash
uv add dagster dagster-webserver pandas numpy
```

Add dev-only packages:

```bash
uv add --dev black pytest isort
```

Remove packages:

```bash
uv remove seaborn
```

---

## Import Existing `requirements.txt` (Optional)

If you still have an old `requirements.txt`, you can import it once:

```bash
uv add -r requirements.txt
```

This converts its entries into your `pyproject.toml`.  
After that, you no longer need `pip install -r requirements.txt`.

---

## Lock and Reproduce Environments

After modifying dependencies:

```bash
uv lock
```

To exactly reproduce the same environment (for CI or teammates):

```bash
uv sync --frozen
```

---

## Run Commands with uv

You can run scripts without manually activating the venv:

```bash
uv run dagster dev
uv run dagster job execute -j training_job
uv run streamlit run streamlit_app/app.py
uv run pytest
```

---

## Clean or Reset

```bash
uv clean        # remove caches and temporary wheels
rm -rf .venv    # delete local environment (if needed)
```

---

## Quick Reference

| Action | Command |
|:--|:--|
| Create environment | `uv venv -p 3.10` |
| Install dependencies | `uv sync` |
| Add package | `uv add <pkg>` |
| Add dev package | `uv add --dev <pkg>` |
| Import requirements.txt | `uv add -r requirements.txt` |
| Update lockfile | `uv lock --upgrade` |
| Reproduce exact env | `uv sync --frozen` |
| Run script | `uv run python your_script.py` |

---

Now your **Stock-Predictor** project is fully managed by **uv** —  
no more `pip install` or `requirements.txt` headaches
