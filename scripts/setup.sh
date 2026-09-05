#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
mkdir -p .tools .cache
if [ ! -x .tools/uv ]; then
  curl --fail --location https://astral.sh/uv/0.12.10/install.sh -o .cache/uv-install.sh
  UV_UNMANAGED_INSTALL="$PWD/.tools" sh .cache/uv-install.sh
fi
.tools/uv python install 3.11.16
if [ ! -d .venv ]; then
  .tools/uv venv --python 3.11.16 .venv
fi
.tools/uv pip install --python .venv/bin/python --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match -r requirements-lock.txt
.tools/uv pip check --python .venv/bin/python
