#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer an explicitly selected runtime, then this clone's virtual environment.
runtime_dir="${CONTACTFORGE_RUNTIME:-${MPHS_RUNTIME:-}}"
if [[ -n "${runtime_dir}" ]]; then
  runtime_python="${runtime_dir}/bin/python"
elif [[ -x "${project_root}/.venv/bin/python" ]]; then
  runtime_python="${project_root}/.venv/bin/python"
else
  runtime_python="$(command -v python || command -v python3)"
fi
if [[ ! -x "${runtime_python}" ]]; then
  echo "Python runtime not found: ${runtime_python}" >&2
  exit 1
fi

cuda_lib="$("${runtime_python}" -c 'import site; from pathlib import Path; print(next((str(Path(p)/"nvidia/cu13/lib") for p in site.getsitepackages() if (Path(p)/"nvidia/cu13/lib/libnvrtc-builtins.so.13.0").is_file()), ""))')"
export PATH="$(dirname -- "${runtime_python}"):${PATH}"
export PYTHONPATH="${project_root}/src"
export PYTHONDONTWRITEBYTECODE=1
if [[ -n "${cuda_lib}" ]]; then
  export LD_LIBRARY_PATH="${cuda_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/contactforge-mpl}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-${TMPDIR:-/tmp}/contactforge-warp}"
cd -- "${project_root}"
if [[ $# -eq 0 ]]; then
  echo "Usage: bash scripts/run_local.sh python scripts/blind_study_job.py --help" >&2
  exit 2
fi
if [[ "$1" == "python" || "$1" == "python3" ]]; then
  shift
  exec "${runtime_python}" "$@"
fi
exec "$@"
