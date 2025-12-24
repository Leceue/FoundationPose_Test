#!/usr/bin/env bash
set -euo pipefail

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Resolve Python interpreter (prefer active env on PATH)
PY_EXEC=${PY_EXEC:-}
if [[ -z "${PY_EXEC}" ]]; then
	PY_EXEC="$(command -v python || true)"
fi
if [[ -z "${PY_EXEC}" ]]; then
	PY_EXEC="$(command -v python3 || true)"
fi
if [[ -z "${PY_EXEC}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
	PY_EXEC="${CONDA_PREFIX}/bin/python"
fi
if [[ -z "${PY_EXEC}" ]]; then
	echo "ERROR: Could not find python. Activate your env or pass PY_EXEC=..."; exit 1;
fi

echo "Using Python: ${PY_EXEC}"
echo "which python: $(command -v python || echo 'not found')"
PYBIN_DIR=$(dirname "${PY_EXEC}")
echo "Python bin dir: ${PYBIN_DIR}"

# Locate pybind11 cmake dir from this python (if available)
PYBIND11_CMAKEDIR=""
if "${PY_EXEC}" -m pybind11 --cmakedir >/dev/null 2>&1; then
	PYBIND11_CMAKEDIR=$("${PY_EXEC}" -m pybind11 --cmakedir || true)
	echo "pybind11 cmakedir: ${PYBIND11_CMAKEDIR}"
fi

# Build mycpp
cd "${PROJ_ROOT}/mycpp/"
if [[ -d build && ! -w build ]]; then
	echo "ERROR: ${PROJ_ROOT}/mycpp/build is not writable (likely created by sudo).";
	echo "Fix once: sudo chown -R \"$USER\":\"$USER\" \"${PROJ_ROOT}/mycpp/build\"";
	exit 1;
fi
rm -rf build
mkdir -p build
cd build
cmake .. -DPYTHON_EXECUTABLE="${PY_EXEC}" \
	$( [[ -n "${PYBIND11_CMAKEDIR}" ]] && echo -Dpybind11_DIR="${PYBIND11_CMAKEDIR}" ) \
	$( [[ -n "${PYBIND11_CMAKEDIR}" ]] && echo -DCMAKE_PREFIX_PATH="${PYBIND11_CMAKEDIR}" )
make -j"$(nproc)"

# Install mycuda (build from current env; require torch present)
cd "${PROJ_ROOT}/bundlesdf/mycuda"
rm -rf build *egg* *.so
"${PY_EXEC}" -m pip --version >/dev/null 2>&1 || { echo "pip not found for ${PY_EXEC}. Activate env or install pip."; exit 1; }

"${PY_EXEC}" - <<'PYCHECK'
import sys
print('sys.executable:', sys.executable)
try:
		import torch
		print('torch OK:', torch.__version__)
except Exception as e:
		print("ERROR: 'import torch' failed:", e)
		raise SystemExit(1)
PYCHECK

# Non-editable install, no isolation, no deps
PIP_USE_PEP517=0 "${PY_EXEC}" -m pip install . --no-build-isolation --no-deps

cd "${PROJ_ROOT}"
echo "Build completed successfully."