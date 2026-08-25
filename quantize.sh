#!/usr/bin/env bash
# =============================================================================
# ExLlamaV3 EXL3 quantizer launcher (DGX Spark / GB10 aarch64 build)
#
# Usage:
#   ./quantize.sh -i /path/to/hf-model -o /path/to/output -b 3.5 [-w /path/to/work] [extra convert.py args]
#
# Required:
#   -i   input model dir (unquantized HF format: config.json, tokenizer.json, *.safetensors)
#   -o   output dir for the EXL3 model
#   -b   target bits per weight (e.g. 3.0, 3.5, 4.0)
#   -w   working dir for checkpoints (needs ~1x output size free; default: <out_dir>.work)
#
# Optional (forwarded to convert.py): -hq, -hb, -ss, -cr, -cc, -d, -r ...
# Example:  ./quantize.sh -i ~/models/foo-hf -o ~/models/foo-exl3-3.5 -b 3.5 -hq
#
# Notes:
#   * Stop/resume: interrupted jobs are resumed with:  ./quantize.sh -w <work> -r
#   * The venv must be built first (see README.md). Re-run nothing; just activate.
#   * TORCH_CUDA_ARCH_LIST is pinned to sm_120/sm_121 for this GB10 box.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")"
PY=.venv/bin/python
test -x "$PY" || { echo "venv not found — run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0;12.1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# Parse -w out of the args so we can create it (convert.py needs it to exist)
WORK=""
ARGS=()
prev=""
for a in "$@"; do
  if [ "$prev" = "-w" ]; then WORK="$a"; fi
  prev="$a"
done
if [ -n "$WORK" ]; then
  mkdir -p "$WORK"
fi

echo "==> exllamav3 $(.venv/bin/python -c 'from exllamav3.version import __version__; print(__version__)')"
echo "==> $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), $(free -g | awk '/Mem:/{print $2" GB RAM"}')"
echo "==> TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

exec "$PY" convert.py "$@"
