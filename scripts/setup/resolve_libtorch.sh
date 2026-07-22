#!/usr/bin/env bash
# Validate a local LibTorch installation and record its resolved path for Lake.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
stamp="$repo_root/.lake/build/libtorch.path"

mkdir -p "$(dirname "$stamp")"

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [LIBTORCH_HOME]" >&2
  exit 2
fi

if [[ $# -eq 1 ]]; then
  libtorch_home="$1"
else
  libtorch_home="$repo_root/libtorch"
fi

if [[ ! -d "$libtorch_home" ]]; then
  cat >&2 <<EOF
LibTorch was not found at:
  $libtorch_home

For CUDA builds that enable the LibTorch SDPA backend, install/extract LibTorch locally and pass:
  lake -R -K cuda=true build -K libtorch_home=/path/to/libtorch

CPU builds do not need LibTorch.
EOF
  exit 1
fi

if [[ ! -d "$libtorch_home/include" || ! -d "$libtorch_home/lib" ]]; then
  echo "LibTorch home must contain include/ and lib/: $libtorch_home" >&2
  exit 1
fi

printf '%s\n' "$libtorch_home" > "$stamp"
