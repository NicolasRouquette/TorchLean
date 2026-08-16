#!/usr/bin/env bash
# Exercise the documented command-line examples as a regression suite.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/checks/example_regression.sh [options]

Run a sequential regression check over the public `lake exe torchlean ...` commands.

Default:
  - check that `import NN.API` exposes the usual `TorchLean.*` names;
  - verify every registered subcommand accepts `--help`;
  - run a compact CPU/tutorial/interop regression set.

Options:
  --cuda             Also run a compact set of real CUDA model checks.
  --extended-cuda    Run a broader sequential CUDA regression pass over model-zoo commands.
  --external-rl      Run optional external-environment RL checks such as ALE/Pong.
  --skip-help       Skip the all-subcommand help audit.
  --skip-default    Skip the default CPU/tutorial/interop regression set.
  -h, --help        Show this help message.

Environment:
  LAKE              Lake executable to use (default: lake).

Notes:
  Commands are intentionally run one at a time. Running several `lake exe`
  invocations in parallel can race while relinking the shared executable.
EOF
}

LAKE="${LAKE:-lake}"
run_cuda=false
run_extended_cuda=false
run_external_rl=false
run_help=true
run_default=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda)
      run_cuda=true
      shift
      ;;
    --extended-cuda)
      run_extended_cuda=true
      shift
      ;;
    --external-rl)
      run_external_rl=true
      shift
      ;;
    --skip-help)
      run_help=false
      shift
      ;;
    --skip-default)
      run_default=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/torchlean-example-regression.XXXXXX")"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

run() {
  printf '\n==>'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  "$@"
}

public_api_check="$tmp_dir/public_api_check.lean"
cat > "$public_api_check" <<'LEAN'
import NN.API

open TorchLean

#check TorchLean.nn.linear
#check TorchLean.nn.lowerToTypedGraph
#check TorchLean.optim.adam
#check TorchLean.Trainer.new
#check TorchLean.Data.tensorDataset
#check TorchLean.Loss.mse
#check TorchLean.Metrics.argmax?
#check Spec.KNN
#check random_forest.Numeric.RegressionForestSpec
#check Spec.pcaForwardSpec
LEAN
run "$LAKE" build +NN.API
run "$LAKE" env lean "$public_api_check"

if [[ "$run_help" == true ]]; then
  run python3 - "$LAKE" <<'PY'
from pathlib import Path
import re
import subprocess
import sys

lake = sys.argv[1]
runner = Path("NN/Examples/Models/Runner.lean").read_text()
commands = re.findall(r'^\s*\| "([^"]+)" =>', runner, re.M)
print(f"help audit: {len(commands)} registered torchlean subcommands")

bad = []
required_help = {
    "rnn": ("Data:", "--data-file PATH", "--tiny-shakespeare", "Runtime:"),
    "lstm": ("Data:", "--data-file PATH", "--tiny-shakespeare", "Runtime:"),
    "transformer": ("Data:", "--data-file PATH", "--tiny-shakespeare", "Runtime:"),
}
for cmd in commands:
    proc = subprocess.run(
        [lake, "exe", "torchlean", cmd, "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=180,
    )
    out = proc.stdout
    required = required_help.get(cmd, ())
    ok = (
        proc.returncode == 0
        and ("Usage:" in out or "usage:" in out.lower())
        and all(item in out for item in required)
    )
    if ok:
        print(f"  ok  {cmd}")
    else:
        print(f"  bad {cmd} (exit={proc.returncode})")
        bad.append((cmd, proc.returncode, out[-1200:]))

if bad:
    print("\nhelp audit failures:")
    for cmd, code, tail in bad:
        print(f"--- {cmd} exit={code} ---")
        print(tail)
    sys.exit(1)
PY
fi

if [[ "$run_default" == true ]]; then
  run "$LAKE" exe torchlean quickstart_tensors
  run "$LAKE" exe torchlean quickstart_autograd
  run "$LAKE" exe torchlean quickstart_mlp \
    --steps 1 --scalar float32 --execution eager --log "$tmp_dir/quickstart_mlp.json"
  run "$LAKE" exe torchlean quickstart_minibatch_mlp \
    --steps 1 --batch 5 --scalar float32 --execution eager --log "$tmp_dir/minibatch_mlp.json"
  run "$LAKE" exe torchlean quickstart_cnn \
    --steps 1 --batch 2 --scalar float32 --execution eager --log "$tmp_dir/quickstart_cnn.json"
  run "$LAKE" exe torchlean data_csv \
    --steps 1 --batch 5 --scalar float32 --execution eager
  run "$LAKE" exe torchlean data_npy \
    --steps 1 --batch 5 --scalar float32 --execution eager
  run "$LAKE" exe torchlean data_cifar10 \
    --check-only --epochs 1 --batch 4 --train-size 8 --n-total 20
  run "$LAKE" exe torchlean pytorch_roundtrip --model mlp --action import
  run "$LAKE" exe torchlean pytorch_export_check
  run "$LAKE" exe torchlean floats_arb_ieee_compare
  run "$LAKE" exe torchlean float32_semantics
  run "$LAKE" exe torchlean numerical_certificate
  run "$LAKE" exe torchlean graphspec --execution eager
  run "$LAKE" exe torchlean ir_axis_ops --scalar float32 --execution eager
  run "$LAKE" exe torchlean one_semantic_universe --samples 3
  run "$LAKE" exe torchlean torch_ir_pytorch
  run "$LAKE" exe torchlean dqn_replay
fi

if [[ "$run_cuda" == true ]]; then
  run "$LAKE" exe -K cuda=true torchlean gpt_adder \
    --device cuda --steps 1 --a 7 --b 8 --log "$tmp_dir/gpt_adder.json"
  run "$LAKE" exe -K cuda=true torchlean cnn \
    --device cuda --steps 1 --n-total 1 --log "$tmp_dir/cnn_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean vit \
    --device cuda --steps 1 --n-total 1 --log "$tmp_dir/vit_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean gpt2 \
    --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0 \
    --log "$tmp_dir/gpt2_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean mamba \
    --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0 \
    --log "$tmp_dir/mamba_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean fno1d_burgers \
    --device cuda --steps 1 \
    --plot-csv "$tmp_dir/fno_predictions.csv" --log "$tmp_dir/fno_cuda.json"
fi

if [[ "$run_extended_cuda" == true ]]; then
  # Supervised and tabular models.
  run "$LAKE" exe -K cuda=true torchlean mlp \
    --device cuda --steps 1 --log "$tmp_dir/mlp_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean kan \
    --device cuda --steps 1 --log "$tmp_dir/kan_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean lstm_regression \
    --device cuda --steps 1 --windows 1 --log "$tmp_dir/lstm_regression_cuda.json"

  # Vision and generative models.
  run "$LAKE" exe -K cuda=true torchlean autoencoder \
    --device cuda --steps 1 --n-total 1 --log "$tmp_dir/autoencoder_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean mae \
    --device cuda --steps 1 --n-total 1 --log "$tmp_dir/mae_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean vae \
    --device cuda --steps 1 --n-total 1 --log "$tmp_dir/vae_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean vqvae \
    --device cuda --steps 1 --n-total 1 --log "$tmp_dir/vqvae_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean gan \
    --device cuda --steps 1 --n-total 1 --log "$tmp_dir/gan_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean diffusion \
    --device cuda --dataset cifar10 --n-total 1 --steps 1 --hidden-c 1 --T 2 \
    --log "$tmp_dir/diffusion_cuda.json" \
    --reference-ppm "$tmp_dir/diffusion_reference.ppm" \
    --sample-ppm "$tmp_dir/diffusion_sample.ppm"

  # Sequence models.
  run "$LAKE" exe -K cuda=true torchlean rnn \
    --device cuda --steps 1 --log "$tmp_dir/rnn_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean lstm \
    --device cuda --steps 1 --log "$tmp_dir/lstm_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean transformer \
    --device cuda --tiny-shakespeare --steps 1 \
    --log "$tmp_dir/transformer_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean chargpt \
    --device cuda --tiny-shakespeare --steps 1 --batch 1 --seq-len 1 --generate 0 \
    --log "$tmp_dir/chargpt_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean gpt2 \
    --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0 \
    --save-checkpoint "$tmp_dir/gpt2_saved.state.json" \
    --log "$tmp_dir/gpt2_extended_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean gpt2_saved \
    --device cuda --checkpoint "$tmp_dir/gpt2_saved.state.json" \
    --prompt "First Citizen:" --generate 0
  run "$LAKE" exe -K cuda=true torchlean text_gpt2 \
    --device cuda --data-file data/real/text/tinystories_valid.txt \
    --allow-small-data --steps 1 --generate 0 \
    --log "$tmp_dir/text_gpt2_cuda.json"

  # RL examples with tiny evaluation windows.
  run "$LAKE" exe -K cuda=true torchlean ppo_cartpole \
    --device cuda --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8 \
    --log "$tmp_dir/ppo_cartpole_cuda.json"
  run "$LAKE" exe -K cuda=true torchlean ppo_gridworld \
    --device cuda --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8 \
    --log "$tmp_dir/ppo_gridworld_cuda.json" \
    --policy "$tmp_dir/ppo_gridworld_policy.json" \
    --path "$tmp_dir/ppo_gridworld_path.json"
fi

if [[ "$run_external_rl" == true ]]; then
  run "$LAKE" exe -K cuda=true torchlean ppo_pong_ram \
    --device cuda --check-env-only
fi

printf '\nTorchLean example regression checks passed.\n'
