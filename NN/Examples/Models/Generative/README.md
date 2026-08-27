# Generative Examples

This folder contains the runnable generative-model commands in TorchLean. They cover the runtime
paths that generative models stress: reconstruction losses, latent representations, adversarial
updates, masked image reconstruction, diffusion noise schedules, image artifacts, CUDA kernels, and
training logs.

The examples are runtime producers. They train small models, write logs or images, and keep the
model-zoo path honest across families whose losses and artifacts look very different from ordinary
classification. The mathematical identities behind the objectives live in the theory layer:
min-max losses for GAN-style updates, masking/reconstruction contracts for MAE-style models, and
denoising/noise-schedule contracts for diffusion. Formal use of a generated artifact begins with the
checker or theorem that consumes it.

## Files

- `Autoencoder.lean`: compact vector autoencoder over real CIFAR image batches.
- `Vae.lean`: supervised reconstruction with two auxiliary latent-statistic vectors.
- `VqVae.lean`: continuous autoencoder with a narrow `tanh` bottleneck.
- `Gan.lean`: compact GAN-style training loop with generator/discriminator updates.
- `Mae.lean`: ViT-MAE-style masked autoencoder path: patch masking, transformer tokens, and image
  reconstruction.
- `Diffusion.lean`: compact unconditional diffusion command with noise schedule, denoiser training,
  DDIM-style replay, and optional PPM image artifacts.

## Data

Most commands use CIFAR-10 through the shared real-data loader:

```bash
python3 scripts/datasets/download_example_data.py --cifar10
```

The diffusion command also supports ImageNet-style folders converted to `.npy`:

```bash
python3 scripts/datasets/torchlean_data_convert.py image-folder \
  --input /path/to/imagenet/train \
  --x-output data/real/imagenet64/imagenet64_train_X.npy \
  --y-output data/real/imagenet64/imagenet64_train_y.npy \
  --height 64 --width 64 --labels-from-dirs --limit 800
```

## Commands

Quick CUDA checks:

```bash
lake -R -K cuda=true exe torchlean autoencoder --device cuda --steps 1 --n-total 1
lake -R -K cuda=true exe torchlean mae --device cuda --steps 1 --n-total 1
lake -R -K cuda=true exe torchlean latent_stats --device cuda --steps 1 --n-total 1
lake -R -K cuda=true exe torchlean tanh_autoencoder --device cuda --steps 1 --n-total 1
lake -R -K cuda=true exe torchlean gan --device cuda --steps 1 --n-total 1
lake -R -K cuda=true exe torchlean diffusion --device cuda --dataset cifar10 --n-total 1 --steps 1 --hidden-c 1 --T 2
```

Longer diffusion run with visual artifacts:

```bash
lake -R -K cuda=true exe torchlean diffusion --device cuda \
  --dataset cifar10 --n-total 800 --steps 200 --hidden-c 8 --T 100 --beta-end 0.12 \
  --reference-ppm data/model_zoo/diffusion_reference.ppm \
  --noisy-ppm data/model_zoo/diffusion_noisy.ppm \
  --reconstruct-ppm data/model_zoo/diffusion_reconstruct.ppm \
  --sample-ppm data/model_zoo/diffusion_sample.ppm \
  --log data/model_zoo/diffusion_trainlog.json
```

## Outputs

The commands produce training logs, generated or reconstructed images, and CUDA runtime output.
Lean checkers can consume exported artifacts, while theory files prove properties of the objective
and model specification.
