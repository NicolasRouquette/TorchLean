# Vision Examples

This folder contains runnable models over the prepared CIFAR-10 tensors. CIFAR is stored in a
channel-first layout, while the TorchLean tensor and layer APIs remain rank-polymorphic: convolution
and pooling consume a declared spatial suffix and preserve any leading axes.

## Files

- `Cnn.lean`: compact convolutional CIFAR-10 classifier. It uses a small crop so the command is a
  practical CUDA regression target while still exercising real convolution/pooling-style data flow.
- `ResNet.lean`: residual classifier with configurable convolution geometry and global average
  pooling over every spatial axis.
- `Vit.lean`: compact ViT-style CIFAR-10 classifier. It uses convolutional patch embedding,
  token reshape, a configurable encoder stack, learned positions, and class-slot pooling.

## Data

Prepare the small real CIFAR-10 subset with:

```bash
python3 scripts/datasets/download_example_data.py --cifar10
```

The loader reads `.npy` arrays under `data/real/cifar10/`. The examples crop the image tensors to
small typed shapes so they run quickly while still crossing the same data boundary as larger image
runs.

## Commands

```bash
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
lake -R -K cuda=true exe torchlean resnet --device cuda --n-total 1 --steps 1
lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```

For runtime profiling or fast kernels:

```bash
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
lake -R -K cuda=true exe torchlean resnet --device cuda --n-total 1 --steps 1
lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```

## What To Inspect

These examples own the image-classification training path. Useful outputs are:

- the training loss and accuracy trace;
- the `TrainLog` JSON if `--log PATH` is passed;
- the typed image shapes in the Lean source;
- CUDA parity and regression evidence when changing image kernels.

For 3D detector certificates or projection verification, use `NN/Examples/Verification` and the
Geometry3D workflow. That path exports detector tensors as certificate artifacts, checks the camera
projection/enclosure conditions in Lean, and renders overlays for inspection.
