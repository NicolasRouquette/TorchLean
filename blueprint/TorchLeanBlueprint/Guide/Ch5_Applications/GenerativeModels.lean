import VersoManual

open Verso.Genre Manual

#doc (Manual) "Generative Models" =>
%%%
tag := "generative-models"
%%%

Generative modeling forces several kinds of reasoning into one program. There is a trainable
network, but also a probability law, a noise source, an objective with several terms, and a
sampling procedure that may not resemble the training pass. TorchLean does not package all of this
under one vague claim of “verified generation.” It gives the pieces separate Lean definitions and
connects them where the current proofs justify the connection.

This chapter begins with one complete diffusion run. It then uses the VAE, VQ-VAE, GAN, and masked
autoencoder examples to show where executable training and formal objective theory currently meet.

# A Diffusion Run From Data To Sample

The maintained
[`diffusion` application](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Diffusion.lean)
supports prepared CIFAR-10 arrays and converted $`64\times64` image folders. Its compact CIFAR
branch crops images to $`2\times2` for a fast end-to-end check.

Prepare CIFAR and run one CPU update:

```
python3 scripts/datasets/download_example_data.py --cifar10

lake exe torchlean diffusion --device cpu \
  --dataset cifar10 --n-total 1 \
  --steps 1 --hidden-c 1 --T 2 \
  --log /tmp/diffusion-trainlog.json
```

The command prints the exact typed network before training:

```
torchlean diffusion: diffusion trainer (device=cpu)
model:
Sequential: [1, 4, 2, 2] -> [1, 3, 2, 2], layers=7, params=15
  [0] Conv2d(4, 1): [1, 4, 2, 2] -> [1, 1, 2, 2]
  [1] ReLU: [1, 1, 2, 2] -> [1, 1, 2, 2]
  [2] Conv2d(1, 1): [1, 1, 2, 2] -> [1, 1, 2, 2]
  [3] ReLU: [1, 1, 2, 2] -> [1, 1, 2, 2]
  [4] Conv2d(1, 1): [1, 1, 2, 2] -> [1, 1, 2, 2]
  [5] ReLU: [1, 1, 2, 2] -> [1, 1, 2, 2]
  [6] Conv2d(1, 3): [1, 1, 2, 2] -> [1, 3, 2, 2]
steps=1 loss0=1.093821 loss1=1.092744
  wrote TrainLog JSON: /tmp/diffusion-trainlog.json
torchlean diffusion: ok
```

Why does the input have four channels while the output has three? The clean image has three RGB
channels. The noised image receives one additional channel containing the normalized timestep.
For batch $`B`, data channels $`C`, and spatial extent $`S`, the model contract is

$$`B\times(C+1)\times S
\longrightarrow
B\times C\times S.`

The reusable constructor
[`nn.models.epsConvNet`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Diffusion.lean)
is parameterized by arbitrary spatial rank. The runnable command instantiates two spatial axes and
uses four $`1\times1` convolutions. A stronger residual same-resolution denoiser also exists in the API,
but it is not the default command path.

# Forward Noising

Let $`\beta_t` be the variance schedule, $`\alpha_t=1-\beta_t`, and

$$`\bar\alpha_t=\prod_{s=0}^{t}\alpha_s.`

The DDPM forward process can sample timestep $`t` directly:

$$`x_t
=\sqrt{\bar\alpha_t}\,x_0
+\sqrt{1-\bar\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).`

The training sample stores the noised image and timestep as input and the same $`\epsilon` as
target. The network therefore learns an epsilon predictor $`\epsilon_\theta(x_t,t)` by mean squared
error.

TorchLean keeps randomness outside the pure noising helper. In
[`NN.API.Models.Diffusion`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Models/Diffusion.lean),
`noisedSampleFromEps` receives an explicit noise tensor. `noisedSample` obtains a reproducible
tensor from a `(seed, step)` pair and then calls the pure helper. This separation makes it possible
to state properties of the noising map without treating ambient randomness as an invisible global
effect.

The mathematical forward law is developed separately in
[`NN.MLTheory.Generative.Diffusion.ForwardGaussian`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Diffusion/ForwardGaussian.lean).
The theorem

```
NN.MLTheory.Generative.Diffusion.forwardGaussian_isGaussian
```

proves that the affine transformation used by `forwardGaussian` is again Gaussian. The sibling
`forwardNoising` development in
*Probability and Local Gradient Proofs* packages the same affine-Gaussian recipe more generally and
adds a Markov kernel, although there is not yet a theorem rewriting one definition into the other.
These are theorems about probability laws, not evidence that the runtime RNG produced independent
standard-normal bits or that a trained denoiser matches the exact score.

# Reverse Sampling

The command uses deterministic DDIM updates for reconstruction and sample artifacts. Given adjacent
schedule values and a predicted noise tensor,

$$`\widehat x_0
=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\widehat\epsilon_t}
       {\sqrt{\bar\alpha_t}},`

$$`x_{t-1}
=\sqrt{\bar\alpha_{t-1}}\,\operatorname{clip}(\widehat x_0,-1,1)
+\sqrt{1-\bar\alpha_{t-1}}\,\widehat\epsilon_t.`

The implementation exposes `ddimPrev` as a dataset-independent helper. The command can write four
different images:

- `--reference-ppm`: the clean input;
- `--noisy-ppm`: a forward-noised input;
- `--reconstruct-ppm`: DDIM reconstruction from a chosen timestep;
- `--sample-ppm`: an unconditional sample beginning from noise.

A small CUDA run that writes all four is:

```
lake -R -K cuda=true exe torchlean diffusion --device cuda \
  --dataset cifar10 --n-total 8 \
  --steps 20 --hidden-c 4 --T 20 \
  --reference-ppm /tmp/reference.ppm \
  --noisy-ppm /tmp/noisy.ppm \
  --reconstruct-ppm /tmp/reconstruct.ppm \
  --sample-ppm /tmp/sample.ppm
```

The sampler theory in
[`NN.MLTheory.Generative.Diffusion.Samplers`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Diffusion/Samplers.lean)
proves local facts such as
`eulerStep_l2_lipschitz_of_rhs_lipschitz` and contraction results for composed DDIM or
probability-flow Euler steps under explicit hypotheses. Their scope is the mathematical step map.
FID, perceptual quality, learned-score accuracy, CUDA equivalence, and the connection between the
native random generator and the formal Gaussian measure remain separate questions. Seeds and
explicit noise tensors make the command replayable without silently answering those questions.

## Two Useful Variations

Increase `--T` while keeping `--steps 1`. The optimizer still takes one update, but the noising
schedule has more timesteps and the reverse artifact requires more model evaluations.

Then keep `--T` fixed and increase `--hidden-c`. The schedule is unchanged; only the denoiser
capacity and parameter count change. Separating those knobs helps distinguish diffusion-process
cost from network cost.

# Autoencoders Before Latent Probability

Diffusion made time and noise explicit. A plain autoencoder is a useful reset: remove the schedule
and sampler, keep only an encoder, a latent vector, a decoder, and a reconstruction loss. Once that
baseline is clear, the extra probability law in a VAE and the finite codebook in a VQ-VAE are much
easier to locate.

The plain autoencoder is the smallest reconstruction baseline:

$$`x
\xrightarrow{\mathrm{encoder}}z
\xrightarrow{\mathrm{decoder}}\widehat x,\qquad
L_{\mathrm{recon}}=\|x-\widehat x\|_2^2.`

The reusable vector model is

```
dataDim -> hiddenDim -> latentDim -> hiddenDim -> dataDim
```

with ReLU hidden activations and a sigmoid output. The runnable example flattens a small prefix of a
CIFAR image into a typed vector.

```
lake exe torchlean autoencoder --device cpu \
  --n-total 1 --steps 1 \
  --log /tmp/autoencoder-trainlog.json
```

Observed output:

```
torchlean autoencoder: CIFAR vector reconstruction (device=cpu)
dataset size = 1
mean_loss(before) = 0.024575
mean_loss(after) = 0.024385
steps=1 loss0=0.024575 loss1=0.024385
torchlean autoencoder: ok
```

This baseline is useful because it isolates data loading, flattening, reconstruction, and optimizer
state before adding a probabilistic interpretation.

# The VAE Objective And The Current Runtime Example

A VAE introduces an approximate posterior $`q_\phi(z\mid x)`, a prior $`p(z)`, and the negative
evidence lower bound

$$`\mathcal L_{\mathrm{VAE}}
=
\mathbb E_{q_\phi(z\mid x)}
  [-\log p_\theta(x\mid z)]
+\beta\,D_{\mathrm{KL}}
  \left(q_\phi(z\mid x)\,\|\,p(z)\right).`

For diagonal Gaussian posterior parameters $`\mu_i` and $`\sigma_i^2`, the KL to a standard normal is

$$`D_{\mathrm{KL}}
=\frac12\sum_i
\left(\mu_i^2+\sigma_i^2-\log\sigma_i^2-1\right)\ge0.`

The nonnegativity theorem is

```
NN.MLTheory.Generative.Latent.diagonalGaussianKlToStandardReal_nonneg
```

and `betaVae_loss_eq_weightedTwoTerm` records the reconstruction-plus-weighted-KL decomposition.
The theory also contains coordinatewise reparameterization laws.

The current executable is intentionally narrower. Its output contains a reconstruction followed by
latent mean and log-variance proxy channels, and its supervised target asks for the image plus zero
latent proxies. It trains that target with MSE. Thus a runnable VAE-shaped network sits beside proved
objective facts, while end-to-end stochastic variational inference remains future work.

```
lake exe torchlean vae --device cpu \
  --n-total 1 --steps 1 \
  --log /tmp/vae-trainlog.json
```

The current run reports:

```
torchlean vae: CIFAR beta-VAE-style training (device=cpu)
dataset size = 1
mean_loss(before) = 0.142191
mean_loss(after) = 0.140895
steps=1 loss0=0.142191 loss1=0.140895
torchlean vae: ok
```

# Finite Codebooks In VQ-VAE

VQ-VAE replaces a continuous latent sample by the nearest entry in a finite codebook. If
$`e_1,\ldots,e_K` are code vectors and $`z_e(x)` is the encoder output, then

$$`k^\star
\in\operatorname*{arg\,min}_{1\le k\le K}
\|z_e(x)-e_k\|_2^2.`

The standard objective combines reconstruction, codebook, and commitment terms:

$$`L
=L_{\mathrm{recon}}
+\|\operatorname{sg}(z_e)-e_{k^\star}\|_2^2
+\beta\|z_e-\operatorname{sg}(e_{k^\star})\|_2^2.`

The theory module
[`NN.MLTheory.Generative.Latent.VQVAE`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Latent/VQVAE.lean)
proves `vqvae_loss_eq_weightedThreeTerm` and
`nearestCode_minimizes_quantization_loss` for the stated finite codebook predicate.

The current `vqvae` command is a compact reconstruction proxy with a narrow `tanh` bottleneck. The
learned discrete lookup and straight-through estimator are not part of that path, so the theorem and
runtime example cover adjacent pieces of the intended architecture rather than one completed proof.

# Two Networks In The GAN Example

For least-squares GAN objectives, one common scalar form is

$$`\begin{aligned}
L_D
&=\mathbb E_x[(D(x)-1)^2]
  +\mathbb E_z[D(G(z))^2],\\
L_G
&=\mathbb E_z[(D(G(z))-1)^2].
\end{aligned}`

The theory in
[`NN.MLTheory.Generative.Latent.GAN`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Generative/Latent/GAN.lean)
packages the weighted generator and discriminator objectives and proves zero-loss facts at the
ideal scalar scores.

The runnable
[`GAN example`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Gan.lean)
uses two trainers, but it chooses a stable warm-up:

- the generator maps deterministic latent noise toward one CIFAR minibatch;
- the discriminator separates that minibatch from deterministic noise images.

It exercises generator state, discriminator state, two optimizers, and combined logging. A full
alternating recipe would additionally feed the generator's latest samples to the discriminator.

```
lake exe torchlean gan --device cpu --n-total 1 --steps 1
```

This is a good source to read when implementing a genuinely alternating trainer because the
two-model state boundary is already explicit.

# Masked Autoencoding

The `mae` command uses real image masking rather than merely a narrow vector bottleneck. It divides
the image into patches, applies a deterministic mask, embeds visible patch tokens with a compact
ViT encoder, and trains a decoder head to reconstruct a flattened image prefix.

For a mask set $`M`, the finite reconstruction objective has the form

$$`L_{\mathrm{MAE}}
=\frac1{|M|}\sum_{i\in M}
\|\widehat x_i-x_i\|_2^2.`

The self-supervised theory proves finite-patch identities such as `maeLoss_append`,
`maeLoss_reverse`, and `exactReconstruction_identity`. Their scope is the exact objective;
downstream representation transfer remains an empirical question.

The executable source is
[`NN/Examples/Models/Generative/Mae.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Generative/Mae.lean);
the mathematical development is under
[`NN/MLTheory/SelfSupervised`](https://github.com/lean-dojo/TorchLean/tree/main/NN/MLTheory/SelfSupervised).

# Results From The Run

The generative stack is strongest when its claims remain compositional:

:::table +header
*
  * Result
  * Meaning
*
  * training loss decreased
  * one executable optimization run completed
*
  * PPM or JSON was written
  * a runtime artifact was produced at the named path
*
  * `forwardGaussian_isGaussian`
  * the formal affine Gaussian law is Gaussian
*
  * sampler Lipschitz theorem
  * the mathematical step satisfies the stated bound under its hypotheses
*
  * KL nonnegativity
  * the formal diagonal-Gaussian KL term is nonnegative
*
  * nearest-code theorem
  * the selected finite code minimizes the stated squared-distance objective
*
  * backend capsule
  * the run records the provider and evidence level of accelerated operations
:::

Read each row at its own scale. A lower loss says something about one run; a theorem says something
about every object satisfying its hypotheses; a backend capsule records how a runtime operation was
provided. Combining them requires an explicit bridge, not a more enthusiastic description of the
sample image.

# References

- Ho, Jain, and Abbeel,
  [*Denoising Diffusion Probabilistic Models*](https://arxiv.org/abs/2006.11239), 2020.
- Song, Meng, and Ermon,
  [*Denoising Diffusion Implicit Models*](https://arxiv.org/abs/2010.02502), 2020/2021.
- Kingma and Welling,
  [*Auto-Encoding Variational Bayes*](https://arxiv.org/abs/1312.6114), 2013/2014.
- van den Oord, Vinyals, and Kavukcuoglu,
  [*Neural Discrete Representation Learning*](https://arxiv.org/abs/1711.00937), 2017.
- Mao et al.,
  [*Least Squares Generative Adversarial Networks*](https://arxiv.org/abs/1611.04076), 2016/2017.
- He et al.,
  [*Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377),
  2021/2022.
