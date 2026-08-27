---
# layout: home
---

<section class="home-intro">
  <figure class="home-overview">
    <a
      href="{{ '/assets/media/figures/torchlean-layout.png' | relative_url }}"
      aria-label="Open the full TorchLean system diagram">
      <img
        src="{{ '/assets/media/figures/torchlean-layout.png' | relative_url }}"
        alt="TorchLean overview: typed tensors, shared graph IR, autograd proofs, IEEE-754 semantics, certificate checking, PyTorch interoperability, CUDA providers, and model analysis."
        loading="eager" />
    </a>
    <figcaption>From a typed model to execution, analysis, and proof.</figcaption>
  </figure>

  <div class="home-intro-copy">
    <p>
      TorchLean is the first unified deep-learning framework built in Lean 4. It brings model
      construction, training, and formal reasoning into one library, so executable neural-network
      code and the mathematics used to study it do not become separate projects.
    </p>

    <p>
      You use it much like an ordinary ML library: define a model, load tensors, and train on CPU
      or GPU. Tensor shapes are part of the types, so incompatible layers and malformed operations
      are caught while the program is being written rather than during a training run.
    </p>

    <p>
      A fixed model can be recorded as a shape-indexed SSA graph for execution and differentiation.
      Supported forward programs can also be lowered to TorchLean's shared operation IR for
      verification and export. Theorems about derivatives are stated separately from these executable
      graph representations, and backend assumptions remain explicit.
    </p>
  </div>

  <div class="home-actions" aria-label="Primary links">
    <a class="primary-link" href="{{ '/blueprint/Introduction/' | relative_url }}">Start reading</a>
    <a class="secondary-link" href="{{ '/examples/' | relative_url }}">View examples</a>
    <a class="secondary-link" href="{{ '/performance/' | relative_url }}">Build performance</a>
    <a class="secondary-link" href="https://arxiv.org/abs/2602.22631">Read the paper</a>
  </div>
</section>

## Explore TorchLean

<div class="workflow-list">
  <a href="{{ '/blueprint/Runtime___-Autograd___-and-Interop/Differentiation-By-Example/' | relative_url }}">
    <span>01</span>
    <strong>Write and run models</strong>
    <em>Define typed tensors and models, then train them with Lean-native autograd.</em>
  </a>
  <a href="{{ '/blueprint/Semantics-and-Graphs/The-Canonical-Graph-IR/' | relative_url }}">
    <span>02</span>
    <strong>Lower to graph IR</strong>
    <em>Inspect operation nodes, shapes, payloads, semantics, and execution traces.</em>
  </a>
  <a href="{{ '/installation/#from-a-model-to-a-kernel' | relative_url }}">
    <span>03</span>
    <strong>Choose a backend</strong>
    <em>Run on CPU or CUDA, with explicit contracts for native and external providers.</em>
  </a>
  <a href="{{ '/blueprint/Verification-and-Certificates/' | relative_url }}">
    <span>04</span>
    <strong>Check verification artifacts</strong>
    <em>Replay robustness bounds and certificates against their Lean predicates.</em>
  </a>
  <a href="{{ '/examples/bug-zoo/' | relative_url }}">
    <span>05</span>
    <strong>Turn bugs into contracts</strong>
    <em>See causal masks, stable losses, normalization, and KV-cache bugs reduced to precise claims.</em>
  </a>
</div>
