---
title: Tools
---

# Tools

<div class="tool-directory">
  <article class="tool-card">
    <p class="tool-kind">Profiling</p>
    <h2>LeanProfiler</h2>
    <p>
      LeanProfiler records nested <code>IO</code> spans, structured runtime metadata, Lean
      heartbeats, and process resource counters. It writes a Trace Event file for Perfetto and a
      JSON summary that can be compared with an earlier run.
    </p>
    <p>
      The TorchLean example wraps an existing model runner. Device completion hooks can include
      synchronized work in a span, and the report labels that measurement as
      completion latency rather than device-kernel time.
    </p>
    <nav class="tool-links" aria-label="LeanProfiler links">
      <a href="https://github.com/lean-dojo/LeanProfiler">Repository</a>
      <a href="https://github.com/lean-dojo/LeanProfiler#add-it-to-a-project">Getting started</a>
      <a href="https://github.com/lean-dojo/LeanProfiler/tree/main/guide">Guide source</a>
    </nav>
  </article>

  <article class="tool-card">
    <p class="tool-kind">Checked case studies</p>
    <h2>TorchLean Verified Examples</h2>
    <p>
      The current examples cover batch-invariant inference and a verifiable transformer
      checkpoint. Each README says what Lean checks and what still depends on Python, CUDA, solver
      output, or hardware.
    </p>
    <nav class="tool-links" aria-label="TorchLean Verified Examples links">
      <a href="https://github.com/Robertboy18/TorchLean-Verified-Examples">
        Repository and overview
      </a>
      <a
        href="https://github.com/Robertboy18/TorchLean-Verified-Examples/tree/master/week-01-batch-invariant-inference">
        Batch-invariant inference
      </a>
      <a
        href="https://github.com/Robertboy18/TorchLean-Verified-Examples/tree/master/week-02-verifiable-transformer-checkpoint">
        Verifiable transformer checkpoint
      </a>
    </nav>
  </article>
</div>
