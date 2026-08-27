/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link.Accumulation

/-!
# Dense Runtime Backward Pass Link

This file proves that the executable dense backward loop produced by graph-to-tape lowering agrees with
the proof-level `backpropAllCtx` semantics. It is the main bridge between the runtime tape engine
and the algebraic reverse-mode model.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Algebra

open Spec
open Tensor

namespace Graph

open Runtime
open Runtime.Autograd

/--
**Main runtime/link theorem**: running the runtime dense backward loop on a tape produced by
`lowerGraphToTape` matches the proved “full backpropagation” `backpropAllCtx`.

This is the formal statement that the executable engine implements the same reverse-mode
accumulation semantics as the proved tape model.
-/
theorem backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx {α : Type} {Δ : Type} [DecidableEq Shape]
  [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : _root_.TorchLean.TensorPack α Γ) (d0 : Δ)
    (seed : _root_.TorchLean.TensorPack α (Γ ++ ss)) :
    Runtime.Autograd.Tape.backwardDenseFrom (t := (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss)
      g x d0).1)
        (grads0 := _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ss) seed) =
      .ok (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ss)
        (backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d0 seed)) := by
  induction g with
  | nil =>
      -- Only leaf nodes; `backwardDenseFromLoop` does nothing because every leaf's `backward` is
      -- `[]`.
      -- (We still need to show the dense array length check passes and every per-node shape check
      -- passes.)
      have hnsize :
          Γ.length =
            (addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x).nodes.size
              := by
        simp [size_addLeaves, Runtime.Autograd.Tape.empty]

      -- Main loop fact for leaf tapes.
      have hloop :
          Runtime.Autograd.Tape.backwardDenseFromLoop
              (t := addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x)
              (addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x).nodes.size
              (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) (_root_.TorchLean.TensorPack.cast (α := α) (h := (List.append_nil Γ))
                seed)) =
            Except.ok
              (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ)
                (_root_.TorchLean.TensorPack.cast (α := α) (h := (List.append_nil Γ)) seed)) := by
        -- Put the tape in a convenient form: it's exactly the `leafNodeOfSomeTensor` image of
        -- `x.toShapeErasedArray`.
        let t :=
          addLeaves (α := α) (t := Runtime.Autograd.Tape.empty (α := α)) (Γ := Γ) x
        have hnodes :
            t.nodes =
              (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x).map (leafNodeOfSomeTensor (α := α)) := by
          -- `nodes_addLeaves` for an empty prefix tape.
          simp [t, nodes_addLeaves, Runtime.Autograd.Tape.empty]

        -- The loop just runs identity steps in reverse order.
        let seedArr :=
          _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) (_root_.TorchLean.TensorPack.cast (α := α) (h := List.append_nil Γ) seed)
        have htlen : t.nodes.size = Γ.length := by
          -- `t` is `addLeaves empty x`
          simpa [t] using hnsize.symm

        -- A small lemma: for any `n ≤ t.size`, the loop is the identity on the dense array.
        have loop_id :
            ∀ n, n ≤ t.nodes.size →
              Runtime.Autograd.Tape.backwardDenseFromLoop (t := t) n seedArr = Except.ok seedArr :=
                by
          intro n hnle
          induction n with
          | zero =>
              rfl
          | succ n ihn =>
              have hnlt : n < t.nodes.size :=
                Nat.lt_of_lt_of_le (Nat.lt_succ_self n) hnle
              have hnle' : n ≤ t.nodes.size :=
                Nat.le_trans (Nat.le_succ n) hnle
              have hidSeed : n < seedArr.size := by
                have : n < Γ.length := by simpa [htlen] using hnlt
                simpa [seedArr, _root_.TorchLean.TensorPack.size_toShapeErasedArray] using this
              have hidX : n < (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x).size := by
                have : n < Γ.length := by simpa [htlen] using hnlt
                simpa [_root_.TorchLean.TensorPack.size_toShapeErasedArray] using this

              -- Identify the node at `n`: it is a leaf node.
              have hnode :
                  t.getNode? n =
                    some
                      (leafNodeOfSomeTensor (α := α)
                        ((_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x)[n]'hidX)) := by
                simp [Runtime.Autograd.Tape.getNode?, hnodes, Array.getElem?_map, leafNodeOfSomeTensor,
                  Array.getElem?_eq_getElem (xs := _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x) (i := n)
                    hidX]

              -- Shape check at `n`: both entries have the same shape.
              have hshape :
                  (seedArr[n]'hidSeed).shape =
                    ((_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x)[n]'hidX).shape := by
                let i : Fin Γ.length := ⟨n, by
                  have : n < Γ.length := by simpa [htlen] using hnlt
                  exact this⟩
                have hx_s :
                    ((_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x)[n]'hidX).shape = Γ.get i := by
                  simpa [i, Spec.SomeTensor.ofTensor] using congrArg Spec.SomeTensor.shape
                    (_root_.TorchLean.TensorPack.get_toShapeErasedArray (α := α) (ss := Γ) x i)
                have hseed_s :
                    (seedArr[n]'hidSeed).shape = Γ.get i := by
                  simpa [seedArr, i, Spec.SomeTensor.ofTensor] using congrArg Spec.SomeTensor.shape
                    (_root_.TorchLean.TensorPack.get_toShapeErasedArray (α := α) (ss := Γ)
                      (_root_.TorchLean.TensorPack.cast (α := α) (h := List.append_nil Γ) seed) i)
                exact hseed_s.trans hx_s.symm

              have hstepn :
                  Runtime.Autograd.Tape.backwardDenseFromStep (t := t) seedArr n = Except.ok seedArr
                    := by
                have hidSeed0 : n < (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ []) seed).size := by
                  have : n < Γ.length := by simpa [htlen] using hnlt
                  simpa [_root_.TorchLean.TensorPack.size_toShapeErasedArray] using this
                -- unfold `backwardDenseFromStep`; `leafNodeOfSomeTensor.backward = []`, so the step is the
                -- identity.
                simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnode, leafNodeOfSomeTensor, seedArr,
                  Array.getElem?_eq_getElem (xs := (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ []) seed))
                    (i := n) hidSeed0]
                have hcond : seed.toShapeErasedArray[n].shape = x.toShapeErasedArray[n].shape := by
                  simpa [seedArr] using hshape
                -- `leafNodeOfSomeTensor.backward` contributes no parent gradients, so the fold is a no-op.
                -- After choosing the `if` branch via `hcond`, this is definitional.
                simp [hcond]
                rfl

              -- Unfold one loop iteration and use the IH.
              simpa [Runtime.Autograd.Tape.backwardDenseFromLoop, hstepn, Bind.bind, Except.bind,
                Pure.pure, Except.pure] using (ihn hnle')

        -- Specialize to `n = t.size`.
        simpa [t, seedArr] using loop_id t.nodes.size (le_rfl)

      -- Put it all together.
      -- `backpropAllCtx` is the identity in the nil case.
      -- Use the size check (`hnsize`) and rewrite away the `cast` on the seed array.
      simpa [lowerGraphToTape, backpropAllCtx, Runtime.Autograd.Tape.backwardDenseFrom, hnsize,
        _root_.TorchLean.TensorPack.toShapeErasedArray_cast] using hloop
  | snoc g node ih =>
      rename_i ssPrev τ
      -- Unpack the lowering of the prefix graph.
      rcases hprev : lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 with ⟨tPrev,
        ctxPrev⟩
      have hctxPrev :
          ctxPrev = Graph.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 := by
        simpa [hprev] using
          (lowerGraphToTape_ctx_eq_eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)
      have htPrevSize : tPrev.nodes.size = Γ.length + ssPrev.length := by
        simpa [hprev] using
          (lowerGraphToTape_nodes_size (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)

      -- Unpack the seed into `(seedPrev, seedOut)` matching the snoc structure.
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : _root_.TorchLean.TensorPack α ((Γ ++ ssPrev) ++ [τ]) := _root_.TorchLean.TensorPack.cast (α := α) (h := assoc.symm) seed
      let seedPrev : _root_.TorchLean.TensorPack α (Γ ++ ssPrev) :=
        (_root_.TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').1
      let seedOut : Tensor α τ :=
        (_root_.TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      have hseed' :
          _root_.TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedPrev seedOut = seed' := by
        simp [seedPrev, seedOut]

      -- The output gradient seed is the last entry of the dense array.
      let outValue : Spec.SomeTensor α := Spec.SomeTensor.ofTensor seedOut

      -- Define the runtime tape for the snoc graph explicitly.
      let y := node.forward ctxPrev d0
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-carrying-graph"
          value := Spec.SomeTensor.ofTensor y
          requiresGrad := true
          parents := #[]
          backward := fun dLdyValue => by
            if h : dLdyValue.shape = τ then
              let dLdy : Tensor α τ := dLdyValue.cast h
              let contribs := node.vjp ctxPrev d0 dLdy
              exact .ok (_root_.TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      let tNext : Runtime.Autograd.Tape α := (Runtime.Autograd.Tape.addNode (t := tPrev)
        runtimeNode).1
      have htNextNodes :
          tNext.nodes = tPrev.nodes.push runtimeNode := by
        simp [tNext, Runtime.Autograd.Tape.addNode]
      have htNextSize :
          tNext.nodes.size = tPrev.nodes.size + 1 := by
        simp [htNextNodes]

      -- Rewrite the initial gradient array as a push of the prefix part plus `seedOut`.
      have hseedArr :
          _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed =
            (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) seedPrev).push outValue := by
        -- First rewrite `seed` to the assoc-cast form `seed'`, then use `toShapeErasedArray_snoc`.
        have hcast :
            _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := (Γ ++ ssPrev) ++ [τ]) seed' =
              _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed := by
          simp [seed']
        -- Replace the LHS by `seed'`, then rewrite `seed'` as a `snoc`.
        rw [← hcast]
        -- `seed' = snoc seedPrev seedOut`
        have : seed' = _root_.TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedPrev seedOut := by
          simpa using hseed'.symm
        -- Now `toShapeErasedArray_snoc` gives the pushed array.
        simp [this, outValue, _root_.TorchLean.TensorPack.toShapeErasedArray_snoc]

      -- From here on we unfold `backwardDenseFrom` into the loop+step decomposition.
      -- `backpropAllCtx` peels off the last seed and recurses on the prefix graph.
      have hsizeCheck :
          (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed).size = tNext.nodes.size :=
            by
        -- LHS: `(Γ ++ ssPrev ++ [τ]).length`, RHS: `tPrev.size + 1`.
        simp [hseedArr, htNextSize, htPrevSize, _root_.TorchLean.TensorPack.size_toShapeErasedArray, Nat.add_assoc]

      -- Expand both sides (`lowerGraphToTape`, `backpropAllCtx`, and `backwardDenseFrom`) and reduce to:
      -- 1) one runtime step for the last node (adding `vjp` contributions to the prefix),
      -- 2) then the IH on the prefix graph.
      let ctx := Graph.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0
      let contrib := node.vjp ctx d0 seedOut
      let seedPrev' := _root_.TorchLean.TensorPack.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev := backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0 seedPrev'
      -- Now compute the runtime `backwardDenseFrom` explicitly:
      -- 1) run one step for the last node, adding `contrib` into the prefix grads,
      -- 2) run the prefix loop, which matches the IH on `g`,
      -- 3) the last gradient entry `seedOut` is never modified afterwards.

      -- Normalize `lowerGraphToTape` and `backpropAllCtx` to our explicit `tNext`/`seedPrev`
      -- decomposition.
      have hTape :
          (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev ++ [τ]) (.snoc (ss := ssPrev) (τ :=
            τ) g node) x d0).1 =
            tNext := by
        simp [lowerGraphToTape, hprev, tNext, y, runtimeNode]

      have hBackpropArr :
          _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ (ssPrev ++ [τ]))
              (backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev ++ [τ])
                (.snoc (ss := ssPrev) (τ := τ) g node) x d0 seed) =
            (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) gradsPrev).push outValue := by
        -- unfold `backpropAllCtx` at the snoc constructor and use `toShapeErasedArray_snoc`
        simp [backpropAllCtx, seed', seedPrev, seedOut, ctx, contrib, seedPrev', gradsPrev, outValue,
          _root_.TorchLean.TensorPack.toShapeErasedArray_cast, _root_.TorchLean.TensorPack.toShapeErasedArray_snoc]

      -- Reduce the main goal to the loop over `tNext`.
      -- After rewriting, the goal is:
      -- `tNext.backwardDenseFrom seedArr0 = .ok (gradsPrevArr.push outValue)`.
      -- The size check passes by `hsizeCheck`.
      have hmain :
          Runtime.Autograd.Tape.backwardDenseFrom (t := tNext)
              (grads0 := _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ (ssPrev ++ [τ])) seed) =
            .ok ((_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) gradsPrev).push outValue) := by
        -- Rewrite the initial array into the pushed form.
        -- Also rewrite `backwardDenseFrom` to its core loop using the size check.
        simp [Runtime.Autograd.Tape.backwardDenseFrom, hseedArr, htNextSize, htPrevSize]
        -- Remaining proof: compute the loop body.
        -- Set up convenient shorthands for the prefix size and gradient arrays.
        let n : Nat := tPrev.nodes.size
        let seedPrevArr : Array (Spec.SomeTensor α) :=
          _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) seedPrev
        let seedPrevArr' : Array (Spec.SomeTensor α) :=
          _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) seedPrev'
        let gradsPrevArr : Array (Spec.SomeTensor α) :=
          _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) gradsPrev

        -- A small helper: the prefix gradient array has size `n`.
        have hsizeSeedPrevArr : seedPrevArr.size = n := by
          simp [seedPrevArr, n, _root_.TorchLean.TensorPack.size_toShapeErasedArray, htPrevSize, List.length_append]

        -- First, compute the last-node step (`id = n`), which just adds `contrib` into the prefix
        -- grads.
        have hnodeLast : Runtime.Autograd.Tape.getNode? (t := tNext) n = some runtimeNode := by
          -- `tNext.nodes = tPrev.nodes.push runtimeNode`
          have : tNext.nodes[n]? = some runtimeNode := by
            simp [htNextNodes, n]
          simpa [Runtime.Autograd.Tape.getNode?, tNext] using this

        have haccLast : (seedPrevArr.push outValue)[n]? = some outValue := by
          have : (seedPrevArr.push outValue)[seedPrevArr.size]? = some outValue := by
            simp
          simpa [hsizeSeedPrevArr] using this

        have hshapeLast : outValue.shape = runtimeNode.value.shape := by
          rfl

        -- Show the `addGradAll` fold for the last node matches `_root_.TorchLean.TensorPack.add` on the prefix, leaving
        -- `[outValue]` untouched.
        have hreqAll :
            ∀ i (hi : i < tPrev.nodes.size), (tPrev.nodes[i]'hi).requiresGrad = true := by
          -- Unfold the `let t := ...` binder in `lowerGraphToTape_requires_grad_true` and rewrite by
          -- `hprev`.
          have hreq0 :
              ∀ i (hi : i < (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x
                d0).1.nodes.size),
                (((lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x
                  d0).1.nodes[i]'hi).requiresGrad = true) := by
            simpa using (lowerGraphToTape_requires_grad_true (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x
              d0)
          simpa [hprev] using hreq0

        have hnodes0 :
            ∀ i (hi : i < (Γ ++ ssPrev).length),
              let id := (0 : Nat) + i
              ∃ node : Runtime.Autograd.Node α,
                tNext.getNode? id = some node ∧ node.requiresGrad = true ∧
                  node.value.shape =
                    ((_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) seedPrev)[i]'(by
                        simpa [_root_.TorchLean.TensorPack.size_toShapeErasedArray] using hi)).shape := by
          intro i hi
          have hiT : i < tPrev.nodes.size := by
            -- `tPrev.nodes.size = (Γ ++ ssPrev).length`
            simpa [htPrevSize, List.length_append, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]
              using hi
          let nodeAt : Runtime.Autograd.Node α := tPrev.nodes[i]'hiT
          have hgetPrev : tPrev.getNode? i = some nodeAt := by
            simp [Runtime.Autograd.Tape.getNode?, nodeAt, Array.getElem?_eq_getElem (xs :=
              tPrev.nodes) (i := i) hiT]
          have hgetNext : tNext.getNode? i = some nodeAt := by
            -- index `< tPrev.nodes.size`, so `push` doesn't change it
            have : (tPrev.nodes.push runtimeNode)[i]? = some (tPrev.nodes[i]'hiT) := by
              simpa using (Array.getElem?_push_lt (xs := tPrev.nodes) (x := runtimeNode) hiT)
            simpa [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, nodeAt] using this
          have hreq : nodeAt.requiresGrad = true := by
            simpa [nodeAt] using hreqAll i hiT
          -- Shapes: both are the `i`th shape in `Γ ++ ssPrev`.
          have hseedShape :
              nodeAt.value.shape =
                ((_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) seedPrev)[i]'(by
                    simpa [_root_.TorchLean.TensorPack.size_toShapeErasedArray] using hi)).shape := by
            let fi : Fin (Γ ++ ssPrev).length := ⟨i, hi⟩
            -- `tPrev.nodes.map value = ctxPrev.toShapeErasedArray`
            have hvals :
                tPrev.nodes.map (fun nd => nd.value) =
                  _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) ctxPrev := by
              simpa [hprev] using
                (lowerGraphToTape_values_eq (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d0)
            have hvalOpt := congrArg (fun a => a[i]?) hvals
            -- Evaluate both sides at `i`.
            have hnodeVal :
                nodeAt.value = (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'(by
                  -- `i < ctxPrev.toShapeErasedArray.size` because it matches `tPrev.nodes.size`
                  simpa [_root_.TorchLean.TensorPack.size_toShapeErasedArray, htPrevSize, List.length_append, Nat.add_assoc] using
                    hiT) := by
              -- Left: map+index gives `some nodeAt.value`
              have hleft :
                  (tPrev.nodes.map (fun nd => nd.value))[i]? = some nodeAt.value := by
                have : tPrev.nodes[i]? = some nodeAt := by
                  simp [nodeAt, Array.getElem?_eq_getElem (xs := tPrev.nodes) (i := i) hiT]
                simp [Array.getElem?_map, this, nodeAt]
              -- Right: in-bounds `getElem?` is `some _`
              have hright :
                  (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]? =
                    some ((_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'(by
                      simpa [_root_.TorchLean.TensorPack.size_toShapeErasedArray, htPrevSize, List.length_append, Nat.add_assoc]
                        using hiT)) := by
                have hiCtx :
                    i < (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) ctxPrev).size := by
                  simpa [_root_.TorchLean.TensorPack.size_toShapeErasedArray, htPrevSize, List.length_append, Nat.add_assoc] using
                    hiT
                simp [Array.getElem?_eq_getElem (xs := (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++
                  ssPrev) ctxPrev)) (i := i) hiCtx]
              -- Combine and extract the value equality.
              have : some nodeAt.value =
                  some ((_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'(by
                    simpa [_root_.TorchLean.TensorPack.size_toShapeErasedArray, htPrevSize, List.length_append, Nat.add_assoc]
                      using hiT)) := by
                -- rewrite both sides of `hvalOpt` using `hleft`/`hright`
                simpa [hleft, hright] using hvalOpt
              simpa using congrArg (fun o => o.getD nodeAt.value) this
            have hnode_s :
                nodeAt.value.shape = (Γ ++ ssPrev).get fi := by
              have hiCtx :
                  i < (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) ctxPrev).size := by
                simpa [_root_.TorchLean.TensorPack.size_toShapeErasedArray, htPrevSize, List.length_append, Nat.add_assoc] using
                  hiT
              have hctx_s :
                  ((_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) ctxPrev)[i]'hiCtx).shape =
                    (Γ ++ ssPrev).get fi := by
                -- `ctxPrev.get fi : Tensor α ((Γ ++ ssPrev).get fi)`, so the RHS shape is
                -- definitional.
                simpa [fi, Spec.SomeTensor.ofTensor] using
                  congrArg Spec.SomeTensor.shape
                    (_root_.TorchLean.TensorPack.get_toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) ctxPrev fi)
              -- rewrite the LHS using `hnodeVal`
              simpa [hnodeVal] using hctx_s
            have hseed_s :
                ((_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) seedPrev)[i]'(by
                    simpa [_root_.TorchLean.TensorPack.size_toShapeErasedArray] using hi)).shape = (Γ ++ ssPrev).get fi := by
              simpa [fi, Spec.SomeTensor.ofTensor] using congrArg Spec.SomeTensor.shape
                (_root_.TorchLean.TensorPack.get_toShapeErasedArray (α := α) (ss := Γ ++ ssPrev) seedPrev fi)
            exact hnode_s.trans hseed_s.symm

          refine ⟨nodeAt, ?_, ?_, ?_⟩
          · simpa [Nat.zero_add] using hgetNext
          · exact hreq
          · simpa using hseedShape

        have hfoldLast :
              (_root_.TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := Γ ++ ssPrev) contrib 0).foldlM
                  (fun acc2 (pid, pg) =>
                    Runtime.Autograd.Tape.addGradAll (t := tNext) acc2 pid pg)
                  (seedPrevArr.push outValue) =
                .ok (seedPrevArr'.push outValue) := by
            -- Apply the generic fold lemma with `pref = #[]` and `suffix = #[outValue]`.
            have hfold :=
              foldlM_addGradAll_toIndexedShapeErasedArray_eq_add (α := α) (t := tNext)
                (pref := (#[] : Array (Spec.SomeTensor α)))
                (seed := seedPrev) (contrib := contrib) (suffix := #[outValue]) hnodes0
            -- Simplify the array concatenations and rewrite `seedPrev'`.
            simpa [seedPrevArr, seedPrevArr', seedPrev', Array.append_assoc, Array.append_empty,
              Array.empty_append, Array.append_singleton] using hfold

        have hstepLast :
              Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (seedPrevArr.push outValue) n =
                .ok (seedPrevArr'.push outValue) := by
            -- Unfold the step: pick out the last node, check shapes, run `backward`, then fold
            -- `addGradAll`.
            -- Eliminate the shape equality so all casts become definitional.
            cases hshapeLast
            have hreqLast : runtimeNode.requiresGrad = true := by
              rfl
            have hout : outValue.shape = τ := by
              rfl
            have hshapeNode : outValue.shape = runtimeNode.value.shape := by
              rfl
            have hbackLast :
                runtimeNode.backward
                    { shape := runtimeNode.value.shape
                      tensor := outValue.cast hshapeNode } =
                  .ok (_root_.TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := Γ ++ ssPrev) contrib 0) := by
              simp [runtimeNode, outValue, hctxPrev, ctx, contrib, Spec.SomeTensor.ofTensor]
            -- Unfold the step and reduce the control flow (`getNode?`, `requiresGrad`, `acc[id]?`,
            -- shape check).
            simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodeLast, hreqLast, haccLast]
            -- Select the shape-check branch, but keep `Tensor.cast_shape` folded so we can rewrite
            -- via `hbackLast`.
            simp [hshapeNode]
            -- Rewrite the `backward` call to its concrete list of contributions.
            rw [hbackLast]
            -- The remaining `foldlM` is exactly `hfoldLast`.
            simpa [seedPrevArr, seedPrevArr', outValue, Bind.bind, Except.bind, Pure.pure,
              Except.pure] using hfoldLast

        -- Run the remaining prefix loop (ids `< n`). This matches the IH on `g` and leaves `outValue`
        -- untouched.
        have ihPrev :
            Runtime.Autograd.Tape.backwardDenseFrom (t := tPrev) (grads0 := seedPrevArr') =
              .ok gradsPrevArr := by
          simpa [hprev, seedPrevArr', gradsPrevArr, gradsPrev] using (ih seedPrev')

        have hsizeSeedPrevArr' : seedPrevArr'.size = n := by
          simp [seedPrevArr', n, _root_.TorchLean.TensorPack.size_toShapeErasedArray, htPrevSize, List.length_append]

        have ihPrevLoop :
            Runtime.Autograd.Tape.backwardDenseFromLoop (t := tPrev) n seedPrevArr' =
              .ok gradsPrevArr := by
          have hsize : seedPrevArr'.size = tPrev.nodes.size := by
            simpa [n] using hsizeSeedPrevArr'
          simpa [Runtime.Autograd.Tape.backwardDenseFrom, hsize, n] using ihPrev

        -- Helper: `addGradAll` commutes with pushing an unused last slot.
        have haddGradAllPush :
            ∀ (acc : Array (Spec.SomeTensor α)) (hacc : acc.size = n)
              (pid : Nat) (pg : Spec.SomeTensor α),
              pid < n →
              Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outValue) pid pg =
                Except.map (fun a => a.push outValue)
                  (Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg) := by
          intro acc hacc pid pg hpid
          have hpidPrev : pid < tPrev.nodes.size := by
            simpa [n] using hpid
          let nodeAt : Runtime.Autograd.Node α := tPrev.nodes[pid]'hpidPrev
          have hnodePrev : Runtime.Autograd.Tape.getNode? (t := tPrev) pid = some nodeAt := by
            simp [Runtime.Autograd.Tape.getNode?, nodeAt,
              Array.getElem?_eq_getElem (xs := tPrev.nodes) (i := pid) hpidPrev]
          have hnodeNext : Runtime.Autograd.Tape.getNode? (t := tNext) pid = some nodeAt := by
            have : (tPrev.nodes.push runtimeNode)[pid]? = some (tPrev.nodes[pid]'hpidPrev) := by
              simpa using (Array.getElem?_push_lt (xs := tPrev.nodes) (x := runtimeNode) hpidPrev)
            simpa [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, nodeAt] using this

          have hpidAcc : pid < acc.size := by
            simpa [hacc] using hpid
          let existing : Spec.SomeTensor α := acc[pid]'hpidAcc
          have hgetPrev : acc[pid]? = some existing := by
            simp [existing]
          have hgetNext : (acc.push outValue)[pid]? = some existing := by
            simpa [existing] using (Array.getElem?_push_lt (xs := acc) (x := outValue) hpidAcc)

          cases hreq : nodeAt.requiresGrad with
          | false =>
              -- If the node does not require grad, `addGradAll` is the identity.
              have hlhs :
                  Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outValue) pid pg =
                    .ok (acc.push outValue) := by
                simp [Runtime.Autograd.Tape.addGradAll, hnodeNext, hreq, throw, throwThe,
                  MonadExceptOf.throw]
                rfl
              have hrhs :
                  Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg = .ok acc :=
                    by
                simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hreq, throw, throwThe,
                  MonadExceptOf.throw]
                rfl
              simp [hlhs, hrhs, Except.map]
          | true =>
              by_cases hshape : pg.shape = nodeAt.value.shape
              · by_cases hex : existing.shape = nodeAt.value.shape
                ·
                    have hcastPg (h : pg.shape = nodeAt.value.shape) :
                        pg.cast h = pg.cast hshape := by
                      rw [Subsingleton.elim h hshape]
                    have hcastExisting (h : existing.shape = nodeAt.value.shape) :
                        existing.cast h = existing.cast hex := by
                      rw [Subsingleton.elim h hex]
                    let pg' : Spec.SomeTensor α := Spec.SomeTensor.ofTensor (pg.cast hshape)
                    let existing' : Spec.SomeTensor α :=
                      Spec.SomeTensor.ofTensor (existing.cast hex)
                    have hidPrev : pid < acc.size := hpidAcc
                    have hidNext : pid < (acc.push outValue).size := by
                      simpa [Array.size_push] using Nat.lt_trans hidPrev (Nat.lt_succ_self acc.size)
                    cases hadd : Runtime.Autograd.SomeTensor.add existing' pg' with
                    | error e =>
                        have haddAny
                            (hex' : existing.shape = nodeAt.value.shape)
                            (hshape' : pg.shape = nodeAt.value.shape) :
                            Runtime.Autograd.SomeTensor.add
                                { shape := nodeAt.value.shape, tensor := existing.cast hex' }
                                { shape := nodeAt.value.shape, tensor := pg.cast hshape' } =
                              .error e := by
                          rw [hcastExisting hex', hcastPg hshape']
                          simpa only [existing', pg', Spec.SomeTensor.ofTensor] using hadd
                        have hprevRes :
                            Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg =
                              .error e := by
                          simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hreq, hshape, hgetPrev,
                            hex, hcastPg, hcastExisting, haddAny, throw, throwThe,
                            MonadExceptOf.throw]
                          rfl
                        have hnextRes :
                            Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outValue)
                              pid pg =
                              .error e := by
                          simp [Runtime.Autograd.Tape.addGradAll, hnodeNext, hreq, hshape, hgetNext,
                            hex, hcastPg, hcastExisting, haddAny, throw, throwThe,
                            MonadExceptOf.throw]
                          rfl
                        simp [hprevRes, hnextRes, Except.map]
                    | ok summed =>
                        have hadd' :
                            Runtime.Autograd.SomeTensor.add
                                (Spec.SomeTensor.ofTensor (existing.cast hex))
                                (Spec.SomeTensor.ofTensor (pg.cast hshape)) =
                              .ok summed := by
                          simpa only [existing', pg'] using hadd
                        have hpid_lt_succ : pid < acc.size + 1 :=
                          Nat.lt_trans hidPrev (Nat.lt_succ_self acc.size)
                        have hprevRes :
                            Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg =
                              .ok (acc.set pid summed (h := hidPrev)) := by
                          have hcond : acc[pid].shape = nodeAt.value.shape := by
                            simpa [existing] using hex
                          have haddAcc :
                              Runtime.Autograd.SomeTensor.add
                                  (Spec.SomeTensor.ofTensor (acc[pid].cast hcond))
                                  (Spec.SomeTensor.ofTensor (pg.cast hshape)) =
                                .ok summed := by
                            simpa [existing, hcond] using hadd'
                          have hcastAcc (h : acc[pid].shape = nodeAt.value.shape) :
                              acc[pid].cast h = acc[pid].cast hcond := by
                            rw [Subsingleton.elim h hcond]
                          have haddAny
                              (haccShape : acc[pid].shape = nodeAt.value.shape)
                              (hpgShape : pg.shape = nodeAt.value.shape) :
                              Runtime.Autograd.SomeTensor.add
                                  { shape := nodeAt.value.shape,
                                    tensor := acc[pid].cast haccShape }
                                  { shape := nodeAt.value.shape, tensor := pg.cast hpgShape } =
                                .ok summed := by
                            rw [hcastAcc haccShape, hcastPg hpgShape]
                            simpa only [Spec.SomeTensor.ofTensor] using haddAcc
                          simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hreq, hshape, hcond,
                            hcastPg, haddAny, hidPrev,
                            throw, throwThe, MonadExceptOf.throw]
                        have hnextRes :
                            Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outValue)
                              pid pg =
                              .ok ((acc.set pid summed (h := hidPrev)).push outValue) := by
                          have hget : (acc.push outValue)[pid] = acc[pid] := by
                            simpa using
                              (Array.getElem_push_lt (xs := acc) (x := outValue) (i := pid) hidPrev)
                          have hcond : acc[pid].shape = nodeAt.value.shape := by
                            simpa [existing] using hex
                          have haddAcc :
                              Runtime.Autograd.SomeTensor.add
                                  (Spec.SomeTensor.ofTensor (acc[pid].cast hcond))
                                  (Spec.SomeTensor.ofTensor (pg.cast hshape)) =
                                .ok summed := by
                            simpa [existing, hcond] using hadd'
                          have hcastAcc (h : acc[pid].shape = nodeAt.value.shape) :
                              acc[pid].cast h = acc[pid].cast hcond := by
                            rw [Subsingleton.elim h hcond]
                          have haddAny
                              (haccShape : acc[pid].shape = nodeAt.value.shape)
                              (hpgShape : pg.shape = nodeAt.value.shape) :
                              Runtime.Autograd.SomeTensor.add
                                  { shape := nodeAt.value.shape,
                                    tensor := acc[pid].cast haccShape }
                                  { shape := nodeAt.value.shape, tensor := pg.cast hpgShape } =
                                .ok summed := by
                            rw [hcastAcc haccShape, hcastPg hpgShape]
                            simpa only [Spec.SomeTensor.ofTensor] using haddAcc
                          simp [Runtime.Autograd.Tape.addGradAll, hnodeNext, hreq, hshape,
                            hcastPg, haddAny, hpid_lt_succ, Array.set_push, hidPrev,
                            hget, hcond, throw,
                              throwThe,
                            MonadExceptOf.throw]
                        simp [hprevRes, hnextRes, Except.map]
                · simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hnodeNext, hreq, hshape,
                  hgetPrev, hgetNext, hex,
                    Except.map, throw, throwThe, MonadExceptOf.throw]
              ·
                  have hprevRes :
                      Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc) pid pg =
                        .error "autograd: gradient contribution has wrong shape for parent" := by
                    simp [Runtime.Autograd.Tape.addGradAll, hnodePrev, hreq, hshape, throw,
                      throwThe,
                      MonadExceptOf.throw]
                  have hnextRes :
                      Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc.push outValue) pid
                        pg =
                        .error "autograd: gradient contribution has wrong shape for parent" := by
                    simp [Runtime.Autograd.Tape.addGradAll, hnodeNext, hreq, hshape, throw,
                      throwThe,
                      MonadExceptOf.throw]
                  simp [hprevRes, hnextRes, Except.map]

        -- Helper: `backwardDenseFromStep` commutes with pushing an unused last slot for ids `< n`.
        have hstepPush :
            ∀ (id : Nat) (hid : id < n) (acc : Array (Spec.SomeTensor α)),
              acc.size = n →
              Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (acc.push outValue) id =
                Except.map (fun a => a.push outValue)
                  (Runtime.Autograd.Tape.backwardDenseFromStep (t := tPrev) acc id) := by
          intro id hid acc hacc
          have hidPrev : id < tPrev.nodes.size := by
            simpa [n] using hid
          let nodeAt : Runtime.Autograd.Node α := tPrev.nodes[id]'hidPrev
          have hnodePrev : Runtime.Autograd.Tape.getNode? (t := tPrev) id = some nodeAt := by
            simp [Runtime.Autograd.Tape.getNode?, nodeAt,
              Array.getElem?_eq_getElem (xs := tPrev.nodes) (i := id) hidPrev]
          have hnodeNext : Runtime.Autograd.Tape.getNode? (t := tNext) id = some nodeAt := by
            have : (tPrev.nodes.push runtimeNode)[id]? = some (tPrev.nodes[id]'hidPrev) := by
              simpa using (Array.getElem?_push_lt (xs := tPrev.nodes) (x := runtimeNode) hidPrev)
            simpa [Runtime.Autograd.Tape.getNode?, tNext, htNextNodes, nodeAt] using this
          have hidAcc : id < acc.size := by
            simpa [hacc] using hid
          have hgetAcc : acc[id]? = some (acc[id]'hidAcc) := by
            simp
          have hgetAccPush : (acc.push outValue)[id]? = some (acc[id]'hidAcc) := by
            simpa using (Array.getElem?_push_lt (xs := acc) (x := outValue) hidAcc)
          cases hreq : nodeAt.requiresGrad with
          | false =>
              have hlhs :
                  Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (acc.push outValue) id =
                    .ok (acc.push outValue) := by
                simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodeNext, hreq, throw, throwThe,
                  MonadExceptOf.throw]
                rfl
              have hrhs :
                  Runtime.Autograd.Tape.backwardDenseFromStep (t := tPrev) acc id = .ok acc := by
                simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hreq, throw, throwThe,
                  MonadExceptOf.throw]
                rfl
              simp [hlhs, hrhs, Except.map]
          | true =>
              by_cases hshape : (acc[id]'hidAcc).shape = nodeAt.value.shape
              · -- shape ok, split on `backward` result
                have hcastDLdy (h : (acc[id]'hidAcc).shape = nodeAt.value.shape) :
                    (acc[id]'hidAcc).cast h = (acc[id]'hidAcc).cast hshape := by
                  rw [Subsingleton.elim h hshape]
                let dLdy : Spec.SomeTensor α :=
                  Spec.SomeTensor.ofTensor ((acc[id]'hidAcc).cast hshape)
                cases hback : nodeAt.backward dLdy with
                | error e =>
                    have hbackAny
                        (hshape' : (acc[id]'hidAcc).shape = nodeAt.value.shape) :
                        nodeAt.backward
                            { shape := nodeAt.value.shape,
                              tensor := (acc[id]'hidAcc).cast hshape' } =
                          .error e := by
                      rw [hcastDLdy hshape']
                      simpa only [dLdy, Spec.SomeTensor.ofTensor] using hback
                    simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                      hgetAcc, hgetAccPush,
                      hshape, hcastDLdy, hbackAny, Except.map, throw, throwThe,
                      MonadExceptOf.throw]
                    rfl
                | ok contribs =>
                    have hbackAny
                        (hshape' : (acc[id]'hidAcc).shape = nodeAt.value.shape) :
                        nodeAt.backward
                            { shape := nodeAt.value.shape,
                              tensor := (acc[id]'hidAcc).cast hshape' } =
                          .ok contribs := by
                      rw [hcastDLdy hshape']
                      simpa only [dLdy, Spec.SomeTensor.ofTensor] using hback
                    have hpids :
                          ∀ {pid : Nat} {pg : Spec.SomeTensor α}, (pid, pg) ∈ contribs → pid < id
                            := by
                        intro pid pg hmem
                        have hgetComp :
                            Runtime.Autograd.Tape.getNode? (t := (lowerGraphToTape (α := α) (Δ := Δ) (Γ :=
                              Γ) (ss := ssPrev) g x d0).1)
                                id =
                              some nodeAt := by
                          simpa [hprev] using hnodePrev
                        exact
                          lowerGraphToTape_backward_pids_lt_id (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g
                            x d0 id nodeAt hgetComp dLdy
                              contribs hback hmem

                    have hfold_push :
                        ∀ (cs : List (Nat × Spec.SomeTensor α))
                          (acc0 : Array (Spec.SomeTensor α)),
                          acc0.size = n →
                          (∀ {pid : Nat} {pg : Spec.SomeTensor α}, (pid, pg) ∈ cs → pid < n) →
                          cs.foldlM (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t :=
                            tNext) acc2 pid pg)
                              (acc0.push outValue) =
                            Except.map (fun a => a.push outValue)
                              (cs.foldlM
                                (fun acc2 (pid, pg) => Runtime.Autograd.Tape.addGradAll (t := tPrev)
                                  acc2 pid pg) acc0) := by
                      intro cs
                      induction cs with
                      | nil =>
                          intro acc0 _hsize _hpids
                          simp [List.foldlM, Except.map]
                          rfl
                      | cons hd tl ih =>
                          intro acc0 hsize hpids
                          rcases hd with ⟨pid, pg⟩
                          have hpid : pid < n := by
                            exact hpids (pid := pid) (pg := pg) (by simp)
                          have hadd :=
                            haddGradAllPush (acc := acc0) (hacc := hsize) (pid := pid) (pg := pg)
                              hpid
                          cases hret : Runtime.Autograd.Tape.addGradAll (t := tPrev) (grads := acc0)
                            pid pg with
                          | error e =>
                              -- both folds error at the first step
                              simp [List.foldlM, hret, hadd, Except.map]
                              rfl
                          | ok acc1 =>
                              have hret' :
                                  Runtime.Autograd.Tape.addGradAll (t := tNext) (grads := acc0.push
                                    outValue) pid pg =
                                    .ok (acc1.push outValue) := by
                                -- unfold `Except.map` in `hadd`
                                simpa [Except.map, hret] using hadd
                              have hsize1 : acc1.size = n := by
                                have := addGradAll_ok_size (t := tPrev) (grads := acc0) (id := pid)
                                  (g := pg)
                                  (grads' := acc1) (by simpa using hret)
                                simpa [hsize] using this
                              have hpids_tl :
                                  ∀ {pid : Nat} {pg : Spec.SomeTensor α}, (pid, pg) ∈ tl → pid < n
                                    := by
                                intro pid pg hmem
                                exact hpids (pid := pid) (pg := pg) (by simp [hmem])
                              have ih' :=
                                ih (acc0 := acc1) hsize1 hpids_tl
                              -- unfold the `foldlM` for the cons case on both sides
                              simpa [List.foldlM, hret, hret', Except.map, Bind.bind, Except.bind,
                                Pure.pure, Except.pure] using ih'

                    have hpids_n :
                        ∀ {pid : Nat} {pg : Spec.SomeTensor α}, (pid, pg) ∈ contribs → pid < n :=
                          by
                      intro pid pg hmem
                      exact Nat.lt_trans (hpids (pid := pid) (pg := pg) hmem) hid

                    have hpids_list :
                        ∀ {pid : Nat} {pg : Spec.SomeTensor α},
                          (pid, pg) ∈ contribs.toList → pid < n := by
                      intro pid pg hmem
                      exact hpids_n (by simpa using hmem)

                    -- Apply the fold lemma.
                    have hfold := hfold_push contribs.toList acc hacc hpids_list
                    -- Unfold the step and rewrite using the fold lemma.
                    simpa [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                      hgetAcc, hgetAccPush,
                      hshape, hcastDLdy, hbackAny, Except.map, throw, throwThe,
                      MonadExceptOf.throw,
                      Bind.bind, Except.bind, Pure.pure, Except.pure, Array.foldlM_toList] using
                        hfold
              · -- shape mismatch
                simp [Runtime.Autograd.Tape.backwardDenseFromStep, hnodePrev, hnodeNext, hreq,
                  hgetAcc, hgetAccPush, hshape,
                  Except.map, throw, throwThe, MonadExceptOf.throw]

        -- The loop itself commutes with pushing an unused last slot.
        have hloopPush :
              ∀ m (hm : m ≤ n) (acc : Array (Spec.SomeTensor α)),
                acc.size = n →
                Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) m (acc.push outValue) =
                  Except.map (fun a => a.push outValue)
                    (Runtime.Autograd.Tape.backwardDenseFromLoop (t := tPrev) m acc) := by
            intro m hm acc hacc
            induction m generalizing acc with
            | zero =>
                simp [Runtime.Autograd.Tape.backwardDenseFromLoop, Except.map]
                rfl
            | succ m ihm =>
                have hm' : m ≤ n := Nat.le_trans (Nat.le_succ m) hm
                have hmid : m < n := Nat.lt_of_lt_of_le (Nat.lt_succ_self m) hm
                have hstep := hstepPush (id := m) (hid := hmid) (acc := acc) hacc
                cases hret : Runtime.Autograd.Tape.backwardDenseFromStep (t := tPrev) acc m with
                | error e =>
                    -- both loops error on this step
                    simp [Runtime.Autograd.Tape.backwardDenseFromLoop, hret, hstep, Except.map]
                    rfl
                | ok acc1 =>
                    have hstep' :
                        Runtime.Autograd.Tape.backwardDenseFromStep (t := tNext) (acc.push outValue) m
                          =
                          .ok (acc1.push outValue) := by
                      simpa [Except.map, hret] using hstep
                    have hsize1 : acc1.size = n := by
                      have := backwardDenseFromStep_ok_size (t := tPrev) (acc := acc) (id := m)
                        (acc' := acc1)
                        (by simpa using hret)
                      simpa [hacc] using this
                    have ih' := ihm (acc := acc1) hm' hsize1
                    simpa [Runtime.Autograd.Tape.backwardDenseFromLoop, hret, hstep', Except.map,
                      Bind.bind, Except.bind, Pure.pure, Except.pure]
                      using ih'

        have hloopFinal :
            Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) n (seedPrevArr'.push outValue) =
              .ok (gradsPrevArr.push outValue) := by
          have h := hloopPush n (le_rfl) seedPrevArr' hsizeSeedPrevArr'
          simpa [ihPrevLoop, Except.map] using h

        -- Use `hstepLast` to reduce the initial step, then apply `hloopFinal`.
        have hloopAll :
            Runtime.Autograd.Tape.backwardDenseFromLoop (t := tNext) (n + 1) (seedPrevArr.push
              outValue) =
              .ok (gradsPrevArr.push outValue) := by
          -- Unfold the loop one step, then rewrite the step result via `hstepLast`.
          -- The remaining goal is exactly `hloopFinal`.
          simp [Runtime.Autograd.Tape.backwardDenseFromLoop, hstepLast]
          simpa [Bind.bind, Except.bind, Pure.pure, Except.pure] using hloopFinal
        simpa [n, htPrevSize, Nat.add_assoc] using hloopAll

      -- Finish: rewrite both sides back to the original statement.
      simpa [hTape, hBackpropArr] using hmain


end Graph

end Algebra
end Autograd
end Proofs
