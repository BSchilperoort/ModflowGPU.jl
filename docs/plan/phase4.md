# Phase 4: solver performance

Task cards for this phase of [PLAN.md](../../PLAN.md). Read PLAN.md sections 1–4 and 6 for the goals, conventions, architecture and validation rules these cards refer to.

Size: S ≈ < 1 day of agent work, M ≈ 1–3 days, L ≈ > 3 days.
`‖` marks cards that can run in parallel with their siblings.

**Phase 4 targets** (adopted 2026-11-08 from the T4.1 baseline,
`benchmark/results/2026-11-07.md`; rocm, RX 7800 XT, tight closures):
- `global5m`: steady state < 5 s in `:double`; daily step < 0.2 s in `:double` and
  < 0.1 s in `:single`; steady state + 30 daily steps < 10 s.
- `indus`: steady state < 0.35 s (mf6 serial speed), < 40 µs per inner iteration.
- In-solver SpMV ≥ 70 % of the T0.4 triad in both precisions (≥ 85 % with SELL-C-σ
  or the DIS stencil).
- Assembly < 4 ms per outer iteration on `global5m`; period data < 10 ms per period.
- AMG device-memory overhead ≤ +60 % of the model's arrays.
Baseline for comparison: `global5m` steady 28.2 s / 9.17 s, daily step 0.618 s /
0.230 s, 31 steps 46.7 s / 16.1 s (`:double` / `:single`); `indus` steady 1.10 s.
Each Phase 4 card reports its effect on these numbers with `benchmark/run.jl`.

**Status after T4.5 (2026-11-08)**, default `:auto` = `:mixed` + `:mc_ilu0`, rocm
(benchmark/results/2026-11-08-T4.5.md). Met: `global5m` steady 2.92 s (`:double`
3.71 s); in-solver SpMV 76–77 % of triad (`:single` 70–72 %); indus steady 0.09 s;
period data 5.6 / 9.1 ms; AMG memory within +60 % except indus `:single` (+67 %).
Carried over to card T4.7 (at the end of this file):
- `global5m` daily step 0.263 s (`:double` 0.324 s, `:single` 0.120 s) vs < 0.2 / < 0.1 s:
  ~11 ms preconditioner setup per outer iteration against ~3 inner iterations.
- `global5m` 31 steps 10.8 s vs < 10 s.
- indus 88 µs per inner iteration (Jacobi 57 µs) vs < 40 µs: 8 kernels per CG
  iteration; needs single-pass reductions and the direction update fused into SpMV.
- Assembly 5.7 ms per outer iteration vs < 4 ms: fewer bytes per entry, or the package
  record kernels fused into the row kernel.

**Status after T4.7 (2026-11-16)**, rocm, MC-ILU0, global5m unless named
(benchmark/results/2026-11-16-T4.7.md; heads and iterations bitwise unchanged):
- daily step `:single` 0.086 s (< 0.1 s): **met**; steady + 30 steps 9.61 s `:double`,
  8.74 s `:mixed` (< 10 s): **met**.
- daily step `:double` 0.217 s (`:mixed` 0.201 s) vs < 0.2 s: **missed by 8 %**; the
  preconditioner setup is gone (5.2 → 0.86 ms per outer iteration), what is left is
  bandwidth-bound assembly, SpMV, MC-ILU0 application and update.
- indus 74 µs per inner iteration (Jacobi 47 µs) vs < 40 µs: **missed**; one CG
  iteration is 4 kernels and one synchronisation, ≈ 5 µs host time per launch and ≈ 18 µs
  to see completion, so < 40 µs needs fewer launches than KA offers (persistent kernels).
- assembly 4.98 ms per outer iteration (`:single` 3.4 ms) vs < 4 ms: **missed** in
  `:mixed`/`:double`; needs fewer bytes per entry (`condsat`/`Δr` without `jas`).

**T4.1 Benchmark harness (S)**
- Depends: T3.4.
- Deliver: `benchmark/run.jl` running `indus`, `continental`, `global5m`,
  steady state and 30 daily steps, on rocm and CPU, plus mf6 6.8.2 (serial) up to
  the size where it finishes in reasonable time. Record wall time per time step,
  outer and inner iterations, achieved GB/s in SpMV, peak device memory, and time
  split (assembly / SpMV / preconditioner / reductions / host transfers) with
  TimerOutputs, as mGV does. Write results to `benchmark/results/<date>.md`.
- Done when: the baseline table exists. Phase 4 targets are then set from it,
  for example a daily step of `global5m` in < 1 s and steady state in minutes.

**T4.2 Chebyshev and multicolour smoothers (M) ‖**
- Depends: T4.1.
- Deliver: `src/solver/precond/chebyshev.jl` (eigenvalue bounds from a few
  Lanczos/CG steps), `multicolor.jl` (red–black colouring for DIS by `(i+j+k)`
  parity; greedy colouring for general graphs on the host), and symmetric
  Gauss–Seidel / ILU0 per colour.
- Done when: inner iteration counts and wall time are reported vs Jacobi on all
  benchmark sizes. Keep whichever is fastest per case.

**T4.3 Algebraic multigrid preconditioner (L) ‖**
- Depends: T4.1.
- Deliver: `src/solver/precond/amg.jl`: smoothed-aggregation hierarchy built on
  the host (AlgebraicMultigrid.jl, or our own if its API does not fit), strength
  threshold tuned for strong K contrasts and the strong vertical coupling of
  2-layer models (try aggregating each vertical column first). V-cycle on the
  device with CSR SpMV, a Chebyshev or Jacobi smoother, and a direct or many-sweep
  coarsest level. Reuse the hierarchy across outer iterations and time steps,
  and rebuild only when the off-diagonals change (convertible/Newton) or
  convergence degrades past a threshold.
- Done when: `global5m` steady state converges to tight closure, and the
  iteration count is roughly independent of grid size (compare `indus` vs
  `continental` vs `global5m`).
- Also (from T2.8): ex-gwf-hwell-a in `:single` fails with Jacobi BiCGSTAB (step 58 of
  60 does not converge, `MFGPU_TEST_HWELL=1`, group `run`); `:double` passes (7.5e-10 m).
  It must pass in `:single` with the new preconditioner (T4.2 smoothers or T4.3 AMG).

**T4.4 `:mixed` preset and production default (M)**
- Depends: T4.3.
- Deliver: the `:mixed` preset (Float64 state and Krylov, Float32 AMG
  hierarchy and smoother data), and a benchmark of `:single`, `:mixed` and
  `:double` on all benchmark sizes. Measure wall time, memory, iterations, and
  the difference from `:double`, especially for the `global5m` steady state,
  where conditioning is worst.
- Done when: `:mixed` meets the `:double` parity tolerances, and
  `docs/precision.md` recommends a production default per backend class (with
  and without Float64) based on the numbers. `:auto` then follows that
  recommendation.

**T4.6 Period-data path (S) ‖ with T4.2, T4.3**
- Depends: T4.1.
- Read: `src/boundaries/framework.jl` (`apply_period!`, `_period_records!`,
  `_apply_multiplier!`, `_bound_offsets!`, `set_bound!`, `_upload_records!`),
  `src/stepping.jl` (mirror push at `prepare_solve`), the T4.1 results
  (64 ms per period on `global5m`, mostly RCH and RIV).
- Deliver:
  1. Structure fast path: when a new PERIOD block has the same cells as the block in
     force (always for array input; ModflowCoupling's RIV changes only stage), skip
     the cellid→reduced-node conversion, the highest-active search, the node→record
     map rebuild and the `nodelist`/`nodeptr`/`noderec` uploads; upload `bound` only.
     Value checks (`check_records`) still run.
  2. One fused, allocation-free host pass: multiplier, `to_offsets` in Float64, cast
     to `T`, into a preallocated staging buffer registered with `pin_memory!`, then
     one `copyto!`. Precision rule 1 is unchanged (offsets on the host in Float64).
     Optionally multithreaded.
  3. Dirty tracking in the stepping API: push only BOUND mirrors that were written.
  4. Optional: prepare the next period's file-based data while the device solves.
- Done when: period data on `global5m` < 10 ms per period (`benchmark/run.jl`), and
  heads, budgets and iteration counts are bitwise unchanged in both presets on CPU and
  rocm (run, stepping, synthetic, boundaries, riv_drn_evt groups).

**T4.5 Kernel fusion and matrix formats (M)**
- Depends: T4.2 or T4.3.
- Deliver: fused PCG/BiCGSTAB update kernels (axpy + dot in one pass),
  SpMV variants (CSR scalar vs vector, SELL-C-σ, and a matrix-free 7-point
  stencil for DIS), selected per grid type by benchmark.
- Done when: SpMV reaches ≥ 70 % of the T0.4 triad bandwidth on `global5m`,
  or the reason why not is documented.
  Also investigate why runs with the section timer enabled are 3–43 % faster than
  untimed runs (T4.1 open issue).

**T4.7 Carried-over Phase 4 targets (M), scheduled after T5.4**
- Depends: T4.5, T5.0 (measure on KA 0.10 only). Start it when the T5.4 timings show
  that ModflowGPU, not mGV, limits the coupled daily step. If mGV dominates, close this
  card by recording the four targets as dropped in the status block above.
- Deliver, each against its target in the status block:
  1. Daily step (`global5m` < 0.2 s in `:double`, < 0.1 s in `:single`; 31 steps
     < 10 s): remove the ~11 ms preconditioner setup per outer iteration, which now
     stands against ~3 inner iterations. Candidate: keep the MC-ILU0 factors while only
     the diagonal changes a little, and refactor when the inner iteration count grows.
  2. Latency (indus < 40 µs per inner iteration; now 8 kernels per CG iteration):
     single-pass reductions and the direction update fused into the SpMV. From the
     T3.1 report: fewer than 2 (CG) or 3 (BiCGSTAB) host synchronisations per iteration
     needs the Krylov scalars on the device with one host check per iteration. That
     changes where the scalars are rounded, so re-run the parity groups.
  3. Assembly (< 4 ms per outer iteration on `global5m`; now 5.7 ms): fewer bytes per
     entry, or the package record kernels fused into the row kernel.
  4. Small items from earlier reports, to do while in this code:
     - T4.4: no per-solve preconditioner switch. On the CPU backend, AMG for the steady
       state would save 3.4 of 17.2 s on `continental`.
     - T4.3: `AMGPreconditioner.fine::Any` and the per-cycle host allocation of the
       smoother coefficients; with DIAGONAL scaling the near-null space is fixed at
       build time; indus `:single` AMG memory is +67 % (target +60 %).
     - T4.6: `:single` period data at 9.1 ms has little margin (single-thread Float32
       cast of RCH, ~4 ms); mirrors compare the whole array per push (~1–2 ms for RIV
       on `global5m`; gone where T5.2 writes device arrays).
     - T6.4: the once-per-session info message in `krylov.jl` says "red–black" also
       for DISV/DISU models, which use greedy colouring.
     - T4.1: `report_mf6` throws a BoundsError if mf6 fails before any `Solving` line.
- Done when: `benchmark/run.jl` shows each target met, or the reason why not is
  documented in the status block; the parity groups pass on CPU and rocm in all presets.
