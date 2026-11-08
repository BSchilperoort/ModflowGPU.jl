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

**T4.5 Kernel fusion and matrix formats (M)**
- Depends: T4.2 or T4.3.
- Deliver: fused PCG/BiCGSTAB update kernels (axpy + dot in one pass),
  SpMV variants (CSR scalar vs vector, SELL-C-σ, and a matrix-free 7-point
  stencil for DIS), selected per grid type by benchmark.
- Done when: SpMV reaches ≥ 70 % of the T0.4 triad bandwidth on `global5m`,
  or the reason why not is documented.
  Also investigate why runs with the section timer enabled are 3–43 % faster than
  untimed runs (T4.1 open issue).
