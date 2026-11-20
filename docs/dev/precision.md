# Precision findings

Measured `:single` (Float32) vs `:double` (Float64) behaviour of the engine (PLAN.md
section 3.7). T2.8 adds the Float32 closures and the head and budget differences on every
parity model; this file starts with the assembly and residual findings of T2.6.

## Assembly and flux-form residual (T2.6)

Kernels: `assemble!` and `flux_residual!` (src/solver/csr.jl). The residual is computed
in flux form, `brhs − bdiag·h' − Σ C·(h'_m − h'_n + Δr)`, with compensated (Neumaier)
summation of the row in `T`. Test: test/parity/assembly_test.jl (group `assembly`).

**Universal residual test** with MF6 6.8.2 tight-closure heads (offsets formed in
Float64, then cast): max |res| / max Σ|terms| over all saved steps, and the bound
`eps(T)·max(Σ C·(|h'_m| + |h'_n| + |Δr|) + |bdiag·h'| + |brhs|) / max Σ|terms|` from
rounding the offset heads to `T`. CPU and ROCm (RX 7800 XT) give bitwise identical results.

| Model | nodes, steps | `:double` | `:single` | `:single` rounding bound |
| --- | --- | --- | --- | --- |
| ex-gwf-toth | 20000, 1 | 5.7e-13 | 4.6e-5 | 1.3e-4 |
| ex-gwf-hanic | 2601, 1 | 3.4e-11 | 2.1e-7 | 1.2e-6 |
| ex-gwf-hanir | 2601, 1 | 3.4e-11 | 2.1e-7 | 1.2e-6 |
| ex-gwf-hwell-a | 153164, 60 | 1.6e-12 | 8.0e-5 | 4.8e-4 |
| fhb_const (synthetic) | 30, 20 | 2.5e-12 | 7.2e-7 | 3.7e-6 |
| coupling_rde (synthetic) | 337, 10 | 8.0e-13 | 2.7e-7 | 8.1e-7 |

The `:double` values are MF6's own closure (the residual left by its inner solver). The
`:single` values are set by rounding the offset heads to Float32: the device assembly
reaches the same level as the host loop of T2.3 (toth 4.6e-5 then and now), and the
`< 1e-8` criterion of PLAN.md section 4 cannot be met in Float32, so `:single` passes
against the rounding bound.

**Why toth and hwell-a are large: the clamp in the reference heads.** Rule 1 sets
`r = clamp(STRT, bot, top)`. In confined cells with heads far above the cell top this
moves `r` away from the heads: toth has STRT 10000 m, cell tops from 50 m, heads
9984–11014 m, so `max |h'| = 10600 m`; hwell-a has STRT 50 m in cells 0–10 m thick,
`max |h'| = 49 m`, and its first time steps are very short (`delt` = 3.5e-6 d, so the
storage terms `rho·h'` are large). Measured on the first step in `:single` (scratch
experiment, not a test):

| r | toth max \|h'\| | toth residual (bound) | hwell-a max \|h'\| | hwell-a residual (bound) |
| --- | --- | --- | --- | --- |
| `clamp(STRT, bot, top)` (current) | 10600 | 4.6e-5 (1.3e-4) | 49 | 1.7e-9 (1.3e-7) |
| `STRT` (no clamp) | 1010 | 2.9e-6 (1.2e-5) | 0.21 | 3.2e-8 (1.6e-7) |
| MF6's converged heads | 0 | 4.1e-8 (1.2e-7) | 0 | 7.8e-8 (1.2e-7) |

Proposal for the owner (adopted after T2.6 and implemented in T2.8, see below): use
`r = STRT` without the clamp in ICELLTYPE 0 cells (the clamp only matters for
convertible cells, T6.1). Re-referencing `r` between stress periods was not adopted.

**Rule 1 as implemented (T2.8).** `reference_heads` (src/gwf/offsets.jl): `r = STRT` for
ICELLTYPE 0, `clamp(STRT, bot, top)` for convertible cells, and the clamp also for a
confined cell whose STRT is not finite or lies more than
`D = max(zmax − zmin, 1000 m)` outside the model's elevation range `[zmin, zmax]`
(`REFERENCE_FALLBACK_MIN_DISTANCE`; such values are sentinels, and the number of cells
is logged). None of the parity models triggers the fallback. Universal residual test,
re-measured (all saved steps, `:single`, rounding bound in brackets): toth 2.9e-6
(1.2e-5; was 4.6e-5), hwell-a 4.8e-6 (1.5e-4, 60 steps; was 8.0e-5), hanic/hanir 2.1e-7,
fhb_const 7.3e-7, coupling_rde 1.3e-7; `:double` unchanged.

**Cancellation test** (a column of 100 cells at 5000 m, 1 cm head differences, K = 3, a
GHB-like term on one cell): max |res_single − res_double| / max |res_double|:

| Form in Float32 | error | 1e-4 criterion |
| --- | --- | --- |
| flux form with offsets (rules 1 and 2) | 6.0e-6 | passes |
| flux form with absolute heads (no rule 1) | 6.3e-2 | fails |
| `b − A·h` with absolute heads (MF6's form) | 1.1e-1 | fails |

**Oracle vs libmf6** (first `solve`, `SLN_1/AMAT` and `b' = RHS − AMAT·r`): `:double`
amat is bitwise equal on all six models and `b'` within 3.6e-16 of
max(|RHS| + Σ|a·r|); `:single` within 7.1e-8 (amat, relative to max |a|) and 5.3e-8
(`b'`), i.e. one Float32 rounding.

## Linear solvers (T2.7)

Kernels: `src/solver/krylov.jl`. Vectors are in `Ts`, the Jacobi inverse diagonal in
`Tp`; every dot product and norm (`L2NORM0`, rho, the denominators, the max-abs `δ` and
residual) is a device reduction accumulated in `Ta = accumulation_type(backend, Ts)`:
Float64 when the backend has it (so also in `:single` on CPU and ROCm), else `Ts`. The
reduction is a pairwise tree in each 256-item work group, then one work group over the
group results; the scalars and the convergence test are Float64 on the host. Note for
T3.1 (no `double` in `:single` device IR): the reduction kernels of `:single` contain
Float64 by design (rule 4) on backends with Float64; the IR check must exempt them or
run with `Ta = Float32`.

Test: test/unit/krylov_test.jl (group `krylov`), random M-matrices (n = 500, about 6
neighbours per row), RELATIVE_RCLOSE 1e-11 (`:double`) or 1e-6 (`:single`); error
`max |x − x\| / max |x\|` (badly scaled: in the scaled variables `√|a_nn|·x`), CPU backend,
iterations in brackets:

| Case | `:double` | `:single` |
| --- | --- | --- |
| SPD, CG | 3.5e-11 (122) | 8.8e-7 (20) |
| SPD, MF6 sign (CG on −A) | 2.6e-11 (108) | 6.6e-7 (20) |
| SPD, rows scaled by e^N(0,1), CG + DIAGONAL | 8.3e-12 (63) | 2.8e-7 (20) |
| SPD, MF6 sign, BiCGSTAB | 4.4e-11 (29) | 7.8e-6 (13) |
| nonsymmetric, BiCGSTAB | 2.5e-11 (16) | 1.5e-6 (10) |
| nonsymmetric, MF6 sign, BiCGSTAB | 2.0e-11 (18) | 1.7e-6 (10) |
| nonsymmetric scaled, BiCGSTAB + DIAGONAL | 2.2e-12 (27) | 2.1e-7 (11) |
| nonsymmetric scaled, BiCGSTAB + L2NORM | 3.2e-12 (2000, see mf6_notes) | 1.2e-7 (11) |

The attainable relative residual of the recursively updated residual is about 1e-12 in
Float64 on these matrices (on one of them CG reaches 1e-9 in 27 iterations and needs
about 1000 for 1e-12, also in a host Float64 port), and about 1e-7 in Float32. In Float32 an absolute
INNER_RCLOSE below `eps(Float32)·‖A‖·‖δ‖` cannot be reached and the solve runs to
INNER_MAXIMUM; T2.8 sets the `:single` closures. ROCm (RX 7800 XT) and the CPU backend give
the same iteration counts in `:double` and `δ` within 1e-10.

Timing on the RX 7800 XT, 1e6 rows (2 × 500 × 1000 DIS grid, 7-point stencil), Jacobi:
CG 0.53 ms per iteration (`:double`) and 0.38 ms (`:single`); BiCGSTAB 1.05 ms and
0.69 ms. Each CG iteration has 2 host synchronisations (one 4-value copy each), BiCGSTAB 3.

## Outer iterations, closures and end-to-end parity (T2.8)

Kernels and host logic: src/solver/picard.jl (outer loop, correction solve with zero
initial guess, rule 3), src/budget.jl (flows in `T`, budget sums as device reductions in
Float64 where available, rule 4). Test: test/parity/run_test.jl (group `run`): every
saved step of our HDS and CBC files and every listing budget table against MF6 6.8.2
tight-closure runs (OUTER_DVCLOSE 1e-9, INNER_DVCLOSE 1e-10, INNER_RCLOSE 1e-10 or the
synthetic models' own tight IMS).

**Float32 closures** (`FLOAT32_CLOSURES`, applied by `PicardSettings` when `T = Float32`;
closures already looser are kept):

| Closure | Float32 floor | Why |
| --- | --- | --- |
| OUTER_DVCLOSE | `max(1e-6 m, eps32 · max_n \|h'_n\|)` (checked every outer iteration) | A Float32 offset head resolves `eps32·\|h'\|`; the correction of a converged state is noise of ~0.25 `eps32·max\|h'\|` (toth: `\|δ\|` stalls at 3.05e-5 m with offsets up to 1010 m, so a fixed 1e-5 never converges) |
| INNER_DVCLOSE | 1e-6 m; at least `eps32 · max_n \|h'_n\|` for the rest of the step once a correction meets OUTER_DVCLOSE but its inner solve did not converge (T5.3, T6.0, below) | the first inner correction of a converged Float32 state is rounding noise of that size |
| INNER_RCLOSE (absolute options) | `≥ eps32 · max_n Σ_m \|terms\|` of the first outer iteration of the step (×√n for the L2 options) | the rounding level of the flux-form residual (rule 2); T2.7: an absolute rclose below ~`eps32·‖A‖·‖δ‖` is unreachable |

Measured: the factor on the rclose floor sets the `:single` budget discrepancy (hanic: 10
→ 0.0055 %, 2 → 0.0016 %, 1 → 0.0008 %, 0.5 → same heads); 1 is used. OUTER_DVCLOSE
floors 1e-5 and 1e-6 both converge on every model; 1e-6 halves the `:single` head error
on coupling_rde (1.1e-5 → 1.5e-6 m vs `:double`) for 15 % more outer iterations.

**End-to-end parity** (CPU backend; ROCm on the RX 7800 XT gives bitwise identical heads
in both presets). Heads: max |Δh| over all saved steps vs MF6; CBC: max |Δq| over all
records (FLOW-JA-FACE, STO-SS, package lists) / total inflow of the step; listing: every
term's rate in and out (incl. the CHD split) beyond the printed rounding / total
inflow; pd: max |percent discrepancy| over the steps; iterations: outer/inner summed over
the steps (MF6 with ILU0, ours with Jacobi until Phase 4).

| Model | steps | `:double` max \|Δh\| | `:single` max \|Δh\| | `:single` vs `:double` (final) | CBC `:double` / `:single` | listing `:single` | pd `:single` | outer/inner `:double`, `:single`, MF6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ex-gwf-toth | 1 | 5.6e-9 | 4.9e-5 | 4.9e-5 | 3.4e-13 / 3.2e-7 | 1.3e-7 | 2.5e-5 % | 3/1247, 3/1435, 2/254 |
| ex-gwf-hanic | 1 | 1.5e-9 | 5.1e-7 | 4.6e-7 | 9.4e-10 / 4.0e-7 | 0 | 1.0e-3 % | 3/199, 3/146, 2/39 |
| ex-gwf-hanir | 1 | 1.4e-9 | 5.0e-7 | 5.3e-7 | 9.0e-10 / 4.9e-7 | 0 | 1.2e-3 % | 4/222, 3/151, 2/39 |
| fhb_const | 20 | 5.0e-11 | 5.9e-6 | 4.1e-7 | 1.6e-11 / 1.4e-6 | 6.7e-8 | 2.3e-4 % | 29/190, 28/149, 29/2251 |
| coupling_rde | 10 | 1.2e-11 | 1.2e-5 | 1.5e-6 | 9.2e-13 / 1.5e-7 | 7.7e-7 | 1.9e-4 % | 103/5722, 55/1101, 142/107439 |
| coupling_rde, SIMPLE 0.8 | 10 | 8.4e-11 | 3.8e-5 | 1.7e-5 | 1.4e-12 / 5.8e-7 | 3.4e-6 | 4.9e-4 % | 137/6402, 75/1825, 137/3807 |
| coupling_rde, COOLEY 0.5 | 10 | 1.3e-10 | 1.3e-5 | 1.7e-5 | 2.4e-12 / 1.7e-7 | 9.6e-7 | 2.2e-4 % | 94/4460, 58/1202, 94/5483 |
| coupling_rde, DBD | 10 | 1.5e-10 | 1.7e-5 | 8.5e-6 | 3.0e-12 / 3.4e-7 | 1.3e-6 | 3.4e-4 % | 112/5363, 66/1295, 113/5346 |
| coupling_rde, MODERATE | 10 | 1.1e-10 | 3.1e-5 | 6.3e-6 | 2.1e-12 / 4.5e-7 | 2.6e-6 | 5.1e-4 % | 94/4351, 55/1086, 94/1028 |
| coupling_rde, COMPLEX | 10 | 1.5e-10 | 2.2e-5 | 1.5e-5 | 3.0e-12 / 3.1e-7 | 1.5e-6 | 3.7e-4 % | 125/5750, 70/3548, 125/4019 |

The under-relaxation variants run with OUTER_DVCLOSE 1e-9, INNER_DVCLOSE 1e-11 and
INNER_RCLOSE 1e-9; their `:double` outer counts equal MF6's within one iteration.
The inner counts reflect Jacobi vs ILU0 and the relative denominator guard of the Krylov
solvers (`_safe_div`, docs/dev/mf6_notes.md): with MF6's absolute `DPREC` guard the
correction solves stalled at |r| ~ 1e-8 and hit INNER_MAXIMUM every outer iteration
(hanir `:double` 16/14402, coupling_rde 226/215792; MF6's own 107439 inner iterations on
coupling_rde show the same stall). Columns other than `:single` vs `:double` were
re-measured after that change. COMPLEXITY COMPLEX enables backtracking, which MF6
used in 3 of 125 outer iterations; the row above was measured before T6.2a, when
ModflowGPU did not backtrack (since T6.2a it backtracks in the same 3 iterations, see
"Backtracking (T6.2a)" below). In `:single` the outer loop stops at the Float32 floors,
hence fewer outer iterations than MF6.

**ex-gwf-hwell-a** (153164 cells, 60 steps, BiCGSTAB; part of the run parity test with
`preconditioner = :amg` since T4.3, where `:single` converges in every step, max |Δh|
3.0e-7 m, 596 inner iterations; `MFGPU_TEST_HWELL=0` skips it). With Jacobi, `:single` on ROCm: steps 1–56 converge in 2–15 outer iterations and
stay within 2e-5 m of MF6 (max |h'| 5.8 m); in step 57 (`delt` = 0.096 d) the Jacobi
BiCGSTAB solve does not converge in 1000 iterations and returns a correction of 3656 m
for a residual of 0.93, after which the outer iterations diverge (NaN by iteration 8).
Until T2.8's last fix the max-abs reduction skipped NaN and the step was reported as
converged; it now propagates NaN and the outer loop stops with a warning (and
`ConvergenceError`). `:double` on ROCm converges in all 60 steps (231 s at T2.8 merge time; 3.7 s after T3.1, see below; 2 outer and 12–30 inner iterations per step in the first steps), within 2.0e-10 m of MF6 in the last step, percent discrepancy 2.8e-8 %. The cure is the Phase 4 preconditioner (MF6 needs 2 outer
and 7–10 inner iterations per step with ILU0).

**Section 4 thresholds.** Confirmed for `:single` on Tier 0 and the synthetic models:
heads < 1e-3 m (largest measured 4.9e-5 m, toth, whose offsets reach 1010 m: the head
error scales with `eps32·max|h'|`), package flows < 1e-4 × total in (largest 3.4e-6),
percent discrepancy < 0.01 % (largest 1.5e-3 %), and `:single` vs `:double` heads
< 1e-3 m. The `:double` criteria hold with large margins (heads ≤ 4.0e-8 m, flows
≤ 1.0e-9 × total in). No threshold is changed; the measured values leave about a factor
20 on heads and 7 on percent discrepancy for the larger models of T3.4 and T5.3.

## Device execution (T3.1)

**Host synchronisations.** Every reduction (2 per CG iteration, 3 per BiCGSTAB iteration,
1 per outer iteration) launches two kernels, synchronises, and reads 4 values on the
host. Measured on the RX 7800 XT for a 360-row vector: kernels plus synchronisation
30 µs, the 4-value device-to-host `copyto!` (unpinned, staged) another 32 µs. The
reduction results now go to host memory registered with HIP (`host_mapped_zeros`, an
extension hook; the CPU backend aliases the same `Array`; backends without the hook keep
the copy; AMDGPU 2.8, which could not unregister wrapped memory, also copied until T5.0
raised the minimum to 2.11), so the copy is gone and a
reduction costs about 30 µs. The algorithms, the number of synchronisations and the
convergence checks per iteration are unchanged (MF6 semantics), and the results are
bitwise the same.

| `:double` on ROCm | before | after | outer / inner iterations |
| --- | --- | --- | --- |
| coupling_rde (360 cells, 10 steps) | 1.15 s (201 µs per inner it.) | 0.63 s (110 µs) | 103 / 5722 |
| ex-gwf-hwell-a (153164 cells, 60 steps) | 4.9 s (338 µs) | 3.7 s (254 µs) | 120 / 14532 |

`:single` coupling_rde: 0.24 s → 0.13 s. The remaining cost is launch latency (about
22 µs from launch to a completed synchronisation). Fewer synchronisations would need the
Krylov scalars (alpha, beta, omega) on the device, with one host check per iteration;
that changes where those scalars are rounded and is left open (Phase 4).

**Argmax.** `sln_get_dxmax` also returns `lrch`, the node of the largest change, for the
IMS convergence tables: the outer-iteration reduction carries the first index of the
largest magnitude (ties go to the lowest index, as MF6's sequential loop; 0 when every
change is zero), `OuterResult.lrch`. The tables themselves are not written yet.

**Kernel hygiene** (rule 5, test group `kernel_hygiene`, ROCm): 50 kernel
specialisations of the 19 kernels compiled by a `:single` workload; no `double` in the
optimised device IR except in the Float64-accumulating `mapreduce_kernel!` and
`finalize_reduction_kernel!`, and none at all when the backend reports no Float64
(`Ta = Float32`).

**Backends.** ROCm and the CPU backend give bitwise identical heads in both presets on
every Phase 2 parity model (run and assembly parity groups), well within the section 4
tolerances.

## Synthetic coupling models (T3.4)

Parity, device memory per preset and precision, and the `:single` BiCGSTAB + Jacobi
divergence on the synthetic models: docs/dev/scaling.md. The section 4 thresholds hold
unchanged (`:single` tiny 1.7e-4 m, indus 1.0e-4 m; pd <= 2.7e-3 %).

## `:mixed` preset and production default (T4.4)

`:mixed`: state, assembly and Krylov vectors in Float64 (`T = Ts = Float64`), every
preconditioner's data in Float32 (`Tp`: Jacobi `minv`, Chebyshev and multicolour operator
copies, the MC-ILU0 factors, the AMG levels below the finest one, their smoother data and
the dense coarsest inverse). The preconditioner kernels read `Tp` and compute in `Ts`
(`Ts(vals[k]) * x`); the Jacobi product is now formed in `Ts` (`x * Ts(minv[n])`; bitwise
unchanged for `Tp = Ts`). Rules 1–5 are unchanged; `:mixed` needs Float64 on the backend
(`resolve_precision` errors otherwise).

Measured with `benchmark/run.jl` (`benchmark/results/2026-11-08-T4.4.md`; rocm, steady
state + 30 daily steps, tight closures), `:mixed` vs `:double`:

| global5m, rocm | steady (s) | daily (s) | 31 steps (s) | peak GiB | Δh vs `:double` Jacobi (steady) |
| --- | --- | --- | --- | --- | --- |
| `:double` mc_ilu0 | 4.18 | 0.441 | 17.42 | 3.75 | 3.6e-9 m |
| `:mixed` mc_ilu0 | **3.36** | **0.386** | **14.94** | 3.56 | 3.6e-9 m |
| `:single` mc_ilu0 | 1.28 | 0.167 | 6.30 | 2.47 | 2.0e-6 m |
| `:mixed` amg | 4.52 | 0.754 | 27.13 | 4.28 | 3.7e-9 m |
| `:mixed` jacobi | 25.88 | 0.524 | 41.60 | 2.99 | 6.8e-13 m |

- Accuracy: `:mixed` has the outer and inner iteration counts of `:double` (all models and
  preconditioners; one exception, indus Jacobi 14292 vs 14306 inner) and heads at the
  closure level of `:double`, also in the global5m steady state (worst conditioning). The
  run parity tests (group `run`, Tier 0–2 and synthetic models) pass in `:mixed` with the
  `:double` tolerances; e.g. hwell-a (AMG) 9.03e-10 m, toth 5.57e-9 m, the same as
  `:double`.
- Speed: −20 % steady and −12 % daily on global5m with MC-ILU0, −16 % over 31 steps with AMG; Jacobi
  gains little (−2 to −7 %). CPU backend: within ±7 % of `:double`.
- Preconditioners: MC-ILU0 is the fastest over 31 steps on every model, backend and
  precision, and in the daily steps; AMG wins only the steady state of indus and
  continental (CPU continental `:mixed` 3.36 vs 6.76 s). A per-solve switch (AMG for
  steady periods) would gain at most 0.2 s on rocm continental and nothing on global5m,
  at +50 % memory for a second preconditioner: not adopted (on the CPU backend it would
  save 3.4 of 17.2 s on continental; an open option for CPU runs with long steady states).

**Production defaults** (`precision = :auto`, `preconditioner = :auto`, the defaults of
`ModflowGPU.run` and `initialize`):

| Backend class | precision | preconditioner | Why |
| --- | --- | --- | --- |
| with Float64 (CPU, ROCm, CUDA; oneAPI/OpenCL devices with fp64) | `:mixed` | `:mc_ilu0` | `:double` accuracy and parity tolerances; on GPU faster and smaller than `:double` (CPU: within ±7 %) |
| without Float64 (Metal; oneAPI/OpenCL without fp64) | `:single` | `:mc_ilu0` | fastest everywhere; converges on hwell-a in `:single` where Jacobi BiCGSTAB diverges (T4.2) |

`:single` on a Float64 backend is the opt-in fast mode (global5m 2.4× faster than `:mixed`
over 31 steps, heads within 2e-6–2e-4 m of `:double` on the synthetic models, i.e. inside
the section 4 `:single` tolerance of 1e-3 m). `:amg` remains the choice for badly
conditioned steady states where MC-ILU0 needs many iterations (e.g. on the CPU backend).
IMS ILU0 maps to `:mc_ilu0` with a logged message; MILU0 / ILUT / MILUT with a warning
(their relaxation and fill levels are not used).

## Coupling parity on the Indus (T5.3)

Test: `test/coupling_parity.jl` in the patched ModflowCoupling (patch 0002; run with
`Pkg.test(test_args = ["parity"])`, `MODFLOWCOUPLING_BACKEND=rocm` for the GPU). The Indus
model (180 × 204 × 2, `period = "month"` 1979-01 to 1979-03 and `period = "day"`
1979-01-01 to 1979-01-31, from the global steady state, synthetic forcing as in the
ModflowCoupling tests) with both engines; every period compared. Heads: max |Δh| over the
active cells; `baseflow_to_routing` and `capillary_rise`: max |Δ| over the land cells
divided by the domain maximum of the mf6 field in that period.

**Tight closures.** OUTER_DVCLOSE 1e-9, INNER_DVCLOSE 1e-10 and INNER_RCLOSE **1e-8**
m³/d (STRICT), OUTER_MAXIMUM 500, INNER_MAXIMUM 2000, for both engines. MF6 does not
converge on the Indus with INNER_RCLOSE 1e-10 (section 4's value): its residual does not
get below the rounding level of the large RIV and DRN terms; 1e-8 is the tightest power
of ten that converges (1e-2 to 1e-8 give the same MF6 heads to the printed digits). `:single` uses
the Float32 floors (`FLOAT32_CLOSURES`; outer 7e-6 to 9e-6 m from the offsets of up to
79 m, rclose 19 to 42 m³/d).

| GPU engine | period | max \|Δh\| | baseflow | capillary rise | outer / inner it. | criterion (heads; fluxes) |
| --- | --- | --- | --- | --- | --- | --- |
| `:double` | 3 months | 6.9e-11 m | 2.2e-12 | 6.4e-14 | 25 / 87 | 1e-4 m; 1e-6 |
| `:mixed` | 3 months | 6.9e-11 m | 2.2e-12 | 6.4e-14 | 25 / 87 | 1e-4 m; 1e-6 |
| `:single` | 3 months | 4.6e-5 m | 1.8e-6 | 1.8e-7 | 20 / 53 | 1e-3 m; 1e-4 |
| `:double` | 31 days | 3.5e-11 m | 1.5e-12 | 4.1e-13 | 210 / 451 | 1e-4 m; 1e-6 |
| `:mixed` | 31 days | 3.5e-11 m | 1.5e-12 | 4.1e-13 | 210 / 451 | 1e-4 m; 1e-6 |
| `:single` | 31 days | 1.9e-5 m | 3.1e-6 | 2.9e-7 | 158 / 286 | 1e-3 m; 1e-4 |

The CPU backend and rocm (RX 7800 XT) give the same numbers to two digits and the same
iteration counts. Every section 4 coupling criterion passes, `:mixed` against the
`:double` ones, with margins of 1e6 (`:double`, `:mixed`) and 20 to 30 (`:single`).

**Fix found on the way (`:single`).** Before T5.3, `:single` with tight closures did not
converge on the Indus: the outer correction stalled at 1.45e-6 m, below the Float32
OUTER_DVCLOSE floor (7.4e-6 m = `eps32·max|h'|`, offsets up to 62 m), but above the fixed
INNER_DVCLOSE floor of 1e-6 m, so the correction solves stagnated after two iterations
without reporting convergence and the outer iterations ran to OUTER_MAXIMUM (MF6
requires inner convergence too, `sln_has_converged` after `imslinear_ap`). Now, in
Float32, when an outer correction is already within OUTER_DVCLOSE and only the inner test
failed, the next inner solves of the step use INNER_DVCLOSE ≥ OUTER_DVCLOSE
(`outer_iteration!`, src/solver/picard.jl; one extra outer iteration per period). A
general floor `INNER_DVCLOSE ≥ eps32·max|h'|` was tried first and rejected: on ex-gwf-toth
(offsets 1010 m, steady, slow CG) it stops the inner solves early and the head error
rises from 6.6e-5 to 1.4e-3 m. T6.0 merged this rule with its STRICT case into one: the
raise is now bounded by `eps32·max|h'|` instead of going to OUTER_DVCLOSE (see "Float32
INNER_DVCLOSE after a converged correction" in the T6.0 section).

**Production closures** (indus.toml: OUTER_DVCLOSE 5, INNER_DVCLOSE 5, INNER_RCLOSE 10),
GPU engine vs mf6 engine (logged, not tested):

| GPU engine | period | max \|Δh\| | baseflow | capillary rise |
| --- | --- | --- | --- | --- |
| `:double`, `:mixed` | 3 months | 1.9 m | 0.59 | 1.0e-8 |
| `:single` | 3 months | 1.9 m | 0.59 | 2.1e-7 |
| `:double`, `:mixed` | 31 days | 6.0e-7 m | 3.3e-8 | 3.9e-10 |
| `:single` | 31 days | 1.9e-5 m | 2.6e-6 | 2.6e-7 |

Monthly, MF6 stops up to 1.9 m from its tight heads (February), and the baseflow differs by 59 %
of its domain maximum in some cells; the GPU engine stops within 8.2e-6 m (`:mixed`). Daily, both
engines stop after the same first outer iteration (the correction is below 5 m), so they
agree with each other, and both are 0.15 m from their tight heads.

### Closure table

Distance to the engine's own tight heads (max over periods and active cells) and median
wall time of a period (`BMI.update`, including the coupling's host work), after a warm-up
run; OUTER_DVCLOSE = INNER_DVCLOSE = the value in the column head, the other settings of
indus.toml. `MODFLOWCOUPLING_CLOSURE_TABLE=1`, run under the GPU benchmark lock;
2026-11-14, RX 7800 XT, mf6 = libmf6 6.3.0 (Modflow6_jll) on the CPU.

| engine | monthly tight | monthly 5 | monthly 0.5 | monthly 0.01 | daily tight | daily 5 | daily 0.5 | daily 0.01 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mf6 | 0.096 s | 1.9 m, 0.021 s | 3.0e-3 m, 0.050 s | 9.8e-10 m, 0.059 s | 0.059 s | 0.15 m, 0.011 s | 0.012 m, 0.019 s | 1.5e-4 m, 0.027 s |
| gpu `:single` rocm | 0.0061 s | 3.0e-3 m, 0.0036 s | 3.0e-3 m, 0.0041 s | 9.5e-7 m, 0.0046 s | 0.0045 s | 0.15 m, 0.0028 s | 0.012 m, 0.0032 s | 1.5e-4 m, 0.0038 s |
| gpu `:mixed` rocm | 0.0087 s | 8.2e-6 m, 0.0039 s | 5.0e-7 m, 0.0042 s | 4.0e-7 m, 0.0047 s | 0.0060 s | 0.15 m, 0.0028 s | 0.012 m, 0.0032 s | 1.5e-4 m, 0.0039 s |
| gpu `:double` rocm | 0.0079 s | 8.2e-6 m, 0.0037 s | 5.0e-7 m, 0.0042 s | 4.0e-7 m, 0.0045 s | 0.0056 s | 0.15 m, 0.0027 s | 0.012 m, 0.0031 s | 1.5e-4 m, 0.0035 s |
| gpu `:single` CPU | 0.048 s | 3.0e-3 m, 0.018 s | 3.0e-3 m, 0.025 s | 9.5e-7 m, 0.031 s | 0.029 s | 0.15 m, 0.0082 s | 0.012 m, 0.013 s | 1.5e-4 m, 0.018 s |
| gpu `:mixed` CPU | 0.064 s | 8.2e-6 m, 0.018 s | 5.0e-7 m, 0.024 s | 4.0e-7 m, 0.030 s | 0.043 s | 0.15 m, 0.0081 s | 0.012 m, 0.013 s | 1.5e-4 m, 0.019 s |
| gpu `:double` CPU | 0.068 s | 8.2e-6 m, 0.017 s | 5.0e-7 m, 0.025 s | 4.0e-7 m, 0.027 s | 0.040 s | 0.15 m, 0.0074 s | 0.012 m, 0.012 s | 1.5e-4 m, 0.017 s |

Notes. The timings are small (180 × 204 × 2 cells) and on rocm near the launch-latency
floor; the mf6 rows are the same in both runs (within 3 %). Daily, the distance is the
same for every engine: it is set by the outer iteration at which the loop stops, not by
the linear solver. At 0.01 the GPU engine stops 4e-7 m from its tight heads monthly,
MF6 1e-9 m; both are far below any coupling tolerance. `:single` at 5 and
0.5 stops 3e-3 m from its tight heads monthly, where `:mixed` stops 8e-6 m away: the
inner solves of `:single` end at the Float32 floors (INNER_RCLOSE ≥ 19 m³/d), probably
the cause (not investigated further).

### Proposal: production closures of the GPU engine (adopted 2026-11-14, PLAN.md section 8, decision 9; the GPU engine default since T5.4)

**Adopted.** Since T5.4 (ModflowCoupling patch 0004) `outer_dvclose` and `inner_dvclose`
default per engine in the patched `config.jl`: 0.01 m for the GPU engine, 5 m for the mf6
engine (`DEFAULT_DVCLOSE`, `dvclose(settings, :outer)`; a value in the config applies to
both engines), with the note in `examples/indus/indus.toml` that MF6 at 5 stops up to
1.9 m from its converged heads on monthly periods. The cost on the 1-year coupled run is
in the T5.4 section below. The proposal as written for T5.3:

1. **Use `outer_dvclose = inner_dvclose = 0.01` m for the GPU engine** (keep
   INNER_RCLOSE 10, the other settings unchanged). Measured on the Indus it is within
   1.5e-4 m (daily) and 4e-7 m (monthly) of the tight heads, against 0.15 m and 1.9 m at
   5, for +20 to +40 % wall time per period on rocm (daily 3.9 vs 2.8 ms, monthly 4.7 vs
   3.9 ms) and 1.7 to 2.3× the time on the CPU backend. The baseflow and capillary rise
   sent to the land-surface model then no longer depend on where the outer loop happened
   to stop, which matters for parity runs and for comparing engines.
2. Keep `5` as the mf6-engine default (the Python reference), but note in indus.toml
   that it stops up to 1.9 m from the converged heads on monthly periods; for mf6, 0.01
   costs 2.4× (daily) to 2.8× (monthly) the time.
3. Tight closures (1e-9) cost 2.1 to 2.2× the 5 m time on the GPU engine (rocm: 6.0 vs
   2.8 ms per day). At global size the ratio is the one to recheck (T5.4/T5.5a: global5m
   tight daily steps take 0.39 s, `:mixed` MC-ILU0); if 0.01 is within 10 % of 5 there
   as well, a tighter value (1e-3) is an option.

(T5.3 left the ModflowCoupling default at 5 for both engines; T5.4 made the per-engine
default above.)

## mGV and ModflowGPU on one GPU: the 1-year coupled Indus run (T5.4)

mGV `main` (42aebaa) and MODFLOW (ModflowCoupling patch 0004, `examples/mgv_coupling`) in
one process, daily exchange, 1979-01-01 to 1979-12-31 (365 days), Indus 180 × 204 × 2
from `steady_head_mgv_coupled.nc`, mGV's Indus inputs (GFDL-ESM4 1979, routing on,
`external_groundwater_routing = true`). mGV on rocm in every run; the GPU engine on rocm
with the device exchange (`set_value_device!` / `get_value_device!`, feedback applied to
mGV's state in kernels), the mf6 engine (libmf6 6.3.0 on the CPU) with the host exchange.
The device and host exchanges give bitwise equal heads, baseflow and capillary rise
(10 days, `:single` and `:double`). Tight closures as T5.3 (1e-9 / 1e-10 / INNER_RCLOSE
1e-8, STRICT).

**Coupled parity** (closed loop: each engine drives its own mGV), against the mf6 engine
at tight closures, max over the 365 days:

| GPU engine | max \|Δh\| | baseflow | capillary rise | outer / inner it. | T5.3 tolerance (heads; fluxes) |
| --- | --- | --- | --- | --- | --- |
| `:double` | 3.1e-7 m | **2.2e-6** | 1.2e-12 | 2274 / 4781 | 1e-4 m; 1e-6 |
| `:mixed` | 3.1e-7 m | **2.2e-6** | 1.2e-12 | 2274 / 4781 | 1e-4 m; 1e-6 |
| `:single` | 8.4e-6 m | 5.1e-6 | 5.2e-7 | 1545 / 2725 | 1e-3 m; 1e-4 |

The baseflow of `:double`/`:mixed` exceeds 1e-6 on 46 of 365 days (median 9e-12, first
31 days ≤ 7.9e-12 as in T5.3). The worst cell (day 301) differs by 6.2e-5 m³/s of 10.28
m³/s with |Δh| = 2.4e-7 m there: the coupled trajectories drift apart slowly through mGV's
state (channel storage, discharge, river stage). **Open loop** (the GPU engine driven by
the daily recharge and discharge the mf6 engine received in its coupled run, same
365 days; the mf6 engine replayed this way reproduces its run bitwise) the engines agree
to `:double`/`:mixed` 3.7e-10 m, baseflow 1.8e-11, capillary 1.2e-12 and `:single` 8.4e-6 m,
5.1e-6, 6.7e-7: every T5.3 tolerance holds with margins of 2.7e5 / 5.7e4 (`:double`) and
120 / 20 (`:single`). The closed-loop baseflow criterion is not met for `:double`/`:mixed`
because of that drift, not because of an engine difference.

**Closures and timings** (median wall time per coupled day from day 3, synchronised:
total, mGV part (feedback kernels + `BMI.update`), MODFLOW part (exchange + `BMI.update`);
the year total excludes day 1, which compiles; distance = max |Δh| to the same engine's
tight run over the year):

| engine | closure | distance to own tight | vs mf6 tight: \|Δh\|, baseflow | ms/day: total, mGV, MODFLOW | days 2–365 (s) | outer / inner it. |
| --- | --- | --- | --- | --- | --- | --- |
| mf6 | tight | – | – | 55.7, 3.3, 52.0 | 21.9 | – |
| mf6 | 5 (default) | 0.5 m | 0.5 m, 0.45 | 14.6, 3.2, 11.4 | 5.7 | – |
| gpu `:mixed` | tight | – | 3.1e-7 m, 2.2e-6 | 4.66, 1.70, 2.97 | 1.74 | 2274 / 4781 |
| gpu `:mixed` | 0.01 (default) | 3.6e-5 m | 3.6e-5 m, 8.3e-5 | 2.91, 1.70, 1.20 | 1.09 | 814 / 1263 |
| gpu `:mixed` | 5 | 0.5 m | 0.5 m, 0.45 | 2.51, 1.69, 0.82 | 1.29 | 365 / 365 |
| gpu `:single` | tight | – | 8.4e-6 m, 5.1e-6 | 3.55, 1.68, 1.85 | 1.32 | 1545 / 2725 |
| gpu `:single` | 0.01 (default) | 3.6e-5 m | 3.6e-5 m, 8.3e-5 | 2.86, 1.70, 1.15 | 1.08 | 814 / 1263 |
| gpu `:single` | 5 | 0.5 m | 0.5 m, 0.45 | 2.49, 1.70, 0.79 | 0.97 | 365 / 365 |
| gpu `:double` | tight | – | 3.1e-7 m, 2.2e-6 | 4.72, 1.67, 3.05 | 1.75 | 2274 / 4781 |
| gpu `:double` | 0.01 (default) | 3.6e-5 m | 3.6e-5 m, 8.3e-5 | 2.92, 1.68, 1.22 | 1.17 | 814 / 1263 |
| gpu `:double` | 5 | 0.5 m | 0.5 m, 0.45 | 2.52, 1.69, 0.83 | 0.93 | 365 / 365 |

**Cost of 0.01 against 5** (decision 9): the MODFLOW part takes 1.46× (`:mixed`
1.20 vs 0.82 ms, `:single` 1.45×, `:double` 1.47×), the whole coupled day 1.16× (2.91 vs
2.51 ms, mGV unchanged), for heads 3.6e-5 m instead of 0.5 m from the tight ones (2.2 outer
iterations per day instead of 1). Year totals (sums of the per-day times) are within the
run-to-run noise of a 1-second sum (1.09 vs 1.29 s `:mixed`, 1.17 vs 0.93 s `:double`), so
the medians are the numbers to use. At 0.01 every precision gives the same heads to the
printed digits (same iteration counts): the Float32 INNER_RCLOSE floor that made `:single`
stop 3e-3 m from its tight heads at 5 and 0.5 on monthly periods (T5.3) does not show at
0.01 on these daily steps, and the default precision of the coupled run is `:mixed`
(`:auto` on rocm), so it was not investigated further.

**Device memory** (`AMDGPU.used_memory`, the stream-ordered pool): mGV 63.9 MiB, MODFLOW
`:mixed` 22.8 MiB (`:single` 15.7, `:double` 23.7), both 86.7 MiB at start and 88.0 MiB at
the end of the year; after `BMI.finalize` of MODFLOW the pool is back to mGV's level plus
1.3 MiB (the example's own device maps and volume accumulators, still referenced).
`mGV.finalize!` does not free mGV's arrays (mGV closes its writer only).

**Float32 river stages.** In `:single` a RIV stage is stored as an offset from the
reference of its cell (the starting head). On the Indus 294 of 15165 RIV records have
|bottom_bankfull − r| > 1000 m (28 above 2000 m; max 2374 m at cell (136, 150): river
bottom 5689 m, head 3315 m): perched mountain rivers, where the leakage depends only on
stage' − rbot', both rounded to eps32/2 · 2374 m = 1.4e-4 m. Not fixed: a smaller
rounding needs a second per-record reference in the RIV formulation (or references at
the river level, which would move the rounding to the heads of those cells); the effect
is within the `:single` tolerances above.

**Transfers per daily step** (rocprofv3 7.16 HIP trace, `:single`, 3 profiled days after
35): two `hipMemcpyWithStream` host → device for the BMI inputs (`set_value`, 47–57 µs
each), and the third is the device-to-device `hold ← h` copy of `start_timestep!`
(2 µs of host time, no host transfer). With the device exchange the two uploads are
gone. The storage `rho` is copied host → device only when the step length changes
(monthly periods); see the T5.4 benchmark file for its cost at global size.

## Confined lgrv examples (T6.0)

ex-gwf-lgrv-gc (9×61×49, 26901 cells) and ex-gwf-lgrv-gr (25×183×147, 672525 cells):
ICELLTYPE 0, one steady period, RIV, RCHA, K33OVERK; IMS CG, ILU0, `INNER_RCLOSE ...
STRICT`. Both are in the assembly and run parity tests (`parity_examples`, test/input_files.jl;
`MFGPU_TEST_LGRV_GR=0` skips -gr, which adds about 2.5 min to the CPU suite; never in CI, which
has no `reference/`).

**Universal residual and oracle** (MF6 tight heads; CPU and ROCm alike): max |res| / max
Σ|terms| in `:single` (rounding bound) and `:double`; libmf6 first solve, `amat` and `b'`.

| Model | `:single` | `:double` | oracle `:double` amat / b' | oracle `:single` amat / b' |
| --- | --- | --- | --- | --- |
| ex-gwf-lgrv-gc | 1.84e-7 (6.08e-7) | 2.04e-13 | 0 / 1.2e-16 | 6.8e-8 / 5.0e-9 |
| ex-gwf-lgrv-gr | 2.61e-7 (8.26e-7) | 1.34e-12 | 0 / 1.0e-16 | 1.2e-7 / 5.3e-9 |

**Run parity** (max |Δh| vs MF6, tolerance `:double`/`:mixed` max(1e-6, 1e-7 × head
range) = 4.0e-6 m (gc) and 3.6e-6 m (gr), `:single` 1e-3 m; CBC max |Δq| / total in;
pd = |percent discrepancy|; listing 0 in every case; CPU backend times):

| Model, preset (preconditioner) | max \|Δh\| | CBC | pd | outer/inner (MF6 2/1001) | CPU time |
| --- | --- | --- | --- | --- | --- |
| gc `:single` (mc_ilu0) | 2.36e-5 | 7.1e-6 | 5.6e-4 % | 4/743 | 1.1 s |
| gc `:mixed` (mc_ilu0) | 3.06e-7 | 8.5e-11 | 5.9e-10 % | 2/566 | 1.0 s |
| gc `:double` (mc_ilu0) | 3.06e-7 | 8.5e-11 | 6.4e-10 % | 2/566 | 0.5 s |
| gr `:single` (amg) | 6.06e-6 | 2.5e-6 | 5.9e-4 % | 6/151 | 19.6 s |
| gr `:mixed` (mc_ilu0) | 2.32e-6 | 2.0e-10 | 2.6e-8 % | 5/3422 | 64 s |
| gr `:double` (mc_ilu0) | 2.32e-6 | 2.0e-10 | 2.6e-8 % | 5/3421 | 68 s |

ROCm (RX 7800 XT): the same iterations and bitwise the same heads as the CPU backend in
every row (max |Δh| = 0, also the assembled systems); run times gc 0.1 / 0.1 / 0.1 s and
gr 3.2 / 2.8 / 2.7 s (`:single` / `:mixed` / `:double`).

**The `:double` differences are MF6's.** Against the exact solution (a sparse Cholesky
solve of our Float64 system, residual 1e-15), MF6's tight heads are 3.1e-7 m (gc) and
2.3e-6 m (gr) off, ours in `:double` 3.5e-10 m and 3.5e-9 m. MF6 stops when one CG
iteration of the STRICT solve changes the heads by less than 1e-9 m; on gr its first
correction solve hits INNER_MAXIMUM (1000), and the remaining error is about 2300 times
the closure. gr therefore uses 64 % of its `:double` head tolerance; this is the reference,
not the engine.

**Float32 INNER_DVCLOSE after a converged correction** (src/solver/picard.jl
`outer_iteration!`; one rule for T5.3 and T6.0). The first inner correction of a converged
Float32 state is rounding noise of about eps32·max|h'| (the residual of rounded offset
heads), and an inner solve cannot get its per-iteration change below a fixed INNER_DVCLOSE
under that size. Two symptoms: on the Indus (T5.3, offsets up to 62 m) the correction
stalled at 1.45e-6 m and the inner solves ran to INNER_MAXIMUM; on ex-gwf-lgrv-gc
(offsets up to 49 m, `INNER_RCLOSE ... STRICT`, which accepts an inner solve only in its
first iteration, ImsLinearBase.f90 `ims_base_testcnvg`) the correction was 2–3e-6 m and
the run did not converge in 500 outer iterations.
- Trigger (Float32 only): from the second outer iteration of a step, the last correction
  met OUTER_DVCLOSE, so only the inner test kept the step from converging. It stays on for
  the rest of the step.
- Effect: INNER_DVCLOSE = max(INNER_DVCLOSE, eps32·max|h'| of the last outer iteration).
  The raise never goes above that noise level, so a loose OUTER_DVCLOSE (say 1e-2) no
  longer loosens the inner test (T5.3 raised it to OUTER_DVCLOSE; the T5.3 review's point).
- Not from the first outer iteration on: a head-resolution INNER_DVCLOSE in every inner
  solve stops slow CG solves early (ex-gwf-toth, offsets 1010 m: 1.4e-3 m instead of
  6.6e-5 m; found by both cards). Float64 (`:double`, `:mixed`) is unchanged.
- Measured with the merged rule (`:single`): Indus 4.6e-5 m monthly (20/53 iterations) and
  1.9e-5 m daily (158/286), the T5.3 numbers; lgrv-gc 2.36e-5 m in 4/743 iterations;
  ex-gwf-toth 6.63e-5 m, hanic 1.05e-6 m (10/1747), hanir 2.93e-7 m (4/78), the same as
  before T6.0. Unit tests: test/unit/picard_test.jl (the T5.3 case with a loose outer
  closure, and STRICT on lgrv-gc).

**ex-gwf-lgrv-gr in `:single` needs AMG.** With the default `:mc_ilu0` the Float32 run does
not converge in 500 outer iterations: the residual of a converged Float32 state (6.7e-8, the
exact heads rounded to Float32) is 4.4 times the STRICT rclose floor eps32·max Σ|terms| =
1.5e-8 (Σ|terms| is small where the offsets are large: 48 m with small flows). An rclose
floor from the head-rounding bound (`bound` of `flux_residual!`, eps32·max bound = 2.1e-7)
makes it converge in 3 outer iterations, but 6.1e-3 m from the exact heads: Float32 CG
with MC-ILU0 converges so slowly on this steady state that it stops on the 1e-6 m
per-iteration change while the error, a smooth mode, is about 5000 times larger, and the
residual cannot see it (7.8e-8 against 6.7e-8 for the rounded exact heads). Neither
INNER_MAXIMUM 10000, nor an INNER_DVCLOSE floor of 1e-7, nor NUMBER_ORTHOGONALIZATIONS 20
(true residual every 20 iterations) gets below 5.8e-3 m. AMG in Float32 reaches 6.7e-6 m
from the exact heads in 5 outer and 133 inner iterations (before the T5.0 merge; 6.06e-6 m
from MF6 in 6/151 now), and `:mixed` (Float64 Krylov, Float32 MC-ILU0) is
fine. That floor was therefore not adopted; the run parity test uses `:amg` for gr in
`:single` (as for ex-gwf-hwell-a), and the section 4 criteria are not met with the default
preconditioner in `:single` on this model. An automatic `:amg` choice for badly
conditioned steady states in `:single` was decided against in T5.5a (section below). The merged
INNER_DVCLOSE rule above does not change this (it does not touch the rclose side; with
`:mc_ilu0` the run still stops at 500 outer iterations).

## Global size: closures, the `:single` spin-up and the rclose floor (T5.5a)

Synthetic `global5m` (6.59 M nodes) through ModflowCoupling's GPU engine, coupled daily
with mGV's real global model for 1979 on rocm (`benchmark/results/2026-11-14-T5.5a.md`):

- **Closures** (decision 9): 5 and 0.01 cost the same here (2 outer / 3 inner iterations
  per day, MODFLOW 91.6 ms per day in `:mixed`); 1e-3 takes 128.8 ms (+41 %) for heads
  1.3e-5 m from those at 0.01. 0.01 stays the default; recheck on the real global model
  (T5.5b).
- **`:single` spin-up preconditioner.** Decision: `:auto` stays MC-ILU0, also for steady
  states in `:single`. On `global5m` the Float32 spin-up with MC-ILU0 takes 16.1 s
  (21 / 325) and ends 1.5e-5 m from `:double`; with AMG 41.5 s (16 / 87) and 9.4e-4 m.
  The lgrv-gr failure (above) shows as non-convergence, not as a wrong answer: run such a
  spin-up with `preconditioner = :amg`, or in `:mixed` (`:auto` where Float64 exists).
  On the Indus (real inputs, ModflowCoupling's spin-up settings, CPU backend) the Float32
  steady state stops about 0.03 m from the converged (`:double`, closure 1e-6) heads
  whatever the preconditioner and closure: at 0.01 MC-ILU0 0.072 m (42 / 14867, 22 s),
  AMG 0.032 m (34 / 16605, 46 s); at 1e-6 both 0.032 m (20886 and 31321 inner
  iterations); `:double` at 0.01 is 0.028 m away, `:mixed` at 1e-6 9.3e-10 m. **Spin-ups
  should run in `:mixed`**; `:single` (backends without Float64) gives centimetre heads on
  such steady states, beyond the 1e-3 m `:single` tolerance, and AMG does not change that.
- **Rclose floor.** Not changed: a floor from the head-rounding bound made lgrv-gr
  "converge" 6.1e-3 m from the exact heads (above), turning a visible failure into a
  silent error.

## Real global model against MODFLOW 6 (T5.6)

The real masked global model (60°N mask, 3,675,450 nodes, year-200 heads of T5.5b), 31
daily steps, open loop: mGV's daily discharge, recharge and channel volume recorded once
from the coupled GPU run and replayed to the mf6 engine (libmf6 6.3.0) and the GPU engine
from the same heads, river-loss limit on, tight closures (T5.3: 1e-9 / 1e-10 / INNER_RCLOSE
1e-8 STRICT) in both engines (`benchmark/results/2026-11-15-T5.6.md`). A replayed GPU run
reproduces the coupled run bitwise.

| GPU engine (rocm) | max \|Δh\| | baseflow | capillary rise | T5.3 tolerance |
| --- | --- | --- | --- | --- |
| `:mixed`, `:double` | 4.4e-11 m | 3.6e-12 | 5.3e-12 | 1e-4 m; 1e-6 |
| `:single` | 2.5e-5 m | 1.2e-5 | 3.9e-6 | 1e-3 m; 1e-4 |

Every T5.3 tolerance holds at global size, with margins 2.8e5 or more (`:mixed`,
`:double`) and 8 or more (`:single`). The differences grow over the month (`:mixed`
3.6e-12 m on day 1, 4.4e-11 m on day 31; `:single` 5.0e-6 to 2.5e-5 m). The largest
`:mixed` head differences are in deep water tables (164–508 m) in the Andes, the largest
`:single` ones in the lower Amazon, where the heads lie up to 105 m above the model top
(largest offsets from the per-cell reference). The budgets agree per term to 1e-12
(`:mixed`, `:double`) and 1e-6 (`:single`); MF6 converges in 7 outer / 14–15 inner
iterations per day, the GPU engine in 7 / 15 (`:single` 5 / 9 at its Float32 floors). The
river-loss limit gives the formula's fluxes in both engines (to 3.5e-7 relative) and no
loss above 0.999 of the channel volume, except in `:single` (117 record-days above the
volume in 31 days, 322 m³: the T5.5b review issue). At equal closures 0.01 the engines agree
to 4.4e-11 m, at 5 to 3.5e-7 m (both 0.51 m from the tight heads after 31 days).

## Time series: ex-gwf-fhb and ts_tas (T6.3)

Time-series values are evaluated on the host in Float64 per time step (TimeSeries.f90
`GetValue`, TimeArraySeries.f90 `GetAverageValues` over `[totimc, totimc + delt]`) and
go through the period-data staging pass (multiplier and offsets in Float64, one cast to
`T`), so they add no Float32 error of their own: a CHD head from a series reaches the
device as `T(h − r)`, as a constant head does. ex-gwf-fhb (3×10 cells, 20 steps; CHD
heads and WEL rates from LINEAREND series) replaces the `fhb_const` stand-in of the
tables above (the T2.x rows are kept as measured then); its reference is the
tight-relative MF6 run (the tight run does not converge in MF6). test/models/ts_tas
(5×10 cells, 15 steps; every TS6/TAS6 form, see docs/dev/support_matrix.md) is run with mf6
at test time.

**Universal residual and oracle** (CPU and ROCm bitwise identical): max |res| / max
Σ|terms| over all steps, `:single` (rounding bound) and `:double`; libmf6 first solve
`amat` / `b'`.

| Model | `:single` | `:double` | oracle `:single` amat / b' | oracle `:double` |
| --- | --- | --- | --- | --- |
| ex-gwf-fhb | 6.5e-7 (3.5e-6) | 9.5e-11 | 2.0e-8 / 1.2e-8 | 0 / 0 |
| ts_tas | 6.3e-7 (2.3e-6) | 2.5e-12 | 3.3e-8 / 2.5e-8 | 0 / 0 |

The ex-gwf-fhb `:double` residual (9.5e-11) is MF6's own closure of the tight-relative
run (RELATIVE_RCLOSE 1e-10). Storage residual test (group sto) on ex-gwf-fhb: STO-SS
max |Δq| / max(max |q|, total in) 1.5e-12 (`:single`) and 3.0e-21 (`:double`).

**Run parity** (default preconditioner `:mc_ilu0`; CPU, and ROCm with bitwise identical
heads in every preset). Heads: max |Δh| over all saved steps; CBC: max |Δq| / total in
over all records; listing beyond the printed rounding / total in; pd: max |percent
discrepancy|; outer/inner iterations (MF6 ILU0 in brackets).

| Model | preset | max \|Δh\| (m) | CBC | listing | pd | outer/inner (MF6) |
| --- | --- | --- | --- | --- | --- | --- |
| ex-gwf-fhb | `:single` | 3.9e-6 | 1.4e-6 | 7.4e-7 | 4.0e-4 % | 40/221 (40/300) |
| ex-gwf-fhb | `:mixed` | 4.4e-10 | 2.0e-10 | 0 | 4.5e-8 % | 40/259 |
| ex-gwf-fhb | `:double` | 2.3e-11 | 1.9e-10 | 0 | 5.6e-13 % | 40/220 |
| ts_tas | `:single` | 3.6e-7 | 3.4e-7 | 8.5e-8 | 2.0e-5 % | 38/347 (38/3126) |
| ts_tas | `:mixed` | 6.4e-12 | 1.4e-12 | 0 | 2.3e-10 % | 38/507 |
| ts_tas | `:double` | 3.4e-12 | 1.3e-12 | 0 | 9.1e-14 % | 38/460 |

`:single` vs `:double` final heads: ex-gwf-fhb 3.3e-7 m, ts_tas 3.6e-7 m. The outer
iteration counts equal MF6's in every preset.

## DISV and DISU: the Tier 5 examples and rad_disu_confined (T6.4)

Models: the DISV examples ex-gwf-u1disv (121 cells, BiCGSTAB, STRICT rclose),
ex-gwf-curve-90 (288) and ex-gwf-curvilin (864) (steady state, CHD, COMPLEXITY COMPLEX);
`rad_disu_confined` (ex-gwf-rad-disu with STO ICONVERT 0: DISU with vertices, 550 cells,
24 transient steps with TSMULT 1.62, five wells and no fixed heads; from MF6's
tight-relative reference inputs, mf6 run at test time); and test/models/ts_tas written as
DISV (TS6 and TAS6 on a vertex grid; test/unit/disv_disu_test.jl).

**Connectivity and conductances** against libmf6 6.8.2: IA, JA, JAS, ISYM, IHC and the
user-node IA/JA of the `.grb` files exact; CL1, CL2, HWVA, AREA, TOP, BOT, XC, YC and
CONDSAT bitwise equal on all four examples; ANGLEX within 1 ulp (2.2e-16, libm `atan2`).
Synthetic DISV/DISU grids with K22/ANGLE1–3 (the K ellipse in the connection direction
`anglex`): CONDSAT within 4.1e-16. The DISV and DISU descriptions of DIS grids give the
same connectivity and conductances as DIS, and with the Jacobi preconditioner bitwise the
same heads and budgets in every preset.

**Universal residual and oracle** (CPU and ROCm alike): max |res| / max Σ|terms|,
`:single` (rounding bound) and `:double`; libmf6 first solve, `amat` / `b'`.

| Model | `:single` | `:double` | oracle `:double` | oracle `:single` |
| --- | --- | --- | --- | --- |
| ex-gwf-u1disv | 3.26e-7 (2.34e-6) | 4.32e-13 | 0 / 0 | 2.7e-10 / 0 |
| ex-gwf-curve-90 | 1.08e-6 (8.61e-6) | 1.01e-10 | 0 / 1.1e-16 | 8.6e-8 / 8.1e-9 |
| ex-gwf-curvilin | 2.59e-6 (1.50e-5) | 7.35e-10 | 0 / 1.3e-16 | 8.4e-8 / 7.7e-9 |
| rad_disu_confined | 4.6e-4 (9.8e-2) | 9.06e-10 | 0 / 1.1e-16 | 3.4e-8 / 1.1e-16 |

The `:double` residuals of curvilin and rad_disu_confined (7e-10, 9e-10) are MF6's
closure: its heads are within 1e-9 m per iteration, not exact.

**Run parity** (max |Δh| vs MF6; CBC max |Δq| / total in; pd = |percent discrepancy|;
listing ≤ 1e-7 everywhere; default preconditioner except rad_disu_confined):

| Model, preset | max \|Δh\| | CBC | pd | outer/inner (MF6) |
| --- | --- | --- | --- | --- |
| u1disv `:single` / `:mixed` / `:double` | 1.45e-7 / 1.56e-12 / 1.6e-12 | 1.1e-7 / 5.6e-13 / 5.9e-13 | 4.4e-5 / 5.3e-10 / 5.1e-10 % | 2/14, 2/17, 2/17 (2/16) |
| curve-90 | 4.48e-6 / 3.77e-10 / 3.8e-10 | 2.0e-7 / 1.3e-11 / 1.3e-11 | 2.5e-4 / 4.6e-11 / 1.4e-10 % | 4/37, 2/33, 2/33 (2/18) |
| curvilin | 9.36e-6 / 1.47e-8 / 1.44e-8 | 1.3e-6 / 1.5e-9 / 1.5e-9 | 5.9e-4 / 1.7e-8 / 4.5e-8 % | 3/202, 3/83, 2/80 (2/36) |
| rad_disu_confined (amg) | 1.49e-6 / 6.17e-9 / 6.17e-9 | 1.9e-4 / 9.8e-10 / 9.8e-10 | 9.0e-6 / 1.0e-9 / 4.3e-10 % | 124/28256, 48/316, 48/324 (53/922) |
| ts_tas as DISV | 3.57e-7 / 6.41e-12 / 3.42e-12 | 3.4e-7 / 1.4e-12 / 1.3e-12 | 2.0e-5 / 2.3e-10 / 9.1e-14 % | 38/347, 38/507, 38/460 (38/3126) |

(heads in the model's length unit: rad_disu_confined is in feet.)

**rad_disu_confined needs AMG.** The late steps (Δt up to 3.8 d) of this radial grid
without fixed heads are nearly singular: with the default MC-ILU0 (greedy colouring)
BiCGSTAB, `:double` stalls at corrections of 1e-8–1e-7 ft from step 21 on and does not
converge in 500 outer iterations; AMG converges in 2 outer iterations per step, as MF6
(ILUT, COMPLEXITY COMPLEX) does. The run parity test uses `:amg` for it, like
ex-gwf-hwell-a.

**`:single` budget criterion loosened for rad_disu_confined** (5e-4 × total in instead of
1e-4; replaced in T6.1 by the rounding bound of the cell flows, see below). The conductances next to the wells reach 5.5e7 ft²/d and the offsets 26 ft, so one
Float32 FLOW-JA-FACE value rounds to C·eps32·(|h'n| + |h'm|) = 81 ft³/d, 2e-2 of the total
inflow (4000 ft³/d); measured 1.9e-4, growing with the drawdown (steps 23–24). The heads
are within 1.5e-6 ft of MF6 and the percent discrepancy is 9e-6 %, so this is the
precision of Float32 flows through very large conductances, not of the solution. Use
`:mixed` where such cell-by-cell flows matter.

ROCm (RX 7800 XT): every row above is the same on ROCm, with the same iteration counts and
bitwise the same heads and assembled systems as the CPU backend (max |Δh| = 0).

## Convertible cells, wetting and drying (T6.1)

Models: the Tier 3 examples ex-gwf-twri01 (VARIABLECV DEWATERED, PERCHED, CG, STRICT),
ex-gwf-bcf2ss-p01a (REWET, RIV, CG), ex-gwf-bump-p01b (REWET, 584 of 2601 cells dry),
ex-gwf-nwt-p02b (14 layers, REWET IHDWET 1, transient, 11778 of 22400 cells dry); the Tier 2
examples that need convertible cells, ex-gwf-advtidal (convertible storage, TS6, EVT NSEG
3, 361 steps), ex-gwf-maw-p03a and -p03c (convertible layer 1); ex-gwf-rad-disu (DISU,
convertible storage; replaces nothing: `rad_disu_confined` stays); and the synthetic
models (mf6 run at test time) test/models/wetdry (2×1×12: every layer-1 cell dries in the
steady period, RCH moves to layer 2, cells rewet in period 3; PERCHED, VARIABLECV
DEWATERED, convertible storage), test/models/disu_staggered (DISU, IHC 2, convertible) and
test/models/afr (WEL AUTO_FLOW_REDUCE). References: tight runs, tight-relative for
advtidal, bcf2ss-p01a, maw-p03c and rad-disu (their tight runs do not converge in MF6).

**Universal residual and oracle** (convertible models through `GwfModel`: MF6's heads and
dry cells in, `formulate!`; only steps whose previous step is saved, nwt-p02b saves the
last step of each period: 1 step). `:single` with the rounding bound in parentheses.

| Model | `:single` | `:double` | oracle `:double` amat / b' | oracle `:single` |
| --- | --- | --- | --- | --- |
| ex-gwf-twri01 | 5.8e-8 (3.0e-7) | 8.8e-13 | 0 / 0 | 5.4e-8 / 2e-17 |
| ex-gwf-advtidal (361 steps) | 1.1e-7 (4.4e-7) | 9.3e-13 | 0 / 5e-17 | 1.6e-8 / 4.6e-9 |
| ex-gwf-bcf2ss-p01a | 4.5e-7 (2.8e-6) | 1.8e-12 | 0 / 0 | 0 / 6e-35 |
| ex-gwf-bump-p01b | 3.0e-6 (9.8e-6) | 7.2e-11 | 1.2e-16 / 6e-44 | 1.2e-7 / 3e-36 |
| ex-gwf-nwt-p02b | 1.6e-6 (7.0e-6) | 7.7e-13 | 0 / 7e-42 | 1.2e-8 / 7e-42 |
| ex-gwf-maw-p03a | 6.6e-6 (1.9e-5) | 1.4e-12 | 0 / 1.9e-17 | 0 / 1.7e-10 |
| ex-gwf-maw-p03c | 4.4e-5 (2.2e-4) | **2.9e-8** | 0 / 5.8e-17 | 5.0e-8 / 1e-16 |
| ex-gwf-rad-disu (24 steps) | 4.7e-2 (2.8e-1) | 2.6e-10 | 0 / 9.3e-17 | 3.4e-8 / 1.2e-9 |
| wetdry (11 steps) | 6.9e-7 (2.3e-6) | 3.6e-12 | 0 / 2.3e-17 | 1.1e-8 / 2.6e-9 |
| disu_staggered | 9.4e-6 (7.4e-5) | 2.4e-11 | 0 / 4.1e-17 | 3.1e-8 / 4.1e-17 |

**ex-gwf-maw-p03c `:double`: criterion 1e-7 instead of 1e-8** (`UNIVERSAL_TOL`,
test/parity/assembly_test.jl). Its reference is the tight-relative run (INNER_RCLOSE 1e-10
RELATIVE_RCLOSE; the strict tight run does not converge), whose heads leave 2.9e-8 of the
row magnitudes; the formulation agrees with libmf6's first AMAT and RHS to 0 and 6e-17.
The `:single` residual of rad-disu (4.7e-2) is within its rounding bound: the storage
coefficients Sy·area/Δt of the first steps (Δt = 5.8e-5 d) multiply offsets of up to
26 ft.

**Run parity** (CPU; max |Δh| vs MF6; CBC max |Δq| / total in, `:single` beyond the
rounding bound below; pd = |percent discrepancy|, `:single` beyond the bound in
parentheses when not 0; listing ≤ 6.0e-6 everywhere; outer/inner, MF6 in parentheses).
Dry cells (HDRY) are the same as MF6's in every saved step of every preset.

| Model | max \|Δh\| `:single` / `:mixed` / `:double` | CBC | pd | outer/inner `:single`, `:mixed`, `:double` (MF6) |
| --- | --- | --- | --- | --- |
| twri01 | 1.19e-5 / 1.42e-8 / 1.42e-8 | 2.7e-9 / 8.1e-12 / 8.1e-12 | 4.7e-7 / 3.4e-10 / 3.4e-10 % | 11/554, 13/926, 13/928 (14/927) |
| advtidal | 5.97e-4 / 4.66e-10 / 4.66e-10 | 5.9e-7 / 9.2e-12 / 9.2e-12 | 1.9e-2 (0) / 4.9e-10 / 4.9e-10 % | 878/5063, 1110/8952, 1110/8952 (1103/6294) |
| bcf2ss-p01a | 2.55e-5 / 9.28e-10 / 9.28e-10 | 1.3e-8 / 4.7e-13 / 4.7e-13 | 6.9e-6 / 6.2e-10 / 6.2e-10 % | 26/513, 39/985, 39/981 (39/632) |
| bump-p01b | 5.67e-6 / 1.89e-9 / 1.89e-9 | 1.6e-7 / 6.7e-11 / 6.7e-11 | 7.4e-4 / 5.8e-9 / 5.8e-9 % | 12/517, 19/1112, 19/1112 (19/5379) |
| nwt-p02b | 1.98e-5 / 2.93e-10 / 2.92e-10 | – (no CBC) | 4.0e-4 / 1.3e-10 / 1.3e-10 % | 223/11037, 321/38435, 321/38246 (339/226156) |
| maw-p03a | 3.72e-5 / 2.81e-8 / 2.81e-8 | – (no CBC) | 5.9e-4 / 1.2e-9 / 1.0e-9 % | 4/827, 7/1729, 7/1706 (25/21764) |
| maw-p03c | 7.78e-6 / 1.72e-9 / 1.72e-9 | 7.7e-5 / 1.2e-7 / 9.7e-8 | 0.60 (0) / 8.9e-5 / 8.9e-5 % | 2/103, 2/228, 2/228 (2/77) |
| rad-disu (amg) | 2.96e-6 / 2.51e-10 / 2.51e-10 | 6e-16 / **6.0e-6** / **6.0e-6** | 48.5 (0) / 2.5e-9 / 2.5e-9 % | 69/266, 77/441, 77/441 (80/6631) |
| wetdry | 1.35e-6 / 8.13e-12 / 8.13e-12 | 1.6e-6 / 7.4e-12 / 7.4e-12 | 3.1e-4 / 1.7e-11 / 1.7e-11 % | 40/139, 53/235, 53/235 (54/1137) |
| disu_staggered | 3.36e-8 / 3.2e-12 / 3.2e-12 | 1.6e-13 / 3.5e-11 / 3.5e-11 | 0 / 1.5e-9 / 1.5e-9 % | 5/15, 7/24, 7/23 (7/49) |

(rad-disu and twri01 heads in feet; MF6's maw-p03c percent discrepancy is 8.9e-5 % too.)

**`:single` budgets: a bound from Float32 rounding of the cell flows** (replaces the 5e-4 ×
total in of rad_disu_confined, T6.4, which now passes 1e-4 with 9.6e-13 beyond the bound).
An engine flow is `C·(h'_m − h'_n + Δr)` or `κ·(h'old − h')` (`κ = (Ss·thk + Sy)·area/Δt`)
from Float32 offsets, so it can differ from MF6's by `C·(4·eps32·(|h'_n| + |h'_m| + |Δr|) +
|δh_n| + |δh_m|)`, `δh` the difference of the converged heads (tested separately against
1e-3 m). `_rounding_bounds` (test/run_parity.jl) computes these per FLOW-JA-FACE entry (C
from MF6's flow and head difference, condsat when MF6's heads are equal; the diagonal, the
cell's net flow, gets the sum over its connections plus its storage bound) and per STO-SS /
STO-SY cell; their sum per step bounds the listing terms and the in − out imbalance. The
criteria (1e-4 × total in, 0.01 %) apply to what is beyond the bound, for rad_disu_confined,
ex-gwf-rad-disu, ex-gwf-maw-p03c and ex-gwf-advtidal only (`single_rounding_bound`,
test/parity/run_test.jl); every other model keeps the raw criteria (reviewer, T6.1). The
bound contains the measured head difference, so it checks that the flows agree with the
heads, not the budget itself: the per-step sum allows a percent discrepancy of up to 47 %
(advtidal), 3400 % (maw-p03c) and 6e5 % (rad-disu), i.e. **the `:single` budgets of these
three models are not verified**. This matters where
large coefficients meet small flows: rad-disu (the STO-SY coefficient of the first steps:
one Float32 offset rounding is thousands of ft³/d against 4000 ft³/d in; raw pd 48.5 %),
maw-p03c (total inflow 18 m³/d; raw CBC 1.0e-3, pd 0.60 %), advtidal (raw pd 1.9e-2 %).
Use `:mixed` where such cell-by-cell flows or budgets matter.
The observed flows of maw-p03c (OBS6, T6.7: flows next to the former well, heads flat at
10.0001 m, large conductances) differ by up to 2.2e-4 × the total inflow in `:single`
(heads 1.6e-6 m); their parity criterion is 1e-3 × total inflow instead of 1e-4
(`single_obs_tol`, test/run_parity.jl).

**ex-gwf-rad-disu `:double` / `:mixed` budget: criterion 1e-5 instead of 1e-6.** The
largest difference is the STO-SY rate of cell 21 in step 2 (MF6 0.0241 ft³/d, ours 0):
with Δt = 9.4e-5 d the coefficient Sy·area/Δt turns the 2.5e-10 ft head difference to
MF6's tight-relative reference into 6.0e-6 × total in. The heads agree to 2.5e-10 ft.

**Float32 STO-SY.** MF6's `rho2·thk·snold − rho2·thk·snnew` cancels in Float32 when the
saturations are near 1 (rad-disu: rates quantised to 4, 8, 64, 512 ft³/d); the engine uses
the equivalent `rho2·(clamp(hold) − clamp(h))` (docs/dev/mf6_notes.md).

**Float32 rclose floor with convertible cells.** The floor `eps32·max Σ|terms|` was taken
from the first formulation of a step (T2.8). With convertible cells the terms grow as cells
saturate (twri01: the first iteration's floor was below the attainable residual, and the
STRICT inner test then failed in every outer iteration while the heads changed by 5e-6 m,
500 outer iterations); it is now the maximum over the step's iterations for convertible
models (confined models unchanged).

**Float32 rclose floor with XT3D (T6.5).** The same rule (maximum over the step's
iterations) applies to XT3D models, and their per-row scale is Σ|c_t·(h_a − h_b + Δr_t)|
over the flux terms of every face (`_xt3d_flux`), not |q| per face: the XT3D terms of a face
cancel, so |q| understates the rounding of the residual, and a model starting from flat
heads (ex-gwf-whirl) has no flow terms in the first formulation. Without it ex-gwf-whirl did
not converge in `:single`. Only `:single` (the floor factor is 0 otherwise), only XT3D models
(`m.xt3d !== nothing`), only the inner INNER_RCLOSE: OUTER_DVCLOSE is unchanged.

**Confined models unchanged.** ICELLTYPE 0 models keep the constant-`condsat` kernels
(`NpfModel.cv === nothing`): the final heads (bits) and the outer/inner iterations of every
step are identical to main (79b2447) for ex-gwf-hanic, -hanir, -toth, -fhb, -lgrv-gc,
-curve-90, -curvilin, -u1disv, coupling_rde and ts_tas in `:single`, `:mixed` and
`:double`, on the CPU backend and on ROCm (60 runs). Synthetic global5m daily step on
ROCm (benchmark/run.jl, `precond=mc_ilu0 timed=0`): daily step 0.221 → 0.214 s (`:double`), 0.201 → 0.198 s (`:mixed`), 0.091 → 0.084 s (`:single`), the same 8.3/26.1 (5.2/13.0) outer/inner iterations per step and the same peak memory (4.204, 4.007, 2.842 GiB); main and branch measured back to back (benchmark/results/2026-11-16-T6.1.md).

ROCm (RX 7800 XT): every convertible model above passes the same criteria on ROCm, with heads bitwise equal to the CPU backend in all three presets (max |Δh| = 0).

## Solver performance changes (T4.7)

The T4.7 changes (incremental red–black MC-ILU0 setup, contiguous diagonal, CG direction
fused into the SpMV, skipped last MC-ILU0 application, scale-free operator rows for
SCALING_METHOD NONE, single-pass reductions, package records formulated in the row
kernel) compute the same operations in the same order: heads and outer/inner iteration
counts are bitwise those of `main` (24ccc38) in `:single`, `:mixed` and `:double`, on CPU
and rocm (coupling_rde, ts_tas, ex-gwf-u1disv, ex-gwf-curve-90, synthetic tiny/indus; see
benchmark/results/2026-11-16-T4.7.md). The accuracy numbers in this file are unchanged.

## Backtracking (T6.2a)

`sln_backtracking` (src/solver/backtracking.jl): the residual L2 norm is a device reduction
of the flux-form residual with Float64 accumulation; the reduction of the head change acts
on the offset heads. Measured on CPU and ROCm (identical numbers; ROCm vs CPU heads equal):

| Model (IMS) | preset | backtracking (step, outer iteration) | outer/inner, ours (MF6) | max \|Δh\| vs MF6 |
| --- | --- | --- | --- | --- |
| coupling_rde (COMPLEX) | `:double` | (1,2), (1,3), (1,4), flag 1, 1 reduction each, as MF6 | 125/3083 (125/4019) | 7.5e-11 m |
| coupling_rde (COMPLEX) | `:mixed` | same | 125/3069 | 7.5e-11 m |
| coupling_rde (COMPLEX) | `:single` | same | 72/3449 | 5.7e-6 m |
| wetdry (COMPLEX, test only) | `:double` | 18 outer iterations of steps 9 and 10, as MF6 (also flags and counts) | 103 (103) | – |
| wetdry (COMPLEX, test only) | `:single` | the same 18; fewer reductions in the two flag-4 rows (5, 6 vs 8, 9: OUTER_DVCLOSE floored to 1e-6) | 76 (103) | – |

The incoming and outgoing norms of the backtracking rows agree with MF6's listing to its
8 printed digits in `:double` (e.g. coupling_rde step 1: 5440.7412 → 2402.1836). Before
T6.2a (no backtracking; the same input with BACKTRACKING_NUMBER 0, MC-ILU0):
coupling_rde `:double` 125/2948, `:single` 71/2239 — MF6 also needs 125 there; wetdry
with COMPLEX `:double` 83 outer iterations against MF6's 103 (steps 9 and 10: 15 and 16
instead of 22 and 29); now equal per step. On DISV (ex-gwf-curve-90 with ICELLTYPE 1)
and DISU (disu_staggered) the incoming norms of the first outer iterations agree with
MF6's within 1e-6 (`:double`) and 1e-3 (`:single`, norms above 1e-3 of the first; below,
the Float32 rounding of the residual dominates).

Models without backtracking are bitwise unchanged: heads and outer/inner iterations per
step equal `main` for coupling_rde (MC-ILU0 and Jacobi), ts_tas, wetdry, disu_staggered,
afr, ex-gwf-u1disv and ex-gwf-advtidal in `:single`, `:mixed` and `:double`, on CPU and
ROCm (24 of 24 cases each). Cost on global5m: benchmark/results/2026-11-16-T6.2a.md
(ModflowCoupling's COMPLEX IMS: +1 ms per daily step, no reduction ever applied).

## Newton–Raphson: the Tier 4 examples (T6.2)

Models: ex-gwf-bcf2ss-p02a, ex-gwf-bump-p01a and -p01c, ex-gwf-nwt-p02a, ex-gwf-nwt-p03a and
-p03b, ex-gwf-zaidel, `capture_saved` (ex-gwf-capture with an OC that saves heads; the
reference run saves none) and the synthetic test/models/newton (convertible storage, WEL
AUTO_FLOW_REDUCE, DRN AUXDEPTHNAME, RCH, CHD; mf6 run at test time). Parity table
(docs/src/assets/parity.toml, commit e719f66, MC-ILU0 BiCGSTAB in every preset): max
|Δh| against MF6 (m, ft for bcf2ss and nwt-p02a; cells whose MF6 head is below the bottom
are dry and left out, see below), the budget difference (the larger of CBC and listing,
× total in), and outer iterations ours/MF6. CPU and ROCm give the same numbers (ROCm vs
CPU heads: 0 in every preset).

| Model | `:double` \|Δh\| / budget / outer | `:mixed` \|Δh\| / budget | `:single` \|Δh\| / budget / outer |
| --- | --- | --- | --- |
| ex-gwf-bcf2ss-p02a | 7.5e-11 / 3.3e-13 / 14 (17) | 7.5e-11 / 3.3e-13 | 1.3e-5 / 6.5e-8 / 13 |
| ex-gwf-bump-p01a | 8.4e-8 / 9.5e-12 / 22 (23) | 8.4e-8 / 9.5e-12 | 1.1e-6 / 3.5e-7 / 17 |
| ex-gwf-bump-p01c | 2.6e-9 / 4.5e-11 / 8 (10) | 2.6e-9 / 4.4e-11 | 1.5e-5 / 8.7e-6 / 7 |
| ex-gwf-nwt-p02a (1 + 11 steps) | 3.5e-10 / 0 / 456 (464) | 3.8e-10 / 0 | 1.7e-4 / 4.6e-5 / 282 |
| ex-gwf-nwt-p03a | 1.2e-9 / 1.1e-16 / 9 (10) | 1.1e-9 / 1.1e-16 | 2.6e-5 / 3.4e-7 / 8 |
| ex-gwf-nwt-p03b | 1.8e-8 / 0 / 96 (159) | 1.7e-8 / 0 | does not converge |
| ex-gwf-zaidel | 6.8e-8 / 0 / 27 (21) | 7.3e-8 / 0 | 1.6e-4 / 0 (rounding bound) / 21 |
| capture_saved | 4.2e-7 / 2.5e-9 / 6 (8) | 4.2e-7 / 2.5e-9 | 3.8e-6 / 1.1e-6 / 7 |
| newton (test model) | 3.1e-12 / 7.7e-12 / 91 (100) | 3.1e-12 / 7.7e-12 | 9.4e-6 / 1.8e-6 / 60 |

Criteria as for the other run parity tests (`:double`/`:mixed` 1e-6 m and 1e-6 × total
in, `:single` 1e-3 m and 1e-4 × total in); budget 0 means MF6 writes no budget file and
the listing tables agree within their printed digits.

**Dry cells.** With Newton–Raphson a cell whose head is below its bottom keeps that head
(no HDRY) and has no outflow; any head between the bottom and the highest wet neighbour
solves its row, so MF6 and ModflowGPU stop at different such heads (e.g. 1e-8 m and 1e-6 m
below the bottom). The parity test checks that a cell below its bottom in MF6 is below its
bottom in ours, and leaves its head out of |Δh| (also out of the ROCm vs CPU and `:single`
vs `:double` comparisons).

**ex-gwf-nwt-p03b `:single`.** The Newton iterations do not converge in Float32 (500
outer iterations, with MC-ILU0 and with AMG). MF6's own path is erratic (inner solves at
INNER_MAXIMUM in outer iterations 1–3, head changes up to 1420 m, 159 outer iterations in
`:double`); in Float32 the unconverged inner solves end far off and the iterations wander
between states with changes of 10–1000 m. With recharge 1000 times that of p03a, MF6's
solution is a water table just above the bottom: median saturated thickness 0.55 mm, 4077
of 6400 cells below 1 cm and 127 within the quadratic smoothing interval (1e-6 of the
150–165 m cell thickness, 1.6e-4 m). The head offsets are about −20 m there (STRT is 20 m
above the bottom), whose Float32 spacing is 1.9e-6 m: the saturated thicknesses, hence the
transmissivities, carry 2–3 significant digits, the smoothing interval is about 80 rounding
units and the OUTER_DVCLOSE floor (`eps(Float32)·max|h'|`, 2.4e-6 m) about one (review of
T6.2). The Float32 formulation itself is consistent: started from MF6's heads (STRT = the
MF6 solution, so the offsets are near 0), `:single` converges in 2 outer iterations to
6.4e-6 m of MF6. The run parity test and the parity table run `:mixed` and `:double` only.
`FLOAT32_CLOSURES.dvclose_head_cap` (1 cm) keeps a diverging Float32 iteration (heads of
1.6e27 m) from raising OUTER_DVCLOSE with its head resolution and ending the step as
converged.

**ex-gwf-zaidel `:single` budget.** K = 1e-4 m/d, head differences of 6 cm per cell next
to heads of 13 m: the raw percent discrepancy is 1.3e-2 % and the raw flow differences
exceed 1e-4 × total in; the criterion applies to the part beyond the Float32 rounding of
the cell flows (`single_rounding_bound`, as for rad-disu, maw-p03c and advtidal; MF6
writes no budget file, so the bound takes C = condsat).

**BiCGSTAB restart (Newton only).** In Newton–Raphson solves, when `rho = dhat·r`
vanishes at the rounding level of the vectors (`|rho| <= eps(Ts)·|dhat|·|r|`), BiCGSTAB
restarts with `dhat = r` (MF6 has no restart). It was added for the first outer iteration
of the zaidel steady state with the red–black ILU0 (`rho` at 3.7e-31·|dhat|·|r| in
`:double`, exactly 0 in `:single`; after that the next step length is noise and the solve
diverges). In Float32, `rho` of the Newton systems also falls to 1e-7 of |dhat|·|r| in
long solves: with these restarts ex-gwf-nwt-p02a `:single` converges (282 outer
iterations; 14 restarts in 282 solves, at most 2 per solve), without them step 1 of period
4 does not (500 outer iterations). nwt-p03b `:double` restarts 3 times (ratios
3.6e-18–2.1e-16). The check uses the `rho` BiCGSTAB already reduces (no extra reduction).

Other systems do not restart, as in MF6 and `main`. Two variants that also restarted them
were tried and dropped: with `eps(Ts)`, coupling_rde with Chebyshev in `:single` restarted
about 10 times per solve (ratios 8e-11–1.2e-7) and step 1 no longer converged in 25 outer
iterations (nor with at most one restart per solve), and rad-disu `:single` changed (110 →
109 outer iterations); with the rounding level of the dot product (`eps(Float64)`, true
breakdowns only: hanic 1.4e-22 `:single` / 5.8e-17 `:double`, hanir 0 / 9.9e-17,
curve-90 `:single` 0), hanic and hanir needed fewer iterations (`:double` 4/270 → 3/100
and 5/368 → 3/94) but one restart at `rho = 0` in rad_disu_confined `:single` (AMG,
nearly singular, most inner solves unconverged) took it from 180 to 341 outer iterations.

**Non-Newton models.** The Newton terms, PTC, NEWTON UNDER_RELAXATION and the BiCGSTAB
restart apply only with NEWTON. Heads (sha256 of the head file), budget files and
outer/inner iterations equal `main` (d0ba506) for coupling_rde, wetdry, disu_staggered,
ex-gwf-u1disv (DISV), ex-gwf-rad-disu (DISU), ex-gwf-twri01, ex-gwf-hanic, ex-gwf-hanir and
ex-gwf-curve-90 in `:single`, `:mixed` and `:double` with the default preconditioner
(MC-ILU0), on CPU and ROCm (ROCm equal to CPU), and every non-Newton row of the parity
table equals the previous table (cf78ba7) or, where T6.2a changed it, a run of
docs/parity/collect.jl on `main`. Cost on global5m: none
(benchmark/results/2026-11-18-T6.2.md).

## Several models and GWF-GWF exchanges: the Tier 7 examples (T6.6)

The models of a solution run as one merged model (src/gwf/exchange.jl), so the precision
rules apply unchanged: each model keeps its own reference heads, and an exchange
connection between two models is a connection of the merged CSR with `Δr = r_m − r_n`
across the models. Run parity against the tight reference runs (CPU, default
preconditioner MC-ILU0; the solution is coloured red–black per model where the exchange
connections allow, else greedily):

| Example | Models, cells | `:double` max \|Δh\| | `:single` max \|Δh\| | `:single` CBC / total in | outer/inner `:double` (MF6) |
| :-- | :-- | --: | --: | --: | :-- |
| ex-gwf-u1gwfgwf-s1 | 2, 121 | 1.6e-12 | 1.5e-7 | 1.7e-7 | 2/17 (2/16) |
| ex-gwf-u1gwfgwf-s2 (XT3D in the models) | 2, 121 | 2.2e-13 | 5.7e-8 | 2.3e-7 | 2/13 (2/12) |
| ex-gwf-u1gwfgwf-s3 (XT3D models and exchange) | 2, 121 | 1.7e-13 | 1.9e-7 | 2.3e-7 | 2/13 (2/11) |
| ex-gwf-u1gwfgwf-s4 (XT3D on the exchange) | 2, 121 | 2.3e-13 | 1.9e-7 | 2.3e-7 | 2/13 (2/14) |
| ex-gwf-lgrv-lgr | 2, 195381 | 3.4e-7 | 3.7e-5 | 7.3e-6 | 2/999 (2/1001) |
| ex-gwf-spbc (periodic) | 1, 19000 | 1.0e-7 | 2.4e-7 | 9.9e-8 | 2/115 (2/1001) |

`:mixed` equals `:double` to the digits shown. ex-gwf-lgrv-lgr and ex-gwf-spbc in `:double`
end 3.4e-7 and 1.0e-7 m from MF6's tight run, within 1e-6 m: these are the accuracy of
the reference runs' linear solves (MF6 needs 1001 inner iterations, INNER_MAXIMUM, in
both); the universal residual of MF6's heads in our merged model is 1.1e-12 and 1.8e-10 of
the largest row. ex-gwf-lgrv-lgr `:single` takes 5 outer iterations (1720 inner) for
3.7e-5 m (K = 1e-6 m/s next to 300 m heads, as ex-gwf-lgrv-gc). Its flows with MF6's heads
differ from MF6's by up to 2.6e-4 of the largest flow in `:single`, all within the Float32
rounding of the flows' terms (the exchange test compares beyond that bound).
