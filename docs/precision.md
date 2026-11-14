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
| INNER_DVCLOSE | 1e-6 m; raised to OUTER_DVCLOSE for the rest of the step once a correction meets OUTER_DVCLOSE but its inner solve did not converge (T5.3, below) | below the outer floor is wasted work |
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
solvers (`_safe_div`, docs/mf6_notes.md): with MF6's absolute `DPREC` guard the
correction solves stalled at |r| ~ 1e-8 and hit INNER_MAXIMUM every outer iteration
(hanir `:double` 16/14402, coupling_rde 226/215792; MF6's own 107439 inner iterations on
coupling_rde show the same stall). Columns other than `:single` vs `:double` were
re-measured after that change. COMPLEXITY COMPLEX enables backtracking, which MF6
used in 3 of 125 outer iterations; ModflowGPU warns and does not backtrack (T6.2), and the
converged heads agree as above. In `:single` the outer loop stops at the Float32 floors,
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
extension hook; the CPU backend aliases the same `Array`; AMDGPU 2.8, the version mGV
allows, keeps the copy because it cannot unregister wrapped memory), so the copy is gone and a
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
divergence on the synthetic models: docs/scaling.md. The section 4 thresholds hold
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
rises from 6.6e-5 to 1.4e-3 m. The conditional rule leaves every Phase 2 parity model
bitwise unchanged (it never triggers there).

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

### Proposal: production closures of the GPU engine (owner decides, PLAN.md section 8)

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

The ModflowCoupling default is not changed by this card (`SolverSettings` keeps 5 for
both engines); adopting the proposal means a per-engine default in the patched
`config.jl` (a later patch).
