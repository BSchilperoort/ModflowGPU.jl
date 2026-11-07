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
| ex-gwf-toth | 1 | 5.6e-9 | 4.9e-5 | 4.9e-5 | 3.4e-13 / 3.2e-7 | 1.3e-7 | 2.5e-5 % | 3/1247, 3/1435, 2/254 |
| ex-gwf-hanic | 1 | 1.5e-9 | 5.1e-7 | 4.6e-7 | 9.4e-10 / 4.0e-7 | 0 | 1.0e-3 % | 3/199, 3/146, 2/39 |
| ex-gwf-hanir | 1 | 1.4e-9 | 5.0e-7 | 5.3e-7 | 9.0e-10 / 4.9e-7 | 0 | 1.2e-3 % | 4/222, 3/151, 2/39 |
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
| INNER_DVCLOSE | 1e-6 m | below the outer floor is wasted work |
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

**ex-gwf-hwell-a** (153164 cells, 60 steps, BiCGSTAB; opt-in `MFGPU_TEST_HWELL=1`, not
part of the suite). `:single` on ROCm: steps 1–56 converge in 2–15 outer iterations and
stay within 2e-5 m of MF6 (max |h'| 5.8 m); in step 57 (`delt` = 0.096 d) the Jacobi
BiCGSTAB solve does not converge in 1000 iterations and returns a correction of 3656 m
for a residual of 0.93, after which the outer iterations diverge (NaN by iteration 8).
Until T2.8's last fix the max-abs reduction skipped NaN and the step was reported as
converged; it now propagates NaN and the outer loop stops with a warning (and
`ConvergenceError`). `:double` on ROCm converges in all 60 steps (231 s; 2 outer and 12–30 inner iterations per step in the first steps), within 2.0e-10 m of MF6 in the last step, percent discrepancy 2.8e-8 %. The cure is the Phase 4 preconditioner (MF6 needs 2 outer
and 7–10 inner iterations per step with ILU0).

**Section 4 thresholds.** Confirmed for `:single` on Tier 0 and the synthetic models:
heads < 1e-3 m (largest measured 4.9e-5 m, toth, whose offsets reach 1010 m: the head
error scales with `eps32·max|h'|`), package flows < 1e-4 × total in (largest 3.4e-6),
percent discrepancy < 0.01 % (largest 1.5e-3 %), and `:single` vs `:double` heads
< 1e-3 m. The `:double` criteria hold with large margins (heads ≤ 4.0e-8 m, flows
≤ 1.0e-9 × total in). No threshold is changed; the measured values leave about a factor
20 on heads and 7 on percent discrepancy for the larger models of T3.4 and T5.3.
