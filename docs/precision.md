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

Proposal for the owner (not implemented; it changes rule 1): use `r = STRT` without
the clamp in ICELLTYPE 0 cells (the clamp only matters for convertible cells, T6.1),
and consider re-referencing `r` to the current heads between stress periods (host
Float64 work: `r`, `Δr` and the head-like boundary columns).

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
