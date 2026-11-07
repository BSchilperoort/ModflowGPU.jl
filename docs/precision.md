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
