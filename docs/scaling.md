# Scaling: synthetic coupling models (T3.4)

The synthetic coupling-like models of `benchmark/synthetic.jl` (module `SyntheticModels`)
stand in for the global inputs that do not exist yet: parity at small sizes, build time
and device memory at large sizes.

## The model

Per preset (`SYNTHETIC_PRESETS`): `ncol × nrow` cells, 2 confined layers (layer 1 = 10 % of
a 50–350 m aquifer, as ModflowCoupling), uniform DELR = DELC (9000 m at 5′, 900 m at 30″).

- Land mask: multi-octave value noise thresholded at the preset's land fraction (lowered
  towards the grid edge for the island presets); the ocean cells next to land form the
  CHD rim (head 0 in both layers); the rest of the ocean is IDOMAIN 0.
- DEM: 1–2001 m, rising from the coast with the land field.
- K: log-normal, log10 K1 = 0.5 + z clipped to [−2.5, 2.5] m/d, layer 2 a correlated lower
  field clipped to [−3, 2.5]: 5–5.5 orders of magnitude contrast; K33 = K/10.
- Storage: STORAGECOEFFICIENT 0.05–0.2 (ModflowCoupling: SS = SY), ICONVERT 0.
- Rivers: D8 routing by priority flood (Barnes et al. 2014) from the rim over the DEM;
  RIV on land cells draining ≥ 10 cells (width 2√acc m, conductance width · 1.5 DELR / 1 d,
  bed 1 + ½ ln acc m below the DEM, stage = bed + bankfull depth × seasonal factor).
- DRN on the other land cells (2 ± 1 m below the DEM, conductance 0.02/d · SC · area, the
  ModflowCoupling linear reservoir), EVT on all land cells (surface = DEM, 1 mm/d, 2 m
  extinction depth, FIXED_CELL, NSEG 1), RCH list on all land cells (0.1–2 mm/d,
  FIXED_CELL).
- TDIS: one steady-state day, then 12 daily transient periods; RCH (×(1 + 0.8 sin)) and
  RIV stage (×(1 + 0.5 cos)) change every period.
- Starting heads: 0.5 m above the drain level, the stage in river cells, 0 on the rim.
  From the drain level (all DRN off in the first Picard iteration, MF6 drn_cf: no
  conductance for h <= elevation) the first outer correction on indus is +254906 m
  (+290441 m with river cells also at the drain level, i.e. below their beds): recharge
  on low-K coastal land (T ~ 1 m²/d) with no active outlet. mf6 6.8.2 gives the same
  corrections (to 7 digits) and converges (31 outer iterations in the steady state), so
  this is the model, not the engine.
- Deterministic: counter-based SplitMix64 hashes of (seed, field, octave, lattice point);
  no dependence on Random or the Julia version for the random numbers, no per-cell
  allocations.

**IMS** (`synthetic_ims(:parity)`): OUTER_DVCLOSE 1e-9, INNER_DVCLOSE 1e-10, INNER_RCLOSE
1e-3 m³/d, CG, 5000 inner / 500 outer iterations, UNDER_RELAXATION DBD (theta 0.9, kappa
1e-4, gamma 0: COMPLEXITY MODERATE's values). Two findings that set it:

- Without under-relaxation the steady state does not converge in mf6 either: the Picard
  iterations cycle between DRN/EVT on/off states (tiny: ±22126 m and ±18532 m forever).
  COOLEY 0.5 and the DBD variant of the run parity test fail in mf6 too; SIMPLE 0.5 and
  the MODERATE DBD values converge (tiny: 369 and 80 outer iterations).
- **BiCGSTAB (ModflowCoupling's choice) with the Jacobi preconditioner diverges in
  `:single`** in the steady state: tiny in outer iteration 4, indus in outer iteration 5,
  each after INNER_MAXIMUM = 5000 inner iterations, returning a non-finite correction (the
  outer loop stops, `ConvergenceError`). `:double` BiCGSTAB converges in all 13 steps
  (tiny 68 outer / 6624 inner, indus 100 / 8506). The matrix is symmetric (confined, no
  Newton), so the parity settings use CG, which converges in both presets. The same
  failure mode as ex-gwf-hwell-a (docs/precision.md); the cure is the Phase 4
  preconditioner, not a looser tolerance.

## Parity vs mf6 6.8.2

Test group `synthetic` (test/parity/synthetic_test.jl, mf6 run at test time in a temp
directory; about 1 min on the CPU backend; `MFGPU_TEST_SYNTHETIC=tiny` for tiny only).
13 steps; heads over every saved step, CBC max |Δq| / total in, listing beyond rounding /
total in, max |percent discrepancy|, outer/inner iterations summed over the steps (mf6
with ILU0, ours with Jacobi). Times include compilation on the first model.

| Model | preset | max \|Δh\| (m) | CBC | listing | pd | outer/inner (mf6) | time CPU / ROCm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tiny (2,690 nodes) | `:double` | 1.4e-9 | 4.6e-12 | 6.3e-12 | 4.2e-10 % | 68/10696 (68/413) | 5.8 s / 0.7 s |
| tiny | `:single` | 1.7e-4 | 8.7e-6 | 2.8e-5 | 2.7e-3 % | 38/5893 | 8.6 s / 0.4 s |
| indus (32,972 nodes) | `:double` | 1.3e-9 | 1.6e-13 | 1.5e-12 | 2.8e-11 % | 99/14915 (100/585) | 9.5 s / 1.6 s |
| indus | `:single` | 1.0e-4 | 2.9e-7 | 2.8e-6 | 2.8e-4 % | 49/7867 | 5.4 s / 0.8 s |

ROCm and the CPU backend give bitwise identical heads (max |Δh| = 0) in both presets;
`:single` vs `:double` final heads differ by 1.7e-4 m (tiny) and 9.5e-5 m (indus). The
`:double` outer counts equal mf6's (within one). mf6 itself takes 0.17 s (tiny) and 2.1 s
(indus). The CPU times of the first model include compilation.

## Build times and device memory

`julia --project=benchmark benchmark/synthetic.jl rocm indus continental global5m regional30s global30s outer=2`
(RX 7800 XT, 16 GB). Fields + spec: host time to generate the fields and build the
`SimulationSpec`; initialize: `ModflowGPU.initialize` on the device (connectivity, NPF,
packages, solver). Device arrays: every array reachable through the Adapt rules of the
model, solver and budget accumulator (`device_bytes`); measured: `AMDGPU.used_memory()`
before and after `initialize` (no further allocations during the outer iterations).

| Preset | grid | nodes | nja | fields + spec | initialize (single / double) | `:single` device | `:double` device | measured (single / double) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tiny | 50 × 40 | 2,690 | 15,764 | < 0.01 s | – | 0.8 MB | 1.3 MB | – |
| indus | 204 × 180 | 32,972 | 195,040 | 0.01 + 0.03 s | 0.01 / 0.03 s (CPU) | 9.7 MiB | 15.3 MiB | 9.7 / 15.3 MiB |
| continental | 1000 × 800 | 888,300 | 5,318,376 | 0.35 + 1.0 s | 0.20 / 0.31 s | 0.256 GiB | 0.407 GiB | 0.256 / 0.407 GiB |
| regional30s | 2040 × 1800 | 3,114,960 | 18,647,468 | 1.6 + 3.1 s | 0.50 / 0.49 s | 0.899 GiB | 1.426 GiB | 0.899 / 1.429 GiB |
| global5m | 4320 × 2160 | 6,592,790 | 39,461,792 | 4.5 + 9.1 s | 1.69 / 1.67 s | 1.903 GiB | 3.019 GiB | 1.903 / 3.019 GiB |
| global30s (estimate) | 43200 × 21600 | 653.6 M | 3.91 G | – | – | 189 GiB | 300 GiB | – |

Per node: 307–310 B in `:single`, 488–492 B in `:double` (CSR values and the Krylov vectors
dominate). Host peak RSS of the whole script (all presets, both precisions): 7.5 GiB.
global5m and regional30s fit on one 16 GB GPU with room for mGV (T5.4); `:double`
global5m uses 19 % of the card.

**Estimate.** `memory_coefficients` divides each array's bytes by what it scales with
(nodes, CSR entries, connections, records of each package; a few KB are fixed), and
`estimate_device_bytes` applies the coefficients of `indus` to other counts: within 0.3 %
of the measured bytes for continental, regional30s and global5m (unit test: tiny →
indus within 1 %). `global30s` counts are scaled from `global5m` (`scaled_counts`: land
and river cells with the area, ×100; the rim with the linear resolution, ×10;
connections per node kept): 653.6 M nodes, 3.91 G CSR entries. That is 189 GiB in
`:single` and 300 GiB in `:double`: 12 resp. 19 RX 7800 XTs at 100 % use, or a 192 GB
MI300X-class card for `:single` only. Also, 3.91 G CSR entries exceed the `Int32`
index type (`Ti`, 2.1 G): one global 30″ model needs `Int64` row pointers or the domain
decomposition of T7.5.

**Speed (first outer iteration of the steady state, ROCm, CG + Jacobi):** about 290 µs
per inner iteration at continental, 940 µs (`:single`) / 1470 µs (`:double`) at
regional30s, and 2.3 ms / 3.2 ms at global5m (outer iteration 2). Full runs: below.

**Full runs on ROCm** (13 steps, `synthetic_ims(:parity)`, no output;
`ModflowGPU.run` of the in-memory spec; measured while a CPU test suite ran on the host):

| Preset | `:double` | `:single` |
| --- | --- | --- |
| continental (888,300 nodes) | 18.3 s, 119 outer / 15185 inner (steady step 18 / 13993), pd 4.7e-12 % | 9.1 s, 65 / 7239 (12 / 6810), pd 8.9e-6 % |
| global5m (6,592,790 nodes) | 48.5 s, 113 / 9447 (13 / 8134), pd 1.4e-11 % | 20.3 s, 72 / 4047 (10 / 3573), pd 3.6e-6 % |

Both converge with CG + Jacobi: the dense coastline (CHD rim) and the strong DRN/RIV
conductances keep the synthetic steady state well conditioned; the real global model
(fewer drains, larger connected land masses) may be harder.

**Benchmark baseline (T4.1)**: steady state + 30 daily steps on rocm, the CPU backend and
mf6, with wall time per step, SpMV GB/s, peak device memory and the time split, in
[benchmark/results/2026-11-07.md](../benchmark/results/2026-11-07.md) (`benchmark/run.jl`).
