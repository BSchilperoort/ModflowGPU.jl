# Comparing results

ModflowGPU solves the same equations as MODFLOW 6, so on the same input it should give
the same heads and budgets, up to the closure criteria of the solver and the rounding of
the chosen precision. This page says how close the results are, how that is checked, and
how to check it on your own model.

## What is compared

Every tested model is run with MODFLOW 6.8.2 and with ModflowGPU, and the results are
compared:

- **Heads:** every saved time step of the head file, every cell.
- **Budgets:** every record of the budget file (the flows between cells, storage and
  each package's flow per boundary cell) and every term of the budget tables in the
  listing file (rate in and rate out per package), per time step. Differences are
  given relative to the total inflow of the time step.
- **Percent discrepancy** of each budget table.

Both programs run with tight closures (`OUTER_DVCLOSE 1e-9`, `INNER_DVCLOSE 1e-10`,
`INNER_RCLOSE 1e-10`, with enough iterations), so that each result is close to the exact
solution of the equations and the comparison measures the equations, not where each solver
happened to stop. With the loose closures of production runs (an `OUTER_DVCLOSE` of
metres) both programs stop early, at different places, and their heads differ by up to
the closure.

The preconditioner is ModflowGPU's default, not MODFLOW 6's ILU (see
[Differences from MODFLOW 6](differences.md)), so the iteration counts differ. Iteration
counts are reported, not compared.

## Acceptance limits

A model passes when all of these hold:

| Comparison | `:double` and `:mixed` | `:single` |
| :-- | :-- | :-- |
| Heads against MODFLOW 6 | max \|Δh\| < 1e-6 m (or 1e-7 × the head range, if larger) | max \|Δh\| < 1e-3 m |
| Budgets against MODFLOW 6 | < 1e-6 × total inflow, per record and per listing term | < 1e-4 × total inflow; percent discrepancy < 0.01 % |
| `:single` against `:double` | – | max \|Δh\| < 1e-3 m |
| GPU against the CPU backend | heads within 1e-10 × the largest \|head\| | heads within 1e-4 m |

`:mixed` keeps the heads and the solver vectors in Float64 and only runs the
preconditioner in Float32, so it must meet the `:double` limits.

Why `:single` has wider limits, in plain terms:

- **Rounding.** A Float32 number has about 7 significant digits. ModflowGPU stores each
  head as a difference from a reference head of its cell, so the 7 digits go to the part
  that changes; still, with heads that vary by hundreds of metres within a model, a
  Float32 head is only resolved to about 1e-5 to 1e-4 m.
- **Closures.** For the same reason the solver cannot meet closures below that
  resolution. In `:single`, closures tighter than what Float32 can reach are raised (to
  at least 1e-6 m for `OUTER_DVCLOSE` and `INNER_DVCLOSE`, more for models with a large
  head range), with an `Info` message in the log. The results are then as close to
  MODFLOW 6 as the rounding allows.
- **Budgets.** Flows are conductance × head difference. Where conductances or storage
  coefficients are very large, the rounding of the heads is multiplied by them. For five
  models (`rad_disu_confined`, `ex-gwf-rad-disu`, `ex-gwf-maw-p03c`, `ex-gwf-advtidal`,
  `ex-gwf-zaidel`) the `:single` budget limit is applied to the part of the difference
  that exceeds this rounding bound, computed per flow; the raw differences are larger (on
  the radial DISU grid, up to 0.4 × total inflow in single flows). Use `:mixed` or
  `:double` when budgets of such models matter.
- **Iterations.** Not all models converge in `:single` with the default preconditioner.
  `ex-gwf-lgrv-gr` is run with `preconditioner = :amg` in `:single` (see the
  [GPU guide](@ref "Which preset to use")). The Newton–Raphson example `ex-gwf-nwt-p03b`
  does not converge in `:single` at all and is tested in `:mixed` and `:double` only.
- **Newton–Raphson.** Cells whose head is below their bottom are dry, and their head is
  not unique ([Differences from MODFLOW 6](@ref "Newton–Raphson")): the comparison checks
  that such cells are below their bottom in both runs and leaves their heads out of the
  differences.

Three models use a criterion of their own, all for documented reasons in the developer
notes: `ex-gwf-rad-disu` in `:double` and `:mixed` (budgets within 1e-5 × total inflow:
one storage flow in a step of 8 seconds amplifies a 2.5e-10 ft head difference), and
`ex-gwf-hwell-a`, `ex-gwf-rad-disu` and `rad_disu_confined`, which run with
`preconditioner = :amg` in every preset.

## Parity table

One row per tested model: the MODFLOW 6 examples (directory names of the
[MODFLOW 6 examples](https://github.com/MODFLOW-ORG/modflow6-examples)), the small test
models of the ModflowGPU test suite, and the synthetic coupling models of the size of
the tests (`synthetic tiny`, 50 × 40, and `synthetic indus`, the 204 × 180 grid of the
Indus basin model). Every model is run in every preset on both backends.

```@eval
Main.CompatTables.parity_source()
```

```@eval
Main.CompatTables.parity_table()
```

Columns: the grid as `DIS layers×rows×columns`, `DISV layers×cells per layer` or
`DISU nodes`, with the number of active cells; the stress packages (STO and NPF
included); max |Δh|, the largest head difference with MODFLOW 6 over all cells, saved
steps and backends; budget, the largest difference of a budget-file flow or a
listing-table rate divided by the total inflow of its time step (for `:single` beyond the
rounding bound where that applies, see above); the wall time of the whole run in
`:mixed` (including writing the output, and for the first model of a session the
compilation). The `ex-` models are compared with MODFLOW 6 runs made with tight closures
beforehand; the other models are run with MODFLOW 6 when the comparison runs.

The ROCm backend against the CPU backend, over all models:

```@eval
Main.CompatTables.parity_backend_table()
```

The table is produced by the script `docs/parity/collect.jl` in the repository, which
runs the parity tests of the test suite on a machine with a GPU and the MODFLOW 6
reference runs. It is not regenerated when the manual is built.

### Feature test models

Some features are not used by any of the models above. They are compared with MODFLOW 6
in the test suite on small models of their own (in `test/models` of the repository, run
with MODFLOW 6 when the test runs), with the same criteria, on the CPU and ROCm backends
(ROCm against CPU: at most 9.5e-7 m in `:single`, identical in `:double`); these models are
not in the table above.

| Feature | Models (test group) | max \|Δh\| (m), `:double` | max \|Δh\| (m), `:single` |
| :-- | :-- | --: | --: |
| Horizontal flow barriers (HFB6) | `hfb_vhfb` (MODFLOW 6 autotest `test_gwf_vhfb.py`: vertical barriers between convertible layers) with its `NEWTON` and logarithmic averaging cases; `hfb_disu` (autotest `test_gwf_disu_hfb_idomain.py`: DISU with `IDOMAIN`); `hfb_conv` (convertible DIS, barriers changing between stress periods) and its `NEWTON` case; `ex-gwf-u1disv` with a line of barriers (DISV); `disu_staggered` with barriers on staggered DISU connections, and its `NEWTON` case (`hfb`) | 2.8e-8 | 8.0e-5 |
| Time-varying K and storage (TVK6, TVS6) | `tvk01` to `tvk06ts` (MODFLOW 6 autotests `test_gwf_npf_tvk01.py` to `tvk06.py`: `K`, `K33` changes per stress period on DIS and DISV, `K33` not given in NPF, a time series); `tvs01` to `tvs05` (autotest `test_gwf_sto_tvs01.py`: one convertible cell with `NEWTON`, `SS` and `SY` changes, a time series, `DISABLE_STORAGE_CHANGE_INTEGRATION`); `tv_conv` (convertible DIS with transient periods of several steps: TVK with a time series and `K33OVERK`, TVS of confined and convertible cells) with its `NEWTON` and `DISABLE_STORAGE_CHANGE_INTEGRATION` cases (`tv`) | 1.0e-9 | 1.9e-4 (heads of 3000 m in `tvs01`; 1.2e-6 in the others) |

## Large models: the Indus basin and the global 5′ model

ModflowGPU was built for coupled runs with a hydrological model, so it was also compared
with MODFLOW 6 on the models it was made for. These comparisons use the coupling
quantities (baseflow to rivers and capillary rise) as well as heads, with tight closures
(`INNER_RCLOSE` 1e-8, as MODFLOW 6 does not converge at 1e-10 on these models), on the
ROCm backend (on the Indus the CPU backend gives the same digits).

**Indus basin** (204 × 180 cells in 2 layers; RIV, DRN, RCH, EVT, CHD, STO; three monthly
stress periods, and 31 daily ones), against MODFLOW 6.3.0 (the version the coupling uses).
Flows are the largest difference relative to the largest value in the domain:

| Preset | Periods | max \|Δh\| (m) | Baseflow | Capillary rise |
| :-- | :-- | --: | --: | --: |
| `:double`, `:mixed` | monthly | 6.9e-11 | 2.2e-12 | 6.4e-14 |
| `:double`, `:mixed` | daily | 3.5e-11 | 1.5e-12 | 4.1e-13 |
| `:single` | monthly | 4.6e-5 | 1.8e-6 | 1.8e-7 |
| `:single` | daily | 1.9e-5 | 3.1e-6 | 2.9e-7 |

**Global 5′ model** (3 675 450 active cells: the land south of 60° N, with constant heads
at sea level along the coast; 31 daily steps driven by recorded output of the hydrological
model), against MODFLOW 6.3.0:

| Preset | max \|Δh\| (m) | Baseflow | Capillary rise | Budget terms per day |
| :-- | --: | --: | --: | --: |
| `:double`, `:mixed` | 4.4e-11 | 3.6e-12 | 5.3e-12 | 1.3e-12 relative |
| `:single` | 2.5e-5 | 1.2e-5 | 3.9e-6 | 1e-6 relative |

Per day MODFLOW 6 needed 7 outer and 14–15 inner iterations; ModflowGPU 7 and 15 (5 and
9 in `:single`).

At production closures (`OUTER_DVCLOSE` of 5 m) MODFLOW 6 stops up to 1.9 m from its own
tight-closure heads on monthly Indus periods, while ModflowGPU stops within 1e-5 m of
them. The two programs then differ visibly, and the difference is MODFLOW 6 stopping
early, not a difference in the equations. With daily periods at the same closure both
stop about 0.15 m from the tight heads, and agree with each other to 1e-6 m.

## Check it yourself

Run MODFLOW 6 and ModflowGPU on the same simulation and compare the output files with
the readers of ModflowGPU, which read the files of both programs. Use tight closures in
the IMS file of both runs (or pass `closures` to [`run_simulation`](@ref)) if you want to
compare the equations rather than the stopping points.

```julia
using ModflowGPU

simdir = "path/to/simulation"          # MODFLOW 6 has run here: model.hds, model.cbc, model.lst
r = run_simulation(simdir; backend = select_backend("auto"), outdir = mktempdir())

# Heads: the largest difference over all saved steps
ours, mf6 = open_head_file(r.head_file), open_head_file(joinpath(simdir, "model.hds"))
dh = maximum(maximum(abs.(read_head(ours, k) .- read_head(mf6, k))) for k in 1:length(mf6))

# A package's flows in the last saved step
a, b = open_budget_file(r.budget_file), open_budget_file(joinpath(simdir, "model.cbc"))
ia = last(ModflowGPU.find_budget_records(a; text = "RIV"))
ib = last(ModflowGPU.find_budget_records(b; text = "RIV"))
dq = maximum(abs.(read_budget(a, ia).q .- read_budget(b, ib).q))

# The listing budget tables: rate in and out per package, and the percent discrepancy
ModflowGPU.listing_rates(r.listing_file, "RIV"), ModflowGPU.listing_rates(joinpath(simdir, "model.lst"), "RIV")
[t.rate_percent_discrepancy for t in read_listing_budgets(r.listing_file)]
```

Dry cells hold the same `HDRY` value in both head files; leave them out of the
difference. The readers are described on [Output files](../api/output.md):
[`open_head_file`](@ref), [`read_head`](@ref), [`open_budget_file`](@ref),
[`find_budget_records`](@ref ModflowGPU.find_budget_records), [`read_budget`](@ref), [`read_listing_budgets`](@ref),
and [`listing_rates`](@ref ModflowGPU.listing_rates).
