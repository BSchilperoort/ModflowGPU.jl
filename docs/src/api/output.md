# Output files

```@meta
CurrentModule = ModflowGPU
DocTestSetup = :(using ModflowGPU)
```

ModflowGPU writes the head file (`.hds`), the budget file (`.cbc`, with `DATA-SPDIS` when
NPF has `SAVE_SPECIFIC_DISCHARGE`), the binary grid file (`.dis.grb`, `.disv.grb`,
`.disu.grb`, unless `NOGRB`), the model listing file (`.lst`), the observation files
(OBS6) and the CSV files (`BUDGETCSV`, `CSV_OUTER_OUTPUT`, `CSV_INNER_OUTPUT`,
`AUTO_FLOW_REDUCE_CSV`) in MF6's formats (see
[Differences from MODFLOW 6](../compatibility/differences.md)). The readers on this page
read these files from ModflowGPU and from MF6 alike, so they can also be used to compare
the two. The binary files can be read by
[FloPy](https://github.com/modflowpy/flopy) as well.

Head and budget files are indexed when opened (the record headers only); the data of a
record is read when asked for.

```julia
using ModflowGPU
hf = open_head_file("path/to/simulation/model.hds")
h = read_head(hf, length(hf))            # the last saved heads, (ncol, nrow, nlay)

bf = open_budget_file("path/to/simulation/model.cbc")
wel = read_budget(bf; text = "WEL")      # one BudgetList per time step
```

```@docs
open_head_file
HeadFile
ModflowGPU.HeadStep
read_head
ModflowGPU.find_head_step
ModflowGPU.head_kstpkper
ModflowGPU.head_times
ModflowGPU.HNOFLO
ModflowGPU.HDRY
open_budget_file
BudgetFile
ModflowGPU.BudgetRecord
BudgetList
read_budget
ModflowGPU.find_budget_records
ModflowGPU.budget_texts
ModflowGPU.budget_kstpkper
ModflowGPU.budget_times
read_grb
BinaryGridFile
ModflowGPU.GrbVariable
read_listing_budgets
ListingBudget
ModflowGPU.ListingBudgetTerm
ModflowGPU.listing_rates
ModflowGPU.percent_discrepancy
ModflowGPU.MF6BinaryFileError
```

## Observation files

The observation output of OBS6 input (CSV or `BINARY`), of ModflowGPU and of MF6; the
paths of a run are in [`RunResult`](@ref)`.obs_files`.

```julia
obs = read_observations("path/to/simulation/model.obs.head.csv")
obs.names, obs.times, obs.values   # values: one row per time step, one column per observation
```

```@docs
read_observations
ObservationFile
ModflowGPU.OBSNODATA
```

## CSV output files

The CSV files of OC `BUDGETCSV`, IMS `CSV_OUTER_OUTPUT` and `CSV_INNER_OUTPUT` and WEL
`AUTO_FLOW_REDUCE_CSV` are written as MF6 writes them (columns on
[Supported packages](@ref "CSV output")); the paths of a run are in
[`RunResult`](@ref)`.csv_files`. They are plain CSV with a header line, readable with
any CSV reader, e.g. with `DelimitedFiles`:

```julia
using DelimitedFiles
result = run_simulation("path/to/simulation")
data, header = readdlm(result.csv_files[1], ','; header = true)
```
