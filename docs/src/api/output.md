# Output files

```@meta
CurrentModule = ModflowGPU
DocTestSetup = :(using ModflowGPU)
```

ModflowGPU writes the head file (`.hds`), the budget file (`.cbc`) and the model listing
file (`.lst`) in MF6's formats. It does not write the binary grid file (`.dis.grb` and
the like), but [`read_grb`](@ref) reads MF6's (see
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
find_head_step
head_kstpkper
head_times
ModflowGPU.HNOFLO
ModflowGPU.HDRY
open_budget_file
BudgetFile
ModflowGPU.BudgetRecord
BudgetList
read_budget
find_budget_records
budget_texts
budget_kstpkper
budget_times
read_grb
BinaryGridFile
ModflowGPU.GrbVariable
read_listing_budgets
ListingBudget
ModflowGPU.ListingBudgetTerm
listing_rates
percent_discrepancy
ModflowGPU.MF6BinaryFileError
```
