# Running a simulation

```@meta
CurrentModule = ModflowGPU
DocTestSetup = :(using ModflowGPU)
```

Run an MF6 simulation directory from start to end with [`run_simulation`](@ref). It
returns a [`RunResult`](@ref) with the final heads and the budget of every time step, and
writes the output files that the OC package of the input asks for. MF6 input is read by
[`load_simulation`](@ref), which [`run_simulation`](@ref) calls for you; call it yourself
to inspect or change a simulation before running it.

```julia
using ModflowGPU
result = run_simulation("path/to/simulation"; backend = select_backend("auto"))
result.converged
percent_discrepancy(result.budgets[end])
```

```@docs
run_simulation
ModflowGPU.run
RunResult
ModelBudget
BudgetTerm
cum_percent_discrepancy
ConvergenceError
load_simulation
MF6UnsupportedError
ModflowGPU.MF6InputError
```
