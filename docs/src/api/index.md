# API reference

```@meta
CurrentModule = ModflowGPU
DocTestSetup = :(using ModflowGPU)
```

The public API of ModflowGPU, one page per area:

- [Running a simulation](running.md): [`run_simulation`](@ref) and its result, loading
  MF6 input, errors.
- [Backends and precision](backends.md): choosing the CPU or a GPU, and the precision
  presets.
- [Stepping API](stepping.md): running a simulation one time step at a time and reading
  or setting its variables, for coupling.
- [Building models](builder.md): a simulation from Julia arrays, written as MF6 input.
- [Output files](output.md): reading head, budget, binary grid and listing files.
- [For backend authors](backend_authors.md): the hooks a GPU package extension
  implements.

Names written as `ModflowGPU.name` on these pages are public but not exported: write them
qualified. All others are exported by `using ModflowGPU`.

`ModflowGPU.run` and `ModflowGPU.finalize`, the names used before version 0.1 exported
anything, are the same functions as [`run_simulation`](@ref) and
[`finalize_simulation`](@ref) and keep working. They are not exported because they would
clash with `Base.run` and `Base.finalize`.

The [BMI](https://bmi.csdms.io) functions are methods of BasicModelInterface's own
functions for [`Simulation`](@ref) (the BMI specification) and [`MF6Model`](@ref)
(libmf6's BMI, as ModflowInterface.jl) and are called qualified, for example
`BasicModelInterface.update(sim)`; see [BMI](../coupling/bmi.md).

```@docs
ModflowGPU
```
