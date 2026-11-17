# Building models

```@meta
CurrentModule = ModflowGPU
DocTestSetup = :(using ModflowGPU)
```

The builder functions make a [`SimulationSpec`](@ref) from Julia arrays, with the checks
MF6 applies to its input. Grid arrays have the shape `(ncol, nrow, nlay)` (MF6's memory
order); most values may also be a scalar or one value per layer. The result can be run
directly with [`run_simulation`](@ref) or [`initialize`](@ref), or written as MF6 input
files with [`write_simulation`](@ref), which MF6 runs too.

Only DIS grids can be built so far. Units are those of the model: lengths in its length
units (`length_units` of [`build_dis`](@ref)), times in its time units (`time_units` of
[`build_tdis`](@ref)).

```jldoctest
julia> dis = build_dis(; delr = 100.0, delc = 100.0, top = 0.0, botm = fill(-10.0, 10, 5, 1));

julia> chd = build_boundary(:CHD, dis; periods = [1 => (; cells = [(1, 1, 1), (1, 5, 10)], values = [0.0, -2.0])]);

julia> gwf = build_model("flow"; dis, npf = build_npf(dis; k = 10.0), ic = build_ic(dis; strt = 0.0), boundaries = [chd]);

julia> sim = build_simulation(gwf; tdis = build_tdis(; perlen = 1.0))
SimulationSpec(1 model(s): FLOW; 1 period(s); 0 exchange(s))
```

```@docs
SimulationSpec
build_dis
build_npf
build_ic
build_sto
oc_record
build_oc
build_boundary
build_array_boundary
build_model
build_tdis
build_ims
build_simulation
write_simulation
```
