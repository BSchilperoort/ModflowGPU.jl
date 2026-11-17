# Stepping API

```@meta
CurrentModule = ModflowGPU
DocTestSetup = :(using ModflowGPU)
```

The stepping API runs a simulation one time step at a time, as libmf6's API (XMI) does,
so that another model can read and change its variables between steps.
[`initialize`](@ref) builds the model; [`update`](@ref) runs one time step, or the calls
it is made of ([`prepare_time_step`](@ref), then [`do_time_step`](@ref) or
[`prepare_solve`](@ref), [`ModflowGPU.solve`](@ref) and [`finalize_solve`](@ref), then
[`finalize_time_step`](@ref)); [`finalize_simulation`](@ref) ends it. Variables are named
by libmf6's addresses, such as `"GWF/X"` (heads) or `"GWF/RIV-1/BOUND"` (river data), and
are read and written as host arrays in Float64 with absolute heads.

```julia
using ModflowGPU
sim = initialize("path/to/simulation"; backend = select_backend("auto"))
riv = get_value_ptr(sim, "GWF/RIV-1/BOUND")   # (3, nriv): stage, cond, rbot
while get_current_time(sim) < get_end_time(sim)
    prepare_time_step(sim)
    riv[1, :] .= my_river_stages(get_current_time(sim))
    do_time_step(sim) || @warn "no convergence"
    finalize_time_step(sim)
    heads = get_value(sim, "GWF/X")
end
finalize_simulation(sim)
```

```@docs
Simulation
StepResult
initialize
prepare_time_step
prepare_solve
ModflowGPU.solve
finalize_solve
finalize_time_step
do_time_step
update
finalize_simulation
ModflowGPU.finalize
get_value
get_value_ptr
set_value
ModflowGPU.device_bound!
get_var_address
var_addresses
input_addresses
get_current_time
get_start_time
get_end_time
get_time_step
```
