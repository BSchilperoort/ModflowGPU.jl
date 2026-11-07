# BasicModelInterface on the stepping API (src/stepping.jl), as libmf6's BMI
# (srcbmi/mf6bmi.f90, mf6bmiGrid.f90): variable names are libmf6 addresses, values are on
# the host in Float64 with absolute heads (Int32 for indices), one grid per model (id 1).

import BasicModelInterface as BMI

"""
    BMI.initialize(::Type{Simulation}, config_file; kwargs...) -> Simulation

`config_file` is the simulation directory or its `mfsim.nam`; the keywords are those of
`initialize` (`backend`, `precision`, `outdir`, `closures`, `write_output`).
"""
function BMI.initialize(::Type{Simulation}, config_file::AbstractString; kwargs...)
    dir = isdir(config_file) ? config_file : dirname(abspath(config_file))
    return initialize(dir; kwargs...)
end

"`BMI.update`: one time step (`update`). Throws a `ConvergenceError` when it fails without CONTINUE."
function BMI.update(s::Simulation)
    if !update(s)
        r = last(s.results)
        throw(ConvergenceError(r.step.kper, r.step.kstp, r.outer.iterations))
    end
    return nothing
end

"`BMI.update_until(sim, time)`: time steps until the current time reaches `time`."
function BMI.update_until(s::Simulation, time::Float64)
    while get_current_time(s) < time && s.istep < length(s.steps)
        BMI.update(s)
    end
    return nothing
end

BMI.finalize(s::Simulation) = (finalize(s); nothing)

BMI.get_component_name(::Simulation) = "ModflowGPU (MODFLOW 6 GWF)"
BMI.get_input_var_names(s::Simulation) = input_addresses(s)
BMI.get_output_var_names(s::Simulation) = var_addresses(s)
BMI.get_input_item_count(s::Simulation) = length(BMI.get_input_var_names(s))
BMI.get_output_item_count(s::Simulation) = length(BMI.get_output_var_names(s))

function BMI.get_var_grid(s::Simulation, name::String)
    _variable(s, name)
    startswith(uppercase(name), uppercase(s.model.spec.name) * "/") ||
        throw(ArgumentError("get_var_grid: \"$name\" is not a model variable"))
    return 1
end

BMI.get_var_type(s::Simulation, name::String) = string(eltype(get_value(s, name)))
BMI.get_var_units(s::Simulation, name::String) = _variable(s, name).units
BMI.get_var_itemsize(s::Simulation, name::String) = sizeof(eltype(get_value(s, name)))
BMI.get_var_nbytes(s::Simulation, name::String) = sizeof(get_value(s, name))
BMI.get_var_location(s::Simulation, name::String) = (_variable(s, name); "node")

BMI.get_current_time(s::Simulation) = get_current_time(s)
BMI.get_start_time(s::Simulation) = get_start_time(s)
BMI.get_end_time(s::Simulation) = get_end_time(s)
BMI.get_time_step(s::Simulation) = get_time_step(s)
BMI.get_time_units(s::Simulation) = lowercase(String(s.spec.tdis.time_units))

BMI.get_value(s::Simulation, name::String, dest::AbstractArray) = copyto!(dest, vec(get_value(s, name)))

"""
    BMI.get_value_ptr(sim, name) -> Array

The host mirror of `name` (`get_value_ptr(sim, name)`; Float64, absolute heads). It is a
copy of the device data, synchronised at the stepping calls (see src/stepping.jl);
writes into it take effect at the next time step.
"""
BMI.get_value_ptr(s::Simulation, name::String) = get_value_ptr(s, name)

BMI.get_value_at_indices(s::Simulation, name::String, dest::AbstractArray, inds::AbstractVector{<:Integer}) =
    copyto!(dest, vec(get_value(s, name))[inds])

BMI.set_value(s::Simulation, name::String, value::AbstractArray) = (set_value(s, name, value); nothing)

function BMI.set_value_at_indices(s::Simulation, name::String, inds::AbstractVector{<:Integer}, value::AbstractArray)
    full = vec(get_value(s, name))
    full[inds] .= vec(value)
    set_value(s, name, full)
    return nothing
end

# Grid 1: the model grid (`mf6bmiGrid.f90`): DIS is "rectilinear" with shape
# (nlay, nrow, ncol) (rank 2 when nlay is 1), DISV/DISU are "unstructured".
function _grid(s::Simulation, grid::Integer)
    grid == 1 || throw(ArgumentError("unknown grid $grid (the model grid is 1)"))
    return s.model.spec.dis
end

BMI.get_grid_type(s::Simulation, grid::Int) = _grid(s, grid) isa DisSpec ? "rectilinear" : "unstructured"

function BMI.get_grid_shape(s::Simulation, grid::Int, shape::AbstractVector{<:Integer})
    d = _grid(s, grid)
    d isa DisSpec || throw(ArgumentError("get_grid_shape: grid $grid is unstructured"))
    sh = d.nlay == 1 ? [d.nrow, d.ncol] : [d.nlay, d.nrow, d.ncol]
    return copyto!(shape, sh)
end

function BMI.get_grid_rank(s::Simulation, grid::Int)
    d = _grid(s, grid)
    return d isa DisSpec && d.nlay == 1 ? 2 : 3
end

BMI.get_grid_size(s::Simulation, grid::Int) = (_grid(s, grid); Int(s.model.conn.nodesuser))
BMI.get_grid_node_count(s::Simulation, grid::Int) = BMI.get_grid_size(s, grid)

"`BMI.get_grid_x`: the column edges, `xorigin + [0; cumsum(delr)]` (DIS)."
function BMI.get_grid_x(s::Simulation, grid::Int, x::AbstractVector{Float64})
    d = _grid(s, grid)
    d isa DisSpec || throw(ArgumentError("get_grid_x: grid $grid is unstructured"))
    return copyto!(x, d.xorigin .+ [0.0; cumsum(d.delr)])
end

"`BMI.get_grid_y`: the row edges from the first (north) row, `yorigin + Σdelc − [0; cumsum(delc)]` (DIS)."
function BMI.get_grid_y(s::Simulation, grid::Int, y::AbstractVector{Float64})
    d = _grid(s, grid)
    d isa DisSpec || throw(ArgumentError("get_grid_y: grid $grid is unstructured"))
    return copyto!(y, d.yorigin + sum(d.delc) .- [0.0; cumsum(d.delc)])
end
