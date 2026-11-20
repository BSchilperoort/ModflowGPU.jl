# MF6Model (T8.10): the BMI and XMI of libmf6 as ModflowInterface.jl wraps them
# (Deltares/ModflowInterface.jl, `ModflowModel`), so that a script written for libmf6 runs
# on ModflowGPU by swapping the model type. A thin handle around a `Simulation`: the
# stepping API (src/stepping.jl) does the work; this file only gives each call libmf6's
# result (srcbmi/mf6bmi.f90, mf6bmiGrid.f90, mf6xmi.F90, MF6 6.8.2):
#
# - one grid per model (id 1, `get_var_grid` of every model address; -1 for SLN_1/TDIS),
#   "rectilinear" for DIS, "unstructured" for DISV and DISU; rank, shape, node and face
#   functions only where libmf6 implements them (DIS: rank, size, shape, x, y; DISU:
#   size, x, y, node count, face count, face nodes, nodes per face; DISV: x, y);
# - DIS x and y are the indices 0:ncol and nrow:-1:0, not coordinates;
#   unstructured x and y the VERTICES as given (model coordinates);
# - `get_value_ptr` with libmf6's shapes (scalars 0-dimensional, BOUND (ncolbnd, maxbound),
#   NODEUSER and NODEREDUCED of length 1 when no cell is excluded) and type strings
#   (`"DOUBLE (57)"`, `"INTEGER"`, `"DOUBLE (1,2)"`);
# - a failure status of libmf6 is an exception here (also kept for `get_last_bmi_error`);
#   functions that libmf6 does not have throw and point to `Simulation`'s BMI.
#
# Simulation's own BMI (src/bmi.jl, T8.9: the BMI specification) is unchanged.

const _MF6MODEL_KEYWORDS = (:backend, :precision, :outdir, :closures, :write_output, :preconditioner, :precond_options)

"""
    MF6Model(path; backend, precision, preconditioner, precond_options, closures, outdir, write_output)
    MF6Model(spec::SimulationSpec; kwargs...)

A model with the BMI of libmf6, used as ModflowInterface.jl's `ModflowModel`: create it
with the simulation directory or its `mfsim.nam` (or a [`SimulationSpec`](@ref)), then
call `BasicModelInterface.initialize(m)`. The keywords are the engine settings of
[`initialize`](@ref); they are kept for every (re-)initialisation.

The BMI functions return what libmf6 returns (variable addresses, shapes and type
strings, one grid per model); the XMI functions of ModflowInterface.jl
([`prepare_time_step`](@ref), [`prepare_solve`](@ref), [`ModflowGPU.solve`](@ref),
[`finalize_solve`](@ref), [`finalize_time_step`](@ref), [`do_time_step`](@ref),
[`get_var_address`](@ref), [`ModflowGPU.get_var_rank`](@ref),
[`ModflowGPU.get_var_shape`](@ref), [`ModflowGPU.get_subcomponent_count`](@ref),
[`ModflowGPU.get_last_bmi_error`](@ref)) take it as their first argument and return it
where ModflowInterface.jl does. BMI functions that libmf6 does not implement throw an
`ArgumentError`; [`Simulation`](@ref) has the full BMI of the specification. The
simulation of an initialised model is `m.simulation` (`nothing` before
`BMI.initialize`). As ModflowInterface.jl's `ModflowModel`, a model that is garbage
collected is finalised (in a new task, after the collection); call `BMI.finalize(m)` to
close the output files and free the device memory at a known point.

# Example
```julia
import BasicModelInterface as BMI
using ModflowGPU
const MF = ModflowGPU

m = MF6Model("path/to/simulation"; backend = select_backend("auto"))
BMI.initialize(m)
head = BMI.get_value_ptr(m, "GWF/X")
while BMI.get_current_time(m) < BMI.get_end_time(m)
    MF.prepare_time_step(m, 0.0)
    MF.prepare_solve(m, 1)
    for _ in 1:BMI.get_value_ptr(m, "SLN_1/MXITER")[]
        MF.solve(m, 1) && break
    end
    MF.finalize_solve(m, 1)
    MF.finalize_time_step(m)
end
BMI.finalize(m)
```
"""
mutable struct MF6Model
    const source::Union{String, SimulationSpec}
    const settings::NamedTuple
    simulation::Union{Nothing, Simulation}
    last_error::String
    # libmf6-shaped arrays that are not host mirrors of the stepping API (the length-1
    # NODEUSER, NODEREDUCED), kept for the run as libmf6 keeps its memory.
    const arrays::Dict{String, Array}
    function MF6Model(source, settings, simulation, last_error, arrays)
        m = new(source, settings, simulation, last_error, arrays)
        # As ModflowInterface.jl's ModflowModel: finalize when garbage collected.
        return finalizer(_mf6_gc_finalize, m)
    end
end

# The finalizer of an MF6Model. It runs inside the garbage collector, where a task switch,
# I/O or a device synchronisation is not allowed, while `finalize_simulation` closes the
# output files and synchronises the device: detach the simulation and finalize it in a
# new task.
function _mf6_gc_finalize(m::MF6Model)
    sim = m.simulation
    isnothing(sim) && return nothing
    m.simulation = nothing
    Threads.@spawn try
        finalize_simulation(sim)
    catch e
        @error "MF6Model finalizer: finalize_simulation failed" exception = (e, catch_backtrace())
    end
    return nothing
end

function _mf6model_settings(kwargs)
    for k in keys(kwargs)
        k in _MF6MODEL_KEYWORDS || throw(ArgumentError("MF6Model: unknown keyword `$k` (expected one of $(join(_MF6MODEL_KEYWORDS, ", ")))"))
    end
    return NamedTuple(kwargs)
end

function MF6Model(path::AbstractString; kwargs...)
    dir = if isdir(path)
        isfile(joinpath(path, "mfsim.nam")) || throw(ArgumentError("MF6Model: no mfsim.nam file found in $path"))
        path
    elseif isfile(path) && lowercase(basename(path)) == "mfsim.nam"
        dirname(abspath(path))
    else
        throw(ArgumentError("MF6Model: $path is not a simulation directory or its mfsim.nam"))
    end
    return MF6Model(normpath(abspath(dir)), _mf6model_settings(kwargs), nothing, "", Dict{String, Array}())
end

MF6Model(spec::SimulationSpec; kwargs...) = MF6Model(spec, _mf6model_settings(kwargs), nothing, "", Dict{String, Array}())

function Base.show(io::IO, m::MF6Model)
    name = m.source isa String ? basename(m.source) : basename(m.source.simdir)
    return print(io, "MF6Model(", repr(name), ", ", isnothing(m.simulation) ? "uninitialized" : "initialized", ")")
end

# Run `f()`, keeping the message of an error for `get_last_bmi_error` (libmf6 keeps the
# message of a failed call).
function _mf6call(f, m::MF6Model)
    try
        return f()
    catch e
        m.last_error = sprint(showerror, e)
        rethrow()
    end
end

function _mf6sim(m::MF6Model)
    isnothing(m.simulation) && throw(ArgumentError("the MF6Model is not initialized (call BasicModelInterface.initialize(m))"))
    return m.simulation
end

_mf6_not_libmf6(f::Symbol) = throw(
    ArgumentError(
        "BMI.$f is not implemented by libmf6, and MF6Model follows libmf6; use the BMI of ModflowGPU.Simulation (BMI.initialize(Simulation, config_file), or m.simulation)",
    ),
)

_mf6_failure(f::Symbol, why) = throw(ArgumentError("BMI.$f: $why (libmf6 returns a failure status)"))

# ---------------------------------------------------------------------------------------
# Initialize, update, finalize, time
# ---------------------------------------------------------------------------------------

"""
    BMI.initialize(m::MF6Model) -> m

Load the simulation and build the model (libmf6's `initialize`). On an initialised model,
finalize it and initialize again, as ModflowInterface.jl does.
"""
function BMI.initialize(m::MF6Model)
    return _mf6call(m) do
        isnothing(m.simulation) || BMI.finalize(m)
        empty!(m.arrays)
        m.simulation = initialize(m.source; m.settings...)
        m
    end
end

"`BMI.initialize(MF6Model)`: the simulation in the current directory, with the default settings (as ModflowInterface.jl)."
BMI.initialize(::Type{MF6Model}) = BMI.initialize(MF6Model(pwd()))

BMI.initialize(::Type{MF6Model}, config_file) = throw(
    ArgumentError("BMI.initialize(MF6Model, config_file) is not supported: use BMI.initialize(MF6Model(path; kwargs...)), or BMI.initialize(Simulation, config_file)"),
)

"""
    BMI.finalize(m::MF6Model) -> m

Close the output files and free the device memory (libmf6's `finalize`); the model can be
initialised again. Does nothing on an uninitialised model.
"""
function BMI.finalize(m::MF6Model)
    isnothing(m.simulation) && return m
    _mf6call(() -> finalize_simulation(m.simulation), m)
    m.simulation = nothing
    return m
end

# libmf6's `finalize_time_step` reports non-convergence (without CONTINUE) as a failure
# status with this message, and goes on; ModflowInterface.jl ignores the status.
function _mf6_converged!(m::MF6Model, ok::Bool)
    ok && return nothing
    r = last(_mf6sim(m).results)
    m.last_error = "simulation failed to converge"
    @warn "MF6Model: time step $(r.step.kstp) of stress period $(r.step.kper) did not converge ($(r.outer.iterations) outer iterations)"
    return nothing
end

"""
    BMI.update(m::MF6Model) -> m

One time step (libmf6's `update`). A step that does not converge does not throw (libmf6
reports success); it gives a warning and sets [`ModflowGPU.get_last_bmi_error`](@ref).
"""
function BMI.update(m::MF6Model)
    ok = _mf6call(() -> update(_mf6sim(m)), m)
    _mf6_converged!(m, ok)
    return m
end

BMI.get_component_name(::MF6Model) = "MODFLOW 6"
BMI.get_start_time(::MF6Model) = 0.0
BMI.get_current_time(m::MF6Model) = _mf6call(() -> get_current_time(_mf6sim(m)), m)
BMI.get_end_time(m::MF6Model) = _mf6call(() -> get_end_time(_mf6sim(m)), m)
BMI.get_time_step(m::MF6Model) = _mf6call(() -> get_time_step(_mf6sim(m)), m)

# ---------------------------------------------------------------------------------------
# XMI (the non-BMI functions of ModflowInterface.jl)
# ---------------------------------------------------------------------------------------

"`prepare_time_step(m::MF6Model, dt) -> m`: [`prepare_time_step`](@ref) of the simulation (`dt` is ignored, as in libmf6)."
prepare_time_step(m::MF6Model, dt::Real) = _mf6call(() -> (prepare_time_step(_mf6sim(m), dt); m), m)

"`prepare_solve(m::MF6Model, component_id = 1) -> m`: [`prepare_solve`](@ref) of the simulation."
prepare_solve(m::MF6Model, component_id::Integer = 1) = _mf6call(() -> (prepare_solve(_mf6sim(m), component_id); m), m)

"`ModflowGPU.solve(m::MF6Model, component_id = 1) -> Bool`: one outer iteration; whether the step converged."
solve(m::MF6Model, component_id::Integer = 1) = _mf6call(() -> solve(_mf6sim(m), component_id), m)

"`finalize_solve(m::MF6Model, component_id = 1) -> m`: [`finalize_solve`](@ref) of the simulation; non-convergence sets [`ModflowGPU.get_last_bmi_error`](@ref)."
function finalize_solve(m::MF6Model, component_id::Integer = 1)
    ok = _mf6call(() -> finalize_solve(_mf6sim(m), component_id), m)
    ok || (m.last_error = "Solution $component_id did not converge")
    return m
end

"`finalize_time_step(m::MF6Model) -> m`: [`finalize_time_step`](@ref) of the simulation; non-convergence (without CONTINUE) warns and sets [`ModflowGPU.get_last_bmi_error`](@ref), as libmf6's failure status."
function finalize_time_step(m::MF6Model)
    ok = _mf6call(() -> finalize_time_step(_mf6sim(m)), m)
    _mf6_converged!(m, ok)
    return m
end

"`do_time_step(m::MF6Model) -> m`: [`do_time_step`](@ref) of the simulation."
function do_time_step(m::MF6Model)
    ok = _mf6call(() -> do_time_step(_mf6sim(m)), m)
    ok || (m.last_error = "Solution 1 did not converge")
    return m
end

"""
    ModflowGPU.get_subcomponent_count(m::MF6Model) -> Int

The number of solutions (libmf6's `get_subcomponent_count`): always 1.
"""
get_subcomponent_count(m::MF6Model) = (_mf6sim(m); 1)

"`get_var_address(m::MF6Model, var, component; subcomponent_name = \"\")`: as for a [`Simulation`](@ref)."
get_var_address(m::MF6Model, var::AbstractString, component::AbstractString; subcomponent_name::AbstractString = "") =
    join(filter(!isempty, uppercase.([component, subcomponent_name, var])), "/")

"""
    ModflowGPU.get_last_bmi_error(m::MF6Model) -> String

The message of the last error of a call on `m` (libmf6's `get_last_bmi_error`), or of the
last time step that did not converge; `""` if there was none.
"""
get_last_bmi_error(m::MF6Model) = m.last_error

# ---------------------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------------------

# Whether `address` (upper case) is a scalar in libmf6 (rank 0).
function _mf6_scalar(parts)
    parts[1] in ("SLN_1", "TDIS") && return true
    return length(parts) == 3 && parts[3] in ("NODES", "NODESUSER", "NBOUND", "MAXBOUND")
end

# The libmf6-shaped array of `address`: the host mirror of the stepping API (a
# 0-dimensional view of it for a scalar), or, for NODEUSER and NODEREDUCED of a model
# without excluded cells, a dummy of length 1 (libmf6 allocates them so; the value is
# meaningless there).
function _mf6_ptr(m::MF6Model, address::AbstractString)
    s = _mf6sim(m)
    parts = split(uppercase(address), '/')
    _variable(s, address)   # checks the address
    c = length(parts) == 3 ? _vconn(_find_view(s, parts[1])) : nothing
    if length(parts) == 3 && parts[3] in ("NODEUSER", "NODEREDUCED") && c.nodes == c.nodesuser
        return get!(() -> Int32[0], m.arrays, join(parts, "/"))
    end
    p = get_value_ptr(s, address)
    return _mf6_scalar(parts) ? reshape(p, ()) : p
end

_mf6_typename(::Type{Float64}) = "DOUBLE"
_mf6_typename(::Type{Int32}) = "INTEGER"

"""
    BMI.get_value_ptr(m::MF6Model, name) -> Array

The variable `name` as libmf6 shapes it: a 0-dimensional array for a scalar, a vector,
or a `(ncolbnd, maxbound)` matrix for BOUND; Float64 with absolute heads, Int32 for
indices. It is the host mirror of [`get_value_ptr`](@ref) (synchronised with the device
at the stepping calls) and stays valid until `BMI.finalize`.
"""
BMI.get_value_ptr(m::MF6Model, name::AbstractString) = _mf6call(() -> _mf6_ptr(m, name), m)
BMI.get_value(m::MF6Model, name::AbstractString) = copy(BMI.get_value_ptr(m, name))
BMI.get_value(m::MF6Model, name::AbstractString, dest::AbstractArray) = copyto!(dest, BMI.get_value_ptr(m, name))

"`BMI.set_value(m::MF6Model, name, values)`: [`set_value`](@ref) of the simulation (libmf6's `set_value`; only the addresses of [`input_addresses`](@ref) are writable here)."
BMI.set_value(m::MF6Model, name::AbstractString, values::AbstractArray) = _mf6call(() -> (set_value(_mf6sim(m), name, values); nothing), m)

BMI.get_input_var_names(m::MF6Model) = _mf6call(() -> var_addresses(_mf6sim(m)), m)
BMI.get_output_var_names(m::MF6Model) = BMI.get_input_var_names(m)
BMI.get_input_item_count(m::MF6Model) = length(BMI.get_input_var_names(m))
BMI.get_output_item_count(m::MF6Model) = length(BMI.get_output_var_names(m))

"`BMI.get_var_type(m::MF6Model, name)`: libmf6's memory type, e.g. `\"DOUBLE (57)\"`, `\"INTEGER\"`, `\"DOUBLE (3,120)\"` (the Fortran dimensions, i.e. Julia's `size`)."
function BMI.get_var_type(m::MF6Model, name::AbstractString)
    p = BMI.get_value_ptr(m, name)
    t = _mf6_typename(eltype(p))
    return ndims(p) == 0 ? t : "$t ($(join(size(p), ',')))"
end
BMI.get_var_itemsize(m::MF6Model, name::AbstractString) = sizeof(eltype(BMI.get_value_ptr(m, name)))
BMI.get_var_nbytes(m::MF6Model, name::AbstractString) = sizeof(BMI.get_value_ptr(m, name))

"`ModflowGPU.get_var_rank(m::MF6Model, name) -> Int`: the number of dimensions of `name` (libmf6's `get_var_rank`; 0 for a scalar)."
get_var_rank(m::MF6Model, name::AbstractString) = ndims(BMI.get_value_ptr(m, name))

"`ModflowGPU.get_var_shape(m::MF6Model, name) -> Tuple`: the size of `name` in Julia order (as ModflowInterface.jl, which reverses libmf6's C-order shape)."
get_var_shape(m::MF6Model, name::AbstractString) = size(BMI.get_value_ptr(m, name))

"""
    BMI.get_var_grid(m::MF6Model, name) -> Int

The grid of the model that `name` belongs to, as libmf6: 1 for every address of the
model (`"<model>/..."`; with several models the model's number in the solution, T6.6), -1
for the solution and TDIS addresses (libmf6 reports a
failure status, which ModflowInterface.jl ignores).
"""
function BMI.get_var_grid(m::MF6Model, name::AbstractString)
    return _mf6call(m) do
        s = _mf6sim(m)
        _variable(s, name)
        # T6.6: model k of a multi-model solution has grid k (libmf6: the model id).
        k = findfirst(v -> uppercase(_vname(v)) == first(split(uppercase(name), '/')), s.views)
        isnothing(k) ? -1 : k
    end
end

for f in (:update_until, :get_var_units, :get_var_location, :get_time_units)
    @eval BMI.$f(::MF6Model, args...) = _mf6_not_libmf6($(QuoteNode(f)))
end
for f in (:get_value_at_indices, :set_value_at_indices, :get_grid_z, :get_grid_spacing, :get_grid_origin)
    @eval BMI.$f(::MF6Model, args...) = _mf6_not_libmf6($(QuoteNode(f)))
end
for f in (:get_grid_edge_count, :get_grid_edge_nodes, :get_grid_face_edges)
    @eval BMI.$f(::MF6Model, args...) = _mf6_not_libmf6($(QuoteNode(f)))
end

# ---------------------------------------------------------------------------------------
# Grids (mf6bmiGrid.f90)
# ---------------------------------------------------------------------------------------

# The DIS spec of grid `grid` (the only grid is 1, the model).
function _mf6_dis(m::MF6Model, f::Symbol, grid::Integer)
    s = _mf6sim(m)
    1 <= grid <= length(s.views) || _mf6_failure(f, "no model for grid id $grid (the models' grids are 1 to $(length(s.views)))")
    return _vspec(s.views[grid]).dis
end

# The DIS spec of grid `grid` if its type is `T` (libmf6's `confirm_grid_type`), else a
# failure.
function _mf6_dis(m::MF6Model, f::Symbol, grid::Integer, ::Type{T}) where {T}
    d = _mf6_dis(m, f, grid)
    d isa T || _mf6_failure(f, "not available for a $(_mf6_dis_type(d)) grid")
    return d
end

_mf6_dis_type(::DisSpec) = "DIS"
_mf6_dis_type(::DisvSpec) = "DISV"
_mf6_dis_type(::DisuSpec) = "DISU"

# libmf6's MSHAPE: (nlay, nrow, ncol), C order.
_mf6_mshape(d::DisSpec) = d.nlay == 1 ? [Int(d.nrow), Int(d.ncol)] : [Int(d.nlay), Int(d.nrow), Int(d.ncol)]

"`BMI.get_grid_type(m::MF6Model, grid)`: `\"rectilinear\"` for DIS, `\"unstructured\"` for DISV and DISU (libmf6)."
BMI.get_grid_type(m::MF6Model, grid::Integer) =
    _mf6call(() -> _mf6_dis(m, :get_grid_type, grid) isa DisSpec ? "rectilinear" : "unstructured", m)

"`BMI.get_grid_rank(m::MF6Model, grid)`: 2 for a DIS grid of one layer, else 3; DIS only (libmf6)."
BMI.get_grid_rank(m::MF6Model, grid::Integer) = _mf6call(() -> length(_mf6_mshape(_mf6_dis(m, :get_grid_rank, grid, DisSpec))), m)

"`BMI.get_grid_size(m::MF6Model, grid)`: nlay · nrow · ncol for DIS, the number of vertices for DISU (libmf6)."
function BMI.get_grid_size(m::MF6Model, grid::Integer)
    return _mf6call(m) do
        d = _mf6_dis(m, :get_grid_size, grid)
        d isa DisSpec ? prod(_mf6_mshape(d)) : Int(_mf6_dis(m, :get_grid_size, grid, DisuSpec).nvert)
    end
end

"""
    BMI.get_grid_shape(m::MF6Model, grid) -> Tuple
    BMI.get_grid_shape(m::MF6Model, grid, shape) -> shape

The shape of a DIS grid: as ModflowInterface.jl, a Tuple in Julia order `(ncol, nrow)` or
`(ncol, nrow, nlay)`; with `shape`, libmf6's C order `[nrow, ncol]` or
`[nlay, nrow, ncol]` written into it.
"""
BMI.get_grid_shape(m::MF6Model, grid::Integer) = Tuple(reverse(_mf6call(() -> _mf6_mshape(_mf6_dis(m, :get_grid_shape, grid, DisSpec)), m)))
function BMI.get_grid_shape(m::MF6Model, grid::Integer, shape::AbstractVector{<:Integer})
    sh = _mf6call(() -> _mf6_mshape(_mf6_dis(m, :get_grid_shape, grid, DisSpec)), m)
    return _mf6_fill!(m, :get_grid_shape, shape, sh)
end

# Write `values` into the start of `dest` (libmf6 writes as many values as it has).
function _mf6_fill!(m::MF6Model, f::Symbol, dest::AbstractVector, values)
    # Never write past the end of `dest` (libmf6 does: its get_grid_y of DIS writes
    # ncol + 1 values).
    length(dest) >= length(values) ||
        _mf6call(() -> throw(ArgumentError("BMI.$f: the destination has length $(length(dest)), expected at least $(length(values))")), m)
    copyto!(dest, 1, values, 1, length(values))
    return dest
end

"""
    BMI.get_grid_x(m::MF6Model, grid, x) -> x

As libmf6: for DIS the column edge indices `0:ncol` (not coordinates), for DISV and DISU
the x of the VERTICES as given (model coordinates, without XORIGIN and ANGROT). `x` must
have at least `ncol + 1` (DIS) or NVERT values, else an `ArgumentError`.
"""
function BMI.get_grid_x(m::MF6Model, grid::Integer, x::AbstractVector{<:Real})
    d = _mf6call(() -> _mf6_dis(m, :get_grid_x, grid), m)
    return _mf6_fill!(m, :get_grid_x, x, d isa DisSpec ? (0:Int(d.ncol)) : _mf6_vertices(d, 1))
end

"""
    BMI.get_grid_y(m::MF6Model, grid, y) -> y

As libmf6: for DIS the row edge indices `nrow:-1:0`, for DISV and DISU the y of the
VERTICES as given. libmf6 6.8.2 writes `ncol:-1:0` for DIS (a bug in `mf6bmiGrid.f90`,
which uses the last MSHAPE entry): fixed here. `y` must have at least `nrow + 1` (DIS) or
NVERT values, else an `ArgumentError`.
"""
function BMI.get_grid_y(m::MF6Model, grid::Integer, y::AbstractVector{<:Real})
    d = _mf6call(() -> _mf6_dis(m, :get_grid_y, grid), m)
    return _mf6_fill!(m, :get_grid_y, y, d isa DisSpec ? (Int(d.nrow):-1:0) : _mf6_vertices(d, 2))
end

_mf6_vertices(d, axis) = d.nvert > 0 ? d.vertices[axis, :] : Float64[]

"`BMI.get_grid_node_count(m::MF6Model, grid)`: the number of vertices (NVERT); DISU only (libmf6)."
BMI.get_grid_node_count(m::MF6Model, grid::Integer) = _mf6call(() -> Int(_mf6_dis(m, :get_grid_node_count, grid, DisuSpec).nvert), m)

"`BMI.get_grid_face_count(m::MF6Model, grid)`: the number of model nodes (cells with IDOMAIN > 0); DISU only (libmf6)."
BMI.get_grid_face_count(m::MF6Model, grid::Integer) =
    _mf6call(() -> (_mf6_dis(m, :get_grid_face_count, grid, DisuSpec); Int(_mf6sim(m).model.conn.nodes)), m)

# libmf6's IAVERT and JAVERT of a DISU grid with vertices (every user node; each polygon
# closed by repeating its first vertex).
function _mf6_iajavert(m::MF6Model, f::Symbol, grid::Integer)
    d = _mf6call(() -> _mf6_dis(m, f, grid, DisuSpec), m)
    d.nvert > 0 || _mf6call(() -> _mf6_failure(f, "the DISU grid has no vertices"), m)
    return _cell_vertex_csr(d.cell_vertices)
end

"`BMI.get_grid_nodes_per_face(m::MF6Model, grid, nodes_per_face)`: the number of vertices of every cell (user node), without the closing vertex; DISU only (libmf6)."
function BMI.get_grid_nodes_per_face(m::MF6Model, grid::Integer, nodes_per_face::AbstractVector{<:Integer})
    ia, _ = _mf6_iajavert(m, :get_grid_nodes_per_face, grid)
    return _mf6_fill!(m, :get_grid_nodes_per_face, nodes_per_face, diff(ia) .- 1)
end

"`BMI.get_grid_face_nodes(m::MF6Model, grid, face_nodes)`: the vertices of every cell (user node) as given, each polygon closed by its first vertex (JAVERT); DISU only (libmf6)."
function BMI.get_grid_face_nodes(m::MF6Model, grid::Integer, face_nodes::AbstractVector{<:Integer})
    _, ja = _mf6_iajavert(m, :get_grid_face_nodes, grid)
    return _mf6_fill!(m, :get_grid_face_nodes, face_nodes, ja)
end
