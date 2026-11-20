# BasicModelInterface on the stepping API (src/stepping.jl), as libmf6's BMI
# (srcbmi/mf6bmi.f90, mf6bmiGrid.f90) where the BMI specification allows: variable names
# are libmf6 addresses, values are on the host in Float64 with absolute heads (Int32 for
# indices). T8.9 brought every function in line with BasicModelInterface.jl 0.1 and the
# CSDMS documentation (bmi.csdms.io); docs/src/coupling/bmi.md lists the remaining
# deviations. Grids:
#
# - 0: scalar (the one-value variables: SLN_1/..., TDIS/..., counts);
# - 1: the model grid, all cells (user nodes): DIS "rectilinear" (one layer, ANGROT 0) or
#   "structured_quadrilateral", nodes at the cell centres; DISV and DISU with vertices
#   "unstructured" (BMI nodes are the vertices, faces the cells of every layer); DISU
#   without vertices "vector";
# - 2: the model nodes (cells with IDOMAIN > 0), "points" at the cell centres ("vector"
#   for a DISU grid without vertices);
# - 1 + 2i and 2 + 2i: the records (NODELIST, SIMVALS) and the BOUND values of stress
#   package i, "vector".
#
# Model-node variables (X, TOP, ...) are on grid 1 when every cell is a model node, else on
# grid 2, so that their length is always the size of their grid.
#
# T6.6: with several models in the solution, the grids above are there for every model, one
# model after the other (model 2's grids follow the 2 + 2·npackages of model 1, ...): the
# grid of a variable of model `<model>` is one of that model's.

import BasicModelInterface as BMI
import TOML

# An error for a BMI function that the model or grid does not support (instead of a
# MethodError), naming the function.
_bmi_unsupported(f::Symbol, why::AbstractString) = throw(ArgumentError("BMI.$f is not supported: $why"))

# ---------------------------------------------------------------------------------------
# Configuration file
# ---------------------------------------------------------------------------------------

const _BMI_CONFIG_KEYS = (
    "simulation", "backend", "precision", "preconditioner", "precond_options", "closures", "outdir", "write_output",
)
const _BMI_CLOSURE_KEYS = (:outer_dvclose, :outer_maximum, :inner_dvclose, :inner_rclose, :inner_maximum)

function _config_error(file, msg)
    throw(ArgumentError("BMI configuration file $file: $msg"))
end

function _config_get(cfg, file, key, ::Type{T}, default) where {T}
    haskey(cfg, key) || return default
    v = cfg[key]
    v isa T || _config_error(file, "`$key` must be a $(T === AbstractDict ? "table" : lowercase(string(nameof(T)))), got $(repr(v))")
    return v
end

# A TOML table as a NamedTuple; string values become Symbols (preconditioner options such
# as AMG's `smoother = "chebyshev"`).
_config_namedtuple(t::AbstractDict) = NamedTuple(Symbol(k) => (v isa AbstractString ? Symbol(v) : v) for (k, v) in t)

# Read a BMI configuration file (see `BMI.initialize`): the simulation directory and the
# keywords of `initialize`. A directory or an `mfsim.nam` gives the defaults.
function _bmi_config(config_file::AbstractString)
    if isdir(config_file)
        return (; simdir = String(config_file), kwargs = (;))
    elseif lowercase(basename(config_file)) == "mfsim.nam"
        isfile(config_file) || throw(ArgumentError("BMI.initialize: $config_file does not exist"))
        return (; simdir = dirname(abspath(config_file)), kwargs = (;))
    end
    endswith(lowercase(config_file), ".toml") || throw(
        ArgumentError(
            "BMI.initialize: config_file must be a TOML configuration file (*.toml), a simulation directory or its mfsim.nam, got \"$config_file\"",
        ),
    )
    isfile(config_file) || throw(ArgumentError("BMI.initialize: configuration file $config_file does not exist"))
    file = abspath(config_file)
    cfg = try
        TOML.parsefile(file)
    catch err
        err isa TOML.ParserError || rethrow()
        _config_error(file, sprint(showerror, err))
    end
    for k in keys(cfg)
        k in _BMI_CONFIG_KEYS || _config_error(file, "unknown key `$k` (expected one of $(join(_BMI_CONFIG_KEYS, ", ")))")
    end
    base = dirname(file)
    haskey(cfg, "simulation") || _config_error(file, "`simulation` (the simulation directory or its mfsim.nam) is missing")
    sim = joinpath(base, _config_get(cfg, file, "simulation", String, ""))
    simdir = isdir(sim) ? sim : lowercase(basename(sim)) == "mfsim.nam" && isfile(sim) ? dirname(sim) :
        _config_error(file, "`simulation`: $sim is not a directory or an mfsim.nam")
    kw = Dict{Symbol, Any}()
    haskey(cfg, "backend") && (kw[:backend] = select_backend(_config_get(cfg, file, "backend", String, "cpu")))
    haskey(cfg, "precision") && (kw[:precision] = Symbol(_config_get(cfg, file, "precision", String, "auto")))
    haskey(cfg, "preconditioner") && (kw[:preconditioner] = Symbol(_config_get(cfg, file, "preconditioner", String, "auto")))
    haskey(cfg, "outdir") && (kw[:outdir] = joinpath(base, _config_get(cfg, file, "outdir", String, "")))
    haskey(cfg, "write_output") && (kw[:write_output] = _config_get(cfg, file, "write_output", Bool, true))
    if haskey(cfg, "precond_options")
        kw[:precond_options] = _config_namedtuple(_config_get(cfg, file, "precond_options", AbstractDict, nothing))
    end
    if haskey(cfg, "closures")
        c = _config_namedtuple(_config_get(cfg, file, "closures", AbstractDict, nothing))
        for k in keys(c)
            k in _BMI_CLOSURE_KEYS || _config_error(file, "unknown closure `$k` (expected one of $(join(_BMI_CLOSURE_KEYS, ", ")))")
        end
        kw[:closures] = c
    end
    return (; simdir = normpath(simdir), kwargs = NamedTuple(kw))
end

"""
    BMI.initialize(::Type{Simulation}, config_file) -> Simulation

Start a simulation from a configuration file (BMI). `config_file` is a TOML file with the
settings of [`initialize`](@ref):

```toml
simulation = "model"          # the simulation directory or its mfsim.nam (required)
backend = "cpu"               # a name for select_backend ("cpu", "rocm", "cuda", "auto", ...)
precision = "auto"            # "auto", "double", "mixed" or "single"
preconditioner = "auto"       # "auto", "jacobi", "chebyshev", "mc_sgs", "mc_ilu0", "amg"
outdir = "output"             # directory for the output files (default: the simulation's)
write_output = true

[closures]                    # IMS overrides (optional)
outer_dvclose = 1.0e-4

[precond_options]             # preconditioner options (optional; strings become Symbols)
relax = 0.97
```

Paths are relative to the configuration file. Keys left out take the defaults of
[`initialize`](@ref). `config_file` can also be the simulation directory or its
`mfsim.nam`, which runs with all defaults (on the CPU). A GPU backend needs its package
loaded (`using AMDGPU`, ...). For keywords, call [`initialize`](@ref) (the stepping API)
directly.
"""
function BMI.initialize(::Type{Simulation}, config_file::AbstractString; kwargs...)
    isempty(kwargs) || throw(
        ArgumentError(
            "BMI.initialize takes no keywords (got $(join(keys(kwargs), ", "))): put the settings in a TOML configuration file, or call ModflowGPU.initialize(simdir; kwargs...)",
        ),
    )
    c = _bmi_config(config_file)
    return initialize(c.simdir; c.kwargs...)
end

BMI.initialize(::Type{Simulation}) = _bmi_unsupported(:initialize, "a Simulation needs a configuration file (BMI.initialize(Simulation, config_file))")

# ---------------------------------------------------------------------------------------
# Control and time
# ---------------------------------------------------------------------------------------

"`BMI.update`: one time step (`update`). Throws a `ConvergenceError` when it fails without CONTINUE."
function BMI.update(s::Simulation)
    if !update(s)
        r = last(s.results)
        throw(ConvergenceError(r.step.kper, r.step.kstp, r.outer.iterations))
    end
    return nothing
end

# Two times are the same when they differ by less than this fraction of the end time
# (the step ends are sums of step lengths, a caller's times may differ in the last bits).
const _BMI_TIME_RTOL = 1.0e-10

"""
    BMI.update_until(sim, time)

Time steps until the current time is `time`, which must be the end of a time step (time
steps are not split) and not before the current time; `time` equal to the current time
does nothing.
"""
function BMI.update_until(s::Simulation, time::Real)
    tol = _BMI_TIME_RTOL * max(1.0, abs(get_end_time(s)))
    now = get_current_time(s)
    abs(time - now) <= tol && return nothing
    time < now && throw(ArgumentError("BMI.update_until: time $time is before the current time $now"))
    k = findfirst(st -> abs(st.totim - time) <= tol, s.steps)
    if isnothing(k)
        time > get_end_time(s) && throw(ArgumentError("BMI.update_until: time $time is after the end time $(get_end_time(s))"))
        throw(
            ArgumentError(
                "BMI.update_until: time $time is not the end of a time step (time steps are not split); the next step ends at $(s.steps[s.istep + 1].totim)",
            ),
        )
    end
    while s.istep < k
        BMI.update(s)
    end
    return nothing
end

BMI.finalize(s::Simulation) = (finalize_simulation(s); nothing)

BMI.get_component_name(::Simulation) = "ModflowGPU (MODFLOW 6 GWF)"
BMI.get_input_var_names(s::Simulation) = input_addresses(s)
BMI.get_output_var_names(s::Simulation) = var_addresses(s)
BMI.get_input_item_count(s::Simulation) = length(BMI.get_input_var_names(s))
BMI.get_output_item_count(s::Simulation) = length(BMI.get_output_var_names(s))

BMI.get_current_time(s::Simulation) = get_current_time(s)
BMI.get_start_time(s::Simulation) = get_start_time(s)
BMI.get_end_time(s::Simulation) = get_end_time(s)
BMI.get_time_step(s::Simulation) = get_time_step(s)

const _BMI_TIME_UNITS = Dict(:seconds => "s", :minutes => "min", :hours => "h", :days => "d", :years => "yr")
const _BMI_LENGTH_UNITS = Dict(:meters => "m", :feet => "ft", :centimeters => "cm")

"`BMI.get_time_units`: the TDIS `TIME_UNITS` in lower case (`\"days\"`, ...); `\"none\"` when not given."
function BMI.get_time_units(s::Simulation)
    u = s.spec.tdis.time_units
    return u === :undefined ? "none" : String(u)
end

# ---------------------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------------------

_bmi_kind(::BoundaryPackage{K}) where {K} = K

# The units of a variable in UDUNITS syntax: the length and time units of the model (DIS
# LENGTH_UNITS, TDIS TIME_UNITS; "L" and "T" when not given), "1" for counts, indices and
# flags. BOUND has one unit only for single-column packages (CHD, WEL, RCH), else "none".
function _bmi_units(s::Simulation, name::AbstractString)
    var = _variable(s, name)
    parts = split(uppercase(name), '/')
    mv = _find_view(s, parts[1])
    L = get(_BMI_LENGTH_UNITS, (isnothing(mv) ? s.model.spec : _vspec(mv)).dis.length_units, "L")
    T = get(_BMI_TIME_UNITS, s.spec.tdis.time_units, "T")
    u = var.units
    if length(parts) == 3 && parts[3] == "BOUND"
        k = _bmi_kind(s.model.packages[_find_package(s, parts[1], parts[2])])
        u = k === :CHD ? "L" : k === :WEL ? "L3/T" : k === :RCH ? "L/T" : "none"
    end
    u == "L" && return L
    u == "L2" && return L * "2"
    u == "L/T" && return "$L $T-1"
    u == "L3/T" && return "$(L)3 $T-1"
    u == "T" && return T
    u == "-" && return "1"
    return u
end

# T6.6: the grids of the models of a multi-model solution, one model after the other: the
# grids of model `k` (1: model grid, 2: nodes, then two per package, as for a single model)
# have the ids `base_k + g - 1`, with `base_1 = 1` (a single model keeps its ids).
_bmi_ngrids(v) = 2 + 2 * length(_vpackages(v))
function _bmi_grid_base(s::Simulation, mv)
    base = 1
    for v in s.views
        v === mv && return base
        base += _bmi_ngrids(v)
    end
    return base
end

# The model (view) of grid `grid` (≥ 1) and the grid's number in that model, or `nothing`.
function _bmi_view_grid(s::Simulation, grid::Integer)
    base = 1
    for v in s.views
        n = _bmi_ngrids(v)
        base <= grid < base + n && return v, grid - base + 1
        base += n
    end
    return nothing, grid
end

# The grid of a variable (see the top of this file).
function _bmi_var_grid(s::Simulation, name::AbstractString)
    _variable(s, name)   # checks the address
    parts = split(uppercase(name), '/')
    parts[1] in ("SLN_1", "TDIS") && return 0
    mv = _find_view(s, parts[1])
    c = _vconn(mv)
    b = _bmi_grid_base(s, mv) - 1
    node_grid = c.nodes == c.nodesuser ? 1 : 2
    length(parts) == 2 && return b + node_grid
    v = parts[3]
    v in ("NODES", "NODESUSER", "NBOUND", "MAXBOUND") && return 0
    parts[2] in _dis_names(_vspec(mv)) && return b + (v == "NODEREDUCED" ? 1 : node_grid)
    ip = findfirst(==(_find_package(s, parts[1], parts[2])), collect(_vpackages(mv)))
    return b + (v == "BOUND" ? 2 + 2ip : 1 + 2ip)
end

BMI.get_var_grid(s::Simulation, name::AbstractString) = _bmi_var_grid(s, name)
BMI.get_var_type(s::Simulation, name::AbstractString) = string(eltype(get_value(s, name)))
BMI.get_var_units(s::Simulation, name::AbstractString) = _bmi_units(s, name)
BMI.get_var_itemsize(s::Simulation, name::AbstractString) = sizeof(eltype(get_value(s, name)))
BMI.get_var_nbytes(s::Simulation, name::AbstractString) = sizeof(get_value(s, name))

"`BMI.get_var_location`: `\"face\"` for a variable on an unstructured model grid (one value per cell), else `\"node\"`."
BMI.get_var_location(s::Simulation, name::AbstractString) =
    _bmi_grid(s, _bmi_var_grid(s, name)).type == "unstructured" ? "face" : "node"

function _check_length(f::Symbol, dest, n)
    length(dest) == n || throw(DimensionMismatch("BMI.$f: the destination has length $(length(dest)), expected $n"))
    return nothing
end

function BMI.get_value(s::Simulation, name::AbstractString, dest::AbstractArray)
    v = get_value(s, name)
    _check_length(:get_value, dest, length(v))
    return copyto!(dest, vec(v))
end

"""
    BMI.get_value_ptr(sim, name) -> Vector

The host mirror of `name` (`get_value_ptr(sim, name)`; Float64, absolute heads),
flattened (a `Vector` that shares the memory of the mirror). It is a copy of the device
data, synchronised at the stepping calls (see src/stepping.jl); writes into it take effect
at the next time step.
"""
BMI.get_value_ptr(s::Simulation, name::AbstractString) = vec(get_value_ptr(s, name))

function BMI.get_value_at_indices(s::Simulation, name::AbstractString, dest::AbstractArray, inds::AbstractVector{<:Integer})
    _check_length(:get_value_at_indices, dest, length(inds))
    return copyto!(dest, vec(get_value(s, name))[inds])
end

BMI.set_value(s::Simulation, name::AbstractString, value::AbstractArray) = (set_value(s, name, value); nothing)

function BMI.set_value_at_indices(s::Simulation, name::AbstractString, inds::AbstractVector{<:Integer}, value::AbstractArray)
    length(value) == length(inds) ||
        throw(DimensionMismatch("BMI.set_value_at_indices: $(length(value)) values for $(length(inds)) indices"))
    full = vec(get_value(s, name))
    full[inds] .= vec(value)
    set_value(s, name, full)
    return nothing
end

# ---------------------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------------------

_has_vertices(d::DisSpec) = true
_has_vertices(d::DisvSpec) = true
_has_vertices(d::DisuSpec) = d.nvert > 0

# (; type, rank, size, shape) of a grid; `shape` is `nothing` for an unstructured grid.
function _bmi_grid(s::Simulation, grid::Integer)
    grid == 0 && return (; type = "scalar", rank = 0, size = 1, shape = Int[])
    mv, g = _bmi_view_grid(s, grid)
    ntot = sum(_bmi_ngrids, s.views)
    isnothing(mv) && throw(ArgumentError("unknown grid $grid (grids are 0 to $ntot)"))
    m = _vmodel(mv)
    d = _vspec(mv).dis
    pkgs = collect(_vpackages(mv))
    np = length(pkgs)
    conn = _vconn(mv)
    nu = Int(conn.nodesuser)
    grid = g
    if grid == 1
        if d isa DisSpec
            nl, nr, nc = Int(d.nlay), Int(d.nrow), Int(d.ncol)
            type = nl == 1 && d.angrot == 0 ? "rectilinear" : "structured_quadrilateral"
            shape = nl == 1 ? [nr, nc] : [nl, nr, nc]
            return (; type, rank = length(shape), size = nu, shape)
        end
        _has_vertices(d) || return (; type = "vector", rank = 1, size = nu, shape = [nu])
        return (; type = "unstructured", rank = 2, size = Int(d.nvert), shape = nothing)
    elseif grid == 2
        n = Int(conn.nodes)
        _has_vertices(d) || return (; type = "vector", rank = 1, size = n, shape = [n])
        return (; type = "points", rank = 3, size = n, shape = nothing)
    elseif 3 <= grid <= 2 + 2np
        pkg = m.packages[pkgs[(grid - 1) ÷ 2]]
        n = isodd(grid) ? length(pkg.nodelist_host) : length(pkg.bound_user)
        return (; type = "vector", rank = 1, size = n, shape = [n])
    end
    throw(ArgumentError("unknown grid $grid (grids are 0 to $ntot)"))
end

BMI.get_grid_type(s::Simulation, grid::Integer) = _bmi_grid(s, grid).type
BMI.get_grid_rank(s::Simulation, grid::Integer) = _bmi_grid(s, grid).rank
BMI.get_grid_size(s::Simulation, grid::Integer) = _bmi_grid(s, grid).size

function _fill(f::Symbol, dest::AbstractVector, values)
    _check_length(f, dest, length(values))
    return copyto!(dest, values)
end

function BMI.get_grid_shape(s::Simulation, grid::Integer, shape::AbstractVector{<:Integer})
    g = _bmi_grid(s, grid)
    isnothing(g.shape) && _bmi_unsupported(:get_grid_shape, "grid $grid is $(g.type) (no shape)")
    return _fill(:get_grid_shape, shape, g.shape)
end

BMI.get_grid_spacing(s::Simulation, grid::Integer, ::AbstractVector{<:Real}) = _bmi_unsupported(
    :get_grid_spacing, "grid $grid is $(_bmi_grid(s, grid).type), not uniform_rectilinear (use get_grid_x and get_grid_y)",
)
BMI.get_grid_origin(s::Simulation, grid::Integer, ::AbstractVector{<:Real}) = _bmi_unsupported(
    :get_grid_origin, "grid $grid is $(_bmi_grid(s, grid).type), not uniform_rectilinear (use get_grid_x and get_grid_y)",
)

# Model coordinates to real-world coordinates: rotation by ANGROT (degrees, counter-
# clockwise) about the lower-left corner, then XORIGIN, YORIGIN (MF6 DisBaseType
# transform_xy).
function _world(d, x, y)
    a = deg2rad(d.angrot)
    return d.xorigin + x * cos(a) - y * sin(a), d.yorigin + x * sin(a) + y * cos(a)
end

# DIS cell centres in model coordinates, row 1 at the top (north).
_dis_xc(d::DisSpec) = cumsum(d.delr) .- d.delr ./ 2
_dis_yc(d::DisSpec) = sum(d.delc) .- (cumsum(d.delc) .- d.delc ./ 2)

# Real-world cell centres (x, y) of every user node.
function _user_centres(d::DisSpec)
    xc, yc = _dis_xc(d), _dis_yc(d)
    xy = [_world(d, xc[j], yc[i]) for j in 1:d.ncol, i in 1:d.nrow, _ in 1:d.nlay]
    return first.(vec(xy)), last.(vec(xy))
end
function _user_centres(d::Union{DisvSpec, DisuSpec})
    n = d isa DisvSpec ? Int(d.nlay * d.ncpl) : Int(d.nodes)
    ncpl = d isa DisvSpec ? Int(d.ncpl) : n
    xy = [_world(d, d.cell_center[1, c], d.cell_center[2, c]) for c in ((nn - 1) % ncpl + 1 for nn in 1:n)]
    return first.(xy), last.(xy)
end

# Cell-centre elevation of every user node.
function _user_zc(d::DisSpec)
    t = reshape(d.top, size(d.top)..., 1)
    tops = cat(t, d.botm[:, :, 1:(end - 1)]; dims = 3)
    return vec((tops .+ d.botm) ./ 2)
end

# Real-world vertices of an unstructured grid.
function _vertices(d)
    xy = [_world(d, d.vertices[1, v], d.vertices[2, v]) for v in 1:size(d.vertices, 2)]
    return first.(xy), last.(xy)
end

function _grid_coord(f::Symbol, s::Simulation, grid::Integer, dest::AbstractVector{<:Real}, axis::Int)
    g = _bmi_grid(s, grid)
    mv = first(_bmi_view_grid(s, max(grid, 1)))  # grid 0 (scalar): no coordinates, below
    d = _vspec(mv).dis
    conn = _vconn(mv)
    nm = (:x, :y, :z)[axis]
    if g.type == "rectilinear"
        axis == 3 && _bmi_unsupported(f, "grid $grid is rectilinear of rank 2 (no z)")
        x, y = _world(d, 0.0, 0.0)
        return _fill(f, dest, axis == 1 ? x .+ _dis_xc(d) : y .+ _dis_yc(d))
    elseif g.type == "structured_quadrilateral"
        if axis == 3
            g.rank == 3 || _bmi_unsupported(f, "grid $grid has rank 2 (no z)")
            return _fill(f, dest, _user_zc(d))
        end
        return _fill(f, dest, _user_centres(d)[axis])
    elseif g.type == "unstructured"
        axis == 3 && _bmi_unsupported(f, "the vertices of grid $grid (DISV/DISU) have no elevation (use the DIS/TOP and DIS/BOT variables)")
        return _fill(f, dest, _vertices(d)[axis])
    elseif g.type == "points"
        axis == 3 && return _fill(f, dest, (conn.top .+ conn.bot) ./ 2)
        return _fill(f, dest, _user_centres(d)[axis][conn.nodeuser])
    end
    return _bmi_unsupported(f, "grid $grid is $(g.type) (no $nm coordinates)")
end

"`BMI.get_grid_x`: the x of the grid nodes (cell centres of DIS, vertices of DISV/DISU), in real-world coordinates (XORIGIN, YORIGIN, ANGROT)."
BMI.get_grid_x(s::Simulation, grid::Integer, x::AbstractVector{<:Real}) = _grid_coord(:get_grid_x, s, grid, x, 1)
"`BMI.get_grid_y`: the y of the grid nodes, as `BMI.get_grid_x` (rows from north to south)."
BMI.get_grid_y(s::Simulation, grid::Integer, y::AbstractVector{<:Real}) = _grid_coord(:get_grid_y, s, grid, y, 2)
"`BMI.get_grid_z`: the cell-centre elevations of a structured_quadrilateral grid of rank 3 or of the points grid."
BMI.get_grid_z(s::Simulation, grid::Integer, z::AbstractVector{<:Real}) = _grid_coord(:get_grid_z, s, grid, z, 3)

BMI.get_grid_node_count(s::Simulation, grid::Integer) = _bmi_grid(s, grid).size

# The faces of an unstructured grid: the vertices of every cell (all layers), counter-
# clockwise (MF6 lists them clockwise; a closing repeat of the first vertex is dropped).
function _faces(f::Symbol, s::Simulation, grid::Integer)
    g = _bmi_grid(s, grid)
    g.type == "unstructured" || _bmi_unsupported(f, "grid $grid is $(g.type), not unstructured")
    mv = first(_bmi_view_grid(s, grid))
    d = _vspec(mv).dis
    ncpl = d isa DisvSpec ? Int(d.ncpl) : Int(d.nodes)
    poly = map(d.cell_vertices) do iv
        v = Int.(iv)
        length(v) > 1 && first(v) == last(v) && pop!(v)
        reverse!(v)
    end
    return [poly[(n - 1) % ncpl + 1] for n in 1:Int(_vconn(mv).nodesuser)]
end

# The edges of an unstructured grid: (tail, head) of every vertex pair that bounds a face,
# numbered in the order first met, and the edges of each face.
function _edges(f::Symbol, s::Simulation, grid::Integer)
    faces = _faces(f, s, grid)
    ids = Dict{Tuple{Int, Int}, Int}()
    edge_nodes = Int[]
    face_edges = Int[]
    for v in faces, k in eachindex(v)
        a, b = v[k], v[mod1(k + 1, length(v))]
        e = get!(ids, minmax(a, b)) do
            append!(edge_nodes, (a, b))
            length(ids) + 1
        end
        push!(face_edges, e)
    end
    return (; edge_nodes, face_edges)
end

BMI.get_grid_face_count(s::Simulation, grid::Integer) = length(_faces(:get_grid_face_count, s, grid))
BMI.get_grid_edge_count(s::Simulation, grid::Integer) = length(_edges(:get_grid_edge_count, s, grid).edge_nodes) ÷ 2
BMI.get_grid_edge_nodes(s::Simulation, grid::Integer, edge_nodes::AbstractVector{<:Integer}) =
    _fill(:get_grid_edge_nodes, edge_nodes, _edges(:get_grid_edge_nodes, s, grid).edge_nodes)
BMI.get_grid_face_edges(s::Simulation, grid::Integer, face_edges::AbstractVector{<:Integer}) =
    _fill(:get_grid_face_edges, face_edges, _edges(:get_grid_face_edges, s, grid).face_edges)
BMI.get_grid_face_nodes(s::Simulation, grid::Integer, face_nodes::AbstractVector{<:Integer}) =
    _fill(:get_grid_face_nodes, face_nodes, reduce(vcat, _faces(:get_grid_face_nodes, s, grid)))
BMI.get_grid_nodes_per_face(s::Simulation, grid::Integer, nodes_per_face::AbstractVector{<:Integer}) =
    _fill(:get_grid_nodes_per_face, nodes_per_face, length.(_faces(:get_grid_nodes_per_face, s, grid)))
