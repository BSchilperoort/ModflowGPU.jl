# Host-side model description: discretization (plain structs, no device arrays).
#
# The spec layer (PLAN.md section 3.2) separates the MF6 file format from the engine: the
# MF6 loader (`load_simulation`, src/input/simulation.jl) and the in-memory builder (T3.2)
# both produce these structs. Rules for every spec struct:
# - host only, plain `Array`s; reals are `Float64`, indices and integer flags `Int32`;
# - arrays keep MF6's memory shape and user node numbering: a DIS array `(ncol, nrow,
#   nlay)` is `A[j, i, k]`, whose linear index is MF6's user node
#   `(k-1)*nrow*ncol + (i-1)*ncol + j`; `(nodes)` arrays are vectors indexed by user node.
#   Reduced (active) node numbers depend on IDOMAIN and are made by T2.1;
# - file references (TS6, OBS6, output files) are kept as written, relative to the
#   simulation directory (`SimulationSpec.simdir`); names of models and packages are upper
#   case, as MF6 stores them.

"""
    AbstractDisSpec

A discretization package of a GWF model: `DisSpec`, `DisvSpec` or `DisuSpec`.
`grid_dims(d)` gives its dimensions (`GridDims`), `nodes_user(d)` the number of user
nodes and `idomain_user(d)` IDOMAIN as a vector by user node.
"""
abstract type AbstractDisSpec end

"""
    DisSpec

A structured grid (DIS6): `nlay × nrow × ncol` cells.

- `delr::Vector{Float64}` `(ncol)`, `delc` `(nrow)`: column widths along a row and row
  widths along a column.
- `top::Matrix{Float64}` `(ncol, nrow)`, `botm::Array{Float64, 3}` `(ncol, nrow, nlay)`.
- `idomain::Array{Int32, 3}` `(ncol, nrow, nlay)`: all ones when not given (MF6 default).
- `xorigin`, `yorigin`, `angrot`: model origin and rotation (degrees).
- `length_units`: `:undefined`, `:feet`, `:meters` or `:centimeters`.
- `nogrb`: no binary grid file; `grb_file`: GRB6 FILEOUT name (`""` = default name).
- `file`: the input file (relative to the simulation directory, `""` for built specs).
"""
Base.@kwdef struct DisSpec <: AbstractDisSpec
    nlay::Int32
    nrow::Int32
    ncol::Int32
    delr::Vector{Float64}
    delc::Vector{Float64}
    top::Matrix{Float64}
    botm::Array{Float64, 3}
    idomain::Array{Int32, 3} = ones(Int32, ncol, nrow, nlay)
    xorigin::Float64 = 0.0
    yorigin::Float64 = 0.0
    angrot::Float64 = 0.0
    length_units::Symbol = :undefined
    nogrb::Bool = false
    grb_file::String = ""
    file::String = ""
end

"""
    DisvSpec

A vertex grid (DISV6): `nlay` layers of `ncpl` cells.

- `top::Vector{Float64}` `(ncpl)`, `botm::Matrix{Float64}` `(ncpl, nlay)`,
  `idomain::Matrix{Int32}` `(ncpl, nlay)`.
- `vertices::Matrix{Float64}` `(2, nvert)`: `xv`, `yv` of each vertex.
- `cell_center::Matrix{Float64}` `(2, ncpl)`: `xc`, `yc` of each cell.
- `cell_vertices::Vector{Vector{Int32}}`: the vertex numbers of each cell (`icvert`, in
  the order given, clockwise in MF6).
- Other fields as in `DisSpec`.
"""
Base.@kwdef struct DisvSpec <: AbstractDisSpec
    nlay::Int32
    ncpl::Int32
    nvert::Int32
    top::Vector{Float64}
    botm::Matrix{Float64}
    idomain::Matrix{Int32} = ones(Int32, ncpl, nlay)
    vertices::Matrix{Float64}
    cell_center::Matrix{Float64}
    cell_vertices::Vector{Vector{Int32}}
    xorigin::Float64 = 0.0
    yorigin::Float64 = 0.0
    angrot::Float64 = 0.0
    length_units::Symbol = :undefined
    nogrb::Bool = false
    grb_file::String = ""
    file::String = ""
end

"""
    DisuSpec

An unstructured grid (DISU6) of `nodes` cells with `nja` connections (diagonal included),
as given in the CONNECTIONDATA block (user node numbers, MF6's `iac`/`ja` layout: row `n`
holds `n` itself first, then its neighbours).

- `top`, `bot`, `area::Vector{Float64}` `(nodes)`, `idomain::Vector{Int32}` `(nodes)`.
- `iac::Vector{Int32}` `(nodes)`: number of entries per row; `ja`, `ihc::Vector{Int32}`
  and `cl12`, `hwva::Vector{Float64}` `(nja)`; `angldegx` `(nja)` or `nothing`.
- `vertices`, `cell_center`, `cell_vertices` as in `DisvSpec`, or empty when NVERT = 0.
- `vertical_offset_tolerance`.
"""
Base.@kwdef struct DisuSpec <: AbstractDisSpec
    nodes::Int32
    nja::Int32
    nvert::Int32 = 0
    top::Vector{Float64}
    bot::Vector{Float64}
    area::Vector{Float64}
    idomain::Vector{Int32} = ones(Int32, nodes)
    iac::Vector{Int32}
    ja::Vector{Int32}
    ihc::Vector{Int32}
    cl12::Vector{Float64}
    hwva::Vector{Float64}
    angldegx::Union{Nothing, Vector{Float64}} = nothing
    vertices::Matrix{Float64} = zeros(2, 0)
    cell_center::Matrix{Float64} = zeros(2, 0)
    cell_vertices::Vector{Vector{Int32}} = Vector{Int32}[]
    vertical_offset_tolerance::Float64 = 0.0
    xorigin::Float64 = 0.0
    yorigin::Float64 = 0.0
    angrot::Float64 = 0.0
    length_units::Symbol = :undefined
    nogrb::Bool = false
    grb_file::String = ""
    file::String = ""
end

grid_dims(d::DisSpec) = dis_dims(d.nlay, d.nrow, d.ncol)
grid_dims(d::DisvSpec) = disv_dims(d.nlay, d.ncpl; nvert = d.nvert)
grid_dims(d::DisuSpec) = disu_dims(d.nodes; nja = d.nja, nvert = d.nvert)

"""
    nodes_user(d::AbstractDisSpec) -> Int

The number of user nodes (cells including inactive ones).
"""
nodes_user(d::AbstractDisSpec) = prod(grid_dims(d).mshape)

"""
    idomain_user(d::AbstractDisSpec) -> Vector{Int32}

IDOMAIN by user node (a view of the IDOMAIN array as a vector).
"""
idomain_user(d::AbstractDisSpec) = vec(d.idomain)
