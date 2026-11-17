# MF6 binary grid files (`.dis.grb`, `.disv.grb`, `.disu.grb`).
#
# Layout (Dis.f90/Disv.f90/Disu.f90 `write_grb`; IO guide "Binary Grid File"): four
# header lines of 50 characters, the last of which is a newline (`GRID DIS`,
# `VERSION 1`, `NTXT n`, `LENTXT 100`), then NTXT variable definitions of LENTXT
# characters, the last again a newline (`NAME TYPE NDIM ndim dims...`, scalars with
# `NDIM 0 # value`), then the data of each variable in definition order: INTEGER as
# 4-byte integers, DOUBLE as 8-byte reals.

"""
    ModflowGPU.GrbVariable

One variable of a binary grid file, an element of `BinaryGridFile.variables`: `name`
(e.g. `"IA"`), `type` (`:INTEGER` or
`:DOUBLE`), `dims` (empty for a scalar) and `definition`, the definition line as written
in the file (LENTXT characters, including the trailing newline).
"""
struct GrbVariable
    name::String
    type::Symbol
    dims::Vector{Int}
    definition::String
end

"""
    BinaryGridFile

Contents of an MF6 binary grid file (see [`read_grb`](@ref)): `grid` (`"DIS"`, `"DISV"`
or `"DISU"`), `version`, the four raw `header` lines, `lentxt`, the `variables` in file
order and their `data` by name (INTEGER as `Int32`, DOUBLE as `Float64`; scalars as
numbers, `NDIM 1` variables as vectors, `NDIM 2` as matrices with the dimensions in the
file's (Fortran) order). `g[name]` and `haskey(g, name)` access `data`.
"""
struct BinaryGridFile
    grid::String
    version::Int
    header::Vector{String}
    lentxt::Int
    variables::Vector{GrbVariable}
    data::Dict{String, Any}
end

Base.getindex(g::BinaryGridFile, name::AbstractString) = g.data[name]
Base.haskey(g::BinaryGridFile, name::AbstractString) = haskey(g.data, name)
Base.keys(g::BinaryGridFile) = (v.name for v in g.variables)

const GRB_HEADER_BYTES = 50

function _grb_header_value(path, line, key, offset)
    w = split(line)
    (length(w) >= 2 && w[1] == key) ||
        throw(MF6BinaryFileError(path, offset, "expected header line '$key ...', found '$(strip(line))'"))
    return w[2]
end

"""
    read_grb(path) -> BinaryGridFile

Read an MF6 binary grid file (`.dis.grb`, `.disv.grb`, `.disu.grb`), written by
ModflowGPU or MF6 for the DIS, DISV or DISU package. It holds the grid and the model's
connectivity (`IA`, `JA`), which give the meaning of the FLOW-JA-FACE budget record.
Throws an [`ModflowGPU.MF6BinaryFileError`](@ref) for a malformed file.

# Example
```julia
g = read_grb("path/to/simulation/model.dis.grb")
g.grid                  # "DIS"
g["NCOL"], g["NROW"], g["NLAY"]
ia, ja = g["IA"], g["JA"]
```
"""
function read_grb(path::AbstractString)
    path = String(path)
    return open(path, "r") do io
        header = String[]
        for _ in 1:4
            eof(io) && throw(MF6BinaryFileError(path, position(io), "file ends inside the header"))
            push!(header, _read_text(io, GRB_HEADER_BYTES))
        end
        grid = String(_grb_header_value(path, header[1], "GRID", 0))
        version = parse(Int, _grb_header_value(path, header[2], "VERSION", 50))
        ntxt = parse(Int, _grb_header_value(path, header[3], "NTXT", 100))
        lentxt = parse(Int, _grb_header_value(path, header[4], "LENTXT", 150))
        variables = GrbVariable[]
        for _ in 1:ntxt
            off = position(io)
            line = _read_text(io, lentxt)
            w = split(line)
            (length(w) >= 4 && w[3] == "NDIM") ||
                throw(MF6BinaryFileError(path, off, "malformed variable definition '$(strip(line))'"))
            type = Symbol(w[2])
            type in (:INTEGER, :DOUBLE) ||
                throw(MF6BinaryFileError(path, off, "unknown variable type $(w[2]) of $(w[1])"))
            ndim = parse(Int, w[4])
            dims = ndim == 0 ? Int[] : parse.(Int, w[5:(4 + ndim)])
            push!(variables, GrbVariable(String(w[1]), type, dims, line))
        end
        data = Dict{String, Any}()
        for v in variables
            T = v.type === :INTEGER ? Int32 : Float64
            n = isempty(v.dims) ? 1 : prod(v.dims)
            off = position(io)
            off + n * sizeof(T) <= filesize(path) ||
                throw(MF6BinaryFileError(path, off, "file ends inside the data of $(v.name)"))
            x = _read_le!(io, Vector{T}(undef, n))
            data[v.name] = isempty(v.dims) ? x[1] : length(v.dims) == 1 ? x : reshape(x, v.dims...)
        end
        eof(io) || throw(MF6BinaryFileError(path, position(io), "unexpected data after the last variable"))
        return BinaryGridFile(grid, version, header, lentxt, variables, data)
    end
end

"""
    write_grb(path, g::BinaryGridFile) -> path

Write a binary grid file with the header lines, variable definitions and data of `g`
(as read by [`read_grb`](@ref)), so `write_grb(p, read_grb(q))` reproduces `q` byte for
byte.
"""
function write_grb(path::AbstractString, g::BinaryGridFile)
    open(path, "w") do io
        for h in g.header
            ncodeunits(h) == GRB_HEADER_BYTES || throw(ArgumentError("header line must have $GRB_HEADER_BYTES characters"))
            write(io, h)
        end
        for v in g.variables
            ncodeunits(v.definition) == g.lentxt || throw(ArgumentError("definition of $(v.name) must have $(g.lentxt) characters"))
            write(io, v.definition)
        end
        for v in g.variables
            T = v.type === :INTEGER ? Int32 : Float64
            x = g.data[v.name]
            _write_le(io, x isa Number ? T[x] : convert(Array{T}, x))
        end
    end
    return path
end
