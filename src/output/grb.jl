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

# --- Writing a model's binary grid file (T6.9; Dis.f90, Disv.f90, Disu.f90 `write_grb`,
# DiscretizationBase.f90 `dis_ar`)

# Fortran `1PGw.d` (`Ee` exponent digits; default 2) as MF6's build writes the scalar
# definitions XORIGIN, YORIGIN and ANGROT: fixed notation `F(w-n).(d-k)` and `n = e + 2`
# blanks for 0.1 ≤ |x| < 10^d (and 0), else `1PEw.d`.
function _fortran_g(x::Float64, w::Integer, d::Integer; e::Integer = 2)
    n = e + 2
    s = x == 0 ? "0." * "0"^(d - 1) : obs_format(x, d; lead0 = true)
    occursin('E', s) || return lpad(s, w - n) * " "^n
    s = @sprintf("%.*E", d - 1, x)
    p = findfirst('E', s)
    ex = parse(Int, s[(p + 1):end])
    return lpad(s[1:(p - 1)] * "E" * (ex < 0 ? "-" : "+") * lpad(string(abs(ex)), e, '0'), w)
end

_grb_line(s, len) = rpad(s, len - 1) * "\n"

"""
    grb_file(dis::AbstractDisSpec, conn::Connectivity, icelltype) -> BinaryGridFile

The binary grid file MF6 writes for the discretization `dis` (Dis.f90, Disv.f90, Disu.f90
`write_grb`, version 1): the grid dimensions, origin and rotation, the input arrays (TOP,
BOTM/BOT, DELR/DELC, VERTICES, CELLX/CELLY, IAVERT/JAVERT: the closed vertex list of each
cell), the connectivity `conn` in user node numbers (`iajausr`), IDOMAIN and ICELLTYPE by
user node (`icelltype` by user node; 0 outside the reduced grid, `dis_ar`). Write it with
[`ModflowGPU.write_grb`](@ref).
"""
function grb_file(dis::AbstractDisSpec, conn::Connectivity, icelltype::AbstractVector{<:Integer})
    nu = Int(conn.nodesuser)
    ict = Int32[conn.nodereduced[u] > 0 ? Int32(icelltype[u]) : Int32(0) for u in 1:nu]
    iausr, jausr = iajausr(conn)
    vars = Tuple{String, Symbol, Vector{Int}, Any, String}[]   # name, type, dims, data, scalar text
    int(name, v) = push!(vars, (name, :INTEGER, Int[], Int32(v), string(v)))
    dbl(name, v, w, e) = push!(vars, (name, :DOUBLE, Int[], Float64(v), _fortran_g(Float64(v), w, 15; e)))
    arr(name, type, x, dims = [length(x)]) = push!(vars, (name, type, dims, x, ""))
    if dis isa DisSpec
        grid, w, e = "DIS", 24, 2
        int("NCELLS", nu)
        int("NLAY", dis.nlay)
        int("NROW", dis.nrow)
        int("NCOL", dis.ncol)
        int("NJA", conn.nja)
    elseif dis isa DisvSpec
        grid, w, e = "DISV", 25, 3
        _, javert = _cell_vertex_csr(dis.cell_vertices)
        int("NCELLS", nu)
        int("NLAY", dis.nlay)
        int("NCPL", dis.ncpl)
        int("NVERT", dis.nvert)
        int("NJAVERT", length(javert))
        int("NJA", conn.nja)
    else
        grid, w, e = "DISU", 24, 2
        int("NODES", nu)
        int("NJA", conn.nja)
    end
    dbl("XORIGIN", dis.xorigin, w, e)
    dbl("YORIGIN", dis.yorigin, w, e)
    dbl("ANGROT", dis.angrot, w, e)
    if dis isa DisSpec
        arr("DELR", :DOUBLE, dis.delr)
        arr("DELC", :DOUBLE, dis.delc)
        arr("TOP", :DOUBLE, vec(dis.top))
        arr("BOTM", :DOUBLE, vec(dis.botm))
    elseif dis isa DisvSpec
        arr("TOP", :DOUBLE, dis.top)
        arr("BOTM", :DOUBLE, vec(dis.botm))
        _vertex_vars!(arr, dis)
    else
        arr("TOP", :DOUBLE, dis.top)
        arr("BOT", :DOUBLE, dis.bot)
    end
    arr("IA", :INTEGER, iausr)
    arr("JA", :INTEGER, jausr)
    arr("IDOMAIN", :INTEGER, Vector{Int32}(idomain_user(dis)))
    arr("ICELLTYPE", :INTEGER, ict)
    dis isa DisuSpec && dis.nvert > 0 && _vertex_vars!(arr, dis)
    # Disv: VERTICES ... JAVERT come before IA (`write_grb` order).
    if dis isa DisvSpec
        order = ["NCELLS", "NLAY", "NCPL", "NVERT", "NJAVERT", "NJA", "XORIGIN", "YORIGIN", "ANGROT", "TOP", "BOTM",
            "VERTICES", "CELLX", "CELLY", "IAVERT", "JAVERT", "IA", "JA", "IDOMAIN", "ICELLTYPE"]
        vars = [vars[findfirst(v -> v[1] == name, vars)] for name in order]
    end
    lentxt = 100
    variables = GrbVariable[]
    data = Dict{String, Any}()
    for (name, type, dims, x, txt) in vars
        def = isempty(dims) ? "$name $type NDIM 0 # $txt" : "$name $type NDIM $(length(dims)) $(join(dims, ' '))"
        push!(variables, GrbVariable(name, type, dims, _grb_line(def, lentxt)))
        data[name] = x
    end
    header = [_grb_line(h, GRB_HEADER_BYTES) for h in ("GRID $grid", "VERSION 1", "NTXT $(length(variables))", "LENTXT $lentxt")]
    return BinaryGridFile(grid, 1, header, lentxt, variables, data)
end

# VERTICES (2, nvert), CELLX, CELLY and the closed vertex lists IAVERT, JAVERT (DISV, DISU
# with vertices).
function _vertex_vars!(arr, dis)
    iavert, javert = _cell_vertex_csr(dis.cell_vertices)
    arr("VERTICES", :DOUBLE, dis.vertices, [2, size(dis.vertices, 2)])
    arr("CELLX", :DOUBLE, dis.cell_center[1, :])
    arr("CELLY", :DOUBLE, dis.cell_center[2, :])
    arr("IAVERT", :INTEGER, iavert)
    arr("JAVERT", :INTEGER, javert)
    return nothing
end
