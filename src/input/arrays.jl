# Array input (READARRAY) for MF6 text input (host only).
#
# Ports src/Utilities/ArrayRead (ArrayReaderBase.f90 `read_control_record`, `fill_*`;
# Double1dReader/Integer1dReader/Double2dReader/Integer2dReader `read_ascii`,
# `read_binary`, `apply_factor`; LayeredArrayReader.f90), the binary header of
# ArrayReaders.f90 (`read_binary_header`, `check_binary_filesize`) and the GRIDDATA tag
# handling of the input data model (LoadMf6File.f90 `load_tag`, `load_double1d_type` ...,
# SourceCommon.f90 `get_shape_from_string`, `get_layered_shape`, `set_model_shape`).
#
# Control record (the line after the array name, or one per layer with LAYERED):
#   CONSTANT c
#   INTERNAL [FACTOR f] [IPRN n]
#   OPEN/CLOSE fname [FACTOR f] [(BINARY)] [IPRN n]
# The words after INTERNAL / OPEN/CLOSE fname are read in any order; FACTOR and IPRN take
# the next word as their value (an integer FACTOR for integer arrays), unknown words are
# ignored (so are trailing comments). FACTOR 0 means no scaling (`apply_factor` skips
# it). CONSTANT takes no FACTOR. IPRN is parsed but nothing is printed.
#
# INTERNAL and ASCII OPEN/CLOSE data are read the way MF6 reads them: with a Fortran
# list-directed `read(unit, *)`, starting at the line after the control record (INTERNAL)
# or at the top of the file (OPEN/CLOSE). Checked against gfortran:
# - values are separated by blanks, tabs or a comma with optional blanks; blank lines
#   are skipped; numbers are Fortran numbers (`parse_double`, `parse_int`: `1.0d0`,
#   `1.5+3`; an integer array rejects `5.` and `1e2`);
# - `r*c` repeats the value c r times;
# - one `read` statement takes as many lines as it needs and skips the rest of its last
#   line, so extra values at the end of the last line are ignored; extra lines are not
#   (the next line is read as the next array name or control record);
# - a 2-D array of the input data model (`(ncol, nrow)`, `(ncpl, nlay)`) is read one
#   row at a time (`read_dbl2d`), so every row starts on a new line; 1-D and 3-D arrays
#   are one read (`read_dbl1d`, `load_double3d_type`); with LAYERED each layer has its
#   own control record, and a DIS layer is read row by row;
# - a null value (`,,`, `r*`) or a `/` before all values were read leaves elements
#   undefined in MF6 (the IO guide says both are not allowed): here they are errors;
# - a word that is not a number is an error (also `#` or `!`: in OPEN/CLOSE files a
#   comment line is an error, as in MF6). Deviation: comment lines inside INTERNAL
#   data are skipped (the block parser of T1.1 drops them), where MF6 fails.
# (BINARY) files are MF6 binary array files (unformatted stream): one header record
# KSTP, KPER (Int32), PERTIM, TOTIM (Float64), TEXT (16 bytes), M1, M2, M3 (Int32), then
# the values as Float64 (double arrays) or Int32 (integer arrays). The file size must be
# exactly header + values (`check_binary_filesize`); M1-M3 are not checked, as in MF6.
#
# Element types: double arrays are `Float64`, integer arrays `Int32` (MF6's `DP` and
# `I4B`; the binary integer format is 4 bytes). The Julia array has the MF6 memory shape:
# the dimensions of the DFN shape in order, column-major, so its linear index is MF6's
# (for a DIS `(ncol, nrow, nlay)` array, `A[j, i, k]` is user node
# `(k-1)*nrow*ncol + (i-1)*ncol + j`; `(nodes)` arrays are vectors indexed by user node).

const BINARY_HEADER_BYTES = 52  # 5 Int32 + 2 Float64 + 16 characters (ArrayReaders.f90)
const BINARY_TEXT_BYTES = 16

# ---------------------------------------------------------------------------------------
# Grid dimensions and array shapes
# ---------------------------------------------------------------------------------------

"""
    GridDims(mshape, dims)

The dimensions array input needs: `mshape` is MF6's `MODEL_SHAPE` (`[nlay, nrow, ncol]`
for DIS, `[nlay, ncpl]` for DISV, `[nodes]` for DISU, `set_model_shape`), and `dims`
maps upper-case dimension names (`"NLAY"`, `"NCPL"`, `"NODES"`, `"NJA"`, ...) to their
values. Build one with `dis_dims`, `disv_dims`, `disu_dims` or `grid_dims`.
"""
struct GridDims
    mshape::Vector{Int}
    dims::Dict{String, Int}
end

"""
    dis_dims(nlay, nrow, ncol) -> GridDims

Dimensions of a DIS grid (`NCPL = nrow*ncol`, `NODES = nlay*nrow*ncol`).
"""
dis_dims(nlay::Integer, nrow::Integer, ncol::Integer) = GridDims(
    [nlay, nrow, ncol],
    Dict("NLAY" => nlay, "NROW" => nrow, "NCOL" => ncol, "NCPL" => nrow * ncol, "NODES" => nlay * nrow * ncol),
)

"""
    disv_dims(nlay, ncpl; nvert = 0) -> GridDims

Dimensions of a DISV grid (`NODES = nlay*ncpl`).
"""
disv_dims(nlay::Integer, ncpl::Integer; nvert::Integer = 0) = GridDims(
    [nlay, ncpl], Dict("NLAY" => nlay, "NCPL" => ncpl, "NODES" => nlay * ncpl, "NVERT" => nvert),
)

"""
    disu_dims(nodes; nja = 0, nvert = 0) -> GridDims

Dimensions of a DISU grid (one layer for LAYERED input, `get_layered_shape`).
"""
disu_dims(nodes::Integer; nja::Integer = 0, nvert::Integer = 0) =
    GridDims([nodes], Dict("NODES" => nodes, "NJA" => nja, "NVERT" => nvert))

"""
    grid_dims(f::InputFile) -> GridDims

The dimensions of a DIS, DISV or DISU file (read with its DFN definition), from its
DIMENSIONS block. A missing or non-positive dimension is an `MF6InputError`, as in
MF6 (`set_model_shape`; NJA and NVERT may be absent).
"""
function grid_dims(f::InputFile)
    isnothing(f.dfn) && throw(ArgumentError("grid_dims: $(f.path) was read without a DFN definition"))
    kind = f.dfn.subcomponent
    kind in ("DIS", "DISV", "DISU") || throw(ArgumentError("grid_dims: $(f.path) is a $(f.dfn.name) file, not DIS, DISV or DISU"))
    b = find_block(f, "DIMENSIONS")
    isnothing(b) && throw(MF6InputError(f.path, 0, "DIMENSIONS block not found"))
    d = Dict{String, Int}()
    for l in b.lines
        tag = keyword(l, 1)
        d[tag] = get_int(l, 2; name = tag)
    end
    function need(n)
        v = get(d, n, 0)
        v >= 1 || input_error(b, "required input dimension \"$n\" not found or not positive")
        return v
    end
    if kind == "DIS"
        return dis_dims(need("NLAY"), need("NROW"), need("NCOL"))
    elseif kind == "DISV"
        return disv_dims(need("NLAY"), need("NCPL"); nvert = get(d, "NVERT", 0))
    else
        return disu_dims(need("NODES"); nja = get(d, "NJA", 0), nvert = get(d, "NVERT", 0))
    end
end

"""
    array_size(shape, g::GridDims) -> Vector{Int}

The size of an array with DFN shape `shape` (`"(ncol, nrow, nlay)"`, `"(nodes)"`,
`"(ncol*nrow; ncpl)"`) in grid `g`, translated as MF6's `dfn2f90.py` and
`get_shape_from_string` do: of `a; b` only `b` is used (`(ncol*nrow; ncpl)` is
`NCPL`), `NODES` is the product of the model shape, and a dimension written as a
product (`ncol*nrow`) is the product of its factors.
"""
function array_size(shape::AbstractString, g::GridDims)
    s = strip(shape)
    startswith(s, "(") && endswith(s, ")") && (s = s[2:(end - 1)])
    i = findlast(==(';'), s)
    isnothing(i) || (s = s[(i + 1):end])
    names = split(uppercase(s), (',', ' ', '\t'); keepempty = false)
    isempty(names) && throw(ArgumentError("array shape \"$shape\" has no dimensions"))
    return [prod(_dim_value(f, g, shape) for f in split(n, '*')) for n in names]
end

function _dim_value(name::AbstractString, g::GridDims, shape)
    name == "NODES" && return prod(g.mshape)
    v = get(g.dims, name, nothing)
    isnothing(v) && throw(ArgumentError("array shape \"$shape\": dimension $name is not known for this grid"))
    return v
end

"""
    layered_shape(g::GridDims) -> (nlay, layer_shape)

How LAYERED input splits an array (`get_layered_shape`): `nlay` control records, each
for `prod(layer_shape)` values; `layer_shape` is `(ncol, nrow)` for DIS (read row by
row), `(ncpl,)` for DISV and `(nodes,)` with `nlay = 1` for DISU.
"""
function layered_shape(g::GridDims)
    m = g.mshape
    length(m) == 1 && return 1, (m[1],)
    length(m) == 2 && return m[1], (m[2],)
    return m[1], (m[3], m[2])
end

_array_eltype(v::DfnVariable) =
    v.type === :double ? Float64 : v.type === :integer ? Int32 :
    throw(ArgumentError("variable $(v.name) of type $(v.type) is not array input"))

# ---------------------------------------------------------------------------------------
# Control record
# ---------------------------------------------------------------------------------------

"""
    ArrayControl{T}

A parsed array control record (`ArrayReaderBase.read_control_record`): `form` is
`:constant`, `:internal` or `:open_close`; `constant` (CONSTANT), `factor` (default 1),
`iprn` (default 0), `binary` and `file` (the OPEN/CLOSE name as written).
"""
struct ArrayControl{T}
    form::Symbol
    constant::T
    factor::T
    iprn::Int
    binary::Bool
    file::String
end

_parse_number(::Type{Float64}, s) = parse_double(s)
function _parse_number(::Type{Int32}, s)
    v = parse_int(s)
    (isnothing(v) || v < typemin(Int32) || v > typemax(Int32)) && return nothing
    return Int32(v)
end
_typename(::Type{Float64}) = "a double precision real"
_typename(::Type{Int32}) = "an integer"

function _control_number(::Type{T}, l::InputLine, t, what, name) where {T}
    isnothing(t) && input_error(l, "missing $what for array $name")
    v = _parse_number(T, t)
    isnothing(v) && input_error(l, "could not read $what of array $name as $(_typename(T)) from \"$t\"")
    return v
end

"""
    read_control_record(T, l::InputLine; name = "array") -> ArrayControl{T}

Parse the control record `l` of an array of element type `T` (`Float64` or `Int32`);
`name` is used in error messages.
"""
function read_control_record(::Type{T}, l::InputLine; name::AbstractString = "array") where {T}
    toks = eachtoken(l)
    st = iterate(toks)
    kw = isnothing(st) ? "" : uppercase(st[1])
    constant = zero(T)
    factor = one(T)
    iprn = 0
    binary = false
    file = ""
    if kw == "CONSTANT"
        st = iterate(toks, st[2])
        constant = _control_number(T, l, isnothing(st) ? nothing : st[1], "CONSTANT value", name)
        return ArrayControl{T}(:constant, constant, factor, iprn, binary, file)
    elseif kw == "INTERNAL"
        form = :internal
    elseif kw == "OPEN/CLOSE"
        form = :open_close
        st = iterate(toks, st[2])
        isnothing(st) && input_error(l, "missing file name after OPEN/CLOSE for array $name")
        file = String(st[1])
    else
        input_error(l, "error reading control record for $name: use CONSTANT, INTERNAL, or OPEN/CLOSE")
    end
    st = iterate(toks, st[2])
    while !isnothing(st)
        w = uppercase(st[1])
        if w == "FACTOR"
            st = iterate(toks, st[2])
            factor = _control_number(T, l, isnothing(st) ? nothing : st[1], "FACTOR", name)
        elseif w == "IPRN"
            st = iterate(toks, st[2])
            iprn = Int(_control_number(Int32, l, isnothing(st) ? nothing : st[1], "IPRN", name))
        elseif w == "(BINARY)"
            binary = true
        end
        isnothing(st) || (st = iterate(toks, st[2]))
    end
    return ArrayControl{T}(form, constant, factor, iprn, binary, file)
end

# ---------------------------------------------------------------------------------------
# Fortran list-directed reading
# ---------------------------------------------------------------------------------------

@inline _ld_blank(c::UInt8) = c == UInt8(' ') || c == UInt8('\t')
@inline _ld_end(c::UInt8) = _ld_blank(c) || c == UInt8(',') || c == UInt8('/')

@noinline function _ld_error(l::InputLine, name, i, n, msg)
    return input_error(l, "array $name, value $i of $n: $msg")
end

@noinline function _ld_eof(lines, k0, ctrl::InputLine, name, got, n, where)
    l = k0 > 1 && k0 - 1 <= length(lines) ? lines[k0 - 1] : ctrl
    return input_error(l, "array $name: $where after $got of $n values")
end

"""
    list_read!(out, first, n, lines, k, ctrl; name, where = "end of block") -> next k

Read `n` values into `out[first:first+n-1]` like one Fortran list-directed
`read(unit, *)` (rules in the header of `arrays.jl`), starting at record `lines[k]`, and
return the index of the first line after the last one used. `ctrl` is the line errors
point to when no data line was read; `where` names the end of `lines` in errors
(`"end of block"`, `"end of file"`).
"""
function list_read!(
        out::AbstractVector{T}, first::Int, n::Int, lines::AbstractVector{InputLine}, k::Int,
        ctrl::InputLine; name::AbstractString = "array", where::AbstractString = "end of block",
    ) where {T}
    # A Fortran read statement always uses at least one record, also for no values.
    if n == 0
        k > length(lines) && _ld_eof(lines, k, ctrl, name, 0, 0, where)
        return k + 1
    end
    i = first
    stop = first + n - 1
    lastcomma = false
    firstrec = true
    while i <= stop
        k > length(lines) && _ld_eof(lines, k, ctrl, name, i - first, n, where)
        l = lines[k]
        s = l.text.string
        p = l.text.offset + 1
        e = l.text.offset + ncodeunits(l.text)
        atstart = firstrec
        @inbounds while i <= stop && p <= e
            c = codeunit(s, p)
            if _ld_blank(c)
                p += 1
                continue
            elseif c == UInt8(',')
                (lastcomma || atstart) && _ld_error(l, name, i - first + 1, n, "null value (empty field between commas); MF6 does not allow null values")
                lastcomma = true
                atstart = false
                p += 1
                continue
            elseif c == UInt8('/')
                _ld_error(l, name, i - first + 1, n, "'/' ends the input before all values were read")
            end
            q = p
            star = 0
            while q <= e
                cq = codeunit(s, q)
                _ld_end(cq) && break
                (cq == UInt8('*') && star == 0) && (star = q)
                q += 1
            end
            if star == 0
                v = _parse_number(T, SubString(s, p, prevind(s, q)))
                isnothing(v) && _ld_error(l, name, i - first + 1, n, "could not read \"$(SubString(s, p, prevind(s, q)))\" as $(_typename(T))")
                out[i] = v
                i += 1
            else
                r = star > p ? parse_int(SubString(s, p, star - 1)) : nothing
                (isnothing(r) || r < 1) && _ld_error(l, name, i - first + 1, n, "bad repeat count in \"$(SubString(s, p, prevind(s, q)))\"")
                star + 1 < q || _ld_error(l, name, i - first + 1, n, "null value \"$(SubString(s, p, prevind(s, q)))\"; MF6 does not allow null values")
                v = _parse_number(T, SubString(s, star + 1, prevind(s, q)))
                isnothing(v) && _ld_error(l, name, i - first + 1, n, "could not read \"$(SubString(s, star + 1, prevind(s, q)))\" as $(_typename(T))")
                m = min(r, stop - i + 1)
                for j in i:(i + m - 1)
                    out[j] = v
                end
                i += m
            end
            lastcomma = false
            atstart = false
            p = q
        end
        firstrec = false
        k += 1
    end
    return k
end

# Read `n` values as `nrow` rows of `rowlen` (read_dbl2d: one read statement per row) or
# as one read (`rowlen = 0`).
function _read_ascii!(out, first, n, rowlen, lines, k, ctrl, name, where)
    if rowlen == 0 || rowlen == n
        return list_read!(out, first, n, lines, k, ctrl; name, where)
    end
    for r in 0:(div(n, rowlen) - 1)
        k = list_read!(out, first + r * rowlen, rowlen, lines, k, ctrl; name, where)
    end
    return k
end

# All lines of a text file as records (comment and blank lines included; a trailing
# carriage return is removed), for list-directed reading of OPEN/CLOSE files.
function _raw_lines(path::String)
    isfile(path) || throw(MF6InputError(path, 0, "file not found"))
    buf = read(path, String)
    n = ncodeunits(buf)
    lines = InputLine[]
    sizehint!(lines, count(==(UInt8('\n')), codeunits(buf)) + 1)
    i = 1
    lineno = 0
    while i <= n
        lineno += 1
        j = i
        while j <= n && codeunit(buf, j) != UInt8('\n')
            j += 1
        end
        stop = j - 1
        stop >= i && codeunit(buf, stop) == UInt8('\r') && (stop -= 1)
        push!(lines, InputLine(SubString(buf, i, stop >= i ? thisind(buf, stop) : i - 1), path, lineno))
        i = j + 1
    end
    return lines
end

# ---------------------------------------------------------------------------------------
# Binary array files
# ---------------------------------------------------------------------------------------

"""
    BinaryArrayHeader

The header record of an MF6 binary array file: `kstp`, `kper`, `pertim`, `totim`,
`text` (up to 16 characters, blank-padded in the file), `m1`, `m2`, `m3` (see the IO
guide, "Description of Binary Array Input Files").
"""
Base.@kwdef struct BinaryArrayHeader
    kstp::Int32 = 1
    kper::Int32 = 1
    pertim::Float64 = 0.0
    totim::Float64 = 0.0
    text::String = ""
    m1::Int32 = 0
    m2::Int32 = 1
    m3::Int32 = 1
end

_binary_eltype(::Type{<:AbstractFloat}) = Float64
_binary_eltype(::Type{<:Integer}) = Int32

"""
    read_binary_array(path, T, n; name = "array") -> (header, values::Vector{T})

Read an MF6 binary array file with `n` values of element type `T` (`Float64`: 8-byte
reals; `Int32`: 4-byte integers; little-endian, as MF6 writes them on x86/ARM). The
file size must be exactly the header plus `n` values (`check_binary_filesize`).
"""
function read_binary_array(path::AbstractString, ::Type{T}, n::Integer; name::AbstractString = "array") where {T}
    path = String(path)
    isfile(path) || throw(MF6InputError(path, 0, "binary file for array $name not found"))
    D = _binary_eltype(T)
    expected = BINARY_HEADER_BYTES + n * sizeof(D)
    found = filesize(path)
    found == expected || throw(
        MF6InputError(
            path, 0, "unexpected file size for binary input array $name: expected $expected bytes " *
                "(header + $n values of $(sizeof(D)) bytes), found $found bytes",
        ),
    )
    return open(path, "r") do io
        kstp = ltoh(read(io, Int32))
        kper = ltoh(read(io, Int32))
        pertim = ltoh(read(io, Float64))
        totim = ltoh(read(io, Float64))
        text = rstrip(String(read(io, BINARY_TEXT_BYTES)), [' ', '\0'])
        m1, m2, m3 = (ltoh(read(io, Int32)) for _ in 1:3)
        data = Vector{D}(undef, n)
        read!(io, data)
        ENDIAN_BOM == 0x04030201 || (data .= ltoh.(data))
        h = BinaryArrayHeader(; kstp, kper, pertim, totim, text = String(text), m1, m2, m3)
        return h, (D === T ? data : convert(Vector{T}, data))
    end
end

"""
    write_binary_array(path, values; kstp = 1, kper = 1, pertim = 0.0, totim = 0.0,
                       text = "", m1 = length(values), m2 = 1, m3 = 1)

Write `values` as an MF6 binary array file (one header record, then the values in
linear order, as Float64 for real and Int32 for integer element types). The defaults
of `m1`-`m3` are those of a non-layered array; for a layer of a DIS grid use
`m1 = ncol, m2 = nrow, m3 = layer` (IO guide). `text` is at most 16 characters.
"""
function write_binary_array(
        path::AbstractString, values::AbstractArray{T}; kstp::Integer = 1, kper::Integer = 1,
        pertim::Real = 0.0, totim::Real = 0.0, text::AbstractString = "",
        m1::Integer = length(values), m2::Integer = 1, m3::Integer = 1,
    ) where {T <: Real}
    ncodeunits(text) <= BINARY_TEXT_BYTES || throw(ArgumentError("binary array text \"$text\" is longer than $BINARY_TEXT_BYTES characters"))
    D = _binary_eltype(T)
    open(path, "w") do io
        write(io, htol(Int32(kstp)), htol(Int32(kper)), htol(Float64(pertim)), htol(Float64(totim)))
        t = collect(codeunits(text))
        write(io, append!(t, fill(UInt8(' '), BINARY_TEXT_BYTES - length(t))))
        write(io, htol(Int32(m1)), htol(Int32(m2)), htol(Int32(m3)))
        write(io, htol.(convert(Vector{D}, vec(values))))
    end
    return path
end

# ---------------------------------------------------------------------------------------
# Reading arrays
# ---------------------------------------------------------------------------------------

"""
    read_array!(out, lines, k; name = "array", simdir = ".", rowlen = 0) -> next k

Read one array with its control record `lines[k]` into `out` (element type `Float64`
or `Int32`), as `ArrayReaderBase.read_array` does, and return the index of the line
after the array. `rowlen = 0` reads the values as one list-directed read (MF6's 1-D
readers); `rowlen > 0` reads rows of `rowlen` values, each starting on a new line (2-D
readers, `length(out)` a multiple of `rowlen`). OPEN/CLOSE files are relative to
`simdir`, the simulation directory.
"""
function read_array!(
        out::AbstractArray{T}, lines::AbstractVector{InputLine}, k::Integer;
        name::AbstractString = "array", simdir::AbstractString = ".", rowlen::Integer = 0,
    ) where {T <: Union{Float64, Int32}}
    k <= length(lines) || throw(ArgumentError("read_array!: no control record for array $name (line index $k)"))
    ctrl = lines[k]
    c = read_control_record(T, ctrl; name)
    v = vec(out)
    n = length(v)
    rowlen > 0 && n % rowlen != 0 && throw(ArgumentError("array $name: $n values are not rows of $rowlen"))
    k += 1
    if c.form === :constant
        fill!(v, c.constant)
        return k
    elseif c.form === :internal
        k = _read_ascii!(v, 1, n, rowlen, lines, k, ctrl, name, "end of block")
    else
        path = _resolve_path(c.file, simdir)
        if c.binary
            _, data = read_binary_array(path, T, n; name)
            copyto!(v, data)
        else
            isfile(path) || input_error(ctrl, "OPEN/CLOSE file \"$(c.file)\" of array $name not found ($path)")
            flines = _raw_lines(path)
            isempty(flines) && input_error(ctrl, "OPEN/CLOSE file \"$(c.file)\" of array $name is empty")
            _read_ascii!(v, 1, n, rowlen, flines, 1, flines[1], name, "end of file")
        end
    end
    # apply_factor: a factor of zero is ignored.
    if c.factor != zero(T) && c.factor != one(T)
        f = c.factor
        @inbounds for i in eachindex(v)
            v[i] *= f
        end
    end
    return k
end

"""
    read_grid_array(lines, k, v::DfnVariable, g::GridDims; simdir = ".") -> (A, next k)

Read the array variable `v` (a DFN variable with `reader == :readarray`) whose name is on
line `lines[k]` (`K`, `BOTM LAYERED`), as MF6's input data model does (`load_double1d_type`,
`load_integer3d_type`, ...): the array has the size of `v.shape` in grid `g`
(`array_size`) and element type `Float64` or `Int32`. When the word after the name is
`LAYERED` and `v` is layered, one control record per layer follows (`layered_shape`);
otherwise one control record for the whole array (LAYERED on an array that is not
layered is ignored, as in MF6). NETCDF input is not supported, and `TIMEARRAYSERIES`
after the name is an error here (period data, T1.3).
"""
function read_grid_array(
        lines::AbstractVector{InputLine}, k::Integer, v::DfnVariable, g::GridDims;
        simdir::AbstractString = ".",
    )
    l = lines[k]
    T = _array_eltype(v)
    sz = try
        array_size(v.shape, g)
    catch e
        e isa ArgumentError || rethrow()
        input_error(l, "array $(uppercase(v.name)): $(e.msg)")
    end
    A = zeros(T, sz...)
    name = uppercase(v.name)
    opt = keyword(l, 2)
    opt == "NETCDF" && input_error(l, "NETCDF array input is not supported")
    # Mf6FileLayerArray.f90: `RECHARGE TIMEARRAYSERIES name` (READASARRAYS PERIOD blocks).
    opt == "TIMEARRAYSERIES" &&
        input_error(l, "$name TIMEARRAYSERIES: time-array series are read with the period data (T1.3), not by read_grid_array")
    if opt == "LAYERED" && v.layered
        nlay, lshape = layered_shape(g)
        ncpl = prod(lshape)
        nlay * ncpl == length(A) || input_error(l, "LAYERED array $name has $(length(A)) values, not $nlay layers of $ncpl")
        rowlen = length(lshape) == 2 ? lshape[1] : 0
        k += 1
        for lay in 1:nlay
            k <= length(lines) || input_error(l, "LAYERED array $name: missing control record for layer $lay of $nlay")
            seg = view(vec(A), ((lay - 1) * ncpl + 1):(lay * ncpl))
            k = read_array!(seg, lines, k; name = "$name (layer $lay)", simdir, rowlen)
        end
        return A, k
    end
    k += 1
    k <= length(lines) || input_error(l, "missing control record for array $name")
    rowlen = ndims(A) == 2 ? size(A, 1) : 0
    k = read_array!(A, lines, k; name, simdir, rowlen)
    return A, k
end

"""
    read_array_block(b::InputBlock, dfn::DfnDefinition, g::GridDims; simdir = ".")
        -> Dict{String, Array}

Read every array of block `b` (GRIDDATA, DISU CONNECTIONDATA, ...) of a file with
definition `dfn` (`LoadMf6File.parse_block`): each line starts with an array name of the
block, followed by its control record(s) and data (`read_grid_array`). The result maps
the DFN variable name (lower case, `"k"`, `"icelltype"`) to the array. A name that is
not a variable of the block (also a stray data line after an array) and a name given
twice are `MF6InputError`s.
"""
function read_array_block(b::InputBlock, dfn::DfnDefinition, g::GridDims; simdir::AbstractString = ".")
    db = dfn_block(dfn, b.name)
    isnothing(db) && input_error(b, "block $(b.name) is not part of a $(dfn.name) file")
    out = Dict{String, Array}()
    lines = b.lines
    k = 1
    while k <= length(lines)
        l = lines[k]
        tag = keyword(l, 1)
        v = dfn_variable(db, tag)
        isnothing(v) && input_error(l, "input tag \"$tag\" not found in block $(b.name) of a $(dfn.name) file")
        v.reader === :readarray || input_error(l, "$tag is not array input")
        haskey(out, v.name) && input_error(l, "array $tag is given more than once in block $(b.name)")
        A, k = read_grid_array(lines, k, v, g; simdir)
        out[v.name] = A
    end
    return out
end
