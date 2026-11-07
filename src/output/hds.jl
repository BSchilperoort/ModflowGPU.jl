# MF6 head files (and other dependent-variable files with the same layout).
#
# Each saved array is written by `record_array` (Dis.f90, Disv.f90, Disu.f90) through
# `ulasav` (InputOutput.f90) as one record per layer:
#
#     KSTP, KPER (int32), PERTIM, TOTIM (real64), TEXT (16 chars), M1, M2, M3 (int32)
#     M1*M2 values (real64)
#
# DIS:  M1 = NCOL, M2 = NROW, M3 = layer, one record per layer;
# DISV: M1 = NCPL, M2 = 1,    M3 = layer, one record per layer;
# DISU: M1 = NODES, M2 = 1,   M3 = 1, one record.
#
# TEXT is the left-justified OC variable name, `"HEAD            "` for GWF
# (gwf-oc.f90). Cells outside the model (IDOMAIN <= 0) hold HNOFLO = 1e30 (`dinact`
# of `record_array`, `dnodata` of OutputControlData.f90); dry cells hold HDRY = -1e30
# (`DHNOFLO`, `DHDRY` in Constants.f90). Values are written as stored in MF6 (Float64).

"""HNOFLO: head written for cells outside the model (`DHNOFLO` in Constants.f90)."""
const HNOFLO = 1.0e30
"""HDRY: head written for dry cells (`DHDRY` in Constants.f90)."""
const HDRY = -1.0e30

const HEAD_HEADER_BYTES = 52  # 2 Int32 + 2 Float64 + 16 characters + 3 Int32

"""
    HeadStep

One saved array of a head file (all its layer records): `kstp`, `kper`, `pertim`,
`totim`, `text` (16 characters as in the file, e.g. `"HEAD            "`), `m1`, `m2`
(per-layer record size), `nlay` (number of layer records) and the byte `offsets` of the
data of each layer record.
"""
struct HeadStep
    kstp::Int32
    kper::Int32
    pertim::Float64
    totim::Float64
    text::String
    m1::Int32
    m2::Int32
    nlay::Int32
    offsets::Vector{Int64}
end

"""
    HeadFile

Index of an MF6 head file, built by [`open_head_file`](@ref): the `path`, the `grid`
layout used to shape arrays (`:dis`, `:disv`, `:disu` or `:auto`) and the saved arrays
(`steps`, one [`HeadStep`](@ref) each, in file order). Data are read lazily by
[`read_head`](@ref).
"""
struct HeadFile
    path::String
    grid::Symbol
    steps::Vector{HeadStep}
end

Base.length(hf::HeadFile) = length(hf.steps)

function Base.show(io::IO, hf::HeadFile)
    print(io, "HeadFile(\"", hf.path, "\", ", length(hf.steps), " arrays")
    isempty(hf.steps) || print(io, ", ", strip(hf.steps[1].text), " ", hf.steps[1].m1, "×", hf.steps[1].m2, "×", hf.steps[1].nlay)
    return print(io, ")")
end

"""
    open_head_file(path; grid = :auto) -> HeadFile

Index the head file at `path`: read every record header (not the data) and group the
layer records of each saved array. `grid` sets the shape of the arrays returned by
[`read_head`](@ref): `:dis` gives `(ncol, nrow, nlay)`, `:disv` `(ncpl, nlay)`, `:disu`
a vector of nodes, and `:auto` the record layout `(m1, m2, nlay)` (which is the DIS
shape). All values are Float64, in MF6's user-node order.
"""
function open_head_file(path::AbstractString; grid::Symbol = :auto)
    grid in (:auto, :dis, :disv, :disu) || throw(ArgumentError("grid must be :auto, :dis, :disv or :disu, got :$grid"))
    path = String(path)
    fsize = filesize(path)
    steps = HeadStep[]
    open(path, "r") do io
        pos = Int64(0)
        while pos < fsize
            pos + HEAD_HEADER_BYTES <= fsize || throw(MF6BinaryFileError(path, pos, "file ends inside a record header"))
            seek(io, pos)
            kstp = _read_le(io, Int32)
            kper = _read_le(io, Int32)
            pertim = _read_le(io, Float64)
            totim = _read_le(io, Float64)
            text = _read_text(io)
            m1, m2, m3 = _read_le(io, Int32), _read_le(io, Int32), _read_le(io, Int32)
            (m1 >= 0 && m2 >= 0) || throw(MF6BinaryFileError(path, pos, "negative array size M1 = $m1, M2 = $m2"))
            data = pos + HEAD_HEADER_BYTES
            next = data + 8 * Int64(m1) * Int64(m2)
            next <= fsize || throw(MF6BinaryFileError(path, pos, "file ends inside the data of a record ($text, KSTP $kstp, KPER $kper, layer $m3)"))
            s = isempty(steps) ? nothing : steps[end]
            if s !== nothing && s.kstp == kstp && s.kper == kper && s.totim == totim && s.text == text &&
                    m3 == s.nlay + 1 && s.m1 == m1 && s.m2 == m2
                push!(s.offsets, data)
                steps[end] = HeadStep(s.kstp, s.kper, s.pertim, s.totim, s.text, s.m1, s.m2, s.nlay + 1, s.offsets)
            else
                m3 == 1 || throw(
                    MF6BinaryFileError(
                        path, pos, "record of layer $m3 ($text, KSTP $kstp, KPER $kper) does not continue " *
                            "the layers of the previous record; MF6 writes layers 1, 2, ... in order",
                    ),
                )
                push!(steps, HeadStep(kstp, kper, pertim, totim, text, m1, m2, 1, [data]))
            end
            pos = next
        end
    end
    return HeadFile(path, grid, steps)
end

"""
    head_kstpkper(hf) -> Vector{Tuple{Int, Int}}

`(kstp, kper)` of each saved array of the head file, in file order.
"""
head_kstpkper(hf::HeadFile) = [(Int(s.kstp), Int(s.kper)) for s in hf.steps]

"""
    head_times(hf) -> Vector{Float64}

Simulation time (`totim`) of each saved array of the head file, in file order.
"""
head_times(hf::HeadFile) = [s.totim for s in hf.steps]

function _head_shape(hf::HeadFile, s::HeadStep)
    n = Int(s.m1) * Int(s.m2)
    hf.grid === :auto && return (Int(s.m1), Int(s.m2), Int(s.nlay))
    hf.grid === :dis && return (Int(s.m1), Int(s.m2), Int(s.nlay))
    hf.grid === :disv && return (n, Int(s.nlay))
    return (n * Int(s.nlay),)
end

"""
    find_head_step(hf; kstp = nothing, kper = nothing, totim = nothing, text = nothing) -> Int

Index of the saved array matching all given keys (`text` is compared without the
padding blanks and case-insensitively). Throws if there is no match or more than one.
"""
function find_head_step(hf::HeadFile; kstp = nothing, kper = nothing, totim = nothing, text = nothing)
    idx = findall(hf.steps) do s
        (kstp === nothing || s.kstp == kstp) && (kper === nothing || s.kper == kper) &&
            (totim === nothing || s.totim == totim) &&
            (text === nothing || uppercase(strip(s.text)) == uppercase(strip(text)))
    end
    keys = join(("$k = $v" for (k, v) in pairs((; kstp, kper, totim, text)) if v !== nothing), ", ")
    isempty(idx) && throw(ArgumentError("no saved array with $keys in $(hf.path)"))
    length(idx) > 1 && throw(ArgumentError("$(length(idx)) saved arrays with $keys in $(hf.path); add keys to select one"))
    return idx[1]
end

"""
    read_head(hf, i) -> Array{Float64}
    read_head(hf; kstp, kper, totim, text) -> Array{Float64}

Read the `i`-th saved array of the head file (or the one selected by
[`find_head_step`](@ref)), shaped as set by the `grid` of [`open_head_file`](@ref).
Inactive cells hold [`HNOFLO`](@ref), dry cells [`HDRY`](@ref).
"""
function read_head(hf::HeadFile, i::Integer)
    s = hf.steps[i]
    n = Int(s.m1) * Int(s.m2)
    out = Vector{Float64}(undef, n * Int(s.nlay))
    open(hf.path, "r") do io
        for (k, off) in enumerate(s.offsets)
            seek(io, off)
            _read_le!(io, out, (k - 1) * n + 1, n)
        end
    end
    return reshape(out, _head_shape(hf, s))
end

read_head(hf::HeadFile; kw...) = read_head(hf, find_head_step(hf; kw...))

"""
    head_output_values(layout, values; nodeuser = nothing, inactive = nothing, dry = nothing)
        -> Vector{Float64}

The Float64 array over the user nodes that MF6 writes for `values` (`record_array`):

- without `nodeuser`, `values` has one value per user node (any array of length
  [`nodesuser`](@ref)`(layout)`, in user-node order);
- with `nodeuser`, `values` are the values of the reduced (active) nodes and
  `nodeuser[i]` is the user node of reduced node `i`; all other user nodes get
  [`HNOFLO`](@ref), as `dinact` in `record_array`;
- `inactive` (a `Bool` mask over the user nodes) sets cells to [`HNOFLO`](@ref);
- `dry` (a `Bool` mask indexed like `values`) sets cells to [`HDRY`](@ref).

The sentinels are set after the conversion to Float64. Pass Float32 heads with masks
rather than storing `1f30` in them: `Float64(1f30) != 1e30`, so a converted Float32
sentinel is not recognised as HNOFLO by MF6-compatible readers.
"""
function head_output_values(
        layout::OutputLayout, values::AbstractArray{<:Real}; nodeuser = nothing, inactive = nothing, dry = nothing,
    )
    nu = nodesuser(layout)
    dry === nothing || length(dry) == length(values) ||
        throw(DimensionMismatch("dry mask has $(length(dry)) values, values has $(length(values))"))
    if nodeuser === nothing
        _check_length(layout, length(values), "head array")
        v = Vector{Float64}(undef, nu)
        copyto!(v, vec(values))  # converts to Float64
        dry === nothing || (v[vec(dry)] .= HDRY)
    else
        length(nodeuser) == length(values) ||
            throw(DimensionMismatch("nodeuser has $(length(nodeuser)) entries, values has $(length(values))"))
        v = fill(HNOFLO, nu)
        for (i, n) in enumerate(nodeuser)
            1 <= n <= nu || throw(ArgumentError("nodeuser[$i] = $n is not a user node of the layout (1:$nu)"))
            v[n] = (dry !== nothing && dry[i]) ? HDRY : Float64(values[i])
        end
    end
    if inactive !== nothing
        length(inactive) == nu || throw(DimensionMismatch("inactive mask has $(length(inactive)) values, the layout has $nu user nodes"))
        v[vec(inactive)] .= HNOFLO
    end
    return v
end

"""
    write_head_record(io, layout::OutputLayout, values; kstp, kper, pertim, totim,
                      text = "HEAD", nodeuser = nothing, inactive = nothing, dry = nothing)
    write_head_record(io, values::AbstractArray; kw...)

Append one saved array to the head file open as `io`, as MF6's `record_array` and
`ulasav` write it, with the records of `layout` ([`OutputLayout`](@ref)):

- `:dis`: one record per layer with M1 = ncol, M2 = nrow, M3 = layer;
- `:disv`: one record per layer with M1 = ncpl, M2 = 1, M3 = layer;
- `:disu`: one record with M1 = nodes, M2 = 1, M3 = 1.

The values written are [`head_output_values`](@ref)`(layout, values; nodeuser, inactive,
dry)`: Float64 over the user nodes, with [`HNOFLO`](@ref) and [`HDRY`](@ref) inserted
from the masks (or already present in Float64 `values`). `text` is left-justified to 16
characters, as MF6's OC names are; a 16-character `text` is written unchanged.

The second form takes the layout from the shape of `values` (see
[`OutputLayout`](@ref)`(values)`): a convenience for arrays in MF6 memory shape only,
since a flat vector is written as DISU.
"""
function write_head_record(
        io::IO, layout::OutputLayout, values::AbstractArray{<:Real}; kstp::Integer, kper::Integer,
        pertim::Real, totim::Real, text::AbstractString = "HEAD", nodeuser = nothing, inactive = nothing,
        dry = nothing,
    )
    t = pad_text(text; justify = :left)
    v = head_output_values(layout, values; nodeuser, inactive, dry)
    m1, m2 = layout.ncol, layout.nrow
    n = m1 * m2
    for k in 1:layout.nlay
        write(io, htol(Int32(kstp)), htol(Int32(kper)), htol(Float64(pertim)), htol(Float64(totim)))
        write(io, t)
        write(io, htol(Int32(m1)), htol(Int32(m2)), htol(Int32(k)))
        _write_le(io, view(v, ((k - 1) * n + 1):(k * n)))
    end
    return nothing
end

write_head_record(io::IO, values::AbstractArray{<:Real}; kw...) = write_head_record(io, OutputLayout(values), values; kw...)

"""
    write_head_file(path, hf::HeadFile) -> path

Write every saved array of `hf` (read with [`read_head`](@ref)) to a new head file with
[`write_head_record`](@ref). For a file written by MF6 the result is byte-identical.
"""
function write_head_file(path::AbstractString, hf::HeadFile)
    open(path, "w") do io
        for (i, s) in enumerate(hf.steps)
            h = read_head(hf, i)
            # The record layout; for a DISU file (one record) this is the same bytes.
            layout = OutputLayout(:dis, s.m1, s.m2, s.nlay)
            write_head_record(io, layout, h; s.kstp, s.kper, s.pertim, s.totim, s.text)
        end
    end
    return path
end
