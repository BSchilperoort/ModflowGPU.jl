# MF6 cell-by-cell budget files (`.cbc`, `.bud`).
#
# MF6 writes only the "compact" budget records (NLAY stored negative, followed by a
# record with IMETH, DELT, PERTIM, TOTIM), with IMETH 1 or 6 (InputOutput.f90):
#
#   header (ubdsv1, ubdsv06):
#     KSTP, KPER (int32), TEXT (16 chars, right-justified), NDIM1, NDIM2, -NDIM3 (int32)
#     IMETH (int32), DELT, PERTIM, TOTIM (real64)
#   IMETH 1 (ubdsv1; `record_array` with a negative unit, `record_connection_array`):
#     NDIM1*NDIM2*NDIM3 values (real64)
#   IMETH 6 (ubdsv06, then ubdsvd per entry; `record_srcdst_list_header` and
#   `record_srcdst_list_entry` of DiscretizationBase.f90):
#     TXT1ID1 (model), TXT2ID1 (package or model), TXT1ID2 (model), TXT2ID2 (package),
#     each 16 chars, left-justified
#     NDAT = NAUX + 1 (int32), NAUX auxiliary names (16 chars each), NLIST (int32)
#     NLIST entries: ID1, ID2 (int32), Q, AUX(1:NAUX) (real64)
#
# NDIM1-3 are the model grid: (NCOL, NROW, NLAY) for DIS, (NCPL, 1, NLAY) for DISV and
# (NODES, 1, 1) for DISU, except for the FLOW-JA-FACE array of `record_connection_array`,
# which is (NJA, 1, 1). ID1/ID2 are user node numbers (or, for exchanges and
# advanced-package entries, the numbers the writer chose). Storage (STO-SS, STO-SY) is an
# IMETH 1 array over all user nodes; boundary packages (CHD, WEL, RCH, ...) and GWF-GWF
# exchanges (text FLOW-JA-FACE, package name the exchange name) are IMETH 6 lists.

const BUDGET_HEADER_BYTES = 4 * 2 + OUTPUT_TEXT_BYTES + 4 * 3 + 4 + 8 * 3

"""
    BudgetRecord

Header of one record of a budget file, as indexed by [`open_budget_file`](@ref):

- `kstp`, `kper`, `text` (16 characters as in the file, e.g. `"    FLOW-JA-FACE"`),
  `ndim1`, `ndim2`, `ndim3` (the grid dimensions; `ndim3` as written, i.e. negative
  for MF6's compact header), `imeth`, `delt`, `pertim`, `totim`;
- for IMETH 6: `modelnam1`, `paknam1`, `modelnam2`, `paknam2` (16 characters each),
  `auxtxt` (auxiliary names, 16 characters each); empty strings for IMETH 1;
- `n`: number of values (IMETH 1) or list entries (IMETH 6), and `offset`, the byte
  offset of the data.
"""
struct BudgetRecord
    kstp::Int32
    kper::Int32
    text::String
    ndim1::Int32
    ndim2::Int32
    ndim3::Int32
    imeth::Int32
    delt::Float64
    pertim::Float64
    totim::Float64
    modelnam1::String
    paknam1::String
    modelnam2::String
    paknam2::String
    auxtxt::Vector{String}
    n::Int64
    offset::Int64
end

"""
    BudgetList

The entries of an IMETH 6 budget record: `id1`, `id2` (int32 node numbers), `q` (flow,
positive into the model cell `id1`), and `aux` (`naux × nlist` matrix of auxiliary values,
in the order of the record's `auxtxt`).
"""
struct BudgetList
    id1::Vector{Int32}
    id2::Vector{Int32}
    q::Vector{Float64}
    aux::Matrix{Float64}
end

BudgetList(id1, id2, q) = BudgetList(id1, id2, q, zeros(Float64, 0, length(q)))

Base.length(l::BudgetList) = length(l.q)
Base.:(==)(a::BudgetList, b::BudgetList) = a.id1 == b.id1 && a.id2 == b.id2 && a.q == b.q && a.aux == b.aux

"""
    BudgetFile

Index of a budget file, built by [`open_budget_file`](@ref): the `path` and the header of
every record (`records`, [`BudgetRecord`](@ref)s in file order). Data are read lazily by
[`read_budget`](@ref).
"""
struct BudgetFile
    path::String
    records::Vector{BudgetRecord}
end

Base.length(bf::BudgetFile) = length(bf.records)

function Base.show(io::IO, bf::BudgetFile)
    return print(io, "BudgetFile(\"", bf.path, "\", ", length(bf.records), " records, ", length(budget_kstpkper(bf)), " time steps)")
end

"""
    open_budget_file(path) -> BudgetFile

Index the budget file at `path`: read the header of every record (not the data). Only
compact records with IMETH 1 or 6, the ones MF6 writes, are supported.
"""
function open_budget_file(path::AbstractString)
    path = String(path)
    fsize = filesize(path)
    records = BudgetRecord[]
    open(path, "r") do io
        pos = Int64(0)
        while pos < fsize
            pos + BUDGET_HEADER_BYTES <= fsize || throw(MF6BinaryFileError(path, pos, "file ends inside a record header"))
            seek(io, pos)
            kstp = _read_le(io, Int32)
            kper = _read_le(io, Int32)
            text = _read_text(io)
            ndim1, ndim2, ndim3 = _read_le(io, Int32), _read_le(io, Int32), _read_le(io, Int32)
            ndim3 < 0 || throw(
                MF6BinaryFileError(path, pos, "record $(strip(text)) has a full (non-compact) header (NDIM3 = $ndim3 > 0); only MF6's compact records are supported"),
            )
            imeth = _read_le(io, Int32)
            delt, pertim, totim = _read_le(io, Float64), _read_le(io, Float64), _read_le(io, Float64)
            m1 = m2 = m3 = m4 = ""
            auxtxt = String[]
            if imeth == 1
                n = Int64(ndim1) * Int64(ndim2) * Int64(-ndim3)
                n >= 0 || throw(MF6BinaryFileError(path, pos, "negative array size in record $(strip(text))"))
                data = position(io)
                next = data + 8n
            elseif imeth == 6
                position(io) + 4 * OUTPUT_TEXT_BYTES + 4 <= fsize || throw(MF6BinaryFileError(path, pos, "file ends inside a record header"))
                m1, m2, m3, m4 = _read_text(io), _read_text(io), _read_text(io), _read_text(io)
                ndat = _read_le(io, Int32)
                ndat >= 1 || throw(MF6BinaryFileError(path, pos, "NDAT = $ndat < 1 in record $(strip(text))"))
                position(io) + OUTPUT_TEXT_BYTES * (ndat - 1) + 4 <= fsize || throw(MF6BinaryFileError(path, pos, "file ends inside a record header"))
                auxtxt = [_read_text(io) for _ in 1:(ndat - 1)]
                n = Int64(_read_le(io, Int32))
                n >= 0 || throw(MF6BinaryFileError(path, pos, "NLIST = $n < 0 in record $(strip(text))"))
                data = position(io)
                next = data + n * (8 + 8 * Int64(ndat))
            else
                throw(MF6BinaryFileError(path, pos, "record $(strip(text)) has IMETH = $imeth; only IMETH 1 and 6 (written by MF6) are supported"))
            end
            next <= fsize || throw(MF6BinaryFileError(path, pos, "file ends inside the data of record $(strip(text)) (KSTP $kstp, KPER $kper)"))
            push!(records, BudgetRecord(kstp, kper, text, ndim1, ndim2, ndim3, imeth, delt, pertim, totim, m1, m2, m3, m4, auxtxt, n, data))
            pos = next
        end
    end
    return BudgetFile(path, records)
end

"""
    budget_texts(bf) -> Vector{String}

The distinct record texts of the budget file (without padding), in order of first
appearance.
"""
budget_texts(bf::BudgetFile) = unique(String(strip(r.text)) for r in bf.records)

"""
    budget_kstpkper(bf) -> Vector{Tuple{Int, Int}}

The distinct `(kstp, kper)` of the budget file, in file order.
"""
budget_kstpkper(bf::BudgetFile) = unique((Int(r.kstp), Int(r.kper)) for r in bf.records)

"""
    budget_times(bf) -> Vector{Float64}

The distinct simulation times (`totim`) of the budget file, in file order.
"""
budget_times(bf::BudgetFile) = unique(r.totim for r in bf.records)

_text_eq(a::AbstractString, b::AbstractString) = uppercase(strip(a)) == uppercase(strip(b))

"""
    find_budget_records(bf; text, kstp, kper, totim, imeth, paknam2, modelnam2) -> Vector{Int}

Indices of the records matching all given keys. Texts and names are compared without
padding and case-insensitively. Note that GWF-GWF exchange flows are IMETH 6 records with
text `FLOW-JA-FACE` (and the exchange name as `paknam2`); use `imeth = 1` for the
model's own FLOW-JA-FACE array.
"""
function find_budget_records(
        bf::BudgetFile; text = nothing, kstp = nothing, kper = nothing, totim = nothing,
        imeth = nothing, paknam2 = nothing, modelnam2 = nothing,
    )
    return findall(bf.records) do r
        (text === nothing || _text_eq(r.text, text)) &&
            (kstp === nothing || r.kstp == kstp) && (kper === nothing || r.kper == kper) &&
            (totim === nothing || r.totim == totim) && (imeth === nothing || r.imeth == imeth) &&
            (paknam2 === nothing || _text_eq(r.paknam2, paknam2)) &&
            (modelnam2 === nothing || _text_eq(r.modelnam2, modelnam2))
    end
end

function _read_budget(io::IO, r::BudgetRecord)
    seek(io, r.offset)
    r.imeth == 1 && return _read_le!(io, Vector{Float64}(undef, r.n))
    naux = length(r.auxtxt)
    recbytes = 8 + 8 * (naux + 1)
    buf = read(io, r.n * recbytes)
    id1 = Vector{Int32}(undef, r.n)
    id2 = Vector{Int32}(undef, r.n)
    q = Vector{Float64}(undef, r.n)
    aux = Matrix{Float64}(undef, naux, r.n)
    @inbounds for i in 1:r.n
        off = (i - 1) * recbytes
        id1[i] = _load_le(Int32, buf, off)
        id2[i] = _load_le(Int32, buf, off + 4)
        q[i] = _load_le(Float64, buf, off + 8)
        for a in 1:naux
            aux[a, i] = _load_le(Float64, buf, off + 8 + 8a)
        end
    end
    return BudgetList(id1, id2, q, aux)
end

"""
    read_budget(bf, i) -> Vector{Float64} or BudgetList
    read_budget(bf; text, kstp, kper, ...) -> Vector

Read the data of record `i`: a vector of `n` values for IMETH 1 (user-node order,
`reshape(v, ndim1, ndim2, -ndim3)` gives the grid shape; FLOW-JA-FACE is a vector of
length NJA), a [`BudgetList`](@ref) for IMETH 6. The keyword form reads all records
matching [`find_budget_records`](@ref), in file order.
"""
function read_budget(bf::BudgetFile, i::Integer)
    r = bf.records[i]
    return open(io -> _read_budget(io, r), bf.path, "r")
end

function read_budget(bf::BudgetFile; kw...)
    idx = find_budget_records(bf; kw...)
    return open(bf.path, "r") do io
        [_read_budget(io, bf.records[i]) for i in idx]
    end
end

function _write_budget_header(io, text, kstp, kper, dims, imeth, delt, pertim, totim)
    write(io, htol(Int32(kstp)), htol(Int32(kper)))
    write(io, pad_text(text; justify = :right))
    write(io, htol(Int32(dims[1])), htol(Int32(dims[2])), htol(-Int32(dims[3])))
    write(io, htol(Int32(imeth)), htol(Float64(delt)), htol(Float64(pertim)), htol(Float64(totim)))
    return nothing
end

"""
    write_budget_array(io, layout::OutputLayout, text, values; kstp, kper, delt, pertim, totim)
    write_budget_array(io, text, values; kstp, kper, delt, pertim, totim, dims = ...)

Append an IMETH 1 record (`ubdsv1`) to the budget file open as `io`.

The first form writes an array over the user nodes of `layout` (e.g. `STO-SS`,
`STO-SY`; any array of length [`nodesuser`](@ref)`(layout)` in user-node order) with the
grid dimensions of the layout as NDIM1-3, as `record_array` does.

The second form takes `dims = (ndim1, ndim2, ndim3)` explicitly. Use it for
`FLOW-JA-FACE` (`record_connection_array`): a vector over the connections of the
reduced (active-node) CSR matrix, i.e. the NJA of the binary grid file, with the default
`dims = (nja, 1, 1)`. Without `dims`, they follow the shape of `values` (3-D: DIS, 2-D:
`(ncpl, 1, nlay)`, vector: `(length, 1, 1)`), a convenience for arrays in MF6 memory
shape only.

NDIM3 is written negative (compact header). `text` is right-justified to 16 characters
(a 16-character `text` is written unchanged). Values are written as Float64.
"""
function write_budget_array(
        io::IO, layout::OutputLayout, text::AbstractString, values::AbstractArray{<:Real}; kw...,
    )
    _check_length(layout, length(values), "budget array $(strip(text))")
    return write_budget_array(io, text, values; kw..., dims = budget_dims(layout))
end

function write_budget_array(
        io::IO, text::AbstractString, values::AbstractArray{<:Real}; kstp::Integer, kper::Integer,
        delt::Real, pertim::Real, totim::Real, dims = budget_dims(OutputLayout(values)),
    )
    prod(dims) == length(values) || throw(DimensionMismatch("dims $dims do not match the $(length(values)) values"))
    _write_budget_header(io, text, kstp, kper, dims, 1, delt, pertim, totim)
    _write_le(io, values isa Array{Float64} ? values : convert(Array{Float64}, values))
    return nothing
end

"""
    write_budget_list(io, layout::OutputLayout, text, list::BudgetList; kstp, kper, delt,
                      pertim, totim, modelnam1, paknam1, modelnam2, paknam2, auxtxt = String[])
    write_budget_list(io, text, list::BudgetList; dims, kw...)

Append an IMETH 6 record (`ubdsv06`, then `ubdsvd` per entry) to the budget file open as
`io`. The header holds the model grid dimensions, [`budget_dims`](@ref)`(layout)` or
`dims = (ndim1, ndim2, ndim3)` (`(ncol, nrow, nlay)`, `(ncpl, 1, nlay)` or
`(nodes, 1, 1)`; NDIM3 is written negative). For a boundary
package MF6 writes `modelnam1 = paknam1 = modelnam2 =` the model name and `paknam2 =` the
package name; for a GWF-GWF exchange `text = "FLOW-JA-FACE"`, `paknam1 = paknam2 =` the
exchange name and `modelnam2` the other model. `auxtxt` has one name per row of
`list.aux`. `text` is right-justified, names upper-cased (MF6 stores model and package
names in upper case) and left-justified to 16 characters, auxiliary names left-justified
as given (16-character strings are written unchanged).
"""
function write_budget_list(io::IO, layout::OutputLayout, text::AbstractString, list::BudgetList; kw...)
    return write_budget_list(io, text, list; kw..., dims = budget_dims(layout))
end

function write_budget_list(
        io::IO, text::AbstractString, list::BudgetList; kstp::Integer, kper::Integer,
        delt::Real, pertim::Real, totim::Real, dims, modelnam1::AbstractString,
        paknam1::AbstractString, modelnam2::AbstractString, paknam2::AbstractString,
        auxtxt::AbstractVector{<:AbstractString} = String[],
    )
    n = length(list.q)
    (length(list.id1) == n && length(list.id2) == n) || throw(ArgumentError("id1, id2 and q must have the same length"))
    naux = length(auxtxt)
    size(list.aux) == (naux, n) || throw(ArgumentError("aux must be $naux × $n (one row per auxiliary name), got $(size(list.aux))"))
    _write_budget_header(io, text, kstp, kper, dims, 6, delt, pertim, totim)
    for s in (modelnam1, paknam1, modelnam2, paknam2)
        write(io, pad_text(uppercase(s); justify = :left))
    end
    write(io, htol(Int32(naux + 1)))
    for a in auxtxt
        write(io, pad_text(a; justify = :left))
    end
    write(io, htol(Int32(n)))
    recbytes = 8 + 8 * (naux + 1)
    buf = Vector{UInt8}(undef, n * recbytes)
    GC.@preserve buf @inbounds for i in 1:n
        p = pointer(buf, (i - 1) * recbytes + 1)
        unsafe_store!(Ptr{Int32}(p), htol(Int32(list.id1[i])))
        unsafe_store!(Ptr{Int32}(p + 4), htol(Int32(list.id2[i])))
        unsafe_store!(Ptr{Float64}(p + 8), htol(Float64(list.q[i])))
        for a in 1:naux
            unsafe_store!(Ptr{Float64}(p + 8 + 8a), htol(Float64(list.aux[a, i])))
        end
    end
    write(io, buf)
    return nothing
end

"""
    write_budget_file(path, bf::BudgetFile) -> path

Write every record of `bf` (read with [`read_budget`](@ref)) to a new budget file with
[`write_budget_array`](@ref) and [`write_budget_list`](@ref). For a file written by MF6
the result is byte-identical.
"""
function write_budget_file(path::AbstractString, bf::BudgetFile)
    open(bf.path, "r") do src
        open(path, "w") do io
            for r in bf.records
                d = _read_budget(src, r)
                kw = (; r.kstp, r.kper, r.delt, r.pertim, r.totim, dims = (r.ndim1, r.ndim2, -r.ndim3))
                if r.imeth == 1
                    write_budget_array(io, r.text, d; kw...)
                else
                    write_budget_list(io, r.text, d; kw..., r.modelnam1, r.paknam1, r.modelnam2, r.paknam2, r.auxtxt)
                end
            end
        end
    end
    return path
end
