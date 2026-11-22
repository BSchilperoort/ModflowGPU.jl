# List and period input for MF6 text input (host only).
#
# Ports the period loading of MF6's input data model (IDM) for the GWF packages in scope:
# - IdmMf6File.f90 `dynamic_rp`, `dynamic_read_ionper`: a PERIOD block is read in the
#   stress period of its number; nothing is read in the periods in between, so the data
#   of a block apply from its period until the next PERIOD block (or the end).
# - Mf6FileList.f90 `ListLoadType%rp` with StructArray.f90 `read_from_parser`,
#   `read_param`, `read_from_binary` and LoadMf6File.f90 `read_control_record`: list
#   input (CHD, WEL, DRN, RIV, GHB, RCH, EVT). Every block replaces the whole list; an
#   empty block gives an empty list (no stresses). The columns in scope come from the DFN
#   record and the options (LoadContext.f90 `in_scope`): `aux` only with AUXILIARY,
#   `boundname` only with BOUNDNAMES, EVT `pxdp`/`petm` only with NSEG > 1 and `petm0`
#   only with SURF_RATE_SPECIFIED.
# - BoundaryPackageExt.f90 `cellid_to_nlist`, `check_cellid`: cellids are checked against
#   the model shape and converted to user node numbers (`get_node`).
# - Mf6FileLayerArray.f90 `LayerArrayLoadType%rp`, `param_load`: READASARRAYS input
#   (RCHA, EVTA). Arrays not given in a block keep their values (all start at zero); the
#   layer array IRCH/IEVT starts at 1 (`default_nodelist`, `nlarray_to_nodelist`).
# - Mf6FileStoInput.f90 `sto_rp` (STO STEADY-STATE/TRANSIENT) and Mf6FileSetting.f90 with
#   OutputControl.f90 `oc_rp` and TimeStepSelect.f90 `read` (OC SAVE/PRINT records).
# - DisConnExchange.f90 `source_data` (GWF-GWF EXCHANGEDATA, static struct array): two
#   cellids, one per model grid, then IHC, CL1, CL2, HWVA, aux, boundname.
#
# List rules (checked against the Fortran):
# - Each line is one record, read column by column with `GetInteger` / `GetDouble` /
#   `GetString` (URWORD words). A missing value is an error, with two exceptions: a
#   missing boundname is blank, and the last column, when it is an optional scalar
#   double that cannot hold a time series (or TS6 is not active), is read with
#   `TryGetDouble` and is DNODATA (3.0e30) when missing (`read_param` tests the
#   time-series branch first; vector columns such as EVT `petm(nseg-1)` always use
#   `GetDouble`). Words after the last column are ignored (comments).
# - Integers (cellid) are Fortran integers (`1.0` is an error). A cellid outside the
#   model shape is an error (`check_cellid`).
# - More records than MAXBOUND is an error ("line count exceeds input dimension").
# - Boundnames and time-series names are converted to upper case (`GetString` with
#   `convertToUpper`, `read_value_or_time_series`); boundnames are cut to 40 characters
#   (LENBOUNDNAME).
# - Time series: when the OPTIONS block has `TS6 FILEIN`, a word that is not a number in
#   a column whose DFN entry has `time_series` is a time-series name
#   (`StructVector.read_token`). It is kept as a `TsRef` and the value is NaN until the
#   boundary framework evaluates the series per time step (T6.3, `advance_time_series!`).
#   Without TS6 such a word is an error, as in MF6.
# - OPEN/CLOSE: a text list file as the first line of the block is a whole-block include
#   (handled by the block parser, `blocks.jl`). `OPEN/CLOSE fname (BINARY)` must be the
#   only line of the block; the file holds unformatted stream records: the cellid
#   (Int32 each), then the doubles and the aux values (Float64), no boundnames
#   (`read_from_binary`). Reading stops after MAXBOUND records or at the end of the file
#   (an incomplete last record is dropped). `OPEN/CLOSE fname SFAC f` is not supported by
#   MF6's IDM list reader (the line is read as a record and fails there): error here too.
#
# Differences from MF6 (documented, stricter or simpler):
# - All PERIOD blocks are read, also those after NPER (which MF6 never reads); the
#   period number is not compared with NPER here (the caller knows NPER).
# - A cellid in an inactive cell (IDOMAIN < 1) is MF6 error "Cell is outside active grid
#   domain" (`cellid_to_nlist`); it needs IDOMAIN, so the loader (T1.4) checks it.
#
# Representation (host only, plain arrays, for `BoundarySpec{Kind}` of T1.4): one
# `StressPeriods{D}` per package, holding the block numbers and one `D` per block;
# `period_data(sp, kper)` gives the data that apply in stress period `kper`. List data are
# `PeriodList`s (user node numbers as Int32, `bound[ncol, nbound]`, `auxvar[naux,
# nbound]`, boundnames, time-series references) and READASARRAYS data `ArrayPeriod`s
# (the full array state after each block).

"""
    DNODATA

MF6's "no data" value (`ConstantsModule.DNODATA = 3.0e30`), stored for an optional double
missing from the last column of a list record.
"""
const DNODATA = 3.0e30

const LENBOUNDNAME = 40  # ConstantsModule
const LENAUXNAME = 16

# ---------------------------------------------------------------------------------------
# Cell ids and user node numbers
# ---------------------------------------------------------------------------------------

"""
    ncelldim(g::GridDims) -> Int

The number of integers in a cellid: 3 (`lay row col`) for DIS, 2 (`lay cell2d`) for
DISV, 1 (`node`) for DISU.
"""
ncelldim(g::GridDims) = length(g.mshape)

"""
    user_node(g::GridDims, cellid) -> Int

The user node number of `cellid` (a tuple or vector of `ncelldim(g)` integers), as MF6's
`get_node`: `(k-1)*nrow*ncol + (i-1)*ncol + j` for DIS, `(k-1)*ncpl + icpl` for DISV, the
node itself for DISU. The cellid is not checked (`valid_cellid`).
"""
function user_node(g::GridDims, c)
    m = g.mshape
    length(m) == 1 && return Int(c[1])
    length(m) == 2 && return (Int(c[1]) - 1) * m[2] + Int(c[2])
    return ((Int(c[1]) - 1) * m[2] + (Int(c[2]) - 1)) * m[3] + Int(c[3])
end

"""
    cellid(g::GridDims, node) -> NTuple

The cellid of user node `node` (inverse of `user_node`): `(k, i, j)` for DIS,
`(k, icpl)` for DISV, `(node,)` for DISU.
"""
function cellid(g::GridDims, node::Integer)
    m = g.mshape
    length(m) == 1 && return (Int(node),)
    if length(m) == 2
        k, c = divrem(Int(node) - 1, m[2])
        return (k + 1, c + 1)
    end
    k, r = divrem(Int(node) - 1, m[2] * m[3])
    i, j = divrem(r, m[3])
    return (k + 1, i + 1, j + 1)
end

"""
    valid_cellid(g::GridDims, cellid) -> Bool

Whether every index of `cellid` is within the model shape (`check_cellid`).
"""
function valid_cellid(g::GridDims, c)
    m = g.mshape
    for d in eachindex(m)
        1 <= c[d] <= m[d] || return false
    end
    return true
end

_shape_string(v) = "(" * join(v, ",") * ")"

# ---------------------------------------------------------------------------------------
# List schema
# ---------------------------------------------------------------------------------------

"""
    ListColumn

One column of a list record, from the DFN record of the PERIOD block (or of another
list block, such as the GWF-GWF EXCHANGEDATA).

- `name`: the DFN name (`"cellid"`, `"head"`, `"pxdp"`, `"aux"`, `"boundname"`,
  `"cellidm1"`, `"ihc"`).
- `kind`: `:cellid` (integers converted to a user node of `grid`), `:int` (integers,
  rows of the integer columns), `:double` (rows of `bound`), `:aux` (rows of `auxvar`) or
  `:boundname`.
- `width`: the number of values (`ncelldim`, 1, `nseg-1`, `naux`).
- `first`: the first row it fills: of `bound` (`:double`), `auxvar` (`:aux`), the
  integer columns (`:int`), or the index of the cellid column (`:cellid`, 1 for the
  first one); 0 for `:boundname`.
- `time_series`: the DFN allows a time-series name instead of a number.
- `optional_last`: a scalar optional double in the last column that cannot hold a
  time series: a missing value (or a word that is not a number) is `DNODATA`
  (`StructArray.read_param`, `TryGetDouble`).
- `grid`: the grid of a `:cellid` column (exchanges connect two grids); `nothing`
  otherwise.
"""
struct ListColumn
    name::String
    kind::Symbol
    width::Int
    first::Int
    time_series::Bool
    optional_last::Bool
    grid::Union{Nothing, GridDims}
end

ListColumn(name, kind, width, first, time_series, optional_last) =
    ListColumn(name, kind, width, first, time_series, optional_last, nothing)

"""
    ListSchema

How the records of a list package are read: the columns in scope (`columns`, in input
order), the grid (`grid`), MAXBOUND (`maxbound`, NEXG for exchanges), the auxiliary
variable names (`auxnames`, upper case), whether boundnames are read (`boundnames`),
whether time-series names are allowed (`ts_active`: TS6 FILEIN in OPTIONS), the number
of rows of `bound` (`nbound_rows`), of the integer columns (`nint_rows`) and the number
of cellid columns (`ncellid`). Build one with `list_schema`.
"""
struct ListSchema
    package::String
    grid::GridDims
    columns::Vector{ListColumn}
    maxbound::Int
    auxnames::Vector{String}
    boundnames::Bool
    ts_active::Bool
    nbound_rows::Int
    nint_rows::Int
    ncellid::Int
end

function Base.show(io::IO, s::ListSchema)
    cols = join((c.width == 1 ? c.name : "$(c.name)($(c.width))" for c in s.columns), " ")
    return print(io, "ListSchema(", s.package, ": ", cols, ", maxbound = ", s.maxbound, ")")
end

"""
    list_column(s::ListSchema, name) -> Union{ListColumn, Nothing}

The column `name` (DFN name, case-insensitive) of `s`, or `nothing` when it is not in
scope. For a `:double` column, `bound[c.first:c.first+c.width-1, :]` holds its values.
"""
function list_column(s::ListSchema, name::AbstractString)
    n = lowercase(name)
    i = findfirst(c -> c.name == n, s.columns)
    return isnothing(i) ? nothing : s.columns[i]
end

"""
    PackageOptions

The OPTIONS and DIMENSIONS entries that decide how period data are read (not a full
options parser; the loader of T1.4 reads the others): `auxnames` (AUXILIARY, upper
case), `boundnames`, `ts_files` (TS6 FILEIN), `tas_files` (TAS6 FILEIN),
`surf_rate_specified` (EVT), `maxbound` and `nseg` (DIMENSIONS; 0 when absent).
"""
Base.@kwdef struct PackageOptions
    auxnames::Vector{String} = String[]
    boundnames::Bool = false
    ts_files::Vector{String} = String[]
    tas_files::Vector{String} = String[]
    surf_rate_specified::Bool = false
    maxbound::Int = 0
    nseg::Int = 0
end

"""
    package_options(f::InputFile) -> PackageOptions

Read the entries of `PackageOptions` from the OPTIONS and DIMENSIONS blocks of `f`.
AUXILIARY takes all words after it as names (`urdaux`); given on two lines it is an
error, as in MF6. Names longer than 16 characters are an error.
"""
function package_options(f::InputFile)
    aux = String[]
    auxline = nothing
    bnames = false
    ts = String[]
    tas = String[]
    srs = false
    b = find_block(f, "OPTIONS")
    if !isnothing(b)
        for l in b.lines
            k = keyword(l, 1)
            if k == "AUXILIARY" || k == "AUX"
                isnothing(auxline) || input_error(l, "auxiliary variables already specified; auxiliary variables must be specified on one line in the options block")
                auxline = l
                for t in Iterators.drop(eachtoken(l), 1)
                    ncodeunits(t) <= LENAUXNAME || input_error(l, "auxiliary variable name \"$t\" is longer than $LENAUXNAME characters")
                    push!(aux, uppercase(t))
                end
            elseif k == "BOUNDNAMES"
                bnames = true
            elseif (k == "TS6" || k == "TAS6") && keyword(l, 2) == "FILEIN"
                fname = token(l, 3)
                isnothing(fname) && input_error(l, "missing file name after $k FILEIN")
                push!(k == "TS6" ? ts : tas, String(fname))
            elseif k == "SURF_RATE_SPECIFIED"
                srs = true
            end
        end
    end
    maxbound = 0
    nseg = 0
    d = find_block(f, "DIMENSIONS")
    if !isnothing(d)
        for l in d.lines
            k = keyword(l, 1)
            k == "MAXBOUND" && (maxbound = get_int(l, 2; name = "MAXBOUND"))
            k == "NSEG" && (nseg = get_int(l, 2; name = "NSEG"))
        end
    end
    return PackageOptions(; auxnames = aux, boundnames = bnames, ts_files = ts, tas_files = tas, surf_rate_specified = srs, maxbound, nseg)
end

# The record (recarray) variable of the PERIOD block of `dfn`.
function _period_recarray(dfn::DfnDefinition)
    pb = dfn_block(dfn, "PERIOD")
    isnothing(pb) && throw(ArgumentError("$(dfn.name) has no PERIOD block"))
    i = findfirst(v -> v.type === :recarray, pb.variables)
    isnothing(i) && throw(ArgumentError("the PERIOD block of $(dfn.name) is not list input"))
    return pb, pb.variables[i]
end

"""
    list_schema(dfn::DfnDefinition, g::GridDims, opts::PackageOptions) -> ListSchema
    list_schema(f::InputFile, g::GridDims) -> ListSchema

The list schema of the PERIOD block of a stress package (`gwf-chd`, `gwf-riv`, ...): the
DFN record members in scope for these options (rules in the header of `lists.jl`), with
a cellid of `ncelldim(g)` integers. MAXBOUND must be positive (`source_dimensions`).
"""
function list_schema(dfn::DfnDefinition, g::GridDims, o::PackageOptions)
    pb, rec = _period_recarray(dfn)
    o.maxbound > 0 || throw(MF6InputError(dfn.name, 0, "MAXBOUND must be an integer greater than zero"))
    cols = ListColumn[]
    nb = 0
    na = 0
    nc = 0
    for m in rec.members
        v = dfn_variable(pb, m)
        isnothing(v) && throw(ArgumentError("$(dfn.name): record member $m is not a variable"))
        if m in ("cellid", "cellid1", "cellid2")
            # T6.11: HFB6 records have two cellids (`cellid1`, `cellid2`) on the same grid.
            nc += 1
            push!(cols, ListColumn(m, :cellid, ncelldim(g), nc, false, false, g))
        elseif m == "boundname"
            o.boundnames && push!(cols, ListColumn(m, :boundname, 1, 0, false, false))
        elseif v.shape == "(naux)"
            isempty(o.auxnames) && continue
            push!(cols, ListColumn(m, :aux, length(o.auxnames), na + 1, v.time_series, false))
            na += length(o.auxnames)
        elseif v.type === :double
            # LoadContext.in_scope: required always; EVT pxdp/petm with NSEG > 1, petm0
            # with SURF_RATE_SPECIFIED.
            width = 1
            if v.shape == "(nseg-1)"
                o.nseg > 1 || continue
                width = o.nseg - 1
            elseif v.optional
                (m == "petm0" && o.surf_rate_specified) || continue
            elseif v.shape != ""
                throw(ArgumentError("$(dfn.name): unsupported shape $(v.shape) of list column $m"))
            end
            push!(cols, ListColumn(m, :double, width, nb + 1, v.time_series, false))
            nb += width
        else
            throw(ArgumentError("$(dfn.name): list column $m of type $(v.type) is not supported"))
        end
    end
    ts_active = !isempty(o.ts_files)
    _mark_optional_last!(cols, pb, ts_active)
    return ListSchema(dfn.name, g, cols, o.maxbound, copy(o.auxnames), o.boundnames, ts_active, nb, 0, max(nc, 1))
end

# StructArray.read_param: the last column, when it is an optional scalar double (MTYPE_DBL,
# no shape), is read with TryGetDouble (DNODATA when missing), unless it may hold a time
# series and TS6 is active: that branch is tested first and always reads a word
# (`read_token`). Vector columns (`pxdp(nseg-1)`) always use GetDouble.
function _mark_optional_last!(cols, db::DfnBlock, ts_active::Bool)
    last = cols[end]
    last.kind === :double || return cols
    v = dfn_variable(db, last.name)
    if v.optional && v.shape == "" && !(v.time_series && ts_active)
        cols[end] = ListColumn(last.name, last.kind, last.width, last.first, last.time_series, true, nothing)
    end
    return cols
end

function list_schema(f::InputFile, g::GridDims)
    isnothing(f.dfn) && throw(ArgumentError("list_schema: $(f.path) was read without a DFN definition"))
    o = package_options(f)
    try
        return list_schema(f.dfn, g, o)
    catch e
        e isa MF6InputError && throw(MF6InputError(f.path, e.line, e.msg))
        rethrow()
    end
end

# ---------------------------------------------------------------------------------------
# List data
# ---------------------------------------------------------------------------------------

"""
    TsRef(row, col, aux, name)

A time-series name in list data (or an array of READASARRAYS input): record `row`, row
`col` of `bound` (or of `auxvar` when `aux`), time series `name` (upper case, as MF6
compares them). The value in `bound`/`auxvar` is NaN; the boundary framework writes the
series value of each time step into its working copy of the block (T6.3).
"""
struct TsRef
    row::Int32
    col::Int32
    aux::Bool
    name::Symbol
end

"""
    PeriodList

The list read from one PERIOD block (`nbound` records):

- `nodeu::Vector{Int32}`: user node number of each record's cellid (`user_node`; the
  cellid itself is `cellid(grid, nodeu[i])`). Reduced (active) node numbers depend on
  IDOMAIN and are left to the loader.
- `bound::Matrix{Float64}` `(nbound_rows, nbound)`: the double columns in input order
  (`list_column(schema, name).first` gives the row of a column).
- `auxvar::Matrix{Float64}` `(naux, nbound)`.
- `boundname::Vector{String}`: upper case, `""` when missing; empty without BOUNDNAMES.
- `ts::Vector{TsRef}`: the time-series names, in reading order.
"""
struct PeriodList
    nodeu::Vector{Int32}
    bound::Matrix{Float64}
    auxvar::Matrix{Float64}
    boundname::Vector{String}
    ts::Vector{TsRef}
end

nbound(p::PeriodList) = length(p.nodeu)

Base.show(io::IO, p::PeriodList) =
    print(io, "PeriodList(", length(p.nodeu), " records, ", size(p.bound, 1), " columns, ", size(p.auxvar, 1), " aux, ", length(p.ts), " time-series references)")

# The columns of `n` records while they are read (one row of `nodes` per cellid column).
struct _Records
    nodes::Matrix{Int32}
    ints::Matrix{Int32}
    bound::Matrix{Float64}
    auxvar::Matrix{Float64}
    boundname::Vector{String}
    ts::Vector{TsRef}
end

_records(s::ListSchema, n::Int) = _Records(
    Matrix{Int32}(undef, s.ncellid, n), Matrix{Int32}(undef, s.nint_rows, n),
    Matrix{Float64}(undef, s.nbound_rows, n), Matrix{Float64}(undef, length(s.auxnames), n),
    s.boundnames ? fill("", n) : String[], TsRef[],
)

# One cellid column: `vec` of a 1×n matrix shares its memory. With two (HFB6, T6.11),
# `nodeu` holds the cellids of each record in turn (`reshape(nodeu, 2, :)`).
PeriodList(r::_Records) = PeriodList(vec(r.nodes), r.bound, r.auxvar, r.boundname, r.ts)

@noinline _missing_value(l::InputLine, c::ListColumn, irow) =
    input_error(l, "list record $irow: missing $(c.name) ($(c.kind in (:cellid, :int) ? "integer" : "double precision"))")

@noinline function _bad_double(l::InputLine, c::ListColumn, t, irow, s::ListSchema)
    hint = c.time_series && !s.ts_active ? " (a time-series name needs TS6 FILEIN in the OPTIONS block)" : ""
    return input_error(l, "list record $irow: could not read $(c.name) as a double precision real from \"$t\"$hint")
end

@noinline _bad_int(l::InputLine, c::ListColumn, t, irow) =
    input_error(l, "list record $irow: could not read $(c.name) as an integer from \"$t\"")

@noinline _cellid_error(l::InputLine, c::ListColumn, irow, cid) = input_error(
    l, "List entry $irow contains $(c.name) $(_shape_string(cid)) but this cellid is invalid for model with shape $(_shape_string(c.grid.mshape))",
)

# A boundname as MF6 stores it (upper case, at most LENBOUNDNAME characters). `names`
# maps the words seen so far to their String, so a name repeated on many records is one
# String (lookup with the SubString does not allocate).
function _boundname!(names::Dict{String, String}, t::SubString{String})
    v = get(names, t, nothing)
    isnothing(v) || return v
    u = uppercase(t)
    length(u) > LENBOUNDNAME && (u = first(u, LENBOUNDNAME))
    names[String(t)] = u
    return u
end

# Read record `irow` from line `l` into `p` (StructArray.read_param per column).
function _read_record!(p::_Records, s::ListSchema, l::InputLine, irow::Int, names::Dict{String, String})
    toks = eachtoken(l)
    st = iterate(toks)
    for c in s.columns
        if c.kind === :cellid
            c1 = c2 = c3 = 0
            for d in 1:c.width
                isnothing(st) && _missing_value(l, c, irow)
                t = st[1]
                v = parse_int(t)
                isnothing(v) && _bad_int(l, c, t, irow)
                d == 1 ? (c1 = v) : d == 2 ? (c2 = v) : (c3 = v)
                st = iterate(toks, st[2])
            end
            cid = (c1, c2, c3)
            g = c.grid::GridDims
            valid_cellid(g, cid) || _cellid_error(l, c, irow, cid[1:c.width])
            p.nodes[c.first, irow] = user_node(g, cid)
        elseif c.kind === :int
            for d in 0:(c.width - 1)
                isnothing(st) && _missing_value(l, c, irow)
                t = st[1]
                v = parse_int(t)
                (isnothing(v) || v < typemin(Int32) || v > typemax(Int32)) && _bad_int(l, c, t, irow)
                p.ints[c.first + d, irow] = v
                st = iterate(toks, st[2])
            end
        elseif c.kind === :double || c.kind === :aux
            M = c.kind === :double ? p.bound : p.auxvar
            for d in 0:(c.width - 1)
                if isnothing(st)
                    c.optional_last || _missing_value(l, c, irow)
                    M[c.first + d, irow] = DNODATA
                    continue
                end
                t = st[1]
                v = parse_double(t)
                if isnothing(v)
                    # read_param tests the time-series branch first.
                    if c.time_series && s.ts_active
                        push!(p.ts, TsRef(irow, c.first + d, c.kind === :aux, Symbol(uppercase(t))))
                        v = NaN
                    elseif c.optional_last
                        v = DNODATA  # TryGetDouble: not a number (a comment) is no data
                    else
                        _bad_double(l, c, t, irow, s)
                    end
                end
                M[c.first + d, irow] = v
                st = iterate(toks, st[2])
            end
        else  # :boundname
            if !isnothing(st)
                p.boundname[irow] = _boundname!(names, st[1])
                st = iterate(toks, st[2])
            end
        end
    end
    return nothing
end

"""
    read_list_block(b::InputBlock, s::ListSchema; simdir = ".") -> PeriodList

Read the list of one PERIOD block (rules in the header of `lists.jl`): one record per
line, or a binary list file (`OPEN/CLOSE fname (BINARY)`, relative to `simdir`).
"""
function read_list_block(b::InputBlock, s::ListSchema; simdir::AbstractString = ".")
    lines = b.lines
    if !isempty(lines) && _keyword_is(lines[1], 1, "OPEN/CLOSE")
        l = lines[1]
        # LoadMf6File.read_control_record: (BINARY) must be the word after the file name.
        if _keyword_is(l, 3, "(BINARY)")
            length(lines) == 1 || input_error(lines[2], "LOOKING FOR 'END $(b.name)' after the binary list file")
            return _read_binary_list(l, s, simdir)
        elseif any(t -> uppercase(t) == "SFAC", eachtoken(l))
            input_error(l, "OPEN/CLOSE with SFAC is not supported for list input (MF6 6.8 reads this line as a list record)")
        end
    end
    n = length(lines)
    n > s.maxbound && input_error(lines[s.maxbound + 1], "Input error: line count exceeds input dimension. Expected rows=$(s.maxbound)")
    return PeriodList(_read_records(lines, s))
end

function _read_records(lines, s::ListSchema)
    n = length(lines)
    p = _records(s, n)
    names = Dict{String, String}()
    for i in 1:n
        _read_record!(p, s, lines[i], i, names)
    end
    return p
end

"""
    write_binary_list(path, s::ListSchema, cellids, bound, auxvar = zeros(0, n))

Write a binary list file for schema `s` (`OPEN/CLOSE path (BINARY)`): per record the
cellid (`cellids[:, i]`, Int32), then `bound[:, i]` and `auxvar[:, i]` (Float64),
little-endian unformatted stream (IO guide, "Description of Binary List Input Files").
"""
function write_binary_list(path::AbstractString, s::ListSchema, cellids::AbstractMatrix, bound::AbstractMatrix, auxvar::AbstractMatrix = zeros(0, size(cellids, 2)))
    n = size(cellids, 2)
    size(cellids, 1) == ncelldim(s.grid) || throw(ArgumentError("cellids must have $(ncelldim(s.grid)) rows"))
    size(bound) == (s.nbound_rows, n) || throw(ArgumentError("bound must be $(s.nbound_rows) × $n"))
    size(auxvar) == (length(s.auxnames), n) || throw(ArgumentError("auxvar must be $(length(s.auxnames)) × $n"))
    open(path, "w") do io
        for i in 1:n
            for c in s.columns
                if c.kind === :cellid
                    foreach(d -> write(io, htol(Int32(cellids[d, i]))), 1:c.width)
                elseif c.kind === :double
                    foreach(d -> write(io, htol(Float64(bound[c.first + d, i]))), 0:(c.width - 1))
                elseif c.kind === :aux
                    foreach(d -> write(io, htol(Float64(auxvar[c.first + d, i]))), 0:(c.width - 1))
                end
            end
        end
    end
    return path
end

# StructArray.read_from_binary: records until MAXBOUND or the end of the file.
function _read_binary_list(l::InputLine, s::ListSchema, simdir)
    s.boundnames && input_error(l, "List style binary inputs not supported for text columns, tag=BOUNDNAME (remove BOUNDNAMES from the OPTIONS block)")
    fname = token(l, 2)
    isnothing(fname) && input_error(l, "missing file name after OPEN/CLOSE")
    path = _resolve_path(fname, simdir)
    isfile(path) || input_error(l, "Specified OPEN/CLOSE file $fname does not exist ($path)")
    nd = ncelldim(s.grid)
    recsize = 4 * nd * s.ncellid + 8 * (s.nbound_rows + length(s.auxnames))
    n = min(s.maxbound, div(filesize(path), recsize))
    p = _records(s, n)
    cid = zeros(Int, nd)
    open(path, "r") do io
        for i in 1:n
            for c in s.columns
                if c.kind === :cellid
                    for d in 1:nd
                        cid[d] = ltoh(read(io, Int32))
                    end
                    valid_cellid(s.grid, cid) || _cellid_error(l, c, i, cid)
                    p.nodes[c.first, i] = user_node(s.grid, cid)
                elseif c.kind === :double
                    for d in 0:(c.width - 1)
                        p.bound[c.first + d, i] = ltoh(read(io, Float64))
                    end
                elseif c.kind === :aux
                    for d in 0:(c.width - 1)
                        p.auxvar[c.first + d, i] = ltoh(read(io, Float64))
                    end
                end
            end
        end
    end
    return PeriodList(p)
end

# ---------------------------------------------------------------------------------------
# Stress periods
# ---------------------------------------------------------------------------------------

"""
    StressPeriods{D}

The period data of one package: `iper[b]` is the number of the `b`-th PERIOD block
(increasing) and `data[b]` what it holds. Block `b` applies from stress period `iper[b]`
until the period before `iper[b+1]`, or to the end of the simulation. Before
`iper[1]` the package has no period data. Use `period_data` and `period_block`.
"""
struct StressPeriods{D}
    iper::Vector{Int}
    data::Vector{D}
end

Base.show(io::IO, sp::StressPeriods{D}) where {D} = print(io, "StressPeriods{", D, "}(periods ", sp.iper, ")")

"""
    period_block(sp::StressPeriods, kper) -> Int

The index of the block that applies in stress period `kper` (the last block with
`iper <= kper`), or 0 when no block applies yet.
"""
period_block(sp::StressPeriods, kper::Integer) = searchsortedlast(sp.iper, kper)

"""
    period_data(sp::StressPeriods{D}, kper) -> Union{D, Nothing}

The data that apply in stress period `kper` (data persist until the next PERIOD block),
or `nothing` before the first block. For list data, an empty PERIOD block gives an empty
`PeriodList` (no stresses).
"""
function period_data(sp::StressPeriods, kper::Integer)
    b = period_block(sp, kper)
    return b == 0 ? nothing : sp.data[b]
end

"""
    read_list_periods(f::InputFile, g::GridDims; simdir = dirname(f.path))
        -> (ListSchema, StressPeriods{PeriodList})

Read every PERIOD block of the list stress package `f` (read with its DFN definition,
e.g. by `read_input_file`) on grid `g`.
"""
function read_list_periods(f::InputFile, g::GridDims; simdir::AbstractString = dirname(f.path))
    s = list_schema(f, g)
    blocks = find_blocks(f, "PERIOD")
    iper = [block_number(b) for b in blocks]
    data = [read_list_block(b, s; simdir) for b in blocks]
    return s, StressPeriods(iper, data)
end

# ---------------------------------------------------------------------------------------
# READASARRAYS (RCHA, EVTA)
# ---------------------------------------------------------------------------------------

"""
    ArrayPeriod

The state of a READASARRAYS package (RCHA, EVTA) after one PERIOD block, as MF6 holds it
in memory (arrays not given in the block keep their values; arrays shared with the
previous state are not copied):

- `layer::Vector{Int32}` `(ncpl)`: the layer of each cell column (IRCH/IEVT; 1 until
  given).
- `values::Dict{String, Vector{Float64}}`: every double array of the DFN PERIOD block
  except `aux` (`"recharge"`; `"surface"`, `"rate"`, `"depth"`), length `ncpl`, zero
  until given.
- `auxvar::Matrix{Float64}` `(naux, ncpl)`.
- `tas::Dict{String, Symbol}`: `TIMEARRAYSERIES` references given in this block, from
  the variable (`"recharge"`, or an auxiliary name in upper case) to the series name
  (upper case). MF6 links them for the periods of this block only, and the next PERIOD
  block first sets every array linked to a series to zero (`rp` → `reset` →
  TimeArraySeriesManager.f90 `Reset`), so a later block that does not give the variable
  has zeros there. In this block the values hold NaN until the boundary framework
  evaluates the series per time step (T6.3, `advance_time_series!`).
- `given::Vector{String}`: the variables given in this block (DFN names, or auxiliary
  names), in input order.
"""
struct ArrayPeriod
    layer::Vector{Int32}
    values::Dict{String, Vector{Float64}}
    auxvar::Matrix{Float64}
    tas::Dict{String, Symbol}
    given::Vector{String}
end

Base.show(io::IO, a::ArrayPeriod) = print(io, "ArrayPeriod(", length(a.layer), " cells, given: ", join(a.given, ", "), ")")

"""
    array_nodes(a::ArrayPeriod, g::GridDims) -> Vector{Int32}

The user node of each cell column of a READASARRAYS package: layer `a.layer[j]` of
column `j` (`nlarray_to_nodelist`; DIS columns in row-major order `(i-1)*ncol + j`).
"""
function array_nodes(a::ArrayPeriod, g::GridDims)
    ncpl = length(a.layer)
    ncpl == prod(g.mshape[2:end]) || throw(ArgumentError("array_nodes: $ncpl cell columns, but the grid has $(prod(g.mshape[2:end]))"))
    return Int32[(a.layer[j] - 1) * ncpl + j for j in 1:ncpl]
end

"""
    read_array_periods(f::InputFile, g::GridDims; simdir = dirname(f.path))
        -> StressPeriods{ArrayPeriod}

Read the PERIOD blocks of a READASARRAYS package (`gwf-rcha`, `gwf-evta`;
`Mf6FileLayerArray.f90`). Each line names a variable (a DFN PERIOD variable such as
`IRCH` or `RECHARGE`, or an auxiliary variable name), optionally followed by
`TIMEARRAYSERIES name` (needs TAS6 FILEIN in OPTIONS). Otherwise the next line is the
array control record and the array has NCPL values, read as one 1-D array as MF6 does
(`read_dbl1d`, `read_int1d`): words after the name such as `LAYERED` are ignored. A layer
number outside 1..NLAY is an error. DISU grids are an error ("READASARRAYS option is not
compatible with selected discretization type").
"""
function read_array_periods(f::InputFile, g::GridDims; simdir::AbstractString = dirname(f.path))
    dfn = f.dfn
    isnothing(dfn) && throw(ArgumentError("read_array_periods: $(f.path) was read without a DFN definition"))
    pb = dfn_block(dfn, "PERIOD")
    isnothing(pb) && throw(ArgumentError("$(dfn.name) has no PERIOD block"))
    length(g.mshape) == 1 && throw(MF6InputError(f.path, 0, "READASARRAYS option is not compatible with selected discretization type"))
    o = package_options(f)
    nlay = g.mshape[1]
    ncpl = prod(g.mshape[2:end])
    naux = length(o.auxnames)
    layername = "i" * lowercase(dfn.subcomponent[1:min(3, end)])  # irch, ievt
    # Initial state (LoadContext.allocate_param: zero; default_nodelist: layer 1).
    layer = fill(Int32(1), ncpl)
    values = Dict{String, Vector{Float64}}()
    for v in pb.variables
        (v.type === :double && v.name != "aux") && (values[v.name] = zeros(ncpl))
    end
    auxvar = zeros(naux, ncpl)
    blocks = find_blocks(f, "PERIOD")
    iper = Int[]
    data = ArrayPeriod[]
    prevtas = Dict{String, Symbol}()
    for b in blocks
        push!(iper, block_number(b))
        values = copy(values)  # new Dict; arrays are replaced (not changed) when given
        tas = Dict{String, Symbol}()
        given = String[]
        auxcopied = false
        # TimeArraySeriesManager.f90 Reset: the arrays the previous block linked to a time-
        # array series are set to zero before this block is read.
        for name in keys(prevtas)
            iaux = findfirst(==(name), o.auxnames)
            if isnothing(iaux)
                values[name] = zeros(ncpl)
            else
                auxcopied || (auxvar = copy(auxvar); auxcopied = true)
                auxvar[iaux, :] .= 0.0
            end
        end
        lines = b.lines
        k = 1
        while k <= length(lines)
            l = lines[k]
            tag = keyword(l, 1)
            iaux = findfirst(==(tag), o.auxnames)
            v = isnothing(iaux) ? dfn_variable(pb, tag) : nothing
            if isnothing(iaux) && (isnothing(v) || v.name == "aux" || v.reader !== :readarray)
                input_error(l, "\"$tag\" is not a PERIOD array of a $(dfn.name) file (nor an auxiliary variable name)")
            end
            name = isnothing(iaux) ? v.name : tag
            push!(given, name)
            kw = keyword(l, 2)
            if kw == "TIMEARRAYSERIES"
                isempty(o.tas_files) && input_error(l, "$tag TIMEARRAYSERIES needs a time-array-series file (TAS6 FILEIN in the OPTIONS block)")
                (isnothing(iaux) && !v.time_series) && input_error(l, "$tag cannot be a time-array series")
                tname = token(l, 3)
                isnothing(tname) && input_error(l, "missing time-array-series name after $tag TIMEARRAYSERIES")
                tas[name] = Symbol(uppercase(tname))
                # The values come from the series (T6.3); NaN until it is evaluated.
                if isnothing(iaux)
                    values[name] = fill(NaN, ncpl)
                else
                    auxcopied || (auxvar = copy(auxvar); auxcopied = true)
                    auxvar[iaux, :] .= NaN
                end
                k += 1
                continue
            end
            kw == "NETCDF" && input_error(l, "NETCDF array input is not supported")
            k + 1 <= length(lines) || input_error(l, "missing control record for array $tag")
            if !isnothing(iaux)
                a = zeros(ncpl)
                k = read_array!(a, lines, k + 1; name = tag, simdir)
                auxcopied || (auxvar = copy(auxvar); auxcopied = true)
                auxvar[iaux, :] .= a
            elseif v.type === :integer
                v.name == layername || input_error(l, "integer array $tag is not supported")
                a = zeros(Int32, ncpl)
                k = read_array!(a, lines, k + 1; name = tag, simdir)
                for j in 1:ncpl
                    1 <= a[j] <= nlay || input_error(l, "Invalid layer number: $(a[j]) (array $tag, cell $j; NLAY = $nlay)")
                end
                layer = a
            else
                a = zeros(ncpl)
                k = read_array!(a, lines, k + 1; name = tag, simdir)
                values[v.name] = a
            end
        end
        push!(data, ArrayPeriod(layer, values, auxvar, tas, given))
        prevtas = tas
    end
    return StressPeriods(iper, data)
end

# ---------------------------------------------------------------------------------------
# STO and OC period blocks
# ---------------------------------------------------------------------------------------

"""
    read_sto_periods(f::InputFile) -> StressPeriods{Symbol}

The STO PERIOD blocks: `:steady_state` or `:transient` per block (`sto_rp`). A block
holds one keyword (STEADY-STATE or TRANSIENT); more lines are an error. An empty block
keeps the previous setting, so it is left out. Before the first block MF6 uses
`:transient` (when the model has STO).
"""
function read_sto_periods(f::InputFile)
    iper = Int[]
    data = Symbol[]
    for b in find_blocks(f, "PERIOD")
        isempty(b.lines) && continue
        l = b.lines[1]
        k = keyword(l, 1)
        s = k == "STEADY-STATE" ? :steady_state : k == "TRANSIENT" ? :transient :
            input_error(l, "expected STEADY-STATE or TRANSIENT in the STO PERIOD block, found \"$k\"")
        length(b.lines) == 1 || input_error(b.lines[2], "LOOKING FOR 'END PERIOD'")
        push!(iper, block_number(b))
        push!(data, s)
    end
    return StressPeriods(iper, data)
end

"""
    OcRecord(action, rtype, setting, steps)

One OC PERIOD record, `SAVE HEAD ALL` or `PRINT BUDGET STEPS 1 5`: `action` is `:save`
or `:print`, `rtype` the record type in upper case (`"HEAD"`, `"BUDGET"`; checked by the
loader), `setting` one of `:all`, `:first`, `:last`, `:frequency`, `:steps`, and `steps`
the integers of FREQUENCY (one) or STEPS (TimeStepSelect.f90 `read`).
"""
struct OcRecord
    action::Symbol
    rtype::String
    setting::Symbol
    steps::Vector{Int}
end

Base.:(==)(a::OcRecord, b::OcRecord) =
    a.action === b.action && a.rtype == b.rtype && a.setting === b.setting && a.steps == b.steps
Base.hash(a::OcRecord, h::UInt) = hash((a.action, a.rtype, a.setting, a.steps), h)

function _oc_record(l::InputLine)
    a = keyword(l, 1)
    action = a == "SAVE" ? :save : a == "PRINT" ? :print : input_error(l, "Looking for PRINT or SAVE. Found: $a")
    rtype = keyword(l, 2)
    isempty(rtype) && input_error(l, "missing record type (HEAD, BUDGET) after $a")
    w = keyword(l, 3)
    steps = Int[]
    if w == "ALL"
        setting = :all
    elseif w == "FIRST"
        setting = :first
    elseif w == "LAST"
        setting = :last
    elseif w == "FREQUENCY"
        setting = :frequency
        push!(steps, get_int(l, 4; name = "FREQUENCY"))
    elseif w == "STEPS"
        setting = :steps
        # urword with ncode 2 and no error: stop at the first word that is not a
        # positive integer.
        for t in Iterators.drop(eachtoken(l), 3)
            v = parse_int(t)
            (isnothing(v) || v <= 0) && break
            push!(steps, v)
        end
    else
        input_error(l, "Looking for ALL, STEPS, FIRST, LAST, OR FREQUENCY. Found: $w")
    end
    return OcRecord(action, rtype, setting, steps)
end

"""
    read_oc_periods(f::InputFile) -> StressPeriods{Vector{OcRecord}}

The OC PERIOD blocks (`oc_rp`): the records of each block replace all settings of the
previous one, and an empty block turns output off (an empty vector).

Within a block the records are cumulative, not exclusive: consumers must OR all records
with the same `rtype` and `action` (`PrintSaveManager%set` adds to one time-step
selection: `SAVE HEAD FIRST` and `SAVE HEAD LAST` select both steps, STEPS lists are
appended, ALL/FIRST/LAST flags accumulate and a later FREQUENCY replaces an earlier one).
"""
function read_oc_periods(f::InputFile)
    blocks = find_blocks(f, "PERIOD")
    return StressPeriods([block_number(b) for b in blocks], [OcRecord[_oc_record(l) for l in b.lines] for b in blocks])
end

# ---------------------------------------------------------------------------------------
# Static lists (TDIS PERIODDATA, ...)
# ---------------------------------------------------------------------------------------

"""
    read_records(b::InputBlock, dfn::DfnDefinition; nrow = 0) -> Dict{String, Vector}

Read the record list of a static block (a DFN `recarray` of scalar integer, double and
string members, such as TDIS PERIODDATA `perlen nstp tsmult`) into one vector per member
(`Int` for integers, `Float64` for doubles, `String` for strings). Missing or malformed
values are errors; words after the last member are ignored. With `nrow > 0`, more lines
than `nrow` are an error ("line count exceeds input dimension") and fewer are an error
as well (MF6 would leave rows undefined).
"""
function read_records(b::InputBlock, dfn::DfnDefinition; nrow::Integer = 0)
    db = dfn_block(dfn, b.name)
    isnothing(db) && input_error(b, "block $(b.name) is not part of a $(dfn.name) file")
    i = findfirst(v -> v.type === :recarray, db.variables)
    isnothing(i) && input_error(b, "block $(b.name) of $(dfn.name) is not a list")
    vars = [dfn_variable(db, m) for m in db.variables[i].members]
    for v in vars
        # A cellid is ncelldim integers (one DFN member): a scalar reader would misread it.
        startswith(v.name, "cellid") && throw(
            ArgumentError("read_records: unsupported: cellid columns ($(v.name) of $(dfn.name) $(b.name)), use the list reader (list_schema, read_exchange_data)"),
        )
        (v.shape == "" && v.type in (:integer, :double, :string)) ||
            throw(ArgumentError("read_records: member $(v.name) of $(dfn.name) $(b.name) is not a scalar"))
    end
    n = length(b.lines)
    if nrow > 0
        n > nrow && input_error(b.lines[nrow + 1], "Input error: line count exceeds input dimension. Expected rows=$nrow")
        n < nrow && input_error(b, "found $n records in block $(b.name), expected $nrow")
    end
    cols = Any[v.type === :integer ? zeros(Int, n) : v.type === :double ? zeros(n) : fill("", n) for v in vars]
    for (r, l) in enumerate(b.lines)
        for (j, v) in enumerate(vars)
            if v.type === :integer
                cols[j][r] = get_int(l, j; name = v.name)
            elseif v.type === :double
                cols[j][r] = get_double(l, j; name = v.name)
            else
                t = token(l, j)
                isnothing(t) && input_error(l, "missing $(v.name)")
                cols[j][r] = v.preserve_case ? String(t) : uppercase(t)
            end
        end
    end
    return Dict{String, Vector}(v.name => c for (v, c) in zip(vars, cols))
end

"""
    read_tdis_perioddata(f::InputFile) -> (; perlen, nstp, tsmult)

The PERIODDATA of a TDIS file: NPER records of PERLEN, NSTP and TSMULT (checks of the
values are the loader's, `check_tdis_timing`).
"""
function read_tdis_perioddata(f::InputFile)
    d = find_block(f, "DIMENSIONS")
    nper = 1  # DFN default
    if !isnothing(d)
        for l in d.lines
            keyword(l, 1) == "NPER" && (nper = get_int(l, 2; name = "NPER"))
        end
    end
    b = find_block(f, "PERIODDATA")
    isnothing(b) && throw(MF6InputError(f.path, 0, "PERIODDATA block not found"))
    r = read_records(b, f.dfn; nrow = nper)
    return (; perlen = r["perlen"]::Vector{Float64}, nstp = r["nstp"]::Vector{Int}, tsmult = r["tsmult"]::Vector{Float64})
end

# ---------------------------------------------------------------------------------------
# GWF-GWF exchange data
# ---------------------------------------------------------------------------------------

"""
    ExchangeData

The EXCHANGEDATA of a GWF6-GWF6 exchange file (`nexg` connections), as
DisConnExchange.f90 `source_data` takes it from the input context:

- `nodem1`, `nodem2::Vector{Int32}`: user node numbers of `cellidm1` in model 1 and
  `cellidm2` in model 2 (reduced node numbers depend on IDOMAIN: the loader converts
  them and reports MF6's "Cell is outside active grid domain").
- `ihc::Vector{Int32}`, `cl1`, `cl2`, `hwva::Vector{Float64}`.
- `auxvar::Matrix{Float64}` `(naux, nexg)`, `auxnames` (upper case).
- `boundname::Vector{String}` (upper case; empty without BOUNDNAMES).
"""
struct ExchangeData
    nodem1::Vector{Int32}
    nodem2::Vector{Int32}
    ihc::Vector{Int32}
    cl1::Vector{Float64}
    cl2::Vector{Float64}
    hwva::Vector{Float64}
    auxvar::Matrix{Float64}
    auxnames::Vector{String}
    boundname::Vector{String}
end

Base.show(io::IO, e::ExchangeData) =
    print(io, "ExchangeData(", length(e.nodem1), " connections, aux ", e.auxnames, ")")

"""
    exchange_schema(dfn::DfnDefinition, g1::GridDims, g2::GridDims, opts::PackageOptions, nexg)
        -> ListSchema

The record schema of the EXCHANGEDATA block of a GWF-GWF exchange: `cellidm1` (cellid of
grid `g1`), `cellidm2` (of `g2`), `ihc` (integer), `cl1 cl2 hwva`, then `aux` with
AUXILIARY and `boundname` with BOUNDNAMES (`LoadContext.in_scope`). Time series are not
allowed in exchange data.
"""
function exchange_schema(dfn::DfnDefinition, g1::GridDims, g2::GridDims, o::PackageOptions, nexg::Integer)
    db = dfn_block(dfn, "EXCHANGEDATA")
    isnothing(db) && throw(ArgumentError("$(dfn.name) has no EXCHANGEDATA block"))
    rec = db.variables[findfirst(v -> v.type === :recarray, db.variables)]
    cols = ListColumn[]
    nb = ni = nc = na = 0
    for m in rec.members
        v = dfn_variable(db, m)
        if startswith(m, "cellid")
            g = nc == 0 ? g1 : g2
            nc += 1
            push!(cols, ListColumn(m, :cellid, ncelldim(g), nc, false, false, g))
        elseif m == "boundname"
            o.boundnames && push!(cols, ListColumn(m, :boundname, 1, 0, false, false))
        elseif v.shape == "(naux)"
            isempty(o.auxnames) && continue
            push!(cols, ListColumn(m, :aux, length(o.auxnames), na + 1, false, false))
            na += length(o.auxnames)
        elseif v.shape == "" && v.type === :integer
            push!(cols, ListColumn(m, :int, 1, ni + 1, false, false))
            ni += 1
        elseif v.shape == "" && v.type === :double
            v.optional && continue
            push!(cols, ListColumn(m, :double, 1, nb + 1, false, false))
            nb += 1
        else
            throw(ArgumentError("$(dfn.name): exchange column $m is not supported"))
        end
    end
    nc == 2 || throw(ArgumentError("$(dfn.name): expected two cellid columns"))
    _mark_optional_last!(cols, db, false)
    return ListSchema(dfn.name, g1, cols, nexg, copy(o.auxnames), o.boundnames, false, nb, ni, nc)
end

"""
    read_exchange_data(f::InputFile, g1::GridDims, g2::GridDims) -> ExchangeData

Read the EXCHANGEDATA of the GWF6-GWF6 exchange file `f` (DFN `exg-gwfgwf`), connecting
a model with grid `g1` (`cellidm1`) to one with grid `g2` (`cellidm2`; the same grid for
an exchange of a model with itself, such as `ex-gwf-spbc`). The records follow the list
rules of `lists.jl` (StructArray.read_param: GetInteger for the cellids and IHC,
GetDouble for CL1, CL2, HWVA and the aux values, GetString for the boundname). There
must be exactly NEXG records (StructArray `read_from_parser` errors on more; fewer would
leave cellids of zero, which MF6 rejects). A cellid outside its model's shape is an
error (MF6 rejects it when it converts it to a node number; the message here is that
of `check_cellid`).
"""
function read_exchange_data(f::InputFile, g1::GridDims, g2::GridDims)
    (isnothing(f.dfn) || f.dfn.name != "exg-gwfgwf") &&
        throw(ArgumentError("read_exchange_data: $(f.path) was not read as a GWF6-GWF6 exchange file"))
    o = package_options(f)
    nexg = 0
    d = find_block(f, "DIMENSIONS")
    if !isnothing(d)
        for l in d.lines
            keyword(l, 1) == "NEXG" && (nexg = get_int(l, 2; name = "NEXG"))
        end
    end
    b = find_block(f, "EXCHANGEDATA")
    isnothing(b) && throw(MF6InputError(f.path, 0, "EXCHANGEDATA block not found"))
    s = exchange_schema(f.dfn, g1, g2, o, nexg)
    n = length(b.lines)
    n > nexg && input_error(b.lines[nexg + 1], "Input error: line count exceeds input dimension. Expected rows=$nexg")
    n < nexg && input_error(b, "found $n exchange records, expected NEXG = $nexg")
    r = _read_records(b.lines, s)
    col(name) = r.bound[list_column(s, name).first, :]
    return ExchangeData(
        r.nodes[1, :], r.nodes[2, :], r.ints[list_column(s, "ihc").first, :],
        col("cl1"), col("cl2"), col("hwva"), r.auxvar, copy(s.auxnames), r.boundname,
    )
end

# ---------------------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------------------

const _LIST_PACKAGES = ("gwf-chd", "gwf-wel", "gwf-drn", "gwf-riv", "gwf-ghb", "gwf-rch", "gwf-evt")
const _ARRAY_PACKAGES = ("gwf-rcha", "gwf-evta")

"""
    read_period_data(f::InputFile, g::GridDims; simdir = dirname(f.path))

The period data of a GWF package with PERIOD blocks, by DFN definition:

- list stress packages (CHD, WEL, DRN, RIV, GHB, RCH, EVT): `(ListSchema,
  StressPeriods{PeriodList})` (`read_list_periods`);
- READASARRAYS packages (RCHA, EVTA): `StressPeriods{ArrayPeriod}`;
- STO: `StressPeriods{Symbol}`; OC: `StressPeriods{Vector{OcRecord}}`.

Other packages (READARRAYGRID variants, advanced packages) are an error naming the
package.
"""
function read_period_data(f::InputFile, g::GridDims; simdir::AbstractString = dirname(f.path))
    isnothing(f.dfn) && throw(ArgumentError("read_period_data: $(f.path) was read without a DFN definition"))
    n = f.dfn.name
    n in _LIST_PACKAGES && return read_list_periods(f, g; simdir)
    n in _ARRAY_PACKAGES && return read_array_periods(f, g; simdir)
    n == "gwf-sto" && return read_sto_periods(f)
    n == "gwf-oc" && return read_oc_periods(f)
    throw(MF6InputError(f.path, 0, "unsupported: period data of package $(f.dfn.subcomponent) ($n)"))
end
