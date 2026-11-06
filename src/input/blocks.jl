# Block parser for MF6 input files (host only).
#
# Ports the block rules of src/Utilities/BlockParser.f90 (`uget_block`,
# `uget_any_block`, `GetNextLine`, `uterminate_block`):
# - A block is `BEGIN name [suffix ...]` … `END name`, compared in upper case. The words
#   after the name on the BEGIN line are the block's suffix (period number, solution-group
#   number, `FILEOUT name.csv` of an OBS `CONTINUOUS` block, ...); words after the name on
#   the END line are ignored (`END SOLUTIONGROUP 1`).
# - Inside a block, a BEGIN line, an END line with another name, or the end of the file
#   is an error (`GetNextLine`, `uterminate_block`).
# - Lines outside blocks are skipped, as MF6 does while it searches for the next BEGIN.
#   An END line outside a block is an error only when it names the block MF6 is
#   searching for at that point (`uget_block`); otherwise it is skipped with a warning.
# - Whole-block OPEN/CLOSE: when the first line of a block that supports it is
#   `OPEN/CLOSE fname`, the block's lines are read from `fname`; at its end, reading goes
#   on in the package file until `END name` (`GetNextLine`), so lines after the
#   OPEN/CLOSE line also belong to the block. An `END name` in the OPEN/CLOSE file ends
#   the block there, and the `END name` of the package file is then a stray END line.
#   The file name keeps its case and is relative to the simulation directory (MF6 opens
#   it relative to its working directory). A line with `(BINARY)` or `SFAC` after the
#   file name is not a whole-block include but list input, and stays in the block.
#
# With a DFN definition (src/input/dfn) `read_input_file` reads the blocks the way MF6
# does (`select_blocks`): MF6 searches for the blocks of a file one after the other in a
# fixed order (DFN order for files loaded through its input data model, IDM,
# LoadMf6File.f90 `load`/`parse_block`; the order of the reader routine for the other
# files, `_LEGACY_PLANS`). Each search stops at the next BEGIN line: if it names another
# block, a required block is an error and an optional block is taken as absent. Blocks
# MF6 therefore never reads (an optional block out of order, a repeated block, an
# unknown block where only optional blocks were searched, anything after the last block
# of a static package) are dropped with a warning naming file and line. After the static
# blocks, stress packages read PERIOD blocks until the end of the file
# (IdmMf6File.f90 `dynamic_read_ionper`): any other block there is an error, and period
# numbers must increase.

"""
    InputBlock

One block of an MF6 input file.

- `name`: the block name in upper case (`"PERIOD"`).
- `suffix`: the words after the name on the BEGIN line, original case (`["1"]` for
  `BEGIN PERIOD 1`, `["FILEOUT", "heads.csv"]` for `BEGIN CONTINUOUS FILEOUT heads.csv`).
- `lines`: the non-comment lines of the block (from the OPEN/CLOSE file, then from the
  package file, for a whole-block OPEN/CLOSE).
- `header`: the BEGIN line (for its file and line number).
- `open_close`: path of the whole-block OPEN/CLOSE file, or `nothing`.
- `stray_ends`: END lines outside any block between the previous block and this one
  (MF6 checks them against the block it searches for, see `select_blocks`).
"""
struct InputBlock
    name::String
    suffix::Vector{SubString{String}}
    lines::Vector{InputLine}
    header::InputLine
    open_close::Union{Nothing, String}
    stray_ends::Vector{InputLine}
end

function Base.show(io::IO, b::InputBlock)
    print(io, "InputBlock(\"", b.name, "\"")
    isempty(b.suffix) || print(io, " ", join(b.suffix, " "))
    print(io, ", ", length(b.lines), " lines, ", b.header.file, ":", b.header.line)
    isnothing(b.open_close) || print(io, ", OPEN/CLOSE ", b.open_close)
    return print(io, ")")
end

"""
    input_error(b::InputBlock, msg)

Throw an `MF6InputError` at the BEGIN line of block `b`.
"""
input_error(b::InputBlock, msg::AbstractString) =
    throw(MF6InputError(b.header.file, b.header.line, string(msg, " (block ", b.name, ")")))

"""
    InputFile

The blocks of one MF6 input file that MF6 reads, in file order. `dfn` is the DFN
definition they were read with, or `nothing` (then all blocks are kept).
"""
struct InputFile
    path::String
    dfn::Union{Nothing, DfnDefinition}
    blocks::Vector{InputBlock}
end

function Base.show(io::IO, f::InputFile)
    print(io, "InputFile(", f.path)
    isnothing(f.dfn) || print(io, ", ", f.dfn.name)
    return print(io, ", blocks = ", join((isempty(b.suffix) ? b.name : b.name * " " * join(b.suffix, " ") for b in f.blocks), ", "), ")")
end

"""
    find_blocks(f::InputFile, name) -> Vector{InputBlock}

All blocks named `name` (case-insensitive), in file order.
"""
find_blocks(f::InputFile, name::AbstractString) = filter(b -> b.name == uppercase(name), f.blocks)

"""
    find_block(f::InputFile, name) -> Union{InputBlock, Nothing}

The block named `name` (case-insensitive), or `nothing` if there is none. Use
`find_blocks` for blocks that may repeat (PERIOD).
"""
function find_block(f::InputFile, name::AbstractString)
    i = findfirst(b -> b.name == uppercase(name), f.blocks)
    return isnothing(i) ? nothing : f.blocks[i]
end

"""
    block_number(b::InputBlock) -> Int

The integer after the block name on the BEGIN line (`BEGIN PERIOD 3` → 3,
`BEGIN SOLUTIONGROUP 1` → 1); an `MF6InputError` when it is missing or not an integer.
"""
function block_number(b::InputBlock)
    isempty(b.suffix) && input_error(b.header, "missing number after BEGIN $(b.name)")
    return get_int(b.header, 3; name = "$(b.name) number")
end

# Whole-block OPEN/CLOSE support without a DFN: as with one, except GRIDDATA (IDM) and
# the TIME block of time-array series (TimeArraySeries.f90), whose first line is an
# array control record that may itself be OPEN/CLOSE.
_supports_open_close(::Nothing, name) = !(name == "GRIDDATA" || name == "TIME")
function _supports_open_close(d::DfnDefinition, name)
    b = dfn_block(d, name)
    return isnothing(b) ? _supports_open_close(nothing, name) : b.open_close
end

_resolve_path(fname, simdir) = isabspath(fname) ? String(fname) : normpath(joinpath(simdir, fname))

_eof_error(l::InputLine, name) =
    throw(MF6InputError(l.file, l.line, "unexpected end of file: BEGIN $name has no END $name"))

# Whether word `i` of `l` is `kw` (upper-case ASCII) ignoring ASCII case, as
# `keyword(l, i) == kw` but without allocating (called for every line of a block).
function _keyword_is(l::InputLine, i::Integer, kw::String)
    k = 0
    for t in eachtoken(l)
        k += 1
        k == i && return _ascii_upper_equal(t, kw)
    end
    return isempty(kw)
end

function _ascii_upper_equal(t::SubString{String}, kw::String)
    ncodeunits(t) == ncodeunits(kw) || return false
    @inbounds for j in 1:ncodeunits(kw)
        c = codeunit(t, j)
        UInt8('a') <= c <= UInt8('z') && (c -= 0x20)
        c == codeunit(kw, j) || return false
    end
    return true
end

# Handle the END line `bl` of block `name`; true when it ends the block.
function _check_end(bl::InputLine, name)
    if _keyword_is(bl, 1, "END")
        _keyword_is(bl, 2, name) ||
            input_error(bl, "END $(keyword(bl, 2)) detected without END $name; END $name must be used to end $name")
        return true
    elseif _keyword_is(bl, 1, "BEGIN")
        input_error(bl, "BEGIN detected before END $name; END $name must be used to end $name")
    end
    return false
end

# Split into blocks; also returns the stray END lines after the last block.
function _parse_blocks(path::AbstractString, simdir::AbstractString, dfn)
    lines = read_input_lines(path)
    blocks = InputBlock[]
    strays = InputLine[]
    n = length(lines)
    i = 1
    while i <= n
        l = lines[i]
        i += 1
        if _keyword_is(l, 1, "END")
            push!(strays, l)
            continue
        elseif !_keyword_is(l, 1, "BEGIN")
            continue  # outside a block: skipped, as in uget_block
        end
        name = keyword(l, 2)
        isempty(name) && input_error(l, "block name missing after BEGIN")
        suffix = collect(Iterators.drop(eachtoken(l), 2))
        body = InputLine[]
        oc = nothing
        ended = false
        if i <= n && _keyword_is(lines[i], 1, "OPEN/CLOSE") && _supports_open_close(dfn, name) &&
                !any(t -> uppercase(t) in ("(BINARY)", "SFAC"), Iterators.drop(eachtoken(lines[i]), 2))
            ocl = lines[i]
            fname = token(ocl, 2)
            isnothing(fname) && input_error(ocl, "missing file name after OPEN/CLOSE")
            oc = _resolve_path(fname, simdir)
            isfile(oc) || input_error(ocl, "OPEN/CLOSE file not found: $oc")
            i += 1
            for el in read_input_lines(oc)
                if _check_end(el, name)
                    ended = true
                    break
                end
                push!(body, el)
            end
        end
        # GetNextLine: read until END name (after an OPEN/CLOSE file, from the package file).
        if !ended
            first = i
            while true
                i <= n || _eof_error(l, name)
                i += 1
                _check_end(lines[i - 1], name) && break
            end
            append!(body, view(lines, first:(i - 2)))  # one allocation per block
        end
        push!(blocks, InputBlock(name, suffix, body, l, oc, strays))
        strays = InputLine[]
    end
    return blocks, strays
end

function _warn_strays(ends)
    for e in ends
        @warn "$(e.file):$(e.line): END $(keyword(e, 2)) outside a block is ignored"
    end
    return nothing
end

"""
    parse_blocks(path; simdir = dirname(path), dfn = nothing) -> Vector{InputBlock}

Split the file `path` into blocks (the rules are in the header of `blocks.jl`), keeping
every block. Whole-block OPEN/CLOSE files are read relative to `simdir`. `dfn` (a
`DfnDefinition` or `nothing`) only decides which blocks support whole-block OPEN/CLOSE;
`read_input_file` also selects the blocks MF6 reads. END lines outside blocks are
skipped with a warning.
"""
function parse_blocks(
        path::AbstractString; simdir::AbstractString = dirname(path),
        dfn::Union{Nothing, DfnDefinition} = nothing,
    )
    blocks, trailing = _parse_blocks(path, simdir, dfn)
    foreach(b -> _warn_strays(b.stray_ends), blocks)
    _warn_strays(trailing)
    return blocks
end

"""
    BlockSearch(name, required, repeat)

One block search of an MF6 reader (`BlockParser%GetBlock`): the block `name`, whether it
is required, and whether it is searched again after it was found (blocks with a block
variable).
"""
struct BlockSearch
    name::String
    required::Bool
    repeat::Bool
end

"""
    ReadPlan(searches, tail, tail_checks_end)

The order in which MF6 reads the blocks of a file: the `searches` one after the other,
then, if `tail` is a block name, only blocks of that name up to the end of the file (any
other block is an error: dynamic PERIOD blocks, OBS CONTINUOUS blocks, TAS TIME blocks).
With `tail = nothing` MF6 reads nothing after the searches. `tail_checks_end` is false
when the tail is read with `uget_any_block`, which does not look at END lines.
"""
struct ReadPlan
    searches::Vector{BlockSearch}
    tail::Union{Nothing, String}
    tail_checks_end::Bool
end

# Files not loaded through IDM, read by their own routines (block, required, repeat).
const _LEGACY_PLANS = Dict(
    # NumericalSolution.f90 sln_read_options / ImsLinearSettings.f90: all optional.
    "sln-ims" => ReadPlan(
        [BlockSearch("OPTIONS", false, false), BlockSearch("NONLINEAR", false, false), BlockSearch("LINEAR", false, false)],
        nothing, true,
    ),
    # TimeSeries.f90: GetBlock default blockRequired = .true.
    "utl-ts" => ReadPlan([BlockSearch("ATTRIBUTES", true, false), BlockSearch("TIMESERIES", true, false)], nothing, true),
    # TimeArraySeries.f90 tas_init / read_next_array: TIME blocks until the end.
    "utl-tas" => ReadPlan([BlockSearch("ATTRIBUTES", true, false), BlockSearch("TIME", true, true)], "TIME", true),
    # Obs.f90 read_obs_options / read_observations: GetBlock('*') then only CONTINUOUS.
    "utl-obs" => ReadPlan([BlockSearch("OPTIONS", false, false)], "CONTINUOUS", false),
    # ats.f90
    "utl-ats" => ReadPlan(
        [BlockSearch("OPTIONS", false, false), BlockSearch("DIMENSIONS", true, false), BlockSearch("PERIODDATA", true, false)],
        nothing, true,
    ),
    # GhostNode.f90
    "gwf-gnc" => ReadPlan(
        [BlockSearch("OPTIONS", false, false), BlockSearch("DIMENSIONS", true, false), BlockSearch("GNCDATA", true, false)],
        nothing, true,
    ),
    # gwf-mvr.f90: static blocks, then PERIOD blocks.
    "gwf-mvr" => ReadPlan(
        [BlockSearch("OPTIONS", false, false), BlockSearch("DIMENSIONS", true, false), BlockSearch("PACKAGES", true, false)],
        "PERIOD", true,
    ),
)

"""
    read_plan(dfn::DfnDefinition) -> ReadPlan

How MF6 reads the blocks of a `dfn` file. IDM files (LoadMf6File.f90 `load`): the DFN
blocks before PERIOD in DFN order, with the DFN's required flags (DISU/DISV1D/DISV2D
VERTICES and CELL2D are only read when NVERT > 0, so they count as optional here), then
PERIOD blocks if the DFN has them. Other files: `_LEGACY_PLANS`.
"""
function read_plan(dfn::DfnDefinition)
    p = get(_LEGACY_PLANS, dfn.name, nothing)
    isnothing(p) || return p
    searches = BlockSearch[]
    tail = nothing
    for b in dfn.blocks
        if b.name == "PERIOD"
            tail = "PERIOD"
            break
        end
        req = b.required && !(b.name in ("VERTICES", "CELL2D") && dfn.subcomponent in ("DISU", "DISV1D", "DISV2D"))
        push!(searches, BlockSearch(b.name, req, !isnothing(b.block_variable)))
    end
    return ReadPlan(searches, tail, true)
end

# The block variable on the BEGIN line, as MF6 reads it.
function _check_block_variable(b::InputBlock, dfn::DfnDefinition)
    db = dfn_block(dfn, b.name)
    bv = isnothing(db) ? nothing : db.block_variable
    isnothing(bv) && return nothing
    what = isempty(bv.longname) ? bv.name : bv.longname
    if bv.type === :integer
        get_int(b.header, 3; name = what)  # GetInteger
    elseif bv.type === :double
        get_double(b.header, 3; name = what)  # GetDouble (TAS TIME)
    elseif dfn.name == "utl-obs"
        # Obs.f90 read_observations: CONTINUOUS FILEOUT fname [BINARY]
        keyword(b.header, 3) == "FILEOUT" ||
            input_error(b.header, "CONTINUOUS keyword must be followed by \"FILEOUT\" then by filename")
        isnothing(token(b.header, 4)) && input_error(b.header, "missing observation output file name")
    end
    return nothing
end

# Stray END lines read by a search for `name`: uget_block errors when one names it.
function _search_strays(ends, name, checks)
    for e in ends
        if checks && keyword(e, 2) == name
            input_error(e, "looking for BEGIN $name but found END $name instead")
        end
        @warn "$(e.file):$(e.line): END $(keyword(e, 2)) outside a block is ignored"
    end
    return nothing
end

_ignored(b::InputBlock, why) =
    @warn "$(b.header.file):$(b.header.line): block $(b.name) is ignored: $why"

"""
    select_blocks(blocks, dfn::DfnDefinition, path; trailing_ends = InputLine[])
        -> Vector{InputBlock}

Read `blocks` (from `parse_blocks`) the way MF6 reads a `dfn` file (`read_plan`, rules in
the header of `blocks.jl`) and return the blocks MF6 reads. Blocks MF6 never reads are
dropped with a warning; what MF6 rejects throws an `MF6InputError` naming file and line.
`trailing_ends` are stray END lines after the last block.
"""
function select_blocks(
        blocks::Vector{InputBlock}, dfn::DfnDefinition, path::AbstractString;
        trailing_ends::Vector{InputLine} = InputLine[],
    )
    plan = read_plan(dfn)
    n = length(blocks)
    kept = InputBlock[]
    p = 1             # next block MF6 has not seen yet
    strays_read = 0   # stray ENDs before block p (or trailing) already read
    # Lines up to the next BEGIN line are read by the first search that reaches them.
    function strays!(name, checks)
        strays_read == p && return nothing
        _search_strays(p <= n ? blocks[p].stray_ends : trailing_ends, name, checks)
        strays_read = p
        return nothing
    end
    for s in plan.searches
        found = false
        while true
            strays!(s.name, true)
            if p <= n && blocks[p].name == s.name
                _check_block_variable(blocks[p], dfn)
                push!(kept, blocks[p])
                p += 1
                found = true
                s.repeat && continue
            elseif s.required && !found
                p <= n || throw(MF6InputError(String(path), 0, "required block $(s.name) not found; found end of file instead ($(dfn.name) file)"))
                input_error(blocks[p].header, "required block $(s.name) not found; found block $(blocks[p].name) instead")
            end
            break
        end
    end
    if isnothing(plan.tail)
        # MF6 reads nothing after the last search: warn about what it never sees.
        for q in p:n
            (q == p && strays_read == p) || _search_strays(blocks[q].stray_ends, "", false)
            what = isnothing(dfn_block(dfn, blocks[q].name)) ? "unknown block" : "out of order or repeated"
            order = join((s.name for s in plan.searches), ", ")
            _ignored(blocks[q], "$what; MF6 searches the blocks of a $(dfn.name) file in the order $order and reads nothing after them")
        end
        strays_read == n + 1 || _search_strays(trailing_ends, "", false)
    else
        last = 0
        while true
            strays!(plan.tail, plan.tail_checks_end)
            p <= n || break
            b = blocks[p]
            b.name == plan.tail ||
                input_error(b.header, "looking for BEGIN $(plan.tail); found block $(b.name) instead")
            _check_block_variable(b, dfn)
            if plan.tail == "PERIOD"
                # IdmMf6File.f90 dynamic_read_ionper
                v = get_int(b.header, 3; name = "stress period number")
                v > last || input_error(b.header, "period numbers not increasing: found $v but last period block was $last")
                last = v
            end
            push!(kept, b)
            p += 1
        end
    end
    return kept
end

# Packages whose DFN depends on READASARRAYS / READARRAYGRID in the OPTIONS block
# (ModflowInput.f90, update_sc_type).
const _ARRAY_VARIANT_PACKAGES = ("CHD", "DRN", "EVT", "GHB", "RCH", "RIV", "SPC", "WEL")

function _array_variant(blocks, dfn::DfnDefinition)
    dfn.subcomponent in _ARRAY_VARIANT_PACKAGES || return dfn
    # update_sc_type reads the first OPTIONS block; the last of the keywords wins.
    i = findfirst(b -> b.name == "OPTIONS", blocks)
    isnothing(i) && return dfn
    variant = dfn
    for l in blocks[i].lines
        k = keyword(l, 1)
        k == "READASARRAYS" && (variant = dfn_definition(dfn.name * "a"))
        k == "READARRAYGRID" && (variant = dfn_definition(dfn.name * "g"))
    end
    return variant
end

"""
    read_input_file(path; dfn = nothing, ftype = nothing, component = "GWF",
                    simdir = dirname(path)) -> InputFile

Read the blocks of the MF6 input file `path`. When a definition is known, keep the
blocks MF6 reads, in the order it reads them (`select_blocks`). Give the definition as
`dfn` (a `DfnDefinition` or its name, e.g. `"gwf-npf"`), or as the file type of a name
file with `ftype` (`"NPF6"`, `"TDIS6"`, `"GWF6-GWF6"`, `"TS6"`; see `dfn_name`), where
`component` is the model type. For CHD, DRN, EVT, GHB, RCH, RIV, SPC and WEL the
READASARRAYS / READARRAYGRID variant (`gwf-rcha`, ...) is selected from the OPTIONS
block, as in MF6. Whole-block OPEN/CLOSE files are read relative to `simdir`, the
simulation directory.

Errors are `MF6InputError`s naming the file and line; input MF6 ignores gives a warning.
"""
function read_input_file(
        path::AbstractString; dfn = nothing, ftype = nothing,
        component::AbstractString = "GWF", simdir::AbstractString = dirname(path),
    )
    if dfn isa AbstractString
        dfn = dfn_definition(dfn)
    elseif isnothing(dfn) && !isnothing(ftype)
        dfn = dfn_definition(dfn_name(ftype; component))
    end
    blocks, trailing = _parse_blocks(path, simdir, dfn)
    if isnothing(dfn)
        foreach(b -> _warn_strays(b.stray_ends), blocks)
        _warn_strays(trailing)
        return InputFile(String(path), nothing, blocks)
    end
    dfn = _array_variant(blocks, dfn)
    return InputFile(String(path), dfn, select_blocks(blocks, dfn, path; trailing_ends = trailing))
end
