# Block parser for MF6 input files (host only).
#
# Ports the block rules of src/Utilities/BlockParser.f90 (`uget_block`,
# `uget_any_block`, `GetNextLine`, `uterminate_block`):
# - A block is `BEGIN name [suffix ...]` … `END name`, compared in upper case. The words
#   after the name on the BEGIN line are the block's suffix (period number, solution-group
#   number, `FILEOUT name.csv` of an OBS `CONTINUOUS` block, ...); words after the name on
#   the END line are ignored (`END SOLUTIONGROUP 1`).
# - Inside a block, a BEGIN line, an END line with another name, or the end of the file
#   is an error.
# - Lines outside blocks are skipped, as MF6 does while it searches for the next BEGIN;
#   an END line outside a block is an error here (MF6 reports it only for the block it is
#   looking for).
# - Whole-block OPEN/CLOSE: when the first line of a block that supports it is
#   `OPEN/CLOSE fname`, the block's lines are read from `fname` instead, until its end
#   (or an `END name` line in it). The file name keeps its case and is relative to the
#   simulation directory (MF6 opens it relative to its working directory, which is the
#   simulation directory). A line with `(BINARY)` or `SFAC` after the file name is not a
#   whole-block include but list input, and stays in the block (`uget_block`). The next
#   line of the package file must end the block (mf6io.pdf, "Specification of Block
#   Information in OPEN/CLOSE File").
#
# With a DFN definition (src/input/dfn) the blocks are also checked: unknown block names,
# block order (DFN order, which is the order MF6 reads them in; MF6 skips out-of-order
# blocks silently), repeated blocks (only blocks with a block variable, such as PERIOD,
# repeat), the block variable on the BEGIN line (PERIOD numbers must increase, as in
# `IdmMf6File.f90` `dynamic_read_ionper`), and required blocks (files read through MF6's
# input data model only).

"""
    InputBlock

One block of an MF6 input file.

- `name`: the block name in upper case (`"PERIOD"`).
- `suffix`: the words after the name on the BEGIN line, original case (`["1"]` for
  `BEGIN PERIOD 1`, `["FILEOUT", "heads.csv"]` for `BEGIN CONTINUOUS FILEOUT heads.csv`).
- `lines`: the non-comment lines between BEGIN and END (or of the OPEN/CLOSE file).
- `header`: the BEGIN line (for its file and line number).
- `open_close`: path of the whole-block OPEN/CLOSE file the lines came from, or `nothing`.
"""
struct InputBlock
    name::String
    suffix::Vector{SubString{String}}
    lines::Vector{InputLine}
    header::InputLine
    open_close::Union{Nothing, String}
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

The blocks of one MF6 input file, in file order. `dfn` is the DFN definition the blocks
were checked against, or `nothing`.
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

"""
    parse_blocks(path; simdir = dirname(path), dfn = nothing) -> Vector{InputBlock}

Split the file `path` into blocks (the rules are in the header of `blocks.jl`).
Whole-block OPEN/CLOSE files are read relative to `simdir`. `dfn` (a `DfnDefinition` or
`nothing`) only decides which blocks support whole-block OPEN/CLOSE; use
`read_input_file` to also check the blocks against it.
"""
function parse_blocks(
        path::AbstractString; simdir::AbstractString = dirname(path),
        dfn::Union{Nothing, DfnDefinition} = nothing,
    )
    lines = read_input_lines(path)
    blocks = InputBlock[]
    n = length(lines)
    i = 1
    while i <= n
        l = lines[i]
        key = keyword(l, 1)
        i += 1
        if key == "END"
            input_error(l, "END $(keyword(l, 2)) without a matching BEGIN")
        elseif key != "BEGIN"
            continue  # outside a block: skipped, as in uget_block
        end
        name = keyword(l, 2)
        isempty(name) && input_error(l, "block name missing after BEGIN")
        suffix = l.tokens[3:end]
        body = InputLine[]
        oc = nothing
        ended = false
        if i <= n && keyword(lines[i], 1) == "OPEN/CLOSE" && _supports_open_close(dfn, name) &&
                !any(t -> uppercase(t) in ("(BINARY)", "SFAC"), lines[i].tokens[3:end])
            ocl = lines[i]
            length(ocl) >= 2 || input_error(ocl, "missing file name after OPEN/CLOSE")
            oc = _resolve_path(ocl[2], simdir)
            isfile(oc) || input_error(ocl, "OPEN/CLOSE file not found: $oc")
            for el in read_input_lines(oc)
                k = keyword(el, 1)
                if k == "BEGIN"
                    input_error(el, "BEGIN detected before END $name; END $name must be used to end $name")
                elseif k == "END"
                    keyword(el, 2) == name || input_error(el, "END $(keyword(el, 2)) detected without END $name; END $name must be used to end $name")
                    ended = true
                    break
                end
                push!(body, el)
            end
            i += 1
            if ended
                # MF6 closes the block at the END in the OPEN/CLOSE file; an END line
                # right after the OPEN/CLOSE line is then outside the block, and skipped.
                i <= n && keyword(lines[i], 1) == "END" && keyword(lines[i], 2) == name && (i += 1)
            else
                i <= n || throw(MF6InputError(l.file, l.line, "unexpected end of file: BEGIN $name has no END $name"))
                k = keyword(lines[i], 1)
                k == "END" || input_error(lines[i], "the line after OPEN/CLOSE must be END $name")
            end
        end
        while !ended
            i <= n || throw(MF6InputError(l.file, l.line, "unexpected end of file: BEGIN $name has no END $name"))
            bl = lines[i]
            i += 1
            k = keyword(bl, 1)
            if k == "END"
                keyword(bl, 2) == name || input_error(bl, "END $(keyword(bl, 2)) detected without END $name; END $name must be used to end $name")
                ended = true
            elseif k == "BEGIN"
                input_error(bl, "BEGIN detected before END $name; END $name must be used to end $name")
            else
                push!(body, bl)
            end
        end
        push!(blocks, InputBlock(name, suffix, body, l, oc))
    end
    return blocks
end

"""
    validate_blocks(blocks, dfn::DfnDefinition, path)

Check `blocks` against the DFN definition `dfn` (the checks are listed in the header of
`blocks.jl`); throws an `MF6InputError` naming the file and line of the first problem.
"""
function validate_blocks(blocks::Vector{InputBlock}, dfn::DfnDefinition, path::AbstractString)
    valid = join((b.name for b in dfn.blocks), ", ")
    last = 0
    lastnum = nothing
    for b in blocks
        k = findfirst(db -> db.name == b.name, dfn.blocks)
        isnothing(k) && input_error(b.header, "unknown block $(b.name) in a $(dfn.name) file (valid blocks: $valid)")
        db = dfn.blocks[k]
        bv = db.block_variable
        if k < last
            input_error(b.header, "block $(b.name) must come before block $(dfn.blocks[last].name)")
        elseif k == last && isnothing(bv)
            input_error(b.header, "block $(b.name) given more than once")
        end
        k == last || (lastnum = nothing)
        last = k
        isnothing(bv) && continue
        what = isempty(bv.longname) ? bv.name : bv.longname
        isempty(b.suffix) && input_error(b.header, "missing $what after BEGIN $(b.name)")
        if bv.type === :integer
            v = get_int(b.header, 3; name = what)
            if !isnothing(lastnum) && v <= lastnum
                input_error(b.header, "$(b.name) numbers not increasing: found $v after $lastnum")
            end
            v >= 1 || input_error(b.header, "$what must be at least 1, found $v")
            lastnum = v
        elseif bv.type === :double
            get_double(b.header, 3; name = what)
        end
    end
    # Required blocks: only for files MF6 loads through IDM; the other readers (IMS, TS,
    # OBS, ...) have their own rules (IMS NONLINEAR and LINEAR are optional, for example).
    dfn.idm || return nothing
    for db in dfn.blocks
        db.required || continue
        # DISU (and DISV1D/DISV2D) read VERTICES and CELL2D only when NVERT > 0
        # (LoadMf6File.f90, parse_block).
        if db.name in ("VERTICES", "CELL2D") && dfn.subcomponent in ("DISU", "DISV1D", "DISV2D")
            continue
        end
        any(b -> b.name == db.name, blocks) ||
            throw(MF6InputError(String(path), 0, "required block $(db.name) not found in a $(dfn.name) file"))
    end
    return nothing
end

# Packages whose DFN depends on READASARRAYS / READARRAYGRID in the OPTIONS block
# (ModflowInput.f90, update_sc_type).
const _ARRAY_VARIANT_PACKAGES = ("CHD", "DRN", "EVT", "GHB", "RCH", "RIV", "SPC", "WEL")

function _array_variant(blocks, dfn::DfnDefinition)
    dfn.subcomponent in _ARRAY_VARIANT_PACKAGES || return dfn
    for b in blocks
        b.name == "OPTIONS" || continue
        for l in b.lines
            k = keyword(l, 1)
            k == "READASARRAYS" && return dfn_definition(dfn.name * "a")
            k == "READARRAYGRID" && return dfn_definition(dfn.name * "g")
        end
    end
    return dfn
end

"""
    read_input_file(path; dfn = nothing, ftype = nothing, component = "GWF",
                    simdir = dirname(path)) -> InputFile

Read the blocks of the MF6 input file `path` and, when a definition is known, check them
against it (`validate_blocks`). Give the definition as `dfn` (a `DfnDefinition` or its
name, e.g. `"gwf-npf"`), or as the file type of a name file with `ftype` (`"NPF6"`,
`"TDIS6"`, `"GWF6-GWF6"`, `"TS6"`; see `dfn_name`), where `component` is the model type.
For CHD, DRN, EVT, GHB, RCH, RIV, SPC and WEL the READASARRAYS / READARRAYGRID variant
(`gwf-rcha`, ...) is selected from the OPTIONS block, as in MF6. Whole-block OPEN/CLOSE
files are read relative to `simdir`, the simulation directory.

Errors are `MF6InputError`s naming the file and line.
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
    blocks = parse_blocks(path; simdir, dfn)
    if !isnothing(dfn)
        dfn = _array_variant(blocks, dfn)
        validate_blocks(blocks, dfn, path)
    end
    return InputFile(String(path), dfn, blocks)
end
