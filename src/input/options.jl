# Keyword blocks (OPTIONS, DIMENSIONS, TIMING) of MF6 input files read through the DFN
# schema (host only).
#
# Ports how MF6's input data model reads a block of tagged values
# (LoadMf6File.f90 `parse_block`, `parse_record_tag`, `load_tag`, `load_keyword_type`,
# `load_string_type`, `load_io_tag`, `load_auxvar_names`, `load_integer_type`,
# `load_double_type`; DefinitionSelect.f90 `get_param_definition_type`,
# `split_record_dfn_tag2`):
# - Each line starts with a tag, the upper-case name of a DFN variable of the block; an
#   unknown tag is an error ('Input file tag not found: "X" in block "OPTIONS".').
# - KEYWORD: present means true. STRING: the next word, in upper case unless the DFN says
#   preserve_case; `AUXILIARY` (shape `naux`) takes all words after it (`urdaux`, `AUX` is
#   an alias). INTEGER / DOUBLE: the next word as a Fortran number. A DFN `valid` list is
#   checked ("Unrecognized input value for X option.").
# - A tag that is the first member of a DFN record (`REWET WETFCT 1.0 IWETIT 1 IHDWET 0`,
#   `TS6 FILEIN name`, `INNER_RCLOSE 1e-6 STRICT`, `HEAD PRINT_FORMAT COLUMNS 10 ...`)
#   reads the record's other members in order: a keyword member must be the next word, a
#   tagged value member is its tag then the value, an untagged one the value alone, and a
#   nested record (OC `formatrecord`) its own members. Optional members may be missing at
#   the end of the line. When several records start with the same tag (OC `HEAD FILEOUT`
#   and `HEAD PRINT_FORMAT`), the second word selects the record. A FILEIN name may be
#   given more than once (several TS6 files); the names are collected in order.
# - Words after the values a tag needs are ignored (trailing comments).
#
# The result is an `OptionSet`: the values by DFN variable name, plus the tags in input
# order for the support checks of the loader (`check_options`, src/input/supported.jl).

"""
    OptionSet

The values of one keyword block (`read_options`): `values` maps DFN variable names
(lower case) to `true` (keywords), a `String`, `Int`, `Float64`, a `Vector{String}`
(AUXILIARY names; FILEIN names); `lines` maps the tag of each line (upper case, as
written: `"TS6"`, `"XT3D"`) to its line, in input order (`tags`). `file` and `block`
name where it was read.
"""
struct OptionSet
    file::String
    block::String
    values::Dict{String, Any}
    tags::Vector{String}
    lines::Vector{InputLine}
end

OptionSet(file::AbstractString, block::AbstractString) =
    OptionSet(String(file), String(block), Dict{String, Any}(), String[], InputLine[])

Base.show(io::IO, o::OptionSet) = print(io, "OptionSet(", o.block, ": ", join(o.tags, ", "), ")")

"""
    option(o::OptionSet, name, default)

The value of DFN variable `name` (lower case), or `default` when it was not given.
"""
option(o::OptionSet, name::AbstractString, default) = get(o.values, name, default)

"""
    has_option(o::OptionSet, name) -> Bool

Whether DFN variable `name` (lower case) was given.
"""
has_option(o::OptionSet, name::AbstractString) = haskey(o.values, name)

"""
    option_line(o::OptionSet, tag) -> Union{InputLine, Nothing}

The (last) line whose first word is `tag` (upper case), or `nothing`.
"""
function option_line(o::OptionSet, tag::AbstractString)
    i = findlast(==(tag), o.tags)
    return isnothing(i) ? nothing : o.lines[i]
end

function _dfn_value(l::InputLine, v::DfnVariable, i::Int)
    if v.type === :integer
        return get_int(l, i; name = uppercase(v.name))
    elseif v.type === :double
        return get_double(l, i; name = uppercase(v.name))
    elseif v.type === :string
        t = token(l, i)
        isnothing(t) && input_error(l, "missing value for $(uppercase(v.name))")
        s = v.preserve_case ? String(t) : uppercase(t)
        if !isempty(v.valid) && !(lowercase(s) in v.valid)
            input_error(l, "Unrecognized input value for $(uppercase(v.name)) option.")
        end
        return s
    end
    throw(ArgumentError("variable $(v.name) of type $(v.type) is not a scalar"))
end

_record_error(l::InputLine, block, tag) =
    input_error(l, "\"$block\" block input record that includes keyword \"$tag\" is not properly formed.")

function _set_value!(vals, v::DfnVariable, x, filein::Bool)
    if filein
        push!(get!(() -> String[], vals, v.name)::Vector{String}, x)
    else
        vals[v.name] = x
    end
    return nothing
end

# Read the members `names` of a record from word `p` of `l`; returns the next word index.
function _read_members!(vals, db::DfnBlock, names, l::InputLine, p::Int, tag)
    filein = false
    for m in names
        mv = dfn_variable(db, m)
        isnothing(mv) && throw(ArgumentError("record member $m is not a variable of block $(db.name)"))
        w = keyword(l, p)
        if mv.type === :keyword
            if w == uppercase(m)
                m == "filein" && (filein = true)
                m in ("filein", "fileout") || (vals[m] = true)
                p += 1
            elseif mv.optional
                continue
            elseif isempty(w)
                _record_error(l, db.name, tag)
            else
                input_error(l, "Expecting record input tag \"$(uppercase(m))\" but instead found \"$w\".")
            end
        elseif mv.type === :record
            p = _read_members!(vals, db, mv.members, l, p, tag)
        else
            if mv.tagged
                if w != uppercase(m)
                    (isempty(w) && mv.optional) && break
                    isempty(w) && _record_error(l, db.name, tag)
                    input_error(l, "Expecting record input tag \"$(uppercase(m))\" but instead found \"$w\".")
                end
                p += 1
            end
            if isnothing(token(l, p))
                mv.optional && break
                _record_error(l, db.name, tag)
            end
            _set_value!(vals, mv, _dfn_value(l, mv, p), filein)
            p += 1
        end
    end
    return p
end

"""
    read_options(b::Union{InputBlock, Nothing}, dfn::DfnDefinition; file = "") -> OptionSet

Read a block of tagged values (OPTIONS, DIMENSIONS, TIMING, ...) of a file with
definition `dfn` the way MF6's input data model does (rules in the header of
`options.jl`). `b = nothing` (an absent optional block) gives an empty set.
"""
function read_options(b::Union{InputBlock, Nothing}, dfn::DfnDefinition; file::AbstractString = "")
    isnothing(b) && return OptionSet(file, "")
    o = OptionSet(isempty(file) ? b.header.file : file, b.name)
    db = dfn_block(dfn, b.name)
    isnothing(db) && input_error(b, "block $(b.name) is not part of a $(dfn.name) file")
    for l in b.lines
        tag = keyword(l, 1)
        name = tag == "AUX" ? "auxiliary" : lowercase(tag)
        v = dfn_variable(db, name)
        isnothing(v) && input_error(l, "Input file tag not found: \"$tag\" in block \"$(b.name)\".")
        push!(o.tags, tag)
        push!(o.lines, l)
        if v.in_record
            # parse_record_tag: the records whose first member is this tag; the second
            # word selects one when there are several.
            recs = filter(r -> r.type === :record && !r.in_record && !isempty(r.members) && r.members[1] == v.name, db.variables)
            if length(recs) > 1
                w2 = lowercase(keyword(l, 2))
                recs = filter(r -> length(r.members) >= 2 && r.members[2] == w2, recs)
            end
            isempty(recs) && _record_error(l, b.name, tag)
            r = recs[1]
            if v.type === :keyword
                o.values[v.name] = true
                p = 2
            else
                # a value as the first member: INNER_RCLOSE 1e-6 STRICT
                isnothing(token(l, 2)) && _record_error(l, b.name, tag)
                o.values[v.name] = _dfn_value(l, v, 2)
                p = 3
            end
            _read_members!(o.values, db, r.members[2:end], l, p, tag)
        elseif v.type === :keyword
            o.values[v.name] = true
        elseif v.type === :string && v.shape == "(naux)"
            haskey(o.values, v.name) && input_error(l, "auxiliary variables already specified; auxiliary variables must be specified on one line in the options block")
            names = String[]
            for t in Iterators.drop(eachtoken(l), 1)
                ncodeunits(t) <= LENAUXNAME || input_error(l, "auxiliary variable name \"$t\" is longer than $LENAUXNAME characters")
                push!(names, uppercase(t))
            end
            o.values[v.name] = names
        elseif v.type in (:integer, :double, :string)
            o.values[v.name] = _dfn_value(l, v, 2)
        else
            input_error(l, "$tag of type $(v.type) cannot be read in block $(b.name)")
        end
    end
    return o
end

"""
    read_options(f::InputFile, block) -> OptionSet

Read the keyword block `block` (`"OPTIONS"`, `"DIMENSIONS"`) of `f` (read with its DFN
definition); empty when the block is absent.
"""
function read_options(f::InputFile, block::AbstractString)
    isnothing(f.dfn) && throw(ArgumentError("read_options: $(f.path) was read without a DFN definition"))
    return read_options(find_block(f, block), f.dfn; file = f.path)
end
