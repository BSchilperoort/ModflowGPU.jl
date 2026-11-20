# Observation input (OBS6 files) on the host (T6.7).
#
# Ports src/Utilities/Observation/:
# - Obs.f90 `read_obs_options` (OPTIONS: DIGITS, PRINT_INPUT), `read_obs_blocks`
#   (`CONTINUOUS FILEOUT fname [BINARY]` blocks, one output file each; a file named twice
#   is an error), `define_fmts`;
# - Observe.f90 `ConstructObservation` (name and type: the first two words in upper case,
#   the name cut to LENOBSNAME = 40 characters; the rest of the line is the ID string);
# - the ID processors: gwf-obs.f90 `gwf_process_head_drawdown_obs_id` (one cellid),
#   `gwf_process_intercell_obs_id` (two cellids; the connection is checked when the model
#   is built), Obs.f90 `DefaultObsIdProcessor` (package observations: a cellid, or a
#   boundary name when the first word is not an integer); DiscretizationBase.f90
#   `noder_from_string`, Dis.f90 `nodeu_from_string` (cells outside the grid or the active
#   domain are errors).
# The observation types each owner supports (`*_df_obs`) are `OBS_TYPES`.

"""
    OBSNODATA

The value MF6 writes for a package observation that matches no boundary in the time step
(`DNODATA`, 3.0e30), e.g. a cell without a record of the package, or `TO-MVR` without a
mover.
"""
const OBSNODATA = 3.0e30

# ConstantsModule: LENOBSNAME, LENBOUNDNAME.
const LENOBSNAME = 40

"""
    ObsEntry

One observation of an OBS6 file (Observe.f90 `ObserveType`): `name` and `type` in upper
case (`"H1"`, `"HEAD"`), `idstring` (the ID string as written, for messages), `cells`
(user nodes: one cell, two for `FLOW-JA-FACE`, none for a boundary name), `boundname`
(upper case; `""` for a cellid), and the `file` and `line` it was read from.
"""
struct ObsEntry
    name::String
    type::String
    idstring::String
    cells::Vector{Int}
    boundname::String
    file::String
    line::Int
end

"""
    ObsOutputSpec

One `CONTINUOUS FILEOUT` block: the output `file` (as written, relative to the simulation
directory), `binary` (BINARY keyword) and its `observations` (`ObsEntry`) in input order,
which is the column order of the output.
"""
struct ObsOutputSpec
    file::String
    binary::Bool
    observations::Vector{ObsEntry}
end

"""
    ObsSpec

An OBS6 file (model or package observations): `file`, `digits` (DIGITS; 0 = MF6's
default, all digits of the value), `print_input` and the `outputs` (`ObsOutputSpec`, one
per CONTINUOUS block, in file order).
"""
struct ObsSpec
    file::String
    digits::Int
    print_input::Bool
    outputs::Vector{ObsOutputSpec}
end

Base.show(io::IO, o::ObsSpec) = print(
    io, "ObsSpec(\"", o.file, "\", ", sum(x -> length(x.observations), o.outputs; init = 0), " observation(s) in ",
    length(o.outputs), " file(s))",
)

"""
    OBS_TYPES

The observation types of each owner (`*_df_obs`): `:GWF` for the model (gwf-obs.f90
`gwf_obs_df`: `HEAD`, `DRAWDOWN`, `FLOW-JA-FACE`), and the stress packages by kind
(`chd_df_obs`, `wel_df_obs`, ...). `TO-MVR` is accepted and gives `OBSNODATA`, since
MVR is not supported; `WEL-REDUCTION` gives it without AUTO_FLOW_REDUCE.
"""
const OBS_TYPES = Dict{Symbol, Vector{String}}(
    :GWF => ["HEAD", "DRAWDOWN", "FLOW-JA-FACE"],
    :CHD => ["CHD"],
    :WEL => ["WEL", "TO-MVR", "WEL-REDUCTION"],
    :DRN => ["DRN", "TO-MVR"],
    :RIV => ["RIV", "TO-MVR"],
    :GHB => ["GHB", "TO-MVR"],
    :RCH => ["RCH"],
    :EVT => ["EVT"],
)

# Dis.f90 `nodeu_from_string` (urword ncode 2) for the cellid at word `i` of `words`:
# the user node, or an error naming the line.
function _obs_cellid(words, i, g::GridDims, idom, l::InputLine)
    nd = length(g.mshape)
    c = zeros(Int, nd)
    for d in 1:nd
        t = i + d - 1 <= length(words) ? words[i + d - 1] : nothing
        isnothing(t) && input_error(l, "Error reading data from ID string: missing cellid ($nd integers expected)")
        v = parse_int(t)
        isnothing(v) && input_error(l, "Error reading data from ID string: could not read an integer from \"$t\"")
        c[d] = v
    end
    valid_cellid(g, c) || input_error(l, "Error reading data from ID string: cell $(_shape_string(c)) is outside of the grid")
    n = user_node(g, c)
    # DiscretizationBase.f90 noder_from_string: a cell outside the active domain.
    idom[n] > 0 || input_error(l, "Cell is outside active grid domain: $(_shape_string(c))")
    return n
end

"""
    read_obs_file(path, owner, dis; simdir = dirname(path), file = path) -> ObsSpec

Read the OBS6 file `path` of `owner` (`:GWF` for the model, else the package kind, e.g.
`:GHB`) on the grid of `dis` (Obs.f90 `obs_ar`): the OPTIONS block (DIGITS, PRINT_INPUT),
then CONTINUOUS blocks, one output file each. Each observation's type must be one of
`OBS_TYPES[owner]` and its ID string a cellid of an active cell (two for
`FLOW-JA-FACE`); package observations may name a boundary instead (BOUNDNAMES). Errors
are `MF6InputError`s with MF6's messages where it has one. `file` is the name kept in
the spec (the loader passes it as written in the input).
"""
function read_obs_file(
        path::AbstractString, owner::Symbol, dis::AbstractDisSpec; simdir::AbstractString = dirname(path),
        file::AbstractString = path,
    )
    f = read_input_file(path; ftype = "OBS6", simdir)
    digits = 0
    print_input = false
    ob = find_block(f, "OPTIONS")
    seen_digits = false
    if !isnothing(ob)
        for l in ob.lines
            k = keyword(l, 1)
            if k == "DIGITS"
                seen_digits && input_error(l, "Error in OBS input: DIGITS has already been defined")
                seen_digits = true
                d = get_int(l, 2; name = "DIGITS")
                d < 0 && input_error(l, "Error in OBS input: Invalid value for DIGITS option")
                digits = d == 0 ? 0 : clamp(d, 2, 16)
            elseif k == "PRINT_INPUT"
                print_input = true
            else
                input_error(l, "Error in OBS input: Unrecognized option: $k")
            end
        end
    end
    g = grid_dims(dis)
    idom = idomain_user(dis)
    types = OBS_TYPES[owner]
    outputs = ObsOutputSpec[]
    for b in f.blocks
        b.name == "CONTINUOUS" || continue
        # Obs.f90 read_obs_blocks (FILEOUT checked by the block parser).
        fname = String(b.suffix[2])
        any(o -> o.file == fname, outputs) && input_error(b, "OBS outfile \"$fname\" is provided more than once.")
        binary = length(b.suffix) >= 3 && uppercase(b.suffix[3]) == "BINARY"
        entries = ObsEntry[]
        for l in b.lines
            words = tokens(l)
            length(words) < 2 && input_error(l, "Error reading observation: name and type expected")
            name = uppercase(words[1])
            name = first(name, LENOBSNAME)
            type = uppercase(words[2])
            type in types || input_error(l, "Observation type not found: $type")
            idstring = join(words[3:end], " ")
            isempty(idstring) && input_error(l, "Error reading data from ID string: no cell or boundary name given")
            cells = Int[]
            boundname = ""
            if owner === :GWF
                push!(cells, _obs_cellid(words, 3, g, idom, l))
                type == "FLOW-JA-FACE" && push!(cells, _obs_cellid(words, 3 + length(g.mshape), g, idom, l))
            elseif isnothing(parse_int(words[3]))
                # DefaultObsIdProcessor: not an integer, a boundary name (upper case).
                boundname = first(uppercase(words[3]), LENOBSNAME)
            else
                n = try
                    _obs_cellid(words, 3, g, idom, l)
                catch err
                    err isa MF6InputError || rethrow()
                    input_error(l, "Error reading data from ID string")
                end
                push!(cells, n)
            end
            push!(entries, ObsEntry(name, type, idstring, cells, boundname, l.file, l.line))
        end
        push!(outputs, ObsOutputSpec(fname, binary, entries))
    end
    return ObsSpec(String(file), digits, print_input, outputs)
end
