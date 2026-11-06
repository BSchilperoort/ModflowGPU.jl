# Schema of MF6 input files, generated from the MF6 definition (DFN) files.
#
# The types below describe one DFN file (`DfnDefinition`), its blocks (`DfnBlock`) and
# its variables (`DfnVariable`). The definitions themselves are in `definitions.jl`,
# written by `generate.jl` (run it again when the DFN files change); do not edit that
# file by hand.

"""
    DfnVariable

One variable (one entry) of a DFN file. Field names and defaults follow the DFN
attributes: `tagged` defaults to `true`, everything else to `false` or empty, and
`optional = false` means required (as in MF6's `dfn2f90.py`).

- `name`: the DFN name, lower case (`"stress_period_data"`); the input keyword is its
  upper-case form (`tag`).
- `type`: `:keyword`, `:integer`, `:double`, `:string`, `:record`, `:recarray` or
  `:keystring`; `members` lists the member names of the last three (lower case, in order).
- `shape`: the DFN shape as written, e.g. `"(naux)"`, `"(ncol, nrow; ncpl)"`, `""`.
- `reader`: `:urword` (keywords, scalars, lists) or `:readarray` (array input, T1.2).
- `block_variable`: the variable is read from the `BEGIN` line (period number, ...).
"""
Base.@kwdef struct DfnVariable
    name::String
    block::String
    type::Symbol
    members::Vector{String} = String[]
    shape::String = ""
    reader::Symbol = :urword
    optional::Bool = false
    tagged::Bool = true
    in_record::Bool = false
    preserve_case::Bool = false
    layered::Bool = false
    time_series::Bool = false
    block_variable::Bool = false
    numeric_index::Bool = false
    jagged_array::String = ""
    default_value::String = ""
    valid::Vector{String} = String[]
    mf6internal::String = ""
    mf6dimension::String = ""
    just_data::Bool = false
    repeating::Bool = false
    support_negative_index::Bool = false
    other_names::Vector{String} = String[]
    extended::Bool = false
    developmode::Bool = false
    netcdf::Bool = false
    removed::String = ""
    deprecated::String = ""
    longname::String = ""
end

"""
    DfnBlock

One block of a DFN file, in DFN order. `name` is upper case.

- `required`: the block must be present (rule of `dfn2f90.py`: some top-level variable
  or the list is not optional). MF6 enforces it for definitions with `idm = true`, except
  for PERIOD blocks (read by the dynamic loader, never required); see `read_plan`.
- `aggregate`: the block holds a list (`recarray`) or keystring data.
- `block_variable`: the variable on the `BEGIN` line (`iper`, `group_num`, ...), or
  `nothing`. Blocks with a block variable may repeat (`BEGIN PERIOD 1`, `BEGIN PERIOD 5`).
- `open_close`: whole-block `OPEN/CLOSE` is allowed (`BlockParser%GetBlock`
  `supportOpenClose`): every block except `GRIDDATA` (IDM, `LoadMf6File.f90`
  `parse_block`) and the time-array-series `TIME` block (`TimeArraySeries.f90`).
"""
Base.@kwdef struct DfnBlock
    name::String
    required::Bool = false
    aggregate::Bool = false
    block_variable::Union{Nothing, DfnVariable} = nothing
    open_close::Bool = true
    variables::Vector{DfnVariable} = DfnVariable[]
end

"""
    DfnDefinition

The definition of one input file type, from one DFN file (`gwf-wel.dfn` → `name =
"gwf-wel"`, `component = "GWF"`, `subcomponent = "WEL"`). `idm` is true when MF6
loads the file through its input data model (`utils/idmloader/dfns.txt`); only then is
the DFN's `required` flag of a block what MF6 enforces. `package_type` is
`"stress-package"`, `"advanced-stress-package"` or `""`; `multi_package` is the flopy
flag (more than one package of this type per model); `subpackages` lists the
`# mf6 subpackage` entries.
"""
Base.@kwdef struct DfnDefinition
    name::String
    component::String
    subcomponent::String
    idm::Bool = false
    multi_package::Bool = false
    package_type::String = ""
    subpackages::Vector{String} = String[]
    blocks::Vector{DfnBlock} = DfnBlock[]
end

function Base.show(io::IO, d::DfnDefinition)
    return print(io, "DfnDefinition(\"", d.name, "\", blocks = ", join((b.name for b in d.blocks), ", "), ")")
end
function Base.show(io::IO, b::DfnBlock)
    return print(io, "DfnBlock(\"", b.name, "\", ", length(b.variables), " variables)")
end
function Base.show(io::IO, v::DfnVariable)
    return print(io, "DfnVariable(\"", v.name, "\", :", v.type, ", block = \"", v.block, "\")")
end

"""
    dfn_tag(v::DfnVariable)

The input keyword of a variable: its name in upper case.
"""
dfn_tag(v::DfnVariable) = uppercase(v.name)

include("definitions.jl")

"""
    DFN_DEFINITIONS

All generated DFN definitions by name (`"gwf-wel"`, `"sim-nam"`, `"utl-ts"`, ...).
`DFN_SOURCE` says which DFN files they were generated from.
"""
const DFN_DEFINITIONS = Dict{String, DfnDefinition}(d.name => d for d in _dfn_definitions())

"""
    dfn_definition(name) -> DfnDefinition

The definition named `name` (case-insensitive, e.g. `"gwf-npf"`). Errors for a name
that was not generated (see `src/input/dfn/generate.jl` for the DFN files in scope).
"""
function dfn_definition(name::AbstractString)
    d = get(DFN_DEFINITIONS, lowercase(name), nothing)
    isnothing(d) && throw(ArgumentError("no DFN definition \"$name\" (unsupported input file type)"))
    return d
end

"""
    dfn_block(d::DfnDefinition, name) -> Union{DfnBlock, Nothing}

The block `name` (case-insensitive) of `d`, or `nothing`.
"""
function dfn_block(d::DfnDefinition, name::AbstractString)
    n = uppercase(name)
    i = findfirst(b -> b.name == n, d.blocks)
    return isnothing(i) ? nothing : d.blocks[i]
end

"""
    dfn_variable(b::DfnBlock, name) -> Union{DfnVariable, Nothing}

The variable `name` (case-insensitive) of block `b`, or `nothing`.
"""
function dfn_variable(b::DfnBlock, name::AbstractString)
    n = lowercase(name)
    i = findfirst(v -> v.name == n, b.variables)
    return isnothing(i) ? nothing : b.variables[i]
end

# File types that are not specific to a model type (name file `ftype` without the `6`).
const _UTL_FTYPES = ("TS", "TAS", "OBS", "TVK", "TVS", "ATS", "HPC", "NCF", "SPC", "SPCA", "LAKTAB", "SFRTAB")

"""
    dfn_name(ftype; component = "GWF") -> String

The DFN name of an input file type as it appears in name files: `"WEL6"` → `"gwf-wel"`,
`"TDIS6"` → `"sim-tdis"`, `"IMS6"` → `"sln-ims"`, `"GWF6-GWF6"` → `"exg-gwfgwf"`,
`"TS6"` → `"utl-ts"`, `"GWF6"` (a model name file) → `"gwf-nam"`, `"SIM"` → `"sim-nam"`.
`component` is the model type of model-level packages. READASARRAYS / READARRAYGRID
variants (`gwf-rcha`, `gwf-welg`) are selected from the OPTIONS block by
`read_input_file`, as MF6 does (`ModflowInput.f90`, `update_sc_type`).
"""
function dfn_name(ftype::AbstractString; component::AbstractString = "GWF")
    f = uppercase(strip(ftype))
    if occursin('-', f)
        # exchange, e.g. GWF6-GWF6 → exg-gwfgwf
        a, b = split(f, '-'; limit = 2)
        return "exg-" * lowercase(rstrip(a, '6') * rstrip(b, '6'))
    end
    f = rstrip(f, '6')
    f == "SIM" && return "sim-nam"
    f == "TDIS" && return "sim-tdis"
    f in ("IMS", "EMS") && return "sln-" * lowercase(f)
    f in _UTL_FTYPES && return "utl-" * lowercase(f)
    f in ("GWF", "GWT", "GWE", "PRT", "CHF", "OLF") && return lowercase(f) * "-nam"
    return lowercase(component) * "-" * lowercase(f)
end
