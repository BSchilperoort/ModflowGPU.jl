# Generate `definitions.jl` (the DFN schema of ModflowGPU's input reader) from the MF6
# definition files.
#
#   julia src/input/dfn/generate.jl [dfn_dir [idm_list]]
#
# `dfn_dir` defaults to `$REFERENCE_DIR/mf6.8.2_dfns` (`REFERENCE_DIR` defaults to
# `<repo root>/reference`, see validation/fetch_reference.sh). `idm_list` is MF6's list
# of the DFNs loaded through its input data model (IDM), default
# `$REFERENCE_DIR/modflow6/utils/idmloader/dfns.txt`; those definitions get `idm = true`
# (their required blocks are checked; other files, such as IMS, TS and OBS, are read by
# their own MF6 routines, which decide themselves what is required). Julia standard library
# only; not part of the package. Run Runic on the output afterwards
# (`julia --project=@runic -m Runic --inplace src/input/dfn`).
#
# Scope: every DFN of the simulation (`sim-*`), the solutions (`sln-*`), GWF models
# (`gwf-*`), the utilities (`utl-*`: ts, tas, obs, tvk, ...) and the GWF-GWF exchange
# (`exg-gwfgwf`). GWT/GWE/PRT/CHF/OLF models and their exchanges are out of scope
# (PLAN.md section 1).
#
# The parsing rules follow MF6's `utils/idmloader/scripts/dfn2f90.py`: an entry starts
# at a `block` line and ends at a blank line; `optional` defaults to false and `tagged`
# to true; a block is required when one of its top-level non-aggregate variables (or
# its aggregate) is not optional, and it is aggregate when it holds a `recarray` or
# `keystring`. As in dfn2f90.py an empty OPTIONS block is prepended to DFNs without one
# (utl-ats: `ats.f90` reads an OPTIONS block; utl-laktab, utl-sfrtab: IDM), except for
# the time-series files, which MF6 reads without looking for OPTIONS (NO_OPTIONS:
# utl-ts, utl-tas). Descriptions are dropped.

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const REFERENCE_DIR = get(ENV, "REFERENCE_DIR", joinpath(REPO_ROOT, "reference"))
const OUTPUT = joinpath(@__DIR__, "definitions.jl")

in_scope(stem) = any(p -> startswith(stem, p), ("sim-", "sln-", "gwf-", "utl-")) || stem == "exg-gwfgwf"

const KNOWN_KEYS = Set(
    [
        "block", "name", "type", "shape", "reader", "optional", "tagged", "in_record",
        "preserve_case", "layered", "time_series", "block_variable", "numeric_index",
        "jagged_array", "default_value", "valid", "mf6internal", "mf6dimension",
        "just_data", "repeating", "support_negative_index", "other_names", "extended",
        "developmode", "netcdf", "removed", "deprecated", "longname", "description",
    ]
)

const NO_OPTIONS = ("utl-ts", "utl-tas")

const TYPE_MAP = Dict(
    "keyword" => :keyword, "integer" => :integer, "double" => :double,
    "string" => :string, "record" => :record, "recarray" => :recarray,
    "keystring" => :keystring,
)

struct Entry
    attrs::Dict{String, String}
    line::Int
end

function read_entries(path)
    entries = Entry[]
    header = String[]
    cur = nothing
    for (i, raw) in enumerate(eachline(path))
        l = rstrip(raw)
        if startswith(l, "#")
            push!(header, strip(l[2:end]))
            continue
        end
        if isempty(strip(l))
            cur = nothing
            continue
        end
        key, val = let p = findfirst(' ', l)
            isnothing(p) ? (String(l), "") : (String(l[1:(p - 1)]), String(strip(l[(p + 1):end])))
        end
        key in KNOWN_KEYS || error("$path:$i: unknown DFN attribute '$key'")
        if key == "block"
            cur = Entry(Dict{String, String}(), i)
            push!(entries, cur)
        elseif isnothing(cur)
            continue  # entries without a block (common.dfn style) are not variables
        end
        haskey(cur.attrs, key) && error("$path:$i: attribute '$key' given twice")
        cur.attrs[key] = val
    end
    return entries, header
end

isflag(a, key) = lowercase(get(a, key, "false")) == "true"

function make_variable(a, path, line)
    haskey(a, "name") || error("$path:$line: entry without a name")
    words = split(get(a, "type", ""))
    isempty(words) && error("$path:$line: entry without a type")
    t = if words[1] == "double" && length(words) >= 2 && words[2] == "precision"
        popfirst!(words)
        popfirst!(words)
        :double
    else
        haskey(TYPE_MAP, words[1]) || error("$path:$line: unknown type '$(a["type"])'")
        TYPE_MAP[popfirst!(words)]
    end
    return (;
        name = lowercase(a["name"]),
        block = uppercase(a["block"]),
        type = t,
        members = lowercase.(String.(words)),
        shape = get(a, "shape", ""),
        reader = Symbol(get(a, "reader", "urword")),
        optional = isflag(a, "optional"),
        tagged = lowercase(get(a, "tagged", "true")) == "true",
        in_record = isflag(a, "in_record"),
        preserve_case = isflag(a, "preserve_case"),
        layered = isflag(a, "layered"),
        time_series = isflag(a, "time_series"),
        block_variable = isflag(a, "block_variable"),
        numeric_index = isflag(a, "numeric_index"),
        jagged_array = get(a, "jagged_array", ""),
        default_value = get(a, "default_value", ""),
        valid = String.(split(get(a, "valid", ""))),
        mf6internal = get(a, "mf6internal", ""),
        mf6dimension = get(a, "mf6dimension", ""),
        just_data = isflag(a, "just_data"),
        repeating = isflag(a, "repeating"),
        support_negative_index = isflag(a, "support_negative_index"),
        other_names = String.(split(get(a, "other_names", ""))),
        extended = isflag(a, "extended"),
        developmode = isflag(a, "developmode"),
        netcdf = isflag(a, "netcdf"),
        removed = get(a, "removed", ""),
        deprecated = get(a, "deprecated", ""),
        longname = get(a, "longname", ""),
    )
end

# Defaults of DfnVariable (src/input/dfn/schema.jl); fields equal to these are omitted.
const VAR_DEFAULTS = (;
    members = String[], shape = "", reader = :urword, optional = false, tagged = true,
    in_record = false, preserve_case = false, layered = false, time_series = false,
    block_variable = false, numeric_index = false, jagged_array = "", default_value = "",
    valid = String[], mf6internal = "", mf6dimension = "", just_data = false,
    repeating = false, support_negative_index = false, other_names = String[],
    extended = false, developmode = false, netcdf = false, removed = "",
    deprecated = "", longname = "",
)

function parse_dfn(path, idm_set)
    stem = splitext(basename(path))[1]
    component, subcomponent = uppercase.(split(stem, '-'; limit = 2))
    entries, header = read_entries(path)
    multi_package = any(h -> occursin("flopy multi-package", h), header)
    package_type = ""
    subpackages = String[]
    for h in header
        m = match(r"^package-type\s+(\S+)", h)
        isnothing(m) || (package_type = m[1])
        m = match(r"^mf6 subpackage\s+(\S+)", h)
        isnothing(m) || push!(subpackages, uppercase(m[1]))
    end
    vars = [make_variable(e.attrs, path, e.line) for e in entries]
    blocknames = unique(v.block for v in vars)
    if !isempty(blocknames) && !("OPTIONS" in blocknames) && !(stem in NO_OPTIONS)
        pushfirst!(blocknames, "OPTIONS")
    end
    blocks = map(blocknames) do bn
        bvars = filter(v -> v.block == bn, vars)
        bvar = filter(v -> v.block_variable, bvars)
        length(bvar) <= 1 || error("$path: block $bn has more than one block variable")
        aggregates = filter(v -> v.type in (:recarray, :keystring) && !v.block_variable, bvars)
        aggregate = !isempty(aggregates)
        # As in dfn2f90.py, also for blocks with a block variable (SOLUTIONGROUP is
        # required; PERIOD blocks are read by the dynamic loader, never as required).
        required = if aggregate
            any(v -> !v.optional, aggregates)
        else
            any(v -> !v.in_record && !v.block_variable && !v.optional, bvars)
        end
        open_close = !(bn == "GRIDDATA" || any(v -> v.reader === :readarray && !v.tagged, bvars))
        (; name = bn, required, aggregate, block_variable = isempty(bvar) ? nothing : bvar[1], open_close, variables = filter(v -> !v.block_variable, bvars))
    end
    idm = stem * ".dfn" in idm_set
    return (; name = stem, component, subcomponent, idm, multi_package, package_type, subpackages, blocks)
end

lit(x::AbstractString) = repr(String(x))
lit(x::Symbol) = repr(x)
lit(x::Bool) = string(x)
lit(x::Vector{String}) = isempty(x) ? "String[]" : "[" * join(lit.(x), ", ") * "]"

function emit_variable(io, v, indent; prefix = "")
    fields = String["name = $(lit(v.name))", "block = $(lit(v.block))", "type = $(lit(v.type))"]
    for k in keys(VAR_DEFAULTS)
        val = getproperty(v, k)
        val == VAR_DEFAULTS[k] && continue
        push!(fields, "$k = $(lit(val))")
    end
    pad = " "^indent
    println(io, pad, prefix, "DfnVariable(")
    for f in fields
        println(io, pad, "    ", f, ",")
    end
    return print(io, pad, ")")
end

function emit_definition(io, d)
    cname = "DFN_" * uppercase(replace(d.name, '-' => '_'))
    println(io, "const ", cname, " = DfnDefinition(")
    println(io, "    name = ", lit(d.name), ",")
    println(io, "    component = ", lit(d.component), ",")
    println(io, "    subcomponent = ", lit(d.subcomponent), ",")
    d.idm && println(io, "    idm = true,")
    d.multi_package && println(io, "    multi_package = true,")
    isempty(d.package_type) || println(io, "    package_type = ", lit(d.package_type), ",")
    isempty(d.subpackages) || println(io, "    subpackages = ", lit(d.subpackages), ",")
    println(io, "    blocks = [")
    for b in d.blocks
        println(io, "        DfnBlock(")
        println(io, "            name = ", lit(b.name), ",")
        b.required && println(io, "            required = true,")
        b.aggregate && println(io, "            aggregate = true,")
        if !isnothing(b.block_variable)
            emit_variable(io, b.block_variable, 12; prefix = "block_variable = ")
            println(io, ",")
        end
        b.open_close || println(io, "            open_close = false,")
        println(io, "            variables = [")
        for v in b.variables
            emit_variable(io, v, 16)
            println(io, ",")
        end
        println(io, "            ],")
        println(io, "        ),")
    end
    println(io, "    ],")
    println(io, ")")
    return cname
end

function main(args)
    dfndir = isempty(args) ? joinpath(REFERENCE_DIR, "mf6.8.2_dfns") : args[1]
    isdir(dfndir) || error("DFN directory not found: $dfndir (run validation/fetch_reference.sh)")
    files = sort(filter(f -> endswith(f, ".dfn") && in_scope(splitext(f)[1]), readdir(dfndir)))
    isempty(files) && error("no in-scope DFN files in $dfndir")
    idmfile = length(args) >= 2 ? args[2] : joinpath(REFERENCE_DIR, "modflow6", "utils", "idmloader", "dfns.txt")
    isfile(idmfile) || error("IDM list not found: $idmfile (run validation/fetch_reference.sh)")
    idm_set = Set(strip.(filter(!isempty ∘ strip, readlines(idmfile))))
    defs = [parse_dfn(joinpath(dfndir, f), idm_set) for f in files]
    io = IOBuffer()
    println(io, "# GENERATED by src/input/dfn/generate.jl from ", basename(normpath(dfndir)), "; do not edit by hand.")
    println(io, "# ", length(defs), " definitions: ", join((d.name for d in defs), ", "))
    println(io)
    println(io, "\"\"\"")
    println(io, "    DFN_SOURCE")
    println(io)
    println(io, "The DFN directory the definitions were generated from.")
    println(io, "\"\"\"")
    println(io, "const DFN_SOURCE = ", lit(basename(normpath(dfndir))))
    names = String[]
    for d in defs
        println(io)
        push!(names, emit_definition(io, d))
    end
    println(io)
    println(io, "_dfn_definitions() = DfnDefinition[")
    for n in names
        println(io, "    ", n, ",")
    end
    println(io, "]")
    write(OUTPUT, take!(io))
    nvar = sum(d -> sum(b -> length(b.variables), d.blocks; init = 0), defs)
    println("Wrote $OUTPUT: $(length(defs)) definitions, $nvar variables")
    noblk = [d.name for d in defs if isempty(d.blocks)]
    isempty(noblk) || println("Without blocks: ", join(noblk, ", "))
    noopt = [d.name for d in defs if !any(b -> b.name == "OPTIONS", d.blocks)]
    isempty(noopt) || println("Without OPTIONS block: ", join(noopt, ", "))
    return nothing
end

main(ARGS)
