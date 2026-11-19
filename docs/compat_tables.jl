# Tables of the "MODFLOW 6 compatibility" pages, built when the manual is built (included
# by docs/make.jl; the pages call them from `@eval` blocks as `Main.CompatTables`):
#
# - the supported packages and options, from the strict-loading tables of
#   src/input/supported.jl, so the page cannot drift from the code;
# - the parity table, from docs/src/assets/parity.toml (docs/parity/collect.jl).
#
# Every name in those tables needs a description below; a missing one fails the build.
module CompatTables

using Markdown: Markdown
using Printf: @sprintf
using TOML: TOML
using ModflowGPU: ModflowGPU

const M = ModflowGPU

# --- Supported packages and options

"Order and user-facing names of the input files in `SUPPORTED_OPTIONS`."
const FILE_NAMES = [
    "sim-nam" => ("Simulation name file", "`mfsim.nam`"),
    "sim-tdis" => ("Time discretisation", "TDIS6"),
    "sln-ims" => ("Iterative model solution", "IMS6"),
    "gwf-nam" => ("GWF model name file", "GWF6"),
    "gwf-dis" => ("Structured grid", "DIS6"),
    "gwf-disv" => ("Vertex grid", "DISV6"),
    "gwf-disu" => ("Unstructured grid", "DISU6"),
    "gwf-ic" => ("Initial conditions", "IC6"),
    "gwf-npf" => ("Node property flow", "NPF6"),
    "gwf-sto" => ("Storage", "STO6"),
    "gwf-oc" => ("Output control", "OC6"),
    "gwf-chd" => ("Constant head", "CHD6"),
    "gwf-wel" => ("Well", "WEL6"),
    "gwf-drn" => ("Drain", "DRN6"),
    "gwf-riv" => ("River", "RIV6"),
    "gwf-ghb" => ("General head boundary", "GHB6"),
    "gwf-rch" => ("Recharge (list input)", "RCH6"),
    "gwf-rcha" => ("Recharge (`READASARRAYS`)", "RCH6"),
    "gwf-evt" => ("Evapotranspiration (list input)", "EVT6"),
    "gwf-evta" => ("Evapotranspiration (`READASARRAYS`)", "EVT6"),
]

"What the features of `SUPPORTED_FEATURES` mean for a user."
const FEATURE_NAMES = Dict(
    :timeseries => "Time series (TS6 files, list values given as time-series names) and time-array series (TAS6 files for RCH and EVT with `READASARRAYS`)",
    :disv => "Vertex grids (DISV6)",
    :disu => "Unstructured grids (DISU6)",
    :convertible => "Convertible cells (NPF `ICELLTYPE` ≠ 0), with wetting and drying (`REWET`, `WETDRY`)",
    :convertible_storage => "Convertible storage (STO `ICONVERT` ≠ 0, specific yield) in transient periods",
    :newton => "Newton–Raphson formulation (`NEWTON` and `NEWTON UNDER_RELAXATION` in the model name file)",
)

"Why each package or model of `UNSUPPORTED_PACKAGES` and `UNSUPPORTED_MODELS` is rejected."
const REJECTED = Dict(
    "SFR6" => ("Streamflow routing", "out of scope"),
    "LAK6" => ("Lakes", "out of scope"),
    "MAW6" => ("Multi-aquifer wells", "out of scope"),
    "UZF6" => ("Unsaturated zone flow", "out of scope"),
    "MVR6" => ("Water mover", "out of scope"),
    "CSUB6" => ("Skeletal storage, compaction and subsidence", "out of scope"),
    "BUY6" => ("Buoyancy (variable density)", "out of scope"),
    "VSC6" => ("Viscosity", "out of scope"),
    "API6" => ("Application programming interface package", "out of scope"),
    "GNC6" => ("Ghost-node correction", "not planned at the moment"),
    "HFB6" => ("Horizontal flow barrier", "not planned at the moment"),
    "GWT6" => ("Groundwater transport model", "out of scope"),
    "GWE6" => ("Groundwater energy transport model", "out of scope"),
    "PRT6" => ("Particle tracking model", "out of scope"),
    "CHF6" => ("Channel flow model", "out of scope"),
    "OLF6" => ("Overland flow model", "out of scope"),
)

_md(s) = Markdown.parse(s)

function _order(dfn)
    i = findfirst(p -> first(p) == dfn, FILE_NAMES)
    isnothing(i) && error("docs/compat_tables.jl: add \"$dfn\" (a key of SUPPORTED_OPTIONS) to FILE_NAMES")
    return i
end

"The option with its accepted values, if `SUPPORTED_VALUES` restricts them."
function _option(dfn, tag)
    vals = get(M.SUPPORTED_VALUES, (dfn, tag), nothing)
    return isnothing(vals) ? "`$tag`" : "`$tag` (" * join(("`$v`" for v in vals), ", ") * ")"
end

"""
    options_table()

One row per input file and block: the options that strict loading accepts
(`SUPPORTED_OPTIONS`, with the values of `SUPPORTED_VALUES`).
"""
function options_table()
    for (dfn, tag) in keys(M.SUPPORTED_VALUES)
        any(k -> first(k) == dfn && tag in M.SUPPORTED_OPTIONS[k], keys(M.SUPPORTED_OPTIONS)) ||
            error("docs/compat_tables.jl: SUPPORTED_VALUES has ($dfn, $tag), which is in no SUPPORTED_OPTIONS block")
    end
    io = IOBuffer()
    println(io, "| Input | File type | Block | Accepted options |")
    println(io, "| :-- | :-- | :-- | :-- |")
    for (dfn, block) in sort!(collect(keys(M.SUPPORTED_OPTIONS)); by = k -> (_order(k[1]), k[2] != "OPTIONS", k[2]))
        name, ftype = last(FILE_NAMES[_order(dfn)])
        tags = M.SUPPORTED_OPTIONS[(dfn, block)]
        opts = isempty(tags) ? "(none)" : join((_option(dfn, t) for t in tags), ", ")
        println(io, "| $name | $ftype | $block | $opts |")
    end
    return _md(String(take!(io)))
end

"""
    features_table()

The model features beyond options that strict loading accepts (`SUPPORTED_FEATURES`).
"""
function features_table()
    io = IOBuffer()
    println(io, "| Feature | Supported |")
    println(io, "| :-- | :-- |")
    for f in sort!(collect(M.SUPPORTED_FEATURES))
        haskey(FEATURE_NAMES, f) || error("docs/compat_tables.jl: add a description of feature :$f to FEATURE_NAMES")
        println(io, "| ", FEATURE_NAMES[f], " | yes |")
    end
    return _md(String(take!(io)))
end

"""
    rejected_table()

The packages and models that are always rejected (`UNSUPPORTED_PACKAGES`,
`UNSUPPORTED_MODELS`), with the reason.
"""
function rejected_table()
    io = IOBuffer()
    println(io, "| File type | What it is | Status |")
    println(io, "| :-- | :-- | :-- |")
    for t in [collect(M.UNSUPPORTED_MODELS); collect(M.UNSUPPORTED_PACKAGES)]
        haskey(REJECTED, t) || error("docs/compat_tables.jl: add $t (UNSUPPORTED_PACKAGES or UNSUPPORTED_MODELS) to REJECTED")
        what, status = REJECTED[t]
        println(io, "| $t | $what | $status |")
    end
    return _md(String(take!(io)))
end

# --- Parity table

const PARITY_FILE = joinpath(@__DIR__, "src", "assets", "parity.toml")

parity_data() = TOML.parsefile(PARITY_FILE)

_e(x) = x == 0 ? "0" : @sprintf("%.1e", x)
_backend(b) = b == "cpu" ? "CPU" : b == "rocm" ? "ROCm" : b

"Max of field `f` over the backends and the presets `ps` of a row."
function _max(row, ps, f)
    v = 0.0
    for (_, res) in row["results"], p in ps
        haskey(res, p) || continue
        v = max(v, f(res[p]))
    end
    return v
end

_budget(r) = max(r["max_budget_cbc"], r["max_budget_listing"])

# `_e(_max(...))`, or "–" when no backend ran the presets `ps` (T6.2: ex-gwf-nwt-p03b has no `:single`).
_ep(row, ps, f) = any(haskey(res, p) for (_, res) in row["results"] for p in ps) ? _e(_max(row, ps, f)) : "–"

function _time(row, b)
    haskey(row["results"], b) || return "–"
    return @sprintf("%.2f", row["results"][b]["mixed"]["wall_time"])
end

"""
    parity_source()

The line that says where the parity table comes from (date, commit, mf6 version).
"""
function parity_source()
    d = parity_data()
    n = length(d["examples"])
    mins = round(d["wall_time"] / 60; digits = 0)
    return _md(
        "Collected on $(d["date"]) at commit `$(d["commit"])` against MODFLOW $(d["mf6_version"]): " *
            "$n models, presets " * join(("`:$p`" for p in d["presets"]), ", ") *
            ", backends " * join(_backend.(d["backends"]), " and ") *
            " (AMD Radeon RX 7800 XT), $(Int(mins)) minutes in total.",
    )
end

"""
    parity_table()

One row per model of docs/src/assets/parity.toml: grid, packages, the largest head and
budget differences against MF6 over the backends (`:double` and `:mixed` together, and
`:single`), the presets and backends tested, and the wall time of the `:mixed` run.
"""
function parity_table()
    d = parity_data()
    io = IOBuffer()
    println(io, "| Model | Grid (active cells) | Steps | Packages | max \\|Δh\\| (m), `:double`/`:mixed` | max \\|Δh\\| (m), `:single` | Budget, `:double`/`:mixed` | Budget, `:single` | Presets | Backends | Time `:mixed` (s), CPU / ROCm |")
    println(io, "| :-- | :-- | --: | :-- | --: | --: | --: | --: | :-- | :-- | --: |")
    dm = ("double", "mixed")
    for row in d["examples"]
        name = row["name"]
        label = row["passed"] ? "`$name`" : "`$name` (failed)"
        grid = "$(row["grid"]) ($(row["active_cells"]))"
        pk = join(row["packages"], ", ")
        presets = sort(unique(p for (_, r) in row["results"] for p in ("single", "mixed", "double") if haskey(r, p)); by = p -> findfirst(==(p), ("single", "mixed", "double")))
        backends = join(_backend.(sort(collect(keys(row["results"])))), ", ")
        println(
            io, "| $label | $grid | $(row["time_steps"]) | $pk | ",
            _e(_max(row, dm, r -> r["max_dh"])), " | ", _ep(row, ("single",), r -> r["max_dh"]), " | ",
            _e(_max(row, dm, _budget)), " | ", _ep(row, ("single",), _budget), " | ",
            join(("`:$p`" for p in presets), ", "), " | $backends | ", _time(row, "cpu"), " / ", _time(row, "rocm"), " |",
        )
    end
    return _md(String(take!(io)))
end

"""
    parity_backend_table()

The largest head difference between the ROCm and the CPU backend per preset, over all
models, with the criterion.
"""
function parity_backend_table()
    d = parity_data()
    io = IOBuffer()
    println(io, "| Preset | Largest \\|Δh\\| ROCm vs CPU (m) | Criterion |")
    println(io, "| :-- | --: | :-- |")
    for (p, crit) in (("double", "1e-10 × the largest \\|head\\|"), ("mixed", "1e-10 × the largest \\|head\\|"), ("single", "1e-4 m"))
        v = 0.0
        for row in d["examples"], (_, r) in row["results"]
            haskey(r, p) && haskey(r[p], "max_dh_vs_cpu") && (v = max(v, r[p]["max_dh_vs_cpu"]))
        end
        println(io, "| `:$p` | ", _e(v), " | $crit |")
    end
    return _md(String(take!(io)))
end

end # module CompatTables
