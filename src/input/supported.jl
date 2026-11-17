# What the engine supports: the central tables the loader checks input against (host
# only). Later cards extend these tables when they implement a feature.
#
# Two levels:
# 1. Out of scope (PLAN.md section 1): models other than GWF6, exchanges other than
#    GWF6-GWF6, EMS6 solutions, and the packages in `UNSUPPORTED_PACKAGES`. They are
#    always an error ("unsupported: SFR6 package 'SFR-1' in model 'GWF'"), also with
#    `strict = false`: there is no spec for them.
# 2. In scope but not implemented yet. `load_simulation(simdir; strict = true)` rejects
#    everything outside `SUPPORTED_OPTIONS`, `SUPPORTED_VALUES` and `SUPPORTED_FEATURES`
#    with an `MF6UnsupportedError` ("unsupported: <thing>"), so that no option is
#    silently ignored. With `strict = false` the loader reads all in-scope input
#    (Tiers 0-7) into the spec without these checks; tests and later cards use it to
#    prove the parser covers the higher tiers.
#
# "Supported" means what the Phase 2-3 cards implement (Milestone A: Tier 0-2 examples
# and the coupling feature set of PLAN.md section 1.1). A few options only change what is
# printed to the listing file (PRINT_INPUT, PRINT_FLOWS, PRINT_OPTION, HEAD PRINT_FORMAT,
# MEMORY_PRINT_OPTION, ...): they are accepted and recorded in the spec, and T2.8
# decides how much of that printing it reproduces. Output files are accepted and
# owned by later cards: DIS GRB6 and OC HEAD/BUDGET FILEOUT by T2.8, OBS6 files by T6.7.
# Time series (T6.3): TS6 files and list values given as time-series names (`TsRef`, in
# any column whose DFN entry allows a time series, auxiliary variables included), TAS6
# files (RCHA, EVTA: `TAS6 FILEIN`) and TIMEARRAYSERIES arrays are the `:timeseries`
# feature, supported since T6.3.

"""
    MF6UnsupportedError(file, msg)

Input that ModflowGPU does not support (yet), thrown by [`load_simulation`](@ref) (and
so by [`run_simulation`](@ref) and [`initialize`](@ref)) before anything runs. `msg`
starts with `"unsupported: "` and names the package, option or feature; `file` is the
input file (or `""`). Nothing unsupported is silently ignored.

# Example
```jldoctest
julia> e = MF6UnsupportedError("model.npf", "unsupported: NPF option XT3D");

julia> sprint(showerror, e)
"MF6UnsupportedError: unsupported: NPF option XT3D (model.npf)"
```
"""
struct MF6UnsupportedError <: Exception
    file::String
    msg::String
end

function Base.showerror(io::IO, e::MF6UnsupportedError)
    print(io, "MF6UnsupportedError: ", e.msg)
    return isempty(e.file) || print(io, " (", e.file, ")")
end

unsupported(file::AbstractString, what::AbstractString) =
    throw(MF6UnsupportedError(String(file), "unsupported: " * what))

"""
    UNSUPPORTED_PACKAGES

GWF package file types that are out of scope (PLAN.md section 1: SFR, LAK, MAW, UZF, MVR,
CSUB, BUY, VSC, API) or not in the plan (GNC, HFB): always an error.
"""
const UNSUPPORTED_PACKAGES = ("SFR6", "LAK6", "MAW6", "UZF6", "MVR6", "CSUB6", "BUY6", "VSC6", "API6", "GNC6", "HFB6")

"""
    UNSUPPORTED_MODELS

Model types that are out of scope: always an error.
"""
const UNSUPPORTED_MODELS = ("GWT6", "GWE6", "PRT6", "CHF6", "OLF6")

# Options every list stress package shares (gwf-chd, -wel, -drn, -riv, -ghb, -rch, -evt).
const _STRESS_OPTIONS = ["AUXILIARY", "AUX", "AUXMULTNAME", "BOUNDNAMES", "PRINT_INPUT", "PRINT_FLOWS", "SAVE_FLOWS", "TS6", "OBS6"]

"""
    SUPPORTED_OPTIONS

The option tags (first word of a line, upper case) accepted with `strict = true`, per
DFN name (`"gwf-npf"`) and block. Anything else in that block is
"unsupported: <ftype> option <TAG>". Extend this table when a card implements an option.
"""
const SUPPORTED_OPTIONS = Dict{Tuple{String, String}, Vector{String}}(
    ("sim-nam", "OPTIONS") => ["CONTINUE", "NOCHECK", "MEMORY_PRINT_OPTION", "PROFILE_OPTION", "MAXERRORS", "PRINT_INPUT"],
    ("sim-tdis", "OPTIONS") => ["TIME_UNITS", "START_DATE_TIME"],
    ("gwf-nam", "OPTIONS") => ["LIST", "PRINT_INPUT", "PRINT_FLOWS", "SAVE_FLOWS"],
    ("gwf-dis", "OPTIONS") => ["LENGTH_UNITS", "NOGRB", "GRB6", "XORIGIN", "YORIGIN", "ANGROT", "CRS"],
    # T6.4: vertex and unstructured grids.
    ("gwf-disv", "OPTIONS") => ["LENGTH_UNITS", "NOGRB", "GRB6", "XORIGIN", "YORIGIN", "ANGROT", "CRS"],
    ("gwf-disu", "OPTIONS") => ["LENGTH_UNITS", "NOGRB", "GRB6", "XORIGIN", "YORIGIN", "ANGROT", "VERTICAL_OFFSET_TOLERANCE", "CRS"],
    # T2.2: harmonic, LOGARITHMIC, AMT-LMK, AMT-HMK, K22/K33/ANGLE*; T2.8: specific discharge.
    ("gwf-npf", "OPTIONS") => [
        "SAVE_FLOWS", "PRINT_FLOWS", "ALTERNATIVE_CELL_AVERAGING", "SAVE_SPECIFIC_DISCHARGE", "K22OVERK", "K33OVERK",
        "THICKSTRT", "VARIABLECV", "DEWATERED", "PERCHED", "REWET", "WETFCT", "IWETIT", "IHDWET",  # T6.1
    ],
    ("gwf-sto", "OPTIONS") => ["SAVE_FLOWS", "STORAGECOEFFICIENT", "SS_CONFINED_ONLY"],
    ("gwf-ic", "OPTIONS") => String[],
    ("gwf-oc", "OPTIONS") => ["BUDGET", "HEAD"],
    ("gwf-chd", "OPTIONS") => _STRESS_OPTIONS,
    ("gwf-wel", "OPTIONS") => [_STRESS_OPTIONS; "AUTO_FLOW_REDUCE"; "FLOW_REDUCTION_LENGTH"; "AUTO_FLOW_REDUCE_AUXNAME"],  # T6.1
    ("gwf-ghb", "OPTIONS") => _STRESS_OPTIONS,
    ("gwf-riv", "OPTIONS") => _STRESS_OPTIONS,
    ("gwf-drn", "OPTIONS") => [_STRESS_OPTIONS; "AUXDEPTHNAME"],  # T2.4: DRN smoothing
    ("gwf-rch", "OPTIONS") => [_STRESS_OPTIONS; "FIXED_CELL"],
    ("gwf-rcha", "OPTIONS") => [_STRESS_OPTIONS; "FIXED_CELL"; "READASARRAYS"; "TAS6"],  # T6.3: TAS6
    ("gwf-evt", "OPTIONS") => [_STRESS_OPTIONS; "FIXED_CELL"; "SURF_RATE_SPECIFIED"],
    ("gwf-evta", "OPTIONS") => [_STRESS_OPTIONS; "FIXED_CELL"; "READASARRAYS"; "TAS6"],  # T6.3: TAS6
    ("sln-ims", "OPTIONS") => ["PRINT_OPTION", "COMPLEXITY"],
    ("sln-ims", "NONLINEAR") => [
        "OUTER_DVCLOSE", "OUTER_MAXIMUM", "UNDER_RELAXATION", "UNDER_RELAXATION_GAMMA",
        "UNDER_RELAXATION_THETA", "UNDER_RELAXATION_KAPPA", "UNDER_RELAXATION_MOMENTUM", "LINEAR_SOLVER",
        # Backtracking: implemented in T6.2; T2.8 warns. Accepted like the implicit
        # backtracking of COMPLEXITY COMPLEX (backtracking_number = 20).
        "BACKTRACKING_NUMBER", "BACKTRACKING_TOLERANCE", "BACKTRACKING_REDUCTION_FACTOR",
        "BACKTRACKING_RESIDUAL_LIMIT",
    ],
    ("sln-ims", "LINEAR") => [
        "INNER_MAXIMUM", "INNER_DVCLOSE", "INNER_RCLOSE", "LINEAR_ACCELERATION", "RELAXATION_FACTOR",
        "PRECONDITIONER_LEVELS", "PRECONDITIONER_DROP_TOLERANCE", "NUMBER_ORTHOGONALIZATIONS",
        "SCALING_METHOD", "REORDERING_METHOD", "RED_BLACK_ORDERING",
    ],
)

"""
    SUPPORTED_VALUES

Values accepted with `strict = true` for options whose value selects a method, per
`(DFN name, TAG)`. Other values are "unsupported: <ftype> <TAG> <value>".
"""
const SUPPORTED_VALUES = Dict{Tuple{String, String}, Vector{String}}(
    ("sln-ims", "UNDER_RELAXATION") => ["NONE", "SIMPLE", "COOLEY", "DBD"],  # T2.8
    ("sln-ims", "SCALING_METHOD") => ["NONE", "DIAGONAL"],                   # T2.7
    ("sln-ims", "REORDERING_METHOD") => ["NONE"],
    ("sln-ims", "LINEAR_SOLVER") => ["DEFAULT", "LINEAR"],
)

"""
    SUPPORTED_FEATURES

Model features beyond options accepted with `strict = true` (names as in the `requires`
field of `validation/examples.toml`). Supported: `:disv`, `:disu` (DISV6 / DISU6 grids,
T6.4), `:convertible` (NPF ICELLTYPE /= 0 in an active cell) and `:convertible_storage`
(STO ICONVERT /= 0 with a transient period), T6.1. Features not in the set are rejected:

- `:newton`: NEWTON in the model name file (T6.2);
- `:multi_model`: more than one GWF model, or more than one solution (T6.6);
- `:exchange`: GWF6-GWF6 exchanges (T6.6);
- `:timeseries`: list values given as TS6 time-series names, or TIMEARRAYSERIES arrays
  (supported since T6.3);

ATS6, TVK6, TVS6 and HPC6 files are options (`ATS6 FILEIN ...`) and are rejected
through `SUPPORTED_OPTIONS`.
"""
const SUPPORTED_FEATURES = Set{Symbol}([:timeseries, :disv, :disu, :convertible, :convertible_storage])

"""
    check_options(o::OptionSet, dfn_name, ftype, ctx)

With strict loading: throw "unsupported: <ftype> option <TAG> (<context>)" for the first
tag of `o` that is not in `SUPPORTED_OPTIONS[(dfn_name, o.block)]`.
"""
function check_options(o::OptionSet, dfn_name::AbstractString, ftype::AbstractString, ctx::AbstractString)
    isempty(o.tags) && return nothing
    ok = get(SUPPORTED_OPTIONS, (String(dfn_name), o.block), String[])
    for t in o.tags
        t in ok || unsupported(o.file, "$ftype option $t ($ctx)")
    end
    return nothing
end

"""
    check_value(file, dfn_name, ftype, tag, value, ctx)

With strict loading: throw "unsupported: <ftype> <TAG> <value>" when `value` is not in
`SUPPORTED_VALUES[(dfn_name, tag)]` (no entry: any value is supported).
"""
function check_value(file, dfn_name::AbstractString, ftype::AbstractString, tag::AbstractString, value::AbstractString, ctx::AbstractString)
    ok = get(SUPPORTED_VALUES, (String(dfn_name), String(tag)), nothing)
    (isnothing(ok) || uppercase(value) in ok) && return nothing
    return unsupported(file, "$ftype $tag $(uppercase(value)) ($ctx)")
end

"""
    check_feature(file, feature::Symbol, what)

With strict loading: throw "unsupported: <what>" when `feature` is not in
`SUPPORTED_FEATURES`.
"""
function check_feature(file, feature::Symbol, what::AbstractString)
    feature in SUPPORTED_FEATURES && return nothing
    return unsupported(file, what)
end
