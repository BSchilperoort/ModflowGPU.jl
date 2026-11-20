# Host-side model description: simulation, timing, solutions, models and exchanges (plain
# structs, no device arrays; rules in the header of src/spec/grid.jl).

"""
    TdisSpec

Time discretization (TDIS6): per stress period the length `perlen`, number of time steps
`nstp::Vector{Int32}` and step multiplier `tsmult`. `time_units` is `:undefined`,
`:seconds`, `:minutes`, `:hours`, `:days` or `:years`; `start_date_time` is as given
(`""` when not given); `ats_file` is the ATS6 FILEIN name (`""` when none). Simulation
time stays Float64 on the host (PLAN.md section 3.7, rule 4).
"""
Base.@kwdef struct TdisSpec
    perlen::Vector{Float64}
    nstp::Vector{Int32}
    tsmult::Vector{Float64}
    time_units::Symbol = :undefined
    start_date_time::String = ""
    ats_file::String = ""
    file::String = ""
end

"""
    nper(t::TdisSpec) -> Int

The number of stress periods.
"""
nper(t::TdisSpec) = length(t.perlen)

"""
    ImsSpec

One IMS6 solution: the file's settings with MF6's COMPLEXITY defaults filled in
(`NumericalSolution.f90` `sln_setouter`, `ImsLinearSettings.f90` `preset_config`), and
the models it solves (`models`, upper case, in `mfsim.nam` order).

- Options: `print_option` (`:none` (default), `:summary`, `:all`), `complexity`
  (`:simple` (default), `:moderate`, `:complex`), `csv_outer_output`, `csv_inner_output`
  (FILEOUT names), `no_ptc` (`:none` when not given, `:all` or `:first`),
  `ats_outer_maximum_fraction` (`NaN` when not given).
- Nonlinear: `outer_dvclose`, `outer_maximum`, `under_relaxation` (`:none`, `:simple`,
  `:cooley`, `:dbd`), `under_relaxation_gamma`, `_theta`, `_kappa`, `_momentum`,
  `backtracking_number`, `backtracking_tolerance`, `backtracking_reduction_factor`,
  `backtracking_residual_limit`. As in MF6, `under_relaxation_theta` is at least 1e-3.
- Linear: `inner_maximum`, `inner_dvclose`, `inner_rclose`, `rclose_option`
  (`:infinity_norm` when INNER_RCLOSE has no keyword, `:strict`, `:l2norm`, `:relative`,
  `:l2norm_relative`; `ims_base_testcnvg` options 0-4), `linear_acceleration` (`:cg`,
  `:bicgstab`), `relaxation_factor`, `preconditioner_levels`,
  `preconditioner_drop_tolerance`, `number_orthogonalizations`, `scaling_method`
  (`:none`, `:diagonal`, `:l2norm`), `reordering_method` (`:none`, `:rcm`, `:md`).
- `given`: the keywords set in the file (upper case), to tell explicit values from
  COMPLEXITY defaults.

The keyword constructor's defaults are those of COMPLEXITY SIMPLE: `ImsSpec(complexity =
:complex)` does not apply the COMPLEX presets; start from `ims_defaults(:complex)`.
"""
Base.@kwdef struct ImsSpec
    models::Vector{String} = String[]
    print_option::Symbol = :none
    complexity::Symbol = :simple
    csv_outer_output::String = ""
    csv_inner_output::String = ""
    no_ptc::Symbol = :none
    ats_outer_maximum_fraction::Float64 = NaN
    outer_dvclose::Float64 = 1.0e-3
    outer_maximum::Int32 = 25
    under_relaxation::Symbol = :none
    under_relaxation_gamma::Float64 = 1.0
    under_relaxation_theta::Float64 = 1.0
    under_relaxation_kappa::Float64 = 0.0
    under_relaxation_momentum::Float64 = 0.0
    backtracking_number::Int32 = 0
    backtracking_tolerance::Float64 = 0.0
    backtracking_reduction_factor::Float64 = 0.0
    backtracking_residual_limit::Float64 = 0.0
    inner_maximum::Int32 = 50
    inner_dvclose::Float64 = 1.0e-3
    inner_rclose::Float64 = 0.1
    rclose_option::Symbol = :infinity_norm
    linear_acceleration::Symbol = :cg
    relaxation_factor::Float64 = 0.0
    preconditioner_levels::Int32 = 0
    preconditioner_drop_tolerance::Float64 = 0.0
    number_orthogonalizations::Int32 = 0
    scaling_method::Symbol = :none
    reordering_method::Symbol = :none
    given::Vector{String} = String[]
    file::String = ""
end

"""
    ims_defaults(complexity) -> ImsSpec

An `ImsSpec` with MF6's defaults for `complexity` (`:simple`, `:moderate`, `:complex`):
`NumericalSolution.f90` `sln_setouter` and `ImsLinearSettings.f90` `preset_config`.
"""
function ims_defaults(complexity::Symbol)
    if complexity === :simple
        return ImsSpec(; complexity)
    elseif complexity === :moderate
        return ImsSpec(;
            complexity, outer_dvclose = 1.0e-2, outer_maximum = 50, under_relaxation = :dbd,
            under_relaxation_gamma = 0.0, under_relaxation_theta = 0.9, under_relaxation_kappa = 1.0e-4,
            inner_maximum = 100, linear_acceleration = :bicgstab, inner_dvclose = 1.0e-2,
            relaxation_factor = 0.97,
        )
    elseif complexity === :complex
        return ImsSpec(;
            complexity, outer_dvclose = 1.0e-1, outer_maximum = 100, under_relaxation = :dbd,
            under_relaxation_gamma = 0.0, under_relaxation_theta = 0.8, under_relaxation_kappa = 1.0e-4,
            backtracking_number = 20, backtracking_tolerance = 1.05, backtracking_reduction_factor = 0.1,
            backtracking_residual_limit = 0.002, inner_maximum = 500, linear_acceleration = :bicgstab,
            inner_dvclose = 1.0e-1, preconditioner_levels = 5, preconditioner_drop_tolerance = 1.0e-4,
            number_orthogonalizations = 2,
        )
    end
    throw(ArgumentError("unknown IMS complexity $complexity (expected :simple, :moderate or :complex)"))
end

"""
    SolutionGroupSpec

One SOLUTIONGROUP of `mfsim.nam`: `mxiter` (MXITER, 1 by default) and its solutions.
"""
Base.@kwdef struct SolutionGroupSpec
    mxiter::Int32 = 1
    solutions::Vector{ImsSpec} = ImsSpec[]
end

"""
    PackageEntry(ftype, file, name)

One line of a model name file's PACKAGES block: file type (`"WEL6"`), file name (as
written) and package name (upper case; MF6's default when not given).
"""
struct PackageEntry
    ftype::String
    file::String
    name::String
end

"""
    GwfSpec

One GWF6 model.

- `name` (upper case), `file` (the model name file).
- Name file options: `list_file`, `print_input`, `print_flows`, `save_flows`, `newton`,
  `newton_under_relaxation` (NEWTON UNDER_RELAXATION).
- `packages`: the PACKAGES block entries in file order (`PackageEntry`).
- `dis` (`DisSpec`, `DisvSpec` or `DisuSpec`), `npf`, `ic`, `sto` (`nothing` without STO:
  every period is steady state), `oc` (`nothing` without OC).
- `boundaries`: the stress packages, in MF6's order (by type: WEL, DRN, RIV, GHB, RCH,
  EVT, CHD, as in `gwf.f90` `GWF_MULTIPKG`, then in name file order within a type). MF6
  formulates and reports them in this order.
- `obs_files`: OBS6 files of the model (the model-level OBS6 packages; package OBS6
  files are in the package specs); `obs`: the model's OBS6 file read (`ObsSpec`, T6.7),
  or `nothing`.
"""
Base.@kwdef struct GwfSpec
    name::String
    list_file::String = ""
    print_input::Bool = false
    print_flows::Bool = false
    save_flows::Bool = false
    newton::Bool = false
    newton_under_relaxation::Bool = false
    packages::Vector{PackageEntry} = PackageEntry[]
    dis::AbstractDisSpec
    npf::NpfSpec
    ic::IcSpec
    sto::Union{Nothing, StoSpec} = nothing
    oc::Union{Nothing, OcSpec} = nothing
    boundaries::Vector{BoundarySpec} = BoundarySpec[]
    obs_files::Vector{String} = String[]
    obs::Union{Nothing, ObsSpec} = nothing
    file::String = ""
end

Base.show(io::IO, m::GwfSpec) = print(
    io, "GwfSpec(\"", m.name, "\", ", nameof(typeof(m.dis)), ", boundaries: ",
    join((b.name for b in m.boundaries), ", "), ")",
)

"""
    ExchangeSpec

A GWF6-GWF6 exchange between models `model1` and `model2` (upper case; the same name for
an exchange of a model with itself, such as `ex-gwf-spbc`).

- Options: `auxnames`, `boundnames`, `print_input`, `print_flows`, `save_flows`,
  `cell_averaging` (`:harmonic` (default), `:logarithmic`, `:amt_lmk`), `variablecv`,
  `dewatered`, `newton`, `xt3d`, `obs_files` (OBS6 FILEIN). GNC6 and MVR6 are rejected.
- `obs`: the OBS6 file read (`ObsSpec`, owner `:GWFGWF`: `FLOW-JA-FACE` by exchange number or
  boundary name; T6.6), or `nothing`.
- `data::ExchangeData`: the EXCHANGEDATA (`read_exchange_data`), cellids as user nodes of
  each model; cells outside the active domain have been rejected.
"""
Base.@kwdef struct ExchangeSpec
    exgtype::String = "GWF6-GWF6"
    model1::String
    model2::String
    auxnames::Vector{String} = String[]
    boundnames::Bool = false
    print_input::Bool = false
    print_flows::Bool = false
    save_flows::Bool = false
    cell_averaging::Symbol = :harmonic
    variablecv::Bool = false
    dewatered::Bool = false
    newton::Bool = false
    xt3d::Bool = false
    obs_files::Vector{String} = String[]
    obs::Union{Nothing, ObsSpec} = nothing
    data::ExchangeData
    file::String = ""
end

"""
    SimulationSpec

A whole simulation: `simdir` (the directory file names are relative to; `""` for specs
built in memory), the `mfsim.nam` options (`continue_on_nonconvergence` (CONTINUE),
`nocheck`, `memory_print_option`, `profile_option`, `maxerrors` (0 = MF6 default),
`print_input`), `tdis`, the GWF `models` (in `mfsim.nam` order), `exchanges` and
`solution_groups`. Look up a model with `model_spec(spec, name)`.

A plain host-side description in Float64, with no device data. It is made by
[`load_simulation`](@ref) (from MF6 input files) or [`build_simulation`](@ref) (from
Julia arrays), and taken by [`run_simulation`](@ref), [`initialize`](@ref) and
[`write_simulation`](@ref). The specs of its parts (`TdisSpec`, `GwfSpec`, `DisSpec`,
...) are not part of the public API; read them, but build them with the builder
functions.

# Example
```julia
spec = load_simulation("path/to/simulation")
spec.models[1].name, spec.tdis.perlen
```
"""
Base.@kwdef struct SimulationSpec
    simdir::String = ""
    continue_on_nonconvergence::Bool = false
    nocheck::Bool = false
    memory_print_option::String = ""
    profile_option::String = ""
    maxerrors::Int32 = 0
    print_input::Bool = false
    tdis::TdisSpec
    models::Vector{GwfSpec}
    exchanges::Vector{ExchangeSpec} = ExchangeSpec[]
    solution_groups::Vector{SolutionGroupSpec}
end

Base.show(io::IO, s::SimulationSpec) = print(
    io, "SimulationSpec(", length(s.models), " model(s): ", join((m.name for m in s.models), ", "),
    "; ", nper(s.tdis), " period(s); ", length(s.exchanges), " exchange(s))",
)

"""
    model_spec(spec::SimulationSpec, name) -> GwfSpec

The model `name` (case-insensitive).
"""
function model_spec(spec::SimulationSpec, name::AbstractString)
    i = findfirst(m -> m.name == uppercase(name), spec.models)
    isnothing(i) && throw(ArgumentError("no model \"$name\" in the simulation (models: $(join((m.name for m in spec.models), ", ")))"))
    return spec.models[i]
end
