# Host-side model description: GWF packages (plain structs, no device arrays; rules in the
# header of src/spec/grid.jl).
#
# Every package spec holds the values MF6 uses after reading its input, with MF6's
# defaults filled in, plus `file` (the input file, relative to the simulation directory;
# `""` for specs built in memory). Arrays not given in the input are `nothing` where MF6
# distinguishes "not given" (NPF K22/K33/ANGLE*/WETDRY, STO SY); MF6 fills them later
# (K22 = K33 = K, angles zero), which is the engine's job (T2.2), so that the spec keeps
# what the user specified.

"""
    TvPeriod(nodeu, var, value, ts)

The settings of one PERIOD block of TVK6 or TVS6 (T6.12), in input order: the user node
`nodeu`, the property `var` (`:K`, `:K22`, `:K33` for TVK; `:SS`, `:SY` for TVS), the value
`value` (`NaN` when linked to a time series) and `ts`, the name of the time series
(upper case) or `Symbol("")` for a literal value.
"""
struct TvPeriod
    nodeu::Vector{Int32}
    var::Vector{Symbol}
    value::Vector{Float64}
    ts::Vector{Symbol}
end

"""
    TvSpec

Time-varying hydraulic conductivity (TVK6, `kind = :TVK`, an option of NPF6) or storage
(TVS6, `kind = :TVS`, an option of STO6), T6.12 (TvBase.f90, gwf-tvk.f90, gwf-tvs.f90):
`print_input`, `disable_storage_change_integration` (TVS only), `ts_files` (TS6 FILEIN
names; with any, every value is re-evaluated each time step, `ts_active`), `timeseries`
(the series of the TS6 files by name), `periods::StressPeriods{TvPeriod}` and `file`.
A setting holds from its period until a later block sets the same property of the cell.
"""
Base.@kwdef struct TvSpec
    kind::Symbol
    print_input::Bool = false
    disable_storage_change_integration::Bool = false
    ts_files::Vector{String} = String[]
    timeseries::Dict{Symbol, TimeSeries} = Dict{Symbol, TimeSeries}()
    periods::StressPeriods{TvPeriod}
    file::String = ""
end

Base.show(io::IO, t::TvSpec) = print(io, "TvSpec(:", t.kind, ", periods ", t.periods.iper, ")")

"""
    NpfSpec

Node property flow (NPF6). Arrays are by user node (`(nodes)`).

- `icelltype::Vector{Int32}`, `k::Vector{Float64}`; `k22`, `k33`, `angle1`, `angle2`,
  `angle3`, `wetdry`: `Vector{Float64}` or `nothing` when not given.
- `cell_averaging`: `:harmonic` (default), `:logarithmic`, `:amt_lmk` or `:amt_hmk`
  (ALTERNATIVE_CELL_AVERAGING).
- `thickstrt`, `variablecv`, `dewatered`, `perched`, `k22overk`, `k33overk`,
  `highest_cell_saturation`: the options of the same name.
- `rewet` with `wetfct`, `iwetit`, `ihdwet` (REWET WETFCT .. IWETIT .. IHDWET ..; MF6
  defaults 1.0, 1, 0 when REWET is not given).
- `xt3d`, `xt3d_rhs` (XT3D [RHS]).
- `save_flows`, `print_flows`, `save_specific_discharge`, `save_saturation`.
- `tvk_files`: TVK6 FILEIN names (time-varying K); `tvk`: the TVK6 file read (`TvSpec`,
  T6.12), or `nothing`.
"""
Base.@kwdef struct NpfSpec
    icelltype::Vector{Int32}
    k::Vector{Float64}
    k22::Union{Nothing, Vector{Float64}} = nothing
    k33::Union{Nothing, Vector{Float64}} = nothing
    angle1::Union{Nothing, Vector{Float64}} = nothing
    angle2::Union{Nothing, Vector{Float64}} = nothing
    angle3::Union{Nothing, Vector{Float64}} = nothing
    wetdry::Union{Nothing, Vector{Float64}} = nothing
    cell_averaging::Symbol = :harmonic
    thickstrt::Bool = false
    variablecv::Bool = false
    dewatered::Bool = false
    perched::Bool = false
    rewet::Bool = false
    wetfct::Float64 = 1.0
    iwetit::Int32 = 1
    ihdwet::Int32 = 0
    xt3d::Bool = false
    xt3d_rhs::Bool = false
    highest_cell_saturation::Bool = false
    k22overk::Bool = false
    k33overk::Bool = false
    save_flows::Bool = false
    print_flows::Bool = false
    save_specific_discharge::Bool = false
    save_saturation::Bool = false
    tvk_files::Vector{String} = String[]
    tvk::Union{Nothing, TvSpec} = nothing
    file::String = ""
end

"""
    StoSpec

Storage (STO6). Arrays are by user node.

- `iconvert::Vector{Int32}`, `ss::Vector{Float64}`; `sy` or `nothing` when not given.
- `storagecoefficient`, `ss_confined_only`, `save_flows`.
- `periods::StressPeriods{Symbol}`: `:steady_state` / `:transient` from the PERIOD blocks;
  before the first block a model with STO is transient (`is_transient`).
- `tvs_files`: TVS6 FILEIN names; `tvs`: the TVS6 file read (`TvSpec`, T6.12), or
  `nothing`.
"""
Base.@kwdef struct StoSpec
    iconvert::Vector{Int32}
    ss::Vector{Float64}
    sy::Union{Nothing, Vector{Float64}} = nothing
    storagecoefficient::Bool = false
    ss_confined_only::Bool = false
    save_flows::Bool = false
    periods::StressPeriods{Symbol} = StressPeriods(Int[], Symbol[])
    tvs_files::Vector{String} = String[]
    tvs::Union{Nothing, TvSpec} = nothing
    file::String = ""
end

"""
    is_transient(sto::Union{Nothing, StoSpec}, kper) -> Bool

Whether stress period `kper` is transient: without STO every period is steady state;
with STO the last PERIOD block at or before `kper` decides, and the default is transient
(`gwf-sto.f90`).
"""
is_transient(::Nothing, kper::Integer) = false
function is_transient(sto::StoSpec, kper::Integer)
    s = period_data(sto.periods, kper)
    return isnothing(s) || s === :transient
end

"""
    IcSpec

Initial conditions (IC6): `strt::Vector{Float64}` by user node.
"""
Base.@kwdef struct IcSpec
    strt::Vector{Float64}
    file::String = ""
end

"""
    PrintFormat(columns, width, digits, format)

OC `HEAD PRINT_FORMAT COLUMNS c WIDTH w DIGITS d {EXPONENTIAL|FIXED|GENERAL|SCIENTIFIC}`.
"""
struct PrintFormat
    columns::Int32
    width::Int32
    digits::Int32
    format::String
end

"""
    OcSpec

Output control (OC6).

- `head_file`, `budget_file`, `budgetcsv_file`: FILEOUT names (`""` when not given).
- `head_print_format`: `PrintFormat` or `nothing`.
- `periods::StressPeriods{Vector{OcRecord}}`: the PERIOD blocks (`read_oc_periods`); record
  types are checked (`HEAD`, `BUDGET`).
"""
Base.@kwdef struct OcSpec
    head_file::String = ""
    budget_file::String = ""
    budgetcsv_file::String = ""
    head_print_format::Union{Nothing, PrintFormat} = nothing
    periods::StressPeriods{Vector{OcRecord}} = StressPeriods(Int[], Vector{OcRecord}[])
    file::String = ""
end

"""
    BOUNDARY_KINDS

The stress packages a `BoundarySpec{Kind}` can hold: `:CHD`, `:WEL`, `:RCH`, `:GHB`,
`:RIV`, `:DRN`, `:EVT`.
"""
const BOUNDARY_KINDS = (:CHD, :WEL, :RCH, :GHB, :RIV, :DRN, :EVT)

"""
    BoundarySpec{Kind}

A stress package of kind `Kind` (one of `BOUNDARY_KINDS`; `boundary_kind(b)`). Period
data are those of T1.3 in user node numbers; cellids in inactive cells have been rejected
by the loader ("Cell is outside active grid domain"), and the conversion to reduced nodes
is T2.1/T2.3's.

- `name`: package name (upper case; MF6's default is `"WEL-1"`, `"WEL-2"`, ... for
  unnamed stress packages); `file`: input file.
- `readasarrays`: READASARRAYS input (RCH, EVT): `periods` holds `ArrayPeriod`s and
  `schema` is `nothing`; otherwise `periods` holds `PeriodList`s and `schema` is their
  `ListSchema` (column layout of `bound`).
- `auxnames` (upper case), `auxmultname` and `auxmultcol` (its index in `auxnames`, 0 when
  none), `auxdepthname` (DRN), `boundnames`.
- `fixed_cell` (RCH, EVT: apply to the given cell even when it is inactive, instead of
  the highest active cell below; the logic is T2.3's), `nseg` and `surf_rate_specified`
  (EVT).
- `maxbound` (0 for READASARRAYS), `print_input`, `print_flows`, `save_flows`.
- `auto_flow_reduce` (WEL; 0 = off), `flow_reduction_length`, `auto_flow_reduce_auxname`,
  `afrcsv_file`; `mover` (MVR, out of scope; always rejected).
- `ts_files`, `tas_files`, `obs_files`: TS6, TAS6 and OBS6 FILEIN names; `obs`: the
  OBS6 file read (`ObsSpec`, T6.7), or `nothing`. List values given as time-series names are `TsRef`s in the period data,
  READASARRAYS arrays given as `TIMEARRAYSERIES name` are in `ArrayPeriod.tas`.
- `timeseries`: the series of the TS6 files by name (T6.3; a later file's series replaces
  one of the same name, as MF6's hash table does); `timearrayseries`: the series of the
  TAS6 files by name (T6.3). Both are evaluated per time step by the boundary framework
  (`advance_time_series!`).
"""
Base.@kwdef struct BoundarySpec{Kind}
    name::String
    readasarrays::Bool = false
    auxnames::Vector{String} = String[]
    auxmultname::String = ""
    auxmultcol::Int32 = 0
    auxdepthname::String = ""
    boundnames::Bool = false
    fixed_cell::Bool = false
    nseg::Int32 = 1
    surf_rate_specified::Bool = false
    maxbound::Int32 = 0
    print_input::Bool = false
    print_flows::Bool = false
    save_flows::Bool = false
    auto_flow_reduce::Float64 = 0.0
    flow_reduction_length::Bool = false
    auto_flow_reduce_auxname::String = ""
    afrcsv_file::String = ""
    mover::Bool = false
    ts_files::Vector{String} = String[]
    tas_files::Vector{String} = String[]
    obs_files::Vector{String} = String[]
    obs::Union{Nothing, ObsSpec} = nothing
    timeseries::Dict{Symbol, TimeSeries} = Dict{Symbol, TimeSeries}()
    timearrayseries::Dict{Symbol, TimeArraySeries} = Dict{Symbol, TimeArraySeries}()
    schema::Union{Nothing, ListSchema} = nothing
    periods::Union{StressPeriods{PeriodList}, StressPeriods{ArrayPeriod}}
    file::String = ""
end

"""
    boundary_kind(b::BoundarySpec) -> Symbol

The kind of a stress package (`:CHD`, `:WEL`, ...).
"""
boundary_kind(::BoundarySpec{K}) where {K} = K

const ChdSpec = BoundarySpec{:CHD}
const WelSpec = BoundarySpec{:WEL}
const RchSpec = BoundarySpec{:RCH}
const GhbSpec = BoundarySpec{:GHB}
const RivSpec = BoundarySpec{:RIV}
const DrnSpec = BoundarySpec{:DRN}
const EvtSpec = BoundarySpec{:EVT}

function Base.show(io::IO, b::BoundarySpec{K}) where {K}
    return print(io, "BoundarySpec{:", K, "}(\"", b.name, "\", ", b.readasarrays ? "arrays" : "list", ", periods ", b.periods.iper, ")")
end

"""
    HfbPeriod(cell1, cell2, hydchr)

The barriers of one HFB6 PERIOD block (T6.11), in input order: the two cells of each
barrier as user node numbers (`cell1`, `cell2`, `Int32`) and the hydraulic characteristic
`hydchr` (barrier K over its width; zero or negative: a multiplier `-hydchr` of the
conductance, gwf-hfb.f90).
"""
struct HfbPeriod
    cell1::Vector{Int32}
    cell2::Vector{Int32}
    hydchr::Vector{Float64}
end

"""
    HfbSpec

Horizontal flow barriers (HFB6, T6.11): `name` (package name), `print_input`, `maxhfb`
(DIMENSIONS MAXHFB), `periods::StressPeriods{HfbPeriod}` (the PERIOD blocks; a block
replaces the barriers of the previous one, as `hfb_rp`) and `file`.
"""
Base.@kwdef struct HfbSpec
    name::String = "HFB"
    print_input::Bool = false
    maxhfb::Int
    periods::StressPeriods{HfbPeriod}
    file::String = ""
end

Base.show(io::IO, h::HfbSpec) = print(io, "HfbSpec(\"", h.name, "\", maxhfb ", h.maxhfb, ", periods ", h.periods.iper, ")")
