# MF6 input files from a `SimulationSpec` (for parity runs of built models with mf6).
#
# Generalised from ModflowCoupling's `mf6_writer.jl` (`fmt`, `write_array`, `write_block`,
# `write_list_package`, `write_simulation`): the same file layout (one file per package,
# `<model>.<ext>`, INTERNAL arrays with one record per model row, LAYERED 3-D arrays,
# shortest round-trip decimals), but driven by the spec instead of ModflowCoupling's
# fixed package set, so that `load_simulation(write_simulation(dir, spec))` gives `spec`
# back (`spec_identical`, apart from `simdir`). Arrays whose values are all bitwise equal
# are written as CONSTANT.
#
# Scope ("small specs"): one or more GWF models on DIS, DISV or DISU grids (T6.9) with the
# packages of `BOUNDARY_KINDS`, STO, OC, OBS6 (T6.9), TDIS, IMS and GWF6-GWF6 exchanges
# (T6.9, with their OBS6). Features that need other extra files (time series, TAS, TVK, TVS,
# ATS) are an `ArgumentError`.

"Shortest round-trip decimal representation, readable by Fortran (ModflowCoupling `fmt`)."
function _mf6_fmt(x::Real)
    x isa Integer && return string(x)
    isfinite(x) || throw(ArgumentError("write_simulation: non-finite value $x in MODFLOW input"))
    return repr(Float64(x))
end

function _write_block(f, io, name)
    println(io, "BEGIN ", name)
    f()
    println(io, "END ", name, "\n")
    return nothing
end

_put(io, words...) = println(io, "  ", join(words, " "))

const _VALUES_PER_LINE = 10

# One 2-D layer `(ncol, nrow)` (or a 1-D array): CONSTANT when all values are bitwise
# equal, else INTERNAL with each row on its own lines (MF6 reads a row per record).
function _write_layer(io, a::AbstractArray)
    if all(x -> isequal(x, first(a)), a)
        println(io, "    CONSTANT ", _mf6_fmt(first(a)))
        return nothing
    end
    println(io, "    INTERNAL")
    rows = ndims(a) == 1 ? (a,) : eachcol(a)
    for row in rows, chunk in Iterators.partition(row, _VALUES_PER_LINE)
        println(io, "      ", join(_mf6_fmt.(chunk), " "))
    end
    return nothing
end

# A 3-D grid array `(ncol, nrow, nlay)`, LAYERED (ModflowCoupling `write_array`).
function _write_array(io, name, a::AbstractArray{<:Any, 3})
    println(io, "  ", name, " LAYERED")
    foreach(layer -> _write_layer(io, layer), eachslice(a; dims = 3))
    return nothing
end

_write_node_array(io, name, a::AbstractVector, dis::DisSpec) =
    _write_array(io, name, reshape(a, Int(dis.ncol), Int(dis.nrow), Int(dis.nlay)))

# T6.9: DISV arrays `(ncpl, nlay)`, LAYERED; DISU arrays by node.
function _write_node_array(io, name, a::AbstractVector, dis::DisvSpec)
    println(io, "  ", name, " LAYERED")
    foreach(layer -> _write_layer(io, layer), eachcol(reshape(a, Int(dis.ncpl), Int(dis.nlay))))
    return nothing
end

function _write_node_array(io, name, a::AbstractVector, ::DisuSpec)
    println(io, "  ", name)
    _write_layer(io, a)
    return nothing
end

# The cells of one layer of a READASARRAYS array (`(ncol, nrow)` on DIS, `ncpl` on DISV).
_layer_shape(a, dis::DisSpec) = reshape(a, Int(dis.ncol), Int(dis.nrow))
_layer_shape(a, ::DisvSpec) = a

function _check_writable(cond, what)
    cond || throw(ArgumentError("write_simulation: $what is not supported by the writer"))
    return nothing
end

_bool_options(io, x, fields) = foreach(f -> getfield(x, f) && _put(io, uppercase(String(f))), fields)

# ---------------------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------------------

const _LENGTH_UNIT_WORDS = Dict(:feet => "FEET", :meters => "METERS", :centimeters => "CENTIMETERS")

function _write_dis(io, d::DisSpec)
    _write_block(io, "OPTIONS") do
        d.length_units === :undefined || _put(io, "LENGTH_UNITS", _LENGTH_UNIT_WORDS[d.length_units])
        d.nogrb && _put(io, "NOGRB")
        isempty(d.grb_file) || _put(io, "GRB6 FILEOUT", d.grb_file)
        d.xorigin === 0.0 || _put(io, "XORIGIN", _mf6_fmt(d.xorigin))
        d.yorigin === 0.0 || _put(io, "YORIGIN", _mf6_fmt(d.yorigin))
        d.angrot === 0.0 || _put(io, "ANGROT", _mf6_fmt(d.angrot))
    end
    _write_block(io, "DIMENSIONS") do
        _put(io, "NLAY", d.nlay)
        _put(io, "NROW", d.nrow)
        _put(io, "NCOL", d.ncol)
    end
    _write_block(io, "GRIDDATA") do
        println(io, "  DELR")
        _write_layer(io, d.delr)
        println(io, "  DELC")
        _write_layer(io, d.delc)
        println(io, "  TOP")
        _write_layer(io, d.top)
        _write_array(io, "BOTM", d.botm)
        _write_array(io, "IDOMAIN", d.idomain)
    end
    return nothing
end

function _write_grid_options(io, d::AbstractDisSpec)
    d.length_units === :undefined || _put(io, "LENGTH_UNITS", _LENGTH_UNIT_WORDS[d.length_units])
    d.nogrb && _put(io, "NOGRB")
    isempty(d.grb_file) || _put(io, "GRB6 FILEOUT", d.grb_file)
    d.xorigin === 0.0 || _put(io, "XORIGIN", _mf6_fmt(d.xorigin))
    d.yorigin === 0.0 || _put(io, "YORIGIN", _mf6_fmt(d.yorigin))
    d.angrot === 0.0 || _put(io, "ANGROT", _mf6_fmt(d.angrot))
    return nothing
end

# VERTICES and CELL2D (Disv.f90 / Disu.f90 `source_vertices`, `source_cell2d`).
function _write_vertices(io, d::Union{DisvSpec, DisuSpec})
    _write_block(io, "VERTICES") do
        for iv in 1:size(d.vertices, 2)
            _put(io, iv, _mf6_fmt(d.vertices[1, iv]), _mf6_fmt(d.vertices[2, iv]))
        end
    end
    _write_block(io, "CELL2D") do
        for (j, iv) in enumerate(d.cell_vertices)
            _put(io, j, _mf6_fmt(d.cell_center[1, j]), _mf6_fmt(d.cell_center[2, j]), length(iv), iv...)
        end
    end
    return nothing
end

# T6.9: DISV6 (Disv.f90 input).
function _write_dis(io, d::DisvSpec)
    _write_block(() -> _write_grid_options(io, d), io, "OPTIONS")
    _write_block(io, "DIMENSIONS") do
        _put(io, "NLAY", d.nlay)
        _put(io, "NCPL", d.ncpl)
        _put(io, "NVERT", d.nvert)
    end
    _write_block(io, "GRIDDATA") do
        println(io, "  TOP")
        _write_layer(io, d.top)
        _write_node_array(io, "BOTM", vec(d.botm), d)
        _write_node_array(io, "IDOMAIN", vec(d.idomain), d)
    end
    _write_vertices(io, d)
    return nothing
end

# T6.9: DISU6 (Disu.f90 input; CONNECTIONDATA as given, user node numbers).
function _write_dis(io, d::DisuSpec)
    _write_block(io, "OPTIONS") do
        _write_grid_options(io, d)
        d.vertical_offset_tolerance === 0.0 || _put(io, "VERTICAL_OFFSET_TOLERANCE", _mf6_fmt(d.vertical_offset_tolerance))
    end
    _write_block(io, "DIMENSIONS") do
        _put(io, "NODES", d.nodes)
        _put(io, "NJA", d.nja)
        d.nvert > 0 && _put(io, "NVERT", d.nvert)
    end
    _write_block(io, "GRIDDATA") do
        for (name, a) in (("TOP", d.top), ("BOT", d.bot), ("AREA", d.area), ("IDOMAIN", d.idomain))
            _write_node_array(io, name, a, d)
        end
    end
    _write_block(io, "CONNECTIONDATA") do
        for (name, a) in (("IAC", d.iac), ("JA", d.ja), ("IHC", d.ihc), ("CL12", d.cl12), ("HWVA", d.hwva))
            _write_node_array(io, name, a, d)
        end
        isnothing(d.angldegx) || _write_node_array(io, "ANGLDEGX", d.angldegx, d)
    end
    d.nvert > 0 && _write_vertices(io, d)
    return nothing
end

const _CELL_AVERAGING_WORDS = Dict(:logarithmic => "LOGARITHMIC", :amt_lmk => "AMT-LMK", :amt_hmk => "AMT-HMK")

function _write_npf(io, n::NpfSpec, dis)
    _check_writable(isempty(n.tvk_files), "NPF TVK6")
    _write_block(io, "OPTIONS") do
        n.cell_averaging === :harmonic || _put(io, "ALTERNATIVE_CELL_AVERAGING", _CELL_AVERAGING_WORDS[n.cell_averaging])
        _bool_options(io, n, (:thickstrt,))
        n.variablecv && _put(io, "VARIABLECV", n.dewatered ? "DEWATERED" : "")
        _bool_options(io, n, (:perched,))
        n.rewet && _put(io, "REWET WETFCT", _mf6_fmt(n.wetfct), "IWETIT", n.iwetit, "IHDWET", n.ihdwet)
        n.xt3d && _put(io, "XT3D", n.xt3d_rhs ? "RHS" : "")
        _bool_options(io, n, (:highest_cell_saturation, :k22overk, :k33overk, :save_flows, :print_flows, :save_specific_discharge, :save_saturation))
    end
    _write_block(io, "GRIDDATA") do
        _write_node_array(io, "ICELLTYPE", n.icelltype, dis)
        _write_node_array(io, "K", n.k, dis)
        for f in (:k22, :k33, :angle1, :angle2, :angle3, :wetdry)
            a = getfield(n, f)
            isnothing(a) || _write_node_array(io, uppercase(String(f)), a, dis)
        end
    end
    return nothing
end

function _write_ic(io, ic::IcSpec, dis)
    _write_block(() -> nothing, io, "OPTIONS")
    _write_block(() -> _write_node_array(io, "STRT", ic.strt, dis), io, "GRIDDATA")
    return nothing
end

function _write_sto(io, s::StoSpec, dis)
    _check_writable(isempty(s.tvs_files), "STO TVS6")
    _write_block(() -> _bool_options(io, s, (:storagecoefficient, :ss_confined_only, :save_flows)), io, "OPTIONS")
    _write_block(io, "GRIDDATA") do
        _write_node_array(io, "ICONVERT", s.iconvert, dis)
        _write_node_array(io, "SS", s.ss, dis)
        isnothing(s.sy) || _write_node_array(io, "SY", s.sy, dis)
    end
    for (k, v) in zip(s.periods.iper, s.periods.data)
        _write_block(() -> _put(io, v === :steady_state ? "STEADY-STATE" : "TRANSIENT"), io, "PERIOD $k")
    end
    return nothing
end

function _oc_text(r::OcRecord)
    w = [uppercase(String(r.action)), r.rtype, uppercase(String(r.setting))]
    append!(w, string.(r.steps))
    return join(w, " ")
end

function _write_oc(io, o::OcSpec)
    _write_block(io, "OPTIONS") do
        isempty(o.budget_file) || _put(io, "BUDGET FILEOUT", o.budget_file)
        isempty(o.budgetcsv_file) || _put(io, "BUDGETCSV FILEOUT", o.budgetcsv_file)
        isempty(o.head_file) || _put(io, "HEAD FILEOUT", o.head_file)
        pf = o.head_print_format
        isnothing(pf) || _put(io, "HEAD PRINT_FORMAT COLUMNS", pf.columns, "WIDTH", pf.width, "DIGITS", pf.digits, pf.format)
    end
    for (k, recs) in zip(o.periods.iper, o.periods.data)
        _write_block(() -> foreach(r -> _put(io, _oc_text(r)), recs), io, "PERIOD $k")
    end
    return nothing
end

function _write_boundary_options(io, b::BoundarySpec)
    _check_writable(isempty(b.ts_files) && isempty(b.tas_files), "time-series input (package $(b.name))")
    _check_writable(b.auto_flow_reduce == 0 && !b.mover, "WEL AUTO_FLOW_REDUCE / MOVER (package $(b.name))")
    _write_block(io, "OPTIONS") do
        b.readasarrays && _put(io, "READASARRAYS")
        b.fixed_cell && _put(io, "FIXED_CELL")
        isempty(b.auxnames) || _put(io, "AUXILIARY", b.auxnames...)
        isempty(b.auxmultname) || _put(io, "AUXMULTNAME", b.auxmultname)
        isempty(b.auxdepthname) || _put(io, "AUXDEPTHNAME", b.auxdepthname)
        _bool_options(io, b, (:boundnames, :print_input, :print_flows, :save_flows, :surf_rate_specified))
        foreach(f -> _put(io, "OBS6 FILEIN", f), b.obs_files)
    end
    return nothing
end

function _write_list_boundary(io, b::BoundarySpec)
    s = b.schema
    _write_boundary_options(io, b)
    _write_block(io, "DIMENSIONS") do
        _put(io, "MAXBOUND", b.maxbound)
        boundary_kind(b) === :EVT && _put(io, "NSEG", b.nseg)
    end
    for (k, p) in zip(b.periods.iper, b.periods.data)
        _check_writable(isempty(p.ts), "time-series values (package $(b.name))")
        _write_block(io, "PERIOD $k") do
            for r in eachindex(p.nodeu)
                w = String[]
                for c in s.columns
                    if c.kind === :cellid
                        append!(w, string.(cellid(s.grid, p.nodeu[r])))
                    elseif c.kind === :double
                        append!(w, _mf6_fmt.(view(p.bound, c.first:(c.first + c.width - 1), r)))
                    elseif c.kind === :aux
                        append!(w, _mf6_fmt.(view(p.auxvar, c.first:(c.first + c.width - 1), r)))
                    elseif c.kind === :boundname
                        isempty(p.boundname[r]) || push!(w, p.boundname[r])
                    end
                end
                _put(io, w...)
            end
        end
    end
    return nothing
end

# T6.11: HFB6 (two cellids and HYDCHR per record).
function _write_hfb(io, h::HfbSpec, dis::AbstractDisSpec)
    g = grid_dims(dis)
    _write_block(() -> h.print_input && _put(io, "PRINT_INPUT"), io, "OPTIONS")
    _write_block(() -> _put(io, "MAXHFB", h.maxhfb), io, "DIMENSIONS")
    for (k, p) in zip(h.periods.iper, h.periods.data)
        _write_block(io, "PERIOD $k") do
            for r in eachindex(p.cell1)
                _put(io, string.(cellid(g, p.cell1[r]))..., string.(cellid(g, p.cell2[r]))..., _mf6_fmt(p.hydchr[r]))
            end
        end
    end
    return nothing
end

function _write_array_boundary(io, b::BoundarySpec, dis::Union{DisSpec, DisvSpec})
    _write_boundary_options(io, b)
    layername = boundary_kind(b) === :RCH ? "irch" : "ievt"
    for (k, p) in zip(b.periods.iper, b.periods.data)
        _check_writable(isempty(p.tas), "TIMEARRAYSERIES input (package $(b.name))")
        _write_block(io, "PERIOD $k") do
            for name in p.given
                iaux = findfirst(==(name), b.auxnames)
                a = !isnothing(iaux) ? p.auxvar[iaux, :] : name == layername ? p.layer : p.values[name]
                println(io, "  ", uppercase(name))
                _write_layer(io, _layer_shape(a, dis))
            end
        end
    end
    return nothing
end

# T6.9: OBS6 (Obs.f90 input): DIGITS, PRINT_INPUT, then one CONTINUOUS block per output
# file with each observation's name, type and ID string as given.
function _write_obs(io, o::ObsSpec)
    _write_block(io, "OPTIONS") do
        o.digits == 0 || _put(io, "DIGITS", o.digits)
        o.print_input && _put(io, "PRINT_INPUT")
    end
    for out in o.outputs
        _write_block(io, "CONTINUOUS FILEOUT $(out.file)$(out.binary ? " BINARY" : "")") do
            foreach(e -> _put(io, e.name, e.type, e.idstring), out.observations)
        end
    end
    return nothing
end

# T6.9: a GWF6-GWF6 exchange file (exg-gwfgwf.f90 input).
function _write_exchange(io, e::ExchangeSpec, m1::GwfSpec, m2::GwfSpec)
    d = e.data
    _write_block(io, "OPTIONS") do
        isempty(e.auxnames) || _put(io, "AUXILIARY", e.auxnames...)
        _bool_options(io, e, (:boundnames, :print_input, :print_flows, :save_flows))
        e.cell_averaging === :harmonic || _put(io, "CELL_AVERAGING", _CELL_AVERAGING_WORDS[e.cell_averaging])
        e.variablecv && _put(io, "VARIABLECV", e.dewatered ? "DEWATERED" : "")
        _bool_options(io, e, (:newton, :xt3d))
        foreach(f -> _put(io, "OBS6 FILEIN", f), e.obs_files)
    end
    _write_block(() -> _put(io, "NEXG", length(d.nodem1)), io, "DIMENSIONS")
    g1, g2 = grid_dims(m1.dis), grid_dims(m2.dis)
    _write_block(io, "EXCHANGEDATA") do
        for i in eachindex(d.nodem1)
            w = String[string.(cellid(g1, d.nodem1[i]))..., string.(cellid(g2, d.nodem2[i]))..., string(d.ihc[i])]
            append!(w, _mf6_fmt.((d.cl1[i], d.cl2[i], d.hwva[i])))
            append!(w, _mf6_fmt.(view(d.auxvar, :, i)))
            (e.boundnames && i <= length(d.boundname) && !isempty(d.boundname[i])) && push!(w, d.boundname[i])
            _put(io, w...)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------------------
# Name files, TDIS, IMS
# ---------------------------------------------------------------------------------------

function _write_model_nam(io, m::GwfSpec)
    _write_block(io, "OPTIONS") do
        isempty(m.list_file) || _put(io, "LIST", m.list_file)
        _bool_options(io, m, (:print_input, :print_flows, :save_flows))
        m.newton && _put(io, "NEWTON", m.newton_under_relaxation ? "UNDER_RELAXATION" : "")
    end
    _write_block(() -> foreach(e -> _put(io, e.ftype, e.file, e.name), m.packages), io, "PACKAGES")
    return nothing
end

const _TIME_UNIT_WORDS = Dict(:seconds => "SECONDS", :minutes => "MINUTES", :hours => "HOURS", :days => "DAYS", :years => "YEARS")

function _write_tdis(io, t::TdisSpec)
    _check_writable(isempty(t.ats_file), "ATS6")
    _write_block(io, "OPTIONS") do
        t.time_units === :undefined || _put(io, "TIME_UNITS", _TIME_UNIT_WORDS[t.time_units])
        isempty(t.start_date_time) || _put(io, "START_DATE_TIME", t.start_date_time)
    end
    _write_block(() -> _put(io, "NPER", nper(t)), io, "DIMENSIONS")
    _write_block(io, "PERIODDATA") do
        for k in 1:nper(t)
            _put(io, _mf6_fmt(t.perlen[k]), t.nstp[k], _mf6_fmt(t.tsmult[k]))
        end
    end
    return nothing
end

const _IMS_BLOCK = Dict(
    f => (
        f in (:print_option, :complexity, :csv_outer_output, :csv_inner_output, :no_ptc, :ats_outer_maximum_fraction) ? "OPTIONS" :
            startswith(String(f), "outer_") || startswith(String(f), "under_") || startswith(String(f), "backtracking_") ? "NONLINEAR" : "LINEAR"
    )
        for f in fieldnames(ImsSpec) if !(f in _IMS_NOT_KEYWORDS)
)

const _RCLOSE_WORDS = Dict(:infinity_norm => "", :strict => "STRICT", :l2norm => "L2NORM_RCLOSE", :relative => "RELATIVE_RCLOSE", :l2norm_relative => "L2NORM_RELATIVE_RCLOSE")

function _ims_value(s::ImsSpec, f::Symbol)
    v = getfield(s, f)
    f === :inner_rclose && return strip(_mf6_fmt(v) * " " * _RCLOSE_WORDS[s.rclose_option])
    f in (:csv_outer_output, :csv_inner_output) && return "FILEOUT $v"
    v isa Symbol && return uppercase(String(v))
    return _mf6_fmt(v)
end

# The keywords given, plus every setting that differs from the COMPLEXITY defaults (in
# field order, which is block order).
function _write_ims(io, s::ImsSpec)
    d = ims_defaults(s.complexity)
    fields = Symbol[]
    for f in fieldnames(ImsSpec)
        f in _IMS_NOT_KEYWORDS && continue
        g = f === :rclose_option ? :inner_rclose : f
        g in fields && continue
        differs = !isequal(getfield(s, f), getfield(d, f))
        (_ims_keyword(f) in s.given || differs || (f === :complexity && s.complexity !== :simple)) && push!(fields, g)
    end
    for block in ("OPTIONS", "NONLINEAR", "LINEAR")
        _write_block(io, block) do
            for f in fields
                _IMS_BLOCK[f] == block && _put(io, _ims_keyword(f), _ims_value(s, f))
            end
        end
    end
    return nothing
end

function _write_mfsim(io, spec::SimulationSpec)
    _write_block(io, "OPTIONS") do
        spec.continue_on_nonconvergence && _put(io, "CONTINUE")
        spec.nocheck && _put(io, "NOCHECK")
        isempty(spec.memory_print_option) || _put(io, "MEMORY_PRINT_OPTION", spec.memory_print_option)
        isempty(spec.profile_option) || _put(io, "PROFILE_OPTION", spec.profile_option)
        spec.maxerrors == 0 || _put(io, "MAXERRORS", spec.maxerrors)
        spec.print_input && _put(io, "PRINT_INPUT")
    end
    _write_block(() -> _put(io, "TDIS6", spec.tdis.file), io, "TIMING")
    _write_block(() -> foreach(m -> _put(io, "GWF6", m.file, m.name), spec.models), io, "MODELS")
    _write_block(() -> foreach(e -> _put(io, e.exgtype, e.file, e.model1, e.model2), spec.exchanges), io, "EXCHANGES")
    for (g, sg) in enumerate(spec.solution_groups)
        _write_block(io, "SOLUTIONGROUP $g") do
            sg.mxiter == 1 || _put(io, "MXITER", sg.mxiter)
            foreach(s -> _put(io, "IMS6", s.file, s.models...), sg.solutions)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------------------

function _write_file(f, dir, name)
    isempty(name) && throw(ArgumentError("write_simulation: a package has no file name (build it with build_model / build_simulation)"))
    isabspath(name) && throw(ArgumentError("write_simulation: absolute file name $name"))
    path = joinpath(dir, name)
    mkpath(dirname(path))
    open(f, path, "w")
    return path
end

"""
    write_simulation(dir, spec::SimulationSpec) -> dir

Write `spec` as MF6 input files into `dir` (created when missing): `mfsim.nam` and every
file under the name in its spec (`file` fields, `PackageEntry`s), as `build_model` and
`build_simulation` set them. `load_simulation(dir)` gives `spec` back
(`spec_identical(spec, load_simulation(dir); ignore = (:simdir,))`), and mf6 runs it.

Supported: GWF models on DIS, DISV and DISU grids with NPF, IC, STO, OC, OBS6, HFB6 (T6.11),
the stress packages of `BOUNDARY_KINDS` (lists, and READASARRAYS on DIS and DISV), TDIS, IMS and
GWF6-GWF6 exchanges (with their OBS6). Time series, TAS, TVK, TVS and ATS input are an
`ArgumentError`. Ported from ModflowCoupling's `write_simulation` (src/mf6_writer.jl
there).

# Example
```jldoctest; setup = :(dis = build_dis(; delr = 100.0, delc = 100.0, top = 0.0, botm = fill(-10.0, 10, 5, 1)); chd = build_boundary(:CHD, dis; periods = [1 => (; cells = [(1, 1, 1), (1, 5, 10)], values = [0.0, -2.0])]); sim = build_simulation(build_model("flow"; dis, npf = build_npf(dis; k = 10.0), ic = build_ic(dis; strt = 0.0), boundaries = [chd]); tdis = build_tdis(; perlen = 1.0)))
julia> dir = write_simulation(mktempdir(), sim);

julia> readdir(dir)
8-element Vector{String}:
 "flow.chd-1"
 "flow.dis"
 "flow.ic"
 "flow.ims"
 "flow.nam"
 "flow.npf"
 "flow.tdis"
 "mfsim.nam"

julia> load_simulation(dir)
SimulationSpec(1 model(s): FLOW; 1 period(s); 0 exchange(s))
```
"""
function write_simulation(dir::AbstractString, spec::SimulationSpec)
    mkpath(dir)
    open(io -> _write_mfsim(io, spec), joinpath(dir, "mfsim.nam"), "w")
    _write_file(io -> _write_tdis(io, spec.tdis), dir, spec.tdis.file)
    for sg in spec.solution_groups, s in sg.solutions
        _write_file(io -> _write_ims(io, s), dir, s.file)
    end
    for m in spec.models
        dis = m.dis
        _write_file(io -> _write_model_nam(io, m), dir, m.file)
        isnothing(m.obs) || _write_file(io -> _write_obs(io, m.obs), dir, only(m.obs_files))
        _write_file(io -> _write_dis(io, dis), dir, dis.file)
        _write_file(io -> _write_npf(io, m.npf, dis), dir, m.npf.file)
        _write_file(io -> _write_ic(io, m.ic, dis), dir, m.ic.file)
        isnothing(m.sto) || _write_file(io -> _write_sto(io, m.sto, dis), dir, m.sto.file)
        isnothing(m.oc) || _write_file(io -> _write_oc(io, m.oc), dir, m.oc.file)
        isnothing(m.hfb) || _write_file(io -> _write_hfb(io, m.hfb, dis), dir, m.hfb.file)
        for b in m.boundaries
            b.readasarrays && _check_writable(!(dis isa DisuSpec), "READASARRAYS on DISU (package $(b.name))")
            _write_file(dir, b.file) do io
                b.readasarrays ? _write_array_boundary(io, b, dis) : _write_list_boundary(io, b)
            end
            isnothing(b.obs) || _write_file(io -> _write_obs(io, b.obs), dir, only(b.obs_files))
        end
    end
    byname = Dict(m.name => m for m in spec.models)
    for e in spec.exchanges
        _write_file(io -> _write_exchange(io, e, byname[e.model1], byname[e.model2]), dir, e.file)
        isnothing(e.obs) || _write_file(io -> _write_obs(io, e.obs), dir, only(e.obs_files))
    end
    return dir
end
