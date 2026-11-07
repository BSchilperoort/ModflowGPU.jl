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
# Scope ("small specs"): one or more GWF models on DIS grids with the packages of
# `BOUNDARY_KINDS`, STO, OC, TDIS and IMS. Features that need extra files (time series,
# TAS, OBS, TVK, TVS, ATS), DISV/DISU grids and exchanges are an `ArgumentError`.

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
    _check_writable(isempty(b.obs_files), "OBS6 input (package $(b.name))")
    _check_writable(b.auto_flow_reduce == 0 && !b.mover, "WEL AUTO_FLOW_REDUCE / MOVER (package $(b.name))")
    _write_block(io, "OPTIONS") do
        b.readasarrays && _put(io, "READASARRAYS")
        b.fixed_cell && _put(io, "FIXED_CELL")
        isempty(b.auxnames) || _put(io, "AUXILIARY", b.auxnames...)
        isempty(b.auxmultname) || _put(io, "AUXMULTNAME", b.auxmultname)
        isempty(b.auxdepthname) || _put(io, "AUXDEPTHNAME", b.auxdepthname)
        _bool_options(io, b, (:boundnames, :print_input, :print_flows, :save_flows, :surf_rate_specified))
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

function _write_array_boundary(io, b::BoundarySpec, dis::DisSpec)
    _write_boundary_options(io, b)
    layername = boundary_kind(b) === :RCH ? "irch" : "ievt"
    for (k, p) in zip(b.periods.iper, b.periods.data)
        _check_writable(isempty(p.tas), "TIMEARRAYSERIES input (package $(b.name))")
        _write_block(io, "PERIOD $k") do
            for name in p.given
                iaux = findfirst(==(name), b.auxnames)
                a = !isnothing(iaux) ? p.auxvar[iaux, :] : name == layername ? p.layer : p.values[name]
                println(io, "  ", uppercase(name))
                _write_layer(io, reshape(a, Int(dis.ncol), Int(dis.nrow)))
            end
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
    _write_block(() -> nothing, io, "EXCHANGES")
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

Supported: GWF models on DIS grids with NPF, IC, STO, OC, the stress packages of
`BOUNDARY_KINDS` (lists and READASARRAYS), TDIS and IMS. Exchanges, DISV/DISU, time
series, TAS, OBS, TVK, TVS and ATS input are an `ArgumentError`. Ported from
ModflowCoupling's `write_simulation` (src/mf6_writer.jl there).
"""
function write_simulation(dir::AbstractString, spec::SimulationSpec)
    _check_writable(isempty(spec.exchanges), "GWF6-GWF6 exchanges")
    mkpath(dir)
    open(io -> _write_mfsim(io, spec), joinpath(dir, "mfsim.nam"), "w")
    _write_file(io -> _write_tdis(io, spec.tdis), dir, spec.tdis.file)
    for sg in spec.solution_groups, s in sg.solutions
        _write_file(io -> _write_ims(io, s), dir, s.file)
    end
    for m in spec.models
        dis = m.dis
        _check_writable(dis isa DisSpec, "$(nameof(typeof(dis))) (model $(m.name))")
        _check_writable(isempty(m.obs_files), "OBS6 input (model $(m.name))")
        _write_file(io -> _write_model_nam(io, m), dir, m.file)
        _write_file(io -> _write_dis(io, dis), dir, dis.file)
        _write_file(io -> _write_npf(io, m.npf, dis), dir, m.npf.file)
        _write_file(io -> _write_ic(io, m.ic, dis), dir, m.ic.file)
        isnothing(m.sto) || _write_file(io -> _write_sto(io, m.sto, dis), dir, m.sto.file)
        isnothing(m.oc) || _write_file(io -> _write_oc(io, m.oc), dir, m.oc.file)
        for b in m.boundaries
            _write_file(dir, b.file) do io
                b.readasarrays ? _write_array_boundary(io, b, dis) : _write_list_boundary(io, b)
            end
        end
    end
    return dir
end
