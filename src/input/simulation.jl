# Simulation loader: MF6 input files → `SimulationSpec` (host only).
#
# Ports the loading of a simulation by MF6 6.8.2:
# - SimulationCreate.f90 (`options_create`, `timing_create`, `models_create`,
#   `check_model_name`, `exchanges_create`, `solution_groups_create`,
#   `solution_group_check`, `check_model_assignment`): mfsim.nam, model names, exchange
#   models, solution groups (consecutive numbers, MXITER, model membership).
# - tdis.f90 (`tdis_source_options`, `tdis_source_dimensions`, `tdis_source_timing`,
#   `check_tdis_timing`).
# - ModelPackageInputs.f90 (`modelpkgs_create`, `modelpkgs_pkgcount`,
#   `LoadablePackageType%add`) and gwf.f90 (`GWF_BASEPKG`, `GWF_MULTIPKG`,
#   `ftype_check`, `package_create`): name file package types, default package names
#   (`WEL-1`, ...), one instance of each base package, required DIS/IC/NPF, unique
#   stress package names, and the order of the stress packages (by type in
#   `GWF_MULTIPKG` order, then name file order).
# - The `source_*` routines of Dis.f90, Disv.f90, Disu.f90, gwf-npf.f90, gwf-sto.f90,
#   gwf-ic.f90, gwf-oc.f90 / OutputControl.f90 (`oc_rp` record types),
#   BoundaryPackageExt.f90 (`source_options`: AUXMULTNAME; `cellid_to_nlist`: "Cell is
#   outside active grid domain"), DisConnExchange.f90 (`source_data`) and the IMS readers
#   (NumericalSolution.f90 `sln_read_options`/`sln_setouter`, ImsLinearSettings.f90
#   `preset_config`/`read_from_file`), with their error messages.
#
# Design:
# - The spec stays in user node numbers. MF6 converts list cellids to reduced nodes when
#   it reads them and fails on a cell outside the active domain (IDOMAIN < 1); the loader
#   raises that error (with the record's file and line) but keeps user nodes: the
#   reduced numbering is made by T2.1, and T2.3 maps the period data. READASARRAYS
#   (RCHA/EVTA) data are not checked: MF6 drops inactive cells there (`nodeu_to_nlist`),
#   and FIXED_CELL / highest-active-cell handling is T2.3's.
# - PERIOD blocks numbered above NPER are never read by MF6 (`dynamic_read_ionper` stops
#   after the last period); the loader drops them with a warning before reading the
#   period data, so their content is not checked either.
# - Support levels and the `strict` keyword are described in src/input/supported.jl:
#   out-of-scope models, exchanges and packages are always an error; with
#   `strict = true` (the default) every option, option value and feature outside the
#   central tables (`SUPPORTED_OPTIONS`, `SUPPORTED_VALUES`, `SUPPORTED_FEATURES`) is an
#   `MF6UnsupportedError` "unsupported: <thing>"; with `strict = false` all in-scope
#   input is read into the spec.
# - Errors in the input are `MF6InputError`s naming the file (and line, where there is
#   one); input MF6 only warns about gives a warning.
#
# Differences from MF6 (documented):
# - Grid checks done at setup in MF6 (cell thickness, active nodes, DISU symmetry) are
#   done by `build_connectivity` (`check_dis_grid`; DISV/DISU in T6.4); missing
#   DELR/DELC/TOP/BOTM (DIS), TOP/BOTM (DISV) or TOP/BOT/AREA (DISU) arrays are an error
#   here (MF6 does not check that they were given and would run with unset values).
#   Value checks of NPF arrays (K <= 0, ...), done by MF6 in `prepcheck`, are done by
#   `npf_properties` (src/gwf/npf.jl, T2.2).
# - READARRAYGRID input (CHDG, WELG, ...) is rejected also with `strict = false`: there is
#   no period-data reader for it yet (an explicit exception to the strict/non-strict rule).
# - NPF option combinations rejected in `check_options` (NEWTON with PERCHED, XT3D with
#   ALTERNATIVE_CELL_AVERAGING, ...) are checked by `check_npf_options` (src/gwf/npf.jl,
#   T2.2), not here.
# - File references (TS6, TAS6, OBS6) must exist; their contents are read by T6.3/T6.7.

const LENMODELNAME = 16    # ConstantsModule
const LENPACKAGENAME = 16

# gwf.f90 GWF_BASEPKG / GWF_MULTIPKG (in scope; the rest is in UNSUPPORTED_PACKAGES).
const _GWF_BASE_FTYPES = ("DIS6", "DISV6", "DISU6", "NPF6", "STO6", "IC6", "OC6", "OBS6")
const _GWF_MULTI_FTYPES = ("WEL6", "DRN6", "RIV6", "GHB6", "RCH6", "EVT6", "CHD6")
const _GWF_ALL_FTYPES = (
    "DIS6", "DISV6", "DISU6", "NPF6", "BUY6", "VSC6", "GNC6", "HFB6", "STO6", "IC6", "CSUB6",
    "MVR6", "OC6", "OBS6", "WEL6", "DRN6", "RIV6", "GHB6", "RCH6", "EVT6", "CHD6", "MAW6", "SFR6",
    "LAK6", "UZF6", "API6",
)

# Loading context.
struct _Loader
    simdir::String
    strict::Bool
end

_rel(ld::_Loader, path) = relpath(path, ld.simdir)

function _read_package(ld::_Loader, fname::AbstractString, ftype::AbstractString, component::AbstractString)
    path = _resolve_path(fname, ld.simdir)
    isfile(path) || throw(MF6InputError(path, 0, "$ftype input file not found"))
    return read_input_file(path; ftype, component, simdir = ld.simdir)
end

# The PERIOD blocks MF6 reads: those numbered 1..nper (dynamic_read_ionper).
function _periods_upto(f::InputFile, nper::Integer)
    keep = InputBlock[]
    for b in f.blocks
        if b.name == "PERIOD" && block_number(b) > nper
            @warn "$(b.header.file):$(b.header.line): PERIOD $(block_number(b)) is ignored: the simulation has NPER = $nper stress periods (MF6 reads no PERIOD block after the last period)"
            continue
        end
        push!(keep, b)
    end
    return InputFile(f.path, f.dfn, keep)
end

# FILEIN names of an option set, checked to exist.
function _filein(ld::_Loader, o::OptionSet, name::AbstractString, tag::AbstractString)
    files = option(o, name, String[])::Vector{String}
    for fname in files
        path = _resolve_path(fname, ld.simdir)
        isfile(path) || input_error(option_line(o, tag), "$tag file \"$fname\" not found ($path)")
    end
    return copy(files)
end

function _required(arrays, name, file, block = "GRIDDATA")
    haskey(arrays, name) || throw(MF6InputError(file, 0, "Error in $block block: $(uppercase(name)) not found."))
    return arrays[name]
end

_opt_array(arrays, name) = haskey(arrays, name) ? vec(arrays[name]) : nothing

# ---------------------------------------------------------------------------------------
# TDIS
# ---------------------------------------------------------------------------------------

const _TIME_UNITS = ("UNKNOWN", "SECONDS", "MINUTES", "HOURS", "DAYS", "YEARS")
const _TIME_UNIT_SYMBOLS = (:undefined, :seconds, :minutes, :hours, :days, :years)

# tdis.f90 check_tdis_timing: the first problem only (it returns after each).
function _check_tdis_timing(path, perlen, nstp, tsmult)
    tstart = 0.0
    for kper in eachindex(perlen)
        nstp[kper] <= 0 && throw(MF6InputError(path, 0, "Number of time steps less than one  for stress period $kper"))
        if perlen[kper] == 0.0
            @warn "$path: PERLEN is zero for stress period $kper. PERLEN must not be zero for transient periods."
            return nothing
        end
        tsmult[kper] <= 0.0 && throw(MF6InputError(path, 0, "TSMULT must be greater than 0.0  for stress period $kper"))
        perlen[kper] < 0.0 && throw(MF6InputError(path, 0, "PERLEN cannot be less than 0.0  for stress period $kper"))
        dt = 0.0
        tend = tstart
        for kstp in 1:nstp[kper]
            if kstp == 1
                dt = perlen[kper] / nstp[kper]
                tsmult[kper] != 1.0 && (dt = perlen[kper] * (1.0 - tsmult[kper]) / (1.0 - tsmult[kper]^nstp[kper]))
            else
                dt *= tsmult[kper]
            end
            tend = tstart + dt
            tstart == tend && throw(MF6InputError(path, 0, "Time step length of $dt is too small in period $kper and time step $kstp"))
        end
        tstart = tend
    end
    return nothing
end

function _load_tdis(ld::_Loader, fname)
    f = _read_package(ld, fname, "TDIS6", "SIM")
    o = read_options(f, "OPTIONS")
    ld.strict && check_options(o, "sim-tdis", "TDIS6", "simulation")
    time_units = :undefined
    if has_option(o, "time_units")
        i = findfirst(==(o.values["time_units"]), _TIME_UNITS)
        isnothing(i) && input_error(option_line(o, "TIME_UNITS"), "Unrecognized input value for TIME_UNITS option.")
        time_units = _TIME_UNIT_SYMBOLS[i]
    end
    ats = _filein(ld, o, "ats6_filename", "ATS6")
    length(ats) > 1 && input_error(option_line(o, "ATS6"), "Multiple ATS6 FILEIN entries; only one is allowed")
    d = read_options(f, "DIMENSIONS")
    np = option(d, "nper", 1)
    np >= 1 || input_error(option_line(d, "NPER"), "NPER must be greater than zero")
    pd = read_tdis_perioddata(f)
    _check_tdis_timing(f.path, pd.perlen, pd.nstp, pd.tsmult)
    return TdisSpec(;
        perlen = pd.perlen, nstp = Int32.(pd.nstp), tsmult = pd.tsmult, time_units,
        start_date_time = option(o, "start_date_time", ""), ats_file = isempty(ats) ? "" : ats[1],
        file = _rel(ld, f.path),
    )
end

# ---------------------------------------------------------------------------------------
# Discretization
# ---------------------------------------------------------------------------------------

const _LENGTH_UNITS = ("FEET", "METERS", "CENTIMETERS")
const _LENGTH_UNIT_SYMBOLS = (:feet, :meters, :centimeters)

# Dis.f90 source_options: an unknown LENGTH_UNITS leaves the unit undefined (no error).
function _dis_options(o::OptionSet)
    lu = option(o, "length_units", "")
    i = findfirst(==(lu), _LENGTH_UNITS)
    return (;
        length_units = isnothing(i) ? :undefined : _LENGTH_UNIT_SYMBOLS[i],
        nogrb = option(o, "nogrb", false), grb_file = option(o, "grb6_filename", ""),
        xorigin = option(o, "xorigin", 0.0), yorigin = option(o, "yorigin", 0.0),
        angrot = option(o, "angrot", 0.0),
    )
end

# VERTICES block: `iv xv yv`, iv = 1..nvert in order (Disv.f90 source_vertices).
function _read_vertices(b::InputBlock, nvert::Integer)
    length(b.lines) == nvert || input_error(b, "found $(length(b.lines)) vertices, expected NVERT = $nvert")
    v = zeros(2, nvert)
    for (r, l) in enumerate(b.lines)
        get_int(l, 1; name = "IV") == r || input_error(l, "VERTICES input sequence violation (expected vertex $r).")
        v[1, r] = get_double(l, 2; name = "XV")
        v[2, r] = get_double(l, 3; name = "YV")
    end
    return v
end

# CELL2D block: `icell2d xc yc ncvert icvert(ncvert)` (Disv.f90 source_cell2d).
function _read_cell2d(b::InputBlock, ncell::Integer, nvert::Integer)
    length(b.lines) == ncell || input_error(b, "found $(length(b.lines)) CELL2D records, expected $ncell")
    xy = zeros(2, ncell)
    icv = Vector{Vector{Int32}}(undef, ncell)
    for (r, l) in enumerate(b.lines)
        get_int(l, 1; name = "ICELL2D") == r || input_error(l, "ICELL2D input sequence violation.")
        xy[1, r] = get_double(l, 2; name = "XC")
        xy[2, r] = get_double(l, 3; name = "YC")
        nc = get_int(l, 4; name = "NCVERT")
        nc >= 0 || input_error(l, "NCVERT must not be negative")
        iv = Vector{Int32}(undef, nc)
        for j in 1:nc
            x = get_int(l, 4 + j; name = "ICVERT")
            1 <= x <= nvert || input_error(l, "ICVERT $x of cell $r is not a vertex number (NVERT = $nvert)")
            iv[j] = x
        end
        icv[r] = iv
    end
    return xy, icv
end

function _load_dis(ld::_Loader, e::PackageEntry, mname)
    f = _read_package(ld, e.file, e.ftype, "GWF")
    ctx = "model '$mname'"
    o = read_options(f, "OPTIONS")
    kind = f.dfn.subcomponent
    if ld.strict
        kind == "DISV" && check_feature(f.path, :disv, "DISV6 discretization ($ctx)")
        kind == "DISU" && check_feature(f.path, :disu, "DISU6 discretization ($ctx)")
        check_options(o, f.dfn.name, e.ftype, ctx)
    end
    g = grid_dims(f)
    gd = find_block(f, "GRIDDATA")
    arrays = read_array_block(gd, f.dfn, g; simdir = ld.simdir)
    req(n) = _required(arrays, n, f.path)
    common = _dis_options(o)
    file = _rel(ld, f.path)
    if kind == "DIS"
        nlay, nrow, ncol = g.mshape
        idomain = haskey(arrays, "idomain") ? arrays["idomain"] : ones(Int32, ncol, nrow, nlay)
        return DisSpec(;
            nlay, nrow, ncol, delr = req("delr"), delc = req("delc"), top = req("top"),
            botm = req("botm"), idomain, common..., file,
        )
    elseif kind == "DISV"
        nlay, ncpl = g.mshape
        nvert = g.dims["NVERT"]
        nvert >= 1 || input_error(find_block(f, "DIMENSIONS"), "required input dimension \"NVERT\" not found or not positive")
        vb = find_block(f, "VERTICES")
        cb = find_block(f, "CELL2D")
        isnothing(vb) && throw(MF6InputError(f.path, 0, "Required Vertex arrays not found."))
        isnothing(cb) && throw(MF6InputError(f.path, 0, "Required cell vertex array(s) [ICELL2D, NCVERT, ICVERT] not found."))
        xy, icv = _read_cell2d(cb, ncpl, nvert)
        idomain = haskey(arrays, "idomain") ? arrays["idomain"] : ones(Int32, ncpl, nlay)
        return DisvSpec(;
            nlay, ncpl, nvert, top = req("top"), botm = req("botm"), idomain,
            vertices = _read_vertices(vb, nvert), cell_center = xy, cell_vertices = icv, common..., file,
        )
    else
        nodes = g.mshape[1]
        nja = g.dims["NJA"]
        nja >= 1 || input_error(find_block(f, "DIMENSIONS"), "required input dimension \"NJA\" not found or not positive")
        nvert = g.dims["NVERT"]
        cd = find_block(f, "CONNECTIONDATA")
        conn = read_array_block(cd, f.dfn, g; simdir = ld.simdir)
        creq(n) = _required(conn, n, f.path, "CONNECTIONDATA")
        vertices = zeros(2, 0)
        xy = zeros(2, 0)
        icv = Vector{Int32}[]
        if nvert > 0
            vb = find_block(f, "VERTICES")
            cb = find_block(f, "CELL2D")
            isnothing(vb) || (vertices = _read_vertices(vb, nvert))
            isnothing(cb) || ((xy, icv) = _read_cell2d(cb, nodes, nvert))
        end
        iac = creq("iac")
        sum(Int, iac) == nja || input_error(cd, "The sum of IAC ($(sum(Int, iac))) does not equal NJA ($nja).")
        return DisuSpec(;
            nodes, nja, nvert, top = req("top"), bot = req("bot"), area = req("area"),
            idomain = haskey(arrays, "idomain") ? arrays["idomain"] : ones(Int32, nodes),
            iac, ja = creq("ja"), ihc = creq("ihc"), cl12 = creq("cl12"), hwva = creq("hwva"),
            angldegx = haskey(conn, "angldegx") ? conn["angldegx"] : nothing,
            vertices, cell_center = xy, cell_vertices = icv,
            vertical_offset_tolerance = option(o, "vertical_offset_tolerance", 0.0), common..., file,
        )
    end
end

# The active cells (IDOMAIN > 0) by user node.
_active(dis::AbstractDisSpec) = idomain_user(dis) .> 0

# ---------------------------------------------------------------------------------------
# NPF, STO, IC, OC
# ---------------------------------------------------------------------------------------

const _CELL_AVERAGING = Dict("LOGARITHMIC" => :logarithmic, "AMT-LMK" => :amt_lmk, "AMT-HMK" => :amt_hmk, "HARMONIC" => :harmonic)

function _load_npf(ld::_Loader, e::PackageEntry, mname, dis::AbstractDisSpec)
    f = _read_package(ld, e.file, e.ftype, "GWF")
    ctx = "model '$mname'"
    o = read_options(f, "OPTIONS")
    ld.strict && check_options(o, f.dfn.name, e.ftype, ctx)
    arrays = read_array_block(find_block(f, "GRIDDATA"), f.dfn, grid_dims(dis); simdir = ld.simdir)
    icelltype = vec(_required(arrays, "icelltype", f.path))
    k = vec(_required(arrays, "k", f.path))
    k22overk = option(o, "k22overk", false)
    k33overk = option(o, "k33overk", false)
    (k33overk && !haskey(arrays, "k33")) && throw(MF6InputError(f.path, 0, "K33OVERK option specified but K33 not specified."))
    (k22overk && !haskey(arrays, "k22")) && throw(MF6InputError(f.path, 0, "K22OVERK option specified but K22 not specified."))
    if ld.strict
        act = _active(dis)
        any(i -> act[i] && icelltype[i] != 0, eachindex(icelltype)) &&
            check_feature(f.path, :convertible, "convertible cells (NPF6 ICELLTYPE /= 0 in an active cell, $ctx)")
    end
    return NpfSpec(;
        icelltype, k, k22 = _opt_array(arrays, "k22"), k33 = _opt_array(arrays, "k33"),
        angle1 = _opt_array(arrays, "angle1"), angle2 = _opt_array(arrays, "angle2"),
        angle3 = _opt_array(arrays, "angle3"), wetdry = _opt_array(arrays, "wetdry"),
        cell_averaging = _CELL_AVERAGING[option(o, "alternative_cell_averaging", "HARMONIC")],
        thickstrt = option(o, "thickstrt", false), variablecv = option(o, "variablecv", false),
        dewatered = option(o, "dewatered", false), perched = option(o, "perched", false),
        rewet = option(o, "rewet", false), wetfct = option(o, "wetfct", 1.0),
        iwetit = option(o, "iwetit", 1), ihdwet = option(o, "ihdwet", 0),
        xt3d = option(o, "xt3d", false), xt3d_rhs = option(o, "rhs", false),
        highest_cell_saturation = option(o, "highest_cell_saturation", false), k22overk, k33overk,
        save_flows = option(o, "save_flows", false), print_flows = option(o, "print_flows", false),
        save_specific_discharge = option(o, "save_specific_discharge", false),
        save_saturation = option(o, "save_saturation", false),
        tvk_files = _filein(ld, o, "tvk6_filename", "TVK6"), file = _rel(ld, f.path),
    )
end

# gwf-sto.f90 source_data: negative SS / SY in an active cell.
function _check_nonnegative(f, name, a, dis::AbstractDisSpec)
    g = grid_dims(dis)
    act = _active(dis)
    for n in eachindex(a)
        if act[n] && a[n] < 0
            throw(MF6InputError(f.path, 0, "Error in $name DATA: $name value in cell $(_shape_string(cellid(g, n))) is less than zero ($(a[n]))."))
        end
    end
    return nothing
end

function _load_sto(ld::_Loader, e::PackageEntry, mname, dis::AbstractDisSpec, np::Integer)
    f = _periods_upto(_read_package(ld, e.file, e.ftype, "GWF"), np)
    ctx = "model '$mname'"
    o = read_options(f, "OPTIONS")
    ld.strict && check_options(o, f.dfn.name, e.ftype, ctx)
    arrays = read_array_block(find_block(f, "GRIDDATA"), f.dfn, grid_dims(dis); simdir = ld.simdir)
    iconvert = vec(_required(arrays, "iconvert", f.path))
    ss = vec(_required(arrays, "ss", f.path))
    sy = _opt_array(arrays, "sy")
    act = _active(dis)
    isconv = any(i -> act[i] && iconvert[i] != 0, eachindex(iconvert))  # MF6 checks reduced nodes
    (isconv && isnothing(sy)) && throw(MF6InputError(f.path, 0, "Error in GRIDDATA block: SY not found."))
    _check_nonnegative(f, "SS", ss, dis)
    isnothing(sy) || _check_nonnegative(f, "SY", sy, dis)
    sto = StoSpec(;
        iconvert, ss, sy, storagecoefficient = option(o, "storagecoefficient", false),
        ss_confined_only = option(o, "ss_confined_only", false), save_flows = option(o, "save_flows", false),
        periods = read_sto_periods(f), tvs_files = _filein(ld, o, "tvs6_filename", "TVS6"), file = _rel(ld, f.path),
    )
    if ld.strict && any(i -> act[i] && iconvert[i] != 0, eachindex(iconvert)) && any(k -> is_transient(sto, k), 1:np)
        check_feature(f.path, :convertible_storage, "convertible storage (STO6 ICONVERT /= 0 in an active cell with a transient period, $ctx)")
    end
    return sto
end

function _load_ic(ld::_Loader, e::PackageEntry, mname, dis::AbstractDisSpec)
    f = _read_package(ld, e.file, e.ftype, "GWF")
    o = read_options(f, "OPTIONS")
    ld.strict && check_options(o, f.dfn.name, e.ftype, "model '$mname'")
    arrays = read_array_block(find_block(f, "GRIDDATA"), f.dfn, grid_dims(dis); simdir = ld.simdir)
    return IcSpec(; strt = vec(_required(arrays, "strt", f.path)), file = _rel(ld, f.path))
end

const _OC_RTYPES = ("BUDGET", "HEAD")  # gwf-oc.f90 oc_ar: the GWF output data objects

function _load_oc(ld::_Loader, e::PackageEntry, mname, np::Integer)
    f = _periods_upto(_read_package(ld, e.file, e.ftype, "GWF"), np)
    o = read_options(f, "OPTIONS")
    ld.strict && check_options(o, f.dfn.name, e.ftype, "model '$mname'")
    periods = read_oc_periods(f)
    # OutputControl.f90 oc_rp
    for (b, recs) in zip(find_blocks(f, "PERIOD"), periods.data)
        for (r, rec) in enumerate(recs)
            rec.rtype in _OC_RTYPES || input_error(b.lines[r], "Input OC period block rtype not found:  \"$(rec.rtype)\".")
        end
    end
    pf = has_option(o, "print_format") ?
        PrintFormat(option(o, "columns", 10), option(o, "width", 15), option(o, "digits", 6), option(o, "format", "GENERAL")) : nothing
    return OcSpec(;
        head_file = option(o, "headfile", ""), budget_file = option(o, "budgetfile", ""),
        budgetcsv_file = option(o, "budgetcsvfile", ""), head_print_format = pf, periods,
        file = _rel(ld, f.path),
    )
end

# ---------------------------------------------------------------------------------------
# Stress packages
# ---------------------------------------------------------------------------------------

# BoundaryPackageExt.f90 cellid_to_nlist: a list cellid must be in an active cell.
function _check_active_list(f::InputFile, sp::StressPeriods{PeriodList}, dis::AbstractDisSpec)
    g = grid_dims(dis)
    idom = idomain_user(dis)
    for (b, p) in zip(find_blocks(f, "PERIOD"), sp.data)
        for (r, n) in enumerate(p.nodeu)
            idom[n] > 0 && continue
            msg = "Cell is outside active grid domain: $(_shape_string(cellid(g, n)))"
            length(b.lines) == length(p.nodeu) ? input_error(b.lines[r], msg) :
                input_error(b, "PERIOD $(block_number(b)), record $r: $msg")
        end
    end
    return nothing
end

# TS6 and TAS6 files of a stress package (T6.3; TimeSeriesManager.f90 `HashBndTimeSeries`:
# a later series of the same name replaces the earlier one), and every name the period
# data use must be one of them (`read_value_or_time_series`, `MakeTasLink`).
function _load_time_series(ld::_Loader, f::InputFile, ts_files, tas_files, periods, g::GridDims)
    ts = Dict{Symbol, TimeSeries}()
    for fname in ts_files, s in read_time_series_file(_resolve_path(fname, ld.simdir); simdir = ld.simdir)
        ts[s.name] = s
    end
    tas = Dict{Symbol, TimeArraySeries}()
    ncpl = prod(g.mshape[2:end])
    for fname in tas_files
        s = read_time_array_series_file(_resolve_path(fname, ld.simdir), ncpl; simdir = ld.simdir)
        tas[s.name] = s
    end
    for (k, p) in zip(periods.iper, periods.data)
        if p isa PeriodList
            for r in p.ts
                haskey(ts, r.name) ||
                    throw(MF6InputError(f.path, 0, "PERIOD $k, record $(r.row): Error in list input. Expected numeric value or time-series name, but found '$(r.name)'."))
            end
        else
            for name in values(p.tas)
                haskey(tas, name) || throw(MF6InputError(f.path, 0, "PERIOD $k: Error: Time-array series \"$name\" not found."))
            end
        end
    end
    return ts, tas
end

function _load_boundary(ld::_Loader, e::PackageEntry, mname, dis::AbstractDisSpec, np::Integer)
    f = _periods_upto(_read_package(ld, e.file, e.ftype, "GWF"), np)
    ctx = "model '$mname'"
    kind = Symbol(rstrip(e.ftype, '6'))
    o = read_options(f, "OPTIONS")
    d = read_options(f, "DIMENSIONS")
    endswith(f.dfn.name, "g") && unsupported(f.path, "$(e.ftype) READARRAYGRID input (package '$(e.name)', $ctx)")
    option(o, "mover", false) && unsupported(f.path, "$(e.ftype) option MOVER (MVR6 is out of scope; package '$(e.name)', $ctx)")
    ld.strict && check_options(o, f.dfn.name, e.ftype, "package '$(e.name)', $ctx")
    auxnames = option(o, "auxiliary", String[])::Vector{String}
    # BoundaryPackageExt.f90 source_options: AUXMULTNAME names an auxiliary variable.
    auxmultname = option(o, "auxmultname", "")
    auxmultcol = 0
    if !isempty(auxmultname)
        l = option_line(o, "AUXMULTNAME")
        isempty(auxnames) && input_error(l, "AUXMULTNAME was specified as $auxmultname but no AUX variables specified.")
        auxmultcol = something(findfirst(==(auxmultname), auxnames), 0)
        auxmultcol == 0 && input_error(l, "AUXMULTNAME was specified as $auxmultname but no AUX variable found with this name.")
    end
    g = grid_dims(dis)
    readasarrays = f.dfn.name in ("gwf-rcha", "gwf-evta")
    if readasarrays
        schema = nothing
        periods = read_array_periods(f, g; simdir = ld.simdir)
    else
        schema, periods = read_list_periods(f, g; simdir = ld.simdir)
        _check_active_list(f, periods, dis)
    end
    if ld.strict
        hasts = readasarrays ? any(p -> !isempty(p.tas), periods.data) : any(p -> !isempty(p.ts), periods.data)
        hasts && check_feature(f.path, :timeseries, "time-series values ($(readasarrays ? "TAS6" : "TS6")) in package '$(e.name)' ($ctx)")
    end
    ts_files = _filein(ld, o, "ts6_filename", "TS6")
    tas_files = _filein(ld, o, "tas6_filename", "TAS6")
    timeseries, timearrayseries = _load_time_series(ld, f, ts_files, tas_files, periods, g)
    return BoundarySpec{kind}(;
        name = e.name, readasarrays, auxnames, auxmultname, auxmultcol,
        auxdepthname = option(o, "auxdepthname", ""), boundnames = option(o, "boundnames", false),
        fixed_cell = option(o, "fixed_cell", false), nseg = option(d, "nseg", 1),
        surf_rate_specified = option(o, "surf_rate_specified", false), maxbound = option(d, "maxbound", 0),
        print_input = option(o, "print_input", false), print_flows = option(o, "print_flows", false),
        save_flows = option(o, "save_flows", false), auto_flow_reduce = option(o, "auto_flow_reduce", 0.0),
        flow_reduction_length = option(o, "flow_reduction_length", false),
        auto_flow_reduce_auxname = option(o, "auto_flow_reduce_auxname", ""),
        afrcsv_file = option(o, "afrcsvfile", ""), mover = false,
        ts_files, tas_files, obs_files = _filein(ld, o, "obs6_filename", "OBS6"), timeseries, timearrayseries,
        schema, periods, file = _rel(ld, f.path),
    )
end

# ---------------------------------------------------------------------------------------
# GWF model
# ---------------------------------------------------------------------------------------

# Name file PACKAGES block (ModelPackageInputs.f90 modelpkgs_create / LoadablePackageType%add).
function _model_packages(nam::InputFile, mname)
    pb = find_block(nam, "PACKAGES")
    entries = PackageEntry[]
    counts = Dict{String, Int}()
    for l in pb.lines
        ftype = keyword(l, 1)
        fname = token(l, 2)
        isnothing(fname) && input_error(l, "missing file name for package type $ftype")
        pname = token(l, 3)
        if ftype in UNSUPPORTED_PACKAGES
            what = isnothing(pname) ? "" : " '$(uppercase(pname))'"
            unsupported(nam.path, "$ftype package$what in model '$mname'")
        end
        ftype in _GWF_ALL_FTYPES || input_error(l, "Model package type not supported [model=$mname, type=$ftype].")
        n = counts[ftype] = get(counts, ftype, 0) + 1
        sub = rstrip(ftype, '6')
        name = isnothing(pname) ? (ftype in _GWF_MULTI_FTYPES ? "$sub-$n" : sub) : uppercase(pname)
        length(name) > LENPACKAGENAME && input_error(l, "Input name \"$name\" exceeds maximum allowed length ($LENPACKAGENAME) for PACKAGENAME.")
        push!(entries, PackageEntry(ftype, String(fname), name))
    end
    for (ftype, n) in counts
        (ftype in _GWF_BASE_FTYPES && n > 1) &&
            input_error(pb, "Multiple instances specified for model base package type [model=$mname, type=$ftype].")
    end
    ndis = count(e -> startswith(e.ftype, "DIS"), entries)
    ndis == 0 && throw(MF6InputError(nam.path, 0, "Discretization package not specified for model \"$mname\"."))
    ndis > 1 && input_error(pb, "more than one discretization package (DIS6, DISV6, DISU6) for model \"$mname\"")
    # gwf.f90 ftype_check
    missing_ = String[]
    any(e -> e.ftype == "IC6", entries) || push!(missing_, "Initial Conditions (IC6) package not specified.")
    any(e -> e.ftype == "NPF6", entries) || push!(missing_, "Node Property Flow (NPF6) Package not specified.")
    isempty(missing_) || throw(MF6InputError(nam.path, 0, join(missing_, " ") * " One or more required package(s) not specified."))
    return entries
end

function _load_gwf(ld::_Loader, fname, mname, np::Integer)
    nam = _read_package(ld, fname, "GWF6", "GWF")
    ctx = "model '$mname'"
    o = read_options(nam, "OPTIONS")
    entries = _model_packages(nam, mname)  # out-of-scope packages fail first
    newton = option(o, "newton", false)
    if ld.strict
        newton && check_feature(nam.path, :newton, "NEWTON ($ctx)")
        check_options(o, "gwf-nam", "GWF6 name file", ctx)
    end
    entry(ftype) = (i = findfirst(e -> e.ftype == ftype, entries); isnothing(i) ? nothing : entries[i])
    dis = _load_dis(ld, entries[findfirst(e -> startswith(e.ftype, "DIS"), entries)], mname)
    npf = _load_npf(ld, entry("NPF6"), mname, dis)
    sto = isnothing(entry("STO6")) ? nothing : _load_sto(ld, entry("STO6"), mname, dis, np)
    ic = _load_ic(ld, entry("IC6"), mname, dis)
    oc = isnothing(entry("OC6")) ? nothing : _load_oc(ld, entry("OC6"), mname, np)
    obs = String[]
    for e in entries
        e.ftype == "OBS6" || continue
        path = _resolve_path(e.file, ld.simdir)
        isfile(path) || throw(MF6InputError(path, 0, "OBS6 input file not found"))
        push!(obs, e.file)
    end
    # Stress packages in MF6's order: by type (GWF_MULTIPKG), then name file order.
    boundaries = BoundarySpec[]
    for ftype in _GWF_MULTI_FTYPES, e in entries
        e.ftype == ftype && push!(boundaries, _load_boundary(ld, e, mname, dis, np))
    end
    # gwf.f90 package_create: unique stress package names.
    for (i, b) in enumerate(boundaries)
        any(c -> c.name == b.name, view(boundaries, 1:(i - 1))) &&
            throw(MF6InputError(nam.path, 0, "Cannot create package.  Package name  already exists: $(b.name)"))
    end
    return GwfSpec(;
        name = mname, list_file = option(o, "list", ""), print_input = option(o, "print_input", false),
        print_flows = option(o, "print_flows", false), save_flows = option(o, "save_flows", false),
        newton, newton_under_relaxation = option(o, "under_relaxation", false), packages = entries,
        dis, npf, ic, sto, oc, boundaries, obs_files = obs, file = _rel(ld, nam.path),
    )
end

# ---------------------------------------------------------------------------------------
# Exchanges
# ---------------------------------------------------------------------------------------

function _load_exchange(ld::_Loader, fname, m1::GwfSpec, m2::GwfSpec)
    f = _read_package(ld, fname, "GWF6-GWF6", "EXG")
    o = read_options(f, "OPTIONS")
    has_option(o, "gnc6_filename") && unsupported(f.path, "GNC6 on a GWF6-GWF6 exchange")
    has_option(o, "mvr6_filename") && unsupported(f.path, "MVR6 on a GWF6-GWF6 exchange (out of scope)")
    data = read_exchange_data(f, grid_dims(m1.dis), grid_dims(m2.dis))
    # DisConnExchange.f90 source_data: both cells must be active.
    b = find_block(f, "EXCHANGEDATA")
    for (m, nodes) in ((m1, data.nodem1), (m2, data.nodem2))
        idom = idomain_user(m.dis)
        g = grid_dims(m.dis)
        for (r, n) in enumerate(nodes)
            idom[n] > 0 || input_error(b.lines[r], "$(m.name) Cell is outside active grid domain $(_shape_string(cellid(g, n))).")
        end
    end
    return ExchangeSpec(;
        model1 = m1.name, model2 = m2.name, auxnames = data.auxnames, boundnames = option(o, "boundnames", false),
        print_input = option(o, "print_input", false), print_flows = option(o, "print_flows", false),
        save_flows = option(o, "save_flows", false),
        cell_averaging = _CELL_AVERAGING[option(o, "cell_averaging", "HARMONIC")],
        variablecv = option(o, "variablecv", false), dewatered = option(o, "dewatered", false),
        newton = option(o, "newton", false), xt3d = option(o, "xt3d", false),
        obs_files = _filein(ld, o, "obs6_filename", "OBS6"), data, file = _rel(ld, f.path),
    )
end

# ---------------------------------------------------------------------------------------
# IMS
# ---------------------------------------------------------------------------------------

function _ims_choice(l::InputLine, choices, what)
    w = keyword(l, 2)
    i = findfirst(==(w), first.(choices))
    isnothing(i) && input_error(l, "Unknown $what ($w).")
    return last(choices[i])
end

_ims_fileout(l::InputLine, k) =
    keyword(l, 2) == "FILEOUT" && !isnothing(token(l, 3)) ? String(token(l, 3)) :
    input_error(l, "Optional $k keyword must be followed by FILEOUT")

# The keywords of the IMS readers (sln_read_options, ImsLinearSettings.read_from_file).
const _IMS_KEYWORDS = Dict(
    "OPTIONS" => (
        "PRINT_OPTION", "COMPLEXITY", "CSV_OUTER_OUTPUT", "CSV_INNER_OUTPUT", "NO_PTC",
        "ATS_OUTER_MAXIMUM_FRACTION",
    ),
    "NONLINEAR" => (
        "OUTER_DVCLOSE", "OUTER_MAXIMUM", "UNDER_RELAXATION", "LINEAR_SOLVER", "UNDER_RELAXATION_THETA",
        "UNDER_RELAXATION_KAPPA", "UNDER_RELAXATION_GAMMA", "UNDER_RELAXATION_MOMENTUM", "BACKTRACKING_NUMBER",
        "BACKTRACKING_TOLERANCE", "BACKTRACKING_REDUCTION_FACTOR", "BACKTRACKING_RESIDUAL_LIMIT",
    ),
    "LINEAR" => (
        "INNER_DVCLOSE", "INNER_RCLOSE", "INNER_MAXIMUM", "LINEAR_ACCELERATION", "SCALING_METHOD",
        "RED_BLACK_ORDERING", "REORDERING_METHOD", "NUMBER_ORTHOGONALIZATIONS", "RELAXATION_FACTOR",
        "PRECONDITIONER_LEVELS", "PRECONDITIONER_DROP_TOLERANCE",
    ),
)

# The IMS file `fname` for `models` (NumericalSolution.f90 sln_read_options,
# ImsLinearSettings.f90 read_from_file; blocks OPTIONS, NONLINEAR, LINEAR, all optional).
function _load_ims(ld::_Loader, fname, models::Vector{String})
    f = _read_package(ld, fname, "IMS6", "SLN")
    given = String[]
    # Unknown keywords are input errors (below); known ones outside the table unsupported.
    strict_check(l, block, k) = (ld.strict && k in _IMS_KEYWORDS[block] && !(k in SUPPORTED_OPTIONS[("sln-ims", block)])) &&
        unsupported(f.path, "IMS6 option $k (line $(l.line))")
    dev(l, k) = input_error(l, "$k is a development option (needs a develop-mode build of MF6)")
    # OPTIONS
    kw = Dict{Symbol, Any}()
    complexity = :simple
    ob = find_block(f, "OPTIONS")
    for l in (isnothing(ob) ? InputLine[] : ob.lines)
        k = keyword(l, 1)
        push!(given, k)
        startswith(k, "DEV_") && dev(l, k)
        strict_check(l, "OPTIONS", k)
        if k == "PRINT_OPTION"
            kw[:print_option] = _ims_choice(l, ("NONE" => :none, "SUMMARY" => :summary, "ALL" => :all), "IMS print option")
        elseif k == "COMPLEXITY"
            complexity = _ims_choice(l, ("SIMPLE" => :simple, "MODERATE" => :moderate, "COMPLEX" => :complex), "IMS COMPLEXITY option")
        elseif k == "CSV_OUTER_OUTPUT"
            kw[:csv_outer_output] = _ims_fileout(l, k)
        elseif k == "CSV_INNER_OUTPUT"
            kw[:csv_inner_output] = _ims_fileout(l, k)
        elseif k == "NO_PTC"
            kw[:no_ptc] = keyword(l, 2) == "FIRST" ? :first : :all
        elseif k == "ATS_OUTER_MAXIMUM_FRACTION"
            v = get_double(l, 2; name = k)
            0.0 <= v <= 0.5 || input_error(l, "Value for ATS_OUTER_MAXIMUM_FRAC must be between 0 and 0.5.  Found $v")
            kw[:ats_outer_maximum_fraction] = v
        else
            input_error(l, "Unknown IMS option ($k).")
        end
    end
    # NONLINEAR (sln_read_options)
    nb = find_block(f, "NONLINEAR")
    dbl(l, k, s) = (kw[s] = get_double(l, 2; name = k))
    int(l, k, s) = (kw[s] = Int32(get_int(l, 2; name = k)))
    for l in (isnothing(nb) ? InputLine[] : nb.lines)
        k = keyword(l, 1)
        push!(given, k)
        strict_check(l, "NONLINEAR", k)
        if k == "OUTER_DVCLOSE"
            dbl(l, k, :outer_dvclose)
        elseif k == "OUTER_MAXIMUM"
            int(l, k, :outer_maximum)
        elseif k == "UNDER_RELAXATION"
            kw[:under_relaxation] = _ims_choice(l, ("NONE" => :none, "SIMPLE" => :simple, "COOLEY" => :cooley, "DBD" => :dbd), "UNDER_RELAXATION specified")
            ld.strict && check_value(f.path, "sln-ims", "IMS6", k, keyword(l, 2), "line $(l.line)")
        elseif k == "LINEAR_SOLVER"
            _ims_choice(l, ("DEFAULT" => :default, "LINEAR" => :default), "LINEAR_SOLVER specified")
        elseif k == "UNDER_RELAXATION_THETA"
            dbl(l, k, :under_relaxation_theta)
        elseif k == "UNDER_RELAXATION_KAPPA"
            dbl(l, k, :under_relaxation_kappa)
        elseif k == "UNDER_RELAXATION_GAMMA"
            dbl(l, k, :under_relaxation_gamma)
        elseif k == "UNDER_RELAXATION_MOMENTUM"
            dbl(l, k, :under_relaxation_momentum)
        elseif k == "BACKTRACKING_NUMBER"
            int(l, k, :backtracking_number)
        elseif k == "BACKTRACKING_TOLERANCE"
            dbl(l, k, :backtracking_tolerance)
        elseif k == "BACKTRACKING_REDUCTION_FACTOR"
            dbl(l, k, :backtracking_reduction_factor)
        elseif k == "BACKTRACKING_RESIDUAL_LIMIT"
            dbl(l, k, :backtracking_residual_limit)
        else
            input_error(l, "Unknown IMS NONLINEAR keyword ($k).")
        end
    end
    # LINEAR (ImsLinearSettings.read_from_file)
    lb = find_block(f, "LINEAR")
    for l in (isnothing(lb) ? InputLine[] : lb.lines)
        k = keyword(l, 1)
        push!(given, k)
        strict_check(l, "LINEAR", k)
        if k == "INNER_DVCLOSE"
            dbl(l, k, :inner_dvclose)
        elseif k == "INNER_RCLOSE"
            dbl(l, k, :inner_rclose)
            w = keyword(l, 3)  # other words are ignored, as in MF6
            opt = w == "STRICT" ? :strict : w == "L2NORM_RCLOSE" ? :l2norm : w == "RELATIVE_RCLOSE" ? :relative :
                w == "L2NORM_RELATIVE_RCLOSE" ? :l2norm_relative : nothing
            isnothing(opt) || (kw[:rclose_option] = opt)
        elseif k == "INNER_MAXIMUM"
            int(l, k, :inner_maximum)
        elseif k == "LINEAR_ACCELERATION"
            kw[:linear_acceleration] = _ims_choice(l, ("CG" => :cg, "BICGSTAB" => :bicgstab), "IMSLINEAR LINEAR_ACCELERATION method")
        elseif k == "SCALING_METHOD"
            kw[:scaling_method] = _ims_choice(l, ("NONE" => :none, "DIAGONAL" => :diagonal, "L2NORM" => :l2norm), "IMSLINEAR SCALING_METHOD")
            ld.strict && check_value(f.path, "sln-ims", "IMS6", k, keyword(l, 2), "line $(l.line)")
        elseif k == "RED_BLACK_ORDERING"
            # ignored by MF6
        elseif k == "REORDERING_METHOD"
            kw[:reordering_method] = _ims_choice(l, ("NONE" => :none, "RCM" => :rcm, "MD" => :md), "IMSLINEAR REORDERING_METHOD")
            ld.strict && check_value(f.path, "sln-ims", "IMS6", k, keyword(l, 2), "line $(l.line)")
        elseif k == "NUMBER_ORTHOGONALIZATIONS"
            int(l, k, :number_orthogonalizations)
        elseif k == "RELAXATION_FACTOR"
            dbl(l, k, :relaxation_factor)
        elseif k == "PRECONDITIONER_LEVELS"
            int(l, k, :preconditioner_levels)
            kw[:preconditioner_levels] < 0 && input_error(l, "IMSLINEAR PRECONDITIONER_LEVELS must be greater than or equal to zero")
        elseif k == "PRECONDITIONER_DROP_TOLERANCE"
            dbl(l, k, :preconditioner_drop_tolerance)
            kw[:preconditioner_drop_tolerance] < 0 && input_error(l, "IMSLINEAR PRECONDITIONER_DROP_TOLERANCE must be greater than or equal to zero")
        else
            input_error(l, "Unknown IMSLINEAR keyword ($k).")
        end
    end
    d = ims_defaults(complexity)
    vals = Dict{Symbol, Any}(n => getfield(d, n) for n in fieldnames(ImsSpec))
    merge!(vals, kw)
    vals[:under_relaxation_theta] = max(vals[:under_relaxation_theta], 1.0e-3)  # sln_read_options
    vals[:outer_maximum] > 0 || throw(MF6InputError(f.path, 0, "Outer iteration number must be > 0."))
    (vals[:under_relaxation] === :simple && vals[:under_relaxation_gamma] == 0) &&
        throw(MF6InputError(f.path, 0, "GAMMA must be greater than zero if SIMPLE under-relaxation is used."))
    if vals[:preconditioner_levels] == 0 && vals[:preconditioner_drop_tolerance] > 0  # check_settings
        @warn "$(f.path): PRECONDITIONER_DROP_TOLERANCE is ignored because PRECONDITIONER_LEVELS equals zero."
    end
    vals[:models] = models
    vals[:given] = given
    vals[:file] = _rel(ld, f.path)
    return ImsSpec(; vals...)
end

# ---------------------------------------------------------------------------------------
# mfsim.nam
# ---------------------------------------------------------------------------------------

# SimulationCreate.f90 check_model_name
function _check_model_name(l::InputLine, name, existing)
    length(name) > LENMODELNAME && input_error(l, "Invalid model name: $name. Name length of $(length(name)) exceeds maximum length of $LENMODELNAME")
    name in existing && input_error(l, "Invalid model name: $name. Model names must be unique.")
    return nothing
end

"""
    load_simulation(simdir; strict = true) -> SimulationSpec

Read the MF6 simulation in directory `simdir` (`mfsim.nam`, TDIS, the GWF6 models and
their packages, GWF6-GWF6 exchanges and IMS6 solutions) into a `SimulationSpec`, with
MF6's checks and error messages (`MF6InputError`; design and differences in the header of
`src/input/simulation.jl`).

Models other than GWF6, exchanges other than GWF6-GWF6, EMS6 and the out-of-scope
packages (`UNSUPPORTED_PACKAGES`) are always an `MF6UnsupportedError`
("unsupported: SFR6 package 'SFR-1' in model 'GWF'"). With `strict = true`, so are
options, values and features not (yet) implemented by the engine (`SUPPORTED_OPTIONS`,
`SUPPORTED_VALUES`, `SUPPORTED_FEATURES` in `src/input/supported.jl`); `strict = false`
reads all in-scope input.
"""
function load_simulation(simdir::AbstractString; strict::Bool = true)
    simdir = abspath(simdir)
    isdir(simdir) || throw(MF6InputError(simdir, 0, "simulation directory not found"))
    ld = _Loader(simdir, strict)
    sim = _read_package(ld, "mfsim.nam", "SIM", "SIM")
    o = read_options(sim, "OPTIONS")
    has_option(o, "hpc6_filename") && unsupported(sim.path, "HPC6 (parallel) input")
    strict && check_options(o, "sim-nam", "mfsim.nam", "simulation")
    # TIMING (timing_create)
    t = read_options(sim, "TIMING")
    tdis6 = option(t, "tdis6", "")
    isempty(tdis6) && throw(MF6InputError(sim.path, 0, "TIMING block variable TDIS6 is unset in simulation control input."))
    # MODELS (models_create); types first, so that out-of-scope models fail early.
    mb = find_block(sim, "MODELS")
    models_in = Tuple{String, String}[]  # (file, name)
    for l in mb.lines
        mtype = keyword(l, 1)
        fname = token(l, 2)
        mname = uppercase(something(token(l, 3), ""))
        (isnothing(fname) || isempty(mname)) && input_error(l, "MODELS record must be: mtype mfname mname")
        mtype in UNSUPPORTED_MODELS && unsupported(sim.path, "$mtype model '$mname' (only GWF6 models are in scope)")
        mtype == "GWF6" || input_error(l, "Unknown simulation model type: $mtype")
        _check_model_name(l, mname, last.(models_in))
        push!(models_in, (String(fname), mname))
    end
    isempty(models_in) && input_error(mb, "no models in the MODELS block")
    strict && length(models_in) > 1 && check_feature(sim.path, :multi_model, "more than one GWF6 model ($(join(last.(models_in), ", ")))")
    # EXCHANGES (exchanges_create)
    eb = find_block(sim, "EXCHANGES")
    exg_in = Tuple{String, String, String}[]
    for l in eb.lines
        etype = keyword(l, 1)
        fname = token(l, 2)
        n1 = uppercase(something(token(l, 3), ""))
        n2 = uppercase(something(token(l, 4), ""))
        (isnothing(fname) || isempty(n1) || isempty(n2)) && input_error(l, "EXCHANGES record must be: exgtype exgfile exgmnamea exgmnameb")
        for n in (n1, n2)
            n in last.(models_in) || input_error(l, "Error in simulation control file.  Could not find model: $n")
        end
        if etype != "GWF6-GWF6"
            occursin(r"^[A-Z]+6-[A-Z]+6$", etype) || input_error(l, "Unknown simulation exchange type: $etype")
            unsupported(sim.path, "$etype exchange (only GWF6-GWF6 exchanges are in scope)")
        end
        strict && check_feature(sim.path, :exchange, "GWF6-GWF6 exchange ($n1 - $n2)")
        push!(exg_in, (String(fname), n1, n2))
    end
    # SOLUTIONGROUP (solution_groups_create)
    groups_in = Tuple{Int, Vector{Tuple{String, Vector{String}}}}[]  # (mxiter, [(file, models)])
    assigned = Dict{String, Int}()
    sgid = 0
    for b in find_blocks(sim, "SOLUTIONGROUP")
        n = block_number(b)
        n == sgid + 1 || input_error(b, "Solution group blocks are not listed consecutively. Found $n when looking for $(sgid + 1).")
        sgid = n
        mxiter = 1
        sols = Tuple{String, Vector{String}}[]
        for l in b.lines
            stype = keyword(l, 1)
            if stype == "MXITER"
                mxiter = get_int(l, 2; name = "MXITER")
            elseif stype == "IMS6"
                fname = token(l, 2)
                isnothing(fname) && input_error(l, "missing IMS6 file name")
                names = [uppercase(t) for t in Iterators.drop(eachtoken(l), 2)]
                for m in names
                    m in last.(models_in) || input_error(l, "Invalid model name: $m")
                    haskey(assigned, m) && input_error(l, "Model $m is already in solution $(assigned[m])")
                    assigned[m] = length(assigned) + 1
                end
                push!(sols, (String(fname), names))
            elseif stype == "EMS6"
                unsupported(sim.path, "EMS6 solution (explicit models are out of scope)")
            else
                input_error(l, "Input file tag not found: \"$stype\" in block \"SOLUTIONGROUP\".")
            end
        end
        isempty(sols) && input_error(b, "There are no solutions for solution group $n")
        (length(sols) == 1 && mxiter > 1) && input_error(b, "MXITER is set to $mxiter but there is only one solution in SOLUTION GROUP $n. Set MXITER to 1 in simulation control file.")
        push!(groups_in, (mxiter, sols))
    end
    isempty(groups_in) && throw(MF6InputError(sim.path, 0, "There are no solution groups."))
    for (_, m) in models_in
        haskey(assigned, m) || throw(MF6InputError(sim.path, 0, "Model was not assigned to a solution: $m"))
    end
    strict && sum(g -> length(g[2]), groups_in) > 1 && check_feature(sim.path, :multi_model, "more than one solution")
    # Load
    tdis = _load_tdis(ld, tdis6)
    np = nper(tdis)
    models = GwfSpec[_load_gwf(ld, fname, mname, np) for (fname, mname) in models_in]
    byname = Dict(m.name => m for m in models)
    exchanges = ExchangeSpec[_load_exchange(ld, fname, byname[n1], byname[n2]) for (fname, n1, n2) in exg_in]
    groups = SolutionGroupSpec[
        SolutionGroupSpec(; mxiter, solutions = ImsSpec[_load_ims(ld, fname, names) for (fname, names) in sols])
            for (mxiter, sols) in groups_in
    ]
    return SimulationSpec(;
        simdir, continue_on_nonconvergence = option(o, "continue", false), nocheck = option(o, "nocheck", false),
        memory_print_option = option(o, "memory_print_option", ""), profile_option = option(o, "profile_option", ""),
        maxerrors = option(o, "maxerrors", 0), print_input = option(o, "print_input", false),
        tdis, models, exchanges, solution_groups = groups,
    )
end
