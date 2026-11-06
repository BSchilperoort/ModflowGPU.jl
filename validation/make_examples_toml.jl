#!/usr/bin/env julia
# Generate validation/examples.toml from the MF6 examples (T0.3).
#
#     REFERENCE_DIR=/path/to/reference julia validation/make_examples_toml.jl [out.toml]
#
# Scans every simulation (directory with mfsim.nam) under
# $REFERENCE_DIR/mf6examples, reads the inputs with the small stdlib-only reader
# in mf6_input.jl, and writes one entry per in-scope GWF simulation plus a list of
# out-of-scope simulations with the reason. Tiers follow PLAN.md section 4.1
# (table TIERS below); simulations not named there get a tier from their
# features and `in_plan = false`. Everything else (grid, sizes, packages,
# convertible flags, ...) is read from the input files, including the actual
# ICELLTYPE / ICONVERT / IDOMAIN arrays.

include(joinpath(@__DIR__, "mf6_input.jl"))
using .MF6Input
using Printf

const REPO = dirname(@__DIR__)
const REFERENCE_DIR = get(ENV, "REFERENCE_DIR", joinpath(REPO, "reference"))
const EXAMPLES = joinpath(REFERENCE_DIR, "mf6examples")

# Packages, models and exchanges outside the project scope (PLAN.md section 1).
const OOS_PACKAGES = Set(["SFR6", "LAK6", "MAW6", "UZF6", "MVR6", "CSUB6", "BUY6", "VSC6", "API6"])
const OOS_MODELS = Set(["GWT6", "GWE6", "PRT6", "CHF6", "OLF6", "SWF6"])

# PLAN.md section 4.1, by simulation name.
const TIERS = Dict{String, Int}(
    "ex-gwf-toth" => 0, "ex-gwf-hanic" => 0, "ex-gwf-hanir" => 0,
    "ex-gwf-fhb" => 2, "ex-gwf-advtidal" => 2,
    "ex-gwf-twri01" => 3, "ex-gwf-bcf2ss-p01a" => 3, "ex-gwf-bump-p01b" => 3,
    "ex-gwf-nwt-p02b" => 3, "ex-gwf-lgrv-gc" => 3, "ex-gwf-lgrv-gr" => 3,
    "ex-gwf-bcf2ss-p02a" => 4, "ex-gwf-bump-p01a" => 4, "ex-gwf-bump-p01c" => 4,
    "ex-gwf-nwt-p02a" => 4, "ex-gwf-nwt-p03a" => 4, "ex-gwf-nwt-p03b" => 4,
    "ex-gwf-zaidel" => 4, "ex-gwf-capture" => 4,
    "ex-gwf-u1disv" => 5, "ex-gwf-curve-90" => 5, "ex-gwf-curvilin" => 5,
    "ex-gwf-rad-disu" => 5,
    "ex-gwf-hanix" => 6, "ex-gwf-u1disv-x" => 6, "ex-gwf-whirl" => 6,
    "ex-gwf-disvmesh" => 6,
    "ex-gwf-u1gwfgwf-s1" => 7, "ex-gwf-u1gwfgwf-s2" => 7, "ex-gwf-u1gwfgwf-s3" => 7,
    "ex-gwf-u1gwfgwf-s4" => 7, "ex-gwf-lgrv-lgr" => 7, "ex-gwf-spbc" => 7,
    # "extra Tier 0-2 size tests once STO is done"
    "ex-gwf-hwell-a" => 2, "ex-gwf-maw-p03a" => 2, "ex-gwf-maw-p03c" => 2,
)

# Hand-checked remarks (see docs/examples_inventory.md).
const NOTES = Dict{String, String}(
    "ex-gwf-hwell-a" => "PLAN 4.1 extra size test (not in a tier row); tier 2 because of STO (transient, all cells confined). Directory in the zip is ex-gwf-hwell/ex-gwf-hwell-a.",
    "ex-gwf-maw-p03a" => "PLAN 4.1 extra size test; no MAW package in this variant. Steady state without STO, but layer 1 is convertible (ICELLTYPE 1), so it needs Tier 3 features, not just STO.",
    "ex-gwf-maw-p03c" => "PLAN 4.1 extra size test; no MAW package in this variant. Steady state without STO, but layer 1 is convertible (ICELLTYPE 1), so it needs Tier 3 features, not just STO.",
    "ex-gwf-advtidal" => "Layer 1 convertible (ICELLTYPE 1) with VARIABLECV DEWATERED and PERCHED, ICONVERT 1 everywhere: needs the Tier 3 convertible formulation although PLAN puts it in Tier 2.",
    "ex-gwf-fhb" => "STO with STORAGECOEFFICIENT; all cells confined.",
    "ex-gwf-lgrv-gc" => "PLAN 4.1 lists it under Tier 3 (convertible), but ICELLTYPE is 0 everywhere: confined. Still a Tier 3 size/RIV/RCHA/K33OVERK test.",
    "ex-gwf-lgrv-gr" => "PLAN 4.1 lists it under Tier 3 (convertible), but ICELLTYPE is 0 everywhere: confined. Still the 25x183x147 size test.",
    "ex-gwf-lgrv-lgr" => "Parent and child models both confined (ICELLTYPE 0).",
    "ex-gwf-rad-disu" => "ICELLTYPE 0 (confined NPF) but STO ICONVERT 1 in a transient period: convertible storage (SY) without convertible conductance.",
    "ex-gwf-u1gwfgwf-s4" => "PLAN 4.1 says no XT3D, but the GWF-GWF exchange has the XT3D option (the models do not).",
    "ex-gwf-u1gwfgwf-s2" => "XT3D in both models' NPF, not on the exchange.",
    "ex-gwf-u1gwfgwf-s3" => "XT3D in both models' NPF and on the exchange.",
    "ex-gwf-curve-90" => "Has STO, but the only period is steady state.",
    "ex-gwf-curvilin" => "Has STO, but the only period is steady state.",
)

# ---------------------------------------------------------------------------
# Ordered TOML tables: Vector{Pair{String,Any}}

const Tbl = Vector{Pair{String, Any}}
(::Type{Tbl})(ps::Pair...) = Pair{String, Any}[ps...]

tomlkey(k) = occursin(r"^[A-Za-z0-9_-]+$", k) ? k : repr(k)
tomlval(v::Bool) = v ? "true" : "false"
tomlval(v::Integer) = string(v)
tomlval(v::AbstractFloat) = isinteger(v) && abs(v) < 1.0e15 ? @sprintf("%.1f", v) : repr(v)
tomlval(v::AbstractString) = repr(String(v))   # Julia string escapes are valid TOML for our content
tomlval(v::AbstractVector) = "[" * join(tomlval.(v), ", ") * "]"

function emit(io, t::Tbl, path::Vector{String})
    for (k, v) in t
        (v isa Tbl || v isa Vector{Tbl} || v === nothing) && continue
        println(io, tomlkey(k), " = ", tomlval(v))
    end
    for (k, v) in t
        if v isa Tbl
            p = [path; k]
            println(io, "\n[", join(tomlkey.(p), "."), "]")
            emit(io, v, p)
        end
    end
    for (k, v) in t
        if v isa Vector{Tbl}
            p = [path; k]
            for e in v
                println(io, "\n[[", join(tomlkey.(p), "."), "]]")
                emit(io, e, p)
            end
        end
    end
    return
end

# ---------------------------------------------------------------------------
# Scanning

upper(s) = uppercase(String(s))

"Run-length summary of a per-period vector, e.g. \"1:SS 2-20:TR\"."
function runs(v::Vector{String})
    parts = String[]
    i = 1
    while i <= length(v)
        j = i
        while j < length(v) && v[j + 1] == v[i]
            j += 1
        end
        push!(parts, (i == j ? "$i" : "$i-$j") * ":" * v[i])
        i = j + 1
    end
    return join(parts, " ")
end

function read_tdis(path)
    bs = read_blocks(path)
    o = option_lines(bs)
    units = option_value(o, "TIME_UNITS")
    nper = parse(Int, option_value(option_lines(bs, "DIMENSIONS"), "NPER")[1])
    pd = find_block(bs, "PERIODDATA").lines
    nstp = [parse(Int, l[2]) for l in pd]
    perlen = [parse(Float64, l[1]) for l in pd]
    ats = option_value(o, "ATS6") !== nothing
    return (; nper, nstp, perlen, units = units === nothing ? "unknown" : lowercase(units[1]), ats)
end

function options_table(lines; skip = ())
    t = Tbl()
    for l in lines
        k = lowercase(l[1])
        k in skip && continue
        push!(t, k => length(l) == 1 ? true : join(l[2:end], " "))
    end
    return t
end

function count_flags(v, active)
    z = p = n = 0
    for (x, a) in zip(v, active)
        a || continue
        x == 0 ? (z += 1) : x > 0 ? (p += 1) : (n += 1)
    end
    return z, p, n
end

struct ModelScan
    table::Tbl
    grid::String
    ncells::Int
    convertible::Bool
    newton::Bool
    xt3d::Bool
    periods::Vector{String}
    ts::Bool
    tas::Bool
    obs::Bool
    storage_convertible::Bool
    requires::Vector{String}
    oos::Vector{String}
end

function scan_gwf(dir, namfile, mname, tdis)
    nam = read_blocks(joinpath(dir, namfile))
    nopts = option_lines(nam)
    pk = [(upper(l[1]), l[2], length(l) >= 3 ? l[3] : lowercase(replace(upper(l[1]), "6" => ""))) for l in option_lines(nam, "PACKAGES")]
    oos = String[]
    for (ft, _, _) in pk
        ft in OOS_PACKAGES && push!(oos, "package $(ft[1:(end - 1)])")
    end
    # discretization
    di = findfirst(p -> p[1] in ("DIS6", "DISV6", "DISU6"), pk)
    dft, dfile, _ = pk[di]
    dbs = read_blocks(joinpath(dir, dfile))
    dims = option_lines(dbs, "DIMENSIONS")
    dv(k) = (v = option_value(dims, k); v === nothing ? 0 : parse(Int, v[1]))
    t = Tbl()
    grid = dft[1:(end - 1)]
    push!(t, "grid" => grid)
    if grid == "DIS"
        nlay, nrow, ncol = dv("NLAY"), dv("NROW"), dv("NCOL")
        ncpl = nrow * ncol
        append!(t, ["nlay" => nlay, "nrow" => nrow, "ncol" => ncol])
        sizef = name -> name == "delr" ? (ncol, false) : name == "delc" ? (nrow, false) :
            name == "top" ? (ncpl, false) : (nlay * ncpl, true)
    elseif grid == "DISV"
        nlay, ncpl = dv("NLAY"), dv("NCPL")
        append!(t, ["nlay" => nlay, "ncpl" => ncpl, "nvert" => dv("NVERT")])
        sizef = name -> name == "top" ? (ncpl, false) : (nlay * ncpl, true)
    else
        nlay, ncpl = 1, dv("NODES")
        append!(t, ["nodes" => ncpl, "nja" => dv("NJA")])
        sizef = name -> (ncpl, false)
    end
    nodes = nlay * ncpl
    dg = find_block(dbs, "GRIDDATA")
    darr = read_griddata(dg, dir, sizef, nlay)
    idomain = get(darr, "idomain", nothing)
    active = idomain === nothing ? trues(nodes) : idomain .> 0
    push!(t, "ncells" => nodes)
    push!(t, "nactive" => count(active))
    idomain !== nothing && any(==(-1), idomain) && push!(t, "idomain_vertical_passthrough" => count(==(-1), idomain))
    dopts = option_lines(dbs)
    for k in ("XORIGIN", "YORIGIN", "ANGROT")
        (v = option_value(dopts, k)) === nothing || push!(t, lowercase(k) => parse(Float64, v[1]))
    end
    # output files (paths relative to the simulation directory)
    if !has_option(dopts, "NOGRB")
        g = option_value(dopts, "GRB6")
        # default: the discretization file name plus ".grb" (gwf-dis.dfn, grb6_filename)
        push!(t, "grb_file" => g !== nothing && length(g) >= 2 && upper(g[1]) == "FILEOUT" ? g[2] : dfile * ".grb")
    end
    oi = findfirst(p -> p[1] == "OC6", pk)
    if oi !== nothing
        oco = option_lines(read_blocks(joinpath(dir, pk[oi][2])))
        for (k, key) in (("HEAD", "head_file"), ("BUDGET", "budget_file"))
            v = option_value(oco, k)
            v !== nothing && length(v) >= 2 && upper(v[1]) == "FILEOUT" && push!(t, key => v[2])
        end
    end

    # packages
    ftypes = String[]
    ts = String[]
    tas = String[]
    obsp = String[]
    pkdetail = Tbl()
    newton = has_option(nopts, "NEWTON")
    nt = Tbl()
    for (ft, f, pn) in pk
        name = ft[1:(end - 1)]
        if ft == "OBS6"
            push!(obsp, "model")
            push!(ftypes, name)
            continue
        end
        bs = read_blocks(joinpath(dir, f))
        o = option_lines(bs)
        if ft in ("RCH6", "EVT6") && has_option(o, "READASARRAYS")
            name *= "A"
        end
        push!(ftypes, name)
        option_value(o, "TS6") !== nothing && push!(ts, pn)
        option_value(o, "TAS6") !== nothing && push!(tas, pn)
        option_value(o, "OBS6") !== nothing && push!(obsp, pn)
        if !(ft in ("DIS6", "DISV6", "DISU6", "IC6", "OC6", "NPF6", "STO6"))
            d = Tbl("type" => name)
            dims = option_lines(bs, "DIMENSIONS")
            (v = option_value(dims, "MAXBOUND")) === nothing || push!(d, "maxbound" => parse(Int, v[1]))
            (v = option_value(dims, "NSEG")) === nothing || push!(d, "nseg" => parse(Int, v[1]))
            (v = option_value(o, "AUXILIARY")) === nothing || push!(d, "auxiliary" => String.(v))
            for k in ("BOUNDNAMES", "MOVER", "FIXED_CELL", "SURF_RATE_SPECIFIED", "MAXIMUM_PERIOD")
                has_option(o, k) && push!(d, lowercase(k) => true)
            end
            for k in ("AUXMULTNAME", "AUXDEPTHNAME", "AUTO_FLOW_REDUCE", "FLOW_REDUCTION_LENGTH")
                (v = option_value(o, k)) === nothing || push!(d, lowercase(k) => isempty(v) ? "true" : join(v, " "))
            end
            push!(pkdetail, pn => d)
        end
    end
    append!(t, ["packages" => ftypes, "newton" => newton])
    newton && push!(t, "newton_under_relaxation" => (v = option_value(nopts, "NEWTON"); v !== nothing && any(x -> upper(x) == "UNDER_RELAXATION", v)))
    isempty(ts) || push!(t, "ts_packages" => ts)
    isempty(tas) || push!(t, "tas_packages" => tas)
    isempty(obsp) || push!(t, "obs_packages" => obsp)

    # NPF
    ni = findfirst(p -> p[1] == "NPF6", pk)
    nbs = read_blocks(joinpath(dir, pk[ni][2]))
    nop = option_lines(nbs)
    narr = read_griddata(find_block(nbs, "GRIDDATA"), dir, _ -> (nodes, true), nlay)
    npf = options_table(nop; skip = ("save_flows", "save_specific_discharge", "print_flows", "save_saturation"))
    ict = get(narr, "icelltype", zeros(nodes))
    if ict === nothing
        push!(npf, "icelltype" => "unreadable (binary)")
        convertible = true
    else
        z, p, n = count_flags(ict, active)
        append!(npf, ["icelltype_confined" => z, "icelltype_positive" => p, "icelltype_negative" => n])
        # per layer, active cells only, run-length encoded: "1:convertible 2-5:confined"
        lay = String[]
        for k in 1:nlay
            r = ((k - 1) * ncpl + 1):(k * ncpl)
            v = ict[r][active[r]]
            push!(lay, isempty(v) ? "inactive" : all(==(0), v) ? "confined" : all(!=(0), v) ? "convertible" : "mixed")
        end
        push!(npf, "icelltype_by_layer" => runs(lay))
        convertible = p + n > 0
    end
    push!(npf, "convertible" => convertible)
    push!(npf, "arrays" => sort!(collect(keys(narr))))
    xt3d = has_option(nop, "XT3D")
    for k in ("k22", "k33")
        haskey(narr, k) || continue
        k1 = narr["k"]
        kk = narr[k]
        if k1 !== nothing && kk !== nothing
            push!(npf, "$(k)_differs_from_k" => any(i -> active[i] && kk[i] != k1[i], eachindex(kk)))
        end
    end
    for k in ("angle1", "angle2", "angle3")
        haskey(narr, k) || continue
        a = narr[k]
        a === nothing || push!(npf, "$(k)_nonzero" => any(!=(0), a[active]))
    end
    if haskey(narr, "wetdry")
        w = narr["wetdry"]
        w === nothing || push!(npf, "wetdry_nonzero" => count(i -> active[i] && w[i] != 0, eachindex(w)))
    end
    push!(t, "npf" => npf)

    # STO and steady/transient
    si = findfirst(p -> p[1] == "STO6", pk)
    periods = fill(si === nothing ? "SS" : "TR", tdis.nper)
    if si !== nothing
        sbs = read_blocks(joinpath(dir, pk[si][2]))
        sop = option_lines(sbs)
        sto = options_table(sop; skip = ("save_flows",))
        sg = find_block(sbs, "GRIDDATA")
        sarr = sg === nothing ? Dict() : read_griddata(sg, dir, _ -> (nodes, true), nlay)
        ic = get(sarr, "iconvert", nothing)
        if ic !== nothing
            z, p, n = count_flags(ic, active)
            append!(sto, ["iconvert_confined" => z, "iconvert_convertible" => p + n])
            sto_conv = p + n > 0
        else
            sto_conv = true  # MF6 initialises ICONVERT to 1 (gwf-sto.f90)
        end
        push!(sto, "arrays" => sort!(collect(keys(sarr))))
        state = "TR"
        bs = sort(find_blocks(sbs, "PERIOD"); by = b -> parse(Int, b.arg))
        bi = 1
        for kper in 1:tdis.nper
            while bi <= length(bs) && parse(Int, bs[bi].arg) == kper
                for l in bs[bi].lines
                    k = upper(l[1])
                    k == "STEADY-STATE" && (state = "SS")
                    k == "TRANSIENT" && (state = "TR")
                end
                bi += 1
            end
            periods[kper] = state
        end
        push!(t, "sto" => sto)
    end
    isempty(pkdetail) || push!(t, "boundaries" => pkdetail)
    # convertible storage only matters in transient periods
    storage_convertible = si !== nothing && sto_conv && any(==("TR"), periods)
    # features beyond a confined, steady, single-model DIS model (see REQUIRES_DOC)
    req = String[]
    grid == "DIS" || push!(req, lowercase(grid))
    convertible && push!(req, "convertible")
    storage_convertible && push!(req, "convertible_storage")
    any(==("TR"), periods) && push!(req, "transient")
    newton && push!(req, "newton")
    xt3d && push!(req, "xt3d")
    isempty(ts) || push!(req, "ts")
    isempty(tas) || push!(req, "tas")
    isempty(obsp) || push!(req, "obs")
    for (k, r) in (
            ("rewet", "rewet"), ("variablecv", "variablecv"), ("perched", "perched"),
            ("thickstrt", "thickstrt"), ("alternative_cell_averaging", "alternative_cell_averaging"),
        )
        any(p -> p[1] == k, npf) && push!(req, r)
    end
    getb(tb, k) = (i = findfirst(p -> p[1] == k, tb); i !== nothing && tb[i][2] === true)
    any(k -> getb(npf, k), ("k22_differs_from_k", "angle1_nonzero", "angle2_nonzero", "angle3_nonzero")) &&
        push!(req, "anisotropy")
    any(p -> p[2] isa Tbl && any(q -> q[1] == "nseg" && q[2] > 1, p[2]), pkdetail) && push!(req, "evt_nseg")
    return ModelScan(
        t, grid, nodes, convertible, newton, xt3d, periods,
        !isempty(ts), !isempty(tas), !isempty(obsp), storage_convertible, req, oos
    )
end

function scan_ims(path)
    bs = read_blocks(path)
    t = Tbl()
    for b in ("OPTIONS", "NONLINEAR", "LINEAR")
        for l in option_lines(bs, b)
            k = lowercase(l[1])
            k in ("print_option", "csv_output", "csv_outer_output", "csv_inner_output") && continue
            push!(t, k => length(l) == 1 ? true : join(l[2:end], " "))
        end
    end
    return t
end

function scan_exchange(dir, e)
    bs = read_blocks(joinpath(dir, e.file))
    o = option_lines(bs)
    t = Tbl("type" => e.etype, "model1" => e.m1, "model2" => e.m2)
    (v = option_value(option_lines(bs, "DIMENSIONS"), "NEXG")) === nothing || push!(t, "nexg" => parse(Int, v[1]))
    e.m1 == e.m2 && push!(t, "self_exchange" => true)
    append!(t, options_table(o; skip = ("print_input", "print_flows", "save_flows")))
    oos = String[]
    e.etype == "GWF6-GWF6" || push!(oos, "exchange $(e.etype)")
    has_option(o, "MVR6") && push!(oos, "exchange MVR6")
    return t, oos
end

function auto_tier(ms::Vector{ModelScan}, nexg)
    (length(ms) > 1 || nexg > 0) && return 7
    any(m -> m.xt3d, ms) && return 6
    any(m -> m.grid != "DIS", ms) && return 5
    any(m -> m.newton, ms) && return 4
    any(m -> m.convertible, ms) && return 3
    any(m -> m.ts || m.tas || any(==("TR"), m.periods), ms) && return 2
    return 0
end

function sim_name(rel)
    parts = splitpath(rel)
    length(parts) == 1 && return parts[1]
    sub = parts[end]
    return startswith(sub, parts[1]) ? sub : join(parts, "__")
end

function scan_sim(dir, rel)
    sim = read_sim(dir)
    name = sim_name(rel)
    oos = String[]
    family = split(splitpath(rel)[1], '-')[2]
    family == "gwf" || push!(oos, "part of a $(uppercase(family)) example")
    for m in sim.models
        m.mtype in OOS_MODELS && push!(oos, "model $(m.mtype[1:(end - 1)])")
    end
    tdis = read_tdis(joinpath(dir, sim.tdis))
    ms = ModelScan[]
    mt = Tbl()
    for m in sim.models
        m.mtype == "GWF6" || continue
        s = scan_gwf(dir, m.namfile, m.name, tdis)
        push!(ms, s)
        push!(mt, m.name => s.table)
        append!(oos, s.oos)
    end
    exs = Tbl[]
    for e in sim.exchanges
        x, o = scan_exchange(dir, e)
        push!(exs, x)
        append!(oos, o)
    end
    unique!(oos)
    if !isempty(oos)
        gwf_only = all(m -> m.mtype == "GWF6", sim.models)
        t = Tbl("path" => rel, "reason" => join(oos, "; "))
        gwf_only && push!(t, "gwf_only" => true)
        if gwf_only && !isempty(ms)
            push!(t, "packages" => sort!(unique(vcat([m.table[findfirst(p -> p[1] == "packages", m.table)][2] for m in ms]...))))
        end
        return name, false, t
    end

    t = Tbl()
    tier = get(TIERS, name, auto_tier(ms, length(exs)))
    push!(t, "path" => rel)
    push!(t, "tier" => tier)
    push!(t, "in_plan" => haskey(TIERS, name))
    haskey(NOTES, name) && push!(t, "notes" => NOTES[name])
    req = Set{String}()
    foreach(m -> union!(req, m.requires), ms)
    length(ms) > 1 && push!(req, "multi_model")
    isempty(exs) || push!(req, "exchange")
    any(x -> any(p -> p[1] == "xt3d", x), exs) && push!(req, "xt3d")
    any(x -> any(p -> p[1] == "obs6", x), exs) && push!(req, "obs")
    push!(t, "requires" => sort!(collect(req)))
    grids = unique([m.grid for m in ms])
    push!(t, "grid" => join(grids, "+"))
    push!(t, "nmodels" => length(ms))
    push!(t, "nexchanges" => length(exs))
    push!(t, "ncells" => sum(m.ncells for m in ms))
    push!(t, "packages" => sort!(unique(vcat([m.table[findfirst(p -> p[1] == "packages", m.table)][2] for m in ms]...))))
    push!(t, "convertible" => any(m -> m.convertible, ms))
    push!(t, "newton" => any(m -> m.newton, ms))
    push!(t, "storage_convertible" => any(m -> m.storage_convertible, ms))
    xt3d_npf = any(m -> m.xt3d, ms)
    xt3d_exg = any(x -> any(p -> p[1] == "xt3d", x), exs)
    push!(t, "xt3d" => xt3d_npf || xt3d_exg)
    xt3d_npf && push!(t, "xt3d_npf" => true)
    xt3d_exg && push!(t, "xt3d_exchange" => true)
    push!(t, "uses_ts" => any(m -> m.ts, ms))
    push!(t, "uses_tas" => any(m -> m.tas, ms))
    push!(t, "uses_obs" => any(m -> m.obs, ms) || any(x -> any(p -> p[1] == "obs6", x), exs))
    per = ms[1].periods
    push!(t, "transient" => any(m -> any(==("TR"), m.periods), ms))
    push!(t, "nper" => tdis.nper)
    push!(t, "nstp_total" => sum(tdis.nstp))
    push!(t, "time_units" => tdis.units)
    tdis.ats && push!(t, "ats" => true)
    push!(t, "periods" => runs(per))
    ims = Tbl[]
    for s in sim.solutions
        it = Tbl("file" => s.file, "models" => String.(s.models))
        append!(it, scan_ims(joinpath(dir, s.file)))
        push!(ims, it)
    end
    push!(t, "models" => mt)
    push!(t, "ims" => ims)
    isempty(exs) || push!(t, "exchanges" => exs)
    return name, true, t
end

function main(out = joinpath(@__DIR__, "examples.toml"))
    isdir(EXAMPLES) || error("no examples at $EXAMPLES; run validation/fetch_reference.sh or set REFERENCE_DIR")
    sims = String[]
    for (root, _, files) in walkdir(EXAMPLES)
        "mfsim.nam" in files && push!(sims, relpath(root, EXAMPLES))
    end
    sort!(sims)
    inscope = Pair{String, Tbl}[]
    oos = Pair{String, Tbl}[]
    failed = String[]
    for rel in sims
        name, ok, t = try
            scan_sim(joinpath(EXAMPLES, rel), rel)
        catch err
            @error "scan failed" rel exception = err
            push!(failed, rel)
            continue
        end
        push!(ok ? inscope : oos, name => t)
    end
    isempty(failed) || error("$(length(failed)) simulation(s) could not be scanned: $(join(failed, ", ")); examples.toml not written")
    sort!(inscope; by = p -> (p[2][findfirst(x -> x[1] == "tier", p[2])][2], p[1]))
    checksum = ""
    ck = joinpath(REFERENCE_DIR, "checksums.txt")
    if isfile(ck)
        for l in eachline(ck)
            occursin("mf6examples", l) && (checksum = split(l)[1])
        end
    end
    open(out, "w") do io
        println(io, "# MF6 example inventory for ModflowGPU (T0.3). GENERATED by")
        println(io, "# validation/make_examples_toml.jl from \$REFERENCE_DIR/mf6examples; do not edit by hand,")
        println(io, "# change the TIERS/NOTES tables in the script and regenerate.")
        println(io, "# Field meanings: docs/examples_inventory.md.")
        println(io, "")
        top = Tbl(
            "mf6examples_sha256" => checksum,
            "n_simulations" => length(sims),
            "n_in_scope" => length(inscope),
            "n_out_of_scope" => length(oos),
        )
        emit(io, top, String[])
        emit(io, Tbl("examples" => Tbl(inscope)), String[])
        emit(io, Tbl("out_of_scope" => Tbl(oos)), String[])
    end
    println("wrote $out: $(length(inscope)) in scope, $(length(oos)) out of scope, of $(length(sims)) simulations")
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end
