# XT3D (T6.5, src/gwf/xt3d.jl):
#
# 1. Host coefficients on a K-orthogonal DIS grid (diagonal K tensor, axes along the grid):
#    XT3D reduces to the two-point flux, so `chat01` equals the harmonic-mean conductance
#    `condsat` of the standard formulation and every other coefficient vanishes; a run with
#    XT3D gives the heads of the run without it.
# 2. Options and errors: XT3D with ANGLE2 and convertible cells is unsupported (T6.14:
#    convertible cells and NEWTON otherwise supported, group xt3d_var), CG with XT3D
#    is MF6's input error, a DISU grid without vertices is MF6's error.
# 3. For the Tier 6 examples (ex-gwf-hanix, ex-gwf-whirl on DIS; ex-gwf-u1disv-x,
#    ex-gwf-disvmesh on DISV; reference runs in reference/runs) and ex-gwf-hanix with the
#    XT3D RHS option (run with mf6 at test time), in `:single` and `:double`:
#    - universal residual: MF6's converged heads through `formulate!` of a GwfModel give
#      max |res| / max Σ|terms| < 1e-8 (`:double`; `:single`: the rounding bound), and the
#      fused formulation is consistent: `rhs − A·h'` (Float64, from the assembled matrix)
#      equals the residual within 1e-12 of the row magnitudes (`:double`);
#    - connection flows: `calc_flows!` with MF6's heads against MF6's FLOW-JA-FACE;
#    - oracle: libmf6's first-solve system (validation/libmf6_first_solve.jl): the system
#      structure (MF6's extended stencil with XT3D, the connectivity with RHS) equals ours,
#      and AMAT and `b' = RHS − AMAT·r` match within 1e-12 (`:double`) or 1e-6 (`:single`).
#    The same for the synthetic model test/models/xt3d (ANGLE1–3, so horizontal connections
#    have a z component; uneven layers, an excluded cell, transient storage; mf6 at test time).
# 4. Run parity of the RHS variant and of test/models/xt3d against mf6 (the Tier 6 examples
#    themselves are in the run parity test, group run).
using ModflowGPU: load_simulation, build_connectivity, npf_setup, npf_properties, xt3d_setup, GwfModel, apply_period!,
    start_timestep!, formulate!, calc_flows!, time_steps, time_step_window, MF6UnsupportedError, MF6InputError,
    open_head_file, read_head, open_budget_file, read_budget, find_budget_records, backend_name

const XT3D_EXAMPLES = ("ex-gwf-hanix", "ex-gwf-whirl", "ex-gwf-u1disv-x", "ex-gwf-disvmesh")
const XT3D_MF6 = get(ENV, "MF6", joinpath(REFERENCE_DIR, "mf6.8.2_linux", "bin", "mf6"))

# A copy of `src` (an MF6 simulation directory, inputs only) with `edit(name, text)` applied
# to every input file.
function _xt3d_variant(src, dst; edit)
    mkpath(dst)
    for f in readdir(src)
        any(e -> endswith(f, e), (".hds", ".cbc", ".lst", ".txt", ".toml", ".grb", ".csv")) && continue
        txt = read(joinpath(src, f), String)
        write(joinpath(dst, f), edit(f, txt))
    end
    return dst
end

# The XT3D RHS variant of ex-gwf-hanix (reference inputs, tight closures).
function _hanix_rhs(dst)
    src = reference_rundir("ex-gwf-hanix")
    isnothing(src) && return nothing
    return _xt3d_variant(src, dst; edit = (f, t) -> endswith(f, ".npf") ? replace(t, r"^(\s*)XT3D\s*$"m => s"\1XT3D  RHS") : t)
end

@testset "XT3D: K-orthogonal DIS grid reduces to the two-point flux" begin
    ncol, nrow, nlay = 5, 4, 3
    g = _Lcg(UInt64(7))
    dis = _dis(;
        delr = [10.0 + _next!(g, 10) for _ in 1:ncol], delc = [8.0 + _next!(g, 5) for _ in 1:nrow],
        top = fill(10.0, ncol, nrow), botm = cat(fill(0.0, ncol, nrow), fill(-12.0, ncol, nrow), fill(-20.0, ncol, nrow); dims = 3),
    )
    nu = ncol * nrow * nlay
    k = [0.5 + _next!(g, 20) / 4 for _ in 1:nu]
    k22 = [0.5 + _next!(g, 20) / 4 for _ in 1:nu]
    k33 = [0.1 + _next!(g, 10) / 10 for _ in 1:nu]
    npf = ModflowGPU.NpfSpec(; icelltype = zeros(Int32, nu), k, k22, k33, xt3d = true, file = "test.npf")
    conn = build_connectivity(dis)
    ic = ModflowGPU.IcSpec(; strt = zeros(nu))
    s = npf_setup(npf, ic, dis, conn)
    x = xt3d_setup(conn, s.properties, s.r)
    @test x.tptr[end] - 1 == length(x.tc)
    worst, cross = 0.0, 0.0
    for j in 1:conn.njas
        t1, t2 = x.tptr[j], x.tptr[j + 1] - 1
        worst = max(worst, abs(x.tc[t1] - s.condsat[j]) / s.condsat[j])
        for t in (t1 + 1):t2
            cross = max(cross, abs(x.tc[t]) / s.condsat[j])
        end
    end
    @test worst < 1.0e-12
    @test cross < 1.0e-12
    # The extended stencil holds the neighbours of neighbours (xt3d_ac): the cells within two
    # steps of the connectivity graph; a cell in the middle layer, row 2, column 3 has 6
    # neighbours and 15 cells two steps away (18 in an unbounded grid).
    within2(n) = Set(k for ii in conn.ia[n]:(conn.ia[n + 1] - 1) for jj in conn.ia[conn.ja[ii]]:(conn.ia[conn.ja[ii] + 1] - 1) for k in (conn.ja[ii], conn.ja[jj]))
    @test all(n -> Set(x.ja[x.ia[n]:(x.ia[n + 1] - 1)]) == within2(n), 1:conn.nodes)
    n = (2 - 1) * ncol * nrow + (2 - 1) * ncol + 3
    @test x.ia[n + 1] - x.ia[n] == 1 + 6 + 15
    @test x.isym[x.isym] == collect(Int32, 1:length(x.ja))
    # Rows of the XT3D matrix sum to zero (flux form), and so do the columns (conservation).
    rs = [sum(x.amat[x.ia[n]:(x.ia[n + 1] - 1)]) for n in 1:conn.nodes]
    cs = zeros(conn.nodes)
    for n in 1:conn.nodes, ii in x.ia[n]:(x.ia[n + 1] - 1)
        cs[x.ja[ii]] += x.amat[ii]
    end
    @test maximum(abs, rs) < 1.0e-12 * maximum(abs, x.amat)
    @test maximum(abs, cs) < 1.0e-12 * maximum(abs, x.amat)
end

@testset "XT3D: options and errors" begin
    dis = _dis(; delr = fill(10.0, 3), delc = fill(10.0, 2), top = fill(10.0, 3, 2), botm = fill(0.0, 3, 2, 1))
    conn = build_connectivity(dis)
    base = (; k = ones(6), xt3d = true, file = "t.npf")
    # T6.14: convertible cells and NEWTON are supported; with ANGLE2 the horizontal
    # connection vectors of convertible cells depend on the saturation: unsupported.
    @test npf_properties(ModflowGPU.NpfSpec(; icelltype = Int32[1, 0, 0, 0, 0, 0], base...), dis, conn).icelltype[1] == 1
    @test npf_properties(ModflowGPU.NpfSpec(; icelltype = zeros(Int32, 6), base...), dis, conn; newton = true) isa ModflowGPU.NpfProperties
    a = (; angle1 = zeros(6), angle2 = zeros(6), angle3 = zeros(6))
    e = try
        npf_properties(ModflowGPU.NpfSpec(; icelltype = Int32[1, 0, 0, 0, 0, 0], base..., a...), dis, conn)
    catch err
        err
    end
    @test e isa MF6UnsupportedError && occursin("XT3D with ANGLE2 and convertible cells", sprint(showerror, e))
    @test npf_properties(ModflowGPU.NpfSpec(; icelltype = zeros(Int32, 6), base..., a...), dis, conn).angle2 == zeros(6)
    # MF6's own option checks still apply.
    e = try
        npf_properties(ModflowGPU.NpfSpec(; icelltype = zeros(Int32, 6), cell_averaging = :logarithmic, base...), dis, conn)
    catch err
        err
    end
    @test e isa MF6InputError && occursin("ALTERNATIVE_CELL_AVERAGING OPTION CANNOT BE USED WITH XT3D", sprint(showerror, e))
    # DISU without vertices (Xt3dInterface.f90 `xt3d_ar`).
    du = _dis_as_disu(dis; vertices = false)
    e = try
        npf_properties(ModflowGPU.NpfSpec(; icelltype = zeros(Int32, 6), base...), du, build_connectivity(du))
    catch err
        err
    end
    @test e isa MF6InputError && occursin("Vertices not specified for discretization package, but XT3D is active", sprint(showerror, e))
    # CG with XT3D (NumericalSolution.f90 `sln_ar`).
    src = reference_rundir("ex-gwf-u1disv-x")
    if isnothing(src)
        @info "Skipping the XT3D CG test: no reference run of ex-gwf-u1disv-x"
    else
        mktempdir() do tmp
            dir = _xt3d_variant(src, joinpath(tmp, "cg"); edit = (f, t) -> endswith(f, ".ims") ? replace(t, r"bicgstab"i => "CG") : t)
            e = try
                ModflowGPU.initialize(dir; write_output = false)
            catch err
                err
            end
            @test e isa MF6InputError && occursin("ASYMMETRIC COEFFICIENT MATRIX", sprint(showerror, e))
        end
    end
end

# Universal residual, flows and consistency of the fused formulation with MF6's heads.
function _xt3d_residual(name, rundir)
    sim = load_simulation(rundir)
    spec = only(sim.models)
    conn = build_connectivity(spec)
    hf = open_head_file(only(filter(f -> endswith(f, ".hds"), readdir(rundir; join = true))))
    habs = vec(read_head(hf, length(hf)))[conn.nodeuser]
    # The old heads of a transient last step: the step before (every step saved).
    hprev = vec(read_head(hf, max(length(hf) - 1, 1)))[conn.nodeuser]
    bf = open_budget_file(only(filter(f -> endswith(f, ".cbc"), readdir(rundir; join = true))))
    fj = vec(read_budget(bf, last(find_budget_records(bf; text = "FLOW-JA-FACE"))))
    steps = time_steps(sim.tdis)
    st = last(steps)
    return for_each_precision() do p
        T = p.T
        m = GwfModel(TEST_BACKEND, T, spec, length(sim.tdis.perlen))
        for kper in 1:st.kper
            apply_period!(m, kper; window = time_step_window(steps, findfirst(s -> s.kper == kper, steps)))
        end
        start_timestep!(m, st.delt)
        copyto!(m.h, T.(habs .- m.setup.r))
        copyto!(m.hold, T.(hprev .- m.setup.r))
        scale = KernelAbstractions.zeros(TEST_BACKEND, T, conn.nodes)
        formulate!(m; scale)
        calc_flows!(m)
        KernelAbstractions.synchronize(TEST_BACKEND)
        res, sc = Float64.(Array(m.res)), Float64.(Array(scale))
        rel = maximum(abs, res) / maximum(sc)
        # `:single`: the rounding of the offset heads and reference differences to T through
        # the XT3D terms, Σ_t |c_t|·(|h'_a| + |h'_b| + |Δr_t|) over the row's connections.
        x = xt3d_setup(conn, m.setup.properties, m.setup.r; rhs = spec.npf.xt3d_rhs)
        hoff = Float64.(Array(m.h))
        # plus the boundary and storage terms |brhs| + |bdiag·h'|.
        rb = abs.(Float64.(Array(m.sys.brhs))) .+ abs.(Float64.(Array(m.sys.bdiag)) .* hoff)
        for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1), t in x.tptr[conn.jas[ii]]:(x.tptr[conn.jas[ii] + 1] - 1)
            rb[n] += abs(x.tc[t]) * (abs(hoff[x.ta[t]]) + abs(hoff[x.tb[t]]) + abs(x.tdr[t]))
        end
        # As the universal residual test (group assembly): 1e-8 of the largest row magnitude
        # (MF6's heads leave that much), in `:single` plus 4 eps of each row's rounding scale
        # (ex-gwf-whirl, -disvmesh: 1.6e-6 and 1.2e-6 relative to the largest row).
        excess = maximum((abs.(res) .- 1.0e-8 * maximum(sc)) ./ (rb .+ floatmin()))
        @test T === Float64 ? rel < 1.0e-8 : excess <= 4 * eps(T)
        rbound = excess
        # rhs − A·h' (Float64) against the residual.
        ia, ja = Array(m.scsr.ia), Array(m.scsr.ja)
        a, b, h = Float64.(Array(m.sys.amat)), Float64.(Array(m.sys.rhs)), Float64.(Array(m.h))
        cons = 0.0
        for n in 1:conn.nodes
            m.ibound_host[n] > 0 || continue
            ah = sum(a[ii] * h[ja[ii]] for ii in ia[n]:(ia[n + 1] - 1))
            cons = max(cons, abs(b[n] - ah - res[n]))
        end
        T === Float64 && @test cons <= 1.0e-12 * maximum(sc)
        # FLOW-JA-FACE with MF6's heads.
        ours = Float64.(Array(m.flowja))
        offd = [ii for n in 1:conn.nodes for ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)]
        dq = maximum(abs.(ours[offd] .- fj[offd])) / maximum(abs, fj[offd])
        @test dq < (T === Float64 ? 1.0e-10 : 1.0e-4)
        @test all(ours[offd] .== .-ours[conn.isym[offd]])
        @info "XT3D universal residual: $name, :$(p.preset), $(backend_name(TEST_BACKEND)), $(conn.nodes) nodes, stencil $(length(ja)) entries (connectivity $(conn.nja)): max |res| / max Σ|terms| = $(round(rel; sigdigits = 3)) (|res| beyond 1e-8: ≤ $(round(max(rbound, 0.0) / eps(T); sigdigits = 3)) eps of the rounding scale), rhs − A·h' vs res $(round(cons / maximum(sc); sigdigits = 3)), FLOW-JA-FACE vs MF6 $(round(dq; sigdigits = 3))"
    end
end

# libmf6's first-solve system against `formulate!` of a GwfModel.
function _xt3d_oracle(name, simdir)
    sim = load_simulation(simdir)
    spec = only(sim.models)
    conn = build_connectivity(spec)
    o = _libmf6_first_solve(simdir)
    steps = time_steps(sim.tdis)
    return for_each_precision() do p
        T = p.T
        m = GwfModel(TEST_BACKEND, T, spec, length(sim.tdis.perlen))
        @test Array(m.scsr.ia) == o.ia && Array(m.scsr.ja) == o.ja
        r = m.setup.r
        bmf = copy(o.rhs)
        bscale = abs.(o.rhs)
        for n in 1:conn.nodes, ii in o.ia[n]:(o.ia[n + 1] - 1)
            t = o.amat[ii] * r[o.ja[ii]]
            bmf[n] -= t
            bscale[n] += abs(t)
        end
        apply_period!(m, 1; window = time_step_window(steps, 1))
        start_timestep!(m, o.delt)
        formulate!(m)
        KernelAbstractions.synchronize(TEST_BACKEND)
        @test maximum(abs.(r .+ Float64.(Array(m.h)) .- o.x0)) <= 4 * eps(T) * max(maximum(abs, o.x0), 1.0)
        ea = maximum(abs.(Float64.(Array(m.sys.amat)) .- o.amat)) / maximum(abs, o.amat)
        eb = maximum(abs.(Float64.(Array(m.sys.rhs)) .- bmf)) / maximum(bscale)
        tol = T === Float64 ? 1.0e-12 : 1.0e-6
        @test ea < tol
        @test eb < tol
        @info "XT3D oracle vs libmf6 (first solve): $name, :$(p.preset), $(backend_name(TEST_BACKEND)), $(length(o.ja)) entries: amat max |Δ| / max |a| = $(round(ea; sigdigits = 3)), b' max |Δ| / max(|RHS| + Σ|a·r|) = $(round(eb; sigdigits = 3))"
    end
end

@testset "XT3D: Tier 6 examples (residual, flows, libmf6 oracle)" begin
    have_libmf6 = libmf6_available()
    have_libmf6 || @info "Skipping the XT3D oracle vs libmf6: libmf6 environment not found at $LIBMF6_ENV"
    for name in XT3D_EXAMPLES
        test_model_selected(name) || continue
        rundir = reference_rundir(name)
        if isnothing(rundir)
            @info "Skipping $name in the XT3D tests: no reference run in $(joinpath(REFERENCE_DIR, "runs", name))"
            continue
        end
        @testset "$name" begin
            _xt3d_residual(name, rundir)
            have_libmf6 && _xt3d_oracle(name, rundir)
        end
    end
    # The synthetic model test/models/xt3d (ANGLE1-3, uneven layers, IDOMAIN 0, storage).
    name = "xt3d"
    if !test_model_selected(name)
    elseif !isfile(XT3D_MF6)
        @info "Skipping $name: mf6 not found at $XT3D_MF6"
    else
        mktempdir() do tmp
            dir = joinpath(tmp, name)
            cp(joinpath(@__DIR__, "..", "models", name), dir)
            chmod(dir, 0o755; recursive = true)
            inputs = joinpath(tmp, "inputs")
            cp(dir, inputs)
            out = read(Cmd(`$XT3D_MF6`; dir), String)
            @test occursin("Normal termination", out)
            @testset "$name" begin
                _xt3d_residual(name, dir)
                have_libmf6 && _xt3d_oracle(name, inputs)
                _run_parity(name, dir; presets = TEST_PRESETS)
            end
        end
    end
    name = "ex-gwf-hanix-rhs"
    if !test_model_selected(name)
    elseif !isfile(XT3D_MF6)
        @info "Skipping $name: mf6 not found at $XT3D_MF6"
    else
        mktempdir() do tmp
            dir = _hanix_rhs(joinpath(tmp, name))
            if isnothing(dir)
                @info "Skipping $name: no reference run of ex-gwf-hanix"
            else
                @test occursin("XT3D  RHS", read(joinpath(dir, "ex-gwf-hanix.npf"), String))
                inputs = _hanix_rhs(joinpath(tmp, "inputs"))
                out = read(Cmd(`$XT3D_MF6`; dir), String)
                @test occursin("Normal termination", out)
                @testset "$name" begin
                    _xt3d_residual(name, dir)
                    have_libmf6 && _xt3d_oracle(name, inputs)
                    # Run parity (the matrix holds the primary terms only; the other terms are
                    # in the residual, so the converged heads are the same).
                    _run_parity(name, dir; presets = TEST_PRESETS)
                end
            end
        end
    end
end
