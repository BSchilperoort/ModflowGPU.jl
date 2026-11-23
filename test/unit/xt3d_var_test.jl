# Head-dependent XT3D and GNC6 (T6.14, src/gwf/xt3d_var.jl):
#
# 1. The factorization of `qconds`: the unscaled terms of `xt3d_var_setup` scaled with the
#    weights of two areas give `qconds` with those areas (random areas, and with a cell
#    made inactive, whose neighbours' stencils lose it: `xvar_ibound!`).
# 2. The head-dependent path on a confined model (forced with `GwfModel(...; variable =
#    true)`) gives the matrix, right-hand side, residual and flows of the constant path
#    (T6.5 XT3D on test/models/xt3d, T6.13 GNC on gnc_disv).
# 3. libmf6 oracle (first solve): the system structure equals MF6's and AMAT, b' match for
#    convertible XT3D (standard formulation, with HFB6), Newton XT3D (`xt3d_fn`) and GNC6
#    with NEWTON (`gnc_fn`) and HFB6 (models without PTC in the first solve).
# 4. Run parity against mf6 (run at test time) in `:single`, `:mixed` and `:double` for the
#    cases of test/models/xt3d_var/variants.jl: the MF6 autotest
#    test_gwf_xt3d_newton_upstream.py (xt3dnr_*), convertible XT3D on DIS (drying and
#    rewetting; NEWTON; RHS) and DISV (with and without NEWTON), XT3D with HFB6 (convertible,
#    NEWTON, confined) and with TVK6, GNC6 with convertible cells (implicit, EXPLICIT,
#    NEWTON), with HFB6 and with TVK6.
using ModflowGPU: load_simulation, build_connectivity, npf_setup, GwfModel, apply_period!, start_timestep!, formulate!,
    calc_flows!, time_steps, time_step_window, backend_name, xt3d_var_setup, Xt3dVarSetup, Xt3dVarNpf

const XV_MF6 = get(ENV, "MF6", joinpath(REFERENCE_DIR, "mf6.8.2_linux", "bin", "mf6"))
include(joinpath(@__DIR__, "..", "models", "xt3d_var", "variants.jl"))

# The coefficients of connection `j` (n < m) from the unscaled terms with areas ar01, ar10
# (the kernel's `qconds` weighting), as (chat01, chati0 per neighbour of n, chat1j per
# neighbour of m).
function _xv_scaled(xs::Xt3dVarSetup, conn, n, m, j, ar01, ar10)
    t1, t2 = xs.tptr[j], xs.tptr[j + 1] - 1
    f0 = f1 = 0.0
    if ar01 != 0.0
        d0 = xs.xa0[j] * ar01
        den = d0 + xs.xa1[j] * ar10
        w1 = abs(den) > eps(Float64) ? d0 / den : 1.0
        f0, f1 = (1.0 - w1) * ar01, w1 * ar10
    end
    c = [xs.tc[t] * (xs.tside[t] == 0 ? f0 : f1) for t in t1:t2]
    nn = Int(conn.ia[n + 1] - conn.ia[n] - 1)
    return c[1], c[2:nn], -c[(nn + 1):end]
end

@testset "XT3D var: the unscaled terms reproduce qconds" begin
    spec = only(load_simulation(joinpath(XT3D_VAR_MODELS, "xt3d_var", "xt3d_conv")).models)
    conn = build_connectivity(spec)
    setup = npf_setup(spec, conn)
    p = setup.properties
    xs = xt3d_var_setup(conn, p, setup.r)
    g = _Lcg(UInt64(3))
    check(ibound) = begin
        err = 0.0
        for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
            m = conn.ja[ii]
            (m > n && ibound[n] != 0 && ibound[m] != 0) || continue
            j = conn.jas[ii]
            c0 = ModflowGPU._xt3d_load(conn, p, ibound, true, n)
            c1 = ModflowGPU._xt3d_load(conn, p, ibound, true, m)
            il0 = ii - conn.ia[n]
            il10 = findfirst(==(n), conn.ja[(conn.ia[m] + 1):(conn.ia[m + 1] - 1)])
            ar01, ar10 = 100.0 * (0.1 + _next!(g, 10)), 100.0 * (0.1 + _next!(g, 10))
            a, bi, bj = ModflowGPU._xt3d_qconds(c0, il0, c1, il10, ar01, ar10, ModflowGPU._XT3D_VCTHRESH)
            o = _xv_scaled(xs, conn, n, m, j, ar01, ar10)
            # The terms skip the primary neighbour; qconds' vectors hold 0 there.
            bi2 = deleteat!(copy(bi[1:c0.nnbr]), il0)
            bj2 = deleteat!(copy(bj[1:c1.nnbr]), il10)
            sc = abs(a) + sum(abs, bi; init = 0.0) + sum(abs, bj; init = 0.0)
            err = max(err, abs(o[1] - a) / sc, maximum(abs.(o[2] .- bi2); init = 0.0) / sc, maximum(abs.(o[3] .- bj2); init = 0.0) / sc)
        end
        err
    end
    @test check(xs.ibound) < 1.0e-13
    # A cell made inactive leaves the stencils of its neighbours (`xt3d_load`):
    # `xvar_ibound!` recomputes their connections.
    x = Xt3dVarNpf(CPU(), Float64, xs)
    ib = copy(xs.ibound)
    k = 30
    ib[k] = 0
    old = copy(xs.tc)
    @test ModflowGPU.xvar_ibound!(xs, x, conn, p, ib)
    @test xs.ibound == ib && xs.tc != old && Array(x.tc) == xs.tc
    @test check(ib) < 1.0e-13
    ib[k] = 1
    @test ModflowGPU.xvar_ibound!(xs, x, conn, p, ib)
    @test xs.tc == old
end

# The forced head-dependent path against the constant one on a confined model.
function _xv_same_as_constant(name, dir)
    sim = load_simulation(dir)
    spec = only(sim.models)
    steps = time_steps(sim.tdis)
    return for_each_precision() do p
        T = p.T
        ms = [GwfModel(TEST_BACKEND, T, spec, length(sim.tdis.perlen); variable = v) for v in (false, true)]
        for m in ms
            apply_period!(m, 1; window = time_step_window(steps, 1))
            start_timestep!(m, steps[1].delt)
            formulate!(m)
            calc_flows!(m)
        end
        KernelAbstractions.synchronize(TEST_BACKEND)
        a, b = ms
        @test b.xvar isa Xt3dVarSetup && isnothing(a.xvar)
        @test Array(a.scsr.ia) == Array(b.scsr.ia) && Array(a.scsr.ja) == Array(b.scsr.ja)
        f(x) = Float64.(Array(x))
        sa = maximum(abs, f(a.sys.amat))
        ea = maximum(abs.(f(a.sys.amat) .- f(b.sys.amat))) / sa
        eb = maximum(abs.(f(a.sys.rhs) .- f(b.sys.rhs))) / max(maximum(abs, f(a.sys.rhs)), sa)
        eq = maximum(abs.(f(a.flowja) .- f(b.flowja))) / maximum(abs, f(a.flowja))
        er = maximum(abs.(f(a.res) .- f(b.res))) / maximum(abs, f(a.flowja))
        tol = T === Float64 ? 1.0e-12 : 1.0e-5
        @test ea < tol
        @test eb < tol
        @test eq < tol
        @test er < tol
        @info "XT3D var vs constant path: $name, :$(p.preset): amat $(round(ea; sigdigits = 3)), rhs $(round(eb; sigdigits = 3)), flowja $(round(eq; sigdigits = 3)), res $(round(er; sigdigits = 3))"
    end
end

@testset "XT3D var: a confined model gives the constant path" begin
    _xv_same_as_constant("xt3d", joinpath(XT3D_VAR_MODELS, "xt3d"))
    _xv_same_as_constant("gnc_disv", joinpath(XT3D_VAR_MODELS, "gnc", "gnc_disv"))
end

# libmf6's first-solve system against `formulate!` (with the Newton terms of a NEWTON model).
function _xv_oracle(name, simdir)
    sim = load_simulation(simdir)
    spec = only(sim.models)
    o = _libmf6_first_solve(simdir)
    amax = maximum(abs, o.amat)
    steps = time_steps(sim.tdis)
    return for_each_precision() do p
        T = p.T
        m = GwfModel(TEST_BACKEND, T, spec, length(sim.tdis.perlen))
        @test m.xvar isa Xt3dVarSetup
        r = m.setup.r
        mf = Dict{Tuple{Int, Int}, Float64}()
        bmf = copy(o.rhs)
        bscale = abs.(o.rhs)
        for n in 1:m.conn.nodes, ii in o.ia[n]:(o.ia[n + 1] - 1)
            mf[(n, Int(o.ja[ii]))] = o.amat[ii]
            t = o.amat[ii] * r[o.ja[ii]]
            bmf[n] -= t
            bscale[n] += abs(t)
        end
        apply_period!(m, 1; window = time_step_window(steps, 1))
        start_timestep!(m, o.delt)
        formulate!(m)
        KernelAbstractions.synchronize(TEST_BACKEND)
        ia, ja, a = Array(m.scsr.ia), Array(m.scsr.ja), Float64.(Array(m.sys.amat))
        @test ia == o.ia && ja == o.ja
        ours = Dict((n, Int(ja[ii])) => a[ii] for n in 1:m.conn.nodes for ii in ia[n]:(ia[n + 1] - 1))
        ea = maximum(k -> abs(get(ours, k, 0.0) - get(mf, k, 0.0)), union(keys(ours), keys(mf))) / amax
        eb = maximum(abs.(Float64.(Array(m.sys.rhs)) .- bmf)) / maximum(bscale)
        tol = T === Float64 ? 1.0e-12 : 1.0e-6
        @test ea < tol
        @test eb < tol
        @info "XT3D var oracle vs libmf6 (first solve): $name, :$(p.preset), $(backend_name(TEST_BACKEND)), $(length(ja)) entries: amat max |Δ| / max |a| = $(round(ea; sigdigits = 3)), b' max |Δ| / max(|RHS| + Σ|a·r|) = $(round(eb; sigdigits = 3))"
    end
end

# No pseudo-transient continuation in the first solve (the oracle compares the system
# without it).
_xv_noptc(f, t) = endswith(f, ".ims") ? replace(t, r"BEGIN options\n"i => "BEGIN options\n  NO_PTC  ALL\n", count = 1) : t

function _xv_edit(dir, edit)
    for f in readdir(dir)
        _xv_input(f) || continue
        write(joinpath(dir, f), edit(f, read(joinpath(dir, f), String)))
    end
    return dir
end

@testset "XT3D var: libmf6 oracle (first solve)" begin
    if !libmf6_available()
        @info "Skipping the XT3D var oracle: libmf6 environment not found at $LIBMF6_ENV"
    else
        for name in ("xt3dnr_cont", "xt3d_conv", "xt3d_hfb_conv", "xt3d_hfb_conv_newton", "xt3d_disv_conv_newton", "gnc_disv_conv_newton", "gnc_disv_hfb")
            test_model_selected(name) || continue
            mktempdir() do tmp
                dir = _xv_edit(xt3d_var_case(name, joinpath(tmp, name)), _xv_noptc)
                @testset "$name" begin
                    _xv_oracle(name, dir)
                end
            end
        end
    end
end

@testset "XT3D var: run parity against mf6" begin
    for (name, _) in xt3d_var_cases()
        if !test_model_selected(name)
        elseif !isfile(XV_MF6)
            @info "Skipping $name: mf6 not found at $XV_MF6"
        else
            mktempdir() do tmp
                dir = xt3d_var_case(name, joinpath(tmp, name))
                out = read(Cmd(`$XV_MF6`; dir), String)
                @test occursin("Normal termination", out)
                @testset "$name" begin
                    _run_parity(name, dir)
                end
            end
        end
    end
end
