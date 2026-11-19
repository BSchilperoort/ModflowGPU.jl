# Newton–Raphson (T6.2): the smoothing functions against MF6's formulas, the Newton matrix
# against a finite-difference Jacobian of the flux-form residual (NPF upstream conductances,
# convertible storage, WEL AUTO_FLOW_REDUCE, DRN AUXDEPTHNAME), the bottoms of NEWTON
# UNDER_RELAXATION, pseudo-transient continuation, and the input checks. Against mf6:
# the Tier 4 examples and test/models/newton (groups run and assembly).
using ModflowGPU: newton_saturation, newton_saturation_derivative, newton_satthk, sq_saturation, sq_saturation_derivative,
    load_simulation, GwfModel, apply_period!, start_timestep!, formulate!, newton_ur_bottoms, ptc_allowed, apply_ptc!,
    PtcState, Reducer, accumulation_type, MF6InputError

# SmoothingFunctions.f90 sQuadraticSaturation / sQuadraticSaturationDerivative, in Float64.
function _quad_sat(top, bot, x, eps = 1.0e-6)
    b = top - bot
    b > 0 || return x < bot ? 0.0 : 1.0
    br = x < bot ? 0.0 : x > top ? 1.0 : (x - bot) / b
    av = 1 / (1 - eps)
    bri = 1 - br
    br < eps && return av * 0.5 * br^2 / eps
    br < 1 - eps && return av * br + 0.5 * (1 - av)
    br < 1 && return 1 - av * 0.5 * bri^2 / eps
    return 1.0
end
function _quad_dsat(top, bot, x, eps = 1.0e-6)
    b = top - bot
    br = x < bot ? 0.0 : x > top ? 1.0 : (x - bot) / b
    av = 1 / (1 - eps)
    y = br < eps ? av * br / eps : br < 1 - eps ? av : br < 1 ? av * (1 - br) / eps : 0.0
    return y / b
end

@testset "Newton: smoothing functions (sQuadraticSaturation, sQSaturation)" begin
    # Offsets: top' 5, bottom' −5 (thickness 10); the smoothing intervals are 1e-5 wide.
    xs = [-6.0, -5.0, -5 + 2.0e-6, -5 + 9.9e-6, -2.5, 0.0, 4.99999, 5 - 3.0e-6, 5.0, 6.0]
    for_each_precision() do p
        T = p.T
        tol = T === Float64 ? 1.0e-14 : 2.0e-6
        for x in xs
            @test newton_saturation(T(5), T(-5), T(x)) ≈ _quad_sat(5.0, -5.0, Float64(T(x))) atol = tol
            # In Float32 `br = (x − bot)/b` next to the top resolves the 1e-6 interval with
            # 16 ulps only: the derivative there is MF6's formula on a rounded `br`.
            d = newton_saturation_derivative(T(5), T(-5), T(x))
            if T === Float32 && 4.99 < x < 5
                @test 0 <= d <= T(0.1000001)
            else
                @test d ≈ _quad_dsat(5.0, -5.0, Float64(T(x))) rtol = 10 * tol atol = tol
            end
            # thk·S without the division.
            @test newton_satthk(T(5), T(-5), T(10), T(x)) ≈ 10 * _quad_sat(5.0, -5.0, Float64(T(x))) atol = 20 * tol
        end
        # A flat cell: a step at the bottom.
        @test newton_saturation(T(1), T(1), T(0.5)) == zero(T) && newton_saturation(T(1), T(1), T(1)) == one(T)
        # sQSaturation and its derivative (c1 = −2, c2 = 3; DRN with Newton: −1, 2).
        for (c1, c2) in ((-2.0, 3.0), (-1.0, 2.0)), x in (-0.1, 0.0, 0.3, 0.7, 1.0, 1.2)
            f(z) = sq_saturation(1.0, 0.0, z, c1, c2)
            @test sq_saturation(T(1), T(0), T(x), T(c1), T(c2)) ≈ T(f(x))
            d = 0 < x < 1 ? (f(x + 1.0e-7) - f(x - 1.0e-7)) / 2.0e-7 : (x == 0 ? 0.0 : 0.0)
            @test sq_saturation_derivative(T(1), T(0), T(x), T(c1), T(c2)) ≈ T(d) atol = 1.0e-5
        end
    end
end

# test/models/newton without HIGHEST_CELL_SATURATION (MF6's Jacobian is not that of the
# conductances then: it takes the derivative of the upstream cell's own saturation).
function _newton_model_dir(dir; highest = false)
    cp(joinpath(@__DIR__, "..", "models", "newton"), dir)
    npf = joinpath(dir, "nwt.npf")
    highest || write(npf, replace(read(npf, String), "  HIGHEST_CELL_SATURATION\n" => ""))
    return dir
end

@testset "Newton: matrix vs the finite-difference Jacobian of the residual" begin
    mktempdir() do tmp
        sim = load_simulation(_newton_model_dir(joinpath(tmp, "nwt")))
        spec = only(sim.models)
        m = GwfModel(CPU(), Float64, spec, 2)
        conn = m.conn
        n = Int(conn.nodes)
        # Period 2 (transient, wells), heads between 5 % and 95 % of each cell, plus noise:
        # no ties between neighbours, every term on a smooth branch.
        apply_period!(m, 1)
        apply_period!(m, 2)
        start_timestep!(m, 10.0)
        frac = [0.05 + 0.9 * mod(0.37 * k + 0.11 * k^2, 1.0) for k in 1:n]
        habs = conn.bot .+ frac .* (conn.top .- conn.bot)
        hold = habs .+ [0.3 * sin(k) for k in 1:n]
        copyto!(m.hold, hold .- m.setup.r)
        h0 = habs .- m.setup.r
        ib = Array(m.ibound)
        residual(h) = (copyto!(m.h, h); formulate!(m); copy(Array(m.res)))
        res0 = residual(h0)
        A = zeros(n, n)
        amat = Array(m.sys.amat)
        for r in 1:n, ii in conn.ia[r]:(conn.ia[r + 1] - 1)
            A[r, conn.ja[ii]] = amat[ii]
        end
        J = zeros(n, n)
        ε = 1.0e-6
        for k in 1:n
            hp = copy(h0); hp[k] += ε
            hm = copy(h0); hm[k] -= ε
            J[:, k] = -(residual(hp) .- residual(hm)) ./ (2ε)
        end
        act = ib .> 0
        err = maximum(abs.(A[act, :] .- J[act, :])) / maximum(abs, A[act, :])
        @test err < 1.0e-6
        # The Newton terms are there: the matrix is not symmetric.
        @test maximum(abs.(A[act, act] .- A[act, act]')) > 1.0e-3 * maximum(abs, A[act, :])
        # The residual does not depend on the Newton terms of NPF, WEL and DRN; the SY term
        # of the backtracking formulation (`jac = false`, `SyTerms`) differs.
        copyto!(m.h, h0)
        formulate!(m; jac = false)
        @test maximum(abs.(Array(m.res) .- res0)) > 0
        @info "Newton Jacobian vs finite differences (test/models/newton, period 2): max |Δ| / max |a| = $(round(err; sigdigits = 3))"
    end
end

@testset "Newton: NEWTON UNDER_RELAXATION bottoms, PTC, input checks" begin
    mktempdir() do tmp
        dir = _newton_model_dir(joinpath(tmp, "nwt"); highest = true)
        sim = load_simulation(dir)
        spec = only(sim.models)
        @test spec.newton && spec.newton_under_relaxation && spec.npf.highest_cell_saturation
        m = GwfModel(CPU(), Float64, spec, 2)
        conn = m.conn
        # Two convertible layers: the bottom of every layer-1 cell is that of layer 2 (0).
        b = newton_ur_bottoms(conn, m.setup)
        for n in 1:conn.nodes
            @test b[n] ≈ 0.0 - m.setup.r[n]
        end
        @test Array(m.nurbot) ≈ b
        @test isnothing(m.wd)  # Newton: no wetting and drying
        # Pseudo-transient continuation (sln_ls): 1/ptcdel = max |res|/V in the first outer
        # iteration; skipped in the first iteration of a later step when the norm did not grow.
        @test ptc_allowed(:none, 1) && !ptc_allowed(:all, 2) && !ptc_allowed(:first, 1) && ptc_allowed(:first, 2)
        apply_period!(m, 1)
        start_timestep!(m, 1.0)
        formulate!(m)
        res = Array(m.res)
        d0 = Array(m.sys.diag)
        red = Reducer(CPU(), Int(conn.nodes), Float64)
        ptc = PtcState()
        v = apply_ptc!(ptc, red, m, 1, 1)
        vol = conn.area .* (conn.top .- conn.bot)
        act = Array(m.ibound) .> 0
        @test v ≈ maximum(abs.(res[act]) ./ vol[act])
        @test Array(m.sys.diag)[act] ≈ d0[act] .- v
        @test ptc.l2norm0 ≈ sqrt(sum(abs2, res[act]))
        @test apply_ptc!(ptc, red, m, 1, 2) == 0.0  # same norm in the first iteration of step 2
        # CG with NEWTON is MF6's input error (sln_ar).
        ims = joinpath(dir, "nwt.ims")
        write(ims, replace(read(ims, String), "LINEAR_ACCELERATION  BICGSTAB" => "LINEAR_ACCELERATION  CG"))
        e = input_error_of(() -> ModflowGPU.initialize(dir; write_output = false))
        @test occursin("PRODUCES AN ASYMMETRIC COEFFICIENT MATRIX", e.msg) && occursin("**NWT**", e.msg)
        # NEWTON with VARIABLECV is MF6's input error (gwf-npf.f90 check_options).
        npf = joinpath(dir, "nwt.npf")
        write(npf, replace(read(npf, String), "  SAVE_FLOWS\n" => "  SAVE_FLOWS\n  VARIABLECV\n"))
        e = input_error_of(() -> GwfModel(CPU(), Float64, only(load_simulation(dir).models), 2))
        @test occursin("NEWTON OPTION CANNOT BE USED WITH VARIABLECV OPTION", e.msg)
    end
end
