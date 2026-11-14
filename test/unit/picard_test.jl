# Unit tests of T2.8: time steps (tdis.f90), output-control selection (TimeStepSelect.f90),
# under-relaxation (sln_underrelax) against a host port, the listing budget writer, and
# the flows and budget reductions of a model (gwf_cq, gwf_bd) against host sums.
using ModflowGPU: TdisSpec, time_steps, OcSpec, OcRecord, StressPeriods, oc_selected, TimeStep, ImsSpec,
    UnderRelaxation, underrelax!, write_listing_budget, read_listing_budgets, load_simulation, GwfModel,
    PicardSettings, PicardSolver, solve_timestep!, start_timestep!, calc_flows!, BudgetAccumulator, model_budget!,
    boundary_kind

@testset "Time steps (tdis_set_delt)" begin
    tdis = TdisSpec(; perlen = [1.0, 30.0, 20.0, 10.0], nstp = Int32[1, 5, 4, 3], tsmult = [1.0, 1.2, 1.0, 0.7])
    steps = time_steps(tdis)
    @test length(steps) == 13
    @test [s.kper for s in steps] == [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]
    @test [s.kstp for s in steps] == [1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3]
    @test steps[1].delt == 1.0 && steps[1].endofperiod && steps[1].totim == 1.0
    d1 = 30.0 * (1 - 1.2) / (1 - 1.2^5)
    @test steps[2].delt == d1
    @test steps[3].delt == 1.2 * d1 && steps[4].delt == 1.2 * 1.2 * d1
    # The last step closes the period exactly: topertim + perlen − totim.
    @test steps[6].delt == (1.0 + 30.0) - steps[5].totim
    @test steps[6].totim == 31.0 && steps[6].pertim == 30.0 && steps[6].endofperiod
    @test all(s -> s.delt == 5.0, steps[7:10]) && steps[10].totim == 51.0
    @test sum(s -> s.delt, steps[11:13]) ≈ 10.0 rtol = 1.0e-15
    @test !steps[5].endofperiod && steps[5].pertim ≈ sum(s -> s.delt, steps[2:5])
end

@testset "Output control selection (TimeStepSelect)" begin
    rec(a, t, s, st = Int[]) = OcRecord(a, t, s, st)
    periods = StressPeriods(
        [2, 4, 6],
        [
            [rec(:save, "HEAD", :all), rec(:print, "BUDGET", :last)],
            [rec(:save, "HEAD", :frequency, [2]), rec(:save, "HEAD", :first), rec(:save, "BUDGET", :steps, [1, 3])],
            OcRecord[],
        ],
    )
    oc = OcSpec(; periods)
    st(kper, kstp, eop = false) = TimeStep(kper, kstp, 1.0, 1.0, 1.0, eop)
    @test !oc_selected(oc, :save, "HEAD", st(1, 1))           # before the first block
    @test oc_selected(oc, :save, "HEAD", st(2, 3))
    @test oc_selected(oc, :save, "HEAD", st(3, 7))            # the block stays in force
    @test !oc_selected(oc, :print, "BUDGET", st(3, 2)) && oc_selected(oc, :print, "BUDGET", st(3, 2, true))
    @test !oc_selected(oc, :save, "BUDGET", st(3, 1))
    @test [oc_selected(oc, :save, "HEAD", st(4, k)) for k in 1:5] == [true, true, false, true, false]
    @test [oc_selected(oc, :save, "BUDGET", st(5, k)) for k in 1:4] == [true, false, true, false]
    @test !oc_selected(oc, :save, "HEAD", st(6, 1)) && !oc_selected(oc, :save, "HEAD", st(9, 2, true))
    @test !oc_selected(nothing, :save, "HEAD", st(1, 1))
end

# Host port of sln_underrelax (Float64) for the reference.
function _host_underrelax!(state, method, ims, x, δ, kiter, bigch)
    n = length(x)
    if method === :simple
        x .+= ims.under_relaxation_gamma .* δ
    elseif method === :cooley
        if kiter == 1
            relax = 1.0
            state[:relaxold] = 1.0
            state[:bigchold] = bigch
        else
            es = bigch / (state[:bigchold] * state[:relaxold])
            relax = es < -1 ? 0.5 / abs(es) : (3 + es) / (3 + abs(es))
        end
        state[:relaxold] = relax
        state[:bigchold] = (1 - ims.under_relaxation_gamma) * bigch + ims.under_relaxation_gamma * state[:bigchold]
        x .+= (relax < 1 ? relax : 1.0) .* δ
    else
        if kiter == 1
            state[:wsave] = ones(n)
            state[:hchold] = fill(1.0e-20, n)
            state[:deold] = zeros(n)
        end
        ws, hc, de = state[:wsave], state[:hchold], state[:deold]
        for i in 1:n
            delx = δ[i]
            ww = de[i] * delx < 0 ? ims.under_relaxation_theta * ws[i] : ws[i] + ims.under_relaxation_kappa
            ww = min(ww, 1.0)
            ws[i] = ww
            hc[i] = kiter == 1 ? delx : (1 - ims.under_relaxation_gamma) * delx + ims.under_relaxation_gamma * hc[i]
            de[i] = delx
            amom = kiter > 4 ? ims.under_relaxation_momentum : 0.0
            x[i] += delx * ww + amom * hc[i]
        end
    end
    return x
end

@testset "Under-relaxation (sln_underrelax): $method" for method in (:simple, :cooley, :dbd)
    ims = ImsSpec(;
        under_relaxation = method, under_relaxation_gamma = 0.3, under_relaxation_theta = 0.7,
        under_relaxation_kappa = 0.05, under_relaxation_momentum = 0.2,
    )
    n = 300
    rng(k) = [sin(0.37 * i * k + 0.1 * k) * 10.0^(-k / 3) for i in 1:n]
    for_each_precision() do p
        T = p.T
        ur = UnderRelaxation(TEST_BACKEND, T, ims, n)
        h = to_test_device(zeros(T, n))
        ib = ones(Int32, n)
        ib[7] = -1
        ibd = to_test_device(ib)
        x = zeros(n)
        state = Dict{Symbol, Any}()
        for kiter in 1:8
            δh = T.(rng(kiter))
            δh[7] = 0
            bigch = Float64(δh[argmax(abs.(δh))])
            underrelax!(ur, h, to_test_device(δh), ibd, kiter, bigch)
            _host_underrelax!(state, method, ims, x, Float64.(δh), kiter, bigch)
        end
        @test maximum(abs.(Float64.(Array(h)) .- x)) <= 50 * eps(T) * maximum(abs, x)
        # A converged iteration takes the full correction.
        h0 = Array(h)
        underrelax!(ur, h, to_test_device(ones(T, n)), ibd, 9, 1.0; converged = true)
        @test Array(h) == h0 .+ T.(ib .> 0)
    end
end

@testset "Listing budget table round trip" begin
    terms = [
        ("STO-SS", "STORAGE", 1477.7609, 0.0, 0.3274, 0.0),
        ("CHD", "CHD_0", 0.0, 69267.8609, 1.3606e-9, 2225.8012),
        ("RCHA", "RCHA", 2.5e12, 0.0, 0.0, 0.0),
    ]
    io = IOBuffer()
    write_listing_budget(io, 5, 2, terms)
    s = String(take!(io))
    @test occursin("VOLUME BUDGET FOR ENTIRE MODEL AT END OF TIME STEP    5, STRESS PERIOD   2", s)
    @test occursin("STO-SS =        1477.7609", s)
    @test occursin("CHD =       1.3606E-09", s)
    path = tempname()
    write(path, s)
    b = only(read_listing_budgets(path))
    @test (b.kstp, b.kper) == (5, 2)
    @test [(t.name, t.package) for t in b.terms] == [(t[1], t[2]) for t in terms]
    @test [t.rate_out for t in b.terms] == [0.0, 2225.8012, 0.0]
    @test b.terms[3].cum_in == 2.5e12 && b.terms[2].rate_in == 1.3606e-9
    @test b.total_rate_in ≈ 0.3274 rtol = 1.0e-4
    rm(path)
end

@testset "Flows and budget reductions (gwf_cq, gwf_bd) on coupling_rde" begin
    sim = load_simulation(joinpath(@__DIR__, "..", "models", "coupling_rde"))
    spec = only(sim.models)
    ims = only(only(sim.solution_groups).solutions)
    steps = time_steps(sim.tdis)
    for_each_precision() do p
        T = p.T
        m = GwfModel(TEST_BACKEND, T, spec, length(sim.tdis.perlen))
        ps = PicardSolver(m, PicardSettings(ims, T))
        acc = BudgetAccumulator(m)
        cum = zeros(length(acc.names))
        for st in steps[1:2]
            st.kstp == 1 && ModflowGPU.apply_period!(m, st.kper)
            start_timestep!(m, st.delt)
            r = solve_timestep!(ps, m, st.kstp)
            @test r.converged
            # hncg and its node lrch (sln_get_dxmax) from the last correction.
            δa = Float64.(Array(m.δ)) .* (Array(m.ibound) .> 0)
            @test r.lrch == argmax(abs.(δa)) && r.hncg == δa[r.lrch]
            calc_flows!(m)
            b = model_budget!(acc, m, st)
            # Host sums (Float64) of the device rates.
            host = Tuple{Float64, Float64}[]
            q = Float64.(Array(m.sto.strgss))
            push!(host, (sum(x for x in q if x > 0; init = 0.0), -sum(x for x in q if x < 0; init = 0.0)))
            for pkg in m.packages
                if boundary_kind(pkg) === :CHD
                    push!(host, (sum(Float64, Array(pkg.ratein)[1:pkg.nbound]), sum(Float64, Array(pkg.rateout)[1:pkg.nbound])))
                else
                    q = Float64.(Array(pkg.simvals)[1:pkg.nbound])
                    push!(host, (sum(x for x in q if x > 0; init = 0.0), -sum(x for x in q if x < 0; init = 0.0)))
                end
            end
            @test length(b.terms) == length(host)
            for (t, hq, i) in zip(b.terms, host, eachindex(host))
                @test t.rate_in ≈ hq[1] rtol = 1.0e-12 atol = 1.0e-300
                @test t.rate_out ≈ hq[2] rtol = 1.0e-12 atol = 1.0e-300
                cum[i] += hq[1] * st.delt
                @test t.cum_in ≈ cum[i] rtol = 1.0e-12 atol = 1.0e-300
            end
            # flowja: antisymmetric; the diagonal is each cell's imbalance (≈ 0).
            fj = Float64.(Array(m.flowja))
            c = m.conn
            @test all(fj[ii] == -fj[c.isym[ii]] for n in 1:c.nodes for ii in (c.ia[n] + 1):(c.ia[n + 1] - 1))
            tin = ModflowGPU.total_rate_in(b)
            @test maximum(abs, fj[c.ia[1:c.nodes]]) < (T === Float64 ? 1.0e-8 : 1.0e-4) * tin
            @test abs(ModflowGPU.percent_discrepancy(b)) < (T === Float64 ? 1.0e-8 : 1.0e-3)
        end
    end
end

@testset "Non-convergence (ConvergenceError, CONTINUE)" begin
    sim = load_simulation(joinpath(@__DIR__, "..", "models", "coupling_rde"))
    for_each_precision() do p
        mktempdir() do out
            err = try
                ModflowGPU.run(sim; backend = TEST_BACKEND, precision = p.preset, outdir = out, closures = (; outer_maximum = 1))
                nothing
            catch e
                e
            end
            @test err isa ModflowGPU.ConvergenceError
            @test (err.kper, err.kstp, err.iterations) == (1, 1, 1)
            # The failed step's output is written first, with its budget table.
            lst = read(joinpath(out, "rde.lst"), String)
            @test occursin("FAILED TO MEET SOLVER CONVERGENCE CRITERIA IN TIME STEP 1 OF STRESS PERIOD 1", lst)
            @test length(read_listing_budgets(joinpath(out, "rde.lst"))) == 1
            @test isfile(joinpath(out, "rde.hds"))
        end
        simc = ModflowGPU.SimulationSpec(; (f => getfield(sim, f) for f in fieldnames(typeof(sim)))..., continue_on_nonconvergence = true)
        r = @test_logs (:warn,) match_mode = :any ModflowGPU.run(simc; backend = TEST_BACKEND, precision = p.preset, write_output = false, closures = (; outer_maximum = 1))
        @test !r.converged && length(r.steps) == 10 && all(s -> s.outer.iterations == 1, r.steps)
        @test r.head_file == "" && r.listing_file == ""
    end
end

@testset "Diverged linear solve (non-finite heads) stops the outer iterations" begin
    sim = load_simulation(joinpath(@__DIR__, "..", "models", "coupling_rde"))
    spec = only(sim.models)
    ims = only(only(sim.solution_groups).solutions)
    for_each_precision() do p
        m = GwfModel(TEST_BACKEND, p.T, spec, 3)
        ps = PicardSolver(m, PicardSettings(ims, p.T))
        ModflowGPU.apply_period!(m, 1)
        start_timestep!(m, 1.0)
        fill!(m.h, p.T(NaN))
        r = @test_logs (:warn, r"not finite") match_mode = :any solve_timestep!(ps, m, 1)
        @test !r.converged && r.iterations == 1 && isnan(r.hncg)
    end
end

# T5.3/T6.0: once a correction meets OUTER_DVCLOSE and only the inner test failed, Float32
# raises INNER_DVCLOSE to the head resolution eps32·max|h'|, never to a loose OUTER_DVCLOSE
# (here 1e-3); Float64 is unchanged. With one inner iteration the step does not converge
# within the 70 outer iterations here (heads up to 6.4 m, so eps32·max|h'| < 1e-6 and the
# floor stays 1e-6: 99 outer iterations in Float32, 83 in Float64). T5.3's raise to
# OUTER_DVCLOSE converged in 46 by accepting inner changes up to 1e-3 m; this test fails on it.
@testset "Float32: INNER_DVCLOSE raised when only the inner test blocks convergence" begin
    sim = load_simulation(joinpath(@__DIR__, "..", "models", "coupling_rde"))
    spec = only(sim.models)
    ims = only(only(sim.solution_groups).solutions)
    cl = (; inner_maximum = 1, outer_dvclose = 1.0e-3, inner_dvclose = 1.0e-6, inner_rclose = 1.0e10, outer_maximum = 70)
    for_each_precision() do p
        m = GwfModel(TEST_BACKEND, p.T, spec, 3)
        ps = PicardSolver(m, PicardSettings(ims, p.T; closures = cl))
        ModflowGPU.apply_period!(m, 1)
        start_timestep!(m, 1.0)
        s = ModflowGPU.OuterState(ps)
        for _ in 1:70
            ModflowGPU.outer_iteration!(s, ps, m, 1) && break
        end
        if p.T === Float32
            hmax = maximum(abs, Float64.(Array(m.h)))
            @test s.noise_floor
            @test 1.0e-6 <= s.krylov.settings.dvclose <= max(1.0e-6, 2 * eps(Float32) * hmax) < 1.0e-3
        else
            @test !s.noise_floor
            @test s.krylov.settings.dvclose == 1.0e-6 && s.krylov.settings == ps.krylov.settings
        end
    end
end

# T6.0: the same rule under STRICT (inner convergence only in the first inner iteration) on
# ex-gwf-lgrv-gc, offsets up to 49 m: Float32 converged in no outer iteration before.
@testset "Float32: STRICT converges at the head resolution (ex-gwf-lgrv-gc)" begin
    rundir = joinpath(REFERENCE_DIR, "runs", "ex-gwf-lgrv-gc", "tight")
    if !isdir(rundir)
        @info "Skipping the STRICT INNER_DVCLOSE test: $rundir not found"
    else
        sim = load_simulation(rundir)
        spec = only(sim.models)
        ims = only(only(sim.solution_groups).solutions)
        for_each_precision() do p
            m = GwfModel(TEST_BACKEND, p.T, spec, 1)
            ps = PicardSolver(m, PicardSettings(ims, p.T))
            @test ps.settings.krylov.icnvgopt == 1
            ModflowGPU.apply_period!(m, 1)
            start_timestep!(m, 1.0)
            s = ModflowGPU.OuterState(ps)
            for _ in 1:20
                ModflowGPU.outer_iteration!(s, ps, m, 1) && break
            end
            @test s.converged && s.kiter <= 5
            if p.T === Float32
                hmax = maximum(abs, Float64.(Array(m.h)))
                @test s.noise_floor
                @test 1.0e-6 < s.krylov.settings.dvclose <= 2 * eps(Float32) * hmax
            else
                @test !s.noise_floor && s.krylov.settings.dvclose == ps.krylov.settings.dvclose
            end
        end
    end
end

# T3.1: Adapt rules on the engine structs. A model, its outer-loop solver and budget
# accumulator built on the CPU and moved with `adapt` to the test backend (and back) give
# the same heads and budgets as ones built there.
@testset "Adapt: moving a model and its solvers between backends" begin
    adapt = ModflowGPU.adapt
    sim = load_simulation(joinpath(@__DIR__, "..", "models", "coupling_rde"))
    spec = only(sim.models)
    steps = time_steps(sim.tdis)[1:3]
    run_steps!(m, ps, acc) = map(steps) do st
        st.kstp == 1 && ModflowGPU.apply_period!(m, st.kper)
        start_timestep!(m, st.delt)
        r = solve_timestep!(ps, m, st.kstp)
        calc_flows!(m)
        b = model_budget!(acc, m, st)
        (r.iterations, r.inner, ModflowGPU.absolute_heads(m), ModflowGPU.total_rate_in(b))
    end
    for_each_precision() do p
        T = p.T
        ims = ImsSpec(; linear_acceleration = :bicgstab, under_relaxation = :dbd, outer_dvclose = 1.0e-6, outer_maximum = 100, inner_maximum = 300, inner_dvclose = 1.0e-7, inner_rclose = 1.0e-6, under_relaxation_theta = 0.7, under_relaxation_kappa = 0.1, under_relaxation_gamma = 0.2)
        m = GwfModel(TEST_BACKEND, T, spec, length(sim.tdis.perlen))
        ps = PicardSolver(m, PicardSettings(ims, T))
        native = run_steps!(m, ps, BudgetAccumulator(m))
        # Built on the CPU (where the reductions accumulate in Float64), then moved.
        mc = GwfModel(CPU(), T, spec, length(sim.tdis.perlen))
        psc = PicardSolver(mc, PicardSettings(ims, T); Ts = T)
        md = adapt(TEST_BACKEND, mc)
        @test md.backend isa typeof(TEST_BACKEND)
        @test typeof(md.h) == typeof(m.h) && typeof(md.packages[1].bound) == typeof(m.packages[1].bound)
        @test typeof(adapt(TEST_BACKEND, psc)) == typeof(ps)
        moved = run_steps!(md, adapt(TEST_BACKEND, psc), adapt(TEST_BACKEND, BudgetAccumulator(mc)))
        @test moved == native
        # The CPU original is untouched (its host period state was copied).
        @test mc.kper == 0 && all(pkg -> pkg.iblock == 0, mc.packages)
        # And back to the CPU: same state.
        mb = adapt(CPU(), md)
        @test mb.h isa Array && Array(md.h) == mb.h && Array(md.sys.amat) == mb.sys.amat
    end
end
