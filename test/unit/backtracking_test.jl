# Unit tests of T6.2a: backtracking of the outer iterations (NumericalSolution.f90
# `sln_backtracking`, `get_backtracking_flag`, `apply_backtracking`, `sln_l2norm`): the
# device routine against hand-computed cases, and runs against mf6 at test time (DIS with
# wetting and drying, where MF6 backtracks; DISV and DISU, where the incoming residual
# norms must match MF6's).
using ModflowGPU: Reducer, accumulation_type, backtrack!, save_hprev!, residual_l2norm, BacktrackRecord, HDRY

@testset "Backtracking: sln_backtracking on device vectors" begin
    for_each_precision() do p
        T = p.T
        dev(x) = ModflowGPU._device_copy(TEST_BACKEND, x)
        ib = dev(Int32[1, 1, -1, 0, 1])
        h0 = T[1, 2, 5, 7, -1]
        red = Reducer(TEST_BACKEND, 5, accumulation_type(TEST_BACKEND, T))
        # The residual of the "formulation" is the head itself; inactive and constant-head
        # rows are left out of the norm.
        setup() = (h = dev(copy(h0)); hprev = dev(zeros(T, 5)); res = dev(copy(h0)); (h, hprev, res))
        norm0 = sqrt(6.0)
        ≈ₜ(a, b) = isapprox(a, b; rtol = 1.0e-5)
        @test residual_l2norm(red, dev(copy(h0)), ib) ≈ₜ norm0
        kw = (; numtrack = 20, btol = 1.05, breduc = 0.1, res_lim = 0.0, dvclose = 1.0e-9)
        run!(h, hprev, res, kiter, res_prev; o...) =
            backtrack!(() -> copyto!(res, h), red, h, hprev, res, ib, kiter, res_prev; merge(kw, (; o...))...)
        # First outer iteration: the norm is stored, nothing is reduced.
        h, hprev, res = setup()
        rec, rp = run!(h, hprev, res, 1, 0.0)
        @test (rec.kiter, rec.flag, rec.count) == (1, 0, 0) && rec.resin ≈ₜ norm0 && rp ≈ₜ norm0
        # No increase beyond the tolerance: nothing is reduced, the norm becomes res_prev.
        rec, rp = run!(h, hprev, res, 2, norm0 / 1.04)
        @test (rec.flag, rec.count) == (0, 0) && rp ≈ₜ norm0 && Array(h) == h0
        # Flag 1: one reduction brings the norm below res_prev·tol; only active cells move.
        rec, rp = run!(h, hprev, res, 2, 1.0)
        @test (rec.flag, rec.count) == (1, 1) && rec.resin ≈ₜ norm0 && rp ≈ₜ 0.1 * norm0
        @test Array(h) ≈ T[0.1, 0.2, 5, 7, -0.1]
        # Flag 2: the last allowed reduction.
        h, hprev, res = setup()
        rec, rp = run!(h, hprev, res, 3, 1.0e-6; numtrack = 2)
        @test (rec.kiter, rec.flag, rec.count) == (3, 2, 2) && rp ≈ₜ 0.01 * norm0 && Array(h) ≈ T[0.01, 0.02, 5, 7, -0.01]
        # Flag 4: the reduced change would be below OUTER_DVCLOSE (0.1·2 < 0.5).
        h, hprev, res = setup()
        rec, rp = run!(h, hprev, res, 2, 1.0; dvclose = 0.5)
        @test (rec.flag, rec.count) == (4, 0) && rp ≈ₜ norm0 && Array(h) == h0
        # Flag 4 after a reduction (0.1·0.2 < 0.05 <= 0.1·2), the residual limit.
        h, hprev, res = setup()
        rec, rp = run!(h, hprev, res, 2, 1.0e-6; dvclose = 0.05)
        @test (rec.flag, rec.count) == (4, 1) && rp ≈ₜ 0.1 * norm0
        h, hprev, res = setup()
        rec, rp = run!(h, hprev, res, 2, 1.0e-6; res_lim = 1.0)
        @test (rec.flag, rec.count) == (0, 1) && rp ≈ₜ 0.1 * norm0
        # xtemp: HDRY for dry cells (MF6's x of a dry cell).
        hp = dev(zeros(T, 5))
        save_hprev!(hp, dev(copy(h0)), ib)
        @test Array(hp) == T[1, 2, 5, T(HDRY), -1]
    end
end

# MF6's outer table rows "Backtracking <kiter> <flag> <count> <resin> <resout>", in order.
_mf6_backtrack_rows(lst) = [
    (; kiter = parse(Int, x[1]), flag = parse(Int, x[2]), count = parse(Int, x[3]), resin = parse(Float64, x[4]), resout = parse(Float64, x[5]))
        for x in eachmatch(r"\n\s*Backtracking\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)", lst)
]

# Runs mf6 and ModflowGPU on a copy of `src` with `ims` as its IMS file (after `edit(rundir)`);
# `f(p, r, rows, lst)` per precision with the ModflowGPU run `r`, MF6's backtracking rows and
# its mfsim.lst.
function _backtracking_vs_mf6(f, src, ims; preconditioner = :auto, edit = identity)
    mktempdir() do tmp
        rundir = joinpath(tmp, "model")
        cp(src, rundir)
        imsf = only(filter(endswith(".ims"), readdir(rundir)))
        write(joinpath(rundir, imsf), ims)
        edit(rundir)
        out = read(Cmd(`$RUN_MF6`; dir = rundir), String)
        @test occursin("Normal termination", out)
        lst = read(joinpath(rundir, "mfsim.lst"), String)
        rows = _mf6_backtrack_rows(lst)
        for_each_precision() do p
            mktempdir() do o
                r = ModflowGPU.run(rundir; backend = TEST_BACKEND, precision = p.preset, outdir = o, preconditioner)
                @test r.converged
                f(p, r, rows, lst)
            end
        end
    end
    return nothing
end

_bt_ims(nonlin = "") = """
BEGIN options
  PRINT_OPTION  SUMMARY
  COMPLEXITY  COMPLEX
END options
BEGIN nonlinear
  OUTER_DVCLOSE  1.0e-9
  OUTER_MAXIMUM  500
  $nonlin
END nonlinear
BEGIN linear
  INNER_MAXIMUM  1000
  INNER_DVCLOSE  1.0e-11
  INNER_RCLOSE  1.0e-9
  LINEAR_ACCELERATION  BICGSTAB
END linear
"""

@testset "Backtracking vs MF6: wetting and drying (DIS)" begin
    if !isfile(RUN_MF6)
        @info "Skipping the backtracking tests against MF6: mf6 not found at $RUN_MF6"
    else
        # test/models/wetdry with COMPLEXITY COMPLEX: MF6 backtracks in 18 outer iterations
        # of steps 9 and 10, where cells rewet (MF6 reduces a rewetted cell's change from
        # HDRY, which dries it again; `save_hprev!`).
        _backtracking_vs_mf6(joinpath(@__DIR__, "..", "models", "wetdry"), _bt_ims()) do p, r, rows, lst
            mf6_bt = _backtracked_iterations(lst)
            bt = _backtracked_iterations(r.steps)
            @test !isempty(bt) && bt == mf6_bt
            ours = [b for s in r.steps for b in s.outer.backtracking if b.count > 0]
            mf6 = [x for x in rows if x.count > 0]
            # Residual norms of the backtracking iterations (MF6 prints 8 digits); :double
            # also the outer iterations of every step and the number of reductions.
            tol = p.T === Float64 ? 1.0e-6 : 1.0e-3
            @test length(ours) == length(mf6)
            @test all(isapprox(a.resin, b.resin; rtol = tol) && isapprox(a.resout, b.resout; rtol = tol) for (a, b) in zip(ours, mf6))
            if p.T === Float64
                @test [b.count for b in ours] == [x.count for x in mf6] && [b.flag for b in ours] == [x.flag for x in mf6]
                @test sum(s -> s.outer.iterations, r.steps) == length(rows)
            end
            @info "Backtracking vs MF6 (wetdry, :$(p.preset)): $(length(ours)) backtracking outer iterations as MF6, outer iterations $(sum(s -> s.outer.iterations, r.steps)) (MF6 $(length(rows)))"
        end
    end
end

@testset "Backtracking vs MF6: residual norms on DISV and DISU" begin
    curve = joinpath(REFERENCE_DIR, "runs", "ex-gwf-curve-90", "tight")
    # ex-gwf-curve-90 with convertible cells (ICELLTYPE 1), so that it is nonlinear.
    function convertible!(dir)
        f = joinpath(dir, "ex-gwf-curve-90.npf")
        write(f, replace(read(f, String), r"(icelltype\s+CONSTANT\s+)0" => s"\g<1>1"))
        return dir
    end
    models = Tuple{String, String, Function}[("disu_staggered (DISU)", joinpath(@__DIR__, "..", "models", "disu_staggered"), identity)]
    isdir(curve) ? push!(models, ("ex-gwf-curve-90, convertible (DISV)", curve, convertible!)) :
        @info "Skipping ex-gwf-curve-90 in the backtracking test: $curve not found"
    if !isfile(RUN_MF6)
        @info "Skipping the backtracking tests against MF6: mf6 not found at $RUN_MF6"
    else
        for (name, src, edit) in models
            # The incoming L2 norms (`sln_l2norm`) of the first outer iterations, before the
            # linear solvers (ILUT vs MC-ILU0) set the outer iterations apart.
            _backtracking_vs_mf6(src, _bt_ims(); edit) do p, r, rows, _
                ours = r.steps[1].outer.backtracking
                # :single: down to 1e-3 of the first norm (the Float32 rounding of the flux-form
                # residual, eps32·Σ|terms|, is a larger part of smaller norms).
                tol = p.T === Float64 ? 1.0e-6 : 1.0e-3
                n = min(5, length(ours), count(x -> x.resin > tol * rows[1].resin, rows))
                @test n >= 3
                @test all(isapprox(ours[i].resin, rows[i].resin; rtol = tol) for i in 1:n)
                @test !any(b -> b.count > 0, ours) && !any(x -> x.count > 0, rows)
                @info "Backtracking vs MF6 ($name, :$(p.preset)): incoming residual norms of $n outer iterations within $tol"
            end
        end
    end
end
