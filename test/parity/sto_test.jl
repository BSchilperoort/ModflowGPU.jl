# Residual test with STO, CHD and WEL (T2.5) on ex-gwf-fhb, with the CHD and WEL time
# series replaced by constants per stress period (test/models/fhb_const; time series
# arrive with T6.3). The copy is run with mf6 6.8.2 ($REFERENCE_DIR/mf6.8.2_linux/bin/mf6,
# or $MF6) at tight closures into a temporary directory; skipped without it.
#
# For every time step (3 periods, 20 steps, all transient) the flux-form residual
# (`flux_residual`, test/residual.jl) is computed in the engine precision `T` with MF6's
# heads at the end of the step (`h'`) and at the end of the previous step (`h'old`; the IC
# offsets for the first step), all converted to offsets in Float64 before the cast to
# `T`; `diag` and `b` hold the boundary terms and the storage terms of `add_to_system!` on
# TEST_BACKEND with `delt` from the budget file. Criterion as in the boundary test:
# < 1e-8, or in `:single` below the bound from rounding the offset heads to Float32. Then
# the STO-SS rates (`calc_storage_rates!`) and budget entry (`storage_budget`) against the
# CBC STO-SS records, relative to max(max |q|, total inflow).
using ModflowGPU: load_simulation, build_connectivity, npf_setup, BoundaryPackage, apply_period!, formulate!,
    add_to_system!, apply_chd_heads!, boundary_ibound, StoragePackage, set_timestep!, calc_storage_rates!,
    storage_budget, initial_heads, to_offsets, open_head_file, read_head, open_budget_file, read_budget,
    find_budget_records

const FHB_CONST_DIR = joinpath(@__DIR__, "..", "models", "fhb_const")
const MF6_EXE = get(ENV, "MF6", joinpath(REFERENCE_DIR, "mf6.8.2_linux", "bin", "mf6"))

@testset "Storage residual test (STO, CHD, WEL) on fhb with constant stresses" begin
    if !isfile(MF6_EXE)
        @info "Skipping the storage residual test: mf6 not found at $MF6_EXE (set REFERENCE_DIR or MF6)"
    else
        mktempdir() do dir
            cp(FHB_CONST_DIR, joinpath(dir, "fhb"))
            rundir = joinpath(dir, "fhb")
            out = read(Cmd(`$MF6_EXE`; dir = rundir), String)
            @test occursin("Normal termination", out)
            m = only(load_simulation(rundir).models)
            conn = build_connectivity(m)
            s = npf_setup(m, conn)
            hf = open_head_file(joinpath(rundir, "ex-gwf-fhb.hds"))
            bf = open_budget_file(joinpath(rundir, "ex-gwf-fhb.cbc"))
            stoidx = find_budget_records(bf; text = "STO-SS")
            stoq = read_budget(bf; text = "STO-SS")
            # Total inflow per step (the WEL injection); storage rates go to ~1e-17 as each
            # period approaches steady state, so errors are scaled by max(max |q|, total in)
            # (PLAN.md section 4, budgets).
            welq = [sum(max.(x.q, 0.0)) for x in read_budget(bf; text = "WEL")]
            nstep = length(hf)
            @test nstep == 20 == length(stoidx)
            hoff = [to_offsets(s.r, vec(read_head(hf, k))[conn.nodeuser]) for k in 1:nstep]
            h0 = initial_heads(m.ic, conn, s.r)
            for_each_precision() do p
                T = p.T
                pk = [BoundaryPackage(TEST_BACKEND, T, b, conn, m.dis) for b in m.boundaries]
                sto = StoragePackage(TEST_BACKEND, T, m.sto, conn, 3)
                idiag = to_test_device(Int32.(1:conn.nodes))
                area = to_test_device(T.(conn.area))
                worst = 0.0
                worst_bound = 0.0
                worst_q = 0.0
                worst_bud = 0.0
                for k in 1:nstep
                    rec = bf.records[stoidx[k]]
                    @test (hf.steps[k].kstp, hf.steps[k].kper) == (rec.kstp, rec.kper)
                    kper = Int(rec.kper)
                    foreach(x -> apply_period!(x, kper, s.r), pk)
                    apply_period!(sto, kper)
                    @test sto.transient
                    set_timestep!(sto, rec.delt)
                    ibh = boundary_ibound(conn, pk)
                    ib = to_test_device(ibh)
                    h = to_test_device(T.(hoff[k]))
                    hold = to_test_device(T.(k == 1 ? h0 : hoff[k - 1]))
                    diag = KernelAbstractions.zeros(TEST_BACKEND, T, conn.nodes)
                    b = KernelAbstractions.zeros(TEST_BACKEND, T, conn.nodes)
                    for x in pk
                        if ModflowGPU.boundary_kind(x) === :CHD
                            apply_chd_heads!(h, x)
                        else
                            formulate!(x, h, ib, area)
                            add_to_system!(diag, b, idiag, x)
                        end
                    end
                    add_to_system!(diag, b, idiag, sto, hold, ib)
                    calc_storage_rates!(sto, h, hold, ib)
                    KernelAbstractions.synchronize(TEST_BACKEND)
                    rel, roundoff = flux_residual(T, conn, T.(s.condsat), T.(s.Δr), Array(h), Array(diag), Array(b), ibh)
                    @test rel < 1.0e-8 || (T === Float32 && rel <= roundoff)
                    worst = max(worst, rel)
                    worst_bound = max(worst_bound, roundoff)
                    # STO-SS rates vs the CBC (user node = reduced node here: no IDOMAIN).
                    qref = stoq[k][conn.nodeuser]
                    q = Float64.(Array(sto.strgss))
                    qscale = max(maximum(abs, qref), welq[k])
                    err = maximum(abs.(q .- qref)) / qscale
                    @test err < (T === Float64 ? 1.0e-10 : 1.0e-4)
                    worst_q = max(worst_q, err)
                    bud = storage_budget(sto)
                    rin = sum(max.(qref, 0.0))
                    rout = -sum(min.(qref, 0.0))
                    berr = max(abs(bud.rin - rin), abs(bud.rout - rout)) / max(rin, rout, welq[k])
                    @test berr < (T === Float64 ? 1.0e-10 : 1.0e-4)
                    worst_bud = max(worst_bud, berr)
                end
                @info "Storage residual test: fhb (constant stresses), :$(p.preset), $nstep steps: max |res| / max Σ|terms| = $worst (rounding bound $worst_bound); STO-SS max |Δq| / max(max |q|, total in) = $worst_q, budget rel. error $worst_bud"
            end
        end
    end
end
