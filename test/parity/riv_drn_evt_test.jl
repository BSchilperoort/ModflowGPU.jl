# Residual test with RIV, DRN and EVT (T2.4), PLAN.md section 4:
#
# 1. ex-gwf-fhb (with its time series since T6.3) is covered by the sto group
#    (it has no RIV, DRN or EVT).
# 2. test/models/coupling_rde: a small synthetic coupling-like model (2 layers, 12 x 15,
#    IDOMAIN 0 corners and a layer-1 hole, variable DELR/DELC, confined) with CHD, WEL,
#    RCHA, RIV (MULTIPLIER, BOUNDNAMES), DRN (AUXDEPTHNAME with positive, negative and zero
#    depths), EVT (NSEG 3, SURF_RATE_SPECIFIED) and EVTA, STO with a steady first period
#    and two transient periods (10 time steps). It is run with mf6 6.8.2
#    ($REFERENCE_DIR/mf6.8.2_linux/bin/mf6, or $MF6) at tight closures in a temporary
#    directory; skipped without it.
#
# For every time step the flux-form residual (`flux_residual`, test/residual.jl) is
# computed in the engine precision `T` with MF6's heads (`h'`, and `h'old` for storage),
# converted to offsets in Float64 before the cast to `T`, and the boundary terms from the
# formulate and add kernels on TEST_BACKEND. Criterion: < 1e-8, or in `:single` below the
# bound from rounding the offset heads to Float32. The RIV, DRN, EVT and EVTA flows per
# record (`calc_simvals!`) are compared with the CBC records, relative to the total
# inflow. The branches of `riv_cf`, `drn_cf` and `evt_cf` hit by MF6's heads are counted
# in Float64; each one must occur.
using ModflowGPU: load_simulation, build_connectivity, npf_setup, BoundaryPackage, apply_period!, formulate!,
    add_to_system!, calc_simvals!, apply_chd_heads!, boundary_ibound, StoragePackage, set_timestep!, initial_heads,
    to_offsets, open_head_file, read_head, open_budget_file, read_budget, find_budget_records

const RDE_DIR = joinpath(@__DIR__, "..", "models", "coupling_rde")
const RDE_MF6 = get(ENV, "MF6", joinpath(REFERENCE_DIR, "mf6.8.2_linux", "bin", "mf6"))

# The branch of one record at absolute head `h` (Float64, from the host records with the
# offsets undone).
function _rde_branch(kind, v, h)
    if kind === :RIV
        return h <= v[3] ? :riv_below_rbot : :riv_above_rbot
    elseif kind === :DRN
        v[3] == 0 && return h <= v[1] ? :drn_off : :drn_on
        top, bot = max(v[1] + v[3], v[1]), min(v[1] + v[3], v[1])
        return h <= bot ? :drn_off : (h >= top ? :drn_on : :drn_scaled)
    else
        h >= v[1] && return :evt_above_surface
        d = v[1] - h
        d >= v[3] && return :evt_below_extinction
        length(v) == 4 && return :evt_single_segment
        return d <= v[5] * v[3] ? :evt_segment1 : (d <= v[6] * v[3] ? :evt_segment2 : :evt_segment3)
    end
end

@testset "Residual test (RIV, DRN, EVT) on the synthetic coupling model" begin
    if !isfile(RDE_MF6)
        @info "Skipping the RIV/DRN/EVT residual test: mf6 not found at $RDE_MF6 (set REFERENCE_DIR or MF6)"
    else
        mktempdir() do dir
            rundir = joinpath(dir, "rde")
            cp(RDE_DIR, rundir)
            out = read(Cmd(`$RDE_MF6`; dir = rundir), String)
            @test occursin("Normal termination", out)
            m = only(load_simulation(rundir; strict = true).models)
            conn = build_connectivity(m)
            s = npf_setup(m, conn)
            @test 300 <= conn.nodes <= 400
            hf = open_head_file(joinpath(rundir, "rde.hds"))
            bf = open_budget_file(joinpath(rundir, "rde.cbc"))
            stoidx = find_budget_records(bf; text = "STO-SS")
            nstep = length(hf)
            @test nstep == 10 == length(stoidx)
            habs = [vec(read_head(hf, k))[conn.nodeuser] for k in 1:nstep]
            hoff = [to_offsets(s.r, x) for x in habs]
            h0 = initial_heads(m.ic, conn, s.r)
            # Total inflow per step (budget scale), from the CBC records of every package.
            totin = zeros(nstep)
            for (k, i) in enumerate(stoidx)
                rec = bf.records[i]
                for j in find_budget_records(bf; kstp = rec.kstp, kper = rec.kper)
                    x = read_budget(bf, j)
                    text = strip(bf.records[j].text)
                    text == "FLOW-JA-FACE" && continue
                    totin[k] += sum(max.(x isa AbstractVector ? x : x.q, 0.0))
                end
            end
            branches = Dict{Symbol, Int}()
            for_each_precision() do p
                T = p.T
                pk = [BoundaryPackage(TEST_BACKEND, T, b, conn, m.dis) for b in m.boundaries]
                @test Set(ModflowGPU.boundary_kind.(pk)) == Set([:CHD, :WEL, :RCH, :RIV, :DRN, :EVT])
                sto = StoragePackage(TEST_BACKEND, T, m.sto, conn, 3)
                idiag = to_test_device(Int32.(1:conn.nodes))
                area = to_test_device(T.(conn.area))
                worst = 0.0
                worst_bound = 0.0
                worst_q = Dict{String, Float64}()
                for k in 1:nstep
                    rec = bf.records[stoidx[k]]
                    kper = Int(rec.kper)
                    foreach(x -> apply_period!(x, kper, s.r), pk)
                    apply_period!(sto, kper)
                    @test sto.transient == (kper > 1)
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
                            calc_simvals!(x, h, ib)
                        end
                    end
                    add_to_system!(diag, b, idiag, sto, hold, ib)
                    KernelAbstractions.synchronize(TEST_BACKEND)
                    rel, roundoff = flux_residual(T, conn, T.(s.condsat), T.(s.Δr), Array(h), Array(diag), Array(b), ibh)
                    @test rel < 1.0e-8 || (T === Float32 && rel <= roundoff)
                    worst = max(worst, rel)
                    worst_bound = max(worst_bound, roundoff)
                    # Flows per record vs the CBC (summed per user node).
                    for x in pk
                        ModflowGPU.boundary_kind(x) in (:RIV, :DRN, :EVT) || continue
                        name = x.spec.name
                        cbc = only(read_budget(bf; kstp = rec.kstp, kper = rec.kper, paknam2 = name))
                        ref = Dict{Int, Float64}()
                        for (id, q) in zip(cbc.id1, cbc.q)
                            ref[id] = get(ref, id, 0.0) + q
                        end
                        sv = Array(x.simvals)
                        got = Dict{Int, Float64}()
                        for i in 1:x.nbound
                            n = x.nodelist_host[i]
                            n > 0 || continue
                            got[conn.nodeuser[n]] = get(got, conn.nodeuser[n], 0.0) + Float64(sv[i])
                            if T === Float64
                                v = ModflowGPU._bound_source(x)[:, i]   # absolute, multiplier applied
                                br = _rde_branch(ModflowGPU.boundary_kind(x), v, habs[k][n])
                                ibh[n] > 0 && (branches[br] = get(branches, br, 0) + 1)
                            end
                        end
                        @test keys(got) == keys(ref)
                        err = maximum(abs(got[id] - ref[id]) for id in keys(ref)) / totin[k]
                        @test err < (T === Float64 ? 1.0e-10 : 1.0e-5)
                        worst_q[name] = max(get(worst_q, name, 0.0), err)
                    end
                end
                @info "RIV/DRN/EVT residual test: coupling_rde, :$(p.preset), $nstep steps: max |res| / max Σ|terms| = $worst (rounding bound $worst_bound); flows vs CBC, max |Δq| / total in: $(join(("$k $(round(v; sigdigits = 3))" for (k, v) in sort(collect(worst_q))), ", "))"
            end
            @info "Branches hit by MF6's heads (records × steps): $(join(("$k $v" for (k, v) in sort(collect(branches))), ", "))"
            for br in (
                    :riv_above_rbot, :riv_below_rbot, :drn_off, :drn_on, :drn_scaled, :evt_above_surface,
                    :evt_segment1, :evt_segment2, :evt_segment3, :evt_single_segment, :evt_below_extinction,
                )
                @test get(branches, br, 0) > 0
            end
        end
    end
end
