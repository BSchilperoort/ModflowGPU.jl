# Residual test with CHD and WEL (T2.3) on ex-gwf-toth (CHD) and ex-gwf-hanic (CHD, WEL),
# using MF6 6.8.2 tight-closure heads (reference/runs/<example>/tight). The assembly of
# T2.6 does not exist yet, so the connection terms are computed here on the host, in flux
# form (PLAN.md section 3.7, rule 2) in the engine precision `T`: per active row `n`,
#
#     res_n = Σ_m condsat_nm·(h'_m − h'_n + Δr_nm) + hcof_n·h'_n − rhs_n
#
# with `hcof_n`, `rhs_n` from the boundary formulate and add kernels on TEST_BACKEND, and
# MF6's heads converted to offsets in Float64 before the cast to `T`. Constant-head cells
# get their heads from `apply_chd_heads!`. Criterion (PLAN.md section 4): max |res_n| /
# max_n Σ|terms_n| < 1e-8; in `:single` it may instead stay below the bound set by rounding
# the offset heads to Float32 (`flux_residual` in test/residual.jl; both are reported). Then the CHD rates (`calc_chd_rate!`, from flowja computed
# with MF6's heads) against MF6's CBC CHD records. Skips without reference/.
using ModflowGPU: load_simulation, build_connectivity, BoundaryPackage, apply_period!, formulate!, add_to_system!, apply_chd_heads!,
    calc_chd_rate!, boundary_ibound, npf_setup, to_offsets, open_head_file, read_head,
    open_budget_file, read_budget

@testset "Boundary residual test (CHD, WEL) vs MF6 tight heads" begin
    for name in ("ex-gwf-toth", "ex-gwf-hanic")
        rundir = joinpath(REFERENCE_DIR, "runs", name, "tight")
        if !isdir(rundir)
            @info "Skipping the boundary residual test on $name: $rundir not found"
            continue
        end
        @testset "$name" begin
            m = only(load_simulation(rundir; strict = false).models)
            conn = build_connectivity(m)
            s = npf_setup(m, conn)
            hds = only(filter(f -> endswith(f, ".hds"), readdir(rundir; join = true)))
            hf = open_head_file(hds)
            habs = vec(read_head(hf, length(hf)))[conn.nodeuser]   # Float64, reduced nodes
            hoff = to_offsets(s.r, habs)
            cbcfile = only(filter(f -> endswith(f, ".cbc") || endswith(f, ".bud"), readdir(rundir; join = true)))
            bf = open_budget_file(cbcfile)
            chdq = only(read_budget(bf; text = "CHD"))
            # flowja from MF6's heads (Float64), for the CHD rates.
            flowja = zeros(conn.nja)
            for n in 1:conn.nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
                flowja[ii] = s.condsat[conn.jas[ii]] * (habs[conn.ja[ii]] - habs[n])
            end
            kinds = Set(ModflowGPU.boundary_kind(b) for b in m.boundaries)
            @test :CHD in kinds && kinds ⊆ Set([:CHD, :WEL])
            for_each_precision() do p
                T = p.T
                pk = [BoundaryPackage(TEST_BACKEND, T, b, conn, m.dis) for b in m.boundaries]
                foreach(x -> apply_period!(x, 1, s.r), pk)
                ibh = boundary_ibound(conn, pk)
                ib = to_test_device(ibh)
                h = to_test_device(T.(hoff))
                area = to_test_device(T.(conn.area))
                diag = KernelAbstractions.zeros(TEST_BACKEND, T, conn.nodes)
                b = KernelAbstractions.zeros(TEST_BACKEND, T, conn.nodes)
                idiag = to_test_device(Int32.(1:conn.nodes))
                for x in pk
                    if ModflowGPU.boundary_kind(x) === :CHD
                        apply_chd_heads!(h, x)
                    else
                        formulate!(x, h, ib, area)
                        add_to_system!(diag, b, idiag, x)
                    end
                end
                KernelAbstractions.synchronize(TEST_BACKEND)
                hh = Array(h)
                # MF6's constant heads are the CHD heads.
                chdnodes = findall(<(0), ibh)
                @test maximum(abs.(Float64.(hh[chdnodes]) .- hoff[chdnodes])) <= 2 * eps(T) * maximum(abs.(hoff[chdnodes]); init = 1.0)
                rel, roundoff = flux_residual(T, conn, T.(s.condsat), T.(s.Δr), hh, Array(diag), Array(b), ibh)
                @info "Boundary residual test: $name, :$(p.preset), $(count(>(0), ibh)) active rows, $(length(chdnodes)) CHD cells: max |res| / max Σ|terms| = $rel (rounding bound $roundoff)"
                @test rel < 1.0e-8 || (T === Float32 && rel <= roundoff)
                # CHD rates vs the CBC CHD records (q per cell).
                ia, ja = to_test_device(conn.ia), to_test_device(conn.ja)
                fj = to_test_device(T.(flowja))
                q = Float64[]
                nodes = Int32[]
                for x in pk
                    ModflowGPU.boundary_kind(x) === :CHD || continue
                    calc_chd_rate!(x, fj, ia, ja, ib)
                    KernelAbstractions.synchronize(TEST_BACKEND)
                    append!(q, Array(x.simvals)[1:x.nbound])
                    append!(nodes, conn.nodeuser[x.nodelist_host[1:x.nbound]])
                end
                order = sortperm(nodes)
                @test nodes[order] == chdq.id1[sortperm(chdq.id1)]
                qref = chdq.q[sortperm(chdq.id1)]
                qscale = maximum(abs, qref)
                err = maximum(abs.(q[order] .- qref)) / qscale
                @info "CHD rates vs MF6 CBC: $name, :$(p.preset), max |Δq| / max |q| = $err"
                @test err < (T === Float64 ? 1.0e-8 : 1.0e-5)
            end
        end
    end
end
