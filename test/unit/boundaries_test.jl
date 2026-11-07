# Boundary framework and CHD, WEL, RCH, GHB (T2.3): period data on the host (reduced
# nodes, MULTIPLIER, head offsets, READASARRAYS, highest active cell), the formulate, add,
# simvals and CHD kernels on TEST_BACKEND in every precision, against host values computed
# in Float64 from the input files.
using ModflowGPU: load_simulation, build_connectivity, BoundaryPackage, apply_period!, formulate!, add_to_system!, calc_simvals!,
    apply_chd_heads!, calc_chd_rate!, boundary_ibound, highest_active, to_offsets,
    reference_heads, MF6UnsupportedError, MF6InputError

# Device array of `x` on TEST_BACKEND.
function _dev(x::AbstractArray)
    d = KernelAbstractions.allocate(TEST_BACKEND, eltype(x), size(x))
    copyto!(d, x)
    return d
end

# A 3-period simulation on the coupling-like grid (IDOMAIN 0 cells and columns with one
# active layer) with CHD, WEL, GHB, RCH (list) and RCHA packages; returns the directory.
function _write_boundary_simulation(dir)
    dis = _synthetic_coupling_dis()
    strt = [dis.top[j, i] - 3.0 - 0.5 * k for j in 1:dis.ncol, i in 1:dis.nrow, k in 1:dis.nlay]
    write_dis_simulation(dir, dis; name = "bnd", strt)
    write_input(
        dir, "sim.tdis", """
        BEGIN dimensions
          NPER  3
        END dimensions
        BEGIN perioddata
          1.0  1  1.0
          1.0  1  1.0
          1.0  1  1.0
        END perioddata
        """,
    )
    write_input(
        dir, "bnd.nam", """
        BEGIN packages
          DIS6  bnd.dis  dis
          NPF6  bnd.npf  npf
          IC6  bnd.ic  ic
          CHD6  bnd.chd  chd
          WEL6  bnd.wel  wel
          GHB6  bnd.ghb  ghb
          RCH6  bnd.rch  rch
          RCH6  bnd.rcha  rcha
        END packages
        """,
    )
    # cellids (layer, row, col): (1,2,2) and (2,2,3) are active; (1,3,4) is a hole.
    write_input(
        dir, "bnd.chd", """
        BEGIN options
          AUXILIARY  mult
          AUXMULTNAME  mult
        END options
        BEGIN dimensions
          MAXBOUND  3
        END dimensions
        BEGIN period  1
          1 2 2  245.5  1.0
          2 5 3  101.25  0.5
        END period
        BEGIN period  3
          1 4 6  240.0  1.0
        END period
        """,
    )
    write_input(
        dir, "bnd.wel", """
        BEGIN dimensions
          MAXBOUND  4
        END dimensions
        BEGIN period  2
          1 3 3  -150.0
          2 2 3  -75.5
          1 3 3  25.25
        END period
        """,
    )
    write_input(
        dir, "bnd.ghb", """
        BEGIN options
          AUXILIARY  f
          AUXMULTNAME  f
        END options
        BEGIN dimensions
          MAXBOUND  2
        END dimensions
        BEGIN period  1
          1 2 5  238.75  12.5  2.0
          2 4 2  100.5  3.0  1.0
        END period
        """,
    )
    write_input(
        dir, "bnd.rch", """
        BEGIN options
          AUXILIARY  m
          AUXMULTNAME  m
        END options
        BEGIN dimensions
          MAXBOUND  2
        END dimensions
        BEGIN period  1
          1 2 2  0.001  0.5
          2 3 4  0.002  1.0
        END period
        """,
    )
    ncpl = dis.nrow * dis.ncol
    irch = ones(Int, ncpl)
    irch[(3 - 1) * dis.ncol + 4] = 2   # the layer-1 hole: layer 2
    irch[(1 - 1) * dis.ncol + 6] = 2   # an excluded column: no cell, node 0
    write_input(
        dir, "bnd.rcha", """
        BEGIN options
          READASARRAYS
        END options
        BEGIN period  1
          irch
            INTERNAL
              $(join(irch, " "))
          recharge
            INTERNAL
              $(join((string(1.0e-3 * (1 + (c % 5))) for c in 1:ncpl), " "))
        END period
        """,
    )
    return dir
end

@testset "Boundary packages (T2.3)" begin
    @testset "to_offsets" begin
        r = [10.0, 5000.0, -3.0]
        @test to_offsets(r, [11.0, 5000.25, -3.0]) == [1.0, 0.25, 0.0]
        @test to_offsets(r, [5001.0, 9.5, 7.0], Int32[2, 1, 0]) == [1.0, -0.5, 7.0]
        @test_throws ArgumentError to_offsets(r, [1.0])
    end

    mktempdir() do dir
        _write_boundary_simulation(dir)
        m = only(load_simulation(dir; strict = true).models)
        conn = build_connectivity(m)
        r = reference_heads(conn, m.ic.strt)
        byname = Dict(b.name => b for b in m.boundaries)
        red(l, i, j) = conn.nodereduced[(l - 1) * 35 + (i - 1) * 7 + j]
        @test red(1, 3, 4) == 0 && red(1, 2, 2) > 0

        @testset "spec checks" begin
            wel = byname["WEL"]
            afr = typeof(wel)(; (f => getfield(wel, f) for f in fieldnames(typeof(wel)))..., auto_flow_reduce = 0.1)
            @test_throws MF6UnsupportedError BoundaryPackage(TEST_BACKEND, Float64, afr, conn, m.dis)
        end

        @testset "highest_active" begin
            ib = ones(Int32, conn.nodes)
            n1 = red(1, 2, 3)
            ib[n1] = 0
            @test highest_active(conn, n1, ib) == red(2, 2, 3)
        end

        for_each_precision() do p
            T = p.T
            pk = Dict(name => BoundaryPackage(TEST_BACKEND, T, b, conn, m.dis) for (name, b) in byname)
            area = _dev(T.(conn.area))
            nodes = conn.nodes
            h = _dev(T.(0.125 .* sin.(1:nodes)))   # offset heads
            hh = Array(h)

            @testset "CHD: period data, IBOUND, heads" begin
                chd = pk["CHD"]
                @test apply_period!(chd, 1, r)
                @test !apply_period!(chd, 2, r)   # no block 2: period 1 data stay
                @test chd.nbound == 2
                n1, n2 = red(1, 2, 2), red(2, 5, 3)
                @test chd.nodelist_host[1:2] == [n1, n2]
                # MULTIPLIER before the offset: 101.25 * 0.5 − r
                @test chd.bound_host[1, 1:2] == [245.5 - r[n1], 101.25 * 0.5 - r[n2]]
                @test Array(chd.bound)[1, 1:2] == T.(chd.bound_host[1, 1:2])
                ib = boundary_ibound(conn, values(pk))
                @test findall(<(0), ib) == sort([n1, n2])
                hc = copy(h)
                apply_chd_heads!(hc, chd)
                KernelAbstractions.synchronize(TEST_BACKEND)
                hca = Array(hc)
                @test hca[n1] == T(245.5 - r[n1]) && hca[n2] == T(50.625 - r[n2])
                @test hca[setdiff(1:nodes, [n1, n2])] == hh[setdiff(1:nodes, [n1, n2])]
                @test apply_period!(chd, 3, r)
                @test chd.nbound == 1 && chd.nodelist_host[1] == red(1, 4, 6)
                # Duplicate constant heads
                @test_throws MF6InputError boundary_ibound(conn, [chd, chd])
            end

            ib_host = boundary_ibound(conn, [pk["CHD"]])   # period 3 CHD
            ib = _dev(ib_host)

            # formulate + add + simvals for one package against Float64 host values;
            # returns (hcof, rhs) on the host.
            function check_package(pkg, hcof_ref, rhs_ref)
                formulate!(pkg, h, ib, area)
                diag = KernelAbstractions.zeros(TEST_BACKEND, T, nodes)
                b = KernelAbstractions.zeros(TEST_BACKEND, T, nodes)
                add_to_system!(diag, b, _dev(Int32.(1:nodes)), pkg)
                calc_simvals!(pkg, h, ib)
                KernelAbstractions.synchronize(TEST_BACKEND)
                n = pkg.nbound
                hcof = Array(pkg.hcof)[1:n]
                rhs = Array(pkg.rhs)[1:n]
                tol = 4 * eps(T)
                @test all(isapprox.(hcof, hcof_ref; rtol = tol, atol = 0))
                @test all(isapprox.(rhs, rhs_ref; rtol = tol, atol = 0))
                # add: per node, records in order (exactly the sequential sum)
                dref = zeros(T, nodes)
                bref = zeros(T, nodes)
                for i in 1:n
                    nd = pkg.nodelist_host[i]
                    nd > 0 || continue
                    dref[nd] += hcof[i]
                    bref[nd] += rhs[i]
                end
                @test Array(diag) == dref
                @test Array(b) == bref
                sv = Array(pkg.simvals)[1:n]
                svref = [pkg.nodelist_host[i] > 0 && ib_host[pkg.nodelist_host[i]] > 0 ? hcof[i] * hh[pkg.nodelist_host[i]] - rhs[i] : zero(T) for i in 1:n]
                @test sv == svref
                return hcof, rhs
            end

            @testset "WEL" begin
                wel = pk["WEL"]
                @test !apply_period!(wel, 1, r) && wel.nbound == 0   # no block yet
                @test apply_period!(wel, 2, r) && wel.nbound == 3
                @test wel.nodelist_host[1] == wel.nodelist_host[3]   # two records in one cell
                hcof, rhs = check_package(wel, zeros(3), [150.0, 75.5, -25.25])
            end

            @testset "GHB" begin
                ghb = pk["GHB"]
                apply_period!(ghb, 1, r)
                n1, n2 = ghb.nodelist_host[1:2]
                @test ghb.bound_host[:, 1] == [238.75 - r[n1], 25.0]   # MULTIPLIER on cond
                @test ghb.bound_host[:, 2] == [100.5 - r[n2], 3.0]
                check_package(ghb, [-25.0, -3.0], [-25.0 * (238.75 - r[n1]), -3.0 * (100.5 - r[n2])])
                # The flow is cond·(bhead − h), independent of the reference.
                sv = Array(ghb.simvals)[1:2]
                habs = r[[n1, n2]] .+ hh[[n1, n2]]
                @test sv ≈ [25.0, 3.0] .* ([238.75, 100.5] .- habs) rtol = 100 * eps(T)
            end

            @testset "RCH list (MULTIPLIER, area)" begin
                rch = pk["RCH"]
                apply_period!(rch, 1, r)
                n1, n2 = rch.nodelist_host[1:2]
                @test n2 == red(2, 3, 4)
                @test rch.bound_host[1, 1:2] == [0.0005, 0.002]
                check_package(rch, zeros(2), [-0.0005 * conn.area[n1], -0.002 * conn.area[n2]])
            end

            @testset "RCHA (READASARRAYS)" begin
                rcha = pk["RCHA"]
                apply_period!(rcha, 1, r)
                @test rcha.nbound == 35
                @test rcha.nodelist_host[(3 - 1) * 7 + 4] == red(2, 3, 4)
                @test rcha.nodelist_host[6] == 0   # excluded column
                @test rcha.nodelist_host[1] == red(1, 1, 1)
                rate = [1.0e-3 * (1 + (c % 5)) for c in 1:35]
                rref = [nd > 0 ? -rate[c] * conn.area[nd] : 0.0 for (c, nd) in enumerate(rcha.nodelist_host[1:35])]
                # The period 3 CHD cell gets no recharge (IBOUND < 0).
                rref[findfirst(==(red(1, 4, 6)), rcha.nodelist_host[1:35])] = 0.0
                check_package(rcha, zeros(35), rref)
            end

            @testset "RCH: highest active cell (no FIXED_CELL)" begin
                rcha = BoundaryPackage(TEST_BACKEND, T, byname["RCHA"], conn, m.dis)
                ibh = ones(Int32, nodes)
                n11 = red(1, 1, 1)
                ibh[n11] = 0
                apply_period!(rcha, 1, r; ibound = ibh)
                @test rcha.nodelist_host[1] == highest_active(conn, n11, ibh) == red(2, 1, 1)
                fixed = typeof(byname["RCHA"])(; (f => getfield(byname["RCHA"], f) for f in fieldnames(typeof(byname["RCHA"])))..., fixed_cell = true)
                rf = BoundaryPackage(TEST_BACKEND, T, fixed, conn, m.dis)
                apply_period!(rf, 1, r; ibound = ibh)
                @test rf.nodelist_host[1] == n11
            end

            @testset "CHD rate (calc_chd_rate)" begin
                chd = pk["CHD"]
                apply_period!(chd, 1, r)
                ib1 = boundary_ibound(conn, [chd])
                # flowja = cond·(h_m − h_n) with unit conductances and the offset heads
                flowja = zeros(T, conn.nja)
                for n in 1:nodes, ii in (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
                    flowja[ii] = hh[conn.ja[ii]] - hh[n]
                end
                fj = _dev(flowja)
                calc_chd_rate!(chd, fj, _dev(conn.ia), _dev(conn.ja), _dev(ib1))
                KernelAbstractions.synchronize(TEST_BACKEND)
                fja = Array(fj)
                for i in 1:2
                    n = chd.nodelist_host[i]
                    rng = (conn.ia[n] + 1):(conn.ia[n + 1] - 1)
                    rate = -sum(flowja[rng])
                    @test Array(chd.simvals)[i] ≈ rate rtol = 10 * eps(T)
                    act = [ii for ii in rng if ib1[conn.ja[ii]] > 0]
                    @test Array(chd.ratein)[i] ≈ sum(-flowja[ii] for ii in act if flowja[ii] < 0; init = zero(T)) rtol = 10 * eps(T)
                    @test Array(chd.rateout)[i] ≈ sum(flowja[ii] for ii in act if flowja[ii] >= 0; init = zero(T)) rtol = 10 * eps(T)
                    @test fja[conn.ia[n]] == Array(chd.simvals)[i]
                end
            end
        end
    end
end
