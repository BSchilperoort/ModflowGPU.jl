# Discretization and connectivity (T2.1) on small hand-built grids. Connectivity is host
# geometry in Float64 with Int32 indices, the same in every precision preset, so these
# tests do not loop over `for_each_precision`. The libmf6 oracle tests are in
# test/parity/connectivity_test.jl.
using ModflowGPU: DisSpec, DisvSpec, Connectivity, build_connectivity, check_dis_grid, reduced_numbering, iajausr,
    getjaindex, connection_lengths, MF6InputError, MF6UnsupportedError

# `_dis` and the generator `_Lcg` / `_next!` are in test/grids.jl.

# Connections of a DIS grid from their definition (MF6 IO guide, DIS): horizontal neighbours
# with IDOMAIN > 0 in the same layer; vertically the first cell with IDOMAIN ≠ < 0 below,
# connected when its IDOMAIN > 0. Returns `(user node low, user node high) => (ihc, cl1,
# cl2, hwva, anglex)` and the cell thicknesses by user node.
function _brute_connections(dis::DisSpec)
    ncol, nrow, nlay = Int(dis.ncol), Int(dis.nrow), Int(dis.nlay)
    id = dis.idomain
    lin(j, i, k) = j + (i - 1) * ncol + (k - 1) * ncol * nrow
    celltop(j, i, k) = k == 1 ? dis.top[j, i] : dis.botm[j, i, k - 1]
    thick(j, i, k) = celltop(j, i, k) - dis.botm[j, i, k]
    conns = Dict{Tuple{Int, Int}, NTuple{5, Float64}}()
    for k in 1:nlay, i in 1:nrow, j in 1:ncol
        id[j, i, k] > 0 || continue
        n = lin(j, i, k)
        if j < ncol && id[j + 1, i, k] > 0
            conns[(n, lin(j + 1, i, k))] = (1, dis.delr[j] / 2, dis.delr[j + 1] / 2, dis.delc[i], 0)
        end
        if i < nrow && id[j, i + 1, k] > 0
            conns[(n, lin(j, i + 1, k))] = (1, dis.delc[i] / 2, dis.delc[i + 1] / 2, dis.delr[j], 1.5 * pi)
        end
        kk = k + 1
        while kk <= nlay && id[j, i, kk] < 0
            kk += 1
        end
        if kk <= nlay && id[j, i, kk] > 0
            conns[(n, lin(j, i, kk))] = (0, thick(j, i, k) / 2, thick(j, i, kk) / 2, dis.delr[j] * dis.delc[i], 0)
        end
    end
    return conns
end

# Check every invariant of `c` against the brute-force connections of `dis`.
function _check_against_brute(c::Connectivity, dis::DisSpec)
    conns = _brute_connections(dis)
    id = vec(dis.idomain)
    active = findall(>(0), id)
    ok = c.nodes == length(active) && c.nodesuser == length(id) && c.nodeuser == active
    ok &= all(c.nodereduced[n] == (id[n] > 0 ? searchsortedfirst(active, n) : id[n] < 0 ? -1 : 0) for n in eachindex(id))
    ok &= c.njas == length(conns) && c.nja == c.nodes + 2 * c.njas
    ok &= length(c.ia) == c.nodes + 1 && c.ia[1] == 1 && c.ia[end] == c.nja + 1
    ipos = 0
    for n in 1:c.nodes
        row = c.ia[n]:(c.ia[n + 1] - 1)
        ok &= c.ja[row[1]] == n && c.jas[row[1]] == 0 && c.isym[row[1]] == row[1]
        offd = c.ja[row[2:end]]
        ok &= issorted(offd) && allunique(offd)
        nu = c.nodeuser[n]
        expected = sort!([c.nodereduced[a == nu ? b : a] for (a, b) in keys(conns) if a == nu || b == nu])
        ok &= offd == expected
        for ii in row[2:end]
            m = c.ja[ii]
            ok &= c.ja[c.isym[ii]] == n && c.isym[c.isym[ii]] == ii && c.jas[c.isym[ii]] == c.jas[ii]
            m > n && (ipos += 1; ok &= c.jas[ii] == ipos)  # upper triangle numbered in CSR order
            g = conns[(min(nu, c.nodeuser[m]), max(nu, c.nodeuser[m]))]
            s = c.jas[ii]
            ok &= (c.ihc[s], c.cl1[s], c.cl2[s], c.hwva[s]) == g[1:4] && c.anglex[s] ≈ g[5]
        end
    end
    return ok
end

@testset "connectivity" begin
    @testset "reduced_numbering" begin
        nr, nu = reduced_numbering(Int32[1, 0, -1, 2, 5, -3, 0])
        @test nr == Int32[1, 0, -1, 2, 3, -1, 0]
        @test nu == Int32[1, 4, 5]
        @test eltype(nr) == Int32 && eltype(nu) == Int32
    end

    @testset "one layer, 2 rows × 3 columns" begin
        dis = _dis(; delr = [10, 20, 30], delc = [5, 7], top = fill(10.0, 3, 2), botm = zeros(3, 2, 1))
        c = build_connectivity(dis)
        @test c isa Connectivity
        @test (c.nodes, c.nodesuser, c.nja, c.njas) == (6, 6, 20, 7)
        @test c.ia == Int32[1, 4, 8, 11, 14, 18, 21]
        @test c.ja == Int32[1, 2, 4, 2, 1, 3, 5, 3, 2, 6, 4, 1, 5, 5, 2, 4, 6, 6, 3, 5]
        @test c.jas == Int32[0, 1, 2, 0, 1, 3, 4, 0, 3, 5, 0, 2, 6, 0, 4, 6, 7, 0, 5, 7]
        # symmetric connections in order: 1-2 right, 1-4 front, 2-3, 2-5, 3-6, 4-5, 5-6
        @test c.ihc == ones(Int32, 7)
        @test c.cl1 == [5, 2.5, 10, 2.5, 2.5, 5, 10]
        @test c.cl2 == [10, 3.5, 15, 3.5, 3.5, 10, 15]
        @test c.hwva == [5, 10, 5, 20, 30, 7, 7]
        @test c.anglex == [0, 1.5pi, 0, 1.5pi, 1.5pi, 0, 0]
        @test c.area == [50, 100, 150, 70, 140, 210]
        @test c.top == fill(10.0, 6) && c.bot == zeros(6)
        @test c.xc == [5, 20, 45, 5, 20, 45]
        @test c.yc == [9.5, 9.5, 9.5, 3.5, 3.5, 3.5]  # row 1 is at the top (largest y)
        @test c.nodeuser == 1:6 && c.nodereduced == 1:6
        @test _check_against_brute(c, dis)
        @test iajausr(c) == (c.ia, c.ja)
        @test getjaindex(c, 2, 5) == 7
        @test getjaindex(c, 5, 2) == 15
        @test getjaindex(c, 3, 3) == 8
        @test getjaindex(c, 1, 6) == 0
        @test getjaindex(c, 0, 1) == -1 && getjaindex(c, 1, 7) == -1
        @test connection_lengths(c, 2, 7) == (2.5, 3.5)   # 2 → 5: cl1 is node 2's
        @test connection_lengths(c, 5, 15) == (3.5, 2.5)  # 5 → 2
        @test sprint(show, c) == "Connectivity(nodes = 6 of 6, nja = 20, njas = 7)"
    end

    @testset "3-layer column: IDOMAIN $(idom)" for (idom, nr, conn) in (
            # (IDOMAIN by layer, nodereduced, vertical connections as (n, m, cl1, cl2))
            ([1, 1, 1], [1, 2, 3], [(1, 2, 1.0, 1.5), (2, 3, 1.5, 2.0)]),
            ([1, -1, 1], [1, -1, 2], [(1, 2, 1.0, 2.0)]),  # pass-through: half thicknesses of 1 and 3
            ([1, 0, 1], [1, 0, 2], Tuple{Int, Int, Float64, Float64}[]),  # an excluded cell blocks it
            ([-1, 1, 1], [-1, 1, 2], [(1, 2, 1.5, 2.0)]),
            ([1, -1, -1], [1, -1, -1], Tuple{Int, Int, Float64, Float64}[]),  # nothing below
            ([1, 1, -1], [1, 2, -1], [(1, 2, 1.0, 1.5)]),
            ([2, -5, 3], [1, -1, 2], [(1, 2, 1.0, 2.0)]),  # only the sign of IDOMAIN matters
        )
        # thicknesses 2, 3, 4; area 6
        dis = _dis(; delr = [2], delc = [3], top = fill(10.0, 1, 1), botm = reshape([8.0, 5.0, 1.0], 1, 1, 3), idomain = reshape(idom, 1, 1, 3))
        c = build_connectivity(dis)
        @test c.nodereduced == nr
        @test c.njas == length(conn)
        for (s, (n, m, l1, l2)) in enumerate(conn)
            ii = getjaindex(c, n, m)
            @test ii > 0 && c.jas[ii] == s
            @test (c.ihc[s], c.cl1[s], c.cl2[s], c.hwva[s], c.anglex[s]) == (0, l1, l2, 6.0, 0.0)
        end
        # top of a cell below an excluded or pass-through cell is that cell's bottom
        @test c.top == [10.0, 8.0, 5.0][idom .> 0]
        @test c.bot == [8.0, 5.0, 1.0][idom .> 0]
        @test _check_against_brute(c, dis)
    end

    @testset "inactive and pass-through cells, 2 layers" begin
        idomain = reshape(Int32[1, 1, 0, 1, -1, 1, 0, 1, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2], 3, 3, 2)
        dis = _dis(; delr = [1, 2, 3], delc = [4, 5, 6], top = fill(5.0, 3, 3), botm = cat(fill(3.0, 3, 3), fill(0.0, 3, 3); dims = 3), idomain)
        c = build_connectivity(dis)
        @test c.nodes == 13 && c.nodesuser == 18
        @test c.nodeuser == Int32[1, 2, 4, 6, 8, 9, 10, 11, 13, 14, 15, 16, 18]
        @test _check_against_brute(c, dis)
        # IAUSR/JAUSR as in the binary grid file: empty rows for cells outside the solution
        iausr, jausr = iajausr(c)
        @test length(iausr) == 19 && iausr[end] == c.nja + 1
        for n in 1:18
            nr = c.nodereduced[n]
            @test iausr[n + 1] - iausr[n] == (nr > 0 ? c.ia[nr + 1] - c.ia[nr] : 0)
        end
        @test jausr == c.nodeuser[c.ja]
    end

    @testset "random grids against the definition" begin
        g = _Lcg(0x1234)
        for trial in 1:200
            ncol, nrow, nlay = 1 + _next!(g, 5), 1 + _next!(g, 4), 1 + _next!(g, 5)
            idomain = [Int32(_next!(g, 4) - 1) for _ in 1:ncol, _ in 1:nrow, _ in 1:nlay]  # -1, 0, 1, 2
            any(>(0), idomain) || (idomain[1] = 1)
            thick = [1.0 + _next!(g, 100) / 7 for _ in 1:ncol, _ in 1:nrow, _ in 1:nlay]
            top = [100.0 + _next!(g, 50) / 3 for _ in 1:ncol, _ in 1:nrow]
            botm = top .- cumsum(thick; dims = 3)
            dis = _dis(; delr = [1.0 + _next!(g, 9) for _ in 1:ncol], delc = [0.5 + _next!(g, 9) for _ in 1:nrow], top, botm, idomain)
            ok = _check_against_brute(build_connectivity(dis), dis)
            @test ok
            ok || (@info "random grid $trial failed" idomain; break)
        end
    end

    @testset "grid checks (MF6 messages)" begin
        top = fill(10.0, 2, 1)
        botm = reshape([5.0, 5.0, 0.0, 0.0], 2, 1, 2)
        err(d) = (
            try
                build_connectivity(d)
            catch e
                e
            end
        )
        e = err(_dis(; delr = [1, 1], delc = [1], top, botm, idomain = zeros(Int32, 2, 1, 2)))
        @test e isa MF6InputError && e.file == "test.dis"
        @test e.msg == "Model does not have any active nodes. Ensure IDOMAIN array has some values greater than zero."
        e = err(_dis(; delr = [1, 1], delc = [1], top, botm, idomain = fill(Int32(-1), 2, 1, 2)))
        @test e isa MF6InputError && occursin("does not have any active nodes", e.msg)
        # zero and negative thicknesses, all reported; IDOMAIN < 1 cells are not checked
        bad = reshape([10.0, 5.0, 0.0, 6.0], 2, 1, 2)
        e = err(_dis(; delr = [1, 1], delc = [1], top, botm = bad))
        @test e isa MF6InputError
        # (number format: gfortran's 1PG24.15)
        @test e.msg == "CELL (1,1,1) THICKNESS <= 0. TOP, BOT:     10.0000000000000        10.0000000000000    \n" *
            "CELL (2,1,2) THICKNESS <= 0. TOP, BOT:     5.00000000000000        6.00000000000000    "
        @test build_connectivity(_dis(; delr = [1, 1], delc = [1], top, botm = bad, idomain = reshape(Int32[0, 1, 1, -1], 2, 1, 2))) isa Connectivity
        @test ModflowGPU._fortran_1pg(-0.25) == "  -0.250000000000000    "
        @test ModflowGPU._fortran_1pg(0.0) == "    0.00000000000000    "
        @test ModflowGPU._fortran_1pg(1.0e20) == "   1.000000000000000E+20"
        @test ModflowGPU._fortran_1pg(-1.5e-3) == "  -1.500000000000000E-03"
        @test ModflowGPU._fortran_1pg(1.0e14) == "    100000000000000.    "
        @test ModflowGPU._fortran_1pg(999999999999999.9) == "   9.999999999999999E+14"
        # dimensions and array shapes
        d0 = DisSpec(; nlay = Int32(0), nrow = Int32(1), ncol = Int32(1), delr = [1.0], delc = [1.0], top = ones(1, 1), botm = zeros(1, 1, 0))
        e = err(d0)
        @test e isa MF6InputError && e.file == "DIS" && e.msg == "NLAY was not specified or was specified incorrectly."
        @test_throws ArgumentError check_dis_grid(_dis(; delr = [1], delc = [1], top, botm))
        # DISV and DISU (T6.4): test/unit/disv_disu_test.jl
    end
end
