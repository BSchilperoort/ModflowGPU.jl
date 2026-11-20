# Vertex and unstructured grids (T6.4): DISV and DISU connectivity on hand-built grids, the
# DISV and DISU descriptions of DIS grids (test/grids.jl `_dis_as_disv`, `_dis_as_disu`)
# against DIS (the same connectivity, conductances and, run end to end in every precision
# preset, the same heads and budgets), and the grid checks with MF6's messages. Connectivity
# is host Float64 geometry (no precision loop); the runs use `for_each_precision`. The
# libmf6 oracle on DISV/DISU grids is in test/parity/connectivity_test.jl.
using ModflowGPU: DisSpec, DisvSpec, DisuSpec, Connectivity, build_connectivity, npf_setup, build_npf, build_ic,
    build_sto, build_boundary, build_array_boundary, build_model, build_tdis, build_ims, build_simulation, build_oc, oc_record,
    check_disv_grid, check_disu_grid, disv_anglex, MF6InputError, grid_colors, is_valid_coloring, load_simulation

_disv_err(d) = (
    try
        build_connectivity(d)
    catch e
        e
    end
)

# Fields of two connectivities that must be equal (`exact`) or equal within `rtol`.
function _same_connectivity(a::Connectivity, b::Connectivity; rtol = 0.0)
    ok = all(getfield(a, f) == getfield(b, f) for f in (:nodes, :nodesuser, :nja, :njas, :nodeuser, :nodereduced, :ia, :ja, :isym, :jas, :ihc))
    for f in (:cl1, :cl2, :hwva, :top, :bot, :area)
        ok &= rtol == 0 ? getfield(a, f) == getfield(b, f) : _rel_close(getfield(a, f), getfield(b, f); rtol)
    end
    return ok
end

@testset "DISV and DISU" begin
    @testset "DISV: hand-built grid (triangle, square, pentagon)" begin
        # Vertices (clockwise cells): square 1 = (0,0)-(0,2)-(2,2)-(2,0); triangle 2 to the
        # right of it; pentagon 3 below both.
        vertices = [0.0 0.0 2.0 2.0 4.0 0.0 4.0 2.0; 0.0 2.0 2.0 0.0 1.0 -1.0 -1.0 -2.0]
        cells = [Int32[1, 2, 3, 4], Int32[4, 3, 5], Int32[1, 4, 5, 7, 8, 6]]
        centers = [1.0 2.8 2.0; 1.0 1.0 -0.5]
        disv = DisvSpec(;
            nlay = Int32(2), ncpl = Int32(3), nvert = Int32(8), top = [10.0, 10.0, 10.0], botm = [6.0 1.0; 5.0 0.5; 4.0 0.0],
            vertices, cell_center = centers, cell_vertices = cells, file = "t.disv",
        )
        c = build_connectivity(disv)
        @test c.grid === :disv && c.nodes == 6 && c.nodesuser == 6
        # rows: diagonal, then sorted neighbours (cell 1 and 2 share edge 3-4; 1 and 3 share
        # 4-1; 2 and 3 share 4-5... cell 3 lists 1,4,5 → edges 1-4, 4-5)
        @test c.ia == Int32[1, 5, 9, 13, 17, 21, 25]
        @test c.ja == Int32[1, 2, 3, 4, 2, 1, 3, 5, 3, 1, 2, 6, 4, 1, 5, 6, 5, 2, 4, 6, 6, 3, 4, 5]
        @test c.area[1:3] == [4.0, 2.0, 7.0]  # shoelace: square, triangle, pentagon
        s12 = c.jas[2]
        @test c.ihc[s12] == 1 && c.hwva[s12] == 2.0 && c.cl1[s12] == 1.0 && c.cl2[s12] ≈ 0.8
        @test c.anglex[s12] == 0.0  # edge 3 → 4 points down; outward normal +x
        s13 = c.jas[3]
        @test c.hwva[s13] == 2.0 && c.cl1[s13] == 1.0 && c.cl2[s13] == 0.5 && c.anglex[s13] ≈ 1.5 * pi  # edge 4 → 1
        s14 = c.jas[4]  # vertical, cell 1 to cell 4 (layer 2)
        @test (c.ihc[s14], c.hwva[s14], c.cl1[s14], c.cl2[s14], c.anglex[s14]) == (0, 4.0, 2.0, 2.5, 0.0)
        # outward normals of the edges of a clockwise cell: top edge (+x) up, bottom edge down
        @test disv_anglex(0.0, 0.0, 1.0, 0.0) == 0.5 * pi && disv_anglex(1.0, 0.0, 0.0, 0.0) ≈ 1.5 * pi
        @test disv_anglex(0.0, 1.0, 0.0, 0.0) == 0.0 && disv_anglex(0.0, 0.0, 0.0, 1.0) == Float64(pi)
        # an open vertex list is closed (the triangle above) and a closed one is the same
        cells2 = [Int32[1, 2, 3, 4, 1], Int32[4, 3, 5, 4], Int32[1, 4, 5, 7, 8, 6]]
        c2 = build_connectivity(DisvSpec(; disv.nlay, disv.ncpl, disv.nvert, disv.top, disv.botm, vertices, cell_center = centers, cell_vertices = cells2))
        @test _same_connectivity(c, c2) && c.anglex == c2.anglex
        # excluded and pass-through cells
        d3 = DisvSpec(; disv.nlay, disv.ncpl, disv.nvert, disv.top, disv.botm, vertices, cell_center = centers, cell_vertices = cells, idomain = Int32[1 1; 0 1; -1 1])
        c3 = build_connectivity(d3)
        @test c3.nodes == 4 && c3.nodeuser == Int32[1, 4, 5, 6] && c3.nodereduced == Int32[1, 0, -1, 2, 3, 4]
        @test c3.ja == Int32[1, 2, 2, 1, 3, 4, 3, 2, 4, 4, 2, 3]
        @test c3.top == [10.0, 6.0, 5.0, 4.0]  # top of a layer-2 cell is the layer-1 bottom
    end

    @testset "DIS described as DISV and DISU" begin
        grids = Any[("coupling-like", _synthetic_coupling_dis()), ("pass-through", _synthetic_passthrough_dis())]
        g = _Lcg(0x5678)
        for trial in 1:30
            ncol, nrow, nlay = 1 + _next!(g, 5), 1 + _next!(g, 4), 1 + _next!(g, 4)
            idomain = [Int32(_next!(g, 4) - 1) for _ in 1:ncol, _ in 1:nrow, _ in 1:nlay]
            any(>(0), idomain) || (idomain[1] = 1)
            thick = [1.0 + _next!(g, 100) / 8 for _ in 1:ncol, _ in 1:nrow, _ in 1:nlay]
            top = [100.0 + _next!(g, 50) / 4 for _ in 1:ncol, _ in 1:nrow]
            push!(grids, ("random $trial", _dis(; delr = [1.0 + _next!(g, 9) / 2 for _ in 1:ncol], delc = [0.5 + _next!(g, 9) for _ in 1:nrow], top, botm = top .- cumsum(thick; dims = 3), idomain)))
        end
        for (label, dis) in grids
            c = build_connectivity(dis)
            # dyadic coordinates: the DISV geometry (normal distances, shoelace areas) is exact
            cv = build_connectivity(_dis_as_disv(dis))
            @test cv.grid === :disv
            okv = _same_connectivity(cv, c) && cv.xc == c.xc && cv.yc == c.yc && _rel_close(cv.anglex, c.anglex)
            @test okv
            okv || (@info "DISV of $label differs from DIS"; break)
            any(<(0), dis.idomain) && continue
            for vertices in (false, true)
                cu = build_connectivity(_dis_as_disu(dis; vertices))
                oku = cu.grid === :disu && _same_connectivity(cu, c) && _rel_close(cu.anglex, c.anglex) &&
                    (vertices ? (cu.xc == c.xc && cu.yc == c.yc) : isempty(cu.xc))
                @test oku
                oku || (@info "DISU of $label differs from DIS"; break)
            end
        end
        # DISU without ANGLDEGX: anglex is DNODATA
        cu = build_connectivity(_dis_as_disu(_synthetic_coupling_dis(); angldegx = false))
        @test all(==(3.0e30), cu.anglex)
    end

    @testset "same conductances as DIS" begin
        dis = _synthetic_coupling_dis()
        sz = size(dis.botm)
        k = [10.0^(((7 * j + 3 * i + 5 * kk) % 13) / 3 - 2) for j in 1:sz[1], i in 1:sz[2], kk in 1:sz[3]]
        k33 = k .* 0.1
        strt = fill(200.0, sz)
        for (label, arrays, opts, rtol) in (
                ("isotropic", (;), (;), 0.0), ("logarithmic", (;), (; cell_averaging = :logarithmic), 0.0),
                ("K22 and ANGLE1", (; k22 = 0.3 .* k, angle1 = fill(30.0, sz)), (;), 1.0e-13), ("K22", (; k22 = 0.3 .* k), (;), 1.0e-13),
            )
            ref = nothing
            for d in (dis, _dis_as_disv(dis), _dis_as_disu(dis))
                v(a) = d isa DisSpec ? a : vec(a)  # by user node: the same numbering
                npf = build_npf(d; k = v(k), k33 = v(k33), map(v, arrays)..., opts...)
                ic = build_ic(d; strt = v(strt))
                s = npf_setup(npf, ic, d, build_connectivity(d))
                if isnothing(ref)
                    ref = s.condsat
                else
                    ok = rtol == 0 ? s.condsat == ref : _rel_close(s.condsat, ref; rtol)
                    @test ok
                    ok || @info "condsat of $(nameof(typeof(d))) differs from DIS ($label)"
                end
            end
        end
        # DISU without ANGLDEGX: K22 needs it (gwf-npf.f90 prepcheck)
        du = _dis_as_disu(dis; angldegx = false)
        npf = build_npf(du; k = vec(k), k22 = vec(k))
        e = try
            npf_setup(npf, build_ic(du; strt = vec(strt)), du, build_connectivity(du))
        catch err
            err
        end
        @test e isa MF6InputError && occursin("ANGLDEGX not provided in discretization file, but K22 was specified", e.msg)
    end

    @testset "time series and TAS6 on DISV (test/models/ts_tas as DISV)" begin
        # ts_tas (T6.3: TS6 on CHD/WEL/GHB, TAS6 on RCHA/EVTA) with its 1×5×10 grid written
        # as DISV and the cellids as (layer, cell2d): bitwise the same run as DIS (Jacobi),
        # and parity against mf6 when it is available.
        src = joinpath(@__DIR__, "..", "models", "ts_tas")
        dis = only(load_simulation(src).models).dis
        mktempdir() do tmp
            dir = joinpath(tmp, "ts_tas_disv")
            cp(src, dir)
            open(io -> _write_grid(io, _dis_as_disv(dis)), joinpath(dir, "tsm.dis"), "w")
            nam = joinpath(dir, "tsm.nam")
            write(nam, replace(read(nam, String), "DIS6  tsm.dis" => "DISV6  tsm.dis"))
            ncol = Int(dis.ncol)
            for f in ("tsm.chd", "tsm.wel", "tsm.ghb")
                p = joinpath(dir, f)
                txt = replace(
                    read(p, String), r"(?m)^(\s+)(\d+) (\d+) (\d+) " => s -> begin
                        m = match(r"^(\s+)(\d+) (\d+) (\d+) ", s)
                        "$(m[1])$(m[2]) $((parse(Int, m[3]) - 1) * ncol + parse(Int, m[4])) "
                    end
                )
                write(p, txt)
            end
            @test only(load_simulation(dir).models).dis isa DisvSpec
            for_each_precision() do p
                r0 = ModflowGPU.run(src; backend = TEST_BACKEND, precision = p.preset, write_output = false, preconditioner = :jacobi)
                r = ModflowGPU.run(dir; backend = TEST_BACKEND, precision = p.preset, write_output = false, preconditioner = :jacobi)
                @test r.converged && r0.converged && r.heads == r0.heads
                @test [ModflowGPU.total_rate_in(b) for b in r.budgets] == [ModflowGPU.total_rate_in(b) for b in r0.budgets]
            end
            if isfile(RUN_MF6)
                out = read(Cmd(`$RUN_MF6`; dir), String)
                @test occursin("Normal termination", out)
                _run_parity("ts_tas as DISV", dir)
            else
                @info "Skipping ts_tas as DISV vs mf6: mf6 not found at $RUN_MF6"
            end
        end
    end

    @testset "colouring of a DISV grid" begin
        c = build_connectivity(_dis_as_disv(_synthetic_coupling_dis()))
        colors = grid_colors(c, nothing)
        @test is_valid_coloring(c.ia, c.ja, colors)
    end

    @testset "DIS, DISV and DISU models give the same run" begin
        dis = _synthetic_coupling_dis()
        sz = size(dis.botm)
        nu = prod(sz)
        act = findall(>(0), vec(dis.idomain))
        k = [5.0 + ((3 * j + 5 * i + 7 * kk) % 11) for j in 1:sz[1], i in 1:sz[2], kk in 1:sz[3]]
        chd = act[[1, 2, 3]]
        wel = act[[10, 20]]
        ghb = act[[15, 25]]
        rch = filter(n -> n <= sz[1] * sz[2], act)
        tdis = build_tdis(; perlen = [1.0, 10.0], nstp = [1, 3], tsmult = [1.0, 1.2])
        riv = act[[30, 31, 32]]
        drn = act[[40, 41]]
        ncpl = sz[1] * sz[2]
        topv = vec(dis.top)
        # `arrays`: also RCHA and EVTA (READASARRAYS: DIS and DISV only, as in MF6); their
        # cells in excluded layer-1 cells move to the highest active cell below.
        function model(d; arrays = false, spdis = false)
            v(a) = d isa DisSpec ? a : vec(a)
            bnd = [
                build_boundary(:CHD, d; periods = [1 => (; cells = chd, values = [200.0, 201.5, 199.25])], save_flows = true),
                build_boundary(:WEL, d; periods = [1 => (; cells = wel, values = [-500.0, -250.0]), 2 => (; cells = wel[1:1], values = [-800.0])], save_flows = true),
                build_boundary(:GHB, d; periods = [1 => (; cells = ghb, values = [198.0 50.0; 202.0 75.0])], save_flows = true),
                build_boundary(:RCH, d; periods = [1 => (; cells = rch, values = fill(1.0e-4, length(rch)))], save_flows = true),
                build_boundary(:RIV, d; periods = [1 => (; cells = riv, values = [199.0 30.0 190.0; 199.5 30.0 190.0; 201.0 40.0 195.0])], save_flows = true),
                build_boundary(:DRN, d; periods = [1 => (; cells = drn, values = [199.0 20.0; 200.5 25.0])], save_flows = true),
                build_boundary(:EVT, d; periods = [1 => (; cells = rch[1:5], values = [topv[rch[1:5]] fill(2.0e-4, 5) fill(60.0, 5)])], save_flows = true),
            ]
            if arrays
                push!(bnd, build_array_boundary(:RCH, d; name = "rcha", periods = [1 => (; recharge = [5.0e-5 + 1.0e-6 * (c % 7) for c in 1:ncpl])], save_flows = true))
                push!(bnd, build_array_boundary(:EVT, d; name = "evta", periods = [1 => (; surface = topv, rate = fill(1.0e-4, ncpl), depth = 80.0)], save_flows = true))
            end
            return build_model(
                "grid"; dis = d, npf = build_npf(d; k = v(k), k33 = v(0.1 .* k), save_flows = true, save_specific_discharge = spdis),
                ic = build_ic(d; strt = v(fill(200.0, sz))), sto = build_sto(d; ss = v(fill(1.0e-5, sz)), periods = [1 => :steady_state, 2 => :transient]),
                oc = build_oc(; head_file = "grid.hds", budget_file = "grid.cbc", periods = [1 => [oc_record(:save, "HEAD"), oc_record(:save, "BUDGET")]]),
                boundaries = bnd,
            )
        end
        ims = build_ims(; outer_dvclose = 1.0e-9, inner_dvclose = 1.0e-10, inner_rclose = 1.0e-9, linear_acceleration = :bicgstab, inner_maximum = 500)
        jrun(d, p; arrays = false) = mktempdir() do out
            ModflowGPU.run(build_simulation([model(d; arrays)]; tdis, ims); backend = TEST_BACKEND, precision = p.preset, outdir = out, preconditioner = :jacobi)
        end
        for_each_precision() do p
            # Jacobi: the same arithmetic as DIS, so bitwise the same heads and budgets (the
            # matrices are equal; only the multicolour orderings would differ).
            for (g, r0, r) in (
                    ("DISV", jrun(dis, p; arrays = true), jrun(_dis_as_disv(dis), p; arrays = true)),
                    ("DISU", jrun(dis, p), jrun(_dis_as_disu(dis), p)),
                )
                @test r0.converged && r.converged && r.heads == r0.heads
                @test [ModflowGPU.total_rate_in(b) for b in r.budgets] == [ModflowGPU.total_rate_in(b) for b in r0.budgets]
                @test [[(t.name, t.rate_in, t.rate_out) for t in b.terms] for b in r.budgets] ==
                    [[(t.name, t.rate_in, t.rate_out) for t in b.terms] for b in r0.budgets]
                @test [s.outer.iterations for s in r.steps] == [s.outer.iterations for s in r0.steps]
            end
            r0 = jrun(dis, p)
            # The default preconditioner (greedy colouring on DISV/DISU, red–black on DIS).
            rv = mktempdir() do out
                ModflowGPU.run(build_simulation([model(_dis_as_disv(dis))]; tdis, ims); backend = TEST_BACKEND, precision = p.preset, outdir = out)
            end
            d = maximum(abs.(rv.heads .- r0.heads))
            @test d < (p.T === Float64 ? 1.0e-7 : 1.0e-3)
            @info "DISV/DISU vs DIS run, :$(p.preset): bitwise equal with Jacobi; default preconditioner max |Δh| = $(round(d; sigdigits = 3)) m"
        end

        # T6.9: DATA-SPDIS and the binary grid file on the three grids (`calc_spdis`,
        # `write_grb`): the same specific discharge (DISV/DISU normals from cos/sin of the
        # angles, DIS exact), the same user-node IA/JA.
        function srun(d, p)
            return mktempdir() do out
                r = ModflowGPU.run(build_simulation([model(d; spdis = true)]; tdis, ims); backend = TEST_BACKEND, precision = p.preset, outdir = out, preconditioner = :jacobi)
                bf = open_budget_file(r.budget_file)
                sp = [read_budget(bf, i).aux for i in find_budget_records(bf; text = "DATA-SPDIS")]
                g = read_grb(only(r.models).grb_file)
                (; sp, g, recs = [strip(x.text) for x in bf.records if x.kstp == 1 && x.kper == 1])
            end
        end
        for_each_precision() do p
            a = srun(dis, p)
            @test a.recs[1:3] == ["STO-SS", "FLOW-JA-FACE", "DATA-SPDIS"] || a.recs[1:2] == ["FLOW-JA-FACE", "DATA-SPDIS"]
            @test length(a.sp) == 4 && a.g.grid == "DIS"
            sc = maximum(maximum(abs, x) for x in a.sp)
            @test sc > 0
            for (grid, d) in (("DISV", _dis_as_disv(dis)), ("DISU", _dis_as_disu(dis; vertices = true)))
                b = srun(d, p)
                @test b.g.grid == grid && b.g["IA"] == a.g["IA"] && b.g["JA"] == a.g["JA"] && b.g["ICELLTYPE"] == a.g["ICELLTYPE"]
                dsp = maximum(maximum(abs.(x .- y)) for (x, y) in zip(a.sp, b.sp)) / sc
                @test dsp < (p.T === Float64 ? 1.0e-12 : 1.0e-5)
                @info "DATA-SPDIS $grid vs DIS, :$(p.preset): max |Δ| / max |q| = $(round(dsp; sigdigits = 3))"
            end
        end
        # MF6's errors: DISU without ANGLDEGX or without vertices.
        for d in (_dis_as_disu(dis; angldegx = false, vertices = true), _dis_as_disu(dis))
            @test_throws ModflowGPU.MF6InputError mktempdir() do out
                ModflowGPU.run(build_simulation([model(d; spdis = true)]; tdis, ims); backend = TEST_BACKEND, outdir = out)
            end
        end
    end

    @testset "grid checks (MF6 messages)" begin
        vertices = [0.0 0.0 1.0 1.0; 0.0 1.0 1.0 0.0]
        mk(; cells = [Int32[1, 2, 3, 4]], top = [1.0], botm = zeros(1, 1), idomain = ones(Int32, 1, 1), nlay = 1) =
            DisvSpec(; nlay = Int32(nlay), ncpl = Int32(1), nvert = Int32(4), top, botm, idomain, vertices, cell_center = fill(0.5, 2, 1), cell_vertices = cells, file = "t.disv")
        @test build_connectivity(mk()).area == [1.0]
        e = _disv_err(mk(; cells = [Int32[1, 4, 3, 2]]))  # counter-clockwise
        @test e isa MF6InputError && e.file == "t.disv" && e.msg == "Calculated CELL2D area less than zero for cell 1.\n" *
            "1 cell(s) have an area less than zero. Calculated cell areas must be greater than zero. Negative areas often mean vertices are not listed in clockwise order."
        e = _disv_err(mk(; cells = [Int32[1, 2, 1]]))
        @test e isa MF6InputError && startswith(e.msg, "Calculated CELL2D area is zero for cell 1.")
        e = _disv_err(mk(; botm = fill(1.0, 1, 1)))
        @test e isa MF6InputError && e.msg == "CELL (1,1) THICKNESS <= 0. TOP, BOT:     1.00000000000000        1.00000000000000    "
        e = _disv_err(mk(; idomain = zeros(Int32, 1, 1)))
        @test e isa MF6InputError && occursin("does not have any active nodes", e.msg)
        @test_throws ArgumentError check_disv_grid(mk(; botm = zeros(1, 2)))

        dis = _dis(; delr = [1, 2], delc = [3], top = fill(10.0, 2, 1), botm = cat(fill(5.0, 2, 1), fill(0.0, 2, 1); dims = 3))
        du = _dis_as_disu(dis)
        with(; kw...) = DisuSpec(; (f => get(kw, f, getfield(du, f)) for f in fieldnames(DisuSpec))...)
        @test check_disu_grid(du) == 0
        ja = copy(du.ja)
        ja[1] = 2
        e = _disv_err(with(; ja))
        @test e isa MF6InputError && e.file == "test.disu" && startswith(e.msg, "Cell 1 must have its first connection be itself.  Found: 2")
        e = _disv_err(with(; idomain = Int32[1, -1, 1, 1]))
        @test e isa MF6InputError && e.msg == "Invalid idomain value -1 specified for node 2"
        hw = copy(du.hwva)
        hw[2] *= 2
        e = _disv_err(with(; hwva = hw))
        @test e isa MF6InputError && startswith(e.msg, "Error in array: HWVA. Array is not symmetric in positions: 2 and ")
        ih = copy(du.ihc)
        ih[5] = 0  # entry 2 → 1 (an entry n → m > n would fail the vertical position check first)
        e = _disv_err(with(; ihc = ih))
        @test e isa MF6InputError && startswith(e.msg, "Error in array: IHC. Array is not symmetric")
        # a cell below another must not reach above its bottom
        e = _disv_err(with(; top = [10.0, 10.0, 6.0, 5.0]))
        @test e isa MF6InputError && occursin("for cell 3 is above bottom elevation", e.msg)
        @test build_connectivity(with(; top = [10.0, 10.0, 6.0, 5.0], bot = [5.0, 5.0, 0.0, 0.0], vertical_offset_tolerance = 1.5)) isa Connectivity
        # inconsistent ANGLDEGX: a warning count, an error only when NPF needs ANGLDEGX
        ang = copy(du.angldegx)
        ang[2] = 10.0
        @test (@test_logs (:warn, r"inconsistent with the values for the reverse connections") check_disu_grid(with(; angldegx = ang))) == 1
        # excluded cells: rows sorted, arrays compressed
        cu = build_connectivity(with(; idomain = Int32[1, 0, 1, 1]))
        @test cu.nodes == 3 && cu.ja == Int32[1, 2, 2, 1, 3, 3, 2] && cu.nodeuser == Int32[1, 3, 4]
    end
end
