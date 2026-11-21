# RIV, DRN, EVT (T2.4): period data on the host (columns, MULTIPLIER, AUXDEPTHNAME, NSEG
# and SURF_RATE_SPECIFIED, offsets, checks), and the formulate / add / simvals kernels on
# TEST_BACKEND in every precision for each branch of `riv_cf`, `drn_cf` (incl. the DRNDEPTH
# scaling) and `evt_cf` (above the surface, within the extinction depth with one and with
# three segments, below it), against MF6's formulas evaluated here in Float64 with
# absolute heads.
using ModflowGPU: load_simulation, build_connectivity, BoundaryPackage, apply_period!, formulate!, add_to_system!, calc_simvals!,
    boundary_ibound, highest_active, reference_heads, linear_saturation, MF6UnsupportedError, MF6InputError

# Records: (layer, row, col) => values, and the absolute head given to that cell.
const _RIV_RECORDS = [  # stage, cond, rbot, mult; head
    ((1, 2, 2), (240.0, 10.0, 235.0, 2.0), 242.0),   # head above rbot (gaining river)
    ((1, 2, 3), (240.0, 10.0, 235.0, 1.0), 230.0),   # head below rbot (constant leakage)
    ((1, 2, 4), (240.0, 7.5, 235.0, 1.0), 235.0),    # head at rbot (constant leakage)
    ((2, 3, 3), (100.0, 4.0, 99.0, 0.5), 99.5),      # head between rbot and stage
]
const _DRN_RECORDS = [  # elev, cond, ddrn; head
    ((1, 3, 2), (230.0, 5.0, 0.0), 231.0),     # on
    ((1, 3, 3), (230.0, 5.0, 0.0), 229.0),     # off
    ((1, 3, 5), (230.0, 5.0, 0.0), 230.0),     # at the elevation: off
    ((1, 4, 2), (230.0, 6.0, 2.0), 231.0),     # scaled, f = 0.5
    ((1, 4, 3), (230.0, 6.0, -2.0), 229.5),    # negative depth: drnbot = 228, f = 0.75
    ((1, 4, 4), (230.0, 6.0, 2.0), 233.0),     # above the scaling range: f = 1
    ((1, 4, 5), (230.0, 6.0, 2.0), 229.0),     # below it: f = 0
]
const _EVT_PXDP = (0.3, 0.6)
const _EVT_PETM = (0.7, 0.2)
const _EVT_RECORDS = [  # surface, rate, depth, petm0; head (NSEG 3, SURF_RATE_SPECIFIED)
    ((1, 5, 2), (245.0, 0.001, 10.0, 0.9), 246.0),   # above the surface: PETM0·rate
    ((1, 3, 6), (245.0, 0.001, 10.0, 0.9), 245.0),   # at the surface
    ((1, 5, 3), (245.0, 0.001, 10.0, 0.9), 243.0),   # segment 1 (d = 2)
    ((1, 5, 4), (245.0, 0.001, 10.0, 0.9), 240.0),   # segment 2 (d = 5)
    ((1, 5, 5), (245.0, 0.002, 10.0, 0.9), 237.0),   # segment 3 (d = 8)
    ((1, 5, 6), (245.0, 0.001, 10.0, 0.9), 230.0),   # below the extinction depth
    ((1, 4, 6), (245.0, 0.001, 10.0, 0.9), 235.0),   # at the extinction depth
]
const _EVT1_RECORDS = [  # surface, rate, depth, mult; head (NSEG 1)
    ((1, 2, 5), (245.0, 0.002, 5.0, 1.5), 243.0),   # within the extinction depth
    ((1, 2, 6), (245.0, 0.002, 5.0, 1.0), 239.0),   # below it
    ((1, 3, 7), (245.0, 0.002, 5.0, 1.0), 247.0),   # above the surface: rate
]

_cellstr(c) = join(c, " ")

function _write_rde_simulation(dir)
    dis = _synthetic_coupling_dis()
    strt = [dis.top[j, i] - 3.0 - 0.5 * k for j in 1:dis.ncol, i in 1:dis.nrow, k in 1:dis.nlay]
    write_dis_simulation(dir, dis; name = "rde", strt)
    write_input(
        dir, "rde.nam", """
        BEGIN packages
          DIS6  rde.dis  dis
          NPF6  rde.npf  npf
          IC6  rde.ic  ic
          RIV6  rde.riv  riv
          DRN6  rde.drn  drn
          EVT6  rde.evt  evt
          EVT6  rde.evt1  evt1
          EVT6  rde.evta  evta
        END packages
        """,
    )
    rows(recs) = join(("  " * _cellstr(c) * "  " * join(v, "  ") for (c, v, _) in recs), "\n")
    write_input(
        dir, "rde.riv", """
        BEGIN options
          AUXILIARY  mult
          AUXMULTNAME  mult
          BOUNDNAMES
        END options
        BEGIN dimensions
          MAXBOUND  4
        END dimensions
        BEGIN period  1
        $(join(("  " * _cellstr(c) * "  " * join(v, "  ") * "  r$k" for (k, (c, v, _)) in enumerate(_RIV_RECORDS)), "\n"))
        END period
        """,
    )
    write_input(
        dir, "rde.drn", """
        BEGIN options
          AUXILIARY  ddrn
          AUXDEPTHNAME  ddrn
        END options
        BEGIN dimensions
          MAXBOUND  7
        END dimensions
        BEGIN period  1
        $(rows(_DRN_RECORDS))
        END period
        """,
    )
    write_input(
        dir, "rde.evt", """
        BEGIN options
          SURF_RATE_SPECIFIED
        END options
        BEGIN dimensions
          MAXBOUND  7
          NSEG  3
        END dimensions
        BEGIN period  1
        $(join(("  " * _cellstr(c) * "  $(v[1])  $(v[2])  $(v[3])  $(join(_EVT_PXDP, " "))  $(join(_EVT_PETM, " "))  $(v[4])" for (c, v, _) in _EVT_RECORDS), "\n"))
        END period
        """,
    )
    write_input(
        dir, "rde.evt1", """
        BEGIN options
          AUXILIARY  m
          AUXMULTNAME  m
          FIXED_CELL
        END options
        BEGIN dimensions
          MAXBOUND  3
        END dimensions
        BEGIN period  1
        $(rows(_EVT1_RECORDS))
        END period
        """,
    )
    ncpl = dis.nrow * dis.ncol
    ievt = ones(Int, ncpl)
    ievt[(3 - 1) * dis.ncol + 4] = 2   # the layer-1 hole: layer 2
    write_input(
        dir, "rde.evta", """
        BEGIN options
          READASARRAYS
        END options
        BEGIN period  1
          ievt
            INTERNAL
              $(join(ievt, " "))
          surface
            INTERNAL
              $(join((string(dis.top[c]) for c in 1:ncpl), " "))
          rate
            CONSTANT  0.0025
          depth
            CONSTANT  8.0
        END period
        """,
    )
    return dir
end

# MF6's formulations in Float64 with absolute heads: (hcof, rhs).
function _riv_ref(stage, c, rbot, h)
    return h <= rbot ? (0.0, -c * (stage - rbot)) : (-c, -c * stage)
end
function _drn_ref(elev, c, ddrn, h)
    if ddrn != 0
        top, bot = max(elev + ddrn, elev), min(elev + ddrn, elev)
        f = clamp((h - bot) / (top - bot), 0.0, 1.0)
    else
        bot = elev
        f = h <= bot ? 0.0 : 1.0
    end
    return (-f * c, -f * c * bot)
end
function _evt_ref(s, c, x, petm0, pxdp, petm, h)
    h >= s && return (0.0, petm0 * c)
    d = s - h
    d < x || return (0.0, 0.0)
    nseg = length(pxdp) + 1
    nseg == 1 && return (-c / x, c - c * s / x)
    p1, e1 = 0.0, petm0
    p2, e2 = 1.0, 0.0
    for iseg in 1:nseg
        p2, e2 = iseg < nseg ? (pxdp[iseg], petm[iseg]) : (1.0, 0.0)
        d <= p2 * x && break
        p1, e1 = p2, e2
    end
    hc = -(e1 - e2) * c / ((p2 - p1) * x)
    return (hc, hc * (s - p1 * x) + e1 * c)
end

@testset "RIV, DRN, EVT (T2.4)" begin
    @testset "linear_saturation (sQuadraticSaturation, eps = 0)" begin
        for T in (Float32, Float64)
            @test linear_saturation(T(2), T(0), T(-1)) === zero(T)
            @test linear_saturation(T(2), T(0), T(0)) === zero(T)
            @test linear_saturation(T(2), T(0), T(0.5)) === T(0.25)
            @test linear_saturation(T(2), T(0), T(2)) === one(T)
            @test linear_saturation(T(2), T(0), T(3)) === one(T)
            @test linear_saturation(T(1), T(1), T(1)) === one(T)    # no range: step at bot
            @test linear_saturation(T(1), T(1), T(0.5)) === zero(T)
        end
    end

    mktempdir() do dir
        _write_rde_simulation(dir)
        m = only(load_simulation(dir; strict = true).models)
        conn = build_connectivity(m)
        r = reference_heads(conn, m.ic.strt)
        byname = Dict(b.name => b for b in m.boundaries)
        red(l, i, j) = conn.nodereduced[(l - 1) * 35 + (i - 1) * 7 + j]
        nodes = conn.nodes
        # Absolute heads: the record heads, elsewhere the reference.
        habs = copy(r)
        for recs in (_RIV_RECORDS, _DRN_RECORDS, _EVT_RECORDS, _EVT1_RECORDS), (c, _, hv) in recs
            habs[red(c...)] = hv
        end
        hoff = habs .- r

        @testset "spec and record checks" begin
            evt = byname["EVT"]
            with(s; kw...) = typeof(s)(; (f => getfield(s, f) for f in fieldnames(typeof(s)))..., kw...)
            @test_throws MF6InputError BoundaryPackage(TEST_BACKEND, Float64, with(evt; nseg = Int32(0)), conn, m.dis)
            evta = byname["EVTA"]
            @test_throws MF6InputError BoundaryPackage(TEST_BACKEND, Float64, with(evta; nseg = Int32(2)), conn, m.dis)
            @test_throws MF6InputError BoundaryPackage(TEST_BACKEND, Float64, with(evta; surf_rate_specified = true), conn, m.dis)
            @test_throws MF6InputError BoundaryPackage(TEST_BACKEND, Float64, with(byname["DRN"]; auxdepthname = "NOPE"), conn, m.dis)
            # riv_ck: stage below rbot; check_pxdp: PXDP not increasing / outside (0, 1).
            riv = byname["RIV"]
            p = riv.periods.data[1]
            bad = copy(p.bound)
            bad[3, 1] = 241.0
            rb = with(riv; periods = typeof(riv.periods)([1], [typeof(p)(p.nodeu, bad, p.auxvar, p.boundname, p.ts)]))
            @test_throws MF6InputError apply_period!(BoundaryPackage(TEST_BACKEND, Float64, rb, conn, m.dis), 1, r)
            p = evt.periods.data[1]
            c = ModflowGPU.list_column(evt.schema, "pxdp").first
            for (a, b) in ((0.6, 0.3), (0.0, 0.5), (0.5, 1.0))
                bad = copy(p.bound)
                bad[c, :] .= a
                bad[c + 1, :] .= b
                eb = with(evt; periods = typeof(evt.periods)([1], [typeof(p)(p.nodeu, bad, p.auxvar, p.boundname, p.ts)]))
                @test_throws MF6InputError apply_period!(BoundaryPackage(TEST_BACKEND, Float64, eb, conn, m.dis), 1, r)
            end
        end

        @testset "unsupported options" begin
            for (pkg, line) in (("rde.drn", "  DEV_CUBIC_SCALING"), ("rde.riv", "  MOVER"))
                d2 = mktempdir()
                cp(dir, d2; force = true)
                f = joinpath(d2, pkg)
                write(f, replace(read(f, String), "BEGIN options\n" => "BEGIN options\n$line\n"; count = 1))
                @test_throws MF6UnsupportedError load_simulation(d2; strict = true)
            end
        end

        @testset "period data on the host" begin
            riv = BoundaryPackage(TEST_BACKEND, Float64, byname["RIV"], conn, m.dis)
            apply_period!(riv, 1, r)
            n1 = red(1, 2, 2)
            @test ModflowGPU.bound_values(riv, r)[:, 1] == [240.0 - r[n1], 20.0, 235.0 - r[n1]]   # MULTIPLIER on cond
            @test riv.boundname == ["R1", "R2", "R3", "R4"] || riv.boundname == ["r1", "r2", "r3", "r4"]
            drn = BoundaryPackage(TEST_BACKEND, Float64, byname["DRN"], conn, m.dis)
            apply_period!(drn, 1, r)
            n = red(1, 4, 3)
            @test ModflowGPU.bound_values(drn, r)[:, 5] == [230.0 - r[n], 6.0, -2.0]
            evt = BoundaryPackage(TEST_BACKEND, Float64, byname["EVT"], conn, m.dis)
            apply_period!(evt, 1, r)
            n = red(1, 5, 2)
            @test size(ModflowGPU.bound_values(evt, r), 1) == 8
            @test ModflowGPU.bound_values(evt, r)[:, 1] == [245.0 - r[n], 0.001, 10.0, 0.9, 0.3, 0.6, 0.7, 0.2]
            evt1 = BoundaryPackage(TEST_BACKEND, Float64, byname["EVT1"], conn, m.dis)
            apply_period!(evt1, 1, r)
            @test ModflowGPU.bound_values(evt1, r)[:, 1] == [245.0 - r[red(1, 2, 5)], 0.003, 5.0, 1.0]   # MULTIPLIER on rate, PETM0 = 1
        end

        @testset "EVT: highest active cell (no FIXED_CELL)" begin
            ibh = ones(Int32, nodes)
            n11 = red(1, 1, 1)
            ibh[n11] = 0
            evta = BoundaryPackage(TEST_BACKEND, Float64, byname["EVTA"], conn, m.dis)
            apply_period!(evta, 1, r; ibound = ibh)
            n21 = highest_active(conn, n11, ibh)
            @test evta.nodelist_host[1] == n21 == red(2, 1, 1)
            @test ModflowGPU.bound_values(evta, r)[1, 1] == m.dis.top[1, 1] - r[n21]   # offset from the moved cell
            evt1 = BoundaryPackage(TEST_BACKEND, Float64, byname["EVT1"], conn, m.dis)
            ibh[red(1, 2, 5)] = 0
            apply_period!(evt1, 1, r; ibound = ibh)
            @test evt1.nodelist_host[1] == red(1, 2, 5)   # FIXED_CELL
        end

        @testset "EVT moves keep device BOUND writes (T6.9)" begin
            for_each_precision() do p
                T = p.T
                n11 = red(1, 1, 1)
                n21 = red(2, 1, 1)
                evta = BoundaryPackage(TEST_BACKEND, T, byname["EVTA"], conn, m.dis)
                apply_period!(evta, 1, r)
                @test !evta.device_written
                # Host values only: the moved record's surface is restaged from the host.
                ibh = ones(Int32, nodes)
                ibh[n11] = 0
                @test ModflowGPU.move_to_highest_active!(evta, ibh, r)
                @test Array(evta.bound) == T.(ModflowGPU.bound_values(evta, r))
                @test ModflowGPU.move_to_highest_active!(evta, ones(Int32, nodes), r)   # back
                # Written on the device (what `device_bound!` marks): every value is kept,
                # the moved record's surface offset is shifted to the new cell's reference.
                written = Array(evta.bound)
                written[1, :] .= T(1.25)
                written[2, :] .= T(0.005)
                copyto!(evta.bound, written)
                evta.device_written = true
                @test ModflowGPU.move_to_highest_active!(evta, ibh, r)
                @test evta.nodelist_host[1] == n21
                moved = Array(evta.bound)
                @test moved[1, 1] == T(1.25) + T(r[n11] - r[n21])
                @test moved[:, 2:end] == written[:, 2:end]
                @test moved[2:end, 1] == written[2:end, 1]
                # Same IBOUND again: no record moves, nothing is written.
                @test !ModflowGPU.move_to_highest_active!(evta, ibh, r)
            end
        end

        for_each_precision() do p
            T = p.T
            area = to_test_device(T.(conn.area))
            h = to_test_device(T.(hoff))
            hh = Array(h)
            ibh = ones(Int32, nodes)
            ib = to_test_device(ibh)
            idiag = to_test_device(Int32.(1:nodes))

            # formulate + add + simvals for one package, against Float64 references (hcof,
            # rhs with absolute heads); returns the device hcof.
            function check(pkg, refs)
                apply_period!(pkg, 1, r)
                formulate!(pkg, h, ib, area)
                diag = KernelAbstractions.zeros(TEST_BACKEND, T, nodes)
                b = KernelAbstractions.zeros(TEST_BACKEND, T, nodes)
                add_to_system!(diag, b, idiag, pkg)
                calc_simvals!(pkg, h, ib)
                KernelAbstractions.synchronize(TEST_BACKEND)
                n = pkg.nbound
                @test n == length(refs)
                hcof = Array(pkg.hcof)[1:n]
                rhs = Array(pkg.rhs)[1:n]
                sv = Array(pkg.simvals)[1:n]
                nl = pkg.nodelist_host[1:n]
                # Offset form: rhs' = rhs − hcof·r; the flow hcof·h − rhs is unchanged.
                qref = [nl[i] > 0 ? refs[i][1] * habs[nl[i]] - refs[i][2] : 0.0 for i in 1:n]
                rref = [nl[i] > 0 ? refs[i][2] - refs[i][1] * r[nl[i]] : 0.0 for i in 1:n]
                hc = [refs[i][1] for i in 1:n]
                # Scale of each record's terms, for the absolute tolerance in Float32.
                scale = [abs(hc[i]) * (abs(hoff[max(nl[i], 1)]) + 1) + abs(rref[i]) + abs(qref[i]) for i in 1:n]
                tol = T === Float64 ? 1.0e-13 : 1.0e-5
                @test all(abs.(hcof .- hc) .<= tol .* abs.(hc))
                @test all(abs.(rhs .- rref) .<= tol .* scale)
                @test all(abs.(sv .- qref) .<= tol .* scale)
                @test sv == [nl[i] > 0 ? hcof[i] * hh[nl[i]] - rhs[i] : zero(T) for i in 1:n]
                dref = zeros(T, nodes)
                bref = zeros(T, nodes)
                for i in 1:n
                    nl[i] > 0 || continue
                    dref[nl[i]] += hcof[i]
                    bref[nl[i]] += rhs[i]
                end
                @test Array(diag) == dref && Array(b) == bref
                return hcof, sv
            end

            @testset "RIV" begin
                refs = [_riv_ref(v[1], v[2] * v[4], v[3], hv) for (_, v, hv) in _RIV_RECORDS]
                hcof, sv = check(BoundaryPackage(TEST_BACKEND, T, byname["RIV"], conn, m.dis), refs)
                @test hcof[2] == hcof[3] == 0 && hcof[1] == -20 && hcof[4] == -2   # rbot branches
                @test sv[2] ≈ 50 && sv[3] ≈ 37.5   # constant leakage C·(stage − rbot) into the cell
            end

            @testset "DRN (DRNDEPTH scaling)" begin
                refs = [_drn_ref(v..., hv) for (_, v, hv) in _DRN_RECORDS]
                hcof, sv = check(BoundaryPackage(TEST_BACKEND, T, byname["DRN"], conn, m.dis), refs)
                @test hcof ≈ T[-5, 0, 0, -3, -4.5, -6, 0]
                @test sv[1] ≈ -5 && sv[4] ≈ -3 && sv[5] ≈ -4.5 * 1.5 && sv[6] ≈ -18
            end

            @testset "EVT, NSEG 3, SURF_RATE_SPECIFIED" begin
                refs = [_evt_ref(v[1], v[2] * conn.area[red(c...)], v[3], v[4], _EVT_PXDP, _EVT_PETM, hv) for (c, v, hv) in _EVT_RECORDS]
                hcof, sv = check(BoundaryPackage(TEST_BACKEND, T, byname["EVT"], conn, m.dis), refs)
                a(c) = conn.area[red(c...)]
                @test hcof[[1, 2, 6, 7]] == zeros(T, 4)
                @test sv[1] ≈ -0.9 * 0.001 * a((1, 5, 2)) && sv[2] ≈ -0.9 * 0.001 * a((1, 3, 6))
                @test sv[6] == 0 && abs(sv[7]) <= 1.0e-6 * 0.001 * a((1, 4, 6))
                # Segment 1 (d = 2, pxdp 0 → 0.3, petm 0.9 → 0.7): rate 0.9 − 0.2·2/3.
                @test sv[3] ≈ -(0.9 - 0.2 * 2 / 3) * 0.001 * a((1, 5, 3)) rtol = 1.0e-5
                @test sv[4] ≈ -(0.7 - 0.5 * 2 / 3) * 0.001 * a((1, 5, 4)) rtol = 1.0e-5
                @test sv[5] ≈ -(0.2 - 0.2 * 2 / 4) * 0.002 * a((1, 5, 5)) rtol = 1.0e-5
            end

            @testset "EVT, NSEG 1, MULTIPLIER" begin
                refs = [_evt_ref(v[1], v[2] * v[4] * conn.area[red(c...)], v[3], 1.0, (), (), hv) for (c, v, hv) in _EVT1_RECORDS]
                hcof, sv = check(BoundaryPackage(TEST_BACKEND, T, byname["EVT1"], conn, m.dis), refs)
                @test sv[1] ≈ -0.003 * conn.area[red(1, 2, 5)] * 0.6 rtol = 1.0e-5
                @test sv[2] == 0 && sv[3] ≈ -0.002 * conn.area[red(1, 3, 7)]
            end

            @testset "EVTA (READASARRAYS)" begin
                evta = BoundaryPackage(TEST_BACKEND, T, byname["EVTA"], conn, m.dis)
                apply_period!(evta, 1, r)
                @test evta.nbound == 35
                nl = evta.nodelist_host[1:35]
                @test nl[(3 - 1) * 7 + 4] == red(2, 3, 4)
                top = vec(m.dis.top)
                refs = [nd > 0 ? _evt_ref(top[c], 0.0025 * conn.area[nd], 8.0, 1.0, (), (), habs[nd]) : (0.0, 0.0) for (c, nd) in enumerate(nl)]
                check(evta, refs)
            end
        end
    end
end
