# Convertible cells (T6.1): saturation functions, conductances and storage terms against
# hand-computed values (both precisions), THICKSTRT, the cell-bottom checks of riv_ck /
# drn_ck / ghb_ck, IBOUND at a period change, the RCH/EVT move to the highest active cell,
# and the confined fast path. Wetting and drying against mf6: test/models/wetdry (groups
# run and assembly).
using ModflowGPU: thksat, storage_saturation, storage_terms, connection_terms, ConvertibleNpf, ConvertibleStorage,
    DisSpec, NpfSpec, IcSpec, build_connectivity, npf_setup, npf_properties, calc_condsat, check_records,
    BoundaryKind, boundary_ibound, MF6InputError, load_simulation, GwfModel

@testset "Convertible: saturation (sgwf_npf_thksat, sQuadraticSaturation)" begin
    for_each_precision() do p
        T = p.T
        # top 10, bottom 0, reference 5: top' = 5, bot' = −5, thk = 10.
        tp, bt, th = T(5), T(-5), T(10)
        @test thksat(T(0), tp, bt, th) === T(0.5)
        @test thksat(T(5), tp, bt, th) === one(T)
        @test thksat(T(7), tp, bt, th) === one(T)
        @test thksat(T(-6), tp, bt, th) ≈ T(-0.1)       # not clamped below (the cell dries)
        @test storage_saturation(T(-6), tp, bt, th) === zero(T)
        @test storage_saturation(T(-2.5), tp, bt, th) === T(0.25)
        @test storage_saturation(T(6), tp, bt, th) === one(T)
    end
end

# A ConvertibleNpf of two cells and one connection (host arrays).
function _cv2(
        T; icelltype = Int32[1, 1], topoff, botoff, sat, condsat, hyl, hyh, cl1 = 50.0, cl2 = 50.0, hwva, ihcx,
        ivarcv = false, idewatcv = false, iperched = false, newton = false, ihighcellsat = false,
    )
    v(x...) = T[x...]
    thk = T.(topoff .- botoff)
    return ConvertibleNpf(
        icelltype, T.(topoff), T.(botoff), thk, T.(sat), v(condsat), v(0.0), v(hyl), v(hyh), v(cl1), v(cl2), v(hwva), Int32[ihcx],
        ivarcv, idewatcv, iperched, Int32(0), newton, ihighcellsat,
    )
end

@testset "Convertible: conductances (hcond, vcond, PERCHED)" begin
    jas = Int32[1, 1]
    ib = Int32[1, 1]
    Δr = [0.0]
    for_each_precision() do p
        T = p.T
        dr = T.(Δr)
        # Horizontal, harmonic: K 2 and 4, thickness 10, saturations 0.5 and 1, cl 50,
        # width 100: t1 = 10, t2 = 40, C = 100·10·40/(10·50 + 40·50) = 16.
        cv = _cv2(T; topoff = [5.0, 5.0], botoff = [-5.0, -5.0], sat = [0.5, 1.0], condsat = 99.0, hyl = 2.0, hyh = 4.0, hwva = 100.0, ihcx = 1)
        h = T[0, 1]
        a, dd, sb, q, c, _ = connection_terms(cv, dr, jas, ib, h, 1, 2, 1)
        @test c ≈ T(16) && a == c && dd == c && q ≈ T(16)
        a2, _, _, q2, c2, _ = connection_terms(cv, dr, jas, ib, h, 2, 1, 1)
        @test c2 == c && q2 == -q
        # Both confined: condsat.
        cvc = _cv2(T; icelltype = Int32[0, 0], topoff = [5.0, 5.0], botoff = [-5.0, -5.0], sat = [0.5, 1.0], condsat = 99.0, hyl = 2.0, hyh = 4.0, hwva = 100.0, ihcx = 1)
        @test connection_terms(cvc, dr, jas, ib, h, 1, 2, 1)[5] == T(99)
        # An inactive (dry) cell: no flow.
        @test connection_terms(cv, dr, jas, Int32[1, 0], h, 1, 2, 1)[5] == zero(T)
        # Vertical, VARIABLECV: upper cell 1 (bottom higher, ihcx 0 + 4) half saturated,
        # lower cell full; K33 1 and 2, thickness 10, area 1000.
        # Without DEWATERED the lower cell's saturation is used as 1: bovk = 0.5·10/2/1 + 1·10/2/2 = 5.
        cvv = _cv2(T; topoff = [5.0, 5.0], botoff = [-5.0, -5.0], sat = [0.5, 1.0], condsat = 7.0, hyl = 1.0, hyh = 2.0, hwva = 1000.0, ihcx = 4, ivarcv = true)
        @test connection_terms(cvv, dr, jas, ib, T[0, 0], 1, 2, 1)[5] ≈ T(200)
        # Both heads at or above the tops: condsat.
        @test connection_terms(cvv, dr, jas, ib, T[5, 5], 1, 2, 1)[5] == T(7)
        # DEWATERED with a partly saturated lower cell: its saturation becomes 0, bovk = 2.5.
        cvd = _cv2(T; topoff = [5.0, 5.0], botoff = [-5.0, -5.0], sat = [0.5, 0.8], condsat = 7.0, hyl = 1.0, hyh = 2.0, hwva = 1000.0, ihcx = 4, ivarcv = true, idewatcv = true)
        @test connection_terms(cvd, dr, jas, ib, T[0, 0], 1, 2, 1)[5] ≈ T(400)
        # T6.2 (from T6.1): DEWATERED with the upper cell at its bottom. The offsets resolve
        # a saturation of 1e-30 (MF6's absolute heads give exactly 0); without the guard
        # `hwva/denom` overflowed to Inf and the flows became NaN
        # (reference/issues/T6.1-perched-dewatered-nan, VARIABLECV DEWATERED PERCHED).
        for s in (T(1.0e-30), floatmin(T), nextfloat(zero(T)))
            cvz = _cv2(T; topoff = [5.0, 5.0], botoff = [-5.0, -5.0], sat = [Float64(s), 0.8], condsat = 7.0, hyl = 1.0, hyh = 2.0, hwva = 1000.0, ihcx = 4, ivarcv = true, idewatcv = true)
            tz = connection_terms(cvz, dr, jas, ib, T[-5, 0], 1, 2, 1)
            @test all(isfinite, tz) && tz[5] == zero(T)
        end
        # Just above eps(T) the conductance is that of the saturation.
        cve = _cv2(T; topoff = [5.0, 5.0], botoff = [-5.0, -5.0], sat = [2 * Float64(eps(T)), 0.8], condsat = 7.0, hyl = 1.0, hyh = 2.0, hwva = 1000.0, ihcx = 4, ivarcv = true, idewatcv = true)
        @test connection_terms(cve, dr, jas, ib, T[-5, 0], 1, 2, 1)[5] ≈ T(1000) / (2 * eps(T) * T(5))
        # PERCHED: lower cell below its top: flow C·(h_1 − bot_1) from cell 1, not A·h.
        cvp = _cv2(T; topoff = [5.0, 5.0], botoff = [-5.0, -5.0], sat = [1.0, 0.8], condsat = 7.0, hyl = 1.0, hyh = 2.0, hwva = 1000.0, ihcx = 4, iperched = true)
        hp = T[2, 3]
        a1, dd1, sb1, q1, c1, _ = connection_terms(cvp, dr, jas, ib, hp, 1, 2, 1)
        @test (a1, dd1, sb1, q1) == (zero(T), T(7), T(35), T(-49))  # q into 1 = 7·(−5 − 2)
        a2, dd2, sb2, q2, _, _ = connection_terms(cvp, dr, jas, ib, hp, 2, 1, 1)
        @test (a2, dd2, sb2, q2) == (T(7), zero(T), T(-35), T(49))
    end
end

@testset "Convertible: storage terms (SsTerms, SyTerms)" begin
    for_each_precision() do p
        T = p.T
        # rho1 = 2, rho2 = 10; top' 5, bot' −5, thk 10; hold' 0 (sn 0.5), h' −2.5 (sn 0.25).
        cs(iconf) = ConvertibleStorage(Int32[1], T[5], T[-5], T[10], T[10], iconf, false)
        rho, hold, h = T[2], T[0], T[-2.5]
        a1, r1, a2, r2 = storage_terms(rho, hold, cs(false), h, 1)
        # SsTerms: aterm = −2·0.25; zold = −2.5, znew = −3.75; rhs = −2·0.5·2.5 − 2·0.25·(−3.75).
        @test a1 ≈ T(-0.5) && r1 ≈ T(-0.625)
        # SyTerms (0 < sn < 1): aterm = −rho2, rhs = −rho2·thk·snold − rho2·bot' = 0.
        @test a2 == T(-10)
        @test abs(r2) <= 100 * eps(T)
        # Rates: SY = rho2·thk·(snold − snnew) = 25 = aterm·h − rhs.
        @test a2 * h[1] - r2 ≈ T(25)
        # SS_CONFINED_ONLY: neither saturation is 1, so no SS term.
        @test storage_terms(rho, hold, cs(true), h, 1)[1:2] == (zero(T), zero(T))
        # Both full: the confined terms.
        b1, s1, b2, s2 = storage_terms(rho, T[6], cs(false), T[7], 1)
        @test b1 == T(-2) && b2 == zero(T) && s2 == zero(T)
        @test b1 * T(7) - s1 ≈ T(2) * (T(6) - T(7))
        # ICONVERT 0: −rho, −rho·hold.
        c0 = ConvertibleStorage(Int32[0], T[5], T[-5], T[10], T[10], false, false)
        @test storage_terms(rho, hold, c0, h, 1) == (T(-2), -T(2) * T(0), zero(T), zero(T))
    end
end

@testset "Convertible: THICKSTRT (preprocess_input, calc_condsat)" begin
    dis = DisSpec(;
        nlay = Int32(1), nrow = Int32(1), ncol = Int32(2), delr = [100.0, 100.0], delc = [100.0],
        top = fill(10.0, 2, 1), botm = zeros(2, 1, 1), idomain = ones(Int32, 2, 1, 1),
    )
    c = build_connectivity(dis)
    npf(ict; kw...) = NpfSpec(; icelltype = Int32.(ict), k = [2.0, 2.0], file = "x.npf", kw...)
    full = npf_setup(npf([1, 1]), IcSpec(; strt = [5.0, 5.0]), dis, c)
    thick = npf_setup(npf([-1, -1]; thickstrt = true), IcSpec(; strt = [5.0, 5.0]), dis, c)
    @test thick.properties.icelltype == [0, 0] && thick.properties.thickstart == [true, true]
    @test thick.properties.sat0 == [0.5, 0.5]
    @test thick.condsat[1] ≈ 0.5 * full.condsat[1]
    # Without THICKSTRT a negative ICELLTYPE is convertible.
    @test npf_setup(npf([-1, 0]), IcSpec(; strt = [5.0, 5.0]), dis, c).properties.icelltype == [1, 0]
    # STRT below the cell bottom is an error.
    @test_throws MF6InputError npf_setup(npf([-1, -1]; thickstrt = true), IcSpec(; strt = [-1.0, 5.0]), dis, c)
end

# A host stand-in for a package in `check_records`.
_ckpkg(icelltype) = (; nodelist_host = Int32[1, 2], icelltype = Int32.(icelltype), conn = (; bot = [0.0, 0.0]), spec = (; file = "x", name = "p"))

@testset "Convertible: riv_ck, drn_ck, ghb_ck cell-bottom and conductance checks" begin
    conv, conf = _ckpkg([1, 1]), _ckpkg([0, 0])
    # RIV: stage, cond, rbot.
    riv = [1.0 1.0; 10.0 10.0; -1.0 0.5]
    @test_throws r"RIVER BOTTOM \(-1.0\) IS LESS THAN CELL BOTTOM" check_records(BoundaryKind{:RIV}(), conv, riv, 2)
    @test isnothing(check_records(BoundaryKind{:RIV}(), conf, riv, 2))
    @test_throws r"CONDUCTANCE \(-1.0\) IS LESS THAN ZERO" check_records(BoundaryKind{:RIV}(), conf, [1.0; -1.0; 0.5;;], 1)
    # GHB: bhead, cond.
    @test_throws r"GHB BOUNDARY \(2\) HEAD \(-2.0\) IS LESS THAN CELL BOTTOM" check_records(BoundaryKind{:GHB}(), conv, [1.0 -2.0; 1.0 1.0], 2)
    @test isnothing(check_records(BoundaryKind{:GHB}(), conf, [1.0 -2.0; 1.0 1.0], 2))
    # DRN: elev, cond, ddrn (scaled conductance bottom = min(elev, elev + ddrn)).
    @test_throws r"SCALED-CONDUCTANCE DRN BOUNDARY \(1\) BOTTOM ELEVATION \(-1.0\)" check_records(BoundaryKind{:DRN}(), conv, [1.0; 1.0; -2.0;;], 1)
    @test isnothing(check_records(BoundaryKind{:DRN}(), conv, [1.0; 1.0; 2.0;;], 1))
end

@testset "Convertible: IBOUND at a period change (chd_rp) and the fast path" begin
    # boundary_ibound with the IBOUND in force: dry cells stay dry, old CHD cells active.
    dis1 = DisSpec(;
        nlay = Int32(1), nrow = Int32(1), ncol = Int32(4), delr = fill(100.0, 4), delc = [100.0],
        top = fill(10.0, 4, 1), botm = zeros(4, 1, 1), idomain = ones(Int32, 4, 1, 1),
    )
    @test boundary_ibound(build_connectivity(dis1), [], Int32[0, -1, 1, 0]) == Int32[0, 1, 1, 0]
    dir = joinpath(REFERENCE_DIR, "runs", "ex-gwf-twri01", "tight")
    if !isdir(dir)
        @info "Skipping the convertible GwfModel checks: $dir not found"
    else
        sim = load_simulation(dir)
        for_each_precision() do p
            m = GwfModel(TEST_BACKEND, p.T, only(sim.models), 1)
            @test !isnothing(m.npf.cv) && !isnothing(m.wd)
            # Head-dependent conductances: the off-diagonals are rewritten at every
            # formulation, with a new id (the preconditioner then takes the full setup).
            ModflowGPU.apply_period!(m, 1)
            ModflowGPU.start_timestep!(m, 1.0)
            ModflowGPU.formulate!(m)
            id1 = m.offdiag_id
            ModflowGPU.formulate!(m)
            @test m.offdiag_id > id1
        end
    end
    # A confined model keeps the constant-conductance path (no ConvertibleNpf, no WetDry).
    mc = joinpath(@__DIR__, "..", "models", "coupling_rde")
    m = GwfModel(TEST_BACKEND, Float64, only(load_simulation(mc).models), 3)
    @test isnothing(m.npf.cv) && isnothing(m.wd)
    ModflowGPU.apply_period!(m, 1)
    ModflowGPU.start_timestep!(m, 1.0)
    ModflowGPU.formulate!(m)
    id1 = m.offdiag_id
    ModflowGPU.formulate!(m)
    @test m.offdiag_id == id1
end
