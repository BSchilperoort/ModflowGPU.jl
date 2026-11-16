# NPF, confined (T2.2): saturated conductance on small grids against hand formulas (each
# averaging method, vertical connections, anisotropy and angles), the reference heads and
# differences of PLAN.md section 3.7 rule 1, MF6's NPF input checks, and the conversion to
# the engine precision. Host setup is Float64; the precision presets enter through
# `NpfModel{T}` and the generic `condmean`/`logmean`. The libmf6 oracle is in
# test/parity/npf_test.jl.
using ModflowGPU: DisSpec, NpfSpec, IcSpec, Connectivity, build_connectivity, npf_setup, npf_properties,
    calc_condsat, hy_eff, hyeff, condmean, logmean, NpfModel, reference_heads, reference_differences,
    reference_difference, CCOND_HMEAN, CCOND_LMEAN, CCOND_AMTLMK, CCOND_AMTHMK, MF6InputError, MF6UnsupportedError

function _npf_dis(; delr, delc, top, botm, idomain = ones(Int32, size(botm)))
    ncol, nrow, nlay = size(botm)
    return DisSpec(;
        nlay = Int32(nlay), nrow = Int32(nrow), ncol = Int32(ncol), delr = Float64.(delr), delc = Float64.(delc),
        top = Float64.(reshape(top, ncol, nrow)), botm = Float64.(botm), idomain = Int32.(idomain), file = "test.dis",
    )
end

_npf(n; kw...) = NpfSpec(; icelltype = zeros(Int32, n), k = ones(n), file = "test.npf", kw...)

_err(f) = try
    f()
    nothing
catch e
    e
end

# Two cells side by side (x direction): widths 100 and 60 m, row width 40 m, thicknesses
# 10 and 8 m.
const _DIS_X2 = _npf_dis(; delr = [100.0, 60.0], delc = [40.0], top = [20.0, 18.0], botm = reshape([10.0, 10.0], 2, 1, 1))

@testset "NPF: averaging methods on two cells" begin
    c = build_connectivity(_DIS_X2)
    @test c.njas == 1
    w, cl1, cl2, b1, b2 = 40.0, 50.0, 30.0, 10.0, 8.0
    for (k1, k2) in ((2.0, 0.5), (3.0, 3.003), (1.0e-6, 25.0))
        t1, t2 = k1 * b1, k2 * b2
        lm(a, b) = 0.995 < b / a < 1.005 ? (a + b) / 2 : (b - a) / log(b / a)
        expected = Dict(
            :harmonic => w * t1 * t2 / (t1 * cl2 + t2 * cl1),
            :logarithmic => lm(t1, t2) * w / (cl1 + cl2),
            :amt_lmk => lm(k1, k2) * (b1 + b2) / 2 * w / (cl1 + cl2),
            :amt_hmk => k1 * k2 / (k1 * cl2 + k2 * cl1) * (b1 + b2) / 2 * w,
        )
        for (avg, ref) in expected
            npf = _npf(2; k = [k1, k2], cell_averaging = avg)
            s = npf_setup(npf, IcSpec(; strt = [15.0, 15.0]), _DIS_X2, c)
            @test s.condsat[1] ≈ ref rtol = 1.0e-14
        end
    end
    # logmean: the arithmetic mean within 0.5 %, and the log mean outside (symmetric).
    @test logmean(2.0, 2.009) == 0.5 * (2.0 + 2.009)
    @test logmean(2.0, 4.0) ≈ 2.0 / log(2.0) rtol = 1.0e-15
    @test logmean(4.0, 2.0) ≈ logmean(2.0, 4.0) rtol = 1.0e-15
    # Zero transmissivity gives zero conductance in every method.
    for m in (CCOND_HMEAN, CCOND_LMEAN, CCOND_AMTLMK, CCOND_AMTHMK)
        @test condmean(0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, m) == 0.0
    end
    # Generic in the precision: no promotion to Float64.
    for_each_precision() do p
        T = p.T
        args = T.((2.0, 0.5, 10.0, 8.0, 50.0, 30.0, 40.0))
        for m in (CCOND_HMEAN, CCOND_LMEAN, CCOND_AMTLMK, CCOND_AMTHMK)
            v = condmean(args..., m)
            @test v isa T
            @test v ≈ condmean(Float64.(args)..., m) rtol = 4 * eps(T)
        end
        @test logmean(T(2), T(2.009)) isa T
    end
end

@testset "NPF: vertical connections" begin
    # One column of three layers with thicknesses 10, 5, 20 and plan area 30 × 20.
    dis = _npf_dis(; delr = [30.0], delc = [20.0], top = [100.0], botm = reshape([90.0, 85.0, 65.0], 1, 1, 3))
    c = build_connectivity(dis)
    k = [1.0, 0.25, 4.0]
    k33 = [0.1, 0.002, 0.5]
    area = 600.0
    vc(b1, k1, b2, k2) = area / (b1 / 2 / k1 + b2 / 2 / k2)
    s = npf_setup(_npf(3; k, k33), IcSpec(; strt = zeros(3)), dis, c)
    @test s.condsat ≈ [vc(10.0, 0.1, 5.0, 0.002), vc(5.0, 0.002, 20.0, 0.5)] rtol = 1.0e-14
    # K33 = K when not given; the averaging method does not apply to vertical connections.
    for avg in (:harmonic, :logarithmic, :amt_lmk, :amt_hmk)
        s = npf_setup(_npf(3; k, cell_averaging = avg), IcSpec(; strt = zeros(3)), dis, c)
        @test s.condsat ≈ [vc(10.0, 1.0, 5.0, 0.25), vc(5.0, 0.25, 20.0, 4.0)] rtol = 1.0e-14
    end
    # K33OVERK: K33 is a ratio.
    s = npf_setup(_npf(3; k, k33 = [0.1, 0.008, 0.125], k33overk = true), IcSpec(; strt = zeros(3)), dis, c)
    @test s.properties.k33 ≈ [0.1, 0.002, 0.5] rtol = 1.0e-15
    @test s.condsat ≈ [vc(10.0, 0.1, 5.0, 0.002), vc(5.0, 0.002, 20.0, 0.5)] rtol = 1.0e-14
end

@testset "NPF: anisotropy and angles (hy_eff)" begin
    # 2 × 2 cells, one layer: connections 1-2 (x, right), 1-3 (y, front), 2-4, 3-4.
    dis = _npf_dis(; delr = [10.0, 10.0], delc = [10.0, 10.0], top = fill(10.0, 4), botm = zeros(2, 2, 1))
    c = build_connectivity(dis)
    k11, k22 = 4.0, 1.0
    ic = IcSpec(; strt = zeros(4))
    # Harmonic mean of equal cells is the cell value, times thickness × width / distance.
    hk(jas, s) = s.condsat[jas] / (10.0 * 10.0 / 10.0)
    jx, jy = c.jas[c.ia[1] + 1], c.jas[c.ia[1] + 2]
    @test c.ihc[jx] == 1 && c.anglex[jx] == 0.0 && c.anglex[jy] == 1.5 * pi
    # K22 without angles: K along x (rows), K22 along y (columns).
    s = npf_setup(_npf(4; k = fill(k11, 4), k22 = fill(k22, 4)), ic, dis, c)
    @test hk(jx, s) == k11
    @test hk(jy, s) == k22
    # K22OVERK.
    s = npf_setup(_npf(4; k = fill(k11, 4), k22 = fill(0.25, 4), k22overk = true), ic, dis, c)
    @test hk(jx, s) == k11 && hk(jy, s) == k22
    # ANGLE1 θ rotates the ellipse: K(x) = k11 k22 / (k22 cos²θ + k11 sin²θ),
    # K(y) = k11 k22 / (k22 sin²θ + k11 cos²θ).
    for θ in (30.0, -45.0, 90.0, 120.0)
        s = npf_setup(_npf(4; k = fill(k11, 4), k22 = fill(k22, 4), angle1 = fill(θ, 4)), ic, dis, c)
        cθ, sθ = cosd(θ), sind(θ)
        @test hk(jx, s) ≈ k11 * k22 / (k22 * cθ^2 + k11 * sθ^2) rtol = 1.0e-14
        @test hk(jy, s) ≈ k11 * k22 / (k22 * sθ^2 + k11 * cθ^2) rtol = 1.0e-14
        @test s.properties.angle1 ≈ fill(deg2rad(θ), 4) rtol = 1.0e-15
    end
    # ANGLE1 without K22 has no effect on horizontal connections.
    s = npf_setup(_npf(4; k = fill(k11, 4), angle1 = fill(30.0, 4)), ic, dis, c)
    @test hk(jx, s) == k11 && hk(jy, s) == k11

    # hyeff: a principal direction matching the vector exactly gives that value.
    @test hyeff(4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0) == 4.0
    @test hyeff(4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0) == 2.0
    @test hyeff(4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0) == 1.0
    # Vertical connections: K33 unless ANGLE2 is given; then the rotated ellipsoid along z:
    # 1/K = sin²a2/k11 + cos²a2 sin²a3/k22 + cos²a2 cos²a3/k33.
    dis3 = _npf_dis(; delr = [10.0], delc = [10.0], top = [20.0], botm = reshape([10.0, 0.0], 1, 1, 2))
    c3 = build_connectivity(dis3)
    k33 = 0.5
    a1, a2, a3 = 30.0, 20.0, 10.0
    kz = 1 / (sind(a2)^2 / k11 + cosd(a2)^2 * sind(a3)^2 / k22 + cosd(a2)^2 * cosd(a3)^2 / k33)
    base = (; k = fill(k11, 2), k22 = fill(k22, 2), k33 = fill(k33, 2))
    ic3 = IcSpec(; strt = zeros(2))
    vcond(kv) = 100.0 / (5.0 / kv + 5.0 / kv)
    s = npf_setup(_npf(2; base..., angle1 = fill(a1, 2)), ic3, dis3, c3)
    @test s.condsat[1] == vcond(k33)
    s = npf_setup(_npf(2; base..., angle1 = fill(a1, 2), angle2 = fill(a2, 2), angle3 = fill(a3, 2)), ic3, dis3, c3)
    @test s.condsat[1] ≈ vcond(kz) rtol = 1.0e-14
    p = s.properties
    @test hy_eff(p, 1, 0, (0.0, 0.0, -1.0)) ≈ kz rtol = 1.0e-14
    @test hy_eff(p, 2, 0, (0.0, 0.0, 1.0)) == hy_eff(p, 1, 0, (0.0, 0.0, -1.0))
    # Horizontal with all three angles: 1/K = Σ ve_i²/k_i with ve = Rᵀ x̂.
    ve = (cosd(a1) * cosd(a2), cosd(a1) * sind(a2) * sind(a3) - sind(a1) * cosd(a3), -cosd(a1) * sind(a2) * cosd(a3) - sind(a1) * sind(a3))
    @test hy_eff(p, 1, 1, (1.0, 0.0, 0.0)) ≈ 1 / (ve[1]^2 / k11 + ve[2]^2 / k22 + ve[3]^2 / k33) rtol = 1.0e-14
end

@testset "NPF: reference heads and differences" begin
    # Two columns of three layers; STRT above the top, inside and below the bottom.
    dis = _npf_dis(; delr = [10.0, 10.0], delc = [10.0], top = [100.0, 102.0], botm = reshape([90.0, 92.0, 80.0, 82.0, 70.0, 72.0], 2, 1, 3))
    c = build_connectivity(dis)
    strt = [105.0, 95.0, 85.0, 81.0, 60.0, 72.0]
    r = reference_heads(c, strt)
    @test r == [100.0, 95.0, 85.0, 82.0, 70.0, 72.0]
    # Confined cells (ICELLTYPE 0) keep STRT (rule 1, revised after T2.6); the clamp only
    # for convertible cells and for implausible STRT (more than max(zmax − zmin, 1000 m)
    # outside the model's elevation range [70, 102]).
    ict = Int32[0, 1, 0, 0, 0, 0]
    @test reference_heads(c, strt; icelltype = ict) == [105.0, 95.0, 85.0, 81.0, 60.0, 72.0]
    @test reference_heads(c, [1.0e30, 95.0, 85.0, -1.0e4, NaN, 72.0]; icelltype = ict) == [100.0, 95.0, 85.0, 82.0, 70.0, 72.0]
    @test reference_heads(c, [1101.0, 95, 85, 81, 60, 72]; icelltype = ict)[1] == 1101.0
    @test reference_heads(c, [1103.0, 95, 85, 81, 60, 72]; icelltype = ict)[1] == 100.0
    @test_throws ArgumentError reference_heads(c, strt; icelltype = Int32[0])
    Δr = reference_differences(c, r)
    @test length(Δr) == c.njas
    for n in 1:c.nodes, ii in (c.ia[n] + 1):(c.ia[n + 1] - 1)
        m = c.ja[ii]
        @test reference_difference(Δr, c, n, ii) == r[m] - r[n]
        m > n && @test Δr[c.jas[ii]] == r[m] - r[n]
    end
    # Excluded cells: by user node in, by reduced node out.
    dis0 = _npf_dis(; delr = [10.0, 10.0], delc = [10.0], top = [100.0, 100.0], botm = reshape([90.0, 90.0], 2, 1, 1), idomain = reshape(Int32[0, 1], 2, 1, 1))
    c0 = build_connectivity(dis0)
    @test reference_heads(c0, [1000.0, 93.5]) == [93.5]
    @test_throws ArgumentError reference_heads(c0, [93.5])
    # The setup carries them, and the precision conversion rounds once.
    s = npf_setup(_npf(6; icelltype = ones(Int32, 6)), IcSpec(; strt), dis, c)
    @test s.r == r && s.Δr == Δr
    @test npf_setup(_npf(6), IcSpec(; strt), dis, c).r == strt
    for_each_precision() do p
        npf = NpfModel{p.T}(s)
        @test npf isa NpfModel{p.T}
        @test npf.Δr == p.T.(Δr) && npf.condsat == p.T.(s.condsat)
        @test NpfModel(p.T, s).condsat == npf.condsat
    end
end

@testset "NPF: ICELLTYPE" begin
    c = build_connectivity(_DIS_X2)
    p = npf_properties(_npf(2; icelltype = Int32[-1, 1]), _DIS_X2, c)
    @test p.icelltype == [1, 1]
    # condsat is the full-thickness conductance whatever ICELLTYPE is (no THICKSTRT).
    ic = IcSpec(; strt = [11.0, 11.0])
    @test npf_setup(_npf(2; icelltype = Int32[-1, 1]), ic, _DIS_X2, c).condsat == npf_setup(_npf(2), ic, _DIS_X2, c).condsat
end

@testset "NPF: input checks" begin
    c = build_connectivity(_DIS_X2)
    ic = IcSpec(; strt = [15.0, 15.0])
    setup(; kw...) = npf_setup(_npf(2; kw...), ic, _DIS_X2, c)

    # gwf-npf.f90 prepcheck: K, K33, K22 <= 0 by cell, with MF6's number format (' ',
    # 1PG15.6; checked against gfortran).
    e = _err(() -> setup(; k = [1.0, -1.0]))
    @test e isa MF6InputError && e.file == "test.npf"
    @test e.msg == "Hydraulic property K is <= 0 for cell (1,1,2)    -1.00000"
    e = _err(() -> setup(; k = [0.0, 2.5e-7]))
    @test e.msg == "Hydraulic property K is <= 0 for cell (1,1,1)     0.00000"
    e = _err(() -> setup(; k33 = [1.0, 0.0]))
    @test e.msg == "Hydraulic property K33 is <= 0 for cell (1,1,2)     0.00000"
    e = _err(() -> setup(; k = [2.0, 2.0], k22 = [-0.5, 1.0], k22overk = true))
    @test e.msg == "Hydraulic property K22 is <= 0 for cell (1,1,1)    -1.00000"
    e = _err(() -> setup(; k = [-1.0e-12, 1.0], k33 = [1.0, -3.0]))
    @test e.msg == "Hydraulic property K is <= 0 for cell (1,1,1)   -1.000000E-12\nHydraulic property K33 is <= 0 for cell (1,1,2)    -3.00000"
    # More than 20 errors: the first 20 and a count.
    dis30 = _npf_dis(; delr = ones(30), delc = [1.0], top = ones(30), botm = zeros(30, 1, 1))
    c30 = build_connectivity(dis30)
    e = _err(() -> npf_setup(_npf(30; k = zeros(30)), IcSpec(; strt = zeros(30)), dis30, c30))
    lines = split(e.msg, "\n")
    @test length(lines) == 21
    @test lines[20] == "Hydraulic property K is <= 0 for cell (1,1,20)     0.00000"
    @test lines[21] == "... 30 additional errors not shown for K"
    # Excluded cells are not checked.
    dis0 = _npf_dis(; delr = [10.0, 10.0], delc = [10.0], top = [100.0, 100.0], botm = reshape([90.0, 90.0], 2, 1, 1), idomain = reshape(Int32[0, 1], 2, 1, 1))
    @test npf_setup(_npf(2; k = [-1.0, 1.0]), IcSpec(; strt = [95.0, 95.0]), dis0, build_connectivity(dis0)).condsat == Float64[]

    # Angles.
    @test _err(() -> setup(; angle2 = [0.0, 0.0])).msg ==
        "ANGLE2 SPECIFIED BUT NOT ANGLE1. ANGLE2 REQUIRES ANGLE1.\nANGLE2 SPECIFIED BUT NOT ANGLE3. SPECIFY BOTH OR NEITHER ONE."
    @test _err(() -> setup(; angle1 = [0.0, 0.0], angle3 = [0.0, 0.0])).msg ==
        "ANGLE3 SPECIFIED BUT NOT ANGLE2. SPECIFY BOTH OR NEITHER ONE."
    # REWET needs WETDRY; K22OVERK needs K22.
    @test _err(() -> setup(; rewet = true)).msg == "Error in GRIDDATA block: WETDRY not found."
    @test _err(() -> setup(; k22overk = true)).msg == "K22OVERK option specified but K22 not specified."
    @test _err(() -> setup(; k33overk = true)).msg == "K33OVERK option specified but K33 not specified."

    # gwf-npf.f90 check_options.
    e = _err(() -> npf_setup(_npf(2; perched = true, variablecv = true, rewet = true, wetdry = zeros(2)), ic, _DIS_X2, c; newton = true))
    @test e isa MF6InputError
    @test e.msg == join(
        [
            "ERROR IN NPF OPTIONS. NEWTON OPTION CANNOT BE USED WITH PERCHED OPTION.",
            "ERROR IN NPF OPTIONS. NEWTON OPTION CANNOT BE USED WITH VARIABLECV OPTION.",
            "ERROR IN NPF OPTIONS. NEWTON OPTION CANNOT BE USED WITH REWET OPTION.",
        ], "\n",
    )
    e = _err(() -> setup(; xt3d = true, cell_averaging = :logarithmic, thickstrt = true, perched = true, variablecv = true))
    @test e isa MF6InputError
    @test length(split(e.msg, "\n")) == 4
    @test startswith(e.msg, "ERROR IN NPF OPTIONS. ALTERNATIVE_CELL_AVERAGING OPTION CANNOT BE USED WITH XT3D OPTION.")
    @test_logs (:warn, r"HIGHEST_CELL_SATURATION option cannot be used when NEWTON option in not specified") setup(; highest_cell_saturation = true)
    @test_logs setup(; perched = true, variablecv = true, dewatered = true)

    # THICKSTRT is supported since T6.1 (test/unit/convertible_test.jl). Not implemented yet:
    @test setup(; thickstrt = true) isa ModflowGPU.NpfSetup
    e = _err(() -> setup(; xt3d = true))
    @test e isa MF6UnsupportedError && occursin("XT3D", e.msg)
    @test_throws ArgumentError setup(; k22 = [1.0])
end
