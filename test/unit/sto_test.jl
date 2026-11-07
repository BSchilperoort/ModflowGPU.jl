# STO (confined) and IC (T2.5): storage capacity, period switching, the add, rate and
# flowja kernels on TEST_BACKEND in every precision against host values computed in
# Float64, the STO-SS budget, the checks for unsupported storage, and the initial heads
# as offsets.
using ModflowGPU: load_simulation, build_connectivity, reference_heads, StoSpec, StressPeriods, StoragePackage,
    storage_capacity, check_storage_spec, apply_period!, set_timestep!, add_to_system!, calc_storage_rates!,
    add_storage_flowja!, storage_budget, initial_heads, MF6UnsupportedError

function _sto_dev(x::AbstractArray)
    d = KernelAbstractions.allocate(TEST_BACKEND, eltype(x), size(x))
    copyto!(d, x)
    return d
end

# A 4-period simulation on the coupling-like grid (IDOMAIN 0 cells) with STO: period 1 is
# transient (no PERIOD block yet), 2 and 4 are steady state, 3 is transient.
function _write_sto_simulation(dir; storagecoefficient = false)
    dis = _synthetic_coupling_dis()
    strt = [dis.top[j, i] - 3.0 - 0.5 * k for j in 1:dis.ncol, i in 1:dis.nrow, k in 1:dis.nlay]
    write_dis_simulation(dir, dis; name = "sto", strt)
    write_input(
        dir, "sim.tdis", """
        BEGIN dimensions
          NPER  4
        END dimensions
        BEGIN perioddata
          1.0  1  1.0
          1.0  1  1.0
          10.0  2  1.0
          1.0  1  1.0
        END perioddata
        """,
    )
    write_input(
        dir, "sto.nam", """
        BEGIN packages
          DIS6  sto.dis  dis
          NPF6  sto.npf  npf
          IC6  sto.ic  ic
          STO6  sto.sto  sto
        END packages
        """,
    )
    write_input(
        dir, "sto.sto", """
        BEGIN options
          SAVE_FLOWS
          $(storagecoefficient ? "STORAGECOEFFICIENT" : "")
          SS_CONFINED_ONLY
        END options
        BEGIN griddata
          iconvert
            CONSTANT  0
          ss  LAYERED
            CONSTANT  1.0e-5
            CONSTANT  2.5e-6
        END griddata
        BEGIN period  2
          STEADY-STATE
        END period  2
        BEGIN period  3
          TRANSIENT
        END period  3
        BEGIN period  4
          STEADY-STATE
        END period  4
        """,
    )
    return dir
end

@testset "STO and IC" begin
    mktempdir() do dir
        for storagecoefficient in (false, true)
            m = only(load_simulation(_write_sto_simulation(joinpath(dir, "sc$storagecoefficient"); storagecoefficient)).models)
            conn = build_connectivity(m)
            sto = m.sto
            @test sto.storagecoefficient == storagecoefficient && sto.ss_confined_only
            @test conn.nodes < conn.nodesuser  # IDOMAIN 0 cells
            # SsCapacity in Float64, by reduced node.
            sc = storage_capacity(sto, conn)
            ssr = sto.ss[conn.nodeuser]
            thick = storagecoefficient ? ones(conn.nodes) : conn.top .- conn.bot
            @test sc == ssr .* thick .* conn.area
            r = reference_heads(conn, m.ic.strt)
            h0 = initial_heads(m.ic, conn, r)
            @test h0 == m.ic.strt[conn.nodeuser] .- r
            nodes = conn.nodes
            # IBOUND with a constant-head (−1) and an inactive (0) cell: no storage there.
            ibh = ones(Int32, nodes)
            ibh[2] = -1
            ibh[5] = 0
            hold64 = h0 .+ [0.01 * sin(n) for n in 1:nodes]
            hnew64 = hold64 .+ [0.25 * cos(3n) - 0.1 for n in 1:nodes]
            for_each_precision() do p
                T = p.T
                @testset "STORAGECOEFFICIENT = $storagecoefficient" begin
                    hd = initial_heads(TEST_BACKEND, T, m.ic, conn, r)
                    @test eltype(hd) == T && Array(hd) == T.(h0)
                    s = StoragePackage(TEST_BACKEND, T, sto, conn, 4)
                    @test eltype(s) == T
                    # Periods: transient (no block yet), steady, transient, steady.
                    @test apply_period!(s, 1) && s.transient
                    @test !apply_period!(s, 1)
                    @test apply_period!(s, 2) && !s.transient
                    @test apply_period!(s, 3) && s.transient
                    @test apply_period!(s, 4) && !s.transient
                    apply_period!(s, 3)
                    delt = 10.0 / 2
                    set_timestep!(s, delt)
                    rho64 = sc .* (1.0 / delt)
                    @test Array(s.rho) == T.(rho64)
                    rho = Float64.(T.(rho64))
                    ib = _sto_dev(ibh)
                    hold = _sto_dev(T.(hold64))
                    hnew = _sto_dev(T.(hnew64))
                    idiag = _sto_dev(Int32.(1:nodes))
                    d0 = [-1.0 - n for n in 1:nodes]
                    b0 = [0.5 * n for n in 1:nodes]
                    diag = _sto_dev(T.(d0))
                    b = _sto_dev(T.(b0))
                    add_to_system!(diag, b, idiag, s, hold, ib)
                    KernelAbstractions.synchronize(TEST_BACKEND)
                    act = ibh .> 0
                    hd_old = Float64.(T.(hold64))
                    dref = d0 .- act .* rho
                    bref = b0 .- act .* rho .* hd_old
                    tol = 4 * eps(T)
                    @test maximum(abs.(Float64.(Array(diag)) .- dref) ./ abs.(dref)) <= tol
                    @test maximum(abs.(Float64.(Array(b)) .- bref) ./ max.(abs.(bref), 1)) <= tol
                    @test Array(diag)[.!act] == T.(d0)[.!act] && Array(b)[.!act] == T.(b0)[.!act]
                    # Rates (STO-SS), flowja diagonal and the budget entry.
                    calc_storage_rates!(s, hnew, hold, ib)
                    KernelAbstractions.synchronize(TEST_BACKEND)
                    qref = act .* rho .* (hd_old .- Float64.(T.(hnew64)))
                    q = Float64.(Array(s.strgss))
                    @test maximum(abs.(q .- qref)) <= tol * maximum(abs, qref)
                    @test all(iszero, q[.!act])
                    flowja = KernelAbstractions.zeros(TEST_BACKEND, T, conn.nja)
                    iad = _sto_dev(Int32.(conn.ia[1:nodes]))
                    add_storage_flowja!(flowja, iad, s)
                    fj = Array(flowja)
                    @test fj[conn.ia[1:nodes]] == Array(s.strgss)
                    @test count(!iszero, fj) == count(!iszero, q)
                    bud = storage_budget(s)
                    @test bud.rin ≈ sum(max.(q, 0)) && bud.rout ≈ -sum(min.(q, 0))
                    @test bud.rin > 0 && bud.rout > 0
                    # Steady state: no terms, zero rates.
                    apply_period!(s, 4)
                    add_to_system!(diag, b, idiag, s, hold, ib)
                    KernelAbstractions.synchronize(TEST_BACKEND)
                    @test maximum(abs.(Float64.(Array(diag)) .- dref) ./ abs.(dref)) <= tol
                    calc_storage_rates!(s, hnew, hold, ib)
                    @test all(iszero, Array(s.strgss))
                    @test storage_budget(s) == (; rin = 0.0, rout = 0.0)
                    # A new delt rescales rho; zero delt is an error in a transient period.
                    set_timestep!(s, 2.0)
                    @test Array(s.rho) == T.(sc .* 0.5)
                    @test set_timestep!(s, 0.0) === s  # steady state: accepted
                    apply_period!(s, 1)
                    @test_throws ArgumentError set_timestep!(s, 0.0)
                    s2 = StoragePackage(TEST_BACKEND, T, sto, conn, 4)
                    apply_period!(s2, 1)
                    @test_throws ArgumentError add_to_system!(diag, b, idiag, s2, hold, ib)
                end
            end
        end
    end

    @testset "unsupported storage" begin
        mktempdir() do dir
            m = only(load_simulation(_write_sto_simulation(dir)).models)
            conn = build_connectivity(m)
            sto = m.sto
            iconv = copy(sto.iconvert)
            iconv[conn.nodeuser[3]] = 1
            conv(periods) = StoSpec(; iconvert = iconv, ss = sto.ss, sy = zeros(length(sto.ss)), periods, file = "x.sto")
            # Convertible cells with a transient period: T6.1.
            e = try
                check_storage_spec(conv(StressPeriods(Int[], Symbol[])), conn, 2)
                nothing
            catch err
                err
            end
            @test e isa MF6UnsupportedError && occursin("T6.1", sprint(showerror, e)) && occursin("ICONVERT", sprint(showerror, e))
            @test_throws MF6UnsupportedError StoragePackage(TEST_BACKEND, Float64, conv(StressPeriods([1, 2], [:steady_state, :transient])), conn, 2)
            # Steady state only: no storage terms, accepted.
            @test isnothing(check_storage_spec(conv(StressPeriods([1, 2], [:steady_state, :transient])), conn, 1))
            # A convertible flag in an IDOMAIN 0 cell does not count.
            iconv .= 0
            iconv[findfirst(<=(0), conn.nodereduced)] = 1
            @test isnothing(check_storage_spec(conv(StressPeriods(Int[], Symbol[])), conn, 2))
            # TVS6.
            tvs = StoSpec(; iconvert = sto.iconvert, ss = sto.ss, tvs_files = ["x.tvs"], file = "x.sto")
            @test_throws MF6UnsupportedError check_storage_spec(tvs, conn, 1)
        end
    end
end
