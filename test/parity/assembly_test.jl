# Assembly and matrix oracle (T2.6), PLAN.md section 4:
#
# 1. Universal residual test. On every Tier 0–2 example that loads with `strict = true`
#    and the confined Tier 3 examples ex-gwf-lgrv-gc and -gr (T6.0, `parity_examples`)
#    (MF6 6.8.2 tight-closure runs in reference/runs/<example>/tight; the others are
#    skipped with the reason) and on the synthetic models test/models/fhb_const and
#    test/models/coupling_rde (run with mf6 at test time), for every saved time step: MF6's
#    heads, converted to offsets in Float64 before the cast to `T`, go through the device
#    formulation (boundary formulate/add kernels, storage, `assemble!`) and the flux-form
#    residual kernel `flux_residual!`. Criterion: max |res| / max Σ|terms| < 1e-8, or in
#    `:single` at most the bound from rounding the offset heads to Float32. In every step
#    the assembled system is checked against the residual (`b' − A·h'` in Float64 from the
#    assembled `amat`, `rhs` equals the residual kernel within 1e-12 of the row magnitudes in
#    `:double`), and on the first step the kernel against the host loop of T2.3
#    (`host_flux_residual`).
# 2. Oracle: MF6's `SLN_1/AMAT` and `SLN_1/RHS` after the first `solve` call
#    (validation/libmf6_first_solve.jl, libmf6 6.8.2 in a subprocess; skipped without it)
#    against our `amat` and `b' = RHS − AMAT·r` (computed in Float64), assembled with the
#    starting heads. Criterion: `:double` < 1e-12, `:single` < 1e-6, relative to max |AMAT|
#    for the matrix and to max_n (|RHS_n| + Σ_j |a_nj·r_j|) for `b'`.
# 3. Cancellation test: a column of 100 cells with heads near 5000 m and 1 cm differences.
#    The `:single` flux-form residual must match the `:double` one within 1e-4 relative
#    (to max |res|); `b − A·h` in Float32 with absolute heads (MF6's form) fails it, as
#    does the flux form without reference offsets.
using ModflowGPU: load_simulation, build_connectivity, npf_setup, BoundaryPackage, apply_period!, formulate!,
    apply_chd_heads!, boundary_ibound, StoragePackage, set_timestep!, initial_heads, to_offsets, open_head_file,
    read_head, CsrStructure, LinearSystem, NpfModel, assemble!, flux_residual!, reset_terms!, add_terms!,
    MF6UnsupportedError, backend_name

const ASM_MF6 = get(ENV, "MF6", joinpath(REFERENCE_DIR, "mf6.8.2_linux", "bin", "mf6"))

# Time step lengths of period `kper` (tdis.f90 `tdis_set_timestep`).
function _step_lengths(tdis, kper)
    perlen, nstp, mult = tdis.perlen[kper], tdis.nstp[kper], tdis.tsmult[kper]
    dt = mult == 1.0 ? perlen / nstp : perlen * (1.0 - mult) / (1.0 - mult^nstp)
    return [dt * mult^(k - 1) for k in 1:nstp]
end

_symmetric(sim) = only(only(sim.solution_groups).solutions).linear_acceleration === :cg

# The device formulation of one model in precision `T`, set up for period `kper` and
# step length `delt`, with offset heads `h` (host, Float64 offsets) and old heads `hold`.
# Returns the device objects; the constant heads are applied to `h`.
struct _AsmCase{T, B, C, S, N, P, ST, V, VI}
    backend::B
    csr::C
    sys::S
    npf::N
    pk::P
    sto::ST
    area::V
    nodes::Int
    ib::VI
end

_todev(backend, x) = (d = KernelAbstractions.allocate(backend, eltype(x), size(x)); copyto!(d, x); d)

function _asm_case(T, m, conn, s, nper; backend = TEST_BACKEND)
    csr = CsrStructure(backend, conn)
    sys = LinearSystem(backend, T, csr)
    npf = NpfModel(backend, T, s)
    pk = [BoundaryPackage(backend, T, b, conn, m.dis) for b in m.boundaries]
    sto = isnothing(m.sto) ? nothing : StoragePackage(backend, T, m.sto, conn, nper)
    area = _todev(backend, T.(conn.area))
    ib = _todev(backend, ones(Int32, conn.nodes))
    return _AsmCase{T, typeof(backend), typeof(csr), typeof(sys), typeof(npf), typeof(pk), typeof(sto), typeof(area), typeof(ib)}(
        backend, csr, sys, npf, pk, sto, area, Int(conn.nodes), ib,
    )
end

# Formulate period `kper`, step length `delt`; returns (h, ibound host) with the device h.
function _formulate!(c::_AsmCase{T}, conn, s, kper, delt, hoff, holdoff) where {T}
    foreach(x -> apply_period!(x, kper, s.r), c.pk)
    if !isnothing(c.sto)
        apply_period!(c.sto, kper)
        set_timestep!(c.sto, delt)
    end
    ibh = boundary_ibound(conn, c.pk)
    copyto!(c.ib, ibh)
    h = _todev(c.backend, T.(hoff))
    hold = _todev(c.backend, T.(holdoff))
    reset_terms!(c.sys)
    for x in c.pk
        if ModflowGPU.boundary_kind(x) === :CHD
            apply_chd_heads!(h, x)
            apply_chd_heads!(hold, x)
        else
            formulate!(x, h, c.ib, c.area)
            add_terms!(c.sys, c.csr, x)
        end
    end
    isnothing(c.sto) || add_terms!(c.sys, c.csr, c.sto, hold, c.ib)
    return h, ibh, hold
end

# max_n |(b' − A·h')_n − res_n| / max_n bound_n, in Float64 from the assembled system.
function _assembly_consistency(conn, amat, rhs, h, res, bound, ibh)
    err = 0.0
    for n in 1:conn.nodes
        ibh[n] > 0 || continue
        r = Float64(rhs[n])
        for ii in conn.ia[n]:(conn.ia[n + 1] - 1)
            r -= Float64(amat[ii]) * Float64(h[conn.ja[ii]])
        end
        err = max(err, abs(r - Float64(res[n])))
    end
    return err / maximum(Float64, bound)
end

# Universal residual test on one model with MF6's heads in `rundir`.
function _universal_residual(name, rundir, m, sim)
    conn = build_connectivity(m)
    s = npf_setup(m, conn)
    hds = only(filter(f -> endswith(f, ".hds"), readdir(rundir; join = true)))
    hf = open_head_file(hds)
    nstep = length(hf)
    habs = [vec(read_head(hf, k))[conn.nodeuser] for k in 1:nstep]
    hoff = [to_offsets(s.r, x) for x in habs]
    h0 = initial_heads(m.ic, conn, s.r)
    symmetric = _symmetric(sim)
    for_each_precision() do p
        T = p.T
        c = _asm_case(T, m, conn, s, length(sim.tdis.perlen))
        res, scale, bound = (KernelAbstractions.zeros(TEST_BACKEND, T, conn.nodes) for _ in 1:3)
        worst, worst_bound, worst_cons, fails = 0.0, 0.0, 0.0, 0
        # The fused `formulate_system!` (T4.5) against the separate kernels, bitwise, with
        # the off-diagonals rewritten only when IBOUND changes (as `formulate!(::GwfModel)`).
        sys2 = LinearSystem(TEST_BACKEND, T, c.csr)
        res2, scale2, bound2 = (KernelAbstractions.zeros(TEST_BACKEND, T, conn.nodes) for _ in 1:3)
        ibprev, fused_ok = nothing, true
        for k in 1:nstep
            st = hf.steps[k]
            delt = _step_lengths(sim.tdis, st.kper)[st.kstp]
            prev = k == 1 ? h0 : hoff[k - 1]
            h, ibh, hold = _formulate!(c, conn, s, Int(st.kper), delt, hoff[k], prev)
            flux_residual!(res, c.sys, c.csr, c.npf, h, c.ib; scale, bound)
            assemble!(c.sys, c.csr, c.npf, h, c.ib; symmetric)
            pkgs = Tuple((x.hcof, x.rhs, x.nodeptr, x.noderec) for x in c.pk if ModflowGPU.boundary_kind(x) !== :CHD && x.nbound > 0)
            sto = !isnothing(c.sto) && c.sto.transient ? (c.sto.rho, hold) : nothing
            ModflowGPU.formulate_system!(sys2, res2, c.csr, c.npf, h, c.ib, pkgs, sto; symmetric, offdiag = ibh != ibprev, scale = scale2, bound = bound2)
            ibprev = ibh
            KernelAbstractions.synchronize(TEST_BACKEND)
            fused_ok &= Array(sys2.amat) == Array(c.sys.amat) && Array(sys2.rhs) == Array(c.sys.rhs) &&
                Array(res2) == Array(res) && Array(scale2) == Array(scale) && Array(bound2) == Array(bound)
            rel, roundoff = residual_stats(T, res, scale, bound)
            ok = rel < 1.0e-8 || (T === Float32 && rel <= roundoff)
            ok || (fails += 1)
            worst, worst_bound = max(worst, rel), max(worst_bound, roundoff)
            hh = Array(h)
            if T === Float64
                cons = _assembly_consistency(conn, Array(c.sys.amat), Array(c.sys.rhs), hh, Array(res), Array(bound), ibh)
                worst_cons = max(worst_cons, cons)
            end
            if k == 1
                # The kernel against the host loop of T2.3 (same formula, plain summation).
                hrel, hround = host_flux_residual(
                    T, conn, T.(s.condsat), T.(s.Δr), hh, Array(c.sys.bdiag), Array(c.sys.brhs), ibh,
                )
                @test hround ≈ roundoff rtol = 1.0e-3
                @test abs(hrel - rel) <= 4 * roundoff
            end
        end
        @test fails == 0
        @test fused_ok
        T === Float64 && @test worst_cons < 1.0e-12
        @info "Universal residual test: $name, :$(p.preset), $(conn.nodes) nodes, $nstep step(s): max |res| / max Σ|terms| = $(round(worst; sigdigits = 3)) (rounding bound $(round(worst_bound; sigdigits = 3)))" *
            (T === Float64 ? ", assembly vs residual $(round(worst_cons; sigdigits = 3))" : "")
    end
    return nothing
end

# Oracle: libmf6's first-solve system vs ours.
function _libmf6_first_solve(simdir)
    return mktempdir() do tmp
        outdir = joinpath(tmp, "out")
        run_libmf6_script("libmf6_first_solve.jl", simdir, outdir)
        index = TOML.parsefile(joinpath(outdir, "index.toml"))
        rd(a, T) = (v = Vector{T}(undef, index[a]["length"]); read!(joinpath(outdir, "$a.bin"), v); v)
        x0 = Vector{Float64}(undef, index["X"]["length"])
        read!(joinpath(outdir, "X0.bin"), x0)
        (; ia = rd("IA", Int32), ja = rd("JA", Int32), amat = rd("AMAT", Float64), rhs = rd("RHS", Float64), x0, delt = index["delt"])
    end
end

function _oracle(name, simdir, m, sim)
    conn = build_connectivity(m)
    s = npf_setup(m, conn)
    o = _libmf6_first_solve(simdir)
    @test o.ia == conn.ia && o.ja == conn.ja
    # b' = RHS − AMAT·r in Float64, and the scale of the terms it subtracts.
    bmf = copy(o.rhs)
    bscale = abs.(o.rhs)
    for n in 1:conn.nodes, ii in conn.ia[n]:(conn.ia[n + 1] - 1)
        t = o.amat[ii] * s.r[conn.ja[ii]]
        bmf[n] -= t
        bscale[n] += abs(t)
    end
    h0 = initial_heads(m.ic, conn, s.r)
    symmetric = _symmetric(sim)
    for_each_precision() do p
        T = p.T
        c = _asm_case(T, m, conn, s, length(sim.tdis.perlen))
        h, ibh = _formulate!(c, conn, s, 1, o.delt, h0, h0)
        assemble!(c.sys, c.csr, c.npf, h, c.ib; symmetric)
        KernelAbstractions.synchronize(TEST_BACKEND)
        # The starting heads (with the constant heads) are MF6's.
        @test maximum(abs.(s.r .+ Float64.(Array(h)) .- o.x0)) <= 4 * eps(T) * maximum(abs, o.x0)
        ea = maximum(abs.(Float64.(Array(c.sys.amat)) .- o.amat)) / maximum(abs, o.amat)
        eb = maximum(abs.(Float64.(Array(c.sys.rhs)) .- bmf)) / maximum(bscale)
        tol = T === Float64 ? 1.0e-12 : 1.0e-6
        @test ea < tol
        @test eb < tol
        if !(TEST_BACKEND isa CPU)
            # The same formulation on the CPU backend (PLAN.md section 4, backends).
            cc = _asm_case(T, m, conn, s, length(sim.tdis.perlen); backend = CPU())
            hc, _ = _formulate!(cc, conn, s, 1, o.delt, h0, h0)
            assemble!(cc.sys, cc.csr, cc.npf, hc, cc.ib; symmetric)
            rg, rc, sg = (zeros(T, conn.nodes) for _ in 1:3)
            resg = KernelAbstractions.zeros(TEST_BACKEND, T, conn.nodes)
            scg = KernelAbstractions.zeros(TEST_BACKEND, T, conn.nodes)
            flux_residual!(resg, c.sys, c.csr, c.npf, h, c.ib; scale = scg)
            flux_residual!(rc, cc.sys, cc.csr, cc.npf, hc, cc.ib)
            KernelAbstractions.synchronize(TEST_BACKEND)
            rel(a, b) = maximum(abs.(Float64.(a) .- Float64.(b))) / max(maximum(abs, Float64.(b)), floatmin())
            dg = (rel(Array(c.sys.amat), cc.sys.amat), rel(Array(c.sys.rhs), cc.sys.rhs), maximum(abs.(Float64.(Array(resg)) .- rc)) / maximum(Array(scg)))
            @test all(<(T === Float64 ? 1.0e-10 : 1.0e-5), dg)
            @info "Assembly $(backend_name(TEST_BACKEND)) vs CPU: $name, :$(p.preset): amat $(dg[1]), rhs $(dg[2]), residual / max Σ|terms| $(dg[3])"
        end
        @info "Assembly oracle vs libmf6 (first solve): $name, :$(p.preset), symmetric = $symmetric, $(count(<(0), ibh)) CHD rows: amat max |Δ| / max |a| = $(round(ea; sigdigits = 3)), b' max |Δ| / max(|RHS| + Σ|a·r|) = $(round(eb; sigdigits = 3))"
    end
    return nothing
end

# The models of parts 1 and 2: (name, directory with MF6 heads, input directory for libmf6).
function _assembly_models(f)
    for (name, _) in parity_examples()
        rundir = joinpath(REFERENCE_DIR, "runs", name, "tight")
        if !isdir(rundir)
            @info "Skipping $name in the assembly test: $rundir not found"
            continue
        end
        sim = try
            load_simulation(rundir; strict = true)
        catch err
            err isa MF6UnsupportedError || rethrow()
            @info "Skipping $name in the assembly test (not loadable): $(sprint(showerror, err))"
            continue
        end
        f(name, rundir, sim, rundir)
    end
    if !isfile(ASM_MF6)
        @info "Skipping the synthetic models in the assembly test: mf6 not found at $ASM_MF6 (set REFERENCE_DIR or MF6)"
        return nothing
    end
    for (name, dir) in (("fhb_const", joinpath(@__DIR__, "..", "models", "fhb_const")), ("coupling_rde", joinpath(@__DIR__, "..", "models", "coupling_rde")))
        mktempdir() do tmp
            rundir = joinpath(tmp, name)
            cp(dir, rundir)
            out = read(Cmd(`$ASM_MF6`; dir = rundir), String)
            @test occursin("Normal termination", out)
            # libmf6 gets a clean copy of the inputs.
            f(name, rundir, load_simulation(rundir; strict = true), dir)
        end
    end
    return nothing
end

@testset "Assembly: universal residual test and libmf6 oracle" begin
    have_libmf6 = libmf6_available()
    have_libmf6 || @info "Skipping the assembly oracle vs libmf6: libmf6 environment not found at $LIBMF6_ENV"
    nmodels = Ref(0)
    _assembly_models() do name, rundir, sim, inputdir
        m = only(sim.models)
        nmodels[] += 1
        @testset "$name" begin
            _universal_residual(name, rundir, m, sim)
            have_libmf6 && _oracle(name, inputdir, m, sim)
        end
    end
    @info "Assembly test: $(nmodels[]) model(s)"
end

@testset "Assembly: cancellation test (heads near 5000 m, 1 cm differences)" begin
    n = 100
    dis = ModflowGPU.DisSpec(;
        nlay = Int32(1), nrow = Int32(n), ncol = Int32(1), delr = [100.0], delc = fill(100.0, n),
        top = fill(5100.0, 1, n), botm = fill(4900.0, 1, n, 1), idomain = ones(Int32, 1, n, 1),
    )
    mktempdir() do dir
        write_dis_simulation(dir, dis; strt = 5000.0, npf_arrays = Dict{String, Any}("icelltype" => 0, "k" => 3.0))
        m = only(load_simulation(dir).models)
        conn = build_connectivity(m)
        s = npf_setup(m, conn)
        @test all(==(5000.0), s.r)
        # Heads 5000 m + 1 cm steps with a non-linear 0–1 cm pattern; a GHB-like term on
        # the first cell (conductance 50, head 5000.005 m).
        habs = [5000.0 + 0.01 * k + 0.01 * ((7919 * k) % 13) / 13 for k in 1:n]
        cg, hg = 50.0, 5000.005
        function residual(T, r)
            c = CsrStructure(TEST_BACKEND, conn)
            sys = LinearSystem(TEST_BACKEND, T, c)
            npf = NpfModel(to_test_device(T.(s.condsat)), to_test_device(T.(ModflowGPU.reference_differences(conn, r))))
            bd, br = zeros(T, n), zeros(T, n)
            bd[1], br[1] = T(-cg), T(-cg * (hg - r[1]))
            copyto!(sys.bdiag, bd)
            copyto!(sys.brhs, br)
            h = to_test_device(T.(to_offsets(r, habs)))
            ib = to_test_device(ones(Int32, n))
            res = KernelAbstractions.zeros(TEST_BACKEND, T, n)
            flux_residual!(res, sys, c, npf, h, ib)
            assemble!(sys, c, npf, h, ib)
            KernelAbstractions.synchronize(TEST_BACKEND)
            return Float64.(Array(res)), Array(sys.amat), Array(sys.rhs), Array(h)
        end
        rd, = residual(Float64, s.r)
        scale = maximum(abs, rd)
        @test scale > 0.01
        for_each_precision() do p
            T = p.T
            rs, = residual(T, s.r)
            err = maximum(abs.(rs .- rd)) / scale
            @test err < 1.0e-4
            msg = "Cancellation test, :$(p.preset): flux form with offsets, max |res − res_double| / max |res_double| = $(round(err; sigdigits = 3))"
            if T === Float32
                # Without offsets (r = 0, absolute heads in Float32): the flux form and
                # MF6's b − A·h (in Float32) from the assembled absolute system.
                r0 = zeros(n)
                ra, amat, b, h = residual(T, r0)
                err_abs = maximum(abs.(ra .- rd)) / scale
                bah = zeros(T, n)
                for i in 1:n
                    acc = b[i]
                    for ii in conn.ia[i]:(conn.ia[i + 1] - 1)
                        acc -= amat[ii] * h[conn.ja[ii]]
                    end
                    bah[i] = acc
                end
                err_bah = maximum(abs.(Float64.(bah) .- rd)) / scale
                @test err_bah > 1.0e-4
                @test err_abs > 1.0e-4
                msg *= "; without offsets: flux form $(round(err_abs; sigdigits = 3)), b − A·h $(round(err_bah; sigdigits = 3)) (both fail 1e-4)"
            end
            @info msg
        end
    end
end
