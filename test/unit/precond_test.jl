# Preconditioners (T4.2): colouring, the application of Chebyshev, multicolour SGS and
# multicolour ILU0 against host Float64 references, symmetry and positivity for CG, and CG /
# BiCGSTAB solves with each of them vs `\`, on TEST_BACKEND in every precision. Uses the
# matrix helpers of krylov_test.jl (same test group).
const _adapt = ModflowGPU.adapt
using ModflowGPU: greedy_colors, is_valid_coloring, dis_parity_colors, grid_colors, precond_apply!, ChebyshevPreconditioner,
    MulticolorPreconditioner, JacobiPreconditioner, PRECONDITIONERS

const _NEW_PRECONDITIONERS = (:chebyshev, :mc_sgs, :mc_ilu0)

# Host references on the operator `A` (Float64, dense) for `z = M⁻¹·r`.
function _host_sgs(A, colors, r)
    n = size(A, 1)
    D = LinearAlgebra.Diagonal([A[i, i] for i in 1:n])
    L = [colors[j] < colors[i] ? A[i, j] : 0.0 for i in 1:n, j in 1:n]
    U = [colors[j] > colors[i] ? A[i, j] : 0.0 for i in 1:n, j in 1:n]
    return ((D + U) \ (D * ((D + L) \ r)))
end

function _host_ilu0(A, colors, r)
    n = size(A, 1)
    o = sortperm(colors; alg = Base.Sort.DEFAULT_STABLE)
    B = A[o, o]
    pat = B .!= 0
    for i in 1:n  # the structural pattern includes the diagonal
        pat[i, i] = true
    end
    F = copy(B)
    for i in 2:n, k in 1:(i - 1)
        pat[i, k] || continue
        F[i, k] /= F[k, k]
        for j in (k + 1):n
            pat[i, j] && pat[k, j] && (F[i, j] -= F[i, k] * F[k, j])
        end
    end
    Lf = LinearAlgebra.UnitLowerTriangular(F)
    Uf = LinearAlgebra.UpperTriangular(F)
    z = zeros(n)
    z[o] = Uf \ (Lf \ r[o])
    return z
end

function _host_chebyshev(A, bounds, degree, r)
    a, b = bounds
    dinv = 1 ./ [A[i, i] for i in 1:size(A, 1)]
    θ, δ = (b + a) / 2, (b - a) / 2
    σ = θ / δ
    ρ = 1 / σ
    d = dinv .* r ./ θ
    x = copy(d)
    for _ in 2:degree
        ρn = 1 / (2σ - ρ)
        d = ρn * ρ .* d .+ (2ρn / δ) .* (dinv .* (r .- A * x))
        x .+= d
        ρ = ρn
    end
    return x
end

# A solver with preconditioner `kind`, set up on `a` (one solve! with maxiter 1).
function _setup_solver(p, csr, a, kind; method = :bicgstab, scaling = :none)
    st = KrylovSettings(; method, maxiter = 1, preconditioner = kind, scaling)
    solver = KrylovSolver(TEST_BACKEND, csr, st; Ts = p.Ts, Tp = p.Tp)
    n = Int(csr.nodes)
    solve!(to_test_device(zeros(p.T, n)), solver, to_test_device(p.T.(a)), to_test_device(ones(p.T, n)))
    return solver
end

function _apply(solver, p, r)
    z = to_test_device(zeros(p.Ts, length(r)))
    precond_apply!(z, solver.precond, to_test_device(p.Ts.(r)))
    return Float64.(Array(z))
end

@testset "preconditioners" begin
    @testset "colouring" begin
        rng = MersenneTwister(11)
        ia, ja, _ = _csr_from_neighbours(_random_graph(rng, 400, 6))
        c = greedy_colors(ia, ja)
        @test is_valid_coloring(ia, ja, c) && minimum(c) == 1
        @test sort(unique(c)) == 1:maximum(c)
        bad = copy(c)
        bad[ja[ia[1] + 1]] = bad[1]
        @test !is_valid_coloring(ia, ja, bad)
        # DIS: red–black by (i + j + k) parity
        dis = _synthetic_coupling_dis()
        conn = build_connectivity(dis)
        rb = dis_parity_colors(conn, dis)
        @test is_valid_coloring(conn.ia, conn.ja, rb) && sort(unique(rb)) == [1, 2]
        for n in 1:conn.nodes
            u = conn.nodeuser[n] - 1
            j, i, k = u % dis.ncol, (u ÷ dis.ncol) % dis.nrow, u ÷ (dis.ncol * dis.nrow)
            @test rb[n] == 1 + (i + j + k) % 2
        end
        @test grid_colors(conn, dis) == rb
        # Pass-through cells connect cells two layers apart: parity is not a colouring there.
        pdis = _synthetic_passthrough_dis()
        pconn = build_connectivity(pdis)
        @test !is_valid_coloring(pconn.ia, pconn.ja, dis_parity_colors(pconn, pdis))
        pc = grid_colors(pconn, pdis)
        @test is_valid_coloring(pconn.ia, pconn.ja, pc) && pc == greedy_colors(pconn.ia, pconn.ja)
        @test_throws ArgumentError KrylovSettings(; preconditioner = :ilut)
        @test KrylovSettings().preconditioner === :jacobi
        @test PRECONDITIONERS == (:jacobi, :chebyshev, :mc_sgs, :mc_ilu0, :amg)
    end

    rng = MersenneTwister(12)
    n = 300
    ia, ja, isym = _csr_from_neighbours(_random_graph(rng, n, 5))
    csr = _device_csr(ia, ja, isym)
    colors = greedy_colors(ia, ja)

    @testset "application vs host references" begin
        anon = _mmatrix(MersenneTwister(13), ia, ja, isym; symmetric = false, negate = true, shift = 0.01)
        A = _dense(ia, ja, anon)
        r = randn(rng, n)
        for_each_precision(; presets = SOLVER_TEST_PRESETS) do p
            tol = p.Tp == Float64 ? 1.0e-11 : 2.0e-4
            for kind in _NEW_PRECONDITIONERS
                s = _setup_solver(p, csr, anon, kind)  # BiCGSTAB, no scaling: the operator is A
                z = _apply(s, p, r)
                ref = kind === :mc_sgs ? _host_sgs(A, colors, r) :
                    kind === :mc_ilu0 ? _host_ilu0(A, colors, r) :
                    _host_chebyshev(A, s.precond.bounds, s.precond.degree, r)
                @test _relerr(z, ref) < tol
            end
            # Red–black colours of a DIS grid (the two-kernel ILU0 path).
            dis = _synthetic_coupling_dis()
            conn = build_connectivity(dis)
            gc = grid_colors(conn, dis)
            @test maximum(gc) == 2
            ga = _mmatrix(MersenneTwister(17), Int.(conn.ia), Int.(conn.ja), Int.(conn.isym); symmetric = false, negate = true, shift = 0.01)
            GA = _dense(Int.(conn.ia), Int.(conn.ja), ga)
            gr = randn(MersenneTwister(18), Int(conn.nodes))
            for kind in (:mc_sgs, :mc_ilu0)
                st = KrylovSettings(; method = :bicgstab, maxiter = 1, preconditioner = kind)
                gs = KrylovSolver(TEST_BACKEND, CsrStructure(TEST_BACKEND, conn), st; Ts = p.Ts, Tp = p.Tp, colors = gc)
                nn = Int(conn.nodes)
                solve!(to_test_device(zeros(p.T, nn)), gs, to_test_device(p.T.(ga)), to_test_device(ones(p.T, nn)))
                ref = kind === :mc_sgs ? _host_sgs(GA, gc, gr) : _host_ilu0(GA, gc, gr)
                @test _relerr(_apply(gs, p, gr), ref) < tol
            end
            # Jacobi through the generic interface equals the fused path's minv·r.
            s = _setup_solver(p, csr, anon, :jacobi)
            @test s.precond isa JacobiPreconditioner && s.precond.minv === s.minv
            @test _relerr(_apply(s, p, r), r ./ [A[i, i] for i in 1:n]) < (p.Tp == Float64 ? 1.0e-14 : 1.0e-6)
        end
    end

    @testset "SPD for CG" begin
        asym = _mmatrix(MersenneTwister(14), ia, ja, isym; symmetric = true, negate = true, shift = 0.01)
        A = -_dense(ia, ja, asym)  # CG iterates on S·A = −A
        u, v = randn(rng, n), randn(rng, n)
        λ = LinearAlgebra.eigvals(LinearAlgebra.Symmetric(LinearAlgebra.Diagonal(1 ./ sqrt.([A[i, i] for i in 1:n])) * A * LinearAlgebra.Diagonal(1 ./ sqrt.([A[i, i] for i in 1:n]))))
        for_each_precision(; presets = SOLVER_TEST_PRESETS) do p
            tol = p.Tp == Float64 ? 1.0e-11 : 1.0e-4
            for kind in _NEW_PRECONDITIONERS
                s = _setup_solver(p, csr, asym, kind; method = :cg)
                mu, mv = _apply(s, p, u), _apply(s, p, v)
                @test abs(dot(v, mu) - dot(u, mv)) <= tol * norm(u) * norm(mv)
                @test dot(u, mu) > 0 && dot(v, mv) > 0
                if kind === :chebyshev
                    # The interval's upper end bounds the spectrum of D⁻¹·Â (SPD polynomial).
                    a, b = s.precond.bounds
                    @test b >= maximum(λ) * (1 - 1.0e-6) && a ≈ b / s.precond.ratio
                end
            end
        end
    end

    @testset "solves vs \\ ($kind)" for kind in _NEW_PRECONDITIONERS
        rowscale = exp.(randn(MersenneTwister(15), n))
        cases = [
            ("SPD, MF6 sign, CG", :cg, (; symmetric = true, negate = true), :none, 0),
            ("SPD badly scaled, CG, DIAGONAL", :cg, (; symmetric = true, negate = true, rowscale), :diagonal, 0),
            ("SPD, CG, NUMBER_ORTHOGONALIZATIONS 3", :cg, (; symmetric = true, negate = true), :none, 3),
            ("nonsymmetric, MF6 sign, BiCGSTAB", :bicgstab, (; symmetric = false, negate = true), :none, 0),
            ("nonsymmetric badly scaled, BiCGSTAB, DIAGONAL", :bicgstab, (; symmetric = false, negate = true, rowscale), :diagonal, 0),
            ("nonsymmetric, BiCGSTAB, NUMBER_ORTHOGONALIZATIONS 3", :bicgstab, (; symmetric = false, negate = true), :none, 3),
        ]
        for (name, method, mkw, scaling, north) in cases
            @testset "$name" begin
                a = _mmatrix(MersenneTwister(hash(name)), ia, ja, isym; shift = 0.01, mkw...)
                A = _dense(ia, ja, a)
                b = A * randn(rng, n)
                xs = A \ b
                w = haskey(mkw, :rowscale) ? sqrt.(abs.([A[i, i] for i in 1:n])) : 1
                for_each_precision(; presets = SOLVER_TEST_PRESETS) do p
                    kw = (; method, maxiter = 2000, dvclose = 0.0, rclose_option = :relative, scaling, north, rclose = p.Ts == Float64 ? 1.0e-11 : 1.0e-6)
                    x, r = _device_solve(p, csr, a, b, KrylovSettings(; kw..., preconditioner = kind))
                    _, rj = _device_solve(p, csr, a, b, KrylovSettings(; kw...))
                    @test r.status === :criterion || p.Ts == Float32
                    @test _relerr(x, xs; w) < (p.Ts == Float64 ? 1.0e-9 : 1.0e-4)
                    kind === :mc_ilu0 && @test r.iterations < rj.iterations
                    kind === :mc_sgs && @test r.iterations <= rj.iterations
                end
            end
        end
    end

    @testset "grid Connectivity with constant heads" begin
        conn = build_connectivity(_synthetic_coupling_dis())
        gcsr = CsrStructure(TEST_BACKEND, conn)
        gia, gja, gisym = Int.(conn.ia), Int.(conn.ja), Int.(conn.isym)
        nn = Int(conn.nodes)
        chd = [1, 7, nn ÷ 2, nn]
        for (method, symmetric) in ((:cg, true), (:bicgstab, false))
            a = _mmatrix(MersenneTwister(16), gia, gja, gisym; symmetric, negate = true, chd, shift = 1.0e-3)
            A = _dense(gia, gja, a)
            b = A * randn(rng, nn)
            b[chd] .= 0
            xs = A \ b
            colors = grid_colors(conn, _synthetic_coupling_dis())
            for_each_precision(; presets = SOLVER_TEST_PRESETS) do p
                for kind in _NEW_PRECONDITIONERS
                    st = _tight(p.Ts, method; rclose = (p.Ts == Float64 ? 1.0e-13 : 1.0e-6) * maximum(abs, b), preconditioner = kind)
                    solver = KrylovSolver(TEST_BACKEND, gcsr, st; Ts = p.Ts, Tp = p.Tp, colors)
                    δ = to_test_device(zeros(p.T, nn))
                    r = solve!(δ, solver, to_test_device(p.T.(a)), to_test_device(p.T.(b)))
                    x = Float64.(Array(δ))
                    @test r.converged
                    @test _relerr(x, xs) < (p.Ts == Float64 ? 1.0e-10 : 1.0e-4)
                    @test all(==(0), x[chd])
                    # Adapt (e.g. to the CPU backend) keeps the preconditioner working.
                    cpu = _adapt(CPU(), solver)
                    @test typeof(cpu.precond).name === typeof(solver.precond).name
                    δc = zeros(p.T, nn)
                    rc = solve!(δc, cpu, p.T.(a), p.T.(b))
                    @test _relerr(Float64.(δc), x) < (p.Ts == Float64 ? 1.0e-10 : 1.0e-4)
                    p.Ts == Float64 && @test rc.iterations == r.iterations
                end
            end
        end
    end

    @testset "incremental red–black ILU0 setup (T4.7)" begin
        conn = build_connectivity(_synthetic_coupling_dis())
        gcsr = CsrStructure(TEST_BACKEND, conn)
        gia, gja, gisym = Int.(conn.ia), Int.(conn.ja), Int.(conn.isym)
        nn = Int(conn.nodes)
        colors = grid_colors(conn, _synthetic_coupling_dis())
        a1 = _mmatrix(MersenneTwister(19), gia, gja, gisym; symmetric = true, negate = true, chd = [1, nn], shift = 1.0e-3)
        # Diagonal changes: a few first- and second-colour rows, one sign flip (its row
        # factor changes for CG), one change below Float32 resolution.
        a2 = copy(a1)
        rows = [3, 4, nn ÷ 3, nn ÷ 2 + 1]
        for r in rows
            a2[gia[r]] *= 1.5
        end
        a2[gia[nn ÷ 4]] = -a2[gia[nn ÷ 4]]
        a2[gia[nn ÷ 5]] *= 1 + 1.0e-12
        diagof(a) = [a[gia[r]] for r in 1:nn]
        for_each_precision(; presets = SOLVER_TEST_PRESETS) do p
            for (method, relax) in ((:cg, 0.0), (:bicgstab, 0.0), (:cg, 0.97))
                st = KrylovSettings(; method, maxiter = 3, preconditioner = :mc_ilu0, precond_options = (; relax))
                inc = KrylovSolver(TEST_BACKEND, gcsr, st; Ts = p.Ts, Tp = p.Tp, colors)
                full = KrylovSolver(TEST_BACKEND, gcsr, st; Ts = p.Ts, Tp = p.Tp, colors)
                δ = to_test_device(zeros(p.T, nn))
                b = to_test_device(ones(p.T, nn))
                da1, da2 = to_test_device(p.T.(a1)), to_test_device(p.T.(a2))
                solve!(δ, inc, da1, b; diag = to_test_device(p.T.(diagof(a1))), offdiag_id = 1_000_001)
                @test inc.precond.key[] == 1_000_001
                r1 = solve!(δ, inc, da2, b; diag = to_test_device(p.T.(diagof(a2))), offdiag_id = 1_000_001)
                x1 = Array(δ)
                r2 = solve!(δ, full, da2, b)
                @test full.precond.key[] == 0
                @test Array(inc.precond.vals) == Array(full.precond.vals)
                @test Array(inc.precond.dinv) == Array(full.precond.dinv)
                @test x1 == Array(δ) && r1.iterations == r2.iterations
                # only the changed rows and their neighbours were flagged
                @test count(!=(0), Array(inc.precond.flag)) == length(rows) + 2 - (p.T == Float32 && Float32(a2[gia[nn ÷ 5]]) == Float32(a1[gia[nn ÷ 5]]))
                # another id: a full setup
                solve!(δ, inc, da1, b; diag = to_test_device(p.T.(diagof(a1))), offdiag_id = 1_000_002)
                solve!(δ, full, da1, b)
                @test Array(inc.precond.vals) == Array(full.precond.vals)
            end
        end
    end

    @testset "run: coupling_rde with each preconditioner (and AMG)" begin
        dir = joinpath(@__DIR__, "..", "models", "coupling_rde")
        for_each_precision(; presets = SOLVER_TEST_PRESETS) do p
            mktempdir() do tmp
                ref = ModflowGPU.run(dir; backend = TEST_BACKEND, precision = p.preset, outdir = tmp, write_output = false, preconditioner = :jacobi)
                @test ref.converged
                # the default (:auto) is mc_ilu0
                ra = ModflowGPU.run(dir; backend = TEST_BACKEND, precision = p.preset, write_output = false)
                rm = ModflowGPU.run(dir; backend = TEST_BACKEND, precision = p.preset, write_output = false, preconditioner = :mc_ilu0)
                @test ra.heads == rm.heads && [s.outer.inner for s in ra.steps] == [s.outer.inner for s in rm.steps]
                for kind in (_NEW_PRECONDITIONERS..., :amg)
                    r = ModflowGPU.run(dir; backend = TEST_BACKEND, precision = p.preset, outdir = tmp, write_output = false, preconditioner = kind)
                    @test r.converged
                    @test maximum(abs, r.heads .- ref.heads) < (p.T == Float64 ? 1.0e-7 : 1.0e-3)
                    kind in (:mc_ilu0, :amg) && @test sum(s.outer.inner for s in r.steps) < sum(s.outer.inner for s in ref.steps)
                end
            end
        end
    end
end
