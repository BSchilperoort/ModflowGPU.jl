# Linear solvers (T2.7): PCG and BiCGSTAB with Jacobi, IMS scaling and inner convergence
# tests, on TEST_BACKEND in every precision. Solutions are compared with `\` (Float64,
# dense) on random SPD and nonsymmetric M-matrices (random graphs and the Connectivity of
# a real grid), and the iteration counts and stopping decisions with a host Float64 port
# of MF6's ims_base_cg / ims_base_bcgs (on `A` itself, Jacobi in place of ILU).
using LinearAlgebra: LinearAlgebra, Diagonal, norm, dot
using Random: Random, MersenneTwister
using ModflowGPU: CsrStructure, KrylovSettings, KrylovSolver, KrylovResult, solve!, spmv!, ims_testcnvg, ims_epfact,
    ims_preconditioner, ims_defaults, ImsSpec, device_dot, device_maxabs, build_connectivity, accumulation_type

# --- matrices in MF6's CSR layout (diagonal first, then the sorted neighbours)

# CSR (ia, ja, isym) of an undirected graph given by its neighbour sets.
function _csr_from_neighbours(nbrs::Vector{Vector{Int}})
    n = length(nbrs)
    ia = Int32[1]
    ja = Int32[]
    for i in 1:n
        push!(ja, i)
        append!(ja, sort(nbrs[i]))
        push!(ia, length(ja) + 1)
    end
    pos = Dict((i, Int(ja[ii])) => ii for i in 1:n for ii in ia[i]:(ia[i + 1] - 1))
    isym = Int32[pos[(Int(ja[ii]), i)] for i in 1:n for ii in ia[i]:(ia[i + 1] - 1)]
    return ia, ja, isym
end

function _device_csr(ia, ja, isym; backend = TEST_BACKEND)
    n = length(ia) - 1
    d(x) = copyto!(KernelAbstractions.allocate(backend, Int32, length(x)), Int32.(x))
    return CsrStructure(Int32(n), Int32(length(ja)), d(ia), d(ja), d(zeros(Int32, length(ja))), d(isym), d(ia[1:n]), d(1:n))
end

function _random_graph(rng, n, degree)
    nbrs = [Int[] for _ in 1:n]
    for i in 2:n  # a path keeps it connected
        push!(nbrs[i], i - 1)
        push!(nbrs[i - 1], i)
    end
    for _ in 1:(n * degree ÷ 2)
        i, j = rand(rng, 1:n), rand(rng, 1:n)
        (i == j || j in nbrs[i]) && continue
        push!(nbrs[i], j)
        push!(nbrs[j], i)
    end
    return nbrs
end

# M-matrix values (positive diagonal, negative off-diagonal, diagonally dominant) on the
# CSR pattern: symmetric or not, rows scaled by `rowscale`, the whole matrix negated for
# MF6's sign convention (`negate`), and identity rows (constant heads, with the matching
# columns of the other rows zeroed as sln_ls does for CG) at `chd`.
function _mmatrix(rng, ia, ja, isym; symmetric = true, negate = false, chd = Int[], rowscale = nothing, shift = 0.05)
    n = length(ia) - 1
    a = zeros(length(ja))
    for i in 1:n, ii in (ia[i] + 1):(ia[i + 1] - 1)
        j = ja[ii]
        if !symmetric || j > i
            a[ii] = -(0.1 + rand(rng))
            symmetric && (a[isym[ii]] = a[ii])
        end
    end
    for i in 1:n
        rs = -sum(a[(ia[i] + 1):(ia[i + 1] - 1)])
        cs = -sum(a[isym[ii]] for ii in (ia[i] + 1):(ia[i + 1] - 1); init = 0.0)
        a[ia[i]] = max(rs, cs) + shift + rand(rng) * shift
    end
    if !isnothing(rowscale)
        # symmetric scaling keeps symmetry: a_ij·s_i·s_j
        for i in 1:n, ii in ia[i]:(ia[i + 1] - 1)
            a[ii] *= rowscale[i] * rowscale[ja[ii]]
        end
    end
    negate && (a .= -a)
    for i in chd
        for ii in ia[i]:(ia[i + 1] - 1)
            a[ii] = ii == ia[i] ? 1.0 : 0.0
            a[isym[ii]] = isym[ii] == ia[ja[ii]] ? a[isym[ii]] : 0.0
        end
    end
    return a
end

function _dense(ia, ja, a)
    n = length(ia) - 1
    A = zeros(n, n)
    for i in 1:n, ii in ia[i]:(ia[i + 1] - 1)
        A[i, ja[ii]] = a[ii]
    end
    return A
end

# Host Float64 port of ims_base_cg / ims_base_bcgs with ims_base_testcnvg, zero initial
# guess, Jacobi preconditioner on `A` itself (MF6's sign), no scaling. Returns (x, iters, icnvg).
function _mf6_reference(A, b, st::KrylovSettings; kstp = 1)
    n = length(b)
    dinv = [A[i, i] != 0 ? 1 / A[i, i] : 1.0 for i in 1:n]
    x = zeros(n)
    d = copy(b)
    l2norm0 = norm(d)
    itmax = l2norm0 == 0 ? 0 : st.maxiter
    icnvg = l2norm0 == 0 ? 1 : 0
    epfact = ims_epfact(st.icnvgopt, kstp)
    sgn(x) = x + copysign(eps(Float64), x)
    close(a, b) = a == b || abs(a - b) <= 100 * eps() * max(abs(a), abs(b))
    iters = 0
    if st.method === :cg
        p = zeros(n)
        rho0 = 0.0
        for it in 1:itmax
            iters += 1
            z = dinv .* d
            rho = dot(d, z)
            p = it == 1 ? z : z .+ (rho / rho0) .* p
            q = A * p
            alpha = rho / sgn(dot(p, q))
            tv = alpha .* p
            x .+= tv
            d .-= alpha .* q
            dvmax = maximum(abs, tv)
            rc = st.icnvgopt >= 2 ? norm(d) : maximum(abs, d)
            icnvg = ims_testcnvg(st.icnvgopt, iters, dvmax, rc, l2norm0, epfact, st.dvclose, st.rclose)
            rc == 0 && (icnvg = 1)
            icnvg != 0 && break
            close(rho, rho0) && break
            if st.north > 0 && mod(it + 1, st.north) == 0
                d = b .- A * x
            end
            rho == 0 && break
            rho0 = rho
        end
    else
        dhat = copy(d)
        p = zeros(n)
        v = zeros(n)
        rho0 = alpha0 = omega0 = 0.0
        for it in 1:itmax
            iters += 1
            rho = dot(dhat, d)
            p = it == 1 ? copy(d) : d .+ ((rho / rho0) * (alpha0 / omega0)) .* (p .- omega0 .* v)
            phat = dinv .* p
            v = A * phat
            alpha = rho / sgn(dot(dhat, v))
            q = d .- alpha .* v
            qhat = dinv .* q
            t = A * qhat
            omega = dot(t, q) / sgn(dot(t, t))
            tv = alpha .* phat .+ omega .* qhat
            x .+= tv
            d = q .- omega .* t
            dvmax = maximum(abs, tv)
            rc = st.icnvgopt >= 2 ? norm(d) : maximum(abs, d)
            icnvg = ims_testcnvg(st.icnvgopt, iters, dvmax, rc, l2norm0, epfact, st.dvclose, st.rclose)
            rc == 0 && (icnvg = 1)
            icnvg != 0 && break
            (close(rho, rho0) || close(alpha, alpha0) || close(omega, omega0)) && break
            if st.north > 0 && mod(it + 1, st.north) == 0
                d = b .- A * x
            end
            rho * omega == 0 && break
            rho0, alpha0, omega0 = rho, alpha, omega
        end
    end
    return x, iters, icnvg < 0 ? 0 : icnvg
end

# Solve on TEST_BACKEND in precision p; returns (x on the host in Float64, result).
function _device_solve(p, csr, a, b, st)
    amat = to_test_device(p.T.(a))
    res = to_test_device(p.T.(b))
    δ = to_test_device(zeros(p.T, length(b)))
    solver = KrylovSolver(TEST_BACKEND, csr, st; Ts = p.Ts, Tp = p.Tp)
    result = solve!(δ, solver, amat, res)
    return Float64.(Array(δ)), result
end

# max |x − y| / max |y|, in the variables `w .* x` (for badly scaled matrices w = √|a_nn|,
# the variables the scaled solver iterates on; `\` itself is only accurate to about
# cond(A)·eps in the unscaled ones).
_relerr(x, y; w = 1) = maximum(abs, w .* (x .- y)) / maximum(abs, w .* y)

# Tight settings: converge on a residual far below the solution accuracy of `T`.
_tight(T, method; kw...) = KrylovSettings(;
    method, maxiter = 2000, dvclose = T == Float64 ? 1.0e-14 : 1.0e-7, rclose = T == Float64 ? 1.0e-13 : 1.0e-6, kw...,
)

@testset "krylov" begin
    @testset "IMS inner convergence test (ims_base_testcnvg)" begin
        # option 0: both criteria
        @test ims_testcnvg(0, 3, 1.0e-4, 0.05, 1.0, 1.0, 1.0e-3, 0.1) == 1
        @test ims_testcnvg(0, 3, -1.0e-4, -0.05, 1.0, 1.0, 1.0e-3, 0.1) == 1
        @test ims_testcnvg(0, 3, 1.0e-2, 0.05, 1.0, 1.0, 1.0e-3, 0.1) == 0
        @test ims_testcnvg(0, 3, 1.0e-4, 0.5, 1.0, 1.0, 1.0e-3, 0.1) == 0
        # option 1 (STRICT): converged only in the first iteration
        @test ims_testcnvg(1, 1, 1.0e-4, 0.05, 1.0, 1.0, 1.0e-3, 0.1) == 1
        @test ims_testcnvg(1, 2, 1.0e-4, 0.05, 1.0, 1.0, 1.0e-3, 0.1) == -1
        @test ims_testcnvg(1, 2, 1.0e-2, 0.05, 1.0, 1.0, 1.0e-3, 0.1) == 0
        # option 2 (L2NORM_RCLOSE): either criterion, or the epfact reduction
        @test ims_testcnvg(2, 5, 1.0e-2, 0.05, 1.0, 0.01, 1.0e-3, 0.1) == 1
        @test ims_testcnvg(2, 5, 1.0e-4, 5.0, 1000.0, 0.01, 1.0e-3, 0.1) == 1
        @test ims_testcnvg(2, 5, 1.0e-2, 5.0, 1000.0, 0.01, 1.0e-3, 0.1) == -1
        @test ims_testcnvg(2, 5, 1.0e-2, 50.0, 1000.0, 0.01, 1.0e-3, 0.1) == 0
        # option 3 (RELATIVE_RCLOSE): dvclose converges, rclose relative stops
        @test ims_testcnvg(3, 5, 1.0e-4, 50.0, 1000.0, 1.0, 1.0e-3, 1.0e-3) == 1
        @test ims_testcnvg(3, 5, 1.0e-2, 0.5, 1000.0, 1.0, 1.0e-3, 1.0e-3) == -1
        @test ims_testcnvg(3, 5, 1.0e-2, 5.0, 1000.0, 1.0, 1.0e-3, 1.0e-3) == 0
        # option 4 (L2NORM_RELATIVE_RCLOSE)
        @test ims_testcnvg(4, 5, 1.0e-4, 0.05, 1000.0, 1.0e-4, 1.0e-3, 0.1) == 1
        @test ims_testcnvg(4, 5, 1.0e-2, 0.05, 1000.0, 1.0e-4, 1.0e-3, 0.1) == -1
        @test ims_testcnvg(4, 5, 1.0e-2, 0.5, 1000.0, 1.0e-4, 1.0e-3, 0.1) == 0
        @test ims_epfact(2, 1) == 0.01 && ims_epfact(2, 3) == 0.1 && ims_epfact(4, 1) == 1.0e-4 && ims_epfact(0, 1) == 1.0
    end

    @testset "IMS option mapping" begin
        @test ims_preconditioner(0, 0.0) === :ilu0
        @test ims_preconditioner(0, 0.97) === :milu0
        @test ims_preconditioner(5, 0.0) === :ilut
        @test ims_preconditioner(5, 0.5) === :milut
        for (cx, method, pc, north) in ((:simple, :cg, "ILU0", 0), (:moderate, :bicgstab, "MILU0", 0), (:complex, :bicgstab, "ILUT", 2))
            ims = ims_defaults(cx)
            st = @test_logs (:warn, Regex("preconditioner $pc .*Jacobi")) KrylovSettings(ims)
            @test st.method === method && st.north == north && st.maxiter == ims.inner_maximum
            @test st.dvclose == ims.inner_dvclose && st.rclose == ims.inner_rclose && st.icnvgopt == 0
        end
        ims = ImsSpec(; rclose_option = :l2norm_relative, scaling_method = :diagonal, reordering_method = :rcm)
        st = @test_logs (:warn,) (:info, r"REORDERING_METHOD RCM") KrylovSettings(ims)
        @test st.icnvgopt == 4 && st.scaling === :diagonal
        @test_throws ArgumentError KrylovSettings(; method = :gmres)
        @test_throws ArgumentError KrylovSettings(; rclose = 0.0)
        @test KrylovSettings(; rclose = 0.0, rclose_option = :relative).rclose == 0.0
    end

    @testset "device reductions" begin
        rng = MersenneTwister(1)
        for_each_precision() do p
            for n in (1, 255, 256, 257, 70_001)
                x = rand(rng, p.Ts, n) .- p.Ts(0.5)
                y = rand(rng, p.Ts, n)
                dx, dy = to_test_device(x), to_test_device(y)
                ref = sum(Float64.(x) .* Float64.(y))
                tol = accumulation_type(TEST_BACKEND, p.Ts) == Float64 ? 1.0e-13 : 1.0e-5
                @test abs(device_dot(dx, dy) - ref) <= tol * sum(abs.(Float64.(x) .* y))
                @test device_maxabs(dx) == Float64(maximum(abs, x))
            end
        end
    end

    @testset "spmv!" begin
        rng = MersenneTwister(2)
        ia, ja, isym = _csr_from_neighbours(_random_graph(rng, 300, 4))
        csr = _device_csr(ia, ja, isym)
        a = _mmatrix(rng, ia, ja, isym; symmetric = false)
        x = rand(rng, 300)
        for_each_precision() do p
            y = to_test_device(zeros(p.T, 300))
            spmv!(y, to_test_device(p.T.(a)), csr, to_test_device(p.T.(x)))
            @test _relerr(Float64.(Array(y)), _dense(ia, ja, a) * x) < (p.T == Float64 ? 1.0e-14 : 1.0e-6)
        end
    end

    @testset "random M-matrices vs \\" begin
        rng = MersenneTwister(3)
        n = 500
        ia, ja, isym = _csr_from_neighbours(_random_graph(rng, n, 6))
        csr = _device_csr(ia, ja, isym)
        rowscale = exp.(randn(rng, n))  # rows over ~±1.5 orders of magnitude
        cases = [
            ("SPD, CG", :cg, (; symmetric = true), :none),
            ("SPD, MF6 sign (CG on -A)", :cg, (; symmetric = true, negate = true), :none),
            ("SPD badly scaled, CG, DIAGONAL", :cg, (; symmetric = true, negate = true, rowscale), :diagonal),
            ("nonsymmetric, BiCGSTAB", :bicgstab, (; symmetric = false), :none),
            ("nonsymmetric, MF6 sign, BiCGSTAB", :bicgstab, (; symmetric = false, negate = true), :none),
            ("nonsymmetric badly scaled, BiCGSTAB, DIAGONAL", :bicgstab, (; symmetric = false, negate = true, rowscale), :diagonal),
            ("nonsymmetric badly scaled, BiCGSTAB, L2NORM", :bicgstab, (; symmetric = false, negate = true, rowscale), :l2norm),
            ("SPD, BiCGSTAB", :bicgstab, (; symmetric = true, negate = true), :none),
        ]
        for (name, method, mkw, scaling) in cases
            @testset "$name" begin
                a = _mmatrix(MersenneTwister(hash(name)), ia, ja, isym; mkw...)
                A = _dense(ia, ja, a)
                xref = randn(rng, n)
                b = A * xref
                xs = A \ b
                w = haskey(mkw, :rowscale) ? sqrt.(abs.([A[i, i] for i in 1:n])) : 1
                for_each_precision() do p
                    # RELATIVE_RCLOSE: stop when ‖r‖₂ <= rclose·‖r₀‖₂ (scaled units).
                    st = KrylovSettings(;
                        method, maxiter = 2000, dvclose = 0.0, rclose_option = :relative, scaling,
                        rclose = p.Ts == Float64 ? 1.0e-11 : 1.0e-6,
                    )
                    x, r = _device_solve(p, csr, a, b, st)
                    @test r.iterations > 1
                    @test _relerr(x, xs; w) < (p.Ts == Float64 ? 1.0e-10 : 1.0e-4)
                    # With L2NORM scaling MF6's BiCGSTAB compares the unscaled L2 norm with
                    # the scaled L2NORM0, so the relative criterion is not reached here.
                    p.Ts == Float64 && scaling !== :l2norm && @test r.status === :criterion
                end
            end
        end
    end

    @testset "grid Connectivity with constant heads" begin
        conn = build_connectivity(_synthetic_coupling_dis())
        csr = CsrStructure(TEST_BACKEND, conn)
        ia, ja, isym = Int.(conn.ia), Int.(conn.ja), Int.(conn.isym)
        n = Int(conn.nodes)
        rng = MersenneTwister(4)
        chd = [1, 7, n ÷ 2, n]
        for (method, symmetric) in ((:cg, true), (:bicgstab, true), (:bicgstab, false))
            a = _mmatrix(rng, ia, ja, isym; symmetric, negate = true, chd, shift = 1.0e-3)
            A = _dense(ia, ja, a)
            b = A * randn(rng, n)
            b[chd] .= 0  # the correction residual of a constant-head row
            xs = A \ b
            for_each_precision() do p
                st = _tight(p.Ts, method; rclose = (p.Ts == Float64 ? 1.0e-13 : 1.0e-6) * maximum(abs, b))
                x, r = _device_solve(p, csr, a, b, st)
                @test _relerr(x, xs) < (p.Ts == Float64 ? 1.0e-10 : 1.0e-4)
                @test all(==(0), x[chd])
            end
        end
    end

    @testset "MF6 semantics: iteration counts and stopping" begin
        rng = MersenneTwister(5)
        n = 400
        ia, ja, isym = _csr_from_neighbours(_random_graph(rng, n, 4))
        csr = _device_csr(ia, ja, isym)
        asym = _mmatrix(rng, ia, ja, isym; symmetric = true, negate = true, shift = 0.01)
        anon = _mmatrix(rng, ia, ja, isym; symmetric = false, negate = true, shift = 0.01)
        b = randn(rng, n)
        configs = [
            (; rclose_option = :infinity_norm, dvclose = 1.0e-6, rclose = 1.0e-5),
            (; rclose_option = :infinity_norm, dvclose = 1.0e-6, rclose = 1.0e-5, maxiter = 7),
            (; rclose_option = :strict, dvclose = 1.0e-6, rclose = 1.0e-5),
            (; rclose_option = :l2norm, dvclose = 1.0e-9, rclose = 1.0e-6),
            (; rclose_option = :relative, dvclose = 1.0e-9, rclose = 1.0e-4),
            (; rclose_option = :l2norm_relative, dvclose = 1.0e-7, rclose = 1.0e-6),
            (; rclose_option = :infinity_norm, dvclose = 1.0e-8, rclose = 1.0e-7, north = 3),
        ]
        for method in (:cg, :bicgstab), cfg in configs
            a = method === :cg ? asym : anon
            A = _dense(ia, ja, a)
            st = KrylovSettings(; method, maxiter = get(cfg, :maxiter, 500), cfg...)
            xr, itr, icr = _mf6_reference(A, b, st)
            for_each_precision() do p
                x, r = _device_solve(p, csr, a, b, st)
                if p.Ts == Float64
                    @test r.iterations == itr
                    @test r.converged == (icr == 1)
                    @test _relerr(x, xr) < 1.0e-10
                end
                # The status agrees with ims_base_testcnvg on the reported norms.
                rc = st.icnvgopt >= 2 ? r.l2norm : r.rmax
                icnvg = ims_testcnvg(st.icnvgopt, r.iterations, r.dvmax, rc, r.l2norm0, 1.0 * ims_epfact(st.icnvgopt, 1), st.dvclose, st.rclose)
                r.status === :converged && @test icnvg == 1
                r.status === :criterion && @test icnvg == -1
                r.status === :maxiter && @test icnvg == 0 && r.iterations == st.maxiter
                cfg.rclose_option === :strict && @test !r.converged && r.status === :criterion
                cfg.rclose_option === :relative && @test r.status === :criterion && r.l2norm <= 1.0e-4 * r.l2norm0
                get(cfg, :maxiter, 0) == 7 && @test r.iterations == 7 && r.status === :maxiter
            end
        end
        # GPU backend vs the CPU backend (PLAN.md section 4): same iterations, δ within
        # 1e-10 relative (:double) or 1e-4 (:single).
        if !(TEST_BACKEND isa CPU)
            for method in (:cg, :bicgstab)
                a = method === :cg ? asym : anon
                st = KrylovSettings(; method, maxiter = 200, dvclose = 1.0e-9, rclose = 1.0e-8)
                for_each_precision() do p
                    x, r = _device_solve(p, csr, a, b, st)
                    cpu = KrylovSolver(CPU(), _device_csr(ia, ja, isym; backend = CPU()), st; Ts = p.Ts, Tp = p.Tp)
                    δ = zeros(p.T, n)
                    rc = solve!(δ, cpu, p.T.(a), p.T.(b))
                    @test _relerr(x, Float64.(δ)) < (p.Ts == Float64 ? 1.0e-10 : 1.0e-4)
                    p.Ts == Float64 && @test r.iterations == rc.iterations
                end
            end
        end
        # zero right-hand side: no iterations, converged
        for_each_precision() do p
            x, r = _device_solve(p, csr, asym, zeros(n), KrylovSettings())
            @test r.iterations == 0 && r.converged && all(==(0), x)
        end
    end
end
