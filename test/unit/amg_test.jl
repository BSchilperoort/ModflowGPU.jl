# Smoothed-aggregation AMG preconditioner (T4.3, src/solver/precond/amg.jl) on TEST_BACKEND
# in every precision: CG and BiCGSTAB solutions against `\`, the hierarchy (vertical
# columns first), iteration counts independent of the grid size, symmetry of the V-cycle,
# the device recomputation of prolongators and Galerkin products against a fresh host
# build, and a whole run of the synthetic `tiny` model against Jacobi.
using LinearAlgebra: LinearAlgebra, dot, norm
const adapt = ModflowGPU.adapt
using ModflowGPU: AMGOptions, AMGPreconditioner, KrylovSettings, KrylovSolver, CsrStructure, solve!, precond_apply!,
    precond_setup!, amg_stats, JacobiPreconditioner, build_preconditioner
isdefined(@__MODULE__, :SyntheticModels) || include(joinpath(@__DIR__, "..", "..", "benchmark", "synthetic.jl"))

# A 2-layer nx × ny grid in MF6's CSR layout (diagonal first, MF6 sign): horizontal
# conductances 0.2–1.2 (pseudo-random), vertical `av` (strong), storage-like `leak` on the
# diagonal, and constant-head identity rows at `chd` (their columns zeroed, as sln_ls does
# for CG).
function _amg_grid(nx, ny; av = 100.0, leak = 1.0e-3, chd = Int[], seed = 1)
    n = 2nx * ny
    id(i, j, k) = (k - 1) * nx * ny + (j - 1) * nx + i
    nbrs = [Int[] for _ in 1:n]
    cond = Dict{Tuple{Int, Int}, Float64}()
    h(x) = 0.2 + abs(sin(12.9898 * x + seed))
    for k in 1:2, j in 1:ny, i in 1:nx
        p = id(i, j, k)
        for (di, dj) in ((1, 0), (0, 1))
            (i + di > nx || j + dj > ny) && continue
            q = id(i + di, j + dj, k)
            c = h(p + 0.37q)
            push!(nbrs[p], q), push!(nbrs[q], p)
            cond[(p, q)] = cond[(q, p)] = c
        end
        if k == 1
            q = id(i, j, 2)
            push!(nbrs[p], q), push!(nbrs[q], p)
            cond[(p, q)] = cond[(q, p)] = av
        end
    end
    ia = Int32[1]
    ja = Int32[]
    a = Float64[]
    for p in 1:n
        push!(ja, p)
        push!(a, 0.0)
        dpos = length(a)
        for q in sort(nbrs[p])
            push!(ja, q)
            push!(a, cond[(p, q)])
            a[dpos] -= cond[(p, q)]
        end
        a[dpos] -= leak
        push!(ia, length(ja) + 1)
    end
    pos = Dict((i, Int(ja[ii])) => ii for i in 1:n for ii in ia[i]:(ia[i + 1] - 1))
    isym = Int32[pos[(Int(ja[ii]), i)] for i in 1:n for ii in ia[i]:(ia[i + 1] - 1)]
    for c in chd, ii in ia[c]:(ia[c + 1] - 1)
        a[ii] = ii == ia[c] ? 1.0 : 0.0
        isym[ii] == ia[ja[ii]] || (a[isym[ii]] = 0.0)
    end
    return ia, ja, isym, a
end

function _amg_csr(ia, ja, isym)
    n = length(ia) - 1
    d(x) = to_test_device(Int32.(collect(x)))
    return CsrStructure(Int32(n), Int32(length(ja)), d(ia), d(ja), d(zeros(Int32, length(ja))), d(isym), d(ia[1:n]), d(1:n))
end

function _amg_dense(ia, ja, a)
    n = length(ia) - 1
    A = zeros(n, n)
    for i in 1:n, ii in ia[i]:(ia[i + 1] - 1)
        A[i, ja[ii]] = a[ii]
    end
    return A
end

_amg_settings(method, T; kw...) = KrylovSettings(;
    method, maxiter = 1000, dvclose = T == Float64 ? 1.0e-12 : 1.0e-6, rclose = T == Float64 ? 1.0e-11 : 1.0e-5,
    preconditioner = :amg, precond_options = (; max_coarse = 40, kw...),
)

function _amg_solve(p, csr, a, b, st)
    solver = KrylovSolver(TEST_BACKEND, csr, st; Ts = p.Ts, Tp = p.Tp)
    δ = to_test_device(zeros(p.T, length(b)))
    result = solve!(δ, solver, to_test_device(p.T.(a)), to_test_device(p.T.(b)))
    return Float64.(Array(δ)), result, solver
end

# z = M⁻¹·r with the preconditioner of `solver` (set up for `amat`), on the host in Float64.
function _amg_apply(solver, r)
    z = similar(solver.r)
    precond_apply!(z, solver.precond, to_test_device(eltype(solver.r).(r)))
    return Float64.(Array(z))
end

@testset "amg" begin
    @testset "options and selection" begin
        @test_throws ArgumentError AMGOptions(; theta = -1.0)
        @test_throws ArgumentError AMGOptions(; smoother = :gauss_seidel)
        @test_throws ArgumentError AMGOptions(; strength = :classical)
        @test_throws ArgumentError AMGOptions(; degree = 0)
        @test_throws ArgumentError KrylovSettings(; preconditioner = :ilu0)
        @test_throws ArgumentError build_preconditioner(:jacobi, TEST_BACKEND, nothing, nothing; Ts = Float64, Tp = Float64, theta = 0.1)
        @test_throws ArgumentError KrylovSettings(; precond_options = (; theta = 0.1))
        # precond_options reach the constructor of each preconditioner (T4.2's settings too).
        gia, gja, gisym, _ = _amg_grid(3, 2)
        gcsr = _amg_csr(gia, gja, gisym)
        ks(pc, o) = KrylovSolver(TEST_BACKEND, gcsr, KrylovSettings(; preconditioner = pc, precond_options = o))
        @test ks(:chebyshev, (; degree = 6, ratio = 20.0)).precond.degree == 6
        @test ks(:mc_ilu0, (; relax = 0.5)).precond.relax == 0.5
        @test ks(:amg, (; theta = 0.2)).precond.options.theta == 0.2
        @test_throws MethodError ks(:chebyshev, (; theta = 0.1))
        @test KrylovSettings().preconditioner === :jacobi
        @test KrylovSettings(; preconditioner = :amg, precond_options = (; theta = 0.2)).precond_options.theta == 0.2
    end

    for_each_precision(; presets = SOLVER_TEST_PRESETS) do p
        T = p.T
        tol = T == Float64 ? 1.0e-8 : 2.0e-3
        @testset "CG and BiCGSTAB vs \\ ($method)" for method in (:cg, :bicgstab)
            nx, ny = 24, 20
            chd = [1, 2, 3, nx * ny + 5]
            ia, ja, isym, a = _amg_grid(nx, ny; chd)
            n = length(ia) - 1
            csr = _amg_csr(ia, ja, isym)
            b = [sin(0.1i) + 0.5 for i in 1:n]
            b[chd] .= 0
            x, res, solver = _amg_solve(p, csr, a, b, _amg_settings(method, T))
            xr = _amg_dense(ia, ja, a) \ b
            @test res.converged
            @test maximum(abs, x .- xr) / maximum(abs, xr) < tol
            @test res.iterations <= 30
            st = amg_stats(solver)
            @test st.builds == 1 && length(st.sizes) >= 3
            # Constant heads are not aggregated, and their correction stays zero.
            @test maximum(abs, x[chd]) <= 10 * eps(T) * maximum(abs, x)
            # Jacobi needs many more iterations.
            jst = KrylovSettings(; method, maxiter = 5000, dvclose = _amg_settings(method, T).dvclose, rclose = _amg_settings(method, T).rclose)
            js = KrylovSolver(TEST_BACKEND, csr, jst; Ts = p.Ts, Tp = p.Tp)
            @test js.precond isa JacobiPreconditioner
            δ = to_test_device(zeros(T, n))
            jres = solve!(δ, js, to_test_device(T.(a)), to_test_device(T.(b)))
            @test jres.iterations > 4 * res.iterations
        end

        @testset "vertical columns first; grid-independent iterations" begin
            iters = Int[]
            for (nx, ny) in ((16, 16), (32, 32), (64, 64))
                ia, ja, isym, a = _amg_grid(nx, ny; seed = 2)
                n = length(ia) - 1
                _, res, solver = _amg_solve(p, _amg_csr(ia, ja, isym), a, [cos(0.05i) for i in 1:n], _amg_settings(:cg, T))
                @test res.converged
                @test amg_stats(solver).sizes[2] == nx * ny  # one aggregate per column
                push!(iters, res.iterations)
            end
            @test maximum(iters) <= 1.6 * minimum(iters)
            @test maximum(iters) <= 25
        end

        @testset "V-cycle symmetric positive definite (CG)" begin
            ia, ja, isym, a = _amg_grid(20, 16; chd = [7])
            n = length(ia) - 1
            _, _, solver = _amg_solve(p, _amg_csr(ia, ja, isym), a, ones(n), _amg_settings(:cg, T))
            x = [sin(0.3i) for i in 1:n]
            y = [cos(0.7i) + 0.1 for i in 1:n]
            x[7] = y[7] = 0  # the constant-head row is the identity
            mx, my = _amg_apply(solver, x), _amg_apply(solver, y)
            @test abs(dot(mx, y) - dot(x, my)) <= (p.Tp == Float64 ? 1.0e-10 : 1.0e-4) * norm(mx) * norm(y)
            @test dot(mx, x) > 0 && dot(my, y) > 0
        end

        @testset "hierarchy reuse: device update equals a fresh build" begin
            ia, ja, isym, a1 = _amg_grid(20, 16; leak = 1.0e-3)
            n = length(ia) - 1
            # Same off-diagonals, a different diagonal (boundary and storage terms).
            a2 = copy(a1)
            for i in 1:n
                a2[ia[i]] -= 0.5 * (1 + sin(0.2i))
            end
            csr = _amg_csr(ia, ja, isym)
            st = _amg_settings(:cg, T)
            solver = KrylovSolver(TEST_BACKEND, csr, st; Ts = p.Ts, Tp = p.Tp)
            δ = to_test_device(zeros(T, n))
            b = to_test_device(T.(ones(n)))
            solve!(δ, solver, to_test_device(T.(a1)), b)
            r2 = solve!(δ, solver, to_test_device(T.(a2)), b)
            stats = amg_stats(solver)
            @test stats.builds == 1 && stats.updates == 1 && r2.converged
            _, _, fresh = _amg_solve(p, csr, a2, ones(n), st)
            v = [sin(0.11i) for i in 1:n]
            z1, z2 = _amg_apply(solver, v), _amg_apply(fresh, v)
            @test maximum(abs, z1 .- z2) <= (p.Tp == Float64 ? 1.0e-10 : 1.0e-4) * maximum(abs, z2)
            # An unchanged matrix keeps the coarse levels.
            solve!(δ, solver, to_test_device(T.(a2)), b)
            @test stats.skipped == 1 && stats.updates == 1
            @test _amg_apply(solver, v) == z1
        end

        @testset "options: Jacobi smoother, plain aggregation, single level" begin
            ia, ja, isym, a = _amg_grid(16, 12; chd = [3])
            n = length(ia) - 1
            b = [sin(0.2i) for i in 1:n]
            b[3] = 0
            xr = _amg_dense(ia, ja, a) \ b
            csr = _amg_csr(ia, ja, isym)
            for opts in ((; smoother = :jacobi, degree = 2), (; smooth = false), (; strength = :symmetric, theta = 0.25))
                x, res, _ = _amg_solve(p, csr, a, b, _amg_settings(:cg, T; opts...))
                @test res.converged && maximum(abs, x .- xr) / maximum(abs, xr) < tol
            end
            # n <= max_coarse: the dense inverse of the whole matrix (a direct solve).
            x, res, solver = _amg_solve(p, csr, a, b, _amg_settings(:cg, T; max_coarse = 1000))
            @test res.converged && res.iterations <= (p.Tp == Float64 ? 2 : 6) && length(amg_stats(solver).sizes) == 1
            @test maximum(abs, x .- xr) / maximum(abs, xr) < tol
        end

        @testset "adapt: an unbuilt preconditioner on the target" begin
            ia, ja, isym, a = _amg_grid(8, 8)
            solver = KrylovSolver(TEST_BACKEND, _amg_csr(ia, ja, isym), _amg_settings(:cg, T); Ts = p.Ts, Tp = p.Tp)
            moved = adapt(TEST_BACKEND, solver)
            @test moved.precond isa AMGPreconditioner && !moved.precond.built
            @test moved.precond.options == solver.precond.options
        end

        @testset "synthetic tiny model: AMG vs Jacobi" begin
            spec = SyntheticModels.synthetic_simulation(:tiny; output = false, ntransient = 2)
            rj = ModflowGPU.run(spec; backend = TEST_BACKEND, precision = p.preset, write_output = false)
            ra = ModflowGPU.run(spec; backend = TEST_BACKEND, precision = p.preset, write_output = false, preconditioner = :amg)
            @test ra.converged && rj.converged
            @test maximum(abs, ra.heads .- rj.heads) < (T == Float64 ? 1.0e-6 : 1.0e-3)
            @test ra.steps[1].outer.inner < rj.steps[1].outer.inner / 4
        end
    end
end
