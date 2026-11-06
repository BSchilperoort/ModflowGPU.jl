# Check a libmf6_probe.jl output directory: is the dumped SLN_1/AMAT, SLN_1/RHS the
# linear system that the first `solve` call solved? Prints the scaled residual
# ‖RHS − AMAT·X‖∞ / max_row Σ|a_ij x_j| with X = SLN_1/X after that call, and how
# constant-head rows look (rows with no off-diagonal entries).
#
# Usage: julia validation/check_probe_system.jl <probe output dir>

function check(dir)
    rd(name, T) = collect(reinterpret(T, read(joinpath(dir, "first_solve_SLN_1_$name.bin"))))
    ia = rd("IA", Int32)
    ja = rd("JA", Int32)
    a = rd("AMAT", Float64)
    b = rd("RHS", Float64)
    x = rd("X", Float64)
    n = length(ia) - 1
    res = 0.0
    scale = 0.0
    ndiagfirst = 0
    isolated = Int[]
    for i in 1:n
        s = 0.0
        t = 0.0
        ja[ia[i]] == i && (ndiagfirst += 1)
        offdiag_nz = false
        for k in ia[i]:(ia[i + 1] - 1)
            s += a[k] * x[ja[k]]
            t += abs(a[k] * x[ja[k]])
            ja[k] != i && a[k] != 0 && (offdiag_nz = true)
        end
        res = max(res, abs(b[i] - s))
        scale = max(scale, t)
        offdiag_nz || push!(isolated, i)
    end
    println(dir)
    println("  rows=$n nnz=$(length(ja)) diagonal-first rows=$ndiagfirst")
    println("  scaled residual of first solve: ", res / scale)
    if !isempty(isolated)
        i = first(isolated)
        println("  rows without off-diagonals (constant head / inactive): $(length(isolated)); e.g. row $i: diag=$(a[ia[i]]) rhs=$(b[i]) x=$(x[i])")
    end
    return nothing
end

foreach(check, ARGS)
