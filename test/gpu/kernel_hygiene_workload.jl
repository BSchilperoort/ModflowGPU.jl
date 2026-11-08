# Workload of the kernel-hygiene test (PLAN.md section 3.7, rule 5), run in a subprocess
# by test/gpu/kernel_hygiene_test.jl:
#
#     julia --project=test test/gpu/kernel_hygiene_workload.jl <pass> <outfile>
#
# Runs representative `:single` work on ROCm under `AMDGPU.@device_code_llvm` (which also
# reports kernels that are already compiled) and writes, per compiled kernel job, a line
# `<kernel name>\t<number of "double" tokens in the optimised device IR>\t<job>` to
# `outfile`, followed by a line `#static\t<names>` with every kernel the package defines.
#
# `pass` is `float64acc` (the normal `:single` preset: state, Krylov and preconditioner in
# Float32, reductions accumulated in Float64, rule 4) or `float32acc` (the backend reports
# no Float64, as Metal does, so the reductions also accumulate in Float32).
using ModflowGPU, AMDGPU
using KernelAbstractions: KernelAbstractions

const PASS = ARGS[1]
const OUTFILE = ARGS[2]
PASS in ("float64acc", "float32acc") || error("unknown pass $PASS")
if PASS == "float32acc"
    # A Float64-less device: resolve_precision(:auto) gives :single, accumulation_type Float32.
    @eval ModflowGPU.supports_float64(::AMDGPU.ROCBackend) = false
end
AMDGPU.allowscalar(false)

const M = ModflowGPU
const BACKEND = M.select_backend("rocm")
const T = Float32
const MODELS = joinpath(@__DIR__, "..", "models")

# coupling_rde (CHD, WEL, RCHA, RIV, DRN, EVT, EVTA, STO) plus a GHB package, so every
# boundary kind is formulated.
function _workload_model(dir)
    rundir = joinpath(dir, "coupling_rde")
    cp(joinpath(MODELS, "coupling_rde"), rundir)
    chmod(rundir, 0o755; recursive = true)
    nam = joinpath(rundir, "rde.nam")
    write(nam, replace(read(nam, String), "  OC6  rde.oc  oc" => "  GHB6  rde.ghb  ghb\n  OC6  rde.oc  oc"))
    write(
        joinpath(rundir, "rde.ghb"), """
        BEGIN dimensions
          MAXBOUND  2
        END dimensions
        BEGIN period  1
          2 8 11  0.5  10.0
          2 4 6  -1.0  5.0
        END period
        """,
    )
    return rundir
end

# The outer loop of a few steps with an IMS variant (inner solver, scaling,
# orthogonalisation, under-relaxation) and a preconditioner (T4.2, T4.3), flows and budgets.
function _steps(sim, ims; pc...)
    spec = only(sim.models)
    m = M.GwfModel(BACKEND, T, spec, M.nper(sim.tdis))
    ps = M.PicardSolver(m, M.PicardSettings(ims, T; pc...))
    acc = M.BudgetAccumulator(m)
    for st in M.time_steps(sim.tdis)[1:3]
        st.kstp == 1 && M.apply_period!(m, st.kper)
        M.start_timestep!(m, st.delt)
        M.solve_timestep!(ps, m, st.kstp)
        M.calc_flows!(m)
        M.model_budget!(acc, m, st)
    end
    return m
end

const IMS_VARIANTS = [
    (; linear_acceleration = :cg, scaling_method = :none, number_orthogonalizations = 0, under_relaxation = :none),
    (; linear_acceleration = :cg, scaling_method = :diagonal, number_orthogonalizations = 3, under_relaxation = :dbd),
    (; linear_acceleration = :cg, scaling_method = :l2norm, number_orthogonalizations = 0, under_relaxation = :cooley),
    (; linear_acceleration = :bicgstab, scaling_method = :diagonal, number_orthogonalizations = 2, under_relaxation = :simple),
    (; linear_acceleration = :bicgstab, scaling_method = :l2norm, number_orthogonalizations = 0, under_relaxation = :dbd),
]

function workload(dir)
    rundir = _workload_model(dir)
    # The whole run path, with output.
    r = M.run(rundir; backend = BACKEND, precision = :single, outdir = joinpath(dir, "out"))
    r.T === T || error("expected a Float32 run")
    sim = M.load_simulation(rundir)
    # (AMG separately below, with a small coarsest level.)
    for v in IMS_VARIANTS, preconditioner in filter(!=(:amg), M.PRECONDITIONERS)
        ims = M.ImsSpec(;
            v..., outer_dvclose = 1.0e-5, outer_maximum = 50, inner_maximum = 200, inner_dvclose = 1.0e-6,
            inner_rclose = 1.0e-4, under_relaxation_gamma = 0.5, under_relaxation_theta = 0.7, under_relaxation_kappa = 0.1,
        )
        m = _steps(sim, ims; preconditioner)
        preconditioner === :jacobi || continue
        # Other entry points: SpMV, the convenience reductions, the residual with bounds.
        y = similar(m.res)
        M.spmv!(y, m.sys.amat, m.csr, m.h)
        M.device_dot(m.h, y)
        M.device_maxabs(y)
        M.device_argmaxabs(y)
        scale = similar(m.res)
        bound = similar(m.res)
        M.flux_residual!(m.res, m.sys, m.csr, m.npf, m.h, m.ibound; scale, bound)
        # The separate assembly kernels (the engine uses `formulate_system!` since T4.5).
        M.reset_terms!(m.sys)
        for pkg in m.packages
            M.boundary_kind(pkg) === :CHD || M.add_terms!(m.sys, m.csr, pkg)
        end
        M.add_terms!(m.sys, m.csr, m.sto, m.hold, m.ibound)
        M.assemble!(m.sys, m.csr, m.npf, m.h, m.ibound; symmetric = true)
    end
    # AMG (T4.3) with several levels: fused and generic finest level, CG and BiCGSTAB.
    for method in (:cg, :bicgstab), fine_degree in (0, 1)
        ims = M.ImsSpec(; linear_acceleration = method, outer_dvclose = 1.0e-5, inner_dvclose = 1.0e-6, inner_rclose = 1.0e-4)
        _steps(sim, ims; preconditioner = :amg, precond_options = (; max_coarse = 20, fine_degree))
    end
    KernelAbstractions.synchronize(BACKEND)
    return nothing
end

# Kernel functions the package defines: `@kernel function f!` defines the launcher `f!`
# and the device function `gpu_f!`.
static_kernels() = sort!(
    [
        String(n)[5:end] for n in names(M; all = true)
            if startswith(String(n), "gpu_") && isdefined(M, Symbol(String(n)[5:end]))
    ],
)

function main()
    io = IOBuffer()
    mktempdir() do dir
        AMDGPU.@device_code_llvm io = io dump_module = true workload(dir)
    end
    text = String(take!(io))
    # Each job's output starts with a "; GPUCompiler.CompilerJob{...}(MethodInstance for
    # <kernel>(<argument types>)..." line. Kernels of other packages (GPUArrays' fill!)
    # are listed as "external".
    chunks = split(text, r"^; (?=(?:GPUCompiler\.)?CompilerJob)"m; keepempty = false)
    open(OUTFILE, "w") do out
        for c in chunks
            header, body = let i = findfirst('\n', c)
                isnothing(i) ? (c, "") : (c[1:(i - 1)], c[(i + 1):end])
            end
            occursin("CompilerJob", header) || continue
            mm = match(r"MethodInstance for ModflowGPU\.gpu_([^\s(]+)\(", header)
            name = isnothing(mm) ? "external" : mm.captures[1]
            ndouble = count(_ -> true, eachmatch(r"\bdouble\b", body))
            println(out, name, '\t', ndouble, '\t', replace(header, '\t' => ' '))
        end
        println(out, "#static\t", join(static_kernels(), ' '))
    end
    return nothing
end

main()
