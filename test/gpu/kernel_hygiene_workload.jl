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
    write(nam, replace(read(nam, String), "  OC6  rde.oc  oc" => "  GHB6  rde.ghb  ghb\n  OC6  rde.oc  oc\n  OBS6  rde.obs  obs"))
    # T6.7: observations (the gather kernel of the heads, FLOWJA and SIMVALS).
    write(joinpath(rundir, "rde.obs"), "BEGIN continuous FILEOUT rde.obs.csv\n  h head 2 8 11\n  q flow-ja-face 2 8 11 2 8 10\nEND continuous\n")
    write(joinpath(rundir, "rde.ghb.obs"), "BEGIN continuous FILEOUT rde.ghb.obs.csv\n  g ghb 2 8 11\nEND continuous\n")
    write(
        joinpath(rundir, "rde.ghb"), """
        BEGIN options
          OBS6 FILEIN rde.ghb.obs
        END options
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
    # Device BOUND writes (T5.4): the check kernel of every package kind at prepare_solve.
    s = M.initialize(rundir; backend = BACKEND, precision = :single, write_output = false)
    M.prepare_time_step(s)
    for p in s.model.packages
        M.device_bound!(s, "$(s.model.spec.name)/$(p.spec.name)/BOUND")
    end
    M.prepare_solve(s)
    M.finalize(s)
    # T6.9: the partial BOUND restage (time-series entries and the EVT surfaces of moved
    # records after device writes).
    bnd = KernelAbstractions.zeros(BACKEND, T, 2, 3)
    for shift in (false, true)
        M._scatter_bound!(bnd, Int32[1, 4], T[1, 2]; shift)
    end
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
        # A reduction split over kernels whose last part is empty: the separate
        # finalisation (`finalize_reduction_kernel!`, T4.7's fallback of `_reduce_last!`).
        red = M.Reducer(M.get_backend(y), length(y), M.accumulation_type(M.get_backend(y), eltype(y)))
        g = M._reduce_stage1!(red, M.get_backend(y), 0, M._dot_term, M._SUM4, length(y), y, y)
        M._reduce_last!(red, M.get_backend(y), g, M._dot_term, M._SUM4, 0, M._NoPost(), red.out, red.out, y, y)
        # Backtracking (T6.2a): xtemp, the norm, the maximum change and the reduction.
        hp = similar(y)
        M.save_hprev!(hp, m.h, m.ibound)
        M.backtrack!(
            () -> nothing, red, y, hp, m.res, m.ibound, 2, 0.0;
            numtrack = 1, btol = 1.0, breduc = 0.5, res_lim = 0.0, dvclose = 0.0,
        )
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
    # Convertible cells (T6.1): wetting and drying, PERCHED, convertible storage, WEL
    # AUTO_FLOW_REDUCE, staggered DISU connections; and the separate assembly, residual and
    # storage kernels with a ConvertibleNpf / ConvertibleStorage.
    # T6.2: the Newton formulation (test/models/newton: PTC, NEWTON UNDER_RELAXATION, Newton
    # storage, DRN and WEL terms, backtracking).
    # T6.11: HFB6 (the scatter kernel; barriers in the convertible conductances).
    for name in ("wetdry", "afr", "disu_staggered", "newton", "hfb_conv")
        d = joinpath(dir, name)
        cp(joinpath(MODELS, name), d)
        chmod(d, 0o755; recursive = true)
        M.run(d; backend = BACKEND, precision = :single, outdir = joinpath(dir, "out_$name"))
        csim = M.load_simulation(d)
        m = M.GwfModel(BACKEND, T, only(csim.models), M.nper(csim.tdis))
        steps = M.time_steps(csim.tdis)
        M.apply_period!(m, 1)
        if length(csim.tdis.perlen) > 1
            M.apply_period!(m, 2)
        end
        M.start_timestep!(m, steps[end].delt)
        M.formulate!(m)
        scale = similar(m.res)
        bound = similar(m.res)
        M.flux_residual!(m.res, m.sys, m.csr, m.npf, m.h, m.ibound; scale, bound)
        M.reset_terms!(m.sys)
        isnothing(m.sto) || M.add_terms!(m.sys, m.csr, m.sto, m.hold, m.ibound; h = m.h)
        M.assemble!(m.sys, m.csr, m.npf, m.h, m.ibound)
    end
    # XT3D (T6.5): the formulation and flow kernels (test/models/xt3d).
    d = joinpath(dir, "xt3d")
    cp(joinpath(MODELS, "xt3d"), d)
    chmod(d, 0o755; recursive = true)
    M.run(d; backend = BACKEND, precision = :single, outdir = joinpath(dir, "out_xt3d"))
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
