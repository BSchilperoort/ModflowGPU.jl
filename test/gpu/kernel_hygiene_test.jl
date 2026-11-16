# Kernel hygiene (PLAN.md section 3.7, rule 5; T3.1): every kernel compiled for `:single`
# on ROCm has no `double` in its device LLVM IR. Runs only with MFGPU_TEST_BACKEND=rocm.
#
# test/gpu/kernel_hygiene_workload.jl runs representative `:single` work (the run path with
# every boundary kind, CG and BiCGSTAB with each scaling, orthogonalisation and
# under-relaxation method, flows, budgets, SpMV and the convenience reductions) in a
# fresh process under `AMDGPU.@device_code_llvm`, twice:
#
# - `float64acc`: the `:single` preset as it runs on ROCm. The reductions accumulate in
#   `Ta = Float64` by design (rule 4), so only `mapreduce_kernel!`, `mapreduce_last_kernel!`,
#   `finalize_reduction_kernel!` and AMG's Galerkin products (`galerkin_kernel!`,
#   `galerkin_store_kernel!`) are exempt.
# - `float32acc`: the backend reports no Float64 (as Metal does), so `Ta = Float32`, and
#   no kernel is exempt.
#
# Every kernel the package defines (`gpu_*` functions of ModflowGPU) must be compiled by
# the workload, so a new kernel cannot escape the test: add work for it to the workload.

const HYGIENE_EXEMPT = ("mapreduce_kernel!", "mapreduce_last_kernel!", "finalize_reduction_kernel!", "galerkin_kernel!", "galerkin_store_kernel!")

function _hygiene_pass(pass)
    out = tempname()
    script = joinpath(@__DIR__, "kernel_hygiene_workload.jl")
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(@__DIR__)) $script $pass $out`
    log = IOBuffer()
    ok = success(pipeline(cmd; stdout = log, stderr = log))
    ok || error("kernel-hygiene workload ($pass) failed:\n" * String(take!(log)))
    jobs = Tuple{String, Int, String}[]
    static = String[]
    for line in eachline(out)
        f = split(line, '\t')
        if f[1] == "#static"
            static = split(f[2])
        else
            push!(jobs, (String(f[1]), parse(Int, f[2]), String(f[3])))
        end
    end
    rm(out)
    return jobs, static
end

if TEST_BACKEND_NAME == "rocm"
    @testset "Kernel hygiene (no double in :single device IR, ROCm)" begin
        compiled = Set{String}()
        for pass in ("float64acc", "float32acc")
            jobs, static = _hygiene_pass(pass)
            own = filter(j -> j[1] != "external", jobs)
            union!(compiled, first.(own))
            @test !isempty(static)
            bad = [(j[1], j[2]) for j in own if j[2] > 0 && !(pass == "float64acc" && j[1] in HYGIENE_EXEMPT)]
            for (name, n, job) in own
                if n > 0 && !(pass == "float64acc" && name in HYGIENE_EXEMPT)
                    @error "Kernel $name has $n `double` tokens in its :single device IR ($pass)" job
                end
            end
            @test isempty(bad)
            if pass == "float64acc"
                # The detector works: the Float64-accumulating reductions do have doubles.
                @test any(j -> j[1] == "mapreduce_kernel!" && j[2] > 0, own)
            else
                # The exempt reductions are clean with Ta = Float32.
                # (galerkin_store_kernel! only runs when Ta differs from Tp.)
                for name in HYGIENE_EXEMPT
                    name == "galerkin_store_kernel!" && continue
                    @test any(j -> j[1] == name, own)
                end
            end
            # Every boundary kind but CHD (which sets heads) is formulated.
            for K in ModflowGPU.BOUNDARY_KINDS
                K === :CHD && continue
                @test any(j -> j[1] == "boundary_formulate_kernel!" && occursin("BoundaryKind{:$K}", j[3]), own)
            end
            missing_kernels = setdiff(static, compiled)
            pass == "float32acc" && !isempty(missing_kernels) &&
                @error "Kernels not compiled by the hygiene workload (add work for them to test/gpu/kernel_hygiene_workload.jl): $(join(missing_kernels, ", "))"
            pass == "float32acc" && @test isempty(missing_kernels)
            @info "Kernel hygiene ($pass): $(length(own)) kernel specialisations of $(length(unique(first.(own)))) kernels checked, $(length(bad)) with double"
        end
    end
else
    @info "Skipping the kernel-hygiene test: it needs MFGPU_TEST_BACKEND=rocm (backend is $TEST_BACKEND_NAME)"
end
