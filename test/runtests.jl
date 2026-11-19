# Test suite of ModflowGPU.
#
# By default everything runs on the CPU backend (this is what `Pkg.test()` and CI do).
# Set MFGPU_TEST_BACKEND to run on a GPU backend as well; GPU tests run only locally:
#   MFGPU_TEST_BACKEND=rocm julia --project=test test/runtests.jl
# Set MFGPU_TEST_GROUPS to a comma-separated list of test groups (see TEST_GROUPS below) to
# run only those; unset runs everything:
#   MFGPU_TEST_GROUPS=connectivity,npf julia --project=test test/runtests.jl
# Set MFGPU_TEST_JOBS=N (N > 1) to run the groups in N parallel Julia processes (test/parallel.jl);
# unset or 1 is the serial run below. `Pkg.test()` is serial unless the variable is set.
#   MFGPU_TEST_JOBS=4 julia --project=test test/runtests.jl
# test/gpu/runtests.jl is the GPU entry: the same suite with MFGPU_TEST_BACKEND (default
# rocm); every parity test then runs on the GPU, and the run and assembly parity tests also
# compare it with the CPU backend (PLAN.md section 4).
using Test

include("test_groups.jl")
const TEST_JOBS = something(tryparse(Int, strip(get(ENV, "MFGPU_TEST_JOBS", "1"))), 1)
if TEST_JOBS > 1
    # The driver starts worker processes that run this file serially (MFGPU_TEST_JOBS=1)
    # and loads nothing of the package itself.
    include("parallel.jl")
    run_parallel_tests(TEST_JOBS) || exit(1)
    exit(0)
end

using ModflowGPU
using KernelAbstractions: KernelAbstractions, CPU

const TEST_BACKEND_NAME = lowercase(get(ENV, "MFGPU_TEST_BACKEND", "cpu"))

# Load the GPU package of the requested backend, which registers it with ModflowGPU.
const TEST_GPU_PACKAGE = ModflowGPU.known_backend_package(TEST_BACKEND_NAME)
if TEST_BACKEND_NAME != "cpu"
    isnothing(TEST_GPU_PACKAGE) && error(
        "Unknown MFGPU_TEST_BACKEND '$TEST_BACKEND_NAME'. Valid options are: " *
            join(ModflowGPU.backend_options()[2:end], ", "),
    )
    @eval using $(Symbol(TEST_GPU_PACKAGE))
end

const TEST_BACKEND = ModflowGPU.select_backend(TEST_BACKEND_NAME)

# No scalar indexing of device arrays in GPU tests (PLAN.md section 2).
if !isnothing(TEST_GPU_PACKAGE)
    @eval $(Symbol(TEST_GPU_PACKAGE)).allowscalar(false)
end

@info "Running tests on backend $(ModflowGPU.backend_name(TEST_BACKEND))"

include("testutils.jl")
include("input_files.jl")
include("libmf6_oracle.jl")
include("grids.jl")
include("residual.jl")
include("run_parity.jl")

const SELECTED_GROUPS = let
    valid = first.(TEST_GROUPS)
    env = strip(get(ENV, "MFGPU_TEST_GROUPS", ""))
    selected = isempty(env) ? valid : [lowercase(strip(g)) for g in split(env, ',') if !isempty(strip(g))]
    unknown = setdiff(selected, valid)
    isempty(unknown) || error(
        "Unknown MFGPU_TEST_GROUPS name(s): $(join(unknown, ", ")). Valid groups are: $(join(valid, ", "))",
    )
    selected
end

@info "Running test groups: $(join(SELECTED_GROUPS, ", "))"

# A worker of the parallel driver (test/parallel.jl) writes its test counts to this file.
const TEST_RESULT_FILE = get(ENV, "MFGPU_TEST_RESULT_FILE", "")

function write_test_counts(counts)
    return isempty(TEST_RESULT_FILE) || write(TEST_RESULT_FILE, join(counts, ' '))
end

try
    ts = @testset "ModflowGPU" begin
        for (group, files) in TEST_GROUPS
            group in SELECTED_GROUPS || continue
            foreach(include, files)
        end
        # The GPU backend tests belong to the "backend" group.
        if !(TEST_BACKEND isa CPU) && "backend" in SELECTED_GROUPS
            include("gpu/backend_gpu_test.jl")
        end
    end
    c = Test.get_test_counts(ts)
    write_test_counts((c.passes + c.cumulative_passes, c.fails + c.cumulative_fails, c.errors + c.cumulative_errors, c.broken + c.cumulative_broken))
catch err
    err isa Test.TestSetException && write_test_counts((err.pass, err.fail, err.error, err.broken))
    rethrow()
end
