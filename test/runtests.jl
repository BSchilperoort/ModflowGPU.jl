# Test suite of ModflowGPU.
#
# By default everything runs on the CPU backend (this is what `Pkg.test()` and CI do).
# Set MFGPU_TEST_BACKEND to run on a GPU backend as well; GPU tests run only locally:
#   MFGPU_TEST_BACKEND=rocm julia --project=test test/runtests.jl
# Set MFGPU_TEST_GROUPS to a comma-separated list of test groups (see TEST_GROUPS below) to
# run only those; unset runs everything:
#   MFGPU_TEST_GROUPS=connectivity,npf julia --project=test test/runtests.jl
# test/gpu/runtests.jl is the GPU entry: the same suite with MFGPU_TEST_BACKEND (default
# rocm); every parity test then runs on the GPU, and the run and assembly parity tests also
# compare it with the CPU backend (PLAN.md section 4).
using Test
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

# Test groups, in run order: group name => test files. A parity file shares the group
# name of its unit counterpart. The helpers above are loaded regardless of selection.
const TEST_GROUPS = [
    "backend" => ["unit/backend_test.jl"],
    "precision" => ["unit/precision_test.jl"],
    "api" => ["unit/api_test.jl"],
    "kernel" => ["unit/kernel_test.jl"],
    "input_tokenizer" => ["unit/input_tokenizer_test.jl"],
    "input_blocks" => ["unit/input_blocks_test.jl"],
    "input_examples" => ["unit/input_examples_test.jl"],
    "input_arrays" => ["unit/input_arrays_test.jl"],
    "input_lists" => ["unit/input_lists_test.jl"],
    "timeseries" => ["unit/timeseries_test.jl"],
    "output" => ["unit/output_test.jl"],
    "simulation_loader" => ["unit/simulation_loader_test.jl"],
    "connectivity" => ["unit/connectivity_test.jl", "parity/connectivity_test.jl"],
    "disv_disu" => ["unit/disv_disu_test.jl"],
    "npf" => ["unit/npf_test.jl", "parity/npf_test.jl"],
    "boundaries" => ["unit/boundaries_test.jl", "parity/boundaries_test.jl"],
    "sto" => ["unit/sto_test.jl", "parity/sto_test.jl"],
    "convertible" => ["unit/convertible_test.jl"],
    "riv_drn_evt" => ["unit/riv_drn_evt_test.jl", "parity/riv_drn_evt_test.jl"],
    "assembly" => ["parity/assembly_test.jl"],
    "krylov" => ["unit/krylov_test.jl", "unit/precond_test.jl"],
    "amg" => ["unit/amg_test.jl"],
    "picard" => ["unit/picard_test.jl"],
    "run" => ["parity/run_test.jl"],
    "stepping" => ["unit/stepping_test.jl"],
    "builder" => ["unit/builder_test.jl"],
    "synthetic" => ["unit/synthetic_test.jl", "parity/synthetic_test.jl"],
    # GPU only (skips with a message on other backends): no `double` in :single kernels.
    "kernel_hygiene" => ["gpu/kernel_hygiene_test.jl"],
]

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

@testset "ModflowGPU" begin
    for (group, files) in TEST_GROUPS
        group in SELECTED_GROUPS || continue
        foreach(include, files)
    end
    # The GPU backend tests belong to the "backend" group.
    if !(TEST_BACKEND isa CPU) && "backend" in SELECTED_GROUPS
        include("gpu/backend_gpu_test.jl")
    end
end
