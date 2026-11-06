# Test suite of ModflowGPU.
#
# By default everything runs on the CPU backend (this is what `Pkg.test()` and CI do).
# Set MFGPU_TEST_BACKEND to run on a GPU backend as well; GPU tests run only locally:
#   MFGPU_TEST_BACKEND=rocm julia --project=test test/runtests.jl
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

@testset "ModflowGPU" begin
    include("unit/backend_test.jl")
    include("unit/precision_test.jl")
    include("unit/kernel_test.jl")
    include("unit/input_tokenizer_test.jl")
    include("unit/input_blocks_test.jl")
    include("unit/input_examples_test.jl")
    if !(TEST_BACKEND isa CPU)
        include("gpu/backend_gpu_test.jl")
    end
end
