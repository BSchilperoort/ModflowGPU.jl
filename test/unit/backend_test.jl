# Ported from mGV/test/backend_test.jl.
@testset "Backend selection" begin
    if TEST_BACKEND isa CPU
        @test isempty(ModflowGPU.BACKEND_REGISTRY)  # no GPU package loaded
        @test ModflowGPU.select_backend() isa CPU
        @test_throws "requires CUDA.jl" ModflowGPU.select_backend("cuda")
        @test_throws "requires AMDGPU.jl" ModflowGPU.select_backend("rocm")
    end
    @test ModflowGPU.select_backend("CPU") isa CPU
    @test ModflowGPU.select_backend(CPU()) isa CPU
    @test ModflowGPU.select_backend(TEST_BACKEND) === TEST_BACKEND
    @test_throws "Unknown backend 'foo'" ModflowGPU.select_backend("foo")
    @test ModflowGPU.backend_options() == ["auto", "cpu", "cuda", "rocm", "metal", "oneapi", "opencl"]
    @test ModflowGPU.backend_name(CPU()) in ("CPU", "POCLBackend")  # KA 0.9, KA 0.10

    # Fake registrations to test the dispatch on registered backends. Restore the registry
    # afterwards, so a real GPU backend registered by an extension stays available.
    saved = copy(ModflowGPU.BACKEND_REGISTRY)
    constructed = String[]
    try
        empty!(ModflowGPU.BACKEND_REGISTRY)
        ModflowGPU.register_backend!("rocm", "AMDGPU", () -> false, () -> (push!(constructed, "rocm"); CPU()))
        ModflowGPU.register_backend!("metal", "Metal", () -> true, () -> (push!(constructed, "metal"); CPU()))
        ModflowGPU.register_backend!("OpenCL", "OpenCL", () -> true, () -> (push!(constructed, "opencl"); CPU()))
        ModflowGPU.register_backend!("other", "Other", () -> true, () -> (push!(constructed, "other"); CPU()))

        @test_throws "AMDGPU is not functional" ModflowGPU.select_backend("rocm")
        # "auto" skips the non-functional rocm backend and prefers metal over opencl
        ModflowGPU.select_backend("auto")
        @test constructed == ["metal"]
        ModflowGPU.select_backend("OpenCL")
        @test constructed == ["metal", "opencl"]
        # backends that are not known in advance can be registered and are listed last
        ModflowGPU.select_backend("other")
        @test constructed == ["metal", "opencl", "other"]
        @test ModflowGPU.backend_options()[end] == "other"
    finally
        empty!(ModflowGPU.BACKEND_REGISTRY)
        merge!(ModflowGPU.BACKEND_REGISTRY, saved)
    end
end
