# Runs only with MFGPU_TEST_BACKEND set to a GPU backend (never in CI).
@testset "GPU backend $(ModflowGPU.backend_name(TEST_BACKEND))" begin
    name = TEST_BACKEND_NAME
    @test !(TEST_BACKEND isa CPU)
    @test haskey(ModflowGPU.BACKEND_REGISTRY, name)
    @test ModflowGPU.select_backend(name) isa typeof(TEST_BACKEND)
    @test ModflowGPU.select_backend("auto") isa typeof(TEST_BACKEND)
    @test ModflowGPU.resolve_precision(TEST_BACKEND, :auto).T ==
        (ModflowGPU.supports_float64(TEST_BACKEND) ? Float64 : Float32)
end
