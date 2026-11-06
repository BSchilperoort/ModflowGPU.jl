# Runs only with MFGPU_TEST_BACKEND set to a GPU backend (never in CI).
using KernelAbstractions: @kernel, @index, @Const

@kernel function axpy_kernel!(y, a, @Const(x))
    i = @index(Global)
    @inbounds y[i] = a * x[i] + y[i]
end

"""
    axpy!(y, a, x)

`y .= a .* x .+ y` with a KernelAbstractions kernel on the backend of `y`.
"""
function axpy!(y, a, x)
    backend = KernelAbstractions.get_backend(y)
    axpy_kernel!(backend)(y, a, x; ndrange = length(y))
    KernelAbstractions.synchronize(backend)
    return y
end

@testset "GPU backend $(ModflowGPU.backend_name(TEST_BACKEND))" begin
    name = TEST_BACKEND_NAME
    @test !(TEST_BACKEND isa CPU)
    @test haskey(ModflowGPU.BACKEND_REGISTRY, name)
    @test ModflowGPU.select_backend(name) isa typeof(TEST_BACKEND)
    @test ModflowGPU.select_backend("auto") isa typeof(TEST_BACKEND)
    @test ModflowGPU.resolve_precision(TEST_BACKEND, :auto).T ==
        (ModflowGPU.supports_float64(TEST_BACKEND) ? Float64 : Float32)

    # A kernel runs on the device in each precision and matches the CPU result.
    for_each_precision() do p
        n = 1000
        x_host = rand(p.T, n)
        y_host = rand(p.T, n)
        x = KernelAbstractions.allocate(TEST_BACKEND, p.T, n)
        y = KernelAbstractions.allocate(TEST_BACKEND, p.T, n)
        copyto!(x, x_host)
        copyto!(y, y_host)
        axpy!(y, p.T(2), x)
        @test eltype(y) == p.T
        @test Array(y) ≈ axpy!(copy(y_host), p.T(2), x_host)
    end
end
