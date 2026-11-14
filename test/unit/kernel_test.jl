# Launches a KA kernel on TEST_BACKEND in each precision (CPU in CI, which is POCL since
# KA 0.10; a GPU with MFGPU_TEST_BACKEND) and compares with a host computation.
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

@testset "Kernel launch on $(ModflowGPU.backend_name(TEST_BACKEND))" begin
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
        @test Array(y) ≈ p.T(2) .* x_host .+ y_host
    end
end
