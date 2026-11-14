module AMDGPUExt

using AMDGPU: AMDGPU, ROCBackend
import ModflowGPU

# Host memory registered with HIP and wrapped as a ROCArray: reductions write their
# results straight into host memory (visible after the stream is synchronised). The
# wrapper keeps the array alive and unregisters it once the device is done with it.
function ModflowGPU.host_mapped_zeros(::ROCBackend, ::Type{T}, n::Integer) where {T}
    host = zeros(T, n)
    return (unsafe_wrap(AMDGPU.ROCArray, host), host)
end

# Page-lock a host array by registering it with HIP (the wrapper keeps the registration
# and the array alive; copies from the array then use DMA without a staging copy).
ModflowGPU.pin_memory!(::ROCBackend, arr::Array) = unsafe_wrap(AMDGPU.ROCArray, arr)

# Return the buffer to the stream-ordered pool now (`finalize`; wrapped host memory is
# unregistered).
ModflowGPU.free_memory!(::ROCBackend, arr::AMDGPU.ROCArray) = (AMDGPU.unsafe_free!(arr); nothing)

function __init__()
    return ModflowGPU.register_backend!("rocm", "AMDGPU", AMDGPU.functional, ROCBackend)
end

end
