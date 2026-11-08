module AMDGPUExt

using AMDGPU: AMDGPU, ROCBackend
import ModflowGPU

# Host memory registered with HIP and wrapped as a ROCArray: reductions write their
# results straight into host memory (visible after the stream is synchronised).
# AMDGPU >= 2.10 wraps an Array: the wrapper keeps it alive and unregisters it once the
# device is done with it. Older versions (2.8, the version mGV allows) only wrap a pointer
# and never unregister it, so freed GC memory would stay registered and a later array at
# the same address (possibly of another size) would reuse the stale registration; there
# the reductions copy their results instead (`nothing`).
function ModflowGPU.host_mapped_zeros(::ROCBackend, ::Type{T}, n::Integer) where {T}
    host = zeros(T, n)
    applicable(unsafe_wrap, AMDGPU.ROCArray, host) || return nothing
    return (unsafe_wrap(AMDGPU.ROCArray, host), host)
end

# Page-lock a host array by registering it with HIP (the wrapper keeps the registration
# and the array alive; copies from the array then use DMA without a staging copy). Not on
# AMDGPU < 2.10, for the reason above.
function ModflowGPU.pin_memory!(::ROCBackend, arr::Array)
    applicable(unsafe_wrap, AMDGPU.ROCArray, arr) || return nothing
    return unsafe_wrap(AMDGPU.ROCArray, arr)
end

function __init__()
    return ModflowGPU.register_backend!("rocm", "AMDGPU", AMDGPU.functional, ROCBackend)
end

end
