module AMDGPUExt

using AMDGPU: AMDGPU, ROCBackend
import ModflowGPU

# Host memory registered with HIP and wrapped as a ROCArray: reductions write their
# results straight into host memory (visible after the stream is synchronised).
function ModflowGPU.host_mapped_zeros(::ROCBackend, ::Type{T}, n::Integer) where {T}
    host = zeros(T, n)
    # AMDGPU >= 2.11 wraps an Array (keeping it alive, unregistering it when freed); older
    # versions (2.8, the version mGV allows) only wrap a pointer, which registers the
    # memory for the rest of the session (the caller keeps `host` alive).
    dev = applicable(unsafe_wrap, AMDGPU.ROCArray, host) ? unsafe_wrap(AMDGPU.ROCArray, host) :
        unsafe_wrap(AMDGPU.ROCArray, pointer(host), size(host))
    return (dev, host)
end

function __init__()
    return ModflowGPU.register_backend!("rocm", "AMDGPU", AMDGPU.functional, ROCBackend)
end

end
