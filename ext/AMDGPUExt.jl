module AMDGPUExt

using AMDGPU: AMDGPU, ROCBackend
import ModflowGPU

# Host memory registered with HIP and wrapped as a ROCArray: reductions write their
# results straight into host memory (visible after the stream is synchronised).
function ModflowGPU.host_mapped_zeros(::ROCBackend, ::Type{T}, n::Integer) where {T}
    host = zeros(T, n)
    return (unsafe_wrap(AMDGPU.ROCArray, host), host)
end

function __init__()
    return ModflowGPU.register_backend!("rocm", "AMDGPU", AMDGPU.functional, ROCBackend)
end

end
