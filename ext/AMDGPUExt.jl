module AMDGPUExt

using AMDGPU: AMDGPU, ROCBackend
import ModflowGPU

function __init__()
    return ModflowGPU.register_backend!("rocm", "AMDGPU", AMDGPU.functional, ROCBackend)
end

end
