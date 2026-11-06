module CUDAExt

using CUDA: CUDA, CUDABackend
import ModflowGPU

ModflowGPU.pin_memory!(::CUDABackend, arr) = CUDA.pin(arr)

function __init__()
    return ModflowGPU.register_backend!("cuda", "CUDA", CUDA.functional, CUDABackend)
end

end
