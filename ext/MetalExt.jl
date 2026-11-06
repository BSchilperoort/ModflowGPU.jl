module MetalExt

using Metal: Metal, MetalBackend
import ModflowGPU

# Apple GPUs have no Float64 (PLAN.md section 3.7).
ModflowGPU.supports_float64(::MetalBackend) = false

function __init__()
    return ModflowGPU.register_backend!("metal", "Metal", Metal.functional, MetalBackend)
end

end
