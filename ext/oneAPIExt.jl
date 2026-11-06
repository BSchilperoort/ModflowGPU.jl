module oneAPIExt

using oneAPI: oneAPI, oneAPIBackend, oneL0
import ModflowGPU

# Intel GPUs (e.g. Arc) may lack Float64; query the Level Zero module flags of the device.
function ModflowGPU.supports_float64(::oneAPIBackend)
    flags = oneL0.module_properties(oneAPI.device()).flags
    return flags & oneL0.ZE_DEVICE_MODULE_FLAG_FP64 != 0
end

function __init__()
    return ModflowGPU.register_backend!("oneapi", "oneAPI", oneAPI.functional, oneAPIBackend)
end

end
