module OpenCLExt

using OpenCL: OpenCL, OpenCLBackend, cl
import ModflowGPU

# OpenCL.jl has no `functional()`; check whether a device can be used instead.
function opencl_functional()
    try
        OpenCL.zeros(Float32, 1)
        return true
    catch
        return false
    end
end

# Float64 is an optional OpenCL extension; query the current device.
ModflowGPU.supports_float64(::OpenCLBackend) = "cl_khr_fp64" in cl.device().extensions

function __init__()
    return ModflowGPU.register_backend!("opencl", "OpenCL", opencl_functional, OpenCLBackend)
end

end
