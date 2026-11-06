# Adapted from mGV/src/backend_setup.jl (same names: `select_backend`, `register_backend!`,
# `backend_options`). The ordered list of known backends is a vector of pairs instead of an
# `OrderedDict`, so ModflowGPU does not need OrderedCollections.

"""
GPU backends are provided by package extensions (see `ext/`), which are loaded when the
corresponding GPU package is loaded alongside ModflowGPU (e.g. `using AMDGPU, ModflowGPU`).
Each extension registers its backend through [`register_backend!`](@ref) in its `__init__`.
"""
struct BackendEntry
    package::String              # package that provides the backend, e.g. "AMDGPU"
    is_functional::Base.Callable # () -> Bool
    construct::Base.Callable     # () -> KernelAbstractions.Backend
end

# Known backend names and the package providing them, in order of preference for "auto".
const KNOWN_BACKENDS = [
    "cuda" => "CUDA",
    "rocm" => "AMDGPU",
    "metal" => "Metal",
    "oneapi" => "oneAPI",
    "opencl" => "OpenCL",
]

const BACKEND_REGISTRY = Dict{String, BackendEntry}()

"""
    register_backend!(name, package, is_functional, construct)

Register a GPU backend under `name`, so it can be selected with `select_backend(name)`.
Called by the package extensions on load.
"""
function register_backend!(name::AbstractString, package::AbstractString, is_functional, construct)
    BACKEND_REGISTRY[lowercase(name)] = BackendEntry(package, is_functional, construct)
    return nothing
end

"""
    backend_name(backend)

Name of the backend type, e.g. `"CPU"` or `"ROCBackend"`.
"""
backend_name(backend::Backend) = string(nameof(typeof(backend)))

# Package that provides the known backend `name`, or `nothing` if `name` is not known.
function known_backend_package(name::AbstractString)
    i = findfirst(p -> first(p) == name, KNOWN_BACKENDS)
    return isnothing(i) ? nothing : last(KNOWN_BACKENDS[i])
end

# GPU backend names in order of preference: known backends first, then any others registered.
gpu_backend_names() = unique([first.(KNOWN_BACKENDS); sort!(collect(keys(BACKEND_REGISTRY)))])

"""
    backend_options()

All valid backend names for [`select_backend`](@ref): `"auto"`, `"cpu"` and the GPU backends.
"""
backend_options() = ["auto"; "cpu"; gpu_backend_names()]

"""
    select_backend(name="auto")

Select the KernelAbstractions backend to run the model on. This is evaluated at
runtime (not during precompilation), so the same precompiled package can be used
on machines with and without a GPU.

`name` is one of `"auto"`, `"cpu"`, $(join(repr.(first.(KNOWN_BACKENDS)), ", ")).
GPU backends are only available once their package is loaded (e.g. `using AMDGPU`).
With `"auto"` the first loaded and functional GPU backend is used, falling back to
CPU. Explicitly requesting a GPU backend that is not loaded or not functional throws
an error.

A `KernelAbstractions.Backend` can also be passed directly, which is returned as is.
"""
select_backend(backend::Backend) = backend

function select_backend(name::AbstractString = "auto")
    name = lowercase(name)
    if name == "auto"
        for key in gpu_backend_names()
            entry = get(BACKEND_REGISTRY, key, nothing)
            if !isnothing(entry) && entry.is_functional()
                return entry.construct()
            end
        end
        return CPU()
    elseif name == "cpu"
        return CPU()
    elseif haskey(BACKEND_REGISTRY, name)
        entry = BACKEND_REGISTRY[name]
        entry.is_functional() || error(
            "Backend '$name' was requested, but $(entry.package) is not functional on this machine.",
        )
        return entry.construct()
    elseif !isnothing(known_backend_package(name))
        package = known_backend_package(name)
        error(
            "Backend '$name' requires $package.jl. Add it to your environment and load it " *
                "alongside ModflowGPU, e.g. `using $package, ModflowGPU`.",
        )
    else
        error("Unknown backend '$name'. Valid options are: $(join(backend_options(), ", ")).")
    end
end

"""
    pin_memory!(backend, arr)

Page-lock host memory to speed up device transfers. Does nothing unless a backend
extension adds a method for its backend.
"""
pin_memory!(::Backend, arr) = nothing
