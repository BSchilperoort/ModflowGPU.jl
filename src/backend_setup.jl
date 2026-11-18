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
    ModflowGPU.register_backend!(name, package, is_functional, construct)

Register a GPU backend under `name` (lower case, e.g. `"rocm"`), so that
[`select_backend(name)`](@ref select_backend) and `select_backend("auto")` can find it.
`package` is the name of the GPU package (e.g. `"AMDGPU"`), `is_functional` a function
`() -> Bool` that tells whether the GPU can be used on this machine, and `construct` a
function `() -> KernelAbstractions.Backend`. Called by each package extension in its
`__init__`. A later registration under the same name replaces the earlier one.

# Example
The registration of `ext/AMDGPUExt.jl`:
```julia
function __init__()
    return ModflowGPU.register_backend!("rocm", "AMDGPU", AMDGPU.functional, ROCBackend)
end
```
"""
function register_backend!(name::AbstractString, package::AbstractString, is_functional, construct)
    BACKEND_REGISTRY[lowercase(name)] = BackendEntry(package, is_functional, construct)
    return nothing
end

"""
    backend_name(backend) -> String

The name of the type of the KernelAbstractions `backend`, e.g. `"POCLBackend"` (KA's
`CPU()`), `"ROCBackend"` (AMDGPU) or `"CUDABackend"`. Used in log messages and the
listing file.

# Example
```jldoctest
julia> backend_name(select_backend("cpu"))
"POCLBackend"
```
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
    backend_options() -> Vector{String}

All valid backend names for [`select_backend`](@ref): `"auto"`, `"cpu"` and the GPU
backends, known ones in order of preference for `"auto"` first. A GPU backend is
listed even when its package is not loaded; selecting it then throws an error that names
the package to load.

# Example
```jldoctest
julia> backend_options()
7-element Vector{String}:
 "auto"
 "cpu"
 "cuda"
 "rocm"
 "metal"
 "oneapi"
 "opencl"
```
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

Returns a `KernelAbstractions.Backend`, to pass as the `backend` keyword of
[`run_simulation`](@ref) and [`initialize`](@ref). The CPU backend of KernelAbstractions
0.10 is `POCLBackend` (`CPU()`); it is meant for tests and small models.

# Example
```jldoctest
julia> backend = select_backend("cpu");

julia> backend_name(backend)
"POCLBackend"
```
With a GPU package loaded:
```julia
using AMDGPU, ModflowGPU
backend = select_backend("auto")   # ROCBackend() when the GPU works, else the CPU
```
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
    ModflowGPU.pin_memory!(backend, arr)

Extension hook. Page-lock the host array `arr` to speed up transfers between it and the
device. Returns `nothing` when nothing was
done (the default, unless a backend extension adds a method for its backend), otherwise a
handle the caller keeps for as long as the memory should stay pinned (CUDA: the array;
ROCm: a registered `ROCArray` wrapper). Copies from pinned memory may run
asynchronously: synchronize before overwriting it.

A backend extension adds a method for its backend type.

# Example
```julia
# in ext/CUDAExt.jl
ModflowGPU.pin_memory!(::CUDABackend, arr) = CUDA.pin(arr)
```
"""
pin_memory!(::Backend, arr) = nothing

"""
    ModflowGPU.free_memory!(backend, arr)

Extension hook. Release the device memory of the device array `arr` now, instead of when
the garbage collector finalises it (two models on one GPU, [`finalize_simulation`](@ref)).
`arr` must not be used
afterwards. The default calls `KernelAbstractions.unsafe_free!`, which GPU packages
implement for their arrays (a no-op for host arrays); extensions add a method where the
backend needs something else. Memory shared with other arrays (`reshape`, `unsafe_wrap`
of the same buffer) is released when the last of them is freed. Returns `nothing`.
"""
free_memory!(::Backend, arr::AbstractArray) = (KernelAbstractions.unsafe_free!(arr); nothing)

"""
    ModflowGPU.free_device_arrays!(x) -> Int

Free ([`ModflowGPU.free_memory!`](@ref)) every device array reachable from `x` through
struct fields, array elements and dictionary entries (host arrays of bits types are
skipped, wrappers such as views are followed to their parent). Returns the number of
arrays freed. Nothing reachable from `x` may use those arrays afterwards. Used by
[`finalize_simulation`](@ref), and by coupling code to free its own device buffers before
another model allocates.

# Example
```julia
buffers = (; recharge = KernelAbstractions.zeros(backend, Float32, n))
ModflowGPU.free_device_arrays!(buffers)   # 1 (0 on the CPU backend)
```
"""
function free_device_arrays!(x)
    seen = IdDict{Any, Nothing}()
    return _free_walk!(x, seen)
end

_free_skip(x) = isbits(x) || x isa Union{Module, Type, Function, IO, AbstractString, Symbol, Task, Core.MethodInstance}

function _free_walk!(x, seen::IdDict{Any, Nothing})
    _free_skip(x) && return 0
    haskey(seen, x) && return 0
    seen[x] = nothing
    if x isa Array
        isbitstype(eltype(x)) && return 0
        n = 0
        for i in eachindex(x)
            isassigned(x, i) && (n += _free_walk!(x[i], seen))
        end
        return n
    elseif x isa AbstractArray && parent(x) === x
        backend = try
            get_backend(x)
        catch
            nothing
        end
        if !isnothing(backend) && !(backend isa CPU)
            free_memory!(backend, x)
            return 1
        end
    elseif x isa AbstractDict
        n = 0
        for (k, v) in x
            n += _free_walk!(k, seen) + _free_walk!(v, seen)
        end
        return n
    end
    n = 0
    for f in 1:nfields(x)
        isdefined(x, f) && (n += _free_walk!(getfield(x, f), seen))
    end
    return n
end

"""
    ModflowGPU.host_mapped_zeros(backend, T, n) -> (device, host) or nothing

A zeroed host vector `host` of `n` values of type `T` and a device array `device` that
aliases its memory (kernels write the host memory directly), or `nothing` when the
backend cannot map host memory (callers then copy with `copyto!`). Used for the small
results of device reductions, which the host reads after every reduction: on ROCm this
saves the device-to-host copy (~30 µs on the RX 7800 XT, docs/dev/precision.md). Extension
hook: the CPU backend returns the same `Array` twice, ROCm a registered host array;
other backends return `nothing` unless their extension adds a method.

# Example
```jldoctest
julia> d, h = ModflowGPU.host_mapped_zeros(select_backend("cpu"), Float64, 2);

julia> d === h
true
```
"""
host_mapped_zeros(::Backend, ::Type, ::Integer) = nothing
host_mapped_zeros(::CPU, ::Type{T}, n::Integer) where {T} = (a = zeros(T, n); (a, a))

# A new array on `backend` with the contents of the host array `x` (`copyto!`, no slicing).
function _device_copy(backend::Backend, x::AbstractArray{T}) where {T}
    d = KernelAbstractions.allocate(backend, T, size(x))
    copyto!(d, x)
    return d
end

# Work-group size of the package's element-wise kernels (T4.5). A static size keeps KA from
# tuning the work group at every launch, and its launch mode from depending on the problem
# size (KA 0.10 launches dynamically sized kernels over large ranges as `LinearLaunch`,
# which compiled the host launch code anew for the first large model of a session).
const _WG = 256
