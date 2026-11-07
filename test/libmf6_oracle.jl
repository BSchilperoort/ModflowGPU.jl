# libmf6 as a test oracle, and a writer for small MF6 simulations made in tests.
#
# libmf6 runs in a subprocess in its own Julia environment (validation/setup_libmf6_envs.sh,
# docs/reference_tooling.md), so ModflowInterface and Modflow6_jll are not test
# dependencies and one libmf6 instance runs per process. The subprocess runs a script of
# validation/ (e.g. libmf6_connectivity.jl) that initialises the simulation and dumps
# arrays to raw files, which the test reads back. Expects `REFERENCE_DIR` (input_files.jl).

using TOML: TOML

"""
Julia environment that loads libmf6 6.8.2: `MFGPU_LIBMF6_ENV`, default
`<REFERENCE_DIR>/julia-env/mf6.8.2`.
"""
const LIBMF6_ENV = get(ENV, "MFGPU_LIBMF6_ENV", joinpath(REFERENCE_DIR, "julia-env", "mf6.8.2"))

"""
    libmf6_available() -> Bool

Whether the libmf6 6.8.2 environment was set up (validation/setup_libmf6_envs.sh).
"""
libmf6_available() = isfile(joinpath(LIBMF6_ENV, ".setup-complete")) && isfile(joinpath(LIBMF6_ENV, "LocalPreferences.toml"))

"""
    run_libmf6_script(script, simdir, outdir)

Run `validation/<script> <simdir> <outdir>` with the libmf6 environment in a fresh Julia
process (the environment variables that select the test environment are removed). Throws
with the subprocess output when it fails.
"""
function run_libmf6_script(script::AbstractString, simdir::AbstractString, outdir::AbstractString)
    path = joinpath(@__DIR__, "..", "validation", script)
    env = copy(ENV)
    for k in ("JULIA_LOAD_PATH", "JULIA_PROJECT", "JULIA_NUM_THREADS")
        delete!(env, k)
    end
    julia = first(Base.julia_cmd().exec)
    cmd = setenv(`$julia --startup-file=no --project=$LIBMF6_ENV $path $simdir $outdir`, env)
    out = IOBuffer()
    ok = success(pipeline(cmd; stdout = out, stderr = out))
    ok || error("libmf6 subprocess failed: $cmd\n" * String(take!(out)))
    return nothing
end

"""
    libmf6_connectivity(simdir) -> Dict{String, Dict{String, Any}}

libmf6's discretization arrays of every GWF model of the simulation in `simdir`, after
initialisation (validation/libmf6_connectivity.jl): `model => Dict("CON/IA" => ..., ...,
"DIS/TOP" => ..., "NODES" => ...)` with upper-case model names.
"""
function libmf6_connectivity(simdir::AbstractString)
    return mktempdir() do tmp
        outdir = joinpath(tmp, "out")
        run_libmf6_script("libmf6_connectivity.jl", simdir, outdir)
        index = TOML.parsefile(joinpath(outdir, "index.toml"))
        result = Dict{String, Dict{String, Any}}()
        for model in index["models"]
            arrays = Dict{String, Any}()
            for (name, info) in index[model]
                if info isa Dict
                    T = info["type"] == "Int32" ? Int32 : info["type"] == "Float64" ? Float64 : error("type $(info["type"])")
                    v = Vector{T}(undef, info["length"])
                    read!(joinpath(outdir, model, replace(name, "/" => "_") * ".bin"), v)
                    arrays[name] = v
                else
                    arrays[name] = info
                end
            end
            result[model] = arrays
        end
        result
    end
end

# A Float64 as MF6 reads it back exactly (shortest round-trip representation).
_mf6_real(x::Real) = repr(Float64(x))

function _write_internal(io, name, a)
    println(io, "  ", name)
    println(io, "    INTERNAL")
    # MF6 reads a 2D/3D array one row (first dimension) per Fortran READ, so every row
    # starts on a new line (a row may continue over several lines).
    for row in Iterators.partition(vec(a), size(a, 1)), chunk in Iterators.partition(row, 10)
        println(io, "      ", join(map(x -> x isa Integer ? string(x) : _mf6_real(x), chunk), " "))
    end
    return nothing
end

"""
    write_dis_simulation(dir, dis::ModflowGPU.DisSpec; name = "synth")

Write a one-model MF6 simulation on the grid `dis` to `dir`: mfsim.nam, TDIS (one
steady period), IMS, and a GWF model with DIS (INTERNAL arrays, reals written so that
they read back exactly), a confined NPF with K = 1 and IC. No boundary packages: enough
for libmf6 to initialise, not to solve.
"""
function write_dis_simulation(dir::AbstractString, dis; name::AbstractString = "synth")
    mkpath(dir)
    write(
        joinpath(dir, "mfsim.nam"), """
        BEGIN options
        END options
        BEGIN timing
          TDIS6  sim.tdis
        END timing
        BEGIN models
          gwf6  $name.nam  $name
        END models
        BEGIN exchanges
        END exchanges
        BEGIN solutiongroup  1
          ims6  sim.ims  $name
        END solutiongroup  1
        """,
    )
    write(
        joinpath(dir, "sim.tdis"), """
        BEGIN options
          TIME_UNITS  days
        END options
        BEGIN dimensions
          NPER  1
        END dimensions
        BEGIN perioddata
          1.0  1  1.0
        END perioddata
        """,
    )
    write(
        joinpath(dir, "sim.ims"), """
        BEGIN options
        END options
        BEGIN nonlinear
          OUTER_DVCLOSE  1.0e-6
          OUTER_MAXIMUM  10
        END nonlinear
        BEGIN linear
          INNER_MAXIMUM  50
          INNER_DVCLOSE  1.0e-6
          INNER_RCLOSE  1.0e-6
        END linear
        """,
    )
    write(
        joinpath(dir, "$name.nam"), """
        BEGIN options
        END options
        BEGIN packages
          DIS6  $name.dis  dis
          NPF6  $name.npf  npf
          IC6  $name.ic  ic
        END packages
        """,
    )
    open(joinpath(dir, "$name.dis"), "w") do io
        println(io, "BEGIN options\n  LENGTH_UNITS  meters\nEND options")
        println(io, "BEGIN dimensions\n  NLAY  $(dis.nlay)\n  NROW  $(dis.nrow)\n  NCOL  $(dis.ncol)\nEND dimensions")
        println(io, "BEGIN griddata")
        _write_internal(io, "delr", dis.delr)
        _write_internal(io, "delc", dis.delc)
        _write_internal(io, "top", dis.top)
        _write_internal(io, "botm", dis.botm)
        _write_internal(io, "idomain", dis.idomain)
        println(io, "END griddata")
    end
    write(
        joinpath(dir, "$name.npf"), """
        BEGIN options
        END options
        BEGIN griddata
          icelltype
            CONSTANT  0
          k
            CONSTANT  1.0
        END griddata
        """,
    )
    write(
        joinpath(dir, "$name.ic"), """
        BEGIN options
        END options
        BEGIN griddata
          strt
            CONSTANT  0.0
        END griddata
        """,
    )
    return dir
end
