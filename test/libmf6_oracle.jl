# libmf6 as a test oracle, and a writer for small MF6 simulations made in tests.
#
# libmf6 runs in a subprocess in its own Julia environment (validation/setup_libmf6_envs.sh,
# docs/dev/reference_tooling.md), so ModflowInterface and Modflow6_jll are not test
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

_grid_ftype(::ModflowGPU.DisSpec) = "DIS6"
_grid_ftype(::ModflowGPU.DisvSpec) = "DISV6"
_grid_ftype(::ModflowGPU.DisuSpec) = "DISU6"

function _write_grid(io, dis::ModflowGPU.DisSpec)
    println(io, "BEGIN options\n  LENGTH_UNITS  meters\nEND options")
    println(io, "BEGIN dimensions\n  NLAY  $(dis.nlay)\n  NROW  $(dis.nrow)\n  NCOL  $(dis.ncol)\nEND dimensions")
    println(io, "BEGIN griddata")
    _write_internal(io, "delr", dis.delr)
    _write_internal(io, "delc", dis.delc)
    _write_internal(io, "top", dis.top)
    _write_internal(io, "botm", dis.botm)
    _write_internal(io, "idomain", dis.idomain)
    println(io, "END griddata")
    return nothing
end

# VERTICES and CELL2D blocks (DISV, DISU with vertices).
function _write_vertices(io, dis)
    println(io, "BEGIN vertices")
    for v in 1:size(dis.vertices, 2)
        println(io, "  ", v, " ", _mf6_real(dis.vertices[1, v]), " ", _mf6_real(dis.vertices[2, v]))
    end
    println(io, "END vertices\nBEGIN cell2d")
    for (c, iv) in enumerate(dis.cell_vertices)
        println(io, "  ", c, " ", _mf6_real(dis.cell_center[1, c]), " ", _mf6_real(dis.cell_center[2, c]), " ", length(iv), " ", join(iv, " "))
    end
    println(io, "END cell2d")
    return nothing
end

function _write_grid(io, dis::ModflowGPU.DisvSpec)
    println(io, "BEGIN options\n  LENGTH_UNITS  meters\nEND options")
    println(io, "BEGIN dimensions\n  NLAY  $(dis.nlay)\n  NCPL  $(dis.ncpl)\n  NVERT  $(dis.nvert)\nEND dimensions")
    println(io, "BEGIN griddata")
    _write_internal(io, "top", dis.top)
    _write_internal(io, "botm", dis.botm)
    _write_internal(io, "idomain", dis.idomain)
    println(io, "END griddata")
    _write_vertices(io, dis)
    return nothing
end

function _write_grid(io, dis::ModflowGPU.DisuSpec)
    println(io, "BEGIN options\n  LENGTH_UNITS  meters\nEND options")
    println(io, "BEGIN dimensions\n  NODES  $(dis.nodes)\n  NJA  $(dis.nja)\n  NVERT  $(dis.nvert)\nEND dimensions")
    println(io, "BEGIN griddata")
    _write_internal(io, "top", dis.top)
    _write_internal(io, "bot", dis.bot)
    _write_internal(io, "area", dis.area)
    _write_internal(io, "idomain", dis.idomain)
    println(io, "END griddata\nBEGIN connectiondata")
    _write_internal(io, "iac", dis.iac)
    _write_internal(io, "ja", dis.ja)
    _write_internal(io, "ihc", dis.ihc)
    _write_internal(io, "cl12", dis.cl12)
    _write_internal(io, "hwva", dis.hwva)
    isnothing(dis.angldegx) || _write_internal(io, "angldegx", dis.angldegx)
    println(io, "END connectiondata")
    dis.nvert > 0 && _write_vertices(io, dis)
    return nothing
end

"""
    write_dis_simulation(dir, dis; name = "synth", npf_options = String[],
                         npf_arrays = Dict("icelltype" => 0, "k" => 1.0), strt = 0.0)

Write a one-model MF6 simulation on the grid `dis` (a `DisSpec`, `DisvSpec` or
`DisuSpec`; the file is `<name>.dis` for all three) to `dir`: mfsim.nam, TDIS (one
steady period), IMS, and a GWF model with the grid (INTERNAL arrays, reals written so
that they read back exactly), NPF and IC. No boundary packages: enough for libmf6 to
initialise, not to solve.

NPF: `npf_options` are lines of the OPTIONS block (`"ALTERNATIVE_CELL_AVERAGING
LOGARITHMIC"`, ...), `npf_arrays` maps GRIDDATA names to a constant or an array of the
grid shape (written INTERNAL, in the order `icelltype`, `k`, `k22`, `k33`, `angle1`,
`angle2`, `angle3`, `wetdry`). IC STRT is a constant or an array.
"""
function write_dis_simulation(
        dir::AbstractString, dis; name::AbstractString = "synth", npf_options = String[],
        npf_arrays = Dict{String, Any}("icelltype" => 0, "k" => 1.0), strt = 0.0,
    )
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
          $(_grid_ftype(dis))  $name.dis  dis
          NPF6  $name.npf  npf
          IC6  $name.ic  ic
        END packages
        """,
    )
    open(io -> _write_grid(io, dis), joinpath(dir, "$name.dis"), "w")
    _write_value(io, key, v) = v isa AbstractArray ? _write_internal(io, key, v) :
        println(io, "  ", key, "\n    CONSTANT  ", v isa Integer ? string(v) : _mf6_real(v))
    open(joinpath(dir, "$name.npf"), "w") do io
        println(io, "BEGIN options")
        foreach(o -> println(io, "  ", o), npf_options)
        println(io, "END options\nBEGIN griddata")
        for key in ("icelltype", "k", "k22", "k33", "angle1", "angle2", "angle3", "wetdry")
            haskey(npf_arrays, key) && _write_value(io, key, npf_arrays[key])
        end
        println(io, "END griddata")
    end
    open(joinpath(dir, "$name.ic"), "w") do io
        println(io, "BEGIN options\nEND options\nBEGIN griddata")
        _write_value(io, "strt", strt)
        println(io, "END griddata")
    end
    return dir
end

# libmf6's system after the first `solve` call of `simdir` (validation/libmf6_first_solve.jl):
# SLN_1 `ia`, `ja`, `amat`, `rhs`, the heads `x0` it was assembled with and `delt`. Used by
# the assembly oracle tests (groups assembly and xt3d).
function _libmf6_first_solve(simdir)
    return mktempdir() do tmp
        outdir = joinpath(tmp, "out")
        run_libmf6_script("libmf6_first_solve.jl", simdir, outdir)
        index = TOML.parsefile(joinpath(outdir, "index.toml"))
        rd(a, T) = (v = Vector{T}(undef, index[a]["length"]); read!(joinpath(outdir, "$a.bin"), v); v)
        x0 = Vector{Float64}(undef, index["X"]["length"])
        read!(joinpath(outdir, "X0.bin"), x0)
        (; ia = rd("IA", Int32), ja = rd("JA", Int32), amat = rd("AMAT", Float64), rhs = rd("RHS", Float64), x0, delt = index["delt"])
    end
end
