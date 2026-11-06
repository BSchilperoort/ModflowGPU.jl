# Probe which memory addresses a libmf6 build exposes through BMI (task card T0.2).
#
# Runs an MF6 simulation through ModflowInterface.jl (the same API ModflowCoupling uses),
# lists every BMI variable address, checks the solution / connectivity addresses that
# ModflowGPU's parity tests need, reads them with `get_value_ptr` after the first `solve`
# call, and then runs the simulation to the end.
#
# Which libmf6 is loaded is decided by the Julia environment, not by this script:
# the environments made by `validation/setup_libmf6_envs.sh` use either Modflow6_jll's
# own 6.3.0 library or the 6.8.2 release library through a `libmf6_path` preference
# (see docs/reference_tooling.md).
#
# Usage:
#   julia --project=<env> validation/libmf6_probe.jl <example dir> <output dir>
#
# The example directory is copied to <output dir>/model (which must not exist) and run
# there; the full address list goes to <output dir>/var_names.txt and the probe result
# to <output dir>/probe.toml.

using ModflowInterface
import ModflowInterface as MF
import BasicModelInterface as BMI
import Modflow6_jll
using TOML
using Libdl

"Version string from the XMI `get_version` call; libmf6 6.3 does not export it."
function libmf6_version()
    f = Libdl.dlsym_e(Modflow6_jll.libmf6_handle, :get_version)
    f == C_NULL && return "unknown (no get_version symbol)"
    buf = zeros(UInt8, 256)
    ccall(f, Cint, (Ptr{UInt8},), buf) == 0 || return "unknown"
    return MF.trimmed_string(buf)
end

"Type, shape and a short summary of the value at `addr`, or the error."
function probe(mf, addr)
    info = Dict{String, Any}("address" => addr)
    try
        type = BMI.get_var_type(mf, addr)
        isempty(type) && error("get_var_type returned nothing (address unknown)")
        info["type"] = type
        info["shape"] = collect(MF.get_var_shape(mf, addr))
        v = BMI.get_value_ptr(mf, addr)
        info["length"] = length(v)
        if !isempty(v)
            info["min"] = minimum(v)
            info["max"] = maximum(v)
            info["nonzeros"] = count(!iszero, v)
            info["first"] = collect(v[1:min(5, end)])
        end
        info["ok"] = true
    catch err
        info["ok"] = false
        info["error"] = sprint(showerror, err)
    end
    return info
end

function main(example, outdir)
    isdir(example) || error("not a directory: $example")
    ispath(outdir) && error("output directory exists: $outdir")
    mkpath(outdir)
    workdir = joinpath(outdir, "model")
    cp(example, workdir)

    mf = ModflowModel(workdir)
    result = Dict{String, Any}(
        "example" => basename(normpath(example)),
        "libmf6" => Modflow6_jll.libmf6_path,
        "version" => libmf6_version(),
        "BMI_LENVARADDRESS" => Int(MF.BMI_LENVARADDRESS),
    )
    println("libmf6: ", result["libmf6"], " (version ", result["version"], ")")
    BMI.initialize(mf)

    innames = BMI.get_input_var_names(mf)
    outnames = BMI.get_output_var_names(mf)
    result["n_input_var_names"] = length(innames)
    result["n_output_var_names"] = length(outnames)
    result["input_equals_output"] = innames == outnames
    open(joinpath(outdir, "var_names.txt"), "w") do io
        foreach(n -> println(io, n), innames)
    end

    # Model name: the component that owns the CON (connectivity) arrays.
    con = filter(n -> endswith(n, "/CON/IA"), innames)
    isempty(con) && error("no <MODEL>/CON/IA in the variable list")
    model = split(first(con), "/")[1]
    result["model"] = model
    # Solution-level arrays live under SLN_1 or a sub-path of it; list them all.
    result["sln_names"] = sort!(filter(n -> startswith(n, "SLN_1/"), innames))

    wanted_init = [
        "SLN_1/IA", "SLN_1/JA", "SLN_1/X", "SLN_1/RHS", "SLN_1/AMAT",
        "$model/CON/IA", "$model/CON/JA", "$model/CON/JAS", "$model/CON/IHC",
        "$model/CON/CL1", "$model/CON/CL2", "$model/CON/HWVA",
        "$model/NPF/CONDSAT", "$model/NPF/K11", "$model/X", "$model/DIS/NODEUSER",
    ]
    # Also probe whatever the list shows for the matrix in this version.
    for n in result["sln_names"]
        any(s -> endswith(n, s), ("/AMAT", "/IA", "/JA", "/RHS", "/X")) && push!(wanted_init, n)
    end
    unique!(wanted_init)
    for a in wanted_init
        result["listed:" * a] = a in innames
    end
    result["after_initialize"] = [probe(mf, a) for a in wanted_init]

    # First time step, first outer iteration: the matrix and RHS are assembled in solve.
    MF.prepare_time_step(mf, 0.0)
    MF.prepare_solve(mf, 1)
    converged = MF.solve(mf, 1)
    result["after_first_solve"] = [probe(mf, a) for a in wanted_init]
    # Raw little-endian dumps (Int32 / Float64) for cross-version comparison.
    for a in ("SLN_1/IA", "SLN_1/JA", "SLN_1/AMAT", "SLN_1/RHS", "SLN_1/X")
        write(joinpath(outdir, "first_solve_" * replace(a, "/" => "_") * ".bin"), BMI.get_value_ptr(mf, a))
    end

    # Finish the simulation.
    nouter = 1
    maxiter = 1000
    while !converged && nouter < maxiter
        converged = MF.solve(mf, 1)
        nouter += 1
    end
    MF.finalize_solve(mf, 1)
    MF.finalize_time_step(mf)
    nsteps = 1
    while BMI.get_current_time(mf) < BMI.get_end_time(mf)
        BMI.update(mf)
        nsteps += 1
    end
    head = BMI.get_value_ptr(mf, "$model/X")
    write(joinpath(outdir, "final_X.bin"), head)
    result["run"] = Dict(
        "first_step_outer_iterations" => nouter,
        "first_step_converged" => converged,
        "time_steps" => nsteps,
        "end_time" => BMI.get_end_time(mf),
        "head_min" => minimum(head),
        "head_max" => maximum(head),
    )
    BMI.finalize(mf)

    open(io -> TOML.print(io, result; sorted = true), joinpath(outdir, "probe.toml"), "w")
    for (stage, list) in (("after_initialize", result["after_initialize"]), ("after_first_solve", result["after_first_solve"]))
        println("--- $stage")
        for p in list
            status = p["ok"] ? "ok   len=$(p["length"]) $(p["type"])" * (haskey(p, "min") ? " range=[$(p["min"]), $(p["max"])] nnz=$(p["nonzeros"])" : "") : "FAIL $(p["error"])"
            println(rpad(p["address"], 28), status)
        end
    end
    println("--- run: ", result["run"])
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: julia --project=<env> libmf6_probe.jl <example dir> <output dir>")
    main(ARGS[1], ARGS[2])
end
