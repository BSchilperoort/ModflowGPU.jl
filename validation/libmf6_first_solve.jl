# Dump libmf6's solution system after the first outer iteration of an MF6 simulation
# (task card T2.6, the assembly oracle test test/parity/assembly_test.jl).
#
# Initialises the simulation through ModflowInterface.jl, runs `prepare_time_step`,
# `prepare_solve` and one `solve` call of solution 1 (the first outer iteration of the
# first time step), and writes raw little-endian files to <outdir>:
#
#   X0.bin    SLN_1/X before the solve (the heads the system is assembled with: the
#             starting heads with the CHD heads of `chd_ad`), Float64
#   IA.bin, JA.bin             SLN_1/IA, SLN_1/JA (Int32)
#   AMAT.bin, RHS.bin, X.bin   SLN_1/AMAT, SLN_1/RHS and SLN_1/X after the solve (Float64)
#
# plus `index.toml` with the lengths, the time step length `delt` and the model names.
# AMAT and RHS after the solve hold the system that iteration assembled, after the
# adjustments of `sln_ls` (docs/reference_tooling.md, section 3); parity runs use no IMS
# scaling.
#
#   julia --project=reference/julia-env/mf6.8.2 validation/libmf6_first_solve.jl <simdir> <outdir>
#
# The simulation is copied to <outdir>/model; <outdir> must not exist. One simulation per
# process: libmf6 keeps global state.

using ModflowInterface
import ModflowInterface as MF
import BasicModelInterface as BMI
using TOML

function main(simdir, outdir)
    isdir(simdir) || error("not a directory: $simdir")
    ispath(outdir) && error("output directory exists: $outdir")
    mkpath(outdir)
    workdir = joinpath(outdir, "model")
    cp(simdir, workdir)

    mf = ModflowModel(workdir)
    BMI.initialize(mf)
    names = BMI.get_input_var_names(mf)
    models = sort!([String(split(n, "/")[1]) for n in names if endswith(n, "/CON/IA") && count('/', n) == 2])
    MF.prepare_time_step(mf, 0.0)
    delt = BMI.get_time_step(mf)
    MF.prepare_solve(mf, 1)
    write(joinpath(outdir, "X0.bin"), collect(vec(BMI.get_value_ptr(mf, "SLN_1/X"))))
    MF.solve(mf, 1)
    index = Dict{String, Any}("models" => models, "delt" => Float64(delt))
    for a in ("IA", "JA", "AMAT", "RHS", "X")
        v = collect(vec(BMI.get_value_ptr(mf, "SLN_1/$a")))
        write(joinpath(outdir, "$a.bin"), v)
        index[a] = Dict("type" => string(eltype(v)), "length" => length(v))
    end
    BMI.finalize(mf)
    open(io -> TOML.print(io, index; sorted = true), joinpath(outdir, "index.toml"), "w")
    return index
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: julia --project=<env> libmf6_first_solve.jl <simdir> <outdir>")
    main(ARGS[1], ARGS[2])
end
