# Run validation/bmi_script.jl through ModflowInterface.jl and libmf6 (task card T8.10),
# for the comparison in test/unit/bmi_test.jl, and serialize what every call returned:
#
#   julia --project=reference/julia-env/mf6.8.2 validation/libmf6_bmi.jl <simdir> <outdir>
#
# The simulation is copied to <outdir>/model (libmf6 writes its output there); the results
# are in <outdir>/result.jls (Serialization, read by the same Julia version). One
# simulation per process: libmf6 keeps global state.

using ModflowInterface
import ModflowInterface as MF
import BasicModelInterface as BMI
using Serialization: serialize

include(joinpath(@__DIR__, "bmi_script.jl"))

function main(simdir, outdir)
    ispath(outdir) && error("output directory exists: $outdir")
    mkpath(outdir)
    workdir = joinpath(outdir, "model")
    cp(simdir, workdir)
    r = cd(() -> bmi_script(MF, ModflowModel, workdir), workdir)
    serialize(joinpath(outdir, "result.jls"), r)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: julia --project=<env> libmf6_bmi.jl <simdir> <outdir>")
    main(ARGS[1], ARGS[2])
end
