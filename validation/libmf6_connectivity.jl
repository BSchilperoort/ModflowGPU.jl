# Dump libmf6's discretization, connectivity and NPF arrays of an MF6 simulation (task
# cards T2.1, T2.2), for the oracle tests (test/parity/connectivity_test.jl,
# test/parity/npf_test.jl).
#
# Initialises the simulation through ModflowInterface.jl (no time step is run) and writes,
# for every GWF model, the arrays below as raw little-endian files `<MODEL>/<NAME>.bin`
# (Int32 or Float64), plus `index.toml` listing model names, each array's type and length
# and the scalars NODES, NODESUSER, NJA, NJAS:
#
#   <MODEL>/CON/{IA,JA,JAS,ISYM,IHC,CL1,CL2,HWVA,ANGLEX,IAUSR,JAUSR}
#   <MODEL>/DIS/{NODEUSER,NODEREDUCED,TOP,BOT,AREA,XC,YC}
#   <MODEL>/NPF/{CONDSAT,K11,K22,K33,ANGLE1,ANGLE2,ANGLE3,ICELLTYPE}, <MODEL>/IC/STRT
#
# The NPF and IC arrays are optional: they are written only when libmf6 lists them with a
# non-zero length (MF6 allocates CONDSAT with length 0 under XT3D, and ANGLE1/2/3 with
# length 0 when they are not given). NPF arrays are after `npf_ar`, i.e. by reduced node,
# with K22/K33 filled in (K where not given, times K with K22OVERK/K33OVERK) and the angles
# in radians.
#
# (DIS/NODEUSER and DIS/NODEREDUCED are dummies of length 1 when no cell is excluded, and
# CON/IAUSR, CON/JAUSR then equal IA, JA.)
#
# Which libmf6 is loaded is decided by the Julia environment (docs/dev/reference_tooling.md):
#
#   julia --project=reference/julia-env/mf6.8.2 validation/libmf6_connectivity.jl <simdir> <outdir>
#
# The simulation is copied to <outdir>/model (MF6 writes its listing and .grb files there);
# <outdir> must not exist. One simulation per process: libmf6 keeps global state.

using ModflowInterface
import ModflowInterface as MF
import BasicModelInterface as BMI
import Modflow6_jll
using TOML

const CON_ARRAYS = ("IA", "JA", "JAS", "ISYM", "IHC", "CL1", "CL2", "HWVA", "ANGLEX", "IAUSR", "JAUSR")
const DIS_ARRAYS = ("NODEUSER", "NODEREDUCED", "TOP", "BOT", "AREA", "XC", "YC")
const OPTIONAL_ARRAYS = (
    ("NPF", ("CONDSAT", "K11", "K22", "K33", "ANGLE1", "ANGLE2", "ANGLE3", "ICELLTYPE")),
    ("IC", ("STRT",)),
)
const DIS_SCALARS = ("NODES", "NODESUSER", "NJA", "NJAS")

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
    index = Dict{String, Any}("libmf6" => Modflow6_jll.libmf6_path, "models" => models)
    for model in models
        mkpath(joinpath(outdir, model))
        info = Dict{String, Any}()
        for (comp, list) in (("CON", CON_ARRAYS), ("DIS", DIS_ARRAYS))
            for a in list
                addr = "$model/$comp/$a"
                v = BMI.get_value_ptr(mf, addr)
                write(joinpath(outdir, model, "$(comp)_$a.bin"), collect(vec(v)))
                info["$comp/$a"] = Dict("type" => string(eltype(v)), "length" => length(v))
            end
        end
        for (comp, list) in OPTIONAL_ARRAYS, a in list
            addr = "$model/$comp/$a"
            (addr in names && BMI.get_var_type(mf, addr) != "" && BMI.get_var_nbytes(mf, addr) > 0) || continue
            v = BMI.get_value_ptr(mf, addr)
            write(joinpath(outdir, model, "$(comp)_$a.bin"), collect(vec(v)))
            info["$comp/$a"] = Dict("type" => string(eltype(v)), "length" => length(v))
        end
        for s in DIS_SCALARS
            info[s] = Int(only(BMI.get_value_ptr(mf, "$model/DIS/$s")))
        end
        index[model] = info
    end
    BMI.finalize(mf)
    open(io -> TOML.print(io, index; sorted = true), joinpath(outdir, "index.toml"), "w")
    return index
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: julia --project=<env> libmf6_connectivity.jl <simdir> <outdir>")
    main(ARGS[1], ARGS[2])
end
