"""
    ModflowGPU

A GPU port of the MODFLOW 6 groundwater flow (GWF) model, written with KernelAbstractions
so it runs on the CPU and on any GPU backend with a package extension (see `ext/`).
"""
module ModflowGPU

using KernelAbstractions: KernelAbstractions, Backend, CPU, get_backend, @kernel, @index, @Const, @localmem, @synchronize
using Printf: @sprintf, @printf
using Adapt: Adapt, adapt
using LinearAlgebra: LinearAlgebra

include("backend_setup.jl")
include("precision.jl")
include("instrument.jl")

# MF6 input (host only)
include("input/tokenizer.jl")
include("input/dfn/schema.jl")
include("input/blocks.jl")
include("input/arrays.jl")
include("input/lists.jl")
include("input/timeseries.jl")
include("input/options.jl")
include("input/supported.jl")

# Host-side model description (plain structs)
include("spec/grid.jl")
include("spec/packages.jl")
include("spec/simulation.jl")

# MF6 simulation loader (input files → SimulationSpec)
include("input/simulation.jl")

# Discretization and connectivity (host only)
include("discretization/connectivity.jl")
include("discretization/dis.jl")
include("discretization/disv.jl")
include("discretization/disu.jl")

# In-memory model construction and the MF6 input writer (host, T3.2)
include("builder.jl")
include("mf6_writer.jl")

# GWF engine setup (host, Float64)
include("gwf/offsets.jl")
include("gwf/npf.jl")
include("gwf/ic.jl")
include("gwf/sto.jl")

# Boundary packages: host period data, KA formulate/add kernels
include("boundaries/framework.jl")
include("boundaries/chd.jl")
include("boundaries/wel.jl")
include("boundaries/rch.jl")
include("boundaries/ghb.jl")
include("boundaries/riv.jl")
include("boundaries/drn.jl")
include("boundaries/evt.jl")

# Assembly and residual (KA kernels)
include("solver/csr.jl")
include("solver/precond/precond.jl")
include("solver/krylov.jl")
include("solver/precond/amg.jl")
include("solver/precond/chebyshev.jl")
include("solver/precond/multicolor.jl")

# Time stepping, the model on a device, outer iterations, flows and budgets
include("timing.jl")
include("gwf/model.jl")
include("solver/underrelax.jl")
include("solver/picard.jl")
include("budget.jl")

# MF6 output files (host only): heads, budgets, binary grid files, listing budgets
include("output/binary.jl")
include("output/grb.jl")
include("output/layout.jl")
include("output/hds.jl")
include("output/cbc.jl")
include("output/listing.jl")
include("output/oc.jl")

# Running a simulation
include("stepping.jl")
include("run.jl")
include("bmi.jl")

end
