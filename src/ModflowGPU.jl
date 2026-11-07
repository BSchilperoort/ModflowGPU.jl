"""
    ModflowGPU

A GPU port of the MODFLOW 6 groundwater flow (GWF) model, written with KernelAbstractions
so it runs on the CPU and on any GPU backend with a package extension (see `ext/`).
"""
module ModflowGPU

using KernelAbstractions: KernelAbstractions, Backend, CPU, get_backend

include("backend_setup.jl")
include("precision.jl")

# MF6 input (host only)
include("input/tokenizer.jl")
include("input/dfn/schema.jl")
include("input/blocks.jl")
include("input/arrays.jl")
include("input/lists.jl")
include("input/options.jl")
include("input/supported.jl")

# Host-side model description (plain structs)
include("spec/grid.jl")
include("spec/packages.jl")
include("spec/simulation.jl")

# MF6 simulation loader (input files → SimulationSpec)
include("input/simulation.jl")

end
