"""
    ModflowGPU

The groundwater-flow (GWF) model of MODFLOW 6 on GPUs and CPUs, written with
KernelAbstractions. It reads MF6 input files, writes MF6 head, budget and listing files,
and can be run one time step at a time for coupling.

Load the package of your GPU next to it to enable that backend (`using AMDGPU,
ModflowGPU`; also CUDA, oneAPI, Metal, OpenCL). The exported names cover:

- running a simulation: [`run_simulation`](@ref), [`load_simulation`](@ref);
- backends and precision: [`select_backend`](@ref), [`resolve_precision`](@ref);
- the stepping API: [`initialize`](@ref), [`do_time_step`](@ref),
  [`finalize_simulation`](@ref), [`get_value`](@ref), [`set_value`](@ref), ...;
- building models in Julia: [`build_dis`](@ref), ..., [`build_simulation`](@ref),
  [`write_simulation`](@ref);
- reading output files: [`open_head_file`](@ref), [`open_budget_file`](@ref),
  [`read_grb`](@ref), [`read_listing_budgets`](@ref).

Other documented names are public but not exported, because their names are generic
(`ModflowGPU.run`, `ModflowGPU.finalize`, `ModflowGPU.solve`, `ModflowGPU.update`) or they are for backend
authors (`ModflowGPU.register_backend!`). BMI is implemented as methods of
BasicModelInterface's functions (`BasicModelInterface.update(sim)`).

# Example
```julia
using ModflowGPU
result = run_simulation("path/to/simulation"; backend = select_backend("auto"))
result.converged
```
"""
module ModflowGPU

# Public API (docs/src/api/). Exported: the names a user calls. `public` (Julia >= 1.11;
# the package needs 1.12): documented names that stay qualified, because they clash with
# Base (`run`, `finalize`), CommonSolve (`solve`) or CUDA (`update`), are exception or component types of
# the exported ones, or are extension hooks for backend authors. test/unit/api_test.jl
# checks that this list matches the @docs blocks of docs/src/api/.

# Running
export run_simulation, RunResult, ModelBudget, BudgetTerm, cum_percent_discrepancy,
    ConvergenceError, load_simulation, MF6UnsupportedError
public run, MF6InputError
# Backends and precision
export select_backend, backend_options, backend_name, supports_float64, resolve_precision
public PRECISION_PRESETS
# Stepping
export Simulation, StepResult, initialize, prepare_time_step, prepare_solve, finalize_solve,
    finalize_time_step, do_time_step, finalize_simulation, get_value, get_value_ptr,
    set_value, get_var_address, var_addresses, input_addresses, get_current_time,
    get_start_time, get_end_time, get_time_step
public solve, update, finalize, device_bound!
# Builder
export SimulationSpec, build_dis, build_npf, build_ic, build_sto, oc_record, build_oc,
    build_boundary, build_array_boundary, build_model, build_tdis, build_ims, build_simulation,
    write_simulation
# Output
export open_head_file, HeadFile, read_head, find_head_step, head_kstpkper, head_times,
    open_budget_file, BudgetFile, BudgetList, read_budget, find_budget_records, budget_texts,
    budget_kstpkper, budget_times, read_grb, BinaryGridFile, read_listing_budgets,
    ListingBudget, listing_rates, percent_discrepancy
public HeadStep, BudgetRecord, GrbVariable, ListingBudgetTerm, MF6BinaryFileError, HNOFLO, HDRY
# For backend authors
public register_backend!, pin_memory!, free_memory!, host_mapped_zeros, free_device_arrays!

using KernelAbstractions: KernelAbstractions, Backend, CPU, get_backend, @kernel, @index, @Const, @localmem, @synchronize
using Printf: @sprintf, @printf
using Adapt: Adapt, adapt
using Atomix: Atomix
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
include("gwf/convertible.jl")
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
