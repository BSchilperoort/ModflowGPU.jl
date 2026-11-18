# Collect the parity table of the manual (docs/src/compatibility/results.md): run the
# end-to-end parity comparisons of test/parity/ (run_test.jl and synthetic_test.jl, with
# their helpers in test/run_parity.jl) on every backend given and in every precision preset,
# and write docs/src/assets/parity.toml. Run it on a machine with the reference runs
# (validation/fetch_reference.sh, validation/run_reference.jl) and mf6; CI does not
# (no GPU, no reference data). The toml is committed.
#
#   REFERENCE_DIR=/path/to/reference julia --project=test docs/parity/collect.jl [backends...]
#
# Backends default to `cpu rocm`. Each backend runs the comparisons in a module of its own
# with `TEST_BACKEND` set to it, so the test helpers are used unchanged; on a GPU backend
# they also compare with the CPU backend. A model whose comparison fails is written with
# `passed = false`. Runtime on the RX 7800 XT machine: see the `wall_time` of the toml.
using Test
using TOML
using Dates
using ModflowGPU
using KernelAbstractions: CPU

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const TESTDIR = joinpath(ROOT, "test")
# PARITY_OUT: another output file; PARITY_ONLY: comma-separated model names (for trying
# the script; the committed toml has every model).
const OUTFILE = get(ENV, "PARITY_OUT", joinpath(ROOT, "docs", "src", "assets", "parity.toml"))
const ONLY = filter(!isempty, strip.(split(get(ENV, "PARITY_ONLY", ""), ',')))
const BACKENDS = isempty(ARGS) ? ["cpu", "rocm"] : lowercase.(ARGS)

for b in BACKENDS
    pkg = ModflowGPU.known_backend_package(b)
    b == "cpu" || isnothing(pkg) || @eval using $(Symbol(pkg))
end

include(joinpath(ROOT, "benchmark", "synthetic.jl"))  # SyntheticModels

"A module with the test helpers of test/ and `TEST_BACKEND = select_backend(name)`."
function helper_module(name)
    mod = Module(Symbol("Parity_", name))
    backend = ModflowGPU.select_backend(name)
    Core.eval(
        mod, quote
            using Test, ModflowGPU
            using KernelAbstractions: KernelAbstractions, CPU
            const TEST_BACKEND = $backend
        end
    )
    pkg = ModflowGPU.known_backend_package(name)
    if name != "cpu" && !isnothing(pkg)
        getfield(Main, Symbol(pkg)).allowscalar(false)
    end
    for f in ("testutils.jl", "input_files.jl", "run_parity.jl")
        Base.include(mod, joinpath(TESTDIR, f))
    end
    return mod
end

# --- Model description (from the loaded input)

function grid_text(d)
    d isa ModflowGPU.DisSpec && return "DIS $(d.nlay)×$(d.nrow)×$(d.ncol)"
    d isa ModflowGPU.DisvSpec && return "DISV $(d.nlay)×$(d.ncpl)"
    return "DISU $(d.nodes)"
end

function describe(rundir)
    sim = ModflowGPU.load_simulation(rundir)
    m = only(sim.models)
    ftypes = [replace(p.ftype, r"6$" => "") for p in m.packages]
    pkgs = String[]
    for t in unique(ftypes)
        t in ("DIS", "DISV", "DISU", "IC", "OC", "OBS") && continue
        n = count(==(t), ftypes)
        push!(pkgs, n > 1 ? "$t ×$n" : t)
    end
    nts = length(ModflowGPU.time_steps(sim.tdis))
    active = count(>(0), ModflowGPU.idomain_user(m.dis))
    return Dict{String, Any}(
        "grid" => grid_text(m.dis), "active_cells" => active, "time_steps" => nts,
        "packages" => pkgs, "newton" => m.newton,
    )
end

# --- Running one model on one backend

"Run `_run_parity` in a testset; returns (stats, passed)."
function run_one(mod, label, rundir, opts)
    stats = Ref{Any}(nothing)
    passed = true
    try
        @testset "$label" begin
            stats[] = mod._run_parity(label, rundir; opts...)
        end
    catch err
        err isa Test.TestSetException || rethrow()
        passed = false
    end
    return stats[], passed
end

r3(x) = round(x; sigdigits = 3)

# One preset on one backend; the fields of `_run_parity` (test/run_parity.jl).
function preset_record(s)
    return Dict{String, Any}(
        "max_dh" => r3(s.dh), "max_budget_cbc" => r3(s.dq), "max_budget_listing" => r3(s.dl),
        "max_percent_discrepancy" => r3(s.pd), "outer" => s.outer, "inner" => s.inner,
        "mf6_outer" => s.mf6_outer, "mf6_inner" => s.mf6_inner, "wall_time" => round(s.time; digits = 2),
        (isnan(s.backend_vs_cpu) ? () : ("max_dh_vs_cpu" => r3(s.backend_vs_cpu),))...,
    )
end

"Every model of the run and synthetic parity tests: (name, kind, setup) with setup(f)."
function model_list(mod)
    out = Tuple{String, String, Any}[]
    for (name, source) in mod.run_parity_models()
        kind = source isa AbstractString ? "mf6 example" : "test model"
        push!(out, (name, kind, f -> mod.with_parity_rundir(f, name, source)))
    end
    for preset in mod.SYNTHETIC_PARITY_MODELS
        setup = function (f)
            return mktempdir() do dir
                SyntheticModels.write_synthetic(dir, Symbol(preset))
                read(Cmd(`$(mod.RUN_MF6)`; dir), String)
                f(dir)
            end
        end
        push!(out, ("synthetic $preset", "synthetic coupling model", setup))
    end
    return out
end

function main()
    t0 = time()
    rows = Dict{String, Dict{String, Any}}()
    order = String[]
    for (b, mod) in zip(BACKENDS, MODULES)
        @info "Parity collection on backend $b"
        for (name, kind, setup) in model_list(mod)
            isempty(ONLY) || name in ONLY || continue
            opts = startswith(name, "synthetic") ? (;) : mod.run_parity_options(name)
            setup() do rundir
                row = get!(rows, name) do
                    push!(order, name)
                    merge(describe(rundir), Dict{String, Any}("name" => name, "kind" => kind, "passed" => true, "results" => Dict{String, Any}()))
                end
                stats, passed = run_one(mod, name, rundir, opts)
                row["passed"] &= passed
                isnothing(stats) && return nothing
                res = Dict{String, Any}(string(p) => preset_record(stats[p]) for p in (:single, :mixed, :double) if haskey(stats, p))
                res["single_vs_double"] = r3(stats[:single_vs_double])
                pc = get(opts, :preconditioner, :auto)
                spc = get(opts, :single_preconditioner, pc)
                res["preconditioner"] = string(pc)
                res["single_preconditioner"] = string(spc)
                row["results"][b] = res
                return nothing
            end
        end
    end
    commit = strip(read(`git -C $ROOT rev-parse --short HEAD`, String))
    dirty = !isempty(strip(read(`git -C $ROOT status --porcelain --untracked-files=no`, String)))
    doc = Dict{String, Any}(
        "date" => string(Dates.today()),
        "commit" => commit * (dirty ? "-dirty" : ""),
        "mf6_version" => "6.8.2",
        "backends" => BACKENDS,
        "presets" => ["single", "mixed", "double"],
        "wall_time" => round(time() - t0; digits = 1),
        "examples" => [rows[n] for n in order],
    )
    mkpath(dirname(OUTFILE))
    open(OUTFILE, "w") do io
        println(io, "# Generated by docs/parity/collect.jl; do not edit by hand.")
        TOML.print(io, doc; sorted = true)
    end
    nfail = count(r -> !r["passed"], values(rows))
    @info "Wrote $OUTFILE: $(length(order)) model(s), $nfail failed, $(round(time() - t0; digits = 1)) s"
    return nfail
end

# Built at top level, so that `main` runs in a world that has their methods.
const MODULES = [helper_module(b) for b in BACKENDS]

exit(main() == 0 ? 0 : 1)
