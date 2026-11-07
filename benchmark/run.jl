# Benchmark harness (T4.1): the synthetic coupling models of benchmark/synthetic.jl
# (`indus`, `continental`, `global5m`), a steady-state step followed by daily transient
# steps, on a ModflowGPU backend in `:double` and `:single`, or with mf6 (serial).
#
#     julia --project=benchmark benchmark/run.jl rocm [presets...] [options]
#     JULIA_KA_CPU_THREADS=8 julia -t 8 --project=benchmark benchmark/run.jl cpu indus
#     REFERENCE_DIR=reference julia --project=benchmark benchmark/run.jl mf6 indus continental
#
# Options (`key=value`): `precision=double,single` (presets, comma-separated), `steps=30`
# (daily transient steps after the steady state), `timed=1` (also a timed run with the
# section timer; `0` skips it), `ims=parity` (`synthetic_ims` kind), `log=<file>` (append
# the Markdown table rows there), `maxsteady=<s>` (skip the transient steps of a case whose
# steady step took longer than this, e.g. on the CPU backend).
#
# Per case (preset × precision) the harness does:
#
# 1. An untimed run (no section timer: the engine as in production): wall time per time
#    step (`update`, then `synchronize`), outer and inner iterations, and the device pool's
#    high-water mark (ROCm: `hipMemPoolAttrUsedMemHigh`, reset before `initialize`).
# 2. SpMV in isolation on the assembled matrix of the last step (`ModflowGPU.spmv!`, 3
#    warm-up and 30 timed calls, median), with the T0.4 "min" byte model:
#    `nja·(sizeof(T) + 4) + (n + 1)·4 + n·sizeof(Ts) (x) + n·sizeof(Ts) (y)`.
# 3. A timed run: a `SyncTimer` installed with `ModflowGPU.set_section_timer!`
#    synchronises the device at each section boundary and records the sections in a
#    TimerOutputs tree. The synchronisations cost time, so the split comes from this run
#    and the wall times from run 1; the overhead is reported. In-solver SpMV GB/s: the
#    `spmv_dot` sections (fused CSR SpMV + dot product, CG: `q = A·p`, `p·q`) with the
#    bytes `nja·(sizeof(T) + 4) + (n + 1)·4 + 4n·sizeof(Ts)` (p, the column and row scale
#    factors, q).
#
# Time split categories (leaf sections of src/; see src/instrument.jl):
# assembly (`assembly`: boundary formulation, CSR assembly, flux residual), SpMV
# (`spmv_dot`, `true_residual`), preconditioner (`precond_setup`: scaling and Jacobi setup;
# the Jacobi application is fused into the update kernels), reductions and vector updates
# (`update_reduce`: axpy + Jacobi + 4 reductions in one kernel, `vector_update`,
# `outer_reduction`), host transfers (`period_data`, `mirrors`, `output`), other
# (`underrelax`, `start_timestep`, `flows`, `budget`, and the host time between sections).

include(joinpath(@__DIR__, "synthetic.jl"))
using .SyntheticModels
using .SyntheticModels: synthetic_ims
using ModflowGPU
using KernelAbstractions: KernelAbstractions as KA
using TimerOutputs
using Printf
using Statistics: median
using Dates

# ---------------------------------------------------------------------------------------
# Section timer: synchronise, then time with TimerOutputs
# ---------------------------------------------------------------------------------------

"""
    SyncTimer(backend)

Section timer for `ModflowGPU.set_section_timer!`: synchronises `backend` at each section
boundary and records the section in the TimerOutputs tree `to`.
"""
struct SyncTimer{B}
    backend::B
    to::TimerOutput
    stack::Vector{Any}
    labels::Dict{Symbol, String}
end
SyncTimer(backend) = SyncTimer(backend, TimerOutput(), Any[], Dict{Symbol, String}())

function ModflowGPU.section_begin(t::SyncTimer, name::Symbol)
    KA.synchronize(t.backend)
    label = get!(() -> String(name), t.labels, name)
    push!(t.stack, TimerOutputs.begin_timed_section!(t.to, label))
    return nothing
end

function ModflowGPU.section_end(t::SyncTimer, ::Symbol)
    KA.synchronize(t.backend)
    TimerOutputs.end_timed_section!(t.to, pop!(t.stack))
    return nothing
end

const CATEGORIES = (
    assembly = ("assembly",),
    spmv = ("spmv_dot", "true_residual"),
    precond = ("precond_setup",),
    reductions = ("update_reduce", "vector_update", "outer_reduction"),
    transfers = ("period_data", "mirrors", "output"),
)

# Seconds and calls of each leaf section name, summed over the tree.
function section_totals(to::TimerOutput)
    tot = Dict{String, Tuple{Float64, Int}}()
    function walk(t)
        for (k, c) in t.inner_timers
            s, n = get(tot, k, (0.0, 0))
            tot[k] = (s + TimerOutputs.time(c) / 1.0e9, n + TimerOutputs.ncalls(c))
            walk(c)
        end
        return
    end
    walk(to)
    return tot
end

# ---------------------------------------------------------------------------------------
# Device memory (ROCm: pool high-water mark; other backends: not measured)
# ---------------------------------------------------------------------------------------

const MEM = Ref{Any}((; reset = () -> nothing, peak = () -> 0, used = () -> 0))
mem_reset() = MEM[].reset()
mem_peak() = MEM[].peak()
mem_used() = MEM[].used()

function setup_rocm_memory!(AMDGPU)
    pool() = AMDGPU.Mem.pool_create(AMDGPU.device())
    high = AMDGPU.HIP.hipMemPoolAttrUsedMemHigh
    MEM[] = (;
        reset = () -> (AMDGPU.synchronize(); AMDGPU.HIP.attribute!(pool(), high, UInt64(0)); nothing),
        peak = () -> (AMDGPU.synchronize(); Int(AMDGPU.HIP.attribute(UInt64, pool(), high))),
        used = () -> (AMDGPU.synchronize(); AMDGPU.used_memory()),
    )
    return nothing
end

# ---------------------------------------------------------------------------------------
# One ModflowGPU case
# ---------------------------------------------------------------------------------------

_gib(b) = b / 2.0^30

function run_steps!(s, backend, nsteps; timer = nothing, maxsteady = Inf)
    rows = NamedTuple{(:time, :outer, :inner, :converged), Tuple{Float64, Int, Int, Bool}}[]
    for k in 1:(nsteps + 1)
        isnothing(timer) || ModflowGPU.section_begin(timer, k == 1 ? :steady_step : :transient_step)
        t0 = time_ns()
        ModflowGPU.update(s)
        KA.synchronize(backend)
        dt = (time_ns() - t0) / 1.0e9
        isnothing(timer) || ModflowGPU.section_end(timer, :step)
        r = last(s.results).outer
        push!(rows, (; time = dt, outer = r.iterations, inner = r.inner, converged = r.converged))
        r.converged || @warn "step $k did not converge ($(r.iterations) outer, $(r.inner) inner)"
        (k == 1 && dt > maxsteady) && (@info "steady step took $(round(dt; digits = 1)) s > maxsteady: skipping the transient steps"; break)
    end
    return rows
end

function spmv_bench(s, backend)
    m = s.model
    Ts = s.Ts
    n = Int(m.conn.nodes)
    x = KA.allocate(backend, Ts, n)
    y = KA.allocate(backend, Ts, n)
    copyto!(x, ones(Ts, n))
    for _ in 1:3
        ModflowGPU.spmv!(y, m.sys.amat, m.csr, x)
    end
    KA.synchronize(backend)
    ts = Float64[]
    for _ in 1:30
        t0 = time_ns()
        ModflowGPU.spmv!(y, m.sys.amat, m.csr, x)
        KA.synchronize(backend)
        push!(ts, (time_ns() - t0) / 1.0e9)
    end
    nja = length(m.sys.amat)
    bytes = nja * (sizeof(eltype(m.sys.amat)) + 4) + (n + 1) * 4 + 2n * sizeof(Ts)
    return (; t = median(ts), gbs = bytes / median(ts) / 1.0e9, bytes)
end

function run_case(preset, pre, backend, bname; nsteps, timed, ims, maxsteady)
    f = synthetic_fields(preset)
    spec = synthetic_simulation(preset; fields = f, output = false, ntransient = nsteps, ims = synthetic_ims(ims))
    c = synthetic_counts(f)
    GC.gc(true)
    mem_reset()
    m0 = mem_used()
    tinit = @elapsed s = ModflowGPU.initialize(spec; backend, precision = pre, write_output = false)
    minit = mem_used() - m0
    rows = run_steps!(s, backend, nsteps; maxsteady)
    peak = mem_peak()
    sp = spmv_bench(s, backend)
    T, Ts = s.T, s.Ts
    s = nothing
    split = nothing
    if timed
        GC.gc(true)
        st = ModflowGPU.initialize(spec; backend, precision = pre, write_output = false)
        timer = SyncTimer(backend)
        prev = ModflowGPU.set_section_timer!(timer)
        trows = try
            run_steps!(st, backend, length(rows) - 1; timer)
        finally
            ModflowGPU.set_section_timer!(prev)
        end
        tot = section_totals(timer.to)
        ttotal = sum(r.time for r in trows)
        cat = Dict(k => sum(get(tot, nm, (0.0, 0))[1] for nm in v) for (k, v) in pairs(CATEGORIES))
        cat[:other] = ttotal - sum(values(cat))
        sd = get(tot, "spmv_dot", (0.0, 0))
        n = c.nodes
        bdot = c.nja * (sizeof(T) + 4) + (n + 1) * 4 + 4n * sizeof(Ts)
        split = (; ttotal, cat, tot, spmv_dot_t = sd[1] / max(sd[2], 1), spmv_dot_gbs = sd[2] > 0 ? bdot * sd[2] / sd[1] / 1.0e9 : NaN, to = timer.to)
        st = nothing
    end
    return (; preset, pre, bname, nodes = c.nodes, nja = c.nja, tinit, minit, peak, rows, sp, split)
end

function report(r; io = stdout)
    st = r.rows[1]
    tr = r.rows[2:end]
    ntr = length(tr)
    tsum = sum(x -> x.time, tr; init = 0.0)
    oi = sum(x -> x.outer, tr; init = 0)
    ii = sum(x -> x.inner, tr; init = 0)
    conv = all(x -> x.converged, r.rows)
    us = 1.0e6 * st.time / max(st.inner, 1)
    trs = ntr == 0 ? "skipped | – | –" : @sprintf("%.3f | %.1f | %.1f", tsum / ntr, oi / ntr, ii / ntr)
    dot = isnothing(r.split) ? "–" : @sprintf("%.0f", r.split.spmv_dot_gbs)
    mem = r.peak > 0 ? @sprintf("%.3f", _gib(r.peak)) : "–"
    @printf(
        io, "| %s | %d | %s | `:%s` | %.2f | %.2f | %d | %d | %.0f | %s | %.2f | %.0f | %s | %s |%s\n",
        r.preset, r.nodes, r.bname, r.pre, r.tinit, st.time, st.outer, st.inner, us, trs, st.time + tsum,
        r.sp.gbs, dot, mem, conv ? "" : " NOT CONVERGED"
    )
    return nothing
end

function report_split(r; io = stdout)
    sp = r.split
    isnothing(sp) && return nothing
    untimed = sum(x -> x.time, r.rows)
    pct(k) = @sprintf("%.1f", 100 * sp.cat[k] / sp.ttotal)
    @printf(
        io, "| %s | %s | `:%s` | %.2f | %.2f (%+.0f %%) | %s | %s | %s | %s | %s | %s | %.0f |\n",
        r.preset, r.bname, r.pre, untimed, sp.ttotal, 100 * (sp.ttotal / untimed - 1), pct(:assembly), pct(:spmv), pct(:precond),
        pct(:reductions), pct(:transfers), pct(:other), 1.0e6 * sp.spmv_dot_t
    )
    return nothing
end

# ---------------------------------------------------------------------------------------
# mf6 (serial)
# ---------------------------------------------------------------------------------------

function mf6_case(preset, mf6exe, workdir; nsteps, ims)
    dir = mktempdir(workdir; prefix = "mf6_$(preset)_")
    tw = @elapsed spec = write_synthetic(dir, preset; maxcells = 10^8, output = false, ntransient = nsteps, ims = synthetic_ims(ims))
    c = synthetic_counts(spec)
    stamps = Float64[]
    t0 = time_ns()
    open(Cmd(`$mf6exe`; dir)) do io
        for line in eachline(io)
            occursin(r"Solving:\s+Stress period:", line) && push!(stamps, (time_ns() - t0) / 1.0e9)
        end
    end
    tend = (time_ns() - t0) / 1.0e9
    push!(stamps, tend)
    steps = diff(stamps)
    txt = read(joinpath(dir, "mfsim.lst"), String)
    its = [(parse(Int, m[1]), parse(Int, m[2])) for m in eachmatch(r"(\d+) CALLS TO NUMERICAL SOLUTION IN TIME STEP \d+ STRESS PERIOD \d+\s*\n\s*(\d+) TOTAL ITERATIONS", txt)]
    ok = occursin("Normal termination", txt)
    rm(dir; recursive = true)
    return (; preset, nodes = c.nodes, twrite = tw, tread = stamps[1], steps, its, tend, ok)
end

function report_mf6(r; io = stdout)
    st = r.steps[1]
    tr = r.steps[2:end]
    ntr = length(tr)
    o = r.its
    oi = sum(x -> x[1], o[2:end]; init = 0)
    ii = sum(x -> x[2], o[2:end]; init = 0)
    @printf(
        io, "| %s | %d | mf6 6.8.2 (serial, ILU0) | `:double` | %.2f | %.2f | %d | %d | %.0f | %.3f | %.1f | %.1f | %.2f | – | – | – |%s\n",
        r.preset, r.nodes, r.tread, st, o[1][1], o[1][2], 1.0e6 * st / max(o[1][2], 1), sum(tr) / ntr, oi / ntr, ii / ntr, r.tend - r.tread,
        r.ok ? "" : " FAILED"
    )
    return nothing
end

# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------

const HEADER = """
| Preset | nodes | engine | precision | init (s) | steady (s) | outer | inner | µs/inner | daily step (s) | outer/step | inner/step | total (s) | SpMV GB/s | in-solver GB/s | peak GiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"""
const SPLIT_HEADER = """
| Preset | engine | precision | untimed (s) | timed (s) | assembly % | SpMV % | precond % | reductions + updates % | host transfers % | other % | µs per spmv_dot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"""

function main(args)
    bname = isempty(args) ? "cpu" : args[1]
    opts = Dict(String(m[1]) => String(m[2]) for m in (match(r"^(\w+)=(.*)$", a) for a in args[2:end]) if !isnothing(m))
    presets = Symbol[Symbol(a) for a in args[2:end] if !occursin('=', a)]
    isempty(presets) && (presets = [:indus, :continental, :global5m])
    nsteps = parse(Int, get(opts, "steps", "30"))
    ims = Symbol(get(opts, "ims", "parity"))
    logf = get(opts, "log", "")
    maxsteady = parse(Float64, get(opts, "maxsteady", "Inf"))
    out = IOBuffer()
    println("# $(now()) $(join(args, " "))")
    if bname == "mf6"
        refdir = get(ENV, "REFERENCE_DIR", joinpath(@__DIR__, "..", "reference"))
        exe = get(ENV, "MF6", joinpath(refdir, "mf6.8.2_linux", "bin", "mf6"))
        workdir = get(opts, "workdir", tempdir())
        for p in presets
            r = mf6_case(p, exe, workdir; nsteps, ims)
            @printf("%s: write %.1f s, read %.2f s, total %.1f s\n", p, r.twrite, r.tread, r.tend)
            report_mf6(r; io = out)
            print(String(take!(copy(out))))
        end
    else
        backend = ModflowGPU.select_backend(bname)
        precisions = Symbol.(split(get(opts, "precision", "double,single"), ","))
        timed = get(opts, "timed", "1") == "1"
        # Warm-up: compile every kernel in both presets on a small model.
        for pre in precisions
            spec = synthetic_simulation(:tiny; output = false, ntransient = 1, ims = synthetic_ims(ims))
            s = ModflowGPU.initialize(spec; backend, precision = pre, write_output = false)
            run_steps!(s, backend, 1)
            spmv_bench(s, backend)
            t = SyncTimer(backend)
            ModflowGPU.set_section_timer!(t)
            s = ModflowGPU.initialize(spec; backend, precision = pre, write_output = false)
            run_steps!(s, backend, 1; timer = t)
            ModflowGPU.set_section_timer!(nothing)
        end
        splits = IOBuffer()
        for p in presets, pre in precisions
            r = run_case(p, pre, backend, bname; nsteps, timed, ims, maxsteady)
            report(r; io = out)
            report_split(r; io = splits)
            println(HEADER)
            print(String(take!(copy(out))))
            if !isnothing(r.split)
                println(SPLIT_HEADER)
                print(String(take!(copy(splits))))
                show(stdout, r.split.to; allocations = false, compact = true, sortby = :firstexec)
                println()
            end
            flush(stdout)
        end
        print(out, "\n", SPLIT_HEADER, "\n", String(take!(splits)))
    end
    txt = String(take!(out))
    isempty(logf) || open(io -> print(io, txt), logf, "a")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    let bname = isempty(ARGS) ? "cpu" : ARGS[1]
        if bname == "rocm"
            @eval using AMDGPU
            AMDGPU.allowscalar(false)
            Base.invokelatest(setup_rocm_memory!, AMDGPU)
        elseif bname ∉ ("cpu", "mf6")
            @eval using $(Symbol(ModflowGPU.known_backend_package(bname)))
        end
        Base.invokelatest(main, ARGS)
    end
end
