# Parallel test driver (T6.8): `MFGPU_TEST_JOBS=N julia --project=test test/runtests.jl`.
#
# Splits the selected test groups into work units, runs the units in worker processes (N at a
# time, the longest first, each worker taking the next unit when it is free), and prints a
# combined summary. Each worker is `test/runtests.jl` run serially (MFGPU_TEST_JOBS=1) with
# `MFGPU_TEST_GROUPS` (and, for the groups that are split by model, `MFGPU_TEST_MODELS`) set to
# its unit, so a worker is exactly the serial run of those groups; its output goes to a log
# file, shown when it fails. The run fails if any worker fails or crashes. A hand-written
# driver instead of ParallelTestRunner.jl: that package takes one test file as a unit, runs
# the files in one process pool with its own test-file discovery and output, and would need
# the test files to be self-contained, while here the helpers are shared and the units are
# selected by two existing environment variables; the driver is about 100 lines.
#
# Needs only Base and Test (loaded by runtests.jl) and `TEST_GROUPS` (test_groups.jl).

"""
Measured serial wall time in seconds (CPU backend, 2026-11, including about 3 s of Julia
start-up and package loading) of each test group, used to order and balance the work
units; groups that are not listed count as `DEFAULT_WEIGHT`. Re-measure when a group changes
a lot (`MFGPU_TEST_GROUPS=<group> julia --project=test test/runtests.jl`).
"""
const GROUP_WEIGHTS = Dict{String, Float64}(
    "backend" => 7,
    "precision" => 2,
    "api" => 10,
    "kernel" => 4,
    "input_tokenizer" => 3,
    "input_blocks" => 4,
    "input_examples" => 6,
    "input_arrays" => 8,
    "input_lists" => 10,
    "timeseries" => 17,
    "output" => 24,
    "simulation_loader" => 12,
    "connectivity" => 32,
    "disv_disu" => 86,
    "npf" => 60,
    "boundaries" => 35,
    "sto" => 26,
    "convertible" => 26,
    "newton" => 22,
    "xt3d" => 60,
    "hfb" => 85,
    "tv" => 60,
    "gnc" => 40,
    "exchange" => 55,
    "riv_drn_evt" => 35,
    "assembly" => 160,
    "krylov" => 97,
    "amg" => 69,
    "picard" => 81,
    "stepping" => 87,
    "bmi" => 92,
    "obs" => 30,
    "builder" => 73,
    "synthetic" => 88,
    "kernel_hygiene" => 2,
)
const DEFAULT_WEIGHT = 30.0

"""
The group that dominates the run time, `run`, is split by model (`MFGPU_TEST_MODELS`, see
`test_model_selected` in input_files.jl): group => `(units, rest)` where `units` are
`(models, weight)` pairs and `rest` is the weight of the last unit, which is every other model
(`MFGPU_TEST_MODELS=!<all listed>`). The weights are the seconds measured for the unit alone.
"""
const GROUP_SPLITS = Dict(
    "run" => (
        [
            ("ex-gwf-lgrv-gr", 205.0),
            ("ex-gwf-nwt-p02a,ex-gwf-nwt-p02b", 215.0),
            ("ex-gwf-hwell-a,ex-gwf-advtidal", 150.0),
            ("coupling_rde", 70.0),
            # T6.6: the Tier 7 examples (several models, exchanges).
            ("ex-gwf-lgrv-lgr,ex-gwf-spbc,ex-gwf-u1gwfgwf-s1,ex-gwf-u1gwfgwf-s2,ex-gwf-u1gwfgwf-s3,ex-gwf-u1gwfgwf-s4", 60.0),
            (
                "ex-gwf-bcf2ss-p01a,ex-gwf-bcf2ss-p02a,ex-gwf-bump-p01a,ex-gwf-bump-p01b,ex-gwf-bump-p01c,ex-gwf-curve-90," *
                    "ex-gwf-curvilin,ex-gwf-fhb,ex-gwf-hanic,ex-gwf-hanir,ex-gwf-lgrv-gc,ex-gwf-maw-p03a,ex-gwf-maw-p03c", 170.0,
            ),
        ],
        170.0,
    ),
)

struct WorkUnit
    name::String
    group::String
    models::String
    weight::Float64
end

"The work units for `groups`; `models` is the user's MFGPU_TEST_MODELS (empty: split as in `GROUP_SPLITS`)."
function work_units(groups::Vector{String}, models::AbstractString)
    reference = get(ENV, "REFERENCE_DIR", joinpath(@__DIR__, "..", "reference"))
    units = WorkUnit[]
    for g in groups
        # Without reference/ (CI) the run group skips every model: one unit is enough.
        if isempty(models) && haskey(GROUP_SPLITS, g) && isdir(reference)
            listed, rest = GROUP_SPLITS[g]
            names = strip.(split(join(first.(listed), ','), ','))
            allunique(names) || error("GROUP_SPLITS[\"$g\"] lists a model twice")
            for (i, (ms, w)) in enumerate(listed)
                push!(units, WorkUnit("$g[$i]", g, ms, w))
            end
            push!(units, WorkUnit("$g[rest]", g, "!" * join(first.(listed), ','), rest))
        else
            push!(units, WorkUnit(g, g, String(models), get(GROUP_WEIGHTS, g, DEFAULT_WEIGHT)))
        end
    end
    return sort!(units; by = u -> -u.weight)
end

"""
    run_parallel_tests(jobs) -> Bool

Run the selected test groups (`MFGPU_TEST_GROUPS`, default all) in `jobs` worker processes
and print a combined summary; `true` when every worker passed. Worker logs are in
`MFGPU_TEST_LOGDIR` (default a temporary directory, removed when everything passed).
"""
function run_parallel_tests(jobs::Int)
    valid = first.(TEST_GROUPS)
    env = strip(get(ENV, "MFGPU_TEST_GROUPS", ""))
    groups = isempty(env) ? valid : [lowercase(strip(g)) for g in split(env, ',') if !isempty(strip(g))]
    unknown = setdiff(groups, valid)
    isempty(unknown) || error("Unknown MFGPU_TEST_GROUPS name(s): $(join(unknown, ", ")). Valid groups are: $(join(valid, ", "))")
    # Keep the run order of TEST_GROUPS for equal weights.
    groups = filter(in(groups), valid)
    units = work_units(groups, strip(get(ENV, "MFGPU_TEST_MODELS", "")))
    jobs = min(jobs, length(units))
    logdir = get(ENV, "MFGPU_TEST_LOGDIR", "")
    keep_logs = !isempty(logdir)
    isempty(logdir) && (logdir = mktempdir())
    mkpath(logdir)
    @info "Running $(length(units)) test units in $jobs parallel workers (logs: $logdir)"

    runtests = joinpath(@__DIR__, "runtests.jl")
    julia = Base.julia_cmd()
    project = Base.active_project()
    threads = "$(Threads.nthreads(:default)),$(Threads.nthreads(:interactive))"  # as Pkg.test
    results = Dict{String, NamedTuple}()
    queue = Channel{WorkUnit}(length(units))
    foreach(u -> put!(queue, u), units)
    close(queue)
    print_lock = ReentrantLock()
    t0 = time()

    function run_unit(u::WorkUnit)
        log = joinpath(logdir, "$(replace(u.name, r"[\[\]!,]" => "_")).log")
        counts = joinpath(logdir, "$(replace(u.name, r"[\[\]!,]" => "_")).counts")
        rm(counts; force = true)
        cenv = copy(ENV)
        cenv["MFGPU_TEST_JOBS"] = "1"
        cenv["MFGPU_TEST_GROUPS"] = u.group
        cenv["MFGPU_TEST_RESULT_FILE"] = counts
        isempty(u.models) ? delete!(cenv, "MFGPU_TEST_MODELS") : (cenv["MFGPU_TEST_MODELS"] = u.models)
        cmd = setenv(`$julia --project=$project --threads=$threads $runtests`, cenv)
        t = time()
        ok = open(log, "w") do io
            success(pipeline(ignorestatus(cmd); stdout = io, stderr = io))
        end
        dt = time() - t
        c = try
            parse.(Int, split(read(counts, String)))
        catch
            nothing  # crashed before the testset ended
        end
        r = (; ok = ok && !isnothing(c), seconds = dt, counts = something(c, [0, 0, 0, 0]), crashed = isnothing(c), log)
        @lock print_lock begin
            results[u.name] = r
            printstyled(stderr, r.ok ? "[ok]   " : "[FAIL] "; color = r.ok ? :green : :red, bold = true)
            println(stderr, rpad(u.name, 20), " ", round(dt; digits = 1), " s  (", round(time() - t0; digits = 1), " s elapsed)")
        end
        return nothing
    end

    workers = [
        @async(
            for u in queue
                run_unit(u)
            end
        ) for _ in 1:jobs
    ]
    foreach(wait, workers)
    wall = time() - t0

    println(stderr)
    println(stderr, "Parallel test summary ($jobs workers, wall time $(round(wall; digits = 1)) s)")
    println(stderr, rpad("unit", 22), lpad("seconds", 9), lpad("pass", 8), lpad("fail", 6), lpad("error", 7), lpad("broken", 8))
    total = zeros(Int, 4)
    for u in sort!(collect(keys(results)))
        r = results[u]
        total .+= r.counts
        println(stderr, rpad(u, 22), lpad(round(r.seconds; digits = 1), 9), lpad(r.counts[1], 8), lpad(r.counts[2], 6), lpad(r.counts[3], 7), lpad(r.counts[4], 8), r.crashed ? "  CRASHED" : "")
    end
    println(stderr, rpad("total", 22), lpad(round(sum(r.seconds for r in values(results)); digits = 1), 9), lpad(total[1], 8), lpad(total[2], 6), lpad(total[3], 7), lpad(total[4], 8))
    failed = [u for u in sort!(collect(keys(results))) if !results[u].ok]
    for u in failed
        println(stderr, "\n===== Worker $u failed; end of its log ($(results[u].log)) =====")
        lines = readlines(results[u].log)
        foreach(l -> println(stderr, l), lines[max(1, end - 80):end])
    end
    isempty(failed) && !keep_logs && rm(logdir; recursive = true, force = true)
    isempty(failed) || println(stderr, "\nFailed workers: ", join(failed, ", "), " (full logs in $logdir)")
    return isempty(failed)
end
