#!/usr/bin/env julia
# Reference runs of the in-scope MF6 examples with mf6 6.8.2 (T0.3).
#
#     REFERENCE_DIR=/path/to/reference julia validation/run_reference.jl [options] [names...]
#
# For every example in validation/examples.toml (or only `names...`) this copies
# the simulation directory to $REFERENCE_DIR/runs/<name>/<mode>/ and runs mf6 in it:
#
#   raw      the example unchanged
#   tight    every IMS file patched to the tight closures of PLAN.md section 4:
#            OUTER_DVCLOSE 1e-9, INNER_DVCLOSE 1e-10, INNER_RCLOSE 1e-10,
#            OUTER_MAXIMUM >= 500, INNER_MAXIMUM >= 1000 (see patch_ims)
#   fallbacks, only run when `tight` fails:
#   tight-relative  as tight, but INNER_RCLOSE 1e-10 RELATIVE_RCLOSE: the inner
#            solve then converges on INNER_DVCLOSE alone. Needed where an absolute
#            residual of 1e-10 is below the rounding floor of the model's flows.
#   relaxed  OUTER_DVCLOSE 1e-8, INNER_DVCLOSE 1e-9, INNER_RCLOSE 1e-8 (keyword
#            kept), same iteration limits
#   Patched modes also get PRINT_OPTION SUMMARY (if the file has no PRINT_OPTION)
#   so that mfsim.lst reports the iteration counts. summary.* names, per
#   example, the first successful mode of tight, tight-relative, relaxed as
#   `reference_mode`: the run whose heads the parity tests should use.
#
# Each run directory gets run.toml (status, exit code, wall time, listing checks)
# and stdout.txt; $REFERENCE_DIR/runs/summary.toml and summary.md collect them.
# Runs that already have a run.toml are skipped unless --force.
#
# Options:
#   --force          rerun even if run.toml exists
#   --summary-only   run nothing: re-derive the listing checks of existing runs
#                    and rewrite run.toml and the summaries
#   --jobs N         parallel mf6 processes (default: number of CPU threads ÷ 2)
#   --modes a,b      modes to run (default raw,tight; relaxed follows a failed tight)
#   --no-relaxed     do not run the fallbacks
#   --tiers a,b      only these tiers
#   --timeout S      seconds per run before it is killed (default 3600)
#
# Environment: REFERENCE_DIR (default <repo>/reference), MF6 (default
# $REFERENCE_DIR/mf6.8.2_linux/bin/mf6). Julia stdlib only.

using TOML, Printf, Dates

const REPO = dirname(@__DIR__)
const REFERENCE_DIR = abspath(get(ENV, "REFERENCE_DIR", joinpath(REPO, "reference")))
const EXAMPLES = joinpath(REFERENCE_DIR, "mf6examples")
const RUNS = joinpath(REFERENCE_DIR, "runs")
const MF6 = get(ENV, "MF6", joinpath(REFERENCE_DIR, "mf6.8.2_linux", "bin", "mf6"))

# rclose_kw: nothing = keep the example's INNER_RCLOSE keyword, else replace it.
const CLOSURES = Dict(
    "tight" => (outer_dvclose = 1.0e-9, inner_dvclose = 1.0e-10, inner_rclose = 1.0e-10, rclose_kw = nothing),
    "tight-relative" => (outer_dvclose = 1.0e-9, inner_dvclose = 1.0e-10, inner_rclose = 1.0e-10, rclose_kw = "RELATIVE_RCLOSE"),
    "relaxed" => (outer_dvclose = 1.0e-8, inner_dvclose = 1.0e-9, inner_rclose = 1.0e-8, rclose_kw = nothing),
)
const FALLBACKS = ["tight-relative", "relaxed"]
# preference order for the reference heads of the parity tests
const REFERENCE_ORDER = ["tight", "tight-relative", "relaxed"]
const MIN_OUTER = 500
const MIN_INNER = 1000
# MF6 defaults by IMS COMPLEXITY (NumericalSolution.f90, ImsLinearSettings.f90);
# the default complexity is SIMPLE.
const DEFAULT_OUTER = Dict("SIMPLE" => 25, "MODERATE" => 50, "COMPLEX" => 100)
const DEFAULT_INNER = Dict("SIMPLE" => 50, "MODERATE" => 100, "COMPLEX" => 500)
# deprecated IMS names, replaced by the DVCLOSE keys when patching
const DEPRECATED = Dict("OUTER_HCLOSE" => "OUTER_DVCLOSE", "INNER_HCLOSE" => "INNER_DVCLOSE")

# ---------------------------------------------------------------------------
# IMS patching

fmt(x) = @sprintf("%.1e", x)

"""
    patch_ims(path, c) -> Dict

Rewrite the IMS file at `path` in place with closures `c`. Keeps every other
line. OUTER_DVCLOSE (NONLINEAR block) and INNER_DVCLOSE / INNER_RCLOSE (LINEAR
block) are replaced or added; an INNER_RCLOSE keyword (STRICT, L2NORM_RCLOSE,
RELATIVE_RCLOSE) is kept, so the new value has the same meaning as the old one.
OUTER_MAXIMUM / INNER_MAXIMUM become max(existing or complexity default, 500 / 1000).
A missing OPTIONS, NONLINEAR or LINEAR block is inserted in MF6's reading
order OPTIONS, NONLINEAR, LINEAR (NumericalSolution.f90 `sln_ar` reads OPTIONS
and NONLINEAR, then ImsLinearSettings reads LINEAR; a NONLINEAR block after
LINEAR is silently ignored). Returns what was set.
"""
function patch_ims(path, c)
    lines = readlines(path)
    tok(l) = split(strip(replace(l, r"[#!].*$" => "")))
    complexity = "SIMPLE"
    for l in lines
        t = tok(l)
        length(t) >= 2 && uppercase(t[1]) == "COMPLEXITY" && (complexity = uppercase(t[2]))
    end
    outer_max = DEFAULT_OUTER[complexity]
    inner_max = DEFAULT_INNER[complexity]
    rclose_kw = ""
    has_print = false
    for l in lines
        t = tok(l)
        isempty(t) && continue
        k = uppercase(t[1])
        k == "OUTER_MAXIMUM" && (outer_max = parse(Int, t[2]))
        k == "INNER_MAXIMUM" && (inner_max = parse(Int, t[2]))
        k == "INNER_RCLOSE" && length(t) >= 3 && (rclose_kw = uppercase(t[3]))
        k == "PRINT_OPTION" && (has_print = true)
    end
    c.rclose_kw === nothing || (rclose_kw = c.rclose_kw)
    set = Dict(
        "NONLINEAR" => ["OUTER_DVCLOSE" => fmt(c.outer_dvclose), "OUTER_MAXIMUM" => string(max(outer_max, MIN_OUTER))],
        # PRINT_OPTION SUMMARY only adds the per-step iteration counts to mfsim.lst
        "OPTIONS" => has_print ? Pair{String, String}[] : ["PRINT_OPTION" => "SUMMARY"],
        "LINEAR" => [
            "INNER_MAXIMUM" => string(max(inner_max, MIN_INNER)),
            "INNER_DVCLOSE" => fmt(c.inner_dvclose),
            "INNER_RCLOSE" => strip(fmt(c.inner_rclose) * " " * rclose_kw),
        ],
    )
    order = ("OPTIONS", "NONLINEAR", "LINEAR")
    seen_blocks = Set{String}()
    for l in lines
        t = tok(l)
        length(t) >= 2 && uppercase(t[1]) == "BEGIN" && push!(seen_blocks, uppercase(t[2]))
    end
    # blocks to add, emitted just before the first existing block that MF6 reads after them
    missing_blocks = [b for b in order if !(b in seen_blocks) && !isempty(set[b])]
    function emit_missing!(out, upto)
        while !isempty(missing_blocks) && findfirst(==(missing_blocks[1]), order) < upto
            b = popfirst!(missing_blocks)
            push!(out, "BEGIN $b")
            for (key, val) in set[b]
                push!(out, "  $key  $val")
            end
            push!(out, "END $b", "")
        end
        return out
    end
    out = String[]
    block = ""
    done = Set{String}()
    for l in lines
        t = tok(l)
        k = isempty(t) ? "" : uppercase(t[1])
        k = get(DEPRECATED, k, k)
        if k == "BEGIN"
            block = uppercase(t[2])
            i = findfirst(==(block), order)
            i === nothing || emit_missing!(out, i)
        elseif k == "END" && haskey(set, block)
            for (key, val) in set[block]
                key in done || push!(out, "  $key  $val")
            end
            block = ""
        elseif haskey(set, block)
            i = findfirst(p -> p[1] == k, set[block])
            if i !== nothing
                key, val = set[block][i]
                push!(out, "  $key  $val  # patched by run_reference.jl")
                push!(done, key)
                continue
            end
        end
        push!(out, l)
    end
    isempty(missing_blocks) || push!(out, "")
    emit_missing!(out, length(order) + 1)
    write(path, join(out, "\n") * "\n")
    return Dict(k => v for b in values(set) for (k, v) in b)
end

"IMS files named in mfsim.nam (SOLUTIONGROUP blocks)."
function ims_files(dir)
    files = String[]
    inblock = false
    for l in eachline(joinpath(dir, "mfsim.nam"))
        t = split(strip(replace(l, r"[#!].*$" => "")))
        isempty(t) && continue
        k = uppercase(t[1])
        if k == "BEGIN"
            inblock = uppercase(t[2]) == "SOLUTIONGROUP"
        elseif k == "END"
            inblock = false
        elseif inblock && k == "IMS6"
            push!(files, t[2])
        end
    end
    return files
end

# ---------------------------------------------------------------------------
# Listing checks

function check_listings(dir)
    sim = joinpath(dir, "mfsim.lst")
    r = Dict{String, Any}(
        "normal_termination" => false, "converged" => false, "nonconverged_steps" => 0,
        "max_abs_percent_discrepancy" => 0.0,
    )
    outer = inner = 0
    found = false
    isfile(sim) || return r
    txt = read(sim, String)
    r["normal_termination"] = occursin("Normal termination of simulation", txt)
    nfail = count(r"Solution Group \d+ did not converge for stress period", txt)
    m = match(r"Simulation convergence failure occurred (\d+) time", txt)
    m === nothing || (nfail = max(nfail, parse(Int, m[1])))
    for f in readdir(dir; join = true)
        endswith(f, ".lst") || continue
        t = f == sim ? txt : read(f, String)
        f == sim || (nfail = max(nfail, count("FAILED TO MEET SOLVER CONVERGENCE CRITERIA", t)))
        for m in eachmatch(r"(\d+) CALLS TO NUMERICAL SOLUTION IN TIME STEP \d+ STRESS PERIOD \d+\s*\n\s*(\d+) TOTAL ITERATIONS", t)
            outer += parse(Int, m[1])
            inner += parse(Int, m[2])
            found = true
        end
        # percent discrepancy recomputed from the printed TOTAL IN / TOTAL OUT
        # (cumulative and rate columns), as MF6 does: 100 (in - out) / ((in + out) / 2).
        # The listing prints the totals with 4 decimals, which bounds the resolution;
        # MF6's own PERCENT DISCREPANCY line has only 0.01 % resolution.
        num = "([-+0-9.Ee]+)"
        tin = [parse.(Float64, (m[1], m[2])) for m in eachmatch(Regex("TOTAL IN =\\s*$num\\s+TOTAL IN =\\s*$num"), t)]
        tout = [parse.(Float64, (m[1], m[2])) for m in eachmatch(Regex("TOTAL OUT =\\s*$num\\s+TOTAL OUT =\\s*$num"), t)]
        for (a, b) in zip(tin, tout), j in 1:2
            s = a[j] + b[j]
            s > 0 && (r["max_abs_percent_discrepancy"] = max(r["max_abs_percent_discrepancy"], abs(200 * (a[j] - b[j]) / s)))
        end
    end
    # iteration counts are printed only with an IMS PRINT_OPTION
    if found
        r["outer_iterations"] = outer
        r["inner_iterations"] = inner
    end
    r["nonconverged_steps"] = nfail
    r["converged"] = r["normal_termination"] && nfail == 0
    # MF6 prints "ERROR REPORT:" followed by numbered messages
    i = findfirst("ERROR REPORT:", txt)
    if i !== nothing
        errs = filter(!isempty, strip.(split(txt[(last(i) + 1):end], '\n')))
        r["errors"] = first(errs, 5)
    end
    return r
end

"""
    check_closures(dir, mode) -> Dict

Closures that mf6 actually applied, parsed from the solver echo in mfsim.lst
(one entry per solution: NumericalSolution.f90 `sln_ar` format 9002 and
ImsLinear.f90 `imslinear_ar` format 02010), compared with the closures requested
for `mode`. A patch that mf6 ignores (e.g. a block in the wrong order) shows up
here. `closures_ok` is false if any applied value differs from the request or
if the echo is missing.
"""
function check_closures(dir, mode)
    sim = joinpath(dir, "mfsim.lst")
    r = Dict{String, Any}()
    isfile(sim) || return r
    txt = read(sim, String)
    num = "([-+0-9.Ee]+)"
    grab(re) = [parse(Float64, m[1]) for m in eachmatch(Regex(re), txt)]
    applied = Dict{String, Any}(
        "outer_dvclose" => grab("OUTER ITERATION CONVERGENCE CRITERION\\s+\\(DVCLOSE\\) =\\s*$num"),
        "outer_maximum" => Int.(grab("MAXIMUM NUMBER OF OUTER ITERATIONS\\s+\\(MXITER\\) =\\s*$num")),
        "inner_maximum" => Int.(grab("MAXIMUM OF (\\d+) INTERNAL ITERATIONS PER CALL")),
        "inner_dvclose" => grab("HEAD CHANGE CRITERION FOR CLOSURE\\s+=\\s*$num"),
        "inner_rclose" => grab("RESIDUAL CHANGE CRITERION FOR CLOSURE\\s+=\\s*$num"),
    )
    r["closures_applied"] = applied
    haskey(CLOSURES, mode) || return r
    c = CLOSURES[mode]
    # the echo has 6 (outer) and 5 (inner) significant digits
    close_to(a, b) = abs(a - b) <= 1.0e-4 * abs(b)
    bad = String[]
    for (key, want) in ("outer_dvclose" => c.outer_dvclose, "inner_dvclose" => c.inner_dvclose, "inner_rclose" => c.inner_rclose)
        isempty(applied[key]) && push!(bad, "$key not echoed in mfsim.lst")
        for v in applied[key]
            close_to(v, want) || push!(bad, @sprintf("%s applied %.6e, requested %.6e", key, v, want))
        end
    end
    for (key, want) in ("outer_maximum" => MIN_OUTER, "inner_maximum" => MIN_INNER)
        isempty(applied[key]) && push!(bad, "$key not echoed in mfsim.lst")
        for v in applied[key]
            v >= want || push!(bad, "$key applied $v, requested >= $want")
        end
    end
    r["closures_ok"] = isempty(bad)
    isempty(bad) || (r["closure_errors"] = bad)
    return r
end

# ---------------------------------------------------------------------------
# Running

function run_one(ex, mode; timeout)
    name = ex["name"]
    src = joinpath(EXAMPLES, ex["path"])
    dest = joinpath(RUNS, name, mode)
    ispath(dest) && rm(dest; recursive = true)
    mkpath(dirname(dest))
    cp(src, dest)
    info = Dict{String, Any}(
        "name" => name, "mode" => mode, "tier" => ex["tier"], "path" => ex["path"],
        "mf6" => MF6, "started" => string(now()),
    )
    if mode != "raw"
        patched = Dict{String, Any}()
        for f in ims_files(dest)
            patched[f] = patch_ims(joinpath(dest, f), CLOSURES[mode])
        end
        info["ims_patch"] = patched
    end
    t0 = time()
    p = run(pipeline(Cmd(`$MF6`; dir = dest); stdout = joinpath(dest, "stdout.txt"), stderr = joinpath(dest, "stderr.txt")); wait = false)
    timed_out = false
    while process_running(p)
        if time() - t0 > timeout
            kill(p)
            timed_out = true
            break
        end
        sleep(0.05)
    end
    wait(p)
    info["wall_s"] = round(time() - t0; digits = 3)
    info["exit_code"] = p.exitcode
    info["timed_out"] = timed_out
    update_listing_fields!(info, dest)
    @info @sprintf(
        "%-22s %-8s %-12s exit=%s %8.2f s outer=%s", name, mode, info["status"], p.exitcode,
        info["wall_s"], get(info, "outer_iterations", "-")
    )
    return info
end

const LISTING_KEYS = (
    "normal_termination", "converged", "nonconverged_steps", "outer_iterations",
    "inner_iterations", "max_abs_percent_discrepancy", "errors", "status",
    "closures_applied", "closures_ok", "closure_errors",
)

"(Re)derive the listing checks and the status of a finished run; rewrite run.toml."
function update_listing_fields!(info, dest)
    foreach(k -> delete!(info, k), LISTING_KEYS)
    merge!(info, check_listings(dest))
    merge!(info, check_closures(dest, info["mode"]))
    info["status"] = info["timed_out"] ? "timeout" :
        !get(info, "closures_ok", true) ? "wrong_closures" :
        (info["exit_code"] == 0 && info["converged"]) ? "ok" :
        info["normal_termination"] ? "nonconverged" :
        info["nonconverged_steps"] > 0 ? "nonconverged" : "error"
    open(io -> TOML.print(io, info; sorted = true), joinpath(dest, "run.toml"), "w")
    return info
end

function load_run(name, mode)
    f = joinpath(RUNS, name, mode, "run.toml")
    return isfile(f) ? TOML.parsefile(f) : nothing
end

function parse_args(args)
    o = Dict{String, Any}(
        "force" => false, "jobs" => max(1, Sys.CPU_THREADS ÷ 2), "modes" => ["raw", "tight"],
        "relaxed" => true, "tiers" => nothing, "timeout" => 3600.0, "names" => String[],
        "summary_only" => false,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--force"
            o["force"] = true
        elseif a == "--summary-only"
            o["summary_only"] = true
        elseif a == "--no-relaxed"
            o["relaxed"] = false
        elseif a == "--jobs"
            o["jobs"] = parse(Int, args[i += 1])
        elseif a == "--modes"
            o["modes"] = split(args[i += 1], ',')
        elseif a == "--tiers"
            o["tiers"] = parse.(Int, split(args[i += 1], ','))
        elseif a == "--timeout"
            o["timeout"] = parse(Float64, args[i += 1])
        elseif startswith(a, "--")
            error("unknown option $a")
        else
            push!(o["names"], a)
        end
        i += 1
    end
    return o
end

function write_summary(examples)
    rows = Dict{String, Any}[]
    for ex in examples
        r = Dict{String, Any}("name" => ex["name"], "tier" => ex["tier"])
        for mode in ("raw", REFERENCE_ORDER...)
            info = load_run(ex["name"], mode)
            info === nothing && continue
            r[mode] = Dict(
                k => info[k] for k in (
                        "status", "exit_code", "wall_s", "normal_termination", "converged",
                        "nonconverged_steps", "outer_iterations", "inner_iterations", "max_abs_percent_discrepancy",
                        "closures_ok",
                    )
                    if haskey(info, k)
            )
        end
        i = findfirst(m -> haskey(r, m) && r[m]["status"] == "ok", REFERENCE_ORDER)
        r["reference_mode"] = i === nothing ? "none" : REFERENCE_ORDER[i]
        # run directory with the reference heads, relative to REFERENCE_DIR
        i === nothing || (r["path"] = join(["runs", ex["name"], REFERENCE_ORDER[i]], "/"))
        push!(rows, r)
    end
    open(joinpath(RUNS, "summary.toml"), "w") do io
        println(io, "# Written by validation/run_reference.jl on $(now())")
        TOML.print(io, Dict("mf6" => MF6, "runs" => rows); sorted = true)
    end
    cell(r, m) = haskey(r, m) ?
        @sprintf("%s (%.1f s, %s outer)", r[m]["status"], r[m]["wall_s"], get(r[m], "outer_iterations", "-")) : "-"
    open(joinpath(RUNS, "summary.md"), "w") do io
        println(io, "# mf6 reference runs\n\nmf6: `$MF6`, written $(now())\n")
        println(io, "| Example | Tier | raw | tight | tight-relative | relaxed | reference |")
        println(io, "| --- | --- | --- | --- | --- | --- | --- |")
        for r in rows
            println(
                io, "| ", r["name"], " | ", r["tier"], " | ", cell(r, "raw"), " | ", cell(r, "tight"), " | ",
                cell(r, "tight-relative"), " | ", cell(r, "relaxed"), " | ", r["reference_mode"], " |"
            )
        end
    end
    return rows
end

function main(args)
    o = parse_args(args)
    isfile(MF6) || error("mf6 not found at $MF6; run validation/fetch_reference.sh or set MF6")
    inv = TOML.parsefile(joinpath(@__DIR__, "examples.toml"))
    examples = [merge(Dict("name" => k), v) for (k, v) in inv["examples"]]
    sort!(examples; by = e -> (e["tier"], e["name"]))
    isempty(o["names"]) || filter!(e -> e["name"] in o["names"], examples)
    o["tiers"] === nothing || filter!(e -> e["tier"] in o["tiers"], examples)
    isempty(examples) && error("no examples selected")
    # largest first so the long runs start early
    order = sort(examples; by = e -> -e["ncells"] * max(1, e["nstp_total"]))
    if o["summary_only"]
        # no mf6 runs: re-derive the listing checks of the existing runs
        for ex in examples, mode in ("raw", REFERENCE_ORDER...)
            info = load_run(ex["name"], mode)
            info === nothing || update_listing_fields!(info, joinpath(RUNS, ex["name"], mode))
        end
        empty!(order)
    end
    sem = Base.Semaphore(o["jobs"])
    @sync for ex in order
        @async begin
            for mode in o["modes"]
                prev = load_run(ex["name"], mode)
                if prev !== nothing && !o["force"]
                    info = prev
                else
                    info = Base.acquire(() -> run_one(ex, mode; timeout = o["timeout"]), sem)
                end
                if mode == "tight" && info["status"] != "ok" && o["relaxed"]
                    for fb in FALLBACKS
                        prev = load_run(ex["name"], fb)
                        (prev === nothing || o["force"]) &&
                            Base.acquire(() -> run_one(ex, fb; timeout = o["timeout"]), sem)
                    end
                end
            end
        end
    end
    all_ex = [merge(Dict("name" => k), v) for (k, v) in inv["examples"]]
    sort!(all_ex; by = e -> (e["tier"], e["name"]))
    rows = write_summary(all_ex)
    println(read(joinpath(RUNS, "summary.md"), String))
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
