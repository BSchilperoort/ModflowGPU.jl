# Helpers of the example gallery (examples/*.jl), for the documentation only: not part of
# ModflowGPU. Loaded by docs/make.jl and by each example script when it runs on its own.
#
# - Inputs: downloads the MODFLOW 6 examples (`mf6examples.zip`) and the MODFLOW 6.8.2
#   Linux binary from the same URLs as validation/fetch_reference.sh, checks their sha256
#   and keeps them in a cache directory (`MFGPU_GALLERY_CACHE`, default `docs/.cache`,
#   gitignored). With `MFGPU_GALLERY_ARCHIVES=<dir>` an existing copy of the archives is
#   used instead of a download (for example `reference/archives`, laid out as
#   `<dir>/mf6examples/mf6examples.zip`); it is only read, and its sha256 is checked too.
# - Runs: copies an example into the cache, sets the tight closures of the parity tests
#   in its IMS file (`patch_ims` of validation/run_reference.jl), runs mf6 there, then
#   ModflowGPU on the same input, and compares heads and budgets as the parity tests do.
# - Plots: CairoMakie helpers for maps, cross-sections, hydrographs and budget bars of
#   structured (DIS) grids, read from MF6's binary grid file.
module ExampleTools

using ModflowGPU
using CairoMakie
using Downloads: Downloads
using Logging: Logging
using Markdown: Markdown
using Printf: @sprintf
using SHA: sha256
using TOML: TOML

export example_run, mf6_run, quietly, cache_dir, repo_path, run_modflowgpu, compare_with_mf6, comparison_table, parity_row,
    DisGrid, read_dis_grid, VertexGrid, read_vertex_grid, read_grid, input_nodes, node, cell_of, layer_values, specific_discharge, budget_cells,
    map_heatmap!, map_contour!, map_grid!, map_cells!, map_vectors!, section_heatmap!, section_grid!,
    section_contour!, budget_bars, difference_map, save_figure, intformat, expformat, BC_COLORS, mask_inactive,
    obs_pair, obs_masked, obs_plot!, obs_difference_table

const DOCS = @__DIR__
const ROOT = dirname(DOCS)

# `patch_ims`, `ims_files`, `CLOSURES` and `check_listings` of the reference runs, so that
# the gallery runs mf6 with the same closures as the reference runs of the parity tests.
module RunReference
    include(joinpath(dirname(@__DIR__), "validation", "run_reference.jl"))
end

# --- Downloads ----------------------------------------------------------------------

const MF6_VERSION = "6.8.2"

"""
Pinned inputs: the same URLs and sha256 as validation/fetch_reference.sh. The examples
come from the rolling `current` release of modflow6-examples, so its digest is the one
of the release used for the parity table (docs/src/assets/parity.toml).
"""
const ARCHIVES = (
    mf6examples = (;
        url = "https://github.com/MODFLOW-ORG/modflow6-examples/releases/download/current/mf6examples.zip",
        sha256 = "cf40b4b900aa6ba19fbd108f0857b192ed90c8bed2a9ebdb7642565c1b9cc406",
        file = "mf6examples.zip", subdir = "mf6examples",
    ),
    mf6 = (;
        url = "https://github.com/MODFLOW-ORG/modflow6/releases/download/$MF6_VERSION/mf$(MF6_VERSION)_linux.zip",
        sha256 = "69847a003a4259d69f0ae637bc191b99097650062847f39ed17d234fd8596f37",
        file = "mf$(MF6_VERSION)_linux.zip", subdir = "mf$(MF6_VERSION)_linux",
    ),
)

"A path in the ModflowGPU repository."
repo_path(parts...) = joinpath(ROOT, parts...)

"The cache directory of the gallery: `MFGPU_GALLERY_CACHE`, default `docs/.cache`."
cache_dir() = mkpath(abspath(get(ENV, "MFGPU_GALLERY_CACHE", joinpath(DOCS, ".cache"))))

file_sha256(path) = bytes2hex(open(sha256, path))

"""
    archive_path(key) -> String

The verified local path of archive `key` (`:mf6examples` or `:mf6`): from
`MFGPU_GALLERY_ARCHIVES` if set and the file is there, else from the cache, downloaded
first if it is not there yet. Throws when the sha256 does not match.
"""
function archive_path(key::Symbol)
    a = ARCHIVES[key]
    local_dir = get(ENV, "MFGPU_GALLERY_ARCHIVES", "")
    if !isempty(local_dir)
        for p in (joinpath(local_dir, a.subdir, a.file), joinpath(local_dir, a.file))
            isfile(p) || continue
            file_sha256(p) == a.sha256 || error("sha256 of $p is not the pinned $(a.sha256)")
            return p
        end
    end
    dir = mkpath(joinpath(cache_dir(), "archives"))
    path = joinpath(dir, a.file)
    if !isfile(path)
        tmp = tempname(dir)
        @info "Downloading $(a.url)"
        Downloads.download(a.url, tmp)
        got = file_sha256(tmp)
        if got != a.sha256
            rm(tmp; force = true)
            error("sha256 of the download of $(a.url) is $got, not the pinned $(a.sha256)")
        end
        mv(tmp, path)
    end
    file_sha256(path) == a.sha256 || error("sha256 of $path is not the pinned $(a.sha256); remove it")
    return path
end

# Extract `members` (unzip patterns) of `zip` into a new directory that then becomes
# `dest`. Archives are untrusted: entries with absolute or `..` paths or symbolic links
# are refused, and nothing is extracted on top of existing files.
function extract!(zip, dest, members...; junk = false)
    isdir(dest) && return dest
    Sys.which("unzip") === nothing && error("the gallery needs the `unzip` program")
    names = readlines(`unzip -Z1 $zip`)
    any(n -> startswith(n, "/") || occursin(r"(^|/)\.\.(/|$)", n), names) && error("$zip contains unsafe paths")
    any(l -> startswith(l, 'l'), readlines(`unzip -Z $zip`)) && error("$zip contains symbolic links")
    tmp = mktempdir(dirname(dest); cleanup = false)
    try
        flags = junk ? ["-q", "-j"] : ["-q"]
        run(pipeline(`unzip $flags $zip $members -d $tmp`; stdout = devnull))
        mv(tmp, dest)
    catch
        rm(tmp; recursive = true, force = true)
        rethrow()
    end
    return dest
end

"Path of the mf6 6.8.2 executable in the cache (extracted from the verified archive)."
function mf6_executable()
    zip = archive_path(:mf6)
    dir = extract!(zip, joinpath(cache_dir(), "mf$(MF6_VERSION)-$(ARCHIVES.mf6.sha256[1:12])"), "$(ARCHIVES.mf6.subdir)/bin/mf6"; junk = true)
    exe = joinpath(dir, "mf6")
    chmod(exe, 0o755)
    return exe
end

"""
Directory of the unmodified input of MF6 example `name` in the cache. `name` is the path
of the example in `mf6examples.zip`, e.g. `"ex-gwf-toth"` or
`"ex-gwf-hwell/ex-gwf-hwell-a"` (the `path` of validation/examples.toml).
"""
function example_input(name::AbstractString)
    zip = archive_path(:mf6examples)
    root = mkpath(joinpath(cache_dir(), "mf6examples-$(ARCHIVES.mf6examples.sha256[1:12])"))
    dest = joinpath(root, name)
    isdir(dest) && return dest
    tmp = joinpath(root, ".x-" * replace(name, "/" => "_"))
    extract!(zip, tmp, "$name/*")
    mkpath(dirname(dest))
    mv(joinpath(tmp, name), dest)
    rm(tmp; recursive = true)
    return dest
end

"Call `f()` with Info messages hidden (warnings are still shown)."
quietly(f) = Logging.with_logger(f, Logging.ConsoleLogger(stderr, Logging.Warn))

# --- Runs ------------------------------------------------------------------------------

"""
    mf6_run(dir) -> Float64

Run mf6 6.8.2 in the simulation directory `dir` (which must hold `mfsim.nam`) and return
the wall time in seconds. Throws if mf6 does not terminate normally or a step does not
converge.
"""
function mf6_run(dir::AbstractString)
    exe = mf6_executable()
    t = @elapsed run(pipeline(Cmd(`$exe`; dir); stdout = joinpath(dir, "mf6-stdout.txt"), stderr = joinpath(dir, "mf6-stderr.txt")))
    chk = RunReference.check_listings(dir)
    chk["converged"] || error("mf6 did not converge in $dir (see mfsim.lst)")
    write(joinpath(dir, "mf6-time.txt"), string(t))
    return t
end

"""
    example_run(name; mode = :auto) -> NamedTuple

The MF6 run of example `name` that the page compares with: the example's input with the
tight closures of the parity tests set in its IMS file (`mode = "tight"`; `:auto` falls back
to `"tight-relative"` where mf6 does not converge with the absolute residual criterion, as
the reference runs do), run with mf6 in the cache. Reused when it exists.
Returns `(; dir, mode, mf6_time)`.
"""
function example_run(name::AbstractString; mode = :auto)
    modes = mode === :auto ? ("tight", "tight-relative") : (String(mode),)
    for m in modes
        dir = joinpath(cache_dir(), "runs", name, m)
        done = joinpath(dir, "mf6-time.txt")
        isfile(done) && return (; dir, mode = m, mf6_time = parse(Float64, read(done, String)))
        m != last(modes) && isfile(joinpath(dir, "mf6-failed.txt")) && continue
        ispath(dir) && rm(dir; recursive = true)
        mkpath(dirname(dir))
        cp(example_input(name), dir)
        for f in RunReference.ims_files(dir)
            RunReference.patch_ims(joinpath(dir, f), RunReference.CLOSURES[m])
        end
        try
            t = mf6_run(dir)
            return (; dir, mode = m, mf6_time = t)
        catch err
            m == last(modes) && rethrow()
            write(joinpath(dir, "mf6-failed.txt"), sprint(showerror, err))
            @info "mf6 did not converge on $name with the $m closures; trying the next ones"
        end
    end
    return error("unreachable")
end

"""
    run_modflowgpu(rundir, precision; kwargs...) -> NamedTuple

Run ModflowGPU on the CPU in `precision` on the simulation in `rundir`, with output in
`rundir/modflowgpu-<precision>`. Info messages (such as the preconditioner replacement)
are not shown. Returns `(; result, time, outdir)`, `time` the wall time in seconds.
"""
function run_modflowgpu(rundir::AbstractString, precision::Symbol; kwargs...)
    outdir = mkpath(joinpath(rundir, "modflowgpu-$precision"))
    local r
    t = Logging.with_logger(Logging.ConsoleLogger(stderr, Logging.Warn)) do
        @elapsed r = run_simulation(rundir; precision, outdir, kwargs...)
    end
    r.converged || error("ModflowGPU did not converge on $rundir")
    return (; result = r, time = t, outdir)
end

_only_file(dir, ext) = only(filter(f -> endswith(f, ext), readdir(dir; join = true)))
_inactive(v) = abs(v) >= 1.0e30

# The rounding of a value printed in a listing table (as test/run_parity.jl).
_listing_round(v) = (v != 0 && (abs(v) < 0.1 || abs(v) >= 9.99999e11)) ? 5.0e-5 * abs(v) : 5.0e-5

"""
    compare_with_mf6(rundir, run) -> NamedTuple

Compare the output of a ModflowGPU run (from [`run_modflowgpu`](@ref)) with mf6's output
in `rundir`, as the parity tests do: `dh`, the largest head difference over all cells and
saved steps (inactive and dry cells left out); `budget_cbc`, the largest difference of a
budget-file flow (all records but DATA-SPDIS) relative to the total inflow of its step;
`budget_listing`, the same for the rates of the listing budget tables (beyond their
printed rounding); `budget`, the larger of the two; and `pd`, the largest percent
discrepancy of ModflowGPU's budgets. With the Newton–Raphson formulation, cells whose
mf6 head is below the cell bottom are dry (their head is not unique) and left out of `dh`,
as in the parity tests; `ndry` counts them (cell-steps). With `shift = true` (a model whose
heads are defined only up to a constant, e.g. ex-gwf-whirl: steady state with wells only)
the mean head difference of each step is removed first, as in the parity tests, and the
largest such shift is returned as `shift`.
"""
function compare_with_mf6(rundir::AbstractString, run; shift::Bool = false)
    r = run.result
    a, b = open_head_file(r.head_file), open_head_file(_only_file(rundir, ".hds"))
    length(a) == length(b) || error("the head files have $(length(a)) and $(length(b)) steps")
    m = only(ModflowGPU.load_simulation(rundir).models)
    bot = nothing
    if m.newton
        conn = ModflowGPU.build_connectivity(m)
        bot = fill(-Inf, Int(conn.nodesuser))
        bot[conn.nodeuser] .= conn.bot
    end
    dh = 0.0
    ndry = 0
    cmax = 0.0
    for k in eachindex(b.steps)
        ha, hb = vec(read_head(a, k)), vec(read_head(b, k))
        if shift
            act = findall(i -> !_inactive(hb[i]), eachindex(hb))
            c = sum(ha[act] .- hb[act]) / length(act)
            cmax = max(cmax, abs(c))
            ha = ha .- c
        end
        for i in eachindex(hb)
            _inactive(hb[i]) && continue
            if !isnothing(bot) && hb[i] < bot[i]
                ndry += 1
                continue
            end
            dh = max(dh, abs(ha[i] - hb[i]))
        end
    end
    tin = Dict((t.kstp, t.kper) => ModflowGPU.total_rate_in(t) for t in r.budgets)
    dq = 0.0
    mf6_cbc = filter(f -> endswith(f, r"\.(cbc|bud)"), readdir(rundir; join = true))
    if !isempty(mf6_cbc) && isfile(r.budget_file)
        ba, bb = open_budget_file(r.budget_file), open_budget_file(only(mf6_cbc))
        for (i, rec) in enumerate(bb.records)
            strip(rec.text) == "DATA-SPDIS" && continue
            pak = rec.imeth == 6 ? strip(rec.paknam2) : nothing
            j = only(ModflowGPU.find_budget_records(ba; text = strip(rec.text), kstp = rec.kstp, kper = rec.kper, paknam2 = pak))
            qa, qb = read_budget(ba, j), read_budget(bb, i)
            va, vb = qa isa ModflowGPU.BudgetList ? (qa.q, qb.q) : (vec(qa), vec(qb))
            isempty(vb) && continue
            dq = max(dq, maximum(abs.(va .- vb)) / tin[(Int(rec.kstp), Int(rec.kper))])
        end
    end
    dl = 0.0
    ta = read_listing_budgets(r.listing_file)
    for tb in read_listing_budgets(_only_file_lst(rundir))
        k = findfirst(x -> (x.kstp, x.kper) == (tb.kstp, tb.kper), ta)
        isnothing(k) && continue
        for (x, y) in zip(ta[k].terms, tb.terms), f in (:rate_in, :rate_out)
            v, w = getfield(x, f), getfield(y, f)
            dl = max(dl, max(abs(v - w) - _listing_round(v) - _listing_round(w), 0.0) / tb.total_rate_in)
        end
    end
    pd = maximum(t -> abs(ModflowGPU.percent_discrepancy(t)), r.budgets)
    return (; dh, budget_cbc = dq, budget_listing = dl, budget = max(dq, dl), pd, ndry, shift = cmax)
end

# The model listing file of mf6 in `dir` (not mfsim.lst).
_only_file_lst(dir) = only(filter(f -> endswith(f, ".lst") && basename(f) != "mfsim.lst", readdir(dir; join = true)))

# --- Comparison table ------------------------------------------------------------------

const PARITY_FILE = joinpath(DOCS, "src", "assets", "parity.toml")

"The row of model `name` in docs/src/assets/parity.toml (the parity table of the manual)."
function parity_row(name::AbstractString)
    d = TOML.parsefile(PARITY_FILE)
    i = findfirst(r -> r["name"] == name, d["examples"])
    isnothing(i) && error("$name is not in $PARITY_FILE")
    return d["examples"][i]
end

_e(x) = x == 0 ? "0" : @sprintf("%.1e", x)

# Acceptance limits of the parity tests (Comparing results page).
const LIMITS = (double = (dh = 1.0e-6, budget = 1.0e-6), single = (dh = 1.0e-3, budget = 1.0e-4))

"""
    comparison_table(parity_name, stats) -> Markdown.MD

The comparison table of an example page: per preset (`stats[:double]`, `stats[:single]`,
each a `(; cmp, time)` with `cmp` from [`compare_with_mf6`](@ref) and `time` the CPU wall
time of the run in this build) the largest head and budget differences against mf6, next
to the values of the parity table (row `parity_name` of docs/src/assets/parity.toml, CPU
backend), and the wall times on the CPU (this build) and on the GPU (parity table, ROCm).
`unit` is the length unit of the model.
Throws if a difference of this build exceeds the acceptance limits of the parity tests
(the head limit is relative to the head range where the parity tests make it so;
`budget_limit_double` and `budget_limit_single` replace the budget limits for the models
whose parity test uses other criteria, see `run_parity_options` in test/run_parity.jl).
"""
function comparison_table(
        parity_name::AbstractString, stats; unit = "m", dh_limit_double = LIMITS.double.dh,
        budget_limit_double = LIMITS.double.budget, budget_limit_single = LIMITS.single.budget,
    )
    row = parity_row(parity_name)
    res = row["results"]
    io = IOBuffer()
    println(io, "| Preset | max \\|Δh\\| ($unit), this build | max \\|Δh\\| ($unit), parity table | Budget, this build | Budget, parity table | CPU time, this build (s) | GPU time, parity table (s) |")
    println(io, "| :-- | --: | --: | --: | --: | --: | --: |")
    for p in (:double, :mixed, :single)
        haskey(stats, p) || continue
        s = stats[p]
        lim = p === :single ? (; dh = LIMITS.single.dh, budget = budget_limit_single) : (; dh = dh_limit_double, budget = budget_limit_double)
        s.cmp.dh < lim.dh || error("$parity_name :$p: max |Δh| = $(s.cmp.dh) exceeds $(lim.dh)")
        s.cmp.budget < lim.budget || error("$parity_name :$p: budget difference $(s.cmp.budget) exceeds $(lim.budget)")
        cpu = res["cpu"][string(p)]
        gpu = haskey(res, "rocm") ? @sprintf("%.2f", res["rocm"][string(p)]["wall_time"]) : "–"
        println(
            io, "| `:$p` | ", _e(s.cmp.dh), " | ", _e(cpu["max_dh"]), " | ", _e(s.cmp.budget), " | ",
            _e(max(cpu["max_budget_cbc"], cpu["max_budget_listing"])), " | ", @sprintf("%.2f", s.time), " | ", gpu, " |",
        )
    end
    println(io)
    println(io, "Budget: the largest difference of a budget-file flow or a listing-table rate, relative to the total inflow of its time step.")
    return Markdown.parse(String(take!(io)))
end

# --- Observation output ----------------------------------------------------------------------

"""
    obs_pair(mf6dir, run, file) -> NamedTuple

The observation file `file` (a file name such as `"ex-gwf-fhb.obs.head.csv"`) of a ModflowGPU
run (`run.result.obs_files`, from [`run_modflowgpu`](@ref)) and of the mf6 run in `mf6dir`,
as `(; gpu, mf6)`, two `ObservationFile`s.
"""
function obs_pair(mf6dir::AbstractString, run, file::AbstractString)
    gpu = only(filter(f -> basename(f) == file, run.result.obs_files))
    return (; gpu = read_observations(gpu), mf6 = read_observations(joinpath(mf6dir, file)))
end

"The column of observation `name` of `obs` as `Float64`s, `NaN` where the package observation matched no record (`ModflowGPU.OBSNODATA`)."
function obs_masked(obs, name::AbstractString)
    k = findfirst(==(uppercase(name)), obs.names)
    k === nothing && error("observation $name is not in the file (has $(obs.names))")
    return [v >= 0.99 * ModflowGPU.OBSNODATA ? NaN : Float64(v) for v in obs.values[:, k]]
end

"""
    obs_plot!(ax, pair, names; colors, stride = 1, labels = names)

Draw the observations `names` of `pair` (from [`obs_pair`](@ref)) on `ax`: mf6 as lines and
ModflowGPU as open circles (every `stride`-th time step), one colour per observation.
"""
function obs_plot!(ax, pair, names; colors = Makie.wong_colors(), stride::Integer = 1, labels = names)
    for (k, name) in enumerate(names)
        color = colors[mod1(k, length(colors))]
        lines!(ax, pair.mf6.times, obs_masked(pair.mf6, name); color, label = "MODFLOW 6 $(labels[k])")
        i = 1:stride:length(pair.gpu.times)
        scatter!(ax, pair.gpu.times[i], obs_masked(pair.gpu, name)[i]; color = :transparent, strokecolor = color, strokewidth = 1, markersize = 6, label = "ModflowGPU $(labels[k])")
    end
    return ax
end

"""
    obs_difference_table(mf6dir, runs, files; unit = "m") -> Markdown.MD

The largest absolute difference between the observations of the ModflowGPU runs `runs` (a
`Dict` precision => run) and of mf6, over all time steps, for every observation of every file
in `files` (steps where either has no data are left out). `unit` is the unit of the values.
Throws if the time steps or the observation names differ.
"""
function obs_difference_table(mf6dir::AbstractString, runs, files; unit = "m")
    precs = [p for p in (:double, :mixed, :single) if haskey(runs, p)]
    io = IOBuffer()
    println(io, "| File | Observation | ", join(("max \\|difference\\| ($unit), `:$p`" for p in precs), " | "), " |")
    println(io, "| :-- | :-- | ", join(("--:" for _ in precs), " | "), " |")
    for file in files
        pairs = Dict(p => obs_pair(mf6dir, runs[p], file) for p in precs)
        for (k, name) in enumerate(pairs[first(precs)].mf6.names)
            ds = map(precs) do p
                pr = pairs[p]
                pr.gpu.names == pr.mf6.names && length(pr.gpu.times) == length(pr.mf6.times) || error("$file: observations of ModflowGPU and mf6 differ")
                a, b = obs_masked(pr.gpu, name), obs_masked(pr.mf6, name)
                d = [abs(x - y) for (x, y) in zip(a, b) if !isnan(x) && !isnan(y)]
                isempty(d) ? NaN : maximum(d)
            end
            println(io, "| `", file, "` | `", name, "` | ", join((_e(d) for d in ds), " | "), " |")
        end
    end
    return Markdown.parse(String(take!(io)))
end

# --- Structured grids --------------------------------------------------------------------

"""
    DisGrid

A structured grid read from MF6's binary grid file (`read_dis_grid`): `nlay`, `nrow`,
`ncol`, `delr`, `delc`, `top` (`(ncol, nrow)`), `botm` and `idomain` (`(ncol, nrow, nlay)`),
`icelltype`, the cell edges `xe` (west to east) and `ye` (north to south, as rows run),
the cell centres `xc` and `yc`, and the connectivity `ia`, `ja` (user nodes, as the
FLOW-JA-FACE record of the budget file).
"""
struct DisGrid
    nlay::Int
    nrow::Int
    ncol::Int
    delr::Vector{Float64}
    delc::Vector{Float64}
    top::Matrix{Float64}
    botm::Array{Float64, 3}
    idomain::Array{Int32, 3}
    icelltype::Array{Int32, 3}
    xe::Vector{Float64}
    ye::Vector{Float64}
    xc::Vector{Float64}
    yc::Vector{Float64}
    ia::Vector{Int32}
    ja::Vector{Int32}
end

"Read the DIS grid of the simulation in `dir` from its `.dis.grb` file."
function read_dis_grid(dir::AbstractString)
    g = read_grb(_only_file(dir, ".grb"))
    g.grid == "DIS" || error("only DIS grids are supported here, got $(g.grid)")
    d = g.data
    nlay, nrow, ncol = Int(d["NLAY"]), Int(d["NROW"]), Int(d["NCOL"])
    delr, delc = d["DELR"], d["DELC"]
    x0, y0 = d["XORIGIN"], d["YORIGIN"]
    xe = x0 .+ [0.0; cumsum(delr)]
    ye = y0 .+ sum(delc) .- [0.0; cumsum(delc)]
    xc = (xe[1:(end - 1)] .+ xe[2:end]) ./ 2
    yc = (ye[1:(end - 1)] .+ ye[2:end]) ./ 2
    return DisGrid(
        nlay, nrow, ncol, delr, delc, reshape(d["TOP"], ncol, nrow), reshape(d["BOTM"], ncol, nrow, nlay),
        reshape(d["IDOMAIN"], ncol, nrow, nlay), reshape(d["ICELLTYPE"], ncol, nrow, nlay), xe, ye, xc, yc, d["IA"], d["JA"],
    )
end

"User node number of cell `(layer, row, column)`."
node(g::DisGrid, l, r, c) = (l - 1) * g.nrow * g.ncol + (r - 1) * g.ncol + c

"Cell `(layer, row, column)` of user node `n`."
function cell_of(g::DisGrid, n)
    l, rem = divrem(n - 1, g.nrow * g.ncol)
    r, c = divrem(rem, g.ncol)
    return (l + 1, r + 1, c + 1)
end

"`a` with inactive and dry cells (|value| ≥ 1e30) set to NaN."
mask_inactive(a) = map(v -> _inactive(v) ? NaN : v, a)

"The `(ncol, nrow)` values of layer `l` of a head array `h[col, row, layer]`, masked."
layer_values(h, l) = mask_inactive(h[:, :, l])
layer_values(::DisGrid, h, l) = layer_values(h, l)

"""
    budget_cells(g, budget_file, text) -> Vector{NTuple{3, Int}}

The cells `(layer, row, column)` of the list records with budget text `text` (e.g.
`"CHD"`, `"WEL"`, `"DRN"`, `"RIV"`) of the first saved step that has any, the boundary
cells of the packages of that type.
"""
function budget_cells(g::DisGrid, budget_file::AbstractString, text::AbstractString)
    return [cell_of(g, Int(n)) for n in _first_step_nodes(budget_file, text)]
end

"""
    input_nodes(dir, ftype) -> Vector{Int}

The user nodes of the list-based boundary packages of type `ftype` (e.g. `"CHD"`) of the
simulation in `dir`, over all stress periods, read from the input with
`load_simulation` (for models that write no budget file; this reads ModflowGPU's
internal input structures, which may change).
"""
function input_nodes(dir::AbstractString, ftype::AbstractString)
    m = only(load_simulation(dir).models)
    nodes = Int[]
    for b in m.boundaries
        (b.schema !== nothing && lowercase(b.schema.package) == "gwf-" * lowercase(ftype)) || continue
        foreach(p -> append!(nodes, p.nodeu), b.periods.data)
    end
    return sort!(unique!(nodes))
end

"""
    specific_discharge(g, flowja, h) -> (qx, qy)

The specific discharge at the cell centres, `(ncol, nrow, nlay)` arrays, from the
FLOW-JA-FACE record `flowja` and the heads `h` (for the saturated thickness of convertible
cells), as MF6 computes it for SAVE_SPECIFIC_DISCHARGE on a regular DIS grid (`calc_spdis`
in gwf-npf.f90): the mean of the face velocities in x and in y; `qx` positive east, `qy`
positive north.
"""
function specific_discharge(g::DisGrid, flowja, h)
    qx = zeros(g.ncol, g.nrow, g.nlay)
    qy = zeros(g.ncol, g.nrow, g.nlay)
    function sat(l, r, c)
        t = l == 1 ? g.top[c, r] : g.botm[c, r, l - 1]
        b = g.botm[c, r, l]
        g.icelltype[c, r, l] == 0 && return t - b
        hv = h[c, r, l]
        return _inactive(hv) ? 0.0 : clamp(hv - b, 0.0, t - b)
    end
    for n in 1:(length(g.ia) - 1)
        l, r, c = cell_of(g, n)
        sx = sy = 0.0
        nx = ny = 0
        for ii in g.ia[n]:(g.ia[n + 1] - 1)
            m = Int(g.ja[ii])
            m == n && continue
            lm, rm, cm = cell_of(g, m)
            lm == l || continue
            dz = (sat(l, r, c) + sat(lm, rm, cm)) / 2
            if rm == r          # x face: flowja is the flow into n from m
                a = g.delc[r] * dz
                v = a > 0 ? flowja[ii] / a : 0.0
                sx += cm > c ? -v : v
                nx += 1
            else                # y face; rows run north to south
                a = g.delr[c] * dz
                v = a > 0 ? flowja[ii] / a : 0.0
                sy += rm < r ? -v : v
                ny += 1
            end
        end
        qx[c, r, l] = nx > 0 ? sx / nx : 0.0
        qy[c, r, l] = ny > 0 ? sy / ny : 0.0
    end
    return qx, qy
end

# --- Unstructured grids (DISV and DISU) --------------------------------------------------

"""
    VertexGrid

A DISV or DISU grid read from MF6's binary grid file (`read_vertex_grid`): `kind`
(`"DISV"` or `"DISU"`), `nlay` and `ncpl` (cells per layer; for DISU `nlay = 1` and `ncpl`
the number of nodes), the cell `polygons` and centres `xc`, `yc` of a layer (DISV: per
cell2d; DISU: per node), and per node `top`, `bot`, `idomain` and `icelltype`, and the
connectivity `ia`, `ja` (user nodes). DISV node `n` is cell2d `mod1(n, ncpl)` of layer
`cld(n, ncpl)`.
"""
struct VertexGrid
    kind::String
    nlay::Int
    ncpl::Int
    polygons::Vector{Vector{Point2d}}
    xc::Vector{Float64}
    yc::Vector{Float64}
    top::Vector{Float64}
    bot::Vector{Float64}
    idomain::Vector{Int32}
    icelltype::Vector{Int32}
    ia::Vector{Int32}
    ja::Vector{Int32}
end

"Read the DISV or DISU grid of the simulation in `dir` from its `.disv.grb` or `.disu.grb` file."
function read_vertex_grid(dir::AbstractString)
    g = read_grb(_only_file(dir, ".grb"))
    d = g.data
    x0, y0 = d["XORIGIN"], d["YORIGIN"]
    v = d["VERTICES"]
    iav, jav = d["IAVERT"], d["JAVERT"]
    polys = map(1:(length(iav) - 1)) do i
        ids = Int.(jav[iav[i]:(iav[i + 1] - 1)])
        length(ids) > 1 && first(ids) == last(ids) && pop!(ids)   # closed rings repeat the first vertex
        [Point2d(x0 + v[1, k], y0 + v[2, k]) for k in ids]
    end
    if g.grid == "DISV"
        nlay, ncpl = Int(d["NLAY"]), Int(d["NCPL"])
        top = repeat(d["TOP"], nlay)
        bot = d["BOTM"]
        for l in 2:nlay, i in 1:ncpl
            top[(l - 1) * ncpl + i] = bot[(l - 2) * ncpl + i]
        end
    elseif g.grid == "DISU"
        nlay, ncpl = 1, Int(d["NODES"])
        top, bot = d["TOP"], d["BOT"]
    else
        error("only DISV and DISU grids are supported here, got $(g.grid)")
    end
    return VertexGrid(g.grid, nlay, ncpl, polys, x0 .+ d["CELLX"], y0 .+ d["CELLY"], top, bot, d["IDOMAIN"], d["ICELLTYPE"], d["IA"], d["JA"])
end

"""
    read_grid(dir) -> DisGrid or VertexGrid

The grid of the simulation in `dir`, from its binary grid file: a [`DisGrid`](@ref) for
DIS, a [`VertexGrid`](@ref) for DISV and DISU.
"""
read_grid(dir::AbstractString) = endswith(_only_file(dir, ".grb"), ".dis.grb") ? read_dis_grid(dir) : read_vertex_grid(dir)

"The values of layer `l` of a DISV head array (`(ncpl, 1, nlay)`), masked, as a vector."
layer_values(g::VertexGrid, h, l) = mask_inactive(vec(h)[((l - 1) * g.ncpl + 1):(l * g.ncpl)])

"The nodes of the list records with budget text `text` of the first saved step that has any (all packages of that type)."
function budget_cells(::VertexGrid, budget_file::AbstractString, text::AbstractString)
    return Int.(_first_step_nodes(budget_file, text))
end

# The nodes of every list record with budget text `text` in the first saved step that has
# a non-empty one.
function _first_step_nodes(budget_file, text)
    bf = open_budget_file(budget_file)
    idx = ModflowGPU.find_budget_records(bf; text)
    for step in unique((bf.records[i].kstp, bf.records[i].kper) for i in idx)
        nodes = reduce(vcat, [read_budget(bf, i).id1 for i in idx if (bf.records[i].kstp, bf.records[i].kper) == step])
        isempty(nodes) || return nodes
    end
    return Int32[]
end

# The edge that the polygons `p` and `q` share: its two end points, or `nothing`.
function _shared_edge(p, q)
    common = [a for a in p if any(b -> isapprox(a, b; atol = 1.0e-6 * max(1.0, maximum(abs, a))), q)]
    return length(common) == 2 ? (common[1], common[2]) : nothing
end

"""
    specific_discharge(g::VertexGrid, flowja, h) -> (qx, qy)

The specific discharge at the cell centres of a DISV grid (vectors over the nodes), from
the FLOW-JA-FACE record `flowja` and the heads `h`, as MF6 computes it for
SAVE_SPECIFIC_DISCHARGE without XT3D (`calc_spdis` in gwf-npf.f90): per horizontal
connection the face velocity `flowja / (face width × saturated thickness)` along the face
normal, combined with MF6's distance weights. The face geometry (normal, width, the
distances `cl1` and `cl2` from the cell centres to the face) comes from the cell polygons.
"""
function specific_discharge(g::VertexGrid, flowja, h)
    g.kind == "DISV" || error("specific discharge is implemented for DISV grids only")
    nn = length(g.ia) - 1
    hv = vec(h)
    qx, qy = zeros(nn), zeros(nn)
    cell(n) = mod1(n, g.ncpl)
    function sat(n)
        t, b = g.top[n], g.bot[n]
        g.icelltype[n] == 0 && return t - b
        return _inactive(hv[n]) ? 0.0 : clamp(hv[n] - b, 0.0, t - b)
    end
    for n in 1:nn
        nix, niy, di, vi = Float64[], Float64[], Float64[], Float64[]
        cn = cell(n)
        pc = Point2d(g.xc[cn], g.yc[cn])
        for ii in g.ia[n]:(g.ia[n + 1] - 1)
            m = Int(g.ja[ii])
            (m == n || cld(m, g.ncpl) != cld(n, g.ncpl)) && continue
            cm = cell(m)
            e = _shared_edge(g.polygons[cn], g.polygons[cm])
            e === nothing && continue
            a, b = Float64.(e[1]), Float64.(e[2])
            t = (b - a) / sqrt(sum(abs2, b - a))
            nrm = [t[2], -t[1]]
            pm = [g.xc[cm], g.yc[cm]]
            p = Float64.(pc)
            sum(nrm .* (pm - p)) < 0 && (nrm = -nrm)      # normal from n towards m
            cl1 = abs(sum(nrm .* (a - p)))
            cl2 = abs(sum(nrm .* (a - pm)))
            dltot = sqrt(sum(abs2, pm - p))
            area = sqrt(sum(abs2, b - a)) * (sat(n) + sat(m)) / 2
            push!(nix, -nrm[1]); push!(niy, -nrm[2])
            push!(di, dltot * cl1 / (cl1 + cl2))
            push!(vi, area > 0 ? flowja[ii] / area : 0.0)
        end
        isempty(vi) && continue
        wix, wiy = di .* abs.(nix), di .* abs.(niy)
        sx, sy = sum(wix) * (1 + 1.0e-10), sum(wiy) * (1 + 1.0e-10)
        wix, wiy = (sx .- wix) .* abs.(nix), (sy .- wiy) .* abs.(niy)
        bix, biy = wix .* sign.(nix), wiy .* sign.(niy)
        sx, sy = sum(wix .* abs.(nix)), sum(wiy .* abs.(niy))
        sx > 0 && (bix ./= sx)
        sy > 0 && (biy ./= sy)
        axy, ayx = sum(bix .* niy), sum(biy .* nix)
        vx = sum((bix .- axy .* biy) .* vi)
        vy = sum((biy .- ayx .* bix) .* vi)
        den = 1 - axy * ayx
        qx[n], qy[n] = den != 0 ? (vx / den, vy / den) : (vx, vy)
    end
    return qx, qy
end

"Colour the cells of a layer of `g` with the values `a` (one per cell of the layer)."
map_heatmap!(ax, g::VertexGrid, a; kwargs...) = poly!(ax, g.polygons; color = Float64.(vec(a)), strokewidth = 0, kwargs...)

"The cell outlines of a layer of `g`."
map_grid!(ax, g::VertexGrid; color = :black, linewidth = 0.5) = poly!(ax, g.polygons; color = :transparent, strokecolor = color, strokewidth = linewidth)

"Fill the cells of the nodes `nodes` (of any layer) of `g`."
map_cells!(ax, g::VertexGrid, nodes; color, kwargs...) = poly!(ax, [g.polygons[mod1(n, g.ncpl)] for n in nodes]; color, kwargs...)

"""
    map_vectors!(ax, g::VertexGrid, qx, qy; normalize = false, scale = 0.8)

Arrows of the specific discharge `qx`, `qy` (one value per cell of the layer) at the cell
centres; the longest arrow (or every arrow, with `normalize`) is `scale` times the median
width (in x) of the cells.
"""
function map_vectors!(ax, g::VertexGrid, qx, qy; normalize = false, color = RGBf(0.75, 0.75, 0.75), scale = 0.8, kwargs...)
    pts = [Point2f(g.xc[i], g.yc[i]) for i in 1:g.ncpl]
    dirs = [Vec2f(qx[i], qy[i]) for i in 1:g.ncpl]
    widths = sort([-(reverse(extrema(first.(p)))...) for p in g.polygons])
    w = widths[cld(length(widths), 2)] * scale
    if normalize
        dirs = [sqrt(sum(abs2, d)) > 0 ? d / sqrt(sum(abs2, d)) * w : d for d in dirs]
    else
        dirs = dirs .* Float32(w / maximum(d -> sqrt(sum(abs2, d)), dirs))
    end
    return arrows2d!(ax, pts, dirs; color, align = :center, shaftwidth = 1, tipwidth = 5, tiplength = 5, kwargs...)
end

# --- Plot helpers --------------------------------------------------------------------------

"Tick labels as whole numbers (`xtickformat = intformat`), instead of `5.0×10³`."
intformat(vs) = [string(round(Int, v)) for v in vs]

"Tick labels with two significant digits in exponent form, for colour bars of small differences."
expformat(vs) = [v == 0 ? "0" : @sprintf("%.1e", v) for v in vs]

"Colours of boundary cells, as FloPy's `plot_bc` draws them."
const BC_COLORS = Dict("CHD" => :navy, "WEL" => :red, "DRN" => :yellow, "RIV" => :teal, "GHB" => :cyan, "EVT" => :peru, "RCH" => :green)

# Values `(ncol, nrow)` in MF6 row order → y increasing for Makie.
_flip(a) = a[:, end:-1:1]

"Heatmap of the `(ncol, nrow)` values `a` of a layer over the cells of `g`."
map_heatmap!(ax, g::DisGrid, a; kwargs...) = heatmap!(ax, g.xe, reverse(g.ye), _flip(a); kwargs...)

"Contours of the `(ncol, nrow)` values `a` of a layer, at the cell centres."
map_contour!(ax, g::DisGrid, a; kwargs...) = contour!(ax, g.xc, reverse(g.yc), _flip(a); kwargs...)

"The cell outlines of a layer of `g`."
function map_grid!(ax, g::DisGrid; color = :black, linewidth = 0.5)
    x0, x1 = extrema(g.xe)
    y0, y1 = extrema(g.ye)
    segs = [Point2f(x, y0) => Point2f(x, y1) for x in g.xe]
    append!(segs, [Point2f(x0, y) => Point2f(x1, y) for y in g.ye])
    return linesegments!(ax, segs; color, linewidth)
end

"Fill the cells `(row, column)` (or `(layer, row, column)`, layer ignored) of a layer of `g`."
function map_cells!(ax, g::DisGrid, cells; color, kwargs...)
    rects = [Rect2f(g.xe[c[end]], g.ye[c[end - 1] + 1], g.delr[c[end]], g.delc[c[end - 1]]) for c in cells]
    return poly!(ax, rects; color, kwargs...)
end

"Arrows of the specific discharge `qx`, `qy` (`(ncol, nrow)`) of a layer, at the cell centres."
function map_vectors!(ax, g::DisGrid, qx, qy; normalize = true, color = RGBf(0.75, 0.75, 0.75), lengthscale = 0.8, kwargs...)
    pts = vec([Point2f(g.xc[c], g.yc[r]) for c in 1:g.ncol, r in 1:g.nrow])
    d = minimum([g.delr; g.delc]) * lengthscale
    function arrow(v)
        s = sqrt(sum(abs2, v))
        return Vec2f(normalize ? (s > 0 ? v / s * d : zero(v)) : v)
    end
    dirs = vec([arrow(Vec2f(qx[c, r], qy[c, r])) for c in 1:g.ncol, r in 1:g.nrow])
    return arrows2d!(ax, pts, dirs; color, align = :center, shaftwidth = 1, tipwidth = 5, tiplength = 5, kwargs...)
end

# Cells of a cross-section along a row (`row = i`: x–z) or a column (`column = j`: y–z):
# horizontal edges, and per (cell, layer) the bottom and top.
function _section(g::DisGrid; row = nothing, column = nothing)
    if row !== nothing
        he = g.xe
        tops = [l == 1 ? g.top[c, row] : g.botm[c, row, l - 1] for c in 1:g.ncol, l in 1:g.nlay]
        bots = [g.botm[c, row, l] for c in 1:g.ncol, l in 1:g.nlay]
        pick = a -> a[:, row, :]
    else
        he = g.ye
        tops = [l == 1 ? g.top[column, r] : g.botm[column, r, l - 1] for r in 1:g.nrow, l in 1:g.nlay]
        bots = [g.botm[column, r, l] for r in 1:g.nrow, l in 1:g.nlay]
        pick = a -> a[column, :, :]
    end
    return he, tops, bots, pick
end

"""
    section_heatmap!(ax, g, a; row or column, kwargs...)

Colour the cells of a cross-section of the `(ncol, nrow, nlay)` values `a` along a row
(x–z) or a column (y–z), each cell drawn from its bottom to its top.
"""
function section_heatmap!(ax, g::DisGrid, a; row = nothing, column = nothing, kwargs...)
    he, tops, bots, pick = _section(g; row, column)
    v = pick(a)
    rects = Rect2f[]
    vals = Float64[]
    for i in axes(tops, 1), l in axes(tops, 2)
        x0, x1 = minmax(he[i], he[i + 1])
        isnan(v[i, l]) && continue
        push!(rects, Rect2f(x0, bots[i, l], x1 - x0, tops[i, l] - bots[i, l]))
        push!(vals, v[i, l])
    end
    return poly!(ax, rects; color = vals, strokewidth = 0, kwargs...)
end

"The cell outlines of a cross-section along a row or a column."
function section_grid!(ax, g::DisGrid; row = nothing, column = nothing, color = :black, linewidth = 0.5)
    he, tops, bots, _ = _section(g; row, column)
    rects = [Rect2f(min(he[i], he[i + 1]), bots[i, l], abs(he[i + 1] - he[i]), tops[i, l] - bots[i, l]) for i in axes(tops, 1) for l in axes(tops, 2)]
    return poly!(ax, rects; color = :transparent, strokecolor = color, strokewidth = linewidth)
end

"""
    section_contour!(ax, g, a; row or column, kwargs...)

Contours of the `(ncol, nrow, nlay)` values `a` in a cross-section, at the cell centres
(for grids whose layer elevations do not vary along the section).
"""
function section_contour!(ax, g::DisGrid, a; row = nothing, column = nothing, kwargs...)
    he, tops, bots, pick = _section(g; row, column)
    hc = (he[1:(end - 1)] .+ he[2:end]) ./ 2
    zc = vec(sum((tops .+ bots) ./ 2; dims = 1) ./ size(tops, 1))
    v = pick(a)
    o = sortperm(hc)
    p = sortperm(zc)
    return contour!(ax, hc[o], zc[p], v[o, p]; kwargs...)
end

"""
    budget_bars(budgets; title) -> Figure

Bars of the rates in and out of each budget term of the ModflowGPU budgets
`budgets` (a `Vector` of `name => ModelBudget`, e.g. `["ModflowGPU" => b1, "MODFLOW 6" =>
b2]`), side by side per term: inflows up, outflows down.
"""
function budget_bars(budgets; title = "", size = (640, 360), unit = "")
    names = [t.name for t in last(first(budgets)).terms]
    fig = Figure(; size)
    ax = Axis(fig[1, 1]; xticks = (1:length(names), names), ylabel = "Rate$(isempty(unit) ? "" : " ($unit)")", title, xticklabelrotation = length(names) > 4 ? π / 6 : 0.0)
    nb = length(budgets)
    w = 0.8 / nb
    colors = [:steelblue, :darkorange, :seagreen, :gray]
    for (k, (lab, b)) in enumerate(budgets)
        x = (1:length(names)) .+ (k - (nb + 1) / 2) * w
        rin = [t.rate_in for t in b.terms]
        rout = [-t.rate_out for t in b.terms]
        barplot!(ax, x, rin; width = w, color = colors[k], label = lab)
        barplot!(ax, x, rout; width = w, color = (colors[k], 0.55))
    end
    hlines!(ax, 0.0; color = :black, linewidth = 0.5)
    axislegend(ax; position = :rt, framevisible = false)
    return fig
end

"""
    difference_map(g, panels; layer = 1, unit = "") -> Figure

Maps of head differences, one panel per `label => Δh` pair (`Δh` a head array as read
from the head file of grid `g`, a [`DisGrid`](@ref) or a [`VertexGrid`](@ref)), each with
its own symmetric colour range; `unit` labels the colour bars.
"""
function difference_map(g::Union{DisGrid, VertexGrid}, panels; layer = 1, unit = "", size = (420 * length(panels), 380), xlabel = "x position", ylabel = "y position")
    fig = Figure(; size)
    for (k, (lab, d)) in enumerate(panels)
        a = layer_values(g, d, layer)
        m = maximum(x -> isnan(x) ? 0.0 : abs(x), a)
        m = m == 0 ? 1.0e-12 : m
        ax = Axis(fig[1, 2k - 1]; aspect = DataAspect(), title = lab, xlabel, ylabel)
        hm = map_heatmap!(ax, g, a; colormap = :balance, colorrange = (-m, m))
        Colorbar(fig[1, 2k], hm; label = isempty(unit) ? "Δh" : "Δh ($unit)", tickformat = expformat)
    end
    return fig
end

"A PNG file shown as an image (on a page: only `image/png`, no HTML representation)."
struct PNGFile
    path::String
end
Base.show(io::IO, ::MIME"image/png", f::PNGFile) = write(io, read(f.path))

"""
    save_figure(fig, name) -> PNGFile

Write `fig` as `<name>.png` into the figure folder of the cache (`<cache>/figures`), so a
script run on its own leaves its figures there, and return the file, which a page shows
as the figure.
"""
function save_figure(fig, name)
    resize_to_layout!(fig)
    path = joinpath(mkpath(joinpath(cache_dir(), "figures")), name * ".png")
    save(path, fig; px_per_unit = 2)
    return PNGFile(path)
end

# --- The gallery and its index ------------------------------------------------------------

"""
The pages of the gallery, in the order of the manual: the script in `examples/` =>
`(title, MF6 examples the page runs)`. Used by docs/make.jl and by
`gallery_index_table`.
"""
const GALLERY = [
    "ex-gwf-toth.jl" => ("Tóth: flow systems in a basin", ["ex-gwf-toth"]),
    "ex-gwf-hani.jl" => ("Horizontal anisotropy", ["ex-gwf-hanir", "ex-gwf-hanic", "ex-gwf-hanix"]),
    "ex-gwf-fhb.jl" => ("Time-series boundaries", ["ex-gwf-fhb"]),
    "ex-gwf-twri01.jl" => ("TWRI: layered aquifers", ["ex-gwf-twri01"]),
    "ex-gwf-bcf2ss.jl" => ("BCF2SS: wetting and drying, and Newton–Raphson", ["ex-gwf-bcf2ss-p01a", "ex-gwf-bcf2ss-p02a"]),
    "ex-gwf-bump.jl" => ("Flow around a high bottom", ["ex-gwf-bump-p01a", "ex-gwf-bump-p01b", "ex-gwf-bump-p01c"]),
    "ex-gwf-nwt-p02.jl" => ("Water table under a recharge pond", ["ex-gwf-nwt-p02a", "ex-gwf-nwt-p02b"]),
    "ex-gwf-nwt-p03.jl" => ("Water table over a rough bottom (Newton–Raphson)", ["ex-gwf-nwt-p03a", "ex-gwf-nwt-p03b"]),
    "ex-gwf-zaidel.jl" => ("Zaidel: a water table over steps (Newton–Raphson)", ["ex-gwf-zaidel"]),
    "ex-gwf-capture.jl" => ("Capture fraction of a river (Newton–Raphson)", ["ex-gwf-capture"]),
    "ex-gwf-advtidal.jl" => ("Highlands next to a tidal estuary", ["ex-gwf-advtidal"]),
    "ex-gwf-maw-p03.jl" => ("A well screened across layers", ["ex-gwf-maw-p03a", "ex-gwf-maw-p03c"]),
    "ex-gwf-hwell.jl" => ("A horizontal well", ["ex-gwf-hwell-a"]),
    "ex-gwf-lgrv-gc.jl" => ("Valley aquifer with a river", ["ex-gwf-lgrv-gc"]),
    "ex-gwf-lgrv-gr.jl" => ("Valley aquifer: the refined model", ["ex-gwf-lgrv-gr"]),
    "ex-gwf-u1disv.jl" => ("A nested grid (DISV, XT3D)", ["ex-gwf-u1disv", "ex-gwf-u1disv-x"]),
    "ex-gwf-curvilinear.jl" => ("Curvilinear grids (DISV)", ["ex-gwf-curve-90", "ex-gwf-curvilin"]),
    "ex-gwf-disvmesh.jl" => ("A triangular mesh (DISV, XT3D)", ["ex-gwf-disvmesh"]),
    "ex-gwf-whirl.jl" => ("Whirl: a rotated conductivity tensor (XT3D)", ["ex-gwf-whirl"]),
    "ex-gwf-rad-disu.jl" => ("An axisymmetric model (DISU)", ["ex-gwf-rad-disu"]),
    "synthetic-coupling.jl" => ("Coupling-type model, step by step", String[]),
]

# The page (notebook) of each in-scope example in the MODFLOW 6 examples documentation,
# where it is not the example's own name.
const NOTEBOOKS = Dict(
    "ex-gwf-hanic" => "ex-gwf-hani", "ex-gwf-hanir" => "ex-gwf-hani", "ex-gwf-hanix" => "ex-gwf-hani",
    "ex-gwf-hwell-a" => "ex-gwf-hwell", "ex-gwf-maw-p03a" => "ex-gwf-maw-p03", "ex-gwf-maw-p03c" => "ex-gwf-maw-p03",
    "ex-gwf-bcf2ss-p01a" => "ex-gwf-bcf2ss", "ex-gwf-bcf2ss-p02a" => "ex-gwf-bcf2ss",
    "ex-gwf-bump-p01a" => "ex-gwf-bump", "ex-gwf-bump-p01b" => "ex-gwf-bump", "ex-gwf-bump-p01c" => "ex-gwf-bump",
    "ex-gwf-lgrv-gc" => "ex-gwf-lgrv", "ex-gwf-lgrv-gr" => "ex-gwf-lgrv", "ex-gwf-lgrv-lgr" => "ex-gwf-lgrv",
    "ex-gwf-nwt-p02a" => "ex-gwf-nwt-p02", "ex-gwf-nwt-p02b" => "ex-gwf-nwt-p02",
    "ex-gwf-nwt-p03a" => "ex-gwf-nwt-p03", "ex-gwf-nwt-p03b" => "ex-gwf-nwt-p03",
    "ex-gwf-twri01" => "ex-gwf-twri", "ex-gwf-curve-90" => "ex-gwf-curvilinear-90", "ex-gwf-curvilin" => "ex-gwf-curvilinear",
    "ex-gwf-u1disv-x" => "ex-gwf-u1disv",
    "ex-gwf-u1gwfgwf-s1" => "ex-gwf-u1gwfgwf", "ex-gwf-u1gwfgwf-s2" => "ex-gwf-u1gwfgwf",
    "ex-gwf-u1gwfgwf-s3" => "ex-gwf-u1gwfgwf", "ex-gwf-u1gwfgwf-s4" => "ex-gwf-u1gwfgwf",
)

# Features (`requires` of validation/examples.toml) of unfinished Phase 6 cards.
const PLANNED = ["exchange" => "GWF–GWF exchanges (T6.6)", "multi_model" => "GWF–GWF exchanges (T6.6)"]

"""
    gallery_index_table() -> Markdown.MD

The table of the gallery's overview page: every in-scope MODFLOW 6 example of
validation/examples.toml (`in_plan = true`) with its grid, size, link to the MODFLOW 6
examples documentation and status: the gallery page that runs it, or "planned" with the
unfinished Phase 6 feature it needs. Throws if an example that ModflowGPU supports has no
page, or a page names an example that is not in scope, so the index cannot drift from
validation/examples.toml.

The links to the pages are written as URLs of the built site (Documenter does not rewrite
`.md` links in the output of an `@eval` block): `prettyurls` must match the setting of
`Documenter.HTML` in docs/make.jl (on in CI).
"""
function gallery_index_table(; prettyurls = get(ENV, "CI", "false") == "true")
    inv = TOML.parsefile(repo_path("validation", "examples.toml"))["examples"]
    pages = Dict(ex => (replace(script, ".jl" => ".md"), title) for (script, (title, exs)) in GALLERY for ex in exs)
    io = IOBuffer()
    println(io, "| MODFLOW 6 example | Grid | Cells | Status |")
    println(io, "| :-- | :-- | --: | :-- |")
    names = sort([k for (k, v) in inv if get(v, "in_plan", false)]; by = k -> (inv[k]["tier"], k))
    for ex in keys(pages)
        ex in names || error("gallery page for $ex, which is not an in-scope example of validation/examples.toml")
    end
    for ex in names
        v = inv[ex]
        nb = get(NOTEBOOKS, ex, ex)
        needs = unique([lab for (f, lab) in PLANNED if f in v["requires"]])
        status = if haskey(pages, ex)
            isempty(needs) || error("$ex has a gallery page but needs $(join(needs, ", "))")
            page = replace(pages[ex][1], ".md" => "")
            "[$(pages[ex][2])]($(prettyurls ? "$page/" : "examples/$page.html"))"
        else
            isempty(needs) && error("$ex is supported but has no gallery page (add one to ExampleTools.GALLERY)")
            "planned: needs " * join(needs, " and ")
        end
        println(io, "| [`$ex`](https://modflow6-examples.readthedocs.io/en/latest/_notebooks/$nb.html) | ", v["grid"], " | ", v["ncells"], " | ", status, " |")
    end
    return Markdown.parse(String(take!(io)))
end

end # module ExampleTools
