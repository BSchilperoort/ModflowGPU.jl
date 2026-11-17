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

export example_run, mf6_run, run_modflowgpu, compare_with_mf6, comparison_table, parity_row,
    DisGrid, read_dis_grid, node, cell_of, layer_values, specific_discharge, budget_cells,
    map_heatmap!, map_contour!, map_grid!, map_cells!, map_vectors!, section_heatmap!, section_grid!,
    section_contour!, budget_bars, difference_map, save_figure, intformat, expformat, BC_COLORS, mask_inactive

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

"Directory of the unmodified input of MF6 example `name` in the cache."
function example_input(name::AbstractString)
    zip = archive_path(:mf6examples)
    root = mkpath(joinpath(cache_dir(), "mf6examples-$(ARCHIVES.mf6examples.sha256[1:12])"))
    dest = joinpath(root, name)
    isdir(dest) && return dest
    extract!(zip, joinpath(root, ".x-" * name), "$name/*")
    mv(joinpath(root, ".x-" * name, name), dest)
    rm(joinpath(root, ".x-" * name); recursive = true)
    return dest
end

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
discrepancy of ModflowGPU's budgets.
"""
function compare_with_mf6(rundir::AbstractString, run)
    r = run.result
    a, b = open_head_file(r.head_file), open_head_file(_only_file(rundir, ".hds"))
    length(a) == length(b) || error("the head files have $(length(a)) and $(length(b)) steps")
    dh = 0.0
    for k in eachindex(b.steps)
        ha, hb = read_head(a, k), read_head(b, k)
        for i in eachindex(hb)
            _inactive(hb[i]) || (dh = max(dh, abs(ha[i] - hb[i])))
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
            j = only(find_budget_records(ba; text = strip(rec.text), kstp = rec.kstp, kper = rec.kper, paknam2 = pak))
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
    pd = maximum(t -> abs(percent_discrepancy(t)), r.budgets)
    return (; dh, budget_cbc = dq, budget_listing = dl, budget = max(dq, dl), pd)
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
Throws if a difference of this build exceeds the acceptance limits of the parity tests
(the head limit is relative to the head range where the parity tests make it so).
"""
function comparison_table(parity_name::AbstractString, stats; dh_limit_double = LIMITS.double.dh)
    row = parity_row(parity_name)
    res = row["results"]
    io = IOBuffer()
    println(io, "| Preset | max \\|Δh\\| (m), this build | max \\|Δh\\| (m), parity table | Budget, this build | Budget, parity table | CPU time, this build (s) | GPU time, parity table (s) |")
    println(io, "| :-- | --: | --: | --: | --: | --: | --: |")
    for p in (:double, :single)
        s = stats[p]
        lim = p === :double ? (; dh = dh_limit_double, budget = LIMITS.double.budget) : LIMITS.single
        s.cmp.dh < lim.dh || error("$parity_name :$p: max |Δh| = $(s.cmp.dh) exceeds $(lim.dh)")
        s.cmp.budget < lim.budget || error("$parity_name :$p: budget difference $(s.cmp.budget) exceeds $(lim.budget)")
        cpu = res["cpu"][string(p)]
        gpu = haskey(res, "rocm") ? @sprintf("%.2f", res["rocm"][string(p)]["wall_time"]) : "–"
        println(
            io, "| `:$p` | ", _e(s.cmp.dh), " | ", _e(cpu["max_dh"]), " | ", _e(s.cmp.budget), " | ",
            _e(max(cpu["max_budget_cbc"], cpu["max_budget_listing"])), " | ", @sprintf("%.2f", s.time), " | ", gpu, " |",
        )
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

"""
    budget_cells(g, budget_file, text) -> Vector{NTuple{3, Int}}

The cells `(layer, row, column)` of the list records with budget text `text` (e.g.
`"CHD"`, `"WEL"`, `"DRN"`, `"RIV"`) of the first saved step, the boundary cells of that
package.
"""
function budget_cells(g::DisGrid, budget_file::AbstractString, text::AbstractString)
    bf = open_budget_file(budget_file)
    recs = read_budget(bf; text)
    isempty(recs) && return NTuple{3, Int}[]
    return [cell_of(g, Int(n)) for n in first(recs).id1]
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
    dirs = vec(
        [
            begin
                    v = Vec2f(qx[c, r], qy[c, r])
                    s = sqrt(sum(abs2, v))
                    normalize ? (s > 0 ? v / s * d : Vec2f(0, 0)) : v
                end for c in 1:g.ncol, r in 1:g.nrow
        ]
    )
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
    names = [t.name == t.package || isempty(t.package) ? t.name : "$(t.name) ($(t.package))" for t in last(first(budgets)).terms]
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
    difference_map(g, panels; layer = 1, title) -> Figure

Maps of head differences, one panel per `label => Δh` pair (`Δh` a `(ncol, nrow, nlay)`
array), each with its own symmetric colour range.
"""
function difference_map(g::DisGrid, panels; layer = 1, size = (420 * length(panels), 380), xlabel = "x position", ylabel = "y position")
    fig = Figure(; size)
    for (k, (lab, d)) in enumerate(panels)
        a = layer_values(d, layer)
        m = maximum(x -> isnan(x) ? 0.0 : abs(x), a)
        m = m == 0 ? 1.0e-12 : m
        ax = Axis(fig[1, 2k - 1]; aspect = DataAspect(), title = lab, xlabel, ylabel)
        hm = map_heatmap!(ax, g, a; colormap = :balance, colorrange = (-m, m))
        Colorbar(fig[1, 2k], hm; label = "Δh", tickformat = expformat)
    end
    return fig
end

"""
    save_figure(fig, name) -> fig

Write `fig` as `<name>.png` into the figure folder of the cache (`<cache>/figures`), so a
script run on its own leaves its figures there, and return `fig` (displayed on the page).
"""
function save_figure(fig, name)
    resize_to_layout!(fig)
    save(joinpath(mkpath(joinpath(cache_dir(), "figures")), name * ".png"), fig; px_per_unit = 2)
    return fig
end

end # module ExampleTools
