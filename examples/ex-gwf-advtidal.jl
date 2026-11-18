# # Highlands next to a tidal estuary
#
# A transient model of highlands bordered by a tidal estuary: three layers (an upper
# aquifer, a confining bed and a lower aquifer) of 15 × 10 cells of 500 m. The estuary is
# a general-head boundary (GHB) in the last column whose stage follows the tide (a time
# series); two rivers cross the area, wells pump from the upper and lower aquifers, and
# recharge falls in three zones, each with its own time series, with
# evapotranspiration from the upper aquifer. A steady-state period is followed by three
# transient periods of 361 time steps in all. Units are metres and days.
#
# This is the MODFLOW 6 example
# [ex-gwf-advtidal](https://modflow6-examples.readthedocs.io/en/latest/_notebooks/ex-gwf-advtidal.html).
# Its time series come from MODFLOW 6's observation (OBS) output; ModflowGPU does not
# write observation files yet, so the same quantities are read here from the head and
# budget files, which hold every time step.
#
# ## Input and the MODFLOW 6 run
#
# `ExampleTools` (`docs/ExampleTools.jl` in the repository) takes the input from the
# released `mf6examples.zip`, sets the tight closures of the [parity tests](@ref
# "Comparing results") in the IMS file, and runs MODFLOW 6.8.2. With the absolute residual
# criterion of the tight closures MODFLOW 6 does not converge on this model, so its run
# uses the relative criterion (`tight-relative`), as the reference run of the parity
# tests does.

isdefined(Main, :ExampleTools) || Base.include(Main, joinpath(@__DIR__, "..", "docs", "ExampleTools.jl")) #hide
using Main.ExampleTools #hide
using ModflowGPU, CairoMakie
CairoMakie.activate!(; type = "png") #hide

mf6 = example_run("ex-gwf-advtidal"; mode = "tight-relative")
mf6.mode, mf6.mf6_time

# ## The ModflowGPU run
#
# In `:double` and in `:single` precision on the CPU, with the default options.

runs = Dict(p => run_modflowgpu(mf6.dir, p) for p in (:double, :single))
r = runs[:double].result
r.converged

# ## The model
#
# (A) Layer 1 with the wells (red) and the river cells (teal), (B) layer 2 and (C)
# layer 3 with the estuary (GHB, cyan) and, in layer 3, the wells, and (D) the three
# recharge zones, as in the MF6 example (which draws the zones as polygons; cells on the
# edge of two zones get recharge from both, with a multiplier).

g = read_dis_grid(mf6.dir)
bf = open_budget_file(r.budget_file)
wells = budget_cells(g, r.budget_file, "WEL")
river = budget_cells(g, r.budget_file, "RIV")
ghb = budget_cells(g, r.budget_file, "GHB")
rch = ModflowGPU.find_budget_records(bf; text = "RCH", kstp = 1, kper = 1)
zones = [strip(bf.records[i].paknam2) => [cell_of(g, n) for n in read_budget(bf, i).id1] for i in rch]
fig = Figure(; size = (620, 900))
panels = [
    (1, 1, "A. Layer 1", [(wells, :red, 1), (river, BC_COLORS["RIV"], 1)]),
    (1, 2, "B. Layer 2", [(ghb, :cyan, 2)]),
    (2, 1, "C. Layer 3", [(ghb, :cyan, 3), (wells, :red, 3)]),
]
for (i, j, title, sets) in panels
    local ax = Axis(fig[i, j]; aspect = DataAspect(), title, xlabel = "x position (m)", ylabel = "y position (m)")
    for (cells, color, l) in sets
        map_cells!(ax, g, filter(c -> c[1] == l, cells); color)
    end
    map_grid!(ax, g)
    limits!(ax, extrema(g.xe)..., extrema(g.ye)...)
end
ax = Axis(fig[2, 2]; aspect = DataAspect(), title = "D. Recharge zones", xlabel = "x position (m)", ylabel = "y position (m)")
for ((name, cells), color) in zip(zones, (:red, :blue, :green))
    map_cells!(ax, g, cells; color = (color, 0.25), label = "Recharge zone $(last(name))")
end
limits!(ax, extrema(g.xe)..., extrema(g.ye)...)
axislegend(ax; position = :rb, labelsize = 9)
save_figure(fig, "ex-gwf-advtidal-grid")

# ## Time series
#
# The head in layer 3, row 13, column 8; the flow between the cells in rows 5 and 6 of
# column 6 in layer 1; and the flows of the estuary: of the GHB cells in row 6 of layers
# 2 and 3, and of all GHB cells of layers 2 and 3 (the boundary names `ESTUARY-L2` and
# `ESTUARY-L3`). These are the observations of the MF6 example, here from ModflowGPU's
# head and budget files (lines) and from MODFLOW 6's observation files (dots, every
# fifth time step).

hf = open_head_file(r.head_file)
t = ModflowGPU.head_times(hf)
head = [read_head(hf, k)[8, 13, 3] for k in 1:length(hf)]
n1, n2 = node(g, 1, 5, 6), node(g, 1, 6, 6)
pos = findfirst(i -> g.ja[i] == n2, g.ia[n1]:(g.ia[n1 + 1] - 1)) + g.ia[n1] - 1
icf1 = [vec(f)[pos] for f in read_budget(bf; text = "FLOW-JA-FACE")]
ghbflows = read_budget(bf; text = "GHB")
layer(n) = cell_of(g, Int(n))[1]
at(rec, l, rw, c) = rec.q[findfirst(==(node(g, l, rw, c)), rec.id1)]
estuary = Dict(
    "ghb_2_6_10" => [at(x, 2, 6, 10) for x in ghbflows], "ghb_3_6_10" => [at(x, 3, 6, 10) for x in ghbflows],
    "estuary2" => [sum(x.q[layer.(x.id1) .== 2]) for x in ghbflows], "estuary3" => [sum(x.q[layer.(x.id1) .== 3]) for x in ghbflows],
)

# The observation files of the MODFLOW 6 run (CSV: time, then one column per observation):
function read_csv(file)
    lines = readlines(joinpath(mf6.dir, file))
    names = lowercase.(split(first(lines), ','))
    data = [parse.(Float64, split(l, ',')) for l in lines[2:end]]
    return Dict(n => [d[k] for d in data] for (k, n) in enumerate(names))
end
obs_head, obs_flow, obs_ghb = read_csv.(("ex-gwf-advtidal.obs.head.csv", "ex-gwf-advtidal.obs.flow.csv", "ex-gwf-advtidal.ghb.obs.csv"))

fig = Figure(; size = (760, 820))
series = [
    ("head (m)", ["h1_13_8" => head], obs_head),
    ("flow (m³/d)", ["icf1" => icf1], obs_flow),
    ("flow (m³/d)", [k => estuary[k] for k in ("ghb_2_6_10", "ghb_3_6_10", "estuary2", "estuary3")], obs_ghb),
]
for (i, (ylabel, curves, obs)) in enumerate(series)
    local ax = Axis(fig[i, 1]; xlabel = "time (d)", ylabel)
    for (k, (name, v)) in enumerate(curves)
        lines!(ax, t, v; color = Makie.wong_colors()[k], label = name)
        scatter!(ax, obs["time"][1:5:end], obs[name][1:5:end]; color = Makie.wong_colors()[k], markersize = 4)
    end
    Legend(fig[i, 2], ax; labelsize = 9, framevisible = false)
end
save_figure(fig, "ex-gwf-advtidal-obs")

# ## Comparison with MODFLOW 6
#
# The head differences between ModflowGPU and MODFLOW 6 in layer 1 at the end of the
# simulation, in both precisions.

hf6 = open_head_file(only(filter(endswith(".hds"), readdir(mf6.dir; join = true))))
h6 = read_head(hf6, length(hf6))
d = [":$p − MODFLOW 6" => (hp = open_head_file(runs[p].result.head_file); read_head(hp, length(hp))) .- h6 for p in (:double, :single)]
save_figure(difference_map(g, d; unit = "m", xlabel = "x (m)", ylabel = "y (m)", size = (700, 520)), "ex-gwf-advtidal-difference")

# The largest head and budget differences over all time steps, next to the values of the
# [parity table](@ref "Parity table"), and the wall times (CPU: this build, including
# compilation for the first run of the session; GPU: the parity table, AMD Radeon RX
# 7800 XT). In `:single` the parity tests compare the budget of this model beyond the
# Float32 rounding of the cell flows, which the parity table shows; the table here shows
# the raw difference.

stats = Dict(p => (; cmp = compare_with_mf6(mf6.dir, runs[p]), runs[p].time) for p in (:double, :single))
comparison_table("ex-gwf-advtidal", stats)
