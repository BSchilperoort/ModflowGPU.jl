# # A valley aquifer with a river: the coarse model
#
# A three-dimensional model of a buried valley aquifer drained by a river, from the
# paper on local grid refinement in MODFLOW by Vilhelmsen and others (2012). The valley
# is filled with sand and gravel of high hydraulic conductivity, set in clay and till of
# low conductivity (three K zones); recharge falls on the whole top, and a river (RIV)
# follows the valley. Units are metres and seconds.
#
# The MODFLOW 6 example
# [ex-gwf-lgrv](https://modflow6-examples.readthedocs.io/en/latest/_notebooks/ex-gwf-lgrv.html)
# runs three versions: a refined model (`-gr`, 25 layers of 183 × 147 cells), a coarse one
# (`-gc`) and the coarse model with a refined child model inside (`-lgr`). This page runs
# the coarse model, `ex-gwf-lgrv-gc`: 9 layers of 61 × 49 cells of 105 m × 75 m. The
# `-lgr` version needs two models coupled by an exchange, which ModflowGPU does not
# support (see [Supported packages](@ref)).
#
# ## Input and the MODFLOW 6 run
#
# `ExampleTools` (`docs/ExampleTools.jl` in the repository) takes the input from the
# released `mf6examples.zip`, sets the tight closures of the [parity tests](@ref
# "Comparing results") in the IMS file, and runs MODFLOW 6.8.2.

isdefined(Main, :ExampleTools) || include(joinpath(@__DIR__, "..", "docs", "ExampleTools.jl")) #hide
using Main.ExampleTools #hide
using ModflowGPU, CairoMakie
CairoMakie.activate!(; type = "png") #hide

mf6 = example_run("ex-gwf-lgrv-gc")
mf6.mode, mf6.mf6_time

# ## The ModflowGPU run
#
# In `:double` and in `:single` precision on the CPU, with the default options.

runs = Dict(p => run_modflowgpu(mf6.dir, p) for p in (:double, :single))
r = runs[:double].result
r.converged

# ## The model
#
# The land surface (the top of layer 1) with the river cells, and a cross-section
# through the column at x = 3000 m that shows the natural logarithm of the hydraulic
# conductivity (K in m/s), as in the MF6 example.

g = read_dis_grid(mf6.dir)
top = [g.idomain[c, r, 1] > 0 ? g.top[c, r] : NaN for c in 1:g.ncol, r in 1:g.nrow]
fig = Figure(; size = (600, 480))
ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x position (m)", ylabel = "y position (m)")
hm = map_heatmap!(ax, g, top; colormap = (:jet, 0.25))
map_cells!(ax, g, budget_cells(g, r.budget_file, "RIV"); color = BC_COLORS["RIV"])
limits!(ax, extrema(g.xe)..., extrema(g.ye)...)
Colorbar(fig[1, 2], hm; label = "Top (m)", height = Relative(0.5))
save_figure(fig, "ex-gwf-lgrv-gc-grid")

#-

k = reshape(only(load_simulation(mf6.dir).models).npf.k, g.ncol, g.nrow, g.nlay)
col = searchsortedlast(g.xe, 3000.0)
fig = Figure(; size = (600, 300))
ax = Axis(fig[1, 1]; xlabel = "y position (m)", ylabel = "z position (m)")
pl = section_heatmap!(ax, g, log.(k); column = col, colormap = (:jet, 0.25))
section_grid!(ax, g; column = col, linewidth = 0.5)
Colorbar(fig[1, 2], pl; label = "ln K (K in m/s)")
save_figure(fig, "ex-gwf-lgrv-gc-xsect")

# ## Heads
#
# The heads in layer 1.

h = read_head(open_head_file(r.head_file), 1)
fig = Figure(; size = (600, 480))
ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x position (m)", ylabel = "y position (m)")
hm = map_heatmap!(ax, g, layer_values(h, 1); colormap = :jet)
limits!(ax, extrema(g.xe)..., extrema(g.ye)...)
Colorbar(fig[1, 2], hm; label = "Head (m)", height = Relative(0.5))
save_figure(fig, "ex-gwf-lgrv-gc-head")

# ## Comparison with MODFLOW 6
#
# The head differences between ModflowGPU and MODFLOW 6 in layer 1, in both precisions.

h6 = read_head(open_head_file(only(filter(endswith(".hds"), readdir(mf6.dir; join = true)))), 1)
d = [":$p − MODFLOW 6" => read_head(open_head_file(runs[p].result.head_file), 1) .- h6 for p in (:double, :single)]
save_figure(difference_map(g, d; xlabel = "x position (m)", ylabel = "y position (m)"), "ex-gwf-lgrv-gc-difference")

# The largest head and budget differences over all layers, next to the values of the
# [parity table](@ref "Parity table"), and the wall times (CPU: this build, including
# compilation for the first run of the session; GPU: the parity table, AMD Radeon RX
# 7800 XT).

stats = Dict(p => (; cmp = compare_with_mf6(mf6.dir, runs[p]), runs[p].time) for p in (:double, :single))
comparison_table("ex-gwf-lgrv-gc", stats)
