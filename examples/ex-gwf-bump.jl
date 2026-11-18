# # Flow diversion around a high bottom
#
# Unconfined flow in a square aquifer of 100 m × 100 m (one layer of 51 × 51 cells)
# whose bottom rises to a bump in the middle, higher than the water table, so that the
# cells on the bump go dry and the flow goes around it. Constant heads of 7.5 m on the
# west edge and 2.5 m on the east edge drive the flow. Units are metres and days.
#
# The MODFLOW 6 example
# [ex-gwf-bump](https://modflow6-examples.readthedocs.io/en/latest/_notebooks/ex-gwf-bump.html)
# solves the problem three ways: with the Newton–Raphson formulation (`-p01a`), with the
# standard formulation and rewetting of dry cells (`-p01b`), and with Newton–Raphson on
# a cylindrical bump (`-p01c`). This page runs `ex-gwf-bump-p01b`; the two Newton
# variants are planned (see [Supported packages](@ref)).
#
# ## Input and the MODFLOW 6 run
#
# `ExampleTools` (`docs/ExampleTools.jl` in the repository) takes the input from the
# released `mf6examples.zip`, sets the tight closures of the [parity tests](@ref
# "Comparing results") in the IMS file, and runs MODFLOW 6.8.2.

isdefined(Main, :ExampleTools) || Base.include(Main, joinpath(@__DIR__, "..", "docs", "ExampleTools.jl")) #hide
using Main.ExampleTools #hide
using ModflowGPU, CairoMakie
CairoMakie.activate!(; type = "png") #hide

mf6 = example_run("ex-gwf-bump-p01b")
mf6.mode, mf6.mf6_time

# ## The ModflowGPU run
#
# In `:double` and in `:single` precision on the CPU, with the default options.

runs = Dict(p => run_modflowgpu(mf6.dir, p) for p in (:double, :single))
r = runs[:double].result
r.converged

# ## The model
#
# The bottom elevation with contours every 2 m and the constant-head cells (cyan), as in
# the MF6 example.

g = read_dis_grid(mf6.dir)
bot = g.botm[:, :, 1]
chd = budget_cells(g, r.budget_file, "CHD")
fig = Figure(; size = (520, 560))
ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x-coordinate, in meters", ylabel = "y-coordinate, in meters")
hm = map_heatmap!(ax, g, bot; colorrange = (0, 10))
map_cells!(ax, g, chd; color = :cyan)
map_contour!(ax, g, bot; levels = 2:2:8, color = :black, linestyle = :dot, linewidth = 1, labels = true, labelsize = 9)
limits!(ax, 0, 100, 0, 100)
Colorbar(fig[2, 1], hm; vertical = false, label = "Bottom elevation (m)", width = Relative(0.5))
save_figure(fig, "ex-gwf-bump-grid")

# ## Heads
#
# The heads with contours every metre (from 3 m to 7 m), the bottom contours, the
# normalized specific discharge (computed from ModflowGPU's FLOW-JA-FACE record as
# MODFLOW 6 computes it for `SAVE_SPECIFIC_DISCHARGE`) and the constant-head cells. Dry
# cells (white) are those whose head is at or below the bottom, as in the MF6 figure.

h = read_head(open_head_file(r.head_file), 1)
flowja = vec(only(read_budget(open_budget_file(r.budget_file); text = "FLOW-JA-FACE")))
qx, qy = specific_discharge(g, flowja, h)
hl = layer_values(h, 1)
hl[hl .<= bot] .= NaN
fig = Figure(; size = (520, 560))
ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x-coordinate, in meters", ylabel = "y-coordinate, in meters")
map_contour!(ax, g, bot; levels = 2:2:8, color = :black, linestyle = :dot, linewidth = 1, labels = true, labelsize = 9)
hm = map_heatmap!(ax, g, hl; colorrange = (2.5, 7.5))
map_contour!(ax, g, hl; levels = 3:1:7, color = :black, linewidth = 0.5, labels = true, labelsize = 9)
map_cells!(ax, g, chd; color = :cyan)
map_vectors!(ax, g, qx[:, :, 1], qy[:, :, 1])
limits!(ax, 0, 100, 0, 100)
Colorbar(fig[2, 1], hm; vertical = false, label = "Head (m)", width = Relative(0.5))
save_figure(fig, "ex-gwf-bump-head")

# ## Comparison with MODFLOW 6
#
# The head differences between ModflowGPU and MODFLOW 6, in both precisions (dry cells
# left out).

h6 = read_head(open_head_file(only(filter(endswith(".hds"), readdir(mf6.dir; join = true)))), 1)
d = [":$p − MODFLOW 6" => mask_inactive(read_head(open_head_file(runs[p].result.head_file), 1)) .- mask_inactive(h6) for p in (:double, :single)]
save_figure(difference_map(g, d; unit = "m", xlabel = "x (m)", ylabel = "y (m)"), "ex-gwf-bump-difference")

# The largest head and budget differences, next to the values of the [parity
# table](@ref "Parity table"), and the wall times (CPU: this build, including compilation
# for the first run of the session; GPU: the parity table, AMD Radeon RX 7800 XT).

stats = Dict(p => (; cmp = compare_with_mf6(mf6.dir, runs[p]), runs[p].time) for p in (:double, :single))
comparison_table("ex-gwf-bump-p01b", stats)
