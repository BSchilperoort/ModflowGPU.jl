# # An unstructured mesh (DISV, XT3D)
#
# An aquifer of about 3 km × 3 km in two layers of 20 m, on an unstructured DISV mesh of
# 5240 quadrilateral cells per layer, with K = 10 m/d and K33 = 0.2 m/d. Every cell of the top
# layer either has a general-head boundary with a stage of 0 m (3360 cells) or receives
# recharge of 0.004 m/d (1880 cells), and the model is steady state. On such a mesh the
# line between two cell centres is in general not perpendicular to their shared face,
# which the standard conductance formulation does not account for; the XT3D formulation of
# NPF (option `XT3D`) does. This is the MODFLOW 6 example
# [ex-gwf-disvmesh](https://modflow6-examples.readthedocs.io/en/latest/_notebooks/ex-gwf-disvmesh.html).
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

mf6 = example_run("ex-gwf-disvmesh")
mf6.mode, mf6.mf6_time

# ## The ModflowGPU run
#
# In `:double` and in `:single` precision on the CPU, with the default options.

runs = Dict(p => run_modflowgpu(mf6.dir, p) for p in (:double, :single))
r = runs[:double].result
r.converged

# ## The mesh and the boundaries
#
# The cells of the mesh, with the general-head boundary cells (cyan) and the recharge
# cells (green) of the top layer, read from ModflowGPU's budget file.

g = read_vertex_grid(mf6.dir)
fig = Figure(; size = (560, 560))
ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x position (m)", ylabel = "y position (m)")
map_cells!(ax, g, budget_cells(g, r.budget_file, "GHB"); color = (BC_COLORS["GHB"], 0.6))
map_cells!(ax, g, budget_cells(g, r.budget_file, "RCH"); color = (BC_COLORS["RCH"], 0.6))
map_grid!(ax, g; linewidth = 0.2)
save_figure(fig, "ex-gwf-disvmesh-grid")

# ## Heads
#
# The heads of both layers.

h = read_head(open_head_file(r.head_file), 1)
fig = Figure(; size = (900, 430))
for l in 1:2
    ax = Axis(fig[1, 2l - 1]; aspect = DataAspect(), xlabel = "x position (m)", ylabel = "y position (m)", title = "Layer $l")
    hm = map_heatmap!(ax, g, layer_values(g, h, l); colormap = :jet)
    Colorbar(fig[1, 2l], hm; label = "Head (m)", height = Relative(0.5))
end
save_figure(fig, "ex-gwf-disvmesh-head")

# ## Comparison with MODFLOW 6
#
# The head differences between ModflowGPU and MODFLOW 6 in the top layer, in both
# precisions.

h6 = read_head(open_head_file(only(filter(endswith(".hds"), readdir(mf6.dir; join = true)))), 1)
d = [":$p − MODFLOW 6" => read_head(open_head_file(runs[p].result.head_file), 1) .- h6 for p in (:double, :single)]
save_figure(difference_map(g, d; unit = "m", xlabel = "x position (m)", ylabel = "y position (m)"), "ex-gwf-disvmesh-difference")

# The largest head and budget differences, next to the values of the [parity
# table](@ref "Parity table"), and the wall times (CPU: this build, including compilation
# for the first run of the session; GPU: the parity table, AMD Radeon RX 7800 XT).

stats = Dict(p => (; cmp = compare_with_mf6(mf6.dir, runs[p]), runs[p].time) for p in (:double, :single))
comparison_table("ex-gwf-disvmesh", stats)
