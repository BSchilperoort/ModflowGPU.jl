# # Horizontal anisotropy: a well in an anisotropic aquifer
#
# A single confined layer of 51 × 51 cells of 10 m, with a well that pumps 1 m³/d from
# the centre cell and constant heads of 0 m on the cells along the edge. The hydraulic
# conductivity is anisotropic in the horizontal plane: 1 m/d along the principal
# direction and 0.01 m/d across it (NPF `K` and `K22`). The direction is set by the NPF
# angle `ANGLE1`:
#
# - `ex-gwf-hanir`: `ANGLE1` = 0°, the principal direction along the rows (x);
# - `ex-gwf-hanic`: `ANGLE1` = 90°, the principal direction along the columns (y).
#
# The cone of depression is stretched along the principal direction. These are two of
# the three simulations of the MODFLOW 6 example
# [ex-gwf-hani](https://modflow6-examples.readthedocs.io/en/latest/_notebooks/ex-gwf-hani.html);
# the third, `ex-gwf-hanix` (25° with XT3D), needs XT3D, which ModflowGPU does not
# support (see [Supported packages](@ref "Supported packages")).
#
# ## Input and the MODFLOW 6 runs
#
# `ExampleTools` (`docs/ExampleTools.jl` in the repository) takes the input of both
# simulations from the released `mf6examples.zip`, sets the tight closures of the
# [parity tests](@ref "Comparing results") in the IMS files, and runs MODFLOW 6.8.2.

isdefined(Main, :ExampleTools) || Base.include(Main, joinpath(@__DIR__, "..", "docs", "ExampleTools.jl")) #hide
using Main.ExampleTools #hide
using ModflowGPU, CairoMakie
CairoMakie.activate!(; type = "png") #hide

sims = ("ex-gwf-hanir", "ex-gwf-hanic")
mf6 = Dict(n => example_run(n) for n in sims)
[(n, mf6[n].mode, mf6[n].mf6_time) for n in sims]

# ## The ModflowGPU runs
#
# Each simulation runs on the CPU in `:double` and in `:single` precision, with the
# default options (`run_simulation(dir; precision, outdir)`).

runs = Dict((n, p) => run_modflowgpu(mf6[n].dir, p) for n in sims, p in (:double, :single))
[(n, p, r.result.converged) for ((n, p), r) in runs]

# ## The grid
#
# The model grid with the constant-head cells (blue) and the well (red), read from the
# budget file of the ModflowGPU run.

g = read_dis_grid(mf6["ex-gwf-hanir"].dir)
cbc = runs[("ex-gwf-hanir", :double)].result.budget_file
fig = Figure(; size = (420, 420))
ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x position (m)", ylabel = "y position (m)")
map_cells!(ax, g, budget_cells(g, cbc, "CHD"); color = BC_COLORS["CHD"])
map_cells!(ax, g, budget_cells(g, cbc, "WEL"); color = BC_COLORS["WEL"])
map_grid!(ax, g)
limits!(ax, extrema(g.xe)..., extrema(g.ye)...)
save_figure(fig, "ex-gwf-hanir-grid")

# ## Drawdown
#
# The drawdown (0 m minus the head) of each simulation, with contours every 0.1 m from
# 0.1 to 0.9 m, as in the MF6 example.

function drawdown_figure(name)
    r = runs[(name, :double)].result
    h = read_head(open_head_file(r.head_file), 1)
    s = 0 .- layer_values(h, 1)
    fig = Figure(; size = (480, 420))
    ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x position (m)", ylabel = "y position (m)", title = name)
    hm = map_heatmap!(ax, g, s; colormap = (:jet, 0.25))
    map_contour!(ax, g, s; levels = 0.1:0.1:0.9, colormap = :viridis)
    Colorbar(fig[1, 2], hm; label = "Drawdown (m)", height = Relative(0.25))
    return save_figure(fig, "$name-head")
end
drawdown_figure("ex-gwf-hanir")

#-

drawdown_figure("ex-gwf-hanic")

# ## Comparison with MODFLOW 6
#
# The head differences between ModflowGPU and MODFLOW 6 in both precisions.

function difference_figure(name)
    h6 = read_head(open_head_file(only(filter(endswith(".hds"), readdir(mf6[name].dir; join = true)))), 1)
    d = [":$p − MODFLOW 6" => read_head(open_head_file(runs[(name, p)].result.head_file), 1) .- h6 for p in (:double, :single)]
    return save_figure(difference_map(g, d; xlabel = "x position (m)", ylabel = "y position (m)"), "$name-difference")
end
difference_figure("ex-gwf-hanir")

#-

difference_figure("ex-gwf-hanic")

# The largest head and budget differences, next to the values of the [parity
# table](@ref "Parity table"), and the wall times (CPU: this build, including compilation
# for the first run of the session; GPU: the parity table, AMD Radeon RX 7800 XT). For
# `ex-gwf-hanir`:

stats(name) = Dict(p => (; cmp = compare_with_mf6(mf6[name].dir, runs[(name, p)]), runs[(name, p)].time) for p in (:double, :single))
comparison_table("ex-gwf-hanir", stats("ex-gwf-hanir"))

# For `ex-gwf-hanic`:

comparison_table("ex-gwf-hanic", stats("ex-gwf-hanic"))
