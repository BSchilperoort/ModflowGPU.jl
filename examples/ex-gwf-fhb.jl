# # Flow and head boundaries from time series
#
# A small transient model that reproduces the test problem of the Flow and Head Boundary
# (FHB) package of MODFLOW-2005 with MODFLOW 6 time series: one confined layer of 3 rows
# and 10 columns of 1000 m, three stress periods of 400, 200 and 400 days (10, 4 and 6
# time steps), a specified head in the three cells of the last column and an injection
# well in the first cell of the middle row. Both the head (CHD) and the injection rate
# (WEL) come from time series files (`.ts`) with `LINEAREND` interpolation, so they
# change at every time step.
#
# This is the MODFLOW 6 example
# [ex-gwf-fhb](https://modflow6-examples.readthedocs.io/en/latest/_notebooks/ex-gwf-fhb.html).
# The example records the heads in the first and last cells of the middle row and the
# flow between the first two cells with the OBS package; ModflowGPU writes the same
# observation files (CSV), which this page reads.
#
# ## Input and the MODFLOW 6 run
#
# `ExampleTools` (`docs/ExampleTools.jl` in the repository) takes the input from the
# released `mf6examples.zip`, sets the tight closures of the [parity tests](@ref
# "Comparing results") in the IMS file, and runs MODFLOW 6.8.2. With an absolute
# `INNER_RCLOSE` of 1e-10 mf6 does not converge on this model, so the closure is relative
# (`RELATIVE_RCLOSE`), as for the reference run of the parity tests.

isdefined(Main, :ExampleTools) || Base.include(Main, joinpath(@__DIR__, "..", "docs", "ExampleTools.jl")) #hide
using Main.ExampleTools #hide
using ModflowGPU, CairoMakie
CairoMakie.activate!(; type = "png") #hide

mf6 = example_run("ex-gwf-fhb")
mf6.mode, mf6.mf6_time

# ## The ModflowGPU run
#
# In `:double` and in `:single` precision on the CPU, with the default options. The time
# series are evaluated on the host in Float64 at every time step, as MODFLOW 6 does.

runs = Dict(p => run_modflowgpu(mf6.dir, p) for p in (:double, :single))
r = runs[:double].result
r.converged, length(r.steps)

# ## The grid
#
# The constant-head cells (blue) and the well (red).

g = read_dis_grid(mf6.dir)
fig = Figure(; size = (480, 300), figure_padding = (8, 24, 8, 8))   # room for the last x tick label
ax = Axis(fig[1, 1]; aspect = DataAspect(), xlabel = "x position (m)", ylabel = "y position (m)", xtickformat = intformat, ytickformat = intformat)
map_cells!(ax, g, budget_cells(g, r.budget_file, "CHD"); color = BC_COLORS["CHD"])
map_cells!(ax, g, budget_cells(g, r.budget_file, "WEL"); color = BC_COLORS["WEL"])
map_grid!(ax, g)
limits!(ax, extrema(g.xe)..., extrema(g.ye)...)
save_figure(fig, "ex-gwf-fhb-grid")

# ## Observations over time
#
# The observations of the MF6 example: `h1_2_1` and `h1_2_10`, the heads in layer 1,
# row 2, columns 1 and 10, and `icf1`, the flow into cell (1, 2, 2) from cell (1, 2, 1),
# at the end of every time step. ModflowGPU writes them as CSV files
# (`result.obs_files`; [`read_observations`](@ref) reads them): lines are MODFLOW 6's
# files, circles ModflowGPU's.

head_file, flow_file = "ex-gwf-fhb.obs.head.csv", "ex-gwf-fhb.obs.flow.csv"
ph, pq = obs_pair(mf6.dir, runs[:double], head_file), obs_pair(mf6.dir, runs[:double], flow_file)
fig = Figure(; size = (600, 360))
ax = Axis(fig[1, 1]; xlabel = "time (d)", ylabel = "head (m)")
obs_plot!(ax, ph, ["h1_2_1", "h1_2_10"])
axislegend(ax; position = :lt, framevisible = false, labelsize = 9)
save_figure(fig, "ex-gwf-fhb-obs-head")

#-

fig = Figure(; size = (600, 360))
ax = Axis(fig[1, 1]; xlabel = "time (d)", ylabel = "flow (m³/d)")
obs_plot!(ax, pq, ["icf1"])
axislegend(ax; position = :lt, framevisible = false, labelsize = 9)
save_figure(fig, "ex-gwf-fhb-obs-flow")

# ## Comparison with MODFLOW 6
#
# The differences of the three observation series between ModflowGPU and MODFLOW 6, at
# every time step, in both precisions.

fig = Figure(; size = (900, 560))
for (k, p) in enumerate((:double, :single))
    gh, gq = obs_pair(mf6.dir, runs[p], head_file), obs_pair(mf6.dir, runs[p], flow_file)
    local ax = Axis(fig[1, k]; title = ":$p − MODFLOW 6", ylabel = "head difference (m)", ytickformat = expformat)
    for name in ("h1_2_1", "h1_2_10")
        scatterlines!(ax, gh.mf6.times, obs_masked(gh.gpu, name) .- obs_masked(gh.mf6, name); label = name)
    end
    k == 2 && axislegend(ax; position = :lb, framevisible = false)
    ax = Axis(fig[2, k]; xlabel = "time (d)", ylabel = "icf1 difference (m³/d)", ytickformat = expformat)
    scatterlines!(ax, gq.mf6.times, obs_masked(gq.gpu, "icf1") .- obs_masked(gq.mf6, "icf1"); color = Makie.wong_colors()[3])
end
save_figure(fig, "ex-gwf-fhb-difference")

# The largest difference of each observation over all time steps:

obs_difference_table(mf6.dir, runs, (head_file, flow_file); flows = (flow_file,))

# The largest head and budget differences over all cells and time steps, next to the
# values of the [parity table](@ref "Parity table"), and the wall times (CPU: this build,
# including compilation for the first run of the session; GPU: the parity table, AMD
# Radeon RX 7800 XT).

stats = Dict(p => (; cmp = compare_with_mf6(mf6.dir, runs[p]), runs[p].time) for p in (:double, :single))
comparison_table("ex-gwf-fhb", stats)
