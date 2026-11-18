# # Zaidel: a water table over steps
#
# Zaidel (2013) gave an analytical solution for steady unconfined flow over a bottom that
# drops in steps. The model is one row of 200 cells of 5 m with a bottom that falls 5 m
# every 40 cells (from 20 m to 0 m), constant heads of 23 m at the west end and 10 m at
# the east end, and a hydraulic conductivity of 1e-4 m/d. At each step the water table
# drops steeply and, just upstream of it, lies close to the bottom, where the saturated
# thickness of the standard formulation would let cells dry out and rewet from one
# iteration to the next; the Newton–Raphson formulation solves it. Units are metres and
# days.
#
# The MODFLOW 6 example
# [ex-gwf-zaidel](https://modflow6-examples.readthedocs.io/en/latest/_notebooks/ex-gwf-zaidel.html)
# compares MODFLOW 6 with the analytical solution; this page compares ModflowGPU with
# MODFLOW 6.
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

mf6 = example_run("ex-gwf-zaidel")
mf6.mode, mf6.mf6_time

# ## The ModflowGPU run
#
# In `:double` and in `:single` precision on the CPU, with the default options.

runs = Dict(p => run_modflowgpu(mf6.dir, p) for p in (:double, :single))
r = runs[:double].result
r.converged

# ## Heads
#
# The heads along the row (ModflowGPU: line, MODFLOW 6: circles every fourth cell) over
# the bottom steps (grey), as in the MF6 example. Where the head is below the bottom the
# cell is dry.

g = read_dis_grid(mf6.dir)
h = vec(read_head(open_head_file(r.head_file), 1))
h6 = vec(read_head(open_head_file(only(filter(endswith(".hds"), readdir(mf6.dir; join = true)))), 1))
bot = vec(g.botm[:, 1, 1])
fig = Figure(; size = (720, 380))
ax = Axis(fig[1, 1]; xlabel = "x-coordinate, in meters", ylabel = "Elevation, in meters")
poly!(ax, [Rect2f(g.xe[c], -1, g.delr[c], bot[c] + 1) for c in 1:g.ncol]; color = RGBf(0.8, 0.8, 0.8))
wet = h .> bot
lines!(ax, g.xc[wet], h[wet]; color = :blue, linewidth = 1.5, label = "ModflowGPU")
scatter!(ax, g.xc[1:4:end], h6[1:4:end]; color = :transparent, strokecolor = :black, strokewidth = 1, markersize = 6, label = "MODFLOW 6")
limits!(ax, 0, 1000, -1, 25)
axislegend(ax; position = :rt, framevisible = false)
save_figure(fig, "ex-gwf-zaidel-head")

# ## Comparison with MODFLOW 6
#
# The head differences between ModflowGPU and MODFLOW 6 along the row, in both precisions
# (dry cells, whose head is not unique, left out; see
# [Differences from MODFLOW 6](@ref "Newton–Raphson")).

fig = Figure(; size = (720, 300))
ax = Axis(fig[1, 1]; xlabel = "x-coordinate, in meters", ylabel = "Head difference, in meters")
for (p, color) in ((:double, :black), (:single, :red))
    local hp = vec(read_head(open_head_file(runs[p].result.head_file), 1))
    local d = [h6[c] > bot[c] ? hp[c] - h6[c] : NaN for c in eachindex(h6)]
    lines!(ax, g.xc, d; color, label = ":$p − MODFLOW 6")
end
axislegend(ax; position = :lb, framevisible = false)
save_figure(fig, "ex-gwf-zaidel-difference")

# The largest head and budget differences, next to the values of the [parity
# table](@ref "Parity table"), and the wall times (CPU: this build, including compilation
# for the first run of the session; GPU: the parity table, AMD Radeon RX 7800 XT). The model
# writes no budget file, so the budget column compares the listing tables only.

stats = Dict(p => (; cmp = compare_with_mf6(mf6.dir, runs[p]), runs[p].time) for p in (:double, :single))
comparison_table("ex-gwf-zaidel", stats)
