# # A rising water table under a recharge pond
#
# Problem 2 of the MODFLOW-NWT documentation (Niswonger and others, 2011), first described
# by McDonald and others (1991) for the rewetting option of MODFLOW: a square aquifer of
# 5000 ft × 5000 ft in 14 layers of 40 × 40 cells, initially dry in its upper layers, with
# constant heads along its edges. Recharge from a pond in one corner raises the water
# table through the dry cells, which become wet again (rewetting with `WETDRY`). Four
# transient stress periods. Units are feet and days.
#
# The MODFLOW 6 example
# [ex-gwf-nwt-p02](https://modflow6-examples.readthedocs.io/en/latest/_notebooks/ex-gwf-nwt-p02.html)
# solves it with the Newton–Raphson formulation (`-p02a`) and with rewetting (`-p02b`), and
# its figure compares the two. ModflowGPU does not support Newton–Raphson yet (planned, see
# [Supported packages](@ref)), so this page runs `ex-gwf-nwt-p02b` and compares
# ModflowGPU's water table with that of MODFLOW 6 on the same input.
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

mf6 = example_run("ex-gwf-nwt-p02b")
mf6.mode, mf6.mf6_time

# ## The ModflowGPU run
#
# In `:double` and in `:single` precision on the CPU, with the default options.

runs = Dict(p => run_modflowgpu(mf6.dir, p) for p in (:double, :single))
r = runs[:double].result
r.converged

# ## The water table
#
# The water table (the head of the highest wet cell) along row 2 at the end of each of
# the four stress periods: ModflowGPU (line) and MODFLOW 6 (circles), with the cells of
# the cross-section and the constant-head cells (cyan), and the mean absolute difference
# of the water table over the whole model, as in the MF6 example (where the line is the
# Newton–Raphson solution and the circles the rewetting solution).

g = read_dis_grid(mf6.dir)
chd = [cell_of(g, n) for n in input_nodes(mf6.dir, "CHD")]
hf = open_head_file(r.head_file)
hf6 = open_head_file(only(filter(endswith(".hds"), readdir(mf6.dir; join = true))))

function water_table(h)
    wt = fill(NaN, g.ncol, g.nrow)
    for c in 1:g.ncol, rw in 1:g.nrow, l in g.nlay:-1:1
        v = h[c, rw, l]
        abs(v) < 1.0e30 && v > g.botm[c, rw, l] && (wt[c, rw] = v)
    end
    return wt
end

row = 2
fig = Figure(; size = (640, 700))
for k in 1:4
    local wt, wt6 = water_table(read_head(hf, k)), water_table(read_head(hf6, k))
    local mae = sum(x -> isnan(x) ? 0.0 : abs(x), wt .- wt6) / (g.ncol * g.nrow)
    local ax = Axis(
        fig[k, 1]; title = "$(('A':'D')[k]). Mean absolute water-table difference $(round(mae; sigdigits = 2)) feet",
        titlesize = 12, titlealign = :right, ylabel = k == 2 ? "Water-table elevation, in feet" : "",
    )
    poly!(ax, [Rect2f(g.xe[c], g.botm[c, row, l], g.delr[c], (l == 1 ? g.top[c, row] : g.botm[c, row, l - 1]) - g.botm[c, row, l]) for (l, rw, c) in chd if rw == row]; color = :cyan)
    section_grid!(ax, g; row, color = RGBf(0.5, 0.5, 0.5))
    lines!(ax, g.xc, wt[:, row]; color = :black, linewidth = 0.75, label = "ModflowGPU")
    scatter!(ax, g.xc, wt6[:, row]; color = :transparent, strokecolor = :blue, strokewidth = 1, markersize = 7, label = "MODFLOW 6")
    ax.yticks = 20:10:60
    limits!(ax, 0, 5000, 20, 65)
    k == 1 && axislegend(ax; position = :rt, framevisible = false, labelsize = 10)
    k < 4 && hidexdecorations!(ax; grid = false)
    k == 4 && (ax.xlabel = "x-coordinate, in feet")
end
save_figure(fig, "ex-gwf-nwt-p02")

# ## Comparison with MODFLOW 6
#
# The head differences between ModflowGPU and MODFLOW 6 in layer 14 (the bottom layer,
# always wet) at the end of the simulation, in both precisions.

h6 = read_head(hf6, 4)
d = [":$p − MODFLOW 6" => read_head(open_head_file(runs[p].result.head_file), 4) .- h6 for p in (:double, :single)]
save_figure(difference_map(g, d; layer = 14, unit = "ft", xlabel = "x (ft)", ylabel = "y (ft)"), "ex-gwf-nwt-p02-difference")

# The largest head differences over all layers and periods, next to the values of the
# [parity table](@ref "Parity table"), and the wall times (CPU: this build, including
# compilation for the first run of the session; GPU: the parity table, AMD Radeon RX
# 7800 XT). The model writes no budget file, so the budget column compares the listing
# tables only.

stats = Dict(p => (; cmp = compare_with_mf6(mf6.dir, runs[p]), runs[p].time) for p in (:double, :single))
comparison_table("ex-gwf-nwt-p02b", stats; unit = "ft")
