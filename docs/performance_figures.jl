# Figures of the Performance page, made when the manual is built (included by docs/make.jl).
#
# The plotted values are in docs/src/assets/performance.toml, each with the result file in
# benchmark/results/ it was copied from. `check_performance_data` fails the build when a
# value is not found on a line of its file. Nothing is measured here.
module PerformanceFigures

using CairoMakie
using Printf: @sprintf
using TOML: TOML

const ROOT = dirname(@__DIR__)
const DATA = joinpath(@__DIR__, "src", "assets", "performance.toml")
const RESULTS = joinpath(ROOT, "benchmark", "results")

# Okabe-Ito colours (colour-blind safe).
const BLUE = "#0072B2"
const ORANGE = "#E69F00"
const GREEN = "#009E73"
const GREY = "#666666"

"""
    check_performance_data(values)

Check that every value of the data file is on a line of its result file that contains the
`row` text and the `text` of the value, and that `text` parses to `y`.
"""
function check_performance_data(values)
    bad = String[]
    cache = Dict{String, Vector{String}}()
    for v in values
        lines = get!(cache, v["file"]) do
            path = joinpath(RESULTS, v["file"])
            isfile(path) ? readlines(path) : String[]
        end
        found = any(l -> occursin(v["row"], l) && occursin(v["text"], l), lines)
        found || push!(bad, "$(v["file"]): no line with \"$(v["row"])\" and \"$(v["text"])\"")
        parse(Float64, v["text"]) == v["y"] || push!(bad, "$(v["file"]): text $(v["text"]) != y $(v["y"])")
    end
    isempty(bad) || error("performance.toml does not match benchmark/results:\n  " * join(bad, "\n  "))
    return nothing
end

function series_of(values, figure)
    out = Dict{String, Vector{Tuple{Float64, Float64}}}()
    for v in values
        v["figure"] == figure || continue
        push!(get!(out, v["series"], Tuple{Float64, Float64}[]), (Float64(v["x"]), Float64(v["y"])))
    end
    return out
end

function figure_scaling(values, outdir)
    s = series_of(values, "scaling")
    fig = Figure(; size = (720, 440), backgroundcolor = :white)
    ax = Axis(
        fig[1, 1]; xscale = log10, yscale = log10,
        xlabel = "Active nodes (2 layers)", ylabel = "Seconds for a steady state and 30 daily steps",
        title = "Total run time, synthetic models (lower is better)",
        xticks = ([3.3e4, 1.0e5, 8.9e5, 6.6e6], ["33 k", "100 k", "890 k", "6.6 M"]),
    )
    for (key, name, color, marker) in [
            ("mf6", "MODFLOW 6.8.2, serial", GREY, :rect),
            ("cpu", "ModflowGPU, CPU backend (8 threads)", ORANGE, :utriangle),
            ("rocm", "ModflowGPU, ROCm (RX 7800 XT)", BLUE, :circle),
        ]
        pts = sort(s[key])
        scatterlines!(ax, first.(pts), last.(pts); color, marker, markersize = 12, linewidth = 2, label = name)
    end
    axislegend(ax; position = :lt)
    save(joinpath(outdir, "performance_scaling.svg"), fig)
    return
end

function figure_global(values, outdir)
    vs = sort(filter(v -> v["figure"] == "global", values); by = v -> v["x"])
    labels = [v["label"] for v in vs]
    ys = [Float64(v["y"]) for v in vs]
    n = length(vs)
    colors = [startswith(l, "MODFLOW 6") ? GREY : (occursin("CPU backend", l) ? ORANGE : BLUE) for l in labels]
    fig = Figure(; size = (860, 420), backgroundcolor = :white)
    ax = Axis(
        fig[1, 1]; xscale = log10, yreversed = true,
        xlabel = "Milliseconds per daily step (median of days 3 to 31)",
        yticks = (1:n, labels), title = "Real global 5′ model (lower is better)",
    )
    barplot!(ax, 1:n, ys; direction = :x, color = colors, fillto = 10.0)
    for i in 1:n
        text!(ax, ys[i], i; text = " " * string(vs[i]["text"]), align = (:left, :center), fontsize = 13)
    end
    xlims!(ax, 10, 2.0e4)
    save(joinpath(outdir, "performance_global.svg"), fig)
    return
end

function figure_precond(values, outdir)
    s = series_of(values, "precond")
    fig = Figure(; size = (720, 420), backgroundcolor = :white)
    ax = Axis(
        fig[1, 1]; ylabel = "Seconds for a steady state and 30 daily steps",
        xticks = (1:3, ["Jacobi", "MC-ILU0 (default)", "AMG"]),
        title = "Preconditioner and precision, synthetic global 5′ model, ROCm",
    )
    offsets = Dict("double" => -0.27, "mixed" => 0.0, "single" => 0.27)
    colors = Dict("double" => GREY, "mixed" => BLUE, "single" => GREEN)
    for p in ("double", "mixed", "single")
        pts = sort(s[p])
        barplot!(ax, first.(pts) .+ offsets[p], last.(pts); width = 0.25, color = colors[p], label = ":" * p)
        for (x, y) in pts
            text!(ax, x + offsets[p], y; text = @sprintf("%.2f", y), align = (:center, :bottom), fontsize = 11)
        end
    end
    ylims!(ax, 0, 50)
    axislegend(ax; position = :rt)
    save(joinpath(outdir, "performance_precond.svg"), fig)
    return
end

"""
    make_figures(outdir)

Check the data file against `benchmark/results/` and write the SVG figures of the
Performance page into `outdir`.
"""
function make_figures(outdir)
    values = TOML.parsefile(DATA)["value"]
    check_performance_data(values)
    mkpath(outdir)
    CairoMakie.activate!(; type = "svg")
    figure_scaling(values, outdir)
    figure_global(values, outdir)
    figure_precond(values, outdir)
    return nothing
end

end # module
