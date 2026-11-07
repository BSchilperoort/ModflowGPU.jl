# Synthetic grids and comparison helpers shared by the connectivity and NPF tests.
using ModflowGPU: DisSpec

# Elementwise relative closeness of two arrays of equal length.
_rel_close(a, b; rtol = 1.0e-14) = length(a) == length(b) && all(abs(x - y) <= rtol * max(abs(x), abs(y)) for (x, y) in zip(a, b))

# A coupling-like grid (PLAN.md section 1.1): 2 layers, variable DELR/DELC, sloping top,
# IDOMAIN 0 outside the domain, 1 in layer 1 and 2 in layer 2, with columns where only one
# layer is active.
function _synthetic_coupling_dis()
    ncol, nrow, nlay = 7, 5, 2
    delr = [1000.0, 950.5, 900.25, 1000.0, 1100.0, 1000.0, 875.125]
    delc = [925.0, 1000.0, 1000.0, 1050.75, 990.0]
    top = [250.0 + 3.25 * j - 7.5 * i + 0.125 * i * j for j in 1:ncol, i in 1:nrow]
    botm = Array{Float64}(undef, ncol, nrow, nlay)
    botm[:, :, 1] = top .- [20.0 + 0.5 * j + i for j in 1:ncol, i in 1:nrow]
    botm[:, :, 2] = botm[:, :, 1] .- [150.0 - 2.0 * j + 3.0 * i for j in 1:ncol, i in 1:nrow]
    idomain = zeros(Int32, ncol, nrow, nlay)
    for i in 1:nrow, j in 1:ncol
        inside = !(i == 1 && j >= 5) && !(j == 1 && i >= 4) && !(i == nrow && j == ncol)
        idomain[j, i, 1] = inside && (i, j) != (3, 4) ? 1 : 0   # a hole in layer 1
        idomain[j, i, 2] = inside && (i, j) != (2, 2) ? 2 : 0   # and one in layer 2
    end
    idomain[5, 1, 2] = 2  # layer 2 active below an excluded layer-1 cell
    return DisSpec(; nlay = Int32(nlay), nrow = Int32(nrow), ncol = Int32(ncol), delr, delc, top, botm, idomain)
end

# A 4-layer grid with vertical pass-through cells (IDOMAIN -1): single and stacked
# pass-through cells, pass-through over and under excluded cells, in the top and bottom
# layers, and next to active cells (no horizontal connections to pass-through cells).
function _synthetic_passthrough_dis()
    ncol, nrow, nlay = 4, 3, 4
    delr = [10.0, 12.5, 7.25, 10.0]
    delc = [8.0, 11.0, 9.5]
    top = fill(100.0, ncol, nrow) .+ [0.5 * j - 0.25 * i for j in 1:ncol, i in 1:nrow]
    thick = [5.0, 7.5, 3.25, 12.0]
    botm = Array{Float64}(undef, ncol, nrow, nlay)
    for k in 1:nlay
        botm[:, :, k] = (k == 1 ? top : botm[:, :, k - 1]) .- (thick[k] .+ [0.1 * j * k + 0.05 * i for j in 1:ncol, i in 1:nrow])
    end
    # columns (j, i) => IDOMAIN by layer
    cols = Dict(
        (1, 1) => (1, -1, 1, 1), (2, 1) => (1, -1, -1, 1), (3, 1) => (-1, 1, 1, -1), (4, 1) => (1, -1, 0, 1),
        (1, 2) => (1, 0, -1, 1), (2, 2) => (-1, -1, -1, 1), (3, 2) => (1, 1, 1, 1), (4, 2) => (1, -1, -1, -1),
        (1, 3) => (0, -1, 1, 1), (2, 3) => (1, 1, -1, 0), (3, 3) => (-1, 0, 1, -1), (4, 3) => (1, 2, -3, 3),
    )
    idomain = Array{Int32}(undef, ncol, nrow, nlay)
    for ((j, i), ids) in cols, k in 1:nlay
        idomain[j, i, k] = ids[k]
    end
    return DisSpec(; nlay = Int32(nlay), nrow = Int32(nrow), ncol = Int32(ncol), delr, delc, top, botm, idomain)
end
