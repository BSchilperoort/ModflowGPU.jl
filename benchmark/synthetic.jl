# Synthetic coupling-model generator (T3.4).
#
# A coupling-like GWF model (ModflowCoupling's package set) of any `ncol × nrow`: 2
# confined layers, a land mask with a constant-head ocean rim (CHD in both layers), a
# DEM, log-normal K with ~5.5 orders of magnitude contrast, a river network (RIV on
# cells of high flow accumulation on the DEM), DRN and EVT on the other land cells, RCH
# on all land cells, uniform DELR/DELC, STORAGECOEFFICIENT storage. Stress periods: one
# steady-state day, then 12 daily transient periods with changing RCH and RIV stage.
#
# Deterministic: every random number is a counter-based hash (SplitMix64) of
# `(seed, field, octave, lattice point)`, so the model does not depend on the Julia
# version or on Random's generators, and fields are computed per cell without per-cell
# allocations.
#
# Use from Julia (tests include this file):
#
#     include("benchmark/synthetic.jl")
#     spec = SyntheticModels.synthetic_simulation(:tiny)   # SimulationSpec in memory
#     SyntheticModels.write_synthetic(dir, :tiny)          # MF6 files (small presets)
#
# As a script (build times, sizes, device memory; `rocm` also measures the device):
#
#     julia --project=benchmark benchmark/synthetic.jl cpu tiny indus continental
#     julia --project=benchmark benchmark/synthetic.jl rocm indus global5m regional30s global30s
#     julia --project=benchmark benchmark/synthetic.jl rocm indus global5m outer=2
#
# `global30s` is only estimated (from `indus`'s bytes per unit and `global5m`'s counts).
module SyntheticModels

    using ModflowGPU
    using ModflowGPU: build_dis, build_npf, build_ic, build_sto, build_oc, build_boundary, build_model,
        build_tdis, build_ims, build_simulation, write_simulation, SimulationSpec
    using ModflowGPU.Adapt: Adapt

    export SYNTHETIC_PRESETS, synthetic_fields, synthetic_simulation, write_synthetic, synthetic_counts,
        device_bytes, memory_coefficients, estimate_device_bytes, scaled_counts, synthetic_main

    """
        SYNTHETIC_PRESETS

    Grid presets: `ncol`, `nrow`, land fraction, cell size (m, DELR = DELC), whether the
    grid edge is ocean (`island`, a continent in the ocean) or the land touches the edge
    (`global`), and whether the model may be built in memory on one host (`global30s`: no,
    memory estimate only).
    """
    const SYNTHETIC_PRESETS = (
        tiny = (; ncol = 50, nrow = 40, land = 0.6, cellsize = 9000.0, island = true, buildable = true),
        indus = (; ncol = 204, nrow = 180, land = 0.42, cellsize = 9000.0, island = true, buildable = true),
        continental = (; ncol = 1000, nrow = 800, land = 0.55, cellsize = 9000.0, island = true, buildable = true),
        global5m = (; ncol = 4320, nrow = 2160, land = 0.35, cellsize = 9000.0, island = false, buildable = true),
        regional30s = (; ncol = 2040, nrow = 1800, land = 0.42, cellsize = 900.0, island = true, buildable = true),
        global30s = (; ncol = 43200, nrow = 21600, land = 0.35, cellsize = 900.0, island = false, buildable = false),
    )

    const DEFAULT_SEED = 0x5eed_0000_0000_0034
    const NPER_TRANSIENT = 12
    const RIVER_ACCUMULATION = 10      # RIV on land cells draining at least this many cells
    const RELIEF = 2000.0              # m, the highest DEM value

    _preset(p::Symbol) = haskey(SYNTHETIC_PRESETS, p) ? SYNTHETIC_PRESETS[p] :
        throw(ArgumentError("unknown preset $p; presets: $(join(keys(SYNTHETIC_PRESETS), ", "))"))
    _preset(p::NamedTuple) = p

    # ---------------------------------------------------------------------------------------
    # Counter-based random numbers and value noise
    # ---------------------------------------------------------------------------------------

    @inline function _splitmix(seed::UInt64, key::UInt64)
        z = seed + key * 0x9e3779b97f4a7c15
        z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
        z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
        return z ⊻ (z >> 31)
    end

    # Uniform in [0, 1) for (field tag, octave, lattice point).
    @inline _u01(seed, tag, o, a, b) =
        (_splitmix(seed, (UInt64(tag) << 58) ⊻ (UInt64(o) << 52) ⊻ (UInt64(a) << 26) ⊻ UInt64(b)) >> 11) * 0x1.0p-53

    @inline _smooth(t) = t * t * (3 - 2t)

    """
        value_noise!(out, seed, tag; spacing, octaves, persistence = 0.5) -> out

    Fill `out` (`(ncol, nrow)`) with multi-octave value noise in about `[-1, 1]`: octave `o`
    interpolates (smoothstep, bilinear) uniform values on a lattice of `spacing / 2^(o-1)`
    cells, with weight `persistence^(o-1)`. Deterministic in `(seed, tag)`.
    """
    function value_noise!(out::AbstractMatrix{Float64}, seed::UInt64, tag::Integer; spacing::Real, octaves::Integer, persistence::Real = 0.5)
        ncol, nrow = size(out)
        fill!(out, 0.0)
        wsum = 0.0
        for o in 1:octaves
            s = spacing / 2.0^(o - 1)
            w = persistence^(o - 1)
            wsum += w
            for i in 1:nrow
                y = (i - 0.5) / s
                J = floor(Int, y)
                fy = _smooth(y - J)
                for j in 1:ncol
                    x = (j - 0.5) / s
                    I = floor(Int, x)
                    fx = _smooth(x - I)
                    v00 = _u01(seed, tag, o, I, J)
                    v10 = _u01(seed, tag, o, I + 1, J)
                    v01 = _u01(seed, tag, o, I, J + 1)
                    v11 = _u01(seed, tag, o, I + 1, J + 1)
                    v = (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy
                    out[j, i] += w * (2v - 1)
                end
            end
        end
        out ./= wsum
        return out
    end

    # Noise lattice: the coarsest octave has ~4 features across the grid, the finest ~3 cells.
    _octaves(ncol, nrow) = (s = max(ncol, nrow) / 4; (s, max(1, floor(Int, log2(s / 3)) + 1)))

    function _noise(seed, tag, ncol, nrow; persistence = 0.5)
        s, o = _octaves(ncol, nrow)
        return value_noise!(Matrix{Float64}(undef, ncol, nrow), seed, tag; spacing = s, octaves = o, persistence)
    end

    # Standardise in place (mean 0, standard deviation 1).
    function _standardize!(a)
        m = sum(a) / length(a)
        s = sqrt(sum(x -> (x - m)^2, a) / length(a))
        a .= (a .- m) ./ s
        return a
    end

    # Approximate quantile from a strided sample of at most ~2^20 values (deterministic).
    function _quantile(a, q)
        stride = max(1, length(a) >> 20)
        v = sort!(a[1:stride:end])
        return v[clamp(round(Int, q * length(v)), 1, length(v))]
    end

    # ---------------------------------------------------------------------------------------
    # Routing: priority flood (Barnes et al. 2014) from the ocean rim over the land cells
    # ---------------------------------------------------------------------------------------

    # Binary min-heap on (key, index); ties go to the lower index, so the order is unique.
    struct _Heap
        key::Vector{Float64}
        idx::Vector{Int32}
    end
    _Heap(n) = (h = _Heap(Float64[], Int32[]); sizehint!(h.key, n); sizehint!(h.idx, n); h)
    @inline _less(h, a, b) = h.key[a] < h.key[b] || (h.key[a] == h.key[b] && h.idx[a] < h.idx[b])
    @inline function _swap!(h, a, b)
        h.key[a], h.key[b] = h.key[b], h.key[a]
        h.idx[a], h.idx[b] = h.idx[b], h.idx[a]
        return nothing
    end
    function _push!(h::_Heap, k, i)
        push!(h.key, k)
        push!(h.idx, i)
        c = length(h.key)
        while c > 1
            p = c >> 1
            _less(h, c, p) || break
            _swap!(h, c, p)
            c = p
        end
        return nothing
    end
    function _pop!(h::_Heap)
        k, i = h.key[1], h.idx[1]
        n = length(h.key)
        h.key[1], h.idx[1] = h.key[n], h.idx[n]
        pop!(h.key)
        pop!(h.idx)
        n -= 1
        c = 1
        while true
            l = 2c
            l > n && break
            m = (l + 1 <= n && _less(h, l + 1, l)) ? l + 1 : l
            _less(h, m, c) || break
            _swap!(h, m, c)
            c = m
        end
        return k, i
    end

    """
        flow_accumulation(kind, dem) -> (down, acc)

    D8 routing of the land cells (`kind == 1`) to the ocean rim (`kind == 2`) by priority
    flood: cells are visited in order of their depression-filled elevation, each draining to
    the cell it was reached from (`down`, a linear index; rim cells and unreached cells 0).
    `acc` is the number of land cells draining through each cell (itself included).
    """
    function flow_accumulation(kind::AbstractMatrix{UInt8}, dem::AbstractMatrix{Float64})
        ncol, nrow = size(kind)
        n = ncol * nrow
        down = zeros(Int32, n)
        visited = falses(n)
        order = Int32[]
        sizehint!(order, count(==(0x01), kind))
        h = _Heap(1 << 16)
        for c in 1:n
            if kind[c] == 0x02
                visited[c] = true
                _push!(h, dem[c], Int32(c))
            end
        end
        while !isempty(h.key)
            z, c = _pop!(h)
            kind[c] == 0x01 && push!(order, c)
            j, i = mod1(c, ncol), (c - 1) ÷ ncol + 1
            for di in -1:1, dj in -1:1
                (di == 0 && dj == 0) && continue
                jj, ii = j + dj, i + di
                (1 <= jj <= ncol && 1 <= ii <= nrow) || continue
                nb = jj + (ii - 1) * ncol
                (visited[nb] || kind[nb] != 0x01) && continue
                visited[nb] = true
                down[nb] = c
                _push!(h, max(dem[nb], z), Int32(nb))
            end
        end
        acc = zeros(Int32, n)
        for c in order
            acc[c] = 1
        end
        for k in length(order):-1:1
            c = order[k]
            d = down[c]
            (d > 0 && kind[d] == 0x01) && (acc[d] += acc[c])
        end
        return down, acc
    end

    # ---------------------------------------------------------------------------------------
    # Fields
    # ---------------------------------------------------------------------------------------

    """
        synthetic_fields(preset; seed = DEFAULT_SEED) -> NamedTuple

    The host fields of a synthetic model, all `(ncol, nrow)` (`preset` a name of
    `SYNTHETIC_PRESETS` or a NamedTuple like its entries): `kind` (0 inactive ocean, 1 land,
    2 ocean rim = CHD), `top` (DEM, 0 on the rim), `botm` (`(ncol, nrow, 2)`; layer 1 is 10 %
    of the aquifer, as ModflowCoupling), `k` (`(ncol, nrow, 2)`, m/d, log-normal),
    `sc` (storage coefficient), `acc` (flow accumulation, cells), `rch` (mean recharge,
    m/d), `ddrn` (drain depth below the DEM, m).
    """
    function synthetic_fields(preset = :tiny; seed::UInt64 = DEFAULT_SEED)
        p = _preset(preset)
        ncol, nrow = p.ncol, p.nrow
        # Land: noise, lowered towards the grid edge for an island continent.
        f = _noise(seed, 1, ncol, nrow; persistence = 0.55)
        if p.island
            for i in 1:nrow, j in 1:ncol
                e = max(abs(2 * (j - 0.5) / ncol - 1), abs(2 * (i - 0.5) / nrow - 1))
                f[j, i] -= 1.5 * e^6
            end
        end
        thr = _quantile(f, 1 - p.land)
        fmax = maximum(f)
        kind = zeros(UInt8, ncol, nrow)
        for c in eachindex(f)
            f[c] > thr && (kind[c] = 0x01)
        end
        for i in 1:nrow, j in 1:ncol
            kind[j, i] == 0x00 || continue
            coast = (j > 1 && kind[j - 1, i] == 0x01) || (j < ncol && kind[j + 1, i] == 0x01) ||
                (i > 1 && kind[j, i - 1] == 0x01) || (i < nrow && kind[j, i + 1] == 0x01)
            coast && (kind[j, i] = 0x02)
        end
        # DEM from the land field: 0 at the coast, rising inland.
        top = f  # reuse the buffer
        for c in eachindex(top)
            top[c] = kind[c] == 0x01 ? 1.0 + RELIEF * ((f[c] - thr) / (fmax - thr))^1.5 : 0.0
        end
        # Aquifer thickness 50–350 m; layer 1 = 10 %.
        aq = _noise(seed, 2, ncol, nrow)
        botm = Array{Float64}(undef, ncol, nrow, 2)
        for c in eachindex(aq)
            a = 200.0 + 150.0 * clamp(aq[c] * 2, -1.0, 1.0)
            botm[c] = top[c] - 0.1a
            botm[c + ncol * nrow] = top[c] - a
        end
        # Log-normal K: log10 K1 ~ N(0.5, 1) clipped to [-2.5, 2.5] (5 orders), layer 2 a
        # correlated, lower field clipped to [-3, 2.5] (5.5 orders over both layers).
        z1 = _standardize!(_noise(seed, 3, ncol, nrow; persistence = 0.6))
        z2 = _standardize!(_noise(seed, 4, ncol, nrow; persistence = 0.6))
        k = Array{Float64}(undef, ncol, nrow, 2)
        for c in eachindex(z1)
            l1 = clamp(0.5 + z1[c], -2.5, 2.5)
            k[c] = 10.0^l1
            k[c + ncol * nrow] = 10.0^clamp(l1 - 0.3 + 0.6 * z2[c], -3.0, 2.5)
        end
        # Storage coefficient 0.05–0.2 (STORAGECOEFFICIENT, ModflowCoupling: SS = SY).
        sc = aq  # reuse
        value_noise!(sc, seed, 5; spacing = _octaves(ncol, nrow)[1], octaves = _octaves(ncol, nrow)[2])
        sc .= 0.125 .+ 0.075 .* clamp.(2 .* sc, -1.0, 1.0)
        # Recharge 0.1–2 mm/d (wetter at low elevation) and drain depth 1–3 m.
        rch = z2  # reuse
        value_noise!(rch, seed, 6; spacing = _octaves(ncol, nrow)[1], octaves = _octaves(ncol, nrow)[2])
        for c in eachindex(rch)
            rch[c] = 1.0e-3 * clamp(1.0 + 1.5 * rch[c] - 0.5 * top[c] / RELIEF, 0.1, 2.0)
        end
        ddrn = z1  # reuse
        value_noise!(ddrn, seed, 7; spacing = _octaves(ncol, nrow)[1] / 4, octaves = 3)
        ddrn .= 2.0 .+ clamp.(2 .* ddrn, -1.0, 1.0)
        _, acc = flow_accumulation(kind, top)
        return (; ncol, nrow, cellsize = p.cellsize, kind, top, botm, k, sc, acc = reshape(acc, ncol, nrow), rch, ddrn)
    end

    # ---------------------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------------------

    # Seasonal factors of the transient periods 2..13 (period 1 is the steady-state mean).
    _rch_factor(kper) = kper == 1 ? 1.0 : 1.0 + 0.8 * sin(2π * (kper - 2) / NPER_TRANSIENT)
    _stage_factor(kper) = kper == 1 ? 1.0 : 1.0 + 0.5 * cos(2π * (kper - 2) / NPER_TRANSIENT)

    """
        synthetic_ims(kind = :parity; settings...) -> ImsSpec

    `:parity`: tight closures for the parity tests (OUTER_DVCLOSE 1e-9, INNER_DVCLOSE
    1e-10, INNER_RCLOSE 1e-3 m³/d, CG, 5000 inner and 500 outer iterations) with DBD
    under-relaxation (theta 0.9, kappa 1e-4, gamma 0: COMPLEXITY MODERATE's values; without
    it the Picard iterations of the steady state cycle between DRN/EVT on/off states). CG,
    not ModflowCoupling's BiCGSTAB: the matrix is symmetric, and BiCGSTAB with the Jacobi
    preconditioner diverges in `:single` in the steady state (docs/scaling.md);
    `:production`: ModflowCoupling's closures (5 m, 5 m, 10 m³/d STRICT, 200/500 iterations,
    BiCGSTAB); `:coupling`: all of ModflowCoupling's IMS (`:production` with COMPLEXITY
    COMPLEX, so backtracking, SIMPLE under-relaxation 0.99, RELAXATION_FACTOR 0; T6.2a).
    `settings` override any of them.
    """
    function synthetic_ims(kind::Symbol = :parity; settings...)
        base = if kind === :parity
            (;
                outer_dvclose = 1.0e-9, outer_maximum = 500, inner_maximum = 5000, inner_dvclose = 1.0e-10, inner_rclose = 1.0e-3,
                linear_acceleration = :cg, under_relaxation = :dbd, under_relaxation_theta = 0.9, under_relaxation_kappa = 1.0e-4, under_relaxation_gamma = 0.0,
            )
        elseif kind === :production || kind === :coupling
            (; outer_dvclose = 5.0, outer_maximum = 200, inner_maximum = 500, inner_dvclose = 5.0, inner_rclose = 10.0, rclose_option = :strict, linear_acceleration = :bicgstab)
        else
            throw(ArgumentError("synthetic_ims: kind must be :parity, :production or :coupling"))
        end
        if kind === :coupling
            base = merge(base, (; complexity = :complex, under_relaxation = :simple, under_relaxation_gamma = 0.99, relaxation_factor = 0.0))
        end
        return build_ims(; print_option = :summary, merge(base, (; settings...))...)
    end

    """
        synthetic_simulation(preset = :tiny; seed, name = "syn", ims = synthetic_ims(),
                             output = true, fields = nothing, ntransient = 12) -> SimulationSpec

    The synthetic coupling-like model of `preset` as a `SimulationSpec` (builder API). `output`
    adds OC with SAVE HEAD/BUDGET ALL (`<name>.hds`, `<name>.cbc`). Pass `fields` (from
    `synthetic_fields`) to reuse them. `output = :none`: an OC without output records (mf6
    then writes no head arrays to its listing; `false` omits OC). `ntransient`: the number of daily transient periods
    after the steady state (the seasonal RCH and stage factors repeat every 12 periods; the
    benchmark, benchmark/run.jl, uses 30).
    """
    function synthetic_simulation(
            preset = :tiny; seed::UInt64 = DEFAULT_SEED, name::AbstractString = "syn", ims = synthetic_ims(),
            output::Union{Bool, Symbol} = true, fields = nothing, ntransient::Integer = NPER_TRANSIENT,
        )
        p = _preset(preset)
        p.buildable || throw(ArgumentError("preset $(preset) is not built in memory (use estimate_device_bytes)"))
        f = something(fields, synthetic_fields(p; seed))
        ncol, nrow = f.ncol, f.nrow
        ncpl = ncol * nrow
        area = f.cellsize^2
        idomain = Int32[k == 0x00 ? 0 : 1 for k in f.kind]
        dis = build_dis(; delr = f.cellsize, delc = f.cellsize, top = f.top, botm = f.botm, idomain, length_units = :meters)
        npf = build_npf(dis; icelltype = 0, k = f.k, k33 = f.k .* 0.1, save_flows = true)
        sc = Array{Float64}(undef, ncol, nrow, 2)
        sc[:, :, 1] .= f.sc
        sc[:, :, 2] .= f.sc
        sto = build_sto(dis; iconvert = 0, ss = sc, storagecoefficient = true, save_flows = true, periods = [1 => :steady_state, 2 => :transient])
        # Cells (user nodes, layer 1 = linear index of (ncol, nrow)).
        land = Int32[c for c in 1:ncpl if f.kind[c] == 0x01]
        rim = Int32[c for c in 1:ncpl if f.kind[c] == 0x02]
        isriv = [f.acc[c] >= RIVER_ACCUMULATION for c in land]
        riv = land[isriv]
        drn = land[.!isriv]
        ntransient >= 1 || throw(ArgumentError("ntransient must be at least 1"))
        output isa Bool || output === :none || throw(ArgumentError("output must be true, false or :none, got $(repr(output))"))
        nper = 1 + ntransient
        # CHD: sea level in both layers of the rim.
        chd = build_boundary(:CHD, dis; name = "chd", save_flows = true, periods = [1 => (; cells = [rim; rim .+ Int32(ncpl)], values = zeros(2length(rim)))])
        # RIV: width 2√acc m (5–500), conductance width × 1.5 cellsize / 1 d; bed 1 + ln(acc)/2 m
        # below the DEM (above the layer bottom), bankfull depth 0.5 + 0.1√acc m (≤ 5 m) times
        # the seasonal factor.
        nr = length(riv)
        cond = Vector{Float64}(undef, nr)
        rbot = Vector{Float64}(undef, nr)
        depth = Vector{Float64}(undef, nr)
        for (r, c) in enumerate(riv)
            a = Float64(f.acc[c])
            cond[r] = clamp(2sqrt(a), 5.0, 500.0) * 1.5 * f.cellsize
            rbot[r] = f.top[c] - min(1.0 + 0.5 * log(a), 0.5 * (f.top[c] - f.botm[c]))
            depth[r] = min(0.5 + 0.1 * sqrt(a), 5.0)
        end
        rivp = [kper => (; cells = riv, values = hcat(rbot .+ depth .* _stage_factor(kper), cond, rbot)) for kper in 1:nper]
        # Starting heads: 0.5 m above the drain level on land (drains active from the start:
        # at the drain level every DRN is off in the first Picard iteration, and recharge on
        # low-K land (T ~ 1 m²/d) sends it 255 km up on indus, in mf6 identically), the
        # stage in river cells (a river cell below its bed infiltrates cond · depth, up to
        # 3e7 m³/d: 290 km with both), 0 on the rim.
        strt = zeros(ncol, nrow, 2)
        for c in land
            strt[c] = strt[c + ncpl] = f.top[c] - f.ddrn[c] + 0.5
        end
        for (r, c) in enumerate(riv)
            strt[c] = strt[c + ncpl] = rbot[r] + depth[r]
        end
        ic = build_ic(dis; strt)
        rivb = build_boundary(:RIV, dis; name = "riv", save_flows = true, periods = rivp)
        # DRN: land cells off the river network, ddrn below the DEM, conductance
        # KQ3 · SC · area with KQ3 = 0.02 / d (ModflowCoupling's linear reservoir).
        drnv = Matrix{Float64}(undef, length(drn), 2)
        for (r, c) in enumerate(drn)
            drnv[r, 1] = f.top[c] - f.ddrn[c]
            drnv[r, 2] = 0.02 * f.sc[c] * area
        end
        drnb = build_boundary(:DRN, dis; name = "drn", save_flows = true, periods = [1 => (; cells = drn, values = drnv)])
        # EVT (capillary rise stand-in): all land cells, surface at the DEM, 1 mm/d, extinction
        # depth 2 m; FIXED_CELL, NSEG 1.
        evtv = Matrix{Float64}(undef, length(land), 3)
        for (r, c) in enumerate(land)
            evtv[r, 1] = f.top[c]
            evtv[r, 2] = 1.0e-3
            evtv[r, 3] = 2.0
        end
        evtb = build_boundary(:EVT, dis; name = "evt", fixed_cell = true, save_flows = true, periods = [1 => (; cells = land, values = evtv)])
        # RCH list on all land cells, FIXED_CELL; the seasonal factor changes it every period.
        r0 = f.rch[land]
        rchb = build_boundary(:RCH, dis; name = "rch", fixed_cell = true, save_flows = true, periods = [kper => (; cells = land, values = r0 .* _rch_factor(kper)) for kper in 1:nper])
        base = lowercase(name)
        # `output = :none`: an OC file with an empty PERIOD 1 block, so mf6 neither saves nor
        # prints (without any OC file mf6 prints the heads of every period to the listing).
        oc = output === true ? build_oc(; head_file = "$base.hds", budget_file = "$base.cbc", periods = [1 => ["SAVE HEAD ALL", "SAVE BUDGET ALL"]]) :
            output === :none ? build_oc(; periods = [1 => String[]]) : nothing
        model = build_model(name; dis, npf, ic, sto, oc, boundaries = [chd, rivb, drnb, evtb, rchb], save_flows = true)
        tdis = build_tdis(; perlen = ones(nper), nstp = 1, tsmult = 1.0, time_units = :days)
        return build_simulation(model; tdis, ims)
    end

    """
        write_synthetic(dir, preset = :tiny; maxcells = 10^6, kwargs...) -> SimulationSpec

    Write the synthetic model of `preset` as MF6 input files in `dir` (`write_simulation`);
    refuses grids of more than `maxcells` cells per layer. `kwargs` go to
    `synthetic_simulation`.
    """
    function write_synthetic(dir::AbstractString, preset = :tiny; maxcells::Integer = 10^6, kwargs...)
        p = _preset(preset)
        p.ncol * p.nrow <= maxcells || throw(ArgumentError("write_synthetic: $(p.ncol) × $(p.nrow) cells per layer exceeds maxcells = $maxcells"))
        spec = synthetic_simulation(p; kwargs...)
        write_simulation(dir, spec)
        return spec
    end

    # ---------------------------------------------------------------------------------------
    # Sizes and device memory
    # ---------------------------------------------------------------------------------------

    """
        synthetic_counts(fields) -> NamedTuple
        synthetic_counts(spec::SimulationSpec) -> NamedTuple

    Model sizes: `nodes` (active cells), `nja` (CSR entries, diagonal included), records per
    stress package (`chd`, `riv`, `drn`, `evt`, `rch`), and land / rim cells per layer.
    From the fields this needs no model build (for `global30s`, use `scaled_counts`).
    """
    function synthetic_counts(f::NamedTuple)
        ncol, nrow = f.ncol, f.nrow
        act(j, i) = f.kind[j, i] != 0x00
        nact = count(!=(0x00), f.kind)
        nh = 0  # horizontal neighbour pairs within one layer
        for i in 1:nrow, j in 1:ncol
            act(j, i) || continue
            (j < ncol && act(j + 1, i)) && (nh += 1)
            (i < nrow && act(j, i + 1)) && (nh += 1)
        end
        nland = count(==(0x01), f.kind)
        nrim = nact - nland
        nriv = count(c -> f.kind[c] == 0x01 && f.acc[c] >= RIVER_ACCUMULATION, eachindex(f.kind))
        nodes = 2nact
        nja = nodes + 2 * (2nh + nact)  # both layers' horizontal pairs and the vertical pairs, both directions
        return (; ncol, nrow, nodes, nja, nland, nrim, chd = 2nrim, riv = nriv, drn = nland - nriv, evt = nland, rch = nland)
    end

    function synthetic_counts(spec::SimulationSpec)
        m = only(spec.models)
        conn = ModflowGPU.build_connectivity(m)
        rec = Dict(lowercase(b.name) => Int(b.maxbound) for b in m.boundaries)
        return (; nodes = Int(conn.nodes), nja = Int(conn.nja), chd = rec["chd"], riv = rec["riv"], drn = rec["drn"], evt = rec["evt"], rch = rec["rch"])
    end

    """
        scaled_counts(c, preset) -> NamedTuple

    Counts of `preset` extrapolated from the counts `c` of a model with the same land fraction
    (e.g. `global5m` for `global30s`): land cells and river cells scale with the area, rim
    cells (the coastline) with the linear resolution, connections per node are kept.
    """
    function scaled_counts(c::NamedTuple, preset)
        p = _preset(preset)
        sa = (p.ncol * p.nrow) / (c.ncol * c.nrow)
        sl = sqrt(sa)
        nland = round(Int, c.nland * sa)
        nrim = round(Int, c.nrim * sl)
        nodes = 2 * (nland + nrim)
        nriv = round(Int, c.riv * sa)
        nja = round(Int, nodes * c.nja / c.nodes)
        return (; ncol = p.ncol, nrow = p.nrow, nodes, nja, nland, nrim, chd = 2nrim, riv = nriv, drn = nland - nriv, evt = nland, rch = nland)
    end

    # Adapt "backend" that records the arrays a structure would move to the device.
    struct _ArrayCollector
        seen::IdDict{Any, Nothing}
        arrays::Vector{Tuple{Int, Int}}  # (bytes, last dimension)
    end
    _ArrayCollector() = _ArrayCollector(IdDict{Any, Nothing}(), Tuple{Int, Int}[])
    function Adapt.adapt_storage(c::_ArrayCollector, x::AbstractArray)
        if !haskey(c.seen, x)
            c.seen[x] = nothing
            push!(c.arrays, (length(x) * sizeof(eltype(x)), ndims(x) == 0 ? 1 : size(x, ndims(x))))
        end
        return x
    end

    """
        device_bytes(sim::ModflowGPU.Simulation) -> NamedTuple

    Bytes of the device arrays of an initialized simulation (model, Picard/Krylov solver,
    budget accumulator; collected through their Adapt rules, each array once), by what they
    scale with: `node` (cells, also `nodes + 1`), `nja` (CSR entries), `njas` (connections,
    `(nja - nodes) / 2`), one entry per stress package (its arrays; records), `other` (fixed
    size, e.g. reduction scratch), and `total`.
    """
    function device_bytes(sim)
        col = _ArrayCollector()
        out = Dict{Symbol, Int}()
        for p in sim.model.packages  # first, so that each package's arrays are its own
            n0 = length(col.arrays)
            Adapt.adapt(col, p)
            out[Symbol(lowercase(p.spec.name))] = sum(first, @view(col.arrays[(n0 + 1):end]); init = 0)
        end
        n0 = length(col.arrays)
        Adapt.adapt(col, sim.model)
        Adapt.adapt(col, sim.solver)
        Adapt.adapt(col, sim.acc)
        nodes, nja = Int(sim.model.conn.nodes), Int(sim.model.conn.nja)
        out[:node] = out[:nja] = out[:njas] = out[:other] = 0
        for (b, l) in @view(col.arrays[(n0 + 1):end])
            cat = l in (nodes, nodes + 1) ? :node : l == nja ? :nja : l == (nja - nodes) ÷ 2 ? :njas : :other
            out[cat] += b
        end
        out[:total] = sum(values(out))
        return (; out...)
    end

    """
        memory_coefficients(sim) -> Dict{Symbol, Float64}

    Device bytes per unit (per node, per CSR entry, per connection, per record of each stress
    package) and the fixed bytes of an initialized simulation (`device_bytes` divided by the
    counts).
    """
    function memory_coefficients(sim)
        b = device_bytes(sim)
        nodes, nja = Int(sim.model.conn.nodes), Int(sim.model.conn.nja)
        c = Dict{Symbol, Float64}(:node => b.node / nodes, :nja => b.nja / nja, :njas => b.njas / ((nja - nodes) ÷ 2), :other => b.other)
        for p in sim.model.packages
            n = Symbol(lowercase(p.spec.name))
            c[n] = getfield(b, n) / p.spec.maxbound
        end
        return c
    end

    """
        estimate_device_bytes(coef, counts) -> Int

    Device bytes of a model with `counts` (`synthetic_counts` / `scaled_counts`) from the
    per-unit coefficients `memory_coefficients` of a smaller model in the same precision.
    """
    estimate_device_bytes(coef::AbstractDict, c::NamedTuple) = round(
        Int, coef[:node] * c.nodes + coef[:nja] * c.nja + coef[:njas] * ((c.nja - c.nodes) ÷ 2) + coef[:other] +
            sum(coef[k] * getfield(c, k) for k in (:chd, :riv, :drn, :evt, :rch))
    )

    # ---------------------------------------------------------------------------------------
    # Benchmark driver (the script at the end of this file)
    # ---------------------------------------------------------------------------------------

    _r(x, d = 2) = string(round(x; digits = d))
    _gib(b) = _r(b / 2.0^30, 3)

    """
        synthetic_main(args, devused = () -> 0)

    The benchmark script: `args` = backend name, preset names (default all), optional
    `outer=N` (run N outer iterations of the first, steady-state, time step). Prints build
    times, counts and device bytes per preset and precision; `devused()` returns the bytes
    the GPU package has allocated (ROCm: `AMDGPU.used_memory()`) for a measured figure. Presets that are not buildable are estimated
    from `indus`'s per-unit bytes and `global5m`'s counts, so list them after those.
    """
    function synthetic_main(args, devused = () -> 0)
        bname = isempty(args) ? "cpu" : args[1]
        backend = ModflowGPU.select_backend(bname)
        names = Symbol[Symbol(a) for a in args[2:end] if !occursin('=', a)]
        isempty(names) && (names = collect(keys(SYNTHETIC_PRESETS)))
        m = match(r"outer=(\d+)", join(args, " "))
        nouter = isnothing(m) ? 0 : parse(Int, m.captures[1])
        measured = devused() != 0 || occursin("rocm", bname)
        coef = Dict{Symbol, Any}()
        counts5m = nothing
        for name in names
            p = SYNTHETIC_PRESETS[name]
            if !p.buildable
                c5 = something(counts5m, synthetic_counts(synthetic_fields(:global5m)))
                c = scaled_counts(c5, name)
                println("$name: estimated $(c.nodes) nodes, $(c.nja) nja, land $(c.nland), rim $(c.nrim), riv $(c.riv)")
                for pre in (:single, :double)
                    haskey(coef, pre) && println("$name $pre: estimated device arrays $(_gib(estimate_device_bytes(coef[pre], c))) GiB")
                end
                haskey(coef, :single) || println("$name: list `indus` before it for the per-unit bytes")
                continue
            end
            tf = @elapsed f = synthetic_fields(name)
            ts = @elapsed spec = synthetic_simulation(name; fields = f, output = false)
            c = synthetic_counts(f)
            name === :global5m && (counts5m = c)
            println("$name: fields $(_r(tf)) s, spec $(_r(ts)) s; $(c.ncol) × $(c.nrow), $(c.nodes) nodes, $(c.nja) nja, land $(c.nland), rim $(c.nrim), riv $(c.riv)")
            for pre in (:single, :double)
                (pre === :single || ModflowGPU.supports_float64(backend)) || continue
                GC.gc(true)
                f0 = devused()
                ti = @elapsed sim = ModflowGPU.initialize(spec; backend, precision = pre, write_output = false)
                f1 = devused()
                b = device_bytes(sim)
                name === :indus && (coef[pre] = memory_coefficients(sim))
                est = haskey(coef, pre) ? " (estimate from indus $(_gib(estimate_device_bytes(coef[pre], c))) GiB)" : ""
                mes = measured ? ", measured $(_gib(f1 - f0)) GiB allocated" : ""
                println("$name $pre: initialize $(_r(ti)) s, device arrays $(_gib(b.total)) GiB, $(_r(b.total / c.nodes, 1)) B/node$est$mes")
                if nouter > 0
                    ModflowGPU.prepare_time_step(sim)
                    ModflowGPU.prepare_solve(sim)
                    prev = 0
                    for k in 1:nouter
                        t = @elapsed ModflowGPU.solve(sim)
                        o = sim.outer
                        ni = o.inner - prev
                        prev = o.inner
                        println("    outer $k: $ni inner, $(_r(t)) s ($(_r(1.0e6 * t / max(ni, 1), 0)) µs per inner), max change $(_r(o.hncg, 4))")
                    end
                    measured && println("    allocated after the outer iterations: $(_gib(devused() - f0)) GiB")
                end
                sim = nothing
            end
        end
        return nothing
    end

end # module SyntheticModels

# ---------------------------------------------------------------------------------------
# Script: build times, sizes and device memory per preset
# ---------------------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    let backend = isempty(ARGS) ? "cpu" : ARGS[1], devused = () -> 0
        if backend == "rocm"
            @eval using AMDGPU
            devused = () -> (GC.gc(true); Base.invokelatest(AMDGPU.synchronize); Base.invokelatest(AMDGPU.used_memory))
        elseif backend != "cpu"
            @eval using $(Symbol(ModflowGPU.known_backend_package(backend)))
        end
        Base.invokelatest(SyntheticModels.synthetic_main, ARGS, devused)
    end
end
