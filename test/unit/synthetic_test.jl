# Synthetic coupling-model generator (T3.4, benchmark/synthetic.jl): determinism, the
# documented field ranges, counts without a build vs the built model, the MF6 round trip,
# and the device-memory estimate (coefficients of `tiny` predict the bytes of `indus`).
isdefined(@__MODULE__, :SyntheticModels) || include(joinpath(@__DIR__, "..", "..", "benchmark", "synthetic.jl"))
using .SyntheticModels: synthetic_fields, synthetic_simulation, synthetic_counts, write_synthetic, device_bytes,
    memory_coefficients, estimate_device_bytes, scaled_counts, SYNTHETIC_PRESETS
using ModflowGPU: spec_identical, spec_difference, load_simulation

@testset "Synthetic coupling models (T3.4)" begin
    @testset "fields" begin
        f = synthetic_fields(:tiny)
        g = synthetic_fields(:tiny)
        @test all(getfield(f, k) == getfield(g, k) for k in keys(f))
        @test synthetic_fields(:tiny; seed = UInt64(1)).k != f.k
        # Counter-based numbers: fixed values, independent of Random and the Julia version.
        @test SyntheticModels._u01(SyntheticModels.DEFAULT_SEED, 1, 1, 0, 0) == 0.22599561359076592
        land = count(==(0x01), f.kind)
        @test abs(land / length(f.kind) - SYNTHETIC_PRESETS.tiny.land) < 0.02
        @test all(f.kind[[1, end], :] .!= 0x01) && all(f.kind[:, [1, end]] .!= 0x01)  # island: ocean edge
        lk = log10.(f.k[cat(f.kind .!= 0x00, f.kind .!= 0x00; dims = 3)])
        @test 4 <= maximum(lk) - minimum(lk) <= 6
        @test all(f.top[f.kind .== 0x01] .>= 1.0) && all(f.top[f.kind .== 0x02] .== 0.0)
        @test all(f.botm[:, :, 1] .< f.top) && all(f.botm[:, :, 2] .< f.botm[:, :, 1])
        # Every land cell drains to the rim; river cells are a minority.
        @test sum(f.acc[f.kind .== 0x01] .== 1) > 0
        nriv = count(f.acc .>= SyntheticModels.RIVER_ACCUMULATION)
        @test 0 < nriv < land ÷ 4
    end

    @testset "model" begin
        spec = synthetic_simulation(:tiny)
        @test spec_identical(spec, synthetic_simulation(:tiny))
        f = synthetic_fields(:tiny)
        c = synthetic_counts(f)
        cs = synthetic_counts(spec)
        @test all(getfield(c, k) == getfield(cs, k) for k in keys(cs))
        m = only(spec.models)
        @test [b.name for b in m.boundaries] == ["DRN", "RIV", "RCH", "EVT", "CHD"]
        @test length(spec.tdis.perlen) == 13 && all(==(1.0), spec.tdis.perlen)
        rch = only(b for b in m.boundaries if b.name == "RCH")
        @test length(rch.periods.iper) == 13 && rch.periods.data[2].bound != rch.periods.data[3].bound
        riv = only(b for b in m.boundaries if b.name == "RIV")
        @test riv.periods.data[2].bound[1, :] != riv.periods.data[3].bound[1, :]  # stage
        mktempdir() do dir
            write_synthetic(dir, :tiny)
            back = load_simulation(dir; strict = true)
            d = spec_difference(spec, back; ignore = (:simdir,))
            isnothing(d) || @info "synthetic tiny round trip: first difference: $d"
            @test isnothing(d)
        end
        @test_throws ArgumentError synthetic_simulation(:global30s)
        @test_throws ArgumentError write_synthetic("unused", :global5m)
    end

    @testset "device memory estimate" begin
        ftiny, findus = synthetic_fields(:tiny), synthetic_fields(:indus)
        for_each_precision() do p
            s1 = ModflowGPU.initialize(synthetic_simulation(:tiny; fields = ftiny); backend = TEST_BACKEND, precision = p.preset, write_output = false)
            s2 = ModflowGPU.initialize(synthetic_simulation(:indus; fields = findus); backend = TEST_BACKEND, precision = p.preset, write_output = false)
            b = device_bytes(s2)
            est = estimate_device_bytes(memory_coefficients(s1), synthetic_counts(findus))
            @test abs(est - b.total) / b.total < 0.01
            @test b.other < 0.01 * b.total
            @info "synthetic: indus device bytes, :$(p.preset): $(b.total) (estimate from tiny $est); $(round(b.total / s2.model.conn.nodes; digits = 1)) B per cell"
        end
        sc = scaled_counts(synthetic_counts(ftiny), :indus)
        @test sc.nodes > 0 && sc.ncol == 204
    end
end
