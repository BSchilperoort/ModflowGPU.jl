# Precision presets (PLAN.md section 3.7), tested with fake backends that do or do not
# support Float64.
struct FakeFloat64Backend <: KernelAbstractions.Backend end
struct FakeNoFloat64Backend <: KernelAbstractions.Backend end
ModflowGPU.supports_float64(::FakeNoFloat64Backend) = false

@testset "Precision presets" begin
    double = (; T = Float64, Ts = Float64, Tp = Float64)
    single = (; T = Float32, Ts = Float32, Tp = Float32)
    mixed = (; T = Float64, Ts = Float64, Tp = Float32)

    @test ModflowGPU.supports_float64(CPU())
    @test ModflowGPU.supports_float64(FakeFloat64Backend())  # default is true
    @test !ModflowGPU.supports_float64(FakeNoFloat64Backend())

    @testset "Backend with Float64" begin
        for backend in (CPU(), FakeFloat64Backend())
            @test ModflowGPU.resolve_precision(backend) == double
            @test ModflowGPU.resolve_precision(backend, :auto) == double
            @test ModflowGPU.resolve_precision(backend, :double) == double
            @test ModflowGPU.resolve_precision(backend, :single) == single
            @test ModflowGPU.resolve_precision(backend, "single") == single
            @test ModflowGPU.resolve_precision(backend, "Double") == double
            @test ModflowGPU.resolve_precision(backend, :mixed) == mixed
            @test ModflowGPU.resolve_precision(backend, "mixed") == mixed
        end
        T, Ts, Tp = ModflowGPU.resolve_precision(CPU(), :single)
        @test (T, Ts, Tp) == (Float32, Float32, Float32)
    end

    @testset "Backend without Float64" begin
        backend = FakeNoFloat64Backend()
        @test ModflowGPU.resolve_precision(backend) == single
        @test ModflowGPU.resolve_precision(backend, :auto) == single
        @test ModflowGPU.resolve_precision(backend, :single) == single
        # the error names the backend and the preset to use instead
        @test_throws "does not support Float64" ModflowGPU.resolve_precision(backend, :double)
        @test_throws "FakeNoFloat64Backend" ModflowGPU.resolve_precision(backend, :double)
        @test_throws "precision = :single" ModflowGPU.resolve_precision(backend, :double)
        @test_throws "does not support Float64" ModflowGPU.resolve_precision(backend, :mixed)
        @test_throws "precision = :single" ModflowGPU.resolve_precision(backend, :mixed)
    end

    @testset "Unknown presets" begin
        @test_throws "Unknown precision preset :quad" ModflowGPU.resolve_precision(CPU(), :quad)
        @test_throws "Unknown precision preset :quad" ModflowGPU.resolve_precision(CPU(), "quad")
    end

    @testset "for_each_precision" begin
        seen = Symbol[]
        for_each_precision(; backend = CPU()) do p
            push!(seen, p.preset)
            @test p.T == p.Ts == p.Tp == (p.preset === :double ? Float64 : Float32)
        end
        @test seen == [:single, :double]

        # presets that need Float64 are skipped on backends without it
        empty!(seen)
        for_each_precision(; backend = FakeNoFloat64Backend()) do p
            push!(seen, p.preset)
            @test p.T == Float32
        end
        @test seen == [:single]

        empty!(seen)
        for_each_precision(; backend = CPU(), presets = SOLVER_TEST_PRESETS) do p
            push!(seen, p.preset)
            @test p.T == p.Ts == Float64 || p.preset === :single
            @test p.Tp == (p.preset === :double ? Float64 : Float32)
        end
        @test seen == [:single, :mixed, :double]
    end
end
