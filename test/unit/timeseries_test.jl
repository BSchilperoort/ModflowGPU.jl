# Time series and time-array series (T6.3): TS6/TAS6 reading, the interpolation methods
# against hand-computed values (TimeSeries.f90 `GetValue`, TimeArraySeries.f90
# `GetAverageValues`), the loader checks, MF6's zeroing of TAS-linked arrays in a later
# PERIOD block, and the boundary framework's per-step evaluation (`apply_period!` with a
# `window`, `advance_time_series!`: multiplier and offsets in Float64, then the cast) in
# both precisions. Run parity against MF6 is in the run group (test/models/ts_tas,
# ex-gwf-fhb).
using ModflowGPU: TimeSeries, TimeArraySeries, ts_value, tas_values!, read_time_series_file,
    read_time_array_series_file, read_array_periods, read_input_file, load_simulation, build_connectivity,
    BoundaryPackage, apply_period!, advance_time_series!, bound_values, bound_user!, set_bound!, MF6InputError,
    time_steps, time_step_window, TdisSpec

_ts(method, times, values) = TimeSeries(:S, method, times, values, "test.ts")

@testset "Time series (TS6, TAS6)" begin
    @testset "ts_value: hand-computed values" begin
        t, v = [0.0, 10.0, 20.0], [0.0, 10.0, 30.0]
        lin, stp, lend = _ts(:linear, t, v), _ts(:stepwise, t, v), _ts(:linearend, t, v)
        # LINEAR: time-weighted average of the interpolant.
        @test ts_value(lin, 0.0, 10.0) == 5.0
        @test ts_value(lin, 5.0, 15.0) == 11.25          # (0.5·5·(5+10) + 0.5·5·(10+20)) / 10
        @test ts_value(lin, 10.0, 20.0) == 20.0
        @test ts_value(lin, 12.0, 12.0) == 14.0          # t0 == t1: value at t0
        # STEPWISE: average of the step function.
        @test ts_value(stp, 5.0, 15.0) == 5.0            # (0·5 + 10·5) / 10
        @test ts_value(stp, 10.0, 20.0) == 10.0
        @test ts_value(stp, 0.0, 20.0) == 5.0
        @test ts_value(stp, 15.0, 15.0) == 10.0
        @test ts_value(stp, 20.0, 20.0) == 10.0          # the record before (get_surrounding_records)
        # LINEAREND: interpolated value at the end of the step; t0 ignored.
        @test ts_value(lend, 5.0, 15.0) == 20.0
        @test ts_value(lend, 0.0, 20.0) == 30.0
        @test ts_value(lend, 2.0, 2.5) == 2.5
        # A series that starts after t0 integrates to zero (MF6); one that ends before t1
        # is an error (no extension to the end of the simulation).
        @test ts_value(_ts(:linear, [5.0, 10.0], [1.0, 1.0]), 0.0, 10.0) == 0.0
        @test_throws MF6InputError ts_value(lin, 15.0, 25.0)
        @test_throws MF6InputError ts_value(stp, 15.0, 25.0)
        @test_throws MF6InputError ts_value(lend, 15.0, 25.0)
        @test ts_value(stp, 15.0, 25.0; extend = true) == (10.0 * 5 + 30.0 * 5) / 10
        # Times within is_close (100·eps relative) of a record are that record.
        @test ts_value(lin, 0.0, 10.0 * (1 + 1.0e-15)) ≈ 5.0
    end

    @testset "TS6 files" begin
        mktempdir() do dir
            f = write_input(
                dir, "a.ts", """
                BEGIN attributes
                  NAMES  first  Second  third
                  METHODS  linear  STEPWISE  linearend
                  SFACS  2.0  1.0  0.5
                  AUTODEALLOCATE  false
                END attributes
                BEGIN timeseries
                  0.0   1.0  3.0e30  4.0
                  10.0  2.0  5.0     6.0
                  20.0  3.0  7.0     3.0e30
                END timeseries
                """,
            )
            s = read_time_series_file(f)
            @test [x.name for x in s] == [:FIRST, :SECOND, :THIRD]
            @test [x.method for x in s] == [:linear, :stepwise, :linearend]
            @test s[1].times == [0.0, 10.0, 20.0] && s[1].values == [2.0, 4.0, 6.0]
            @test s[2].times == [10.0, 20.0] && s[2].values == [5.0, 7.0]       # DNODATA skipped
            @test s[3].times == [0.0, 10.0] && s[3].values == [2.0, 3.0]
            @test ts_value(s[1], 0.0, 10.0) == 3.0
            # METHOD and SFAC apply to all series.
            f = write_input(dir, "b.ts", "BEGIN attributes\n NAME a b\n METHOD stepwise\n SFAC 10\nEND attributes\nBEGIN timeseries\n 0 1 2\nEND timeseries\n")
            @test [(x.method, x.values) for x in read_time_series_file(f)] == [(:stepwise, [10.0]), (:stepwise, [20.0])]
            # Errors.
            for (attr, msg) in (
                    (" NAME a\n METHOD cubic\n", "Unknown interpolation method: \"CUBIC\""),
                    (" NAME a\n", "Interpolation method was not set"),
                    (" METHOD linear\n NAME a\n", "NAME attribute not provided before METHOD"),
                    (" NAME a b\n METHODS linear\n", "METHODS attribute does not list a method for all time series"),
                    (" NAME a\n METHOD linear\n COLOUR red\n", "Unknown option found in ATTRIBUTES block: \"COLOUR\""),
                    (" NAME a b\n METHOD linear\n SFACS 1.0\n", "SFACS"),
                )
                f = write_input(dir, "e.ts", "BEGIN attributes\n$(attr)END attributes\nBEGIN timeseries\n 0 1 2\nEND timeseries\n")
                e = input_error_of(() -> read_time_series_file(f))
                ok = occursin(msg, e.msg)
                ok || @info "unexpected message" msg e.msg
                @test ok
            end
            f = write_input(dir, "e.ts", "BEGIN attributes\n NAME a b\n METHOD linear\nEND attributes\nBEGIN timeseries\n 0 1\nEND timeseries\n")
            @test input_error_of(() -> read_time_series_file(f)) isa MF6InputError
            f = write_input(dir, "e.ts", "BEGIN attributes\n NAME a\n METHOD linear\nEND attributes\nBEGIN timeseries\nEND timeseries\n")
            @test occursin("No time-series data", input_error_of(() -> read_time_series_file(f)).msg)
        end
    end

    @testset "TAS6 files and tas_values!: hand-computed values" begin
        mktempdir() do dir
            tas(method) = write_input(
                dir, "a.tas", """
                BEGIN attributes
                  NAME  RchArr
                  METHOD  $method
                  SFAC  0.5
                END attributes
                BEGIN time 0.0
                  INTERNAL
                    2.0 4.0
                END time
                BEGIN time 10.0
                  INTERNAL FACTOR 2.0
                    3.0 6.0
                END time
                BEGIN time 20.0
                  CONSTANT 5.0
                END time
                """,
            )
            lin = read_time_array_series_file(tas("linear"), 2)
            @test lin.name === :RCHARR && lin.method === :linear && lin.times == [0.0, 10.0, 20.0]
            @test lin.arrays == [[1.0, 2.0], [3.0, 6.0], [2.5, 2.5]]
            out = zeros(2)
            @test tas_values!(out, lin, 0.0, 10.0) == [2.0, 4.0]
            # [5, 15]: (0.5·5·(2+3) + 0.5·5·(3+2.75)) / 10, (0.5·5·(4+6) + 0.5·5·(6+4.25)) / 10
            @test tas_values!(out, lin, 5.0, 15.0) ≈ [2.6875, 5.0625] rtol = 1.0e-15
            @test tas_values!(out, lin, 15.0, 15.0) == [2.75, 4.25]
            stp = read_time_array_series_file(tas("stepwise"), 2)
            @test tas_values!(out, stp, 5.0, 15.0) == [2.0, 4.0]    # (1·5 + 3·5)/10, (2·5 + 6·5)/10
            @test tas_values!(out, stp, 12.0, 12.0) == [3.0, 6.0]
            @test_throws MF6InputError tas_values!(out, lin, 15.0, 25.0)
            @test_throws MF6InputError tas_values!(out, stp, 15.0, 25.0)
            f = write_input(dir, "b.tas", "BEGIN attributes\n NAME x\n METHOD linearend\nEND attributes\nBEGIN time 0\n CONSTANT 1\nEND time\n")
            @test occursin("Unknown interpolation method", input_error_of(() -> read_time_array_series_file(f, 2)).msg)
            f = write_input(dir, "c.tas", "BEGIN attributes\n NAME x\nEND attributes\nBEGIN time 0\n CONSTANT 1\nEND time\n")
            @test occursin("Interpolation method not specified", input_error_of(() -> read_time_array_series_file(f, 2)).msg)
        end
    end

    @testset "time_step_window" begin
        tdis = TdisSpec(; perlen = [10.0, 6.0], nstp = Int32[2, 3], tsmult = [1.0, 2.0])
        steps = time_steps(tdis)
        @test time_step_window(steps, 1) == (0.0, 5.0)
        @test time_step_window(steps, 2) == (5.0, 10.0)
        @test all(time_step_window(steps, i) == (steps[i - 1].totim, steps[i].totim) for i in 2:length(steps))
    end

    @testset "READASARRAYS: arrays linked to a TAS are zero in the next block (Reset)" begin
        mktempdir() do dir
            text = """
            BEGIN OPTIONS
             READASARRAYS
             TAS6 FILEIN r.tas
             AUXILIARY m
            END OPTIONS
            BEGIN PERIOD 1
             RECHARGE TIMEARRAYSERIES rchts
             m TIMEARRAYSERIES mts
            END PERIOD
            BEGIN PERIOD 2
             IRCH
              CONSTANT 1
            END PERIOD
            BEGIN PERIOD 3
             RECHARGE
              CONSTANT 0.5
            END PERIOD
            """
            f = read_input_file(write_input(dir, "t.rch", text); dfn = "gwf-rch")
            sp = read_array_periods(f, ModflowGPU.dis_dims(2, 2, 3))
            a1, a2, a3 = sp.data
            @test all(isnan, a1.values["recharge"]) && all(isnan, a1.auxvar)
            @test a2.values["recharge"] == zeros(6) && a2.auxvar == zeros(1, 6) && isempty(a2.tas)
            @test a3.values["recharge"] == fill(0.5, 6) && a3.auxvar == zeros(1, 6)
        end
    end

    # A small simulation with time series in CHD (head, offsets), WEL (rate and the
    # AUXMULTNAME multiplier) and RCHA (TAS).
    files = Dict(
        "mfsim.nam" => "BEGIN timing\n TDIS6 s.tdis\nEND timing\nBEGIN models\n gwf6 g.nam g\nEND models\nBEGIN exchanges\nEND exchanges\nBEGIN solutiongroup 1\n ims6 s.ims g\nEND solutiongroup 1\n",
        "s.tdis" => "BEGIN dimensions\n NPER 2\nEND dimensions\nBEGIN perioddata\n 10.0 2 1.0\n 10.0 1 1.0\nEND perioddata\n",
        "s.ims" => "BEGIN nonlinear\n OUTER_DVCLOSE 1e-6\nEND nonlinear\n",
        "g.nam" => "BEGIN packages\n DIS6 g.dis\n NPF6 g.npf\n IC6 g.ic\n CHD6 g.chd\n WEL6 g.wel\n RCH6 g.rch\nEND packages\n",
        "g.dis" => "BEGIN dimensions\n NLAY 1\n NROW 2\n NCOL 3\nEND dimensions\nBEGIN griddata\n delr\n  CONSTANT 10\n delc\n  CONSTANT 10\n top\n  CONSTANT 500\n botm\n  CONSTANT 0\nEND griddata\n",
        "g.npf" => "BEGIN griddata\n icelltype\n  CONSTANT 0\n k\n  CONSTANT 1\nEND griddata\n",
        "g.ic" => "BEGIN griddata\n strt\n  INTERNAL\n  100 200 300 400 450 480\nEND griddata\n",
        "g.chd" => "BEGIN options\n TS6 FILEIN g.chd.ts\nEND options\nBEGIN dimensions\n MAXBOUND 1\nEND dimensions\nBEGIN period 1\n 1 1 1 hd\nEND period\n",
        "g.chd.ts" => "BEGIN attributes\n NAME hd\n METHOD linear\nEND attributes\nBEGIN timeseries\n 0 100\n 20 140\nEND timeseries\n",
        "g.wel" => "BEGIN options\n AUXILIARY mlt\n AUXMULTNAME mlt\n TS6 FILEIN g.wel.ts\nEND options\nBEGIN dimensions\n MAXBOUND 2\nEND dimensions\nBEGIN period 1\n 1 2 2 q mlt\n 1 2 3 -1.0 mlt\nEND period\n",
        "g.wel.ts" => "BEGIN attributes\n NAMES q mlt\n METHODS stepwise linearend\nEND attributes\nBEGIN timeseries\n 0 -2 1\n 5 -4 3\n 20 -8 5\nEND timeseries\n",
        "g.rch" => "BEGIN options\n READASARRAYS\n TAS6 FILEIN g.rch.tas\nEND options\nBEGIN period 1\n RECHARGE TIMEARRAYSERIES r\nEND period\n",
        "g.rch.tas" => "BEGIN attributes\n NAME r\n METHOD stepwise\nEND attributes\nBEGIN time 0\n CONSTANT 0.001\nEND time\nBEGIN time 10\n INTERNAL\n 1 2 3 4 5 6\nEND time\nBEGIN time 20\n CONSTANT 0\nEND time\n",
    )

    @testset "loader: names must be in the series files" begin
        mktempdir() do dir
            foreach(((k, v),) -> write_input(dir, k, v), files)
            s = load_simulation(dir; strict = true)
            b = Dict(ModflowGPU.boundary_kind(x) => x for x in s.models[1].boundaries)
            @test sort(collect(keys(b[:WEL].timeseries))) == [:MLT, :Q] && collect(keys(b[:RCH].timearrayseries)) == [:R]
            write_input(dir, "g.chd", replace(files["g.chd"], " hd\n" => " nosuch\n"))
            @test occursin("Expected numeric value or time-series name, but found 'NOSUCH'", input_error_of(() -> load_simulation(dir)).msg)
            write_input(dir, "g.chd", files["g.chd"])
            write_input(dir, "g.rch", replace(files["g.rch"], "SERIES r" => "SERIES other"))
            @test occursin("Time-array series \"OTHER\" not found", input_error_of(() -> load_simulation(dir)).msg)
        end
    end

    @testset "boundary framework: per-step values" begin
        mktempdir() do dir
            foreach(((k, v),) -> write_input(dir, k, v), files)
            sim = load_simulation(dir; strict = true)
            m = only(sim.models)
            conn = build_connectivity(m)
            r = Float64.(m.ic.strt)   # any per-cell reference
            steps = time_steps(sim.tdis)
            w = [time_step_window(steps, i) for i in 1:3]
            @test w == [(0.0, 5.0), (5.0, 10.0), (10.0, 20.0)]
            for_each_precision() do p
                T = p.T
                pk = Dict(ModflowGPU.boundary_kind(b) => BoundaryPackage(TEST_BACKEND, T, b, conn, m.dis) for b in m.boundaries)
                chd, wel, rch = pk[:CHD], pk[:WEL], pk[:RCH]
                @test_throws ArgumentError apply_period!(chd, 1, r)   # time series need the step
                for (k, kper) in ((1, 1), (2, 1), (3, 2))
                    for pkg in (chd, wel, rch)
                        apply_period!(pkg, kper, r; window = w[k])
                        advance_time_series!(pkg, w[k], r)
                    end
                    t0, t1 = w[k]
                    # CHD head: LINEAR average = value at the midpoint; stored as an offset.
                    hd = 100.0 + 2.0 * (t0 + t1) / 2
                    @test bound_values(chd, r)[1, 1] == hd - r[1]
                    @test Array(chd.bound)[1, 1] == T(hd - r[1])
                    # WEL: rate STEPWISE average times the multiplier LINEAREND at t1.
                    q = k == 1 ? -2.0 : -4.0
                    mlt = 3.0 + 2.0 * (t1 - 5.0) / 15.0
                    @test bound_values(wel, r)[1, 1:2] ≈ [q * mlt, -1.0 * mlt] rtol = 1.0e-15
                    @test Array(wel.bound)[1, 1:2] == T.(bound_values(wel, r)[1, 1:2])
                    @test bound_user!(wel)[1, 1:2] == [q, -1.0]          # before the multiplier
                    # RCH: STEPWISE TAS (the recharge rate; the kernel multiplies by the area).
                    rv = k < 3 ? fill(0.001, 6) : Float64.(1:6)
                    @test bound_values(rch, r)[1, 1:6] == rv
                end
                # Already evaluated for this window: nothing to do.
                @test !advance_time_series!(chd, w[3], r)
                # A stepping API write is kept for columns without a series; the series
                # entries replace their columns at the next step.
                u = bound_user!(wel)
                u[1, 2] = -7.0
                u[1, 1] = 99.0
                set_bound!(wel, r)
                @test advance_time_series!(wel, (20.0, 20.0), r)
                # q STEPWISE at t = 20 is the record before (-4), the multiplier at t = 20 is 5.
                @test bound_user!(wel)[1, 1:2] == [-4.0, -7.0]
                @test bound_values(wel, r)[1, 1:2] == [-4.0 * 5.0, -7.0 * 5.0]
            end
        end
    end
end
