# Observations (OBS6, T6.7): input (src/input/obs.jl), the number format and the files
# (src/output/obs.jl), and runs against mf6 at test time: test/models/coupling_rde with
# model observations (HEAD, DRAWDOWN, FLOW-JA-FACE; also BINARY) and observations of every
# stress package (by cell, by boundary name, cells without records, TO-MVR; DIGITS), and
# test/models/afr with WEL and WEL-REDUCTION. The MF6 examples with observations
# (ex-gwf-advtidal, ex-gwf-fhb, ex-gwf-rad-disu, ex-gwf-maw-p03c) are compared in the run
# parity test (`_obs_difference`, test/run_parity.jl).
using ModflowGPU: obs_format, read_obs_file, read_observations, OBSNODATA, ObsSpec

@testset "obs_format: MF6's G0 and G20.13" begin
    # Values and times as MF6 6.8.2 writes them (reference runs).
    @test obs_format(19.25173278310135, 16) == "19.25173278310135"
    @test obs_format(0.1302931596091205, 16) == ".1302931596091205"
    @test obs_format(-0.01118514052768522, 16) == "-.1118514052768522E-01"
    @test obs_format(-72112.0217109044, 16) == "-72112.02171090440"
    @test obs_format(-9.930409663606566e-5, 16) == "-.9930409663606566E-04"
    @test obs_format(1.0, 13; lead0 = true) == "1.000000000000"
    @test obs_format(40.0, 13; lead0 = true) == "40.00000000000"
    @test obs_format(5.808117703354e-5, 13; lead0 = true) == "0.5808117703354E-04"
    # Fortran G editing after rounding (gfortran G20.13 / G18.10E3 agree).
    @test obs_format(0.099999999999999995, 13; lead0 = true) == "0.1000000000000"
    @test obs_format(99999.99999999999, 13; lead0 = true) == "100000.0000000"
    @test obs_format(0.09999999, 10; lead0 = true, edigits = 3) == "0.9999999000E-001"
    @test obs_format(3.0e30, 10; lead0 = true, edigits = 3) == "0.3000000000E+031"
    @test obs_format(OBSNODATA, 16) == ".3000000000000000E+31"
    @test obs_format(1.0e100, 16) == ".1000000000000000E+101"
    @test obs_format(0.0, 16) == "0.000000000000000"
    # 16 significant digits read back.
    for k in 1:200
        x = sin(1.7 * k) * 10.0^(k % 41 - 20)
        @test parse(Float64, obs_format(x, 16)) ≈ x rtol = 1.0e-15
    end
end

@testset "OBS6 input" begin
    mktempdir() do dir
        dis = build_dis(; delr = 10.0, delc = 10.0, top = 10.0, botm = cat(fill(5.0, 4, 3), fill(0.0, 4, 3); dims = 3), idomain = (i = ones(Int, 4, 3, 2); i[4, 3, 1] = 0; i))
        f = write_input(
            dir, "a.obs", """
            BEGIN options
              DIGITS 12
              PRINT_INPUT
            END options
            BEGIN continuous FILEOUT a.csv
              h1 head 1 2 3
              dd drawdown 2 3 4
            END continuous
            BEGIN continuous FILEOUT a.bin BINARY
              q flow-ja-face 1 1 1 1 1 2
            END continuous
            """,
        )
        o = read_obs_file(f, :GWF, dis; file = "a.obs")
        @test o isa ObsSpec
        @test o.file == "a.obs" && o.digits == 12 && o.print_input
        @test [x.file for x in o.outputs] == ["a.csv", "a.bin"]
        @test [x.binary for x in o.outputs] == [false, true]
        e = o.outputs[1].observations
        @test [x.name for x in e] == ["H1", "DD"] && [x.type for x in e] == ["HEAD", "DRAWDOWN"]
        @test e[1].cells == [(1 - 1) * 12 + (2 - 1) * 4 + 3] && e[2].cells == [12 + 8 + 4]
        @test only(o.outputs[2].observations).cells == [1, 2]
        # DIGITS limits (read_obs_options): 1 → 2, 20 → 16, 0 = G0.
        for (d, want) in ((1, 2), (20, 16), (0, 0))
            write_input(dir, "d.obs", "BEGIN options\n  DIGITS $d\nEND options\nBEGIN continuous FILEOUT d.csv\n  h head 1 1 1\nEND continuous\n")
            @test read_obs_file(joinpath(dir, "d.obs"), :GWF, dis).digits == want
        end
        # Package observations: a cellid or a boundary name.
        write_input(dir, "p.obs", "BEGIN continuous FILEOUT p.csv\n  g1 ghb 1 1 1\n  g2 GHB my-name\n  m to-mvr 2 1 1\nEND continuous\n")
        p = read_obs_file(joinpath(dir, "p.obs"), :GHB, dis)
        e = p.outputs[1].observations
        @test e[1].cells == [1] && e[1].boundname == ""
        @test isempty(e[2].cells) && e[2].boundname == "MY-NAME"
        @test e[3].type == "TO-MVR"
        # Errors, with MF6's messages where it has one.
        bad(text, owner = :GWF) = (write_input(dir, "b.obs", text); input_error_of(() -> read_obs_file(joinpath(dir, "b.obs"), owner, dis)).msg)
        @test occursin("Observation type not found: CHD", bad("BEGIN continuous FILEOUT b.csv\n  x chd 1 1 1\nEND continuous\n"))
        @test occursin("Observation type not found: HEAD", bad("BEGIN continuous FILEOUT b.csv\n  x head 1 1 1\nEND continuous\n", :WEL))
        @test occursin("Observation type not found: TO-MVR", bad("BEGIN continuous FILEOUT b.csv\n  x to-mvr 1 1 1\nEND continuous\n", :CHD))
        @test occursin("outside of the grid", bad("BEGIN continuous FILEOUT b.csv\n  x head 3 1 1\nEND continuous\n"))
        @test occursin("Cell is outside active grid domain: (1,3,4)", bad("BEGIN continuous FILEOUT b.csv\n  x head 1 3 4\nEND continuous\n"))
        @test occursin("Error reading data from ID string", bad("BEGIN continuous FILEOUT b.csv\n  x wel 1 3 4\nEND continuous\n", :WEL))
        @test occursin("could not read an integer", bad("BEGIN continuous FILEOUT b.csv\n  x head 1 a 1\nEND continuous\n"))
        @test occursin("provided more than once", bad("BEGIN continuous FILEOUT b.csv\n  x head 1 1 1\nEND continuous\nBEGIN continuous FILEOUT b.csv\n  y head 1 1 2\nEND continuous\n"))
        @test occursin("Unrecognized option: PRECISION", bad("BEGIN options\n  PRECISION double\nEND options\n"))
        @test occursin("Invalid value for DIGITS", bad("BEGIN options\n  DIGITS -1\nEND options\n"))
        @test occursin("FILEOUT", bad("BEGIN continuous b.csv\n  x head 1 1 1\nEND continuous\n"))
    end
end

# test/models/coupling_rde with observations: the model's (CSV and BINARY) and one OBS6
# file per stress package.
function _rde_with_obs(dir)
    cp(joinpath(@__DIR__, "..", "models", "coupling_rde"), dir)
    write_input(
        dir, "rde.obs", """
        BEGIN continuous FILEOUT rde.obs.csv
          h_1_5_5 head 1 5 5
          h_2_8_11 HEAD 2 8 11
          dd_2_8_11 drawdown 2 8 11
          q_h flow-ja-face 1 5 5 1 5 6
          q_v flow-ja-face 1 8 11 2 8 11
          q_chd flow-ja-face 1 2 2 1 2 1
        END continuous
        BEGIN continuous FILEOUT rde.obs.bin BINARY
          h_2_4_6 head 2 4 6
          q_w flow-ja-face 2 4 6 2 4 7
        END continuous
        """,
    )
    pkg_obs = Dict(
        "rde.chd" => ("chd", "  c1 chd 1 1 1\n  c2 chd 1 8 1\n  c3 chd 1 9 9\n"),
        "rde.wel" => ("wel", "  w1 wel 2 8 11\n  w2 wel 2 4 6\n  m to-mvr 2 8 11\n"),
        "rde.riv" => ("riv", "  r1 riv 1 1 7\n  river riv river\n  none riv nobody\n"),
        "rde.drn" => ("drn", "  d1 drn 1 2 3\n  d2 drn 1 2 11\n"),
        "rde.evt" => ("evt", "  e1 evt 1 1 1\n  e2 evt 1 1 9\n"),
        "rde.evta" => ("evt", "  ea evt 1 6 6\n"),
        "rde.rcha" => ("rch", "  ra rch 1 6 6\n  rb rch 1 1 12\n"),
    )
    for (file, (t, lines)) in pkg_obs
        # DIGITS on the RIV observations; the others MF6's default (G0).
        opts = file == "rde.riv" ? "BEGIN options\n  DIGITS 10\nEND options\n" : ""
        write_input(dir, "$file.obs", "$(opts)BEGIN continuous FILEOUT $file.obs.csv\n$(lines)END continuous\n")
        txt = read(joinpath(dir, file), String)
        txt = replace(txt, r"BEGIN options"i => "BEGIN options\n  OBS6 FILEIN $file.obs"; count = 1)
        write(joinpath(dir, file), txt)
    end
    nam = read(joinpath(dir, "rde.nam"), String)
    write(joinpath(dir, "rde.nam"), replace(nam, "  OC6  rde.oc  oc" => "  OC6  rde.oc  oc\n  OBS6  rde.obs  obs"))
    return dir
end

# test/models/afr with WEL and WEL-REDUCTION observations.
function _afr_with_obs(dir)
    cp(joinpath(@__DIR__, "..", "models", "afr"), dir)
    for (file, cell) in (("afr.wel", "1 1 6"), ("afr_len.wel", "1 1 4"))
        write_input(dir, "$file.obs", "BEGIN continuous FILEOUT $file.obs.csv\n  q wel $cell\n  red wel-reduction $cell\nEND continuous\n")
        txt = read(joinpath(dir, file), String)
        write(joinpath(dir, file), replace(txt, r"BEGIN options"i => "BEGIN options\n  OBS6 FILEIN $file.obs"; count = 1))
    end
    write_input(dir, "afr.obs", "BEGIN continuous FILEOUT afr.obs.csv\n  h6 head 1 1 6\n  h4 head 1 1 4\nEND continuous\n")
    nam = read(joinpath(dir, "afr.nam"), String)
    write(joinpath(dir, "afr.nam"), replace(nam, "  OC6  afr.oc  oc" => "  OC6  afr.oc  oc\n  OBS6  afr.obs  obs"))
    return dir
end

@testset "Observations vs mf6 at test time" begin
    if !isfile(RUN_MF6)
        @info "Skipping the observation runs: mf6 not found at $RUN_MF6 (set REFERENCE_DIR or MF6)"
    else
        for (name, make) in (("coupling_rde", _rde_with_obs), ("afr", _afr_with_obs))
            mktempdir() do tmp
                rundir = make(joinpath(tmp, name))
                out = read(Cmd(`$RUN_MF6`; dir = rundir), String)
                @test occursin("Normal termination", out)
                m = only(load_simulation(rundir).models)
                outputs = _obs_outputs(m)
                @testset "$name" begin
                    for_each_precision() do p
                        mktempdir() do ours
                            r = run_simulation(rundir; backend = TEST_BACKEND, precision = p.preset, outdir = ours)
                            @test r.converged
                            @test r.obs_files == [joinpath(ours, f) for (f, _) in outputs]
                            # The CSV headers and times are MF6's text; values to 16 digits.
                            for (f, _) in outputs
                                endswith(f, ".bin") && continue
                                a, b = readlines(joinpath(ours, f)), readlines(joinpath(rundir, f))
                                @test a[1] == b[1]
                                @test first.(split.(a, ',')) == first.(split.(b, ','))
                            end
                            tins = [ModflowGPU.total_rate_in(b) for b in r.budgets]
                            dh, dq, nv = _obs_difference(m, ours, rundir, tins; htol = p.T === Float64 ? 1.0e-6 : 1.0e-3, ftol = p.T === Float64 ? 1.0e-6 : 1.0e-4)
                            @info "Observations vs mf6: $name, :$(p.preset), $(backend_name(TEST_BACKEND)): max |Δ| heads $(round(dh; sigdigits = 3)), flows $(round(dq; sigdigits = 3)) × scale ($nv values, $(length(outputs)) files)"
                            if name == "coupling_rde"
                                w = read_observations(joinpath(ours, "rde.wel.obs.csv"))
                                # No WEL block in period 1; well 2 4 6 dropped in period 3; no mover.
                                @test w.values[1, :] == fill(OBSNODATA, 3)
                                @test all(w.values[:, 3] .== OBSNODATA)
                                @test w.values[end, 2] == OBSNODATA && w.values[2, 2] != OBSNODATA
                                rv = read_observations(joinpath(ours, "rde.riv.obs.csv"))
                                @test all(rv.values[:, 3] .== OBSNODATA)
                                # DIGITS 10: G18.10E3.
                                @test occursin(r"^0\.\d{10}E\+\d{3}$", split(readlines(joinpath(ours, "rde.riv.obs.csv"))[2], ',')[4])
                                b = read_observations(joinpath(ours, "rde.obs.bin"))
                                @test b.names == ["H_2_4_6", "Q_W"] && length(b.times) == length(r.steps)
                                @test read(joinpath(ours, "rde.obs.bin"), 100) == read(joinpath(rundir, "rde.obs.bin"), 100)
                            end
                        end
                    end
                end
            end
        end
    end
end

@testset "Observations: stepping API and no OBS" begin
    if !isfile(RUN_MF6)
        @info "Skipping: mf6 not found at $RUN_MF6"
    else
        mktempdir() do tmp
            rundir = _afr_with_obs(joinpath(tmp, "afr"))
            # Stepping: one row per finalize_time_step.
            mktempdir() do ours
                sim = initialize(rundir; backend = TEST_BACKEND, precision = :double, outdir = ours)
                ModflowGPU.update(sim)
                # T6.9: a new period with the same records keeps the gather plan and its
                # device buffers (`_obs_plan!`).
                o = only(sim.outputs).obs
                idx, vals = o.idx, o.vals
                ModflowGPU.obs_rp!(o, sim.model)
                @test o.idx === idx && o.vals === vals
                ModflowGPU.update(sim)
                f = joinpath(ours, "afr.obs.csv")
                @test length(readlines(f)) == 3
                finalize_simulation(sim)
                @test length(read_observations(f).times) == 2
            end
            # T6.9: write_output = false still writes the observation files (MF6 writes OBS6
            # output in every run), the same as with output.
            mktempdir() do ours
                r0 = run_simulation(rundir; backend = TEST_BACKEND, precision = :double, outdir = joinpath(ours, "a"))
                r = run_simulation(rundir; backend = TEST_BACKEND, precision = :double, write_output = false, outdir = joinpath(ours, "b"))
                @test length(r.obs_files) == length(r0.obs_files) > 0 && only(r.models).obs_files == r.obs_files
                @test basename.(r.obs_files) == basename.(r0.obs_files)
                @test all(((a, b),) -> read(a) == read(b), zip(r.obs_files, r0.obs_files))
                @test !isfile(joinpath(ours, "b", "afr.lst")) && isempty(only(r.models).head_file)
                # The stepping API without output (BMI-like), one row per time step.
                sim = initialize(rundir; backend = TEST_BACKEND, precision = :double, write_output = false, outdir = joinpath(ours, "c"))
                ModflowGPU.update(sim)
                finalize_simulation(sim)
                @test length(read_observations(joinpath(ours, "c", "afr.obs.csv")).times) == 1
            end
        end
    end
    # FLOW-JA-FACE between cells without a connection (gwf_process_intercell_obs_id).
    mktempdir() do tmp
        rundir = joinpath(tmp, "afr")
        cp(joinpath(@__DIR__, "..", "models", "afr"), rundir)
        write_input(rundir, "afr.obs", "BEGIN continuous FILEOUT afr.obs.csv\n  q flow-ja-face 1 1 1 1 1 3\nEND continuous\n")
        nam = read(joinpath(rundir, "afr.nam"), String)
        write(joinpath(rundir, "afr.nam"), replace(nam, "  OC6  afr.oc  oc" => "  OC6  afr.oc  oc\n  OBS6  afr.obs  obs"))
        e = input_error_of(() -> run_simulation(rundir; backend = TEST_BACKEND, precision = :double, outdir = joinpath(tmp, "out")))
        @test occursin("No connection exists between cells identified in text: 1 1 1 1 1 3", e.msg)
        @test !isfile(joinpath(tmp, "out", "afr.obs.csv"))
    end
    # A model without OBS6 input: no observation output.
    r = run_simulation(joinpath(@__DIR__, "..", "models", "afr"); backend = TEST_BACKEND, precision = :double, write_output = false)
    @test isempty(r.obs_files)
end
