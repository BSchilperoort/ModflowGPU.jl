# MF6 block parser and DFN schema (T1.1). Uses `write_input` from input_tokenizer_test.jl.
using ModflowGPU: MF6InputError, InputFile, InputBlock, parse_blocks, read_input_file,
    find_block, find_blocks, block_number, keyword, DFN_DEFINITIONS, dfn_definition,
    dfn_block, dfn_variable, dfn_name, dfn_tag

"The `MF6InputError` thrown by `f()` (test fails if there is none)."
function input_error_of(f)
    try
        f()
    catch e
        e isa MF6InputError && return e
        rethrow()
    end
    @test false
    return nothing
end

@testset "Block parser" begin
    @testset "blocks, suffixes, comments" begin
        mktempdir() do dir
            path = write_input(
                dir, "x.wel", """
                # comment before the first block
                stray text outside a block is skipped, as in MF6
                begin Options
                  # comment in a block
                  BOUNDNAMES
                  ts6 filein My.TS
                end OPTIONS

                BEGIN dimensions
                  MAXBOUND 2
                END dimensions

                BEGIN period 1
                  1 1 1 -100.0 well_a  # trailing comment
                END period 1
                BEGIN PERIOD  3
                END PERIOD
                BEGIN continuous FILEOUT Heads.CSV binary
                  h1 HEAD 1 1 1
                END continuous
                """,
            )
            bs = parse_blocks(path)
            @test [b.name for b in bs] == ["OPTIONS", "DIMENSIONS", "PERIOD", "PERIOD", "CONTINUOUS"]
            opt = bs[1]
            @test isempty(opt.suffix)
            @test [l.tokens for l in opt.lines] == [["BOUNDNAMES"], ["ts6", "filein", "My.TS"]]
            @test opt.header.line == 3 && opt.header.file == path
            @test [l.line for l in opt.lines] == [5, 6]
            @test isnothing(opt.open_close)
            @test bs[3].suffix == ["1"]
            @test block_number(bs[3]) == 1
            @test block_number(bs[4]) == 3
            @test isempty(bs[4].lines)
            @test bs[5].suffix == ["FILEOUT", "Heads.CSV", "binary"]  # case preserved
            @test_throws MF6InputError block_number(bs[1])
            @test_throws MF6InputError block_number(bs[5])

            f = InputFile(path, nothing, bs)
            @test find_block(f, "options") === opt
            @test isnothing(find_block(f, "griddata"))
            @test length(find_blocks(f, "Period")) == 2
            @test occursin("PERIOD 3", sprint(show, f))
            @test occursin("OPTIONS", sprint(show, opt))
        end
    end

    @testset "block errors" begin
        mktempdir() do dir
            cases = [
                ("eof", "BEGIN options\n  A\n", 1, "unexpected end of file"),
                ("wrongend", "BEGIN options\n  A\nEND dimensions\n", 3, "END DIMENSIONS"),
                ("noname", "BEGIN options\n  A\nEND\n", 3, "END OPTIONS must be used"),
                ("nested", "BEGIN options\nBEGIN dimensions\nEND dimensions\n", 2, "BEGIN detected before END OPTIONS"),
                ("beginname", "\n\nBEGIN\nEND\n", 3, "block name missing"),
                ("strayend", "BEGIN options\nEND options\nEND options\n", 3, "without a matching BEGIN"),
            ]
            for (name, text, line, msg) in cases
                path = write_input(dir, name, text)
                err = input_error_of(() -> parse_blocks(path))
                @test err.file == path
                @test err.line == line
                @test occursin(msg, sprint(showerror, err))
            end
        end
    end

    @testset "OPEN/CLOSE" begin
        mktempdir() do simdir
            # The package file is in a subdirectory; OPEN/CLOSE names are relative to the
            # simulation directory, as MF6 opens them from its working directory.
            write_input(simdir, "data/Period1.txt", "# comment\n 1 1 1 -5.0\n\n 1 1 2 -6.0\n")
            write_input(simdir, "data/with_end.txt", "2 2 2 1.0\nEND period\nignored\n")
            path = write_input(
                simdir, "model/x.wel", """
                BEGIN options
                  open/close data/opts.txt
                END options
                BEGIN period 1
                  OPEN/CLOSE data/Period1.txt
                END period
                BEGIN period 2
                  OPEN/CLOSE data/list.bin (BINARY)
                END period
                BEGIN period 3
                  open/close data/list.txt sfac 2.0
                  1 1 1 1.0
                END period
                BEGIN period 4
                  OPEN/CLOSE data/with_end.txt
                END period
                """,
            )
            write_input(simdir, "data/opts.txt", "BOUNDNAMES\n")
            bs = parse_blocks(path; simdir)
            @test bs[1].open_close == normpath(joinpath(simdir, "data/opts.txt"))
            @test [l.tokens for l in bs[1].lines] == [["BOUNDNAMES"]]
            p1 = bs[2]
            @test p1.open_close == normpath(joinpath(simdir, "data/Period1.txt"))  # case kept
            @test [l.tokens for l in p1.lines] == [["1", "1", "1", "-5.0"], ["1", "1", "2", "-6.0"]]
            @test [l.line for l in p1.lines] == [2, 4]
            @test all(l -> l.file == p1.open_close, p1.lines)
            @test p1.header.file == path
            # (BINARY) or SFAC: list input, not a whole-block include (uget_block).
            @test isnothing(bs[3].open_close)
            @test keyword(bs[3].lines[1], 1) == "OPEN/CLOSE" && length(bs[3].lines) == 1
            @test isnothing(bs[4].open_close) && length(bs[4].lines) == 2
            # END in the OPEN/CLOSE file ends the block.
            @test [l.tokens for l in bs[5].lines] == [["2", "2", "2", "1.0"]]

            # Relative to the package file's directory by default.
            write_input(simdir, "model/data/opts.txt", "PRINT_INPUT\n")
            path2 = write_input(simdir, "model/y.wel", "BEGIN options\n  OPEN/CLOSE data/opts.txt\nEND options\n")
            bs2 = parse_blocks(path2)
            @test [l.tokens for l in bs2[1].lines] == [["PRINT_INPUT"]]

            # GRIDDATA and time-array-series TIME blocks: OPEN/CLOSE is an array control.
            g = write_input(simdir, "g.npf", "BEGIN griddata\n  OPEN/CLOSE k.txt\nEND griddata\nBEGIN time 1.0\n  OPEN/CLOSE r.txt FACTOR 1.0\nEND time\n")
            bg = parse_blocks(g; simdir)
            @test isnothing(bg[1].open_close) && keyword(bg[1].lines[1], 1) == "OPEN/CLOSE"
            @test isnothing(bg[2].open_close) && length(bg[2].lines) == 1

            cases = [
                ("missing", "BEGIN period 1\n  OPEN/CLOSE nofile.txt\nEND period\n", 2, "not found"),
                ("noname", "BEGIN period 1\n  OPEN/CLOSE\nEND period\n", 2, "missing file name"),
                ("extra", "BEGIN period 1\n  OPEN/CLOSE data/Period1.txt\n  1 1 1 2.0\nEND period\n", 3, "must be END PERIOD"),
                ("eof", "BEGIN period 1\n  OPEN/CLOSE data/Period1.txt\n", 1, "unexpected end of file"),
            ]
            for (name, text, line, msg) in cases
                p = write_input(simdir, name, text)
                err = input_error_of(() -> parse_blocks(p; simdir))
                @test err.line == line
                @test occursin(msg, sprint(showerror, err))
            end
            write_input(simdir, "data/nested.txt", "1 1 1 1.0\nBEGIN period 2\n")
            p = write_input(simdir, "nested", "BEGIN period 1\n  OPEN/CLOSE data/nested.txt\nEND period\n")
            err = input_error_of(() -> parse_blocks(p; simdir))
            @test endswith(err.file, "nested.txt") && err.line == 2
        end
    end

    @testset "DFN schema" begin
        names = keys(DFN_DEFINITIONS)
        for n in (
                "sim-nam", "sim-tdis", "sln-ims", "gwf-nam", "gwf-dis", "gwf-disv", "gwf-disu", "gwf-npf",
                "gwf-sto", "gwf-ic", "gwf-oc", "gwf-chd", "gwf-wel", "gwf-rch", "gwf-rcha", "gwf-evt",
                "gwf-evta", "gwf-ghb", "gwf-riv", "gwf-drn", "utl-ts", "utl-tas", "utl-obs", "exg-gwfgwf",
            )
            @test n in names
        end
        @test !any(n -> startswith(n, "gwt-") || startswith(n, "gwe-") || startswith(n, "prt-"), names)

        npf = dfn_definition("GWF-NPF")
        @test npf.idm
        @test [b.name for b in npf.blocks] == ["OPTIONS", "GRIDDATA"]
        gd = dfn_block(npf, "griddata")
        @test gd.required && !gd.open_close
        ict = dfn_variable(gd, "ICELLTYPE")
        @test ict.type === :integer && ict.reader === :readarray && ict.layered
        @test occursin("nodes", ict.shape)
        @test dfn_tag(ict) == "ICELLTYPE"
        @test isnothing(dfn_variable(gd, "nonexistent"))
        @test isnothing(dfn_block(npf, "PERIOD"))

        wel = dfn_definition("gwf-wel")
        @test wel.multi_package && wel.package_type == "stress-package"
        per = dfn_block(wel, "PERIOD")
        @test per.aggregate && !per.required && per.open_close
        @test per.block_variable.name == "iper" && per.block_variable.type === :integer
        spd = dfn_variable(per, "stress_period_data")
        @test spd.type === :recarray && spd.members == ["cellid", "q", "aux", "boundname"]
        @test dfn_variable(per, "q").time_series
        @test dfn_variable(dfn_block(wel, "options"), "ts6_filename").preserve_case
        @test dfn_block(wel, "DIMENSIONS").required

        @test dfn_block(dfn_definition("utl-tas"), "TIME").block_variable.type === :double
        @test !dfn_block(dfn_definition("utl-tas"), "TIME").open_close
        @test dfn_block(dfn_definition("utl-obs"), "CONTINUOUS").block_variable.type === :record
        @test isnothing(dfn_block(dfn_definition("utl-ts"), "OPTIONS"))
        @test !dfn_definition("sln-ims").idm
        @test [b.name for b in dfn_definition("sim-nam").blocks] == ["OPTIONS", "TIMING", "MODELS", "EXCHANGES", "SOLUTIONGROUP"]
        @test_throws ArgumentError dfn_definition("gwt-adv")

        @test dfn_name("WEL6") == "gwf-wel"
        @test dfn_name("dis6") == "gwf-dis"
        @test dfn_name("TDIS6") == "sim-tdis"
        @test dfn_name("IMS6") == "sln-ims"
        @test dfn_name("GWF6-GWF6") == "exg-gwfgwf"
        @test dfn_name("TS6") == "utl-ts"
        @test dfn_name("TAS6") == "utl-tas"
        @test dfn_name("OBS6") == "utl-obs"
        @test dfn_name("GWF6") == "gwf-nam"
        @test dfn_name("SIM") == "sim-nam"
        @test dfn_name("IC6"; component = "GWT") == "gwt-ic"
    end

    @testset "validation against the DFN" begin
        mktempdir() do dir
            good = """
            BEGIN options
              READASARRAYS
            END options
            BEGIN period 1
              recharge
                constant 0.001
            END period
            BEGIN period 4
              recharge
                constant 0.002
            END period
            """
            f = read_input_file(write_input(dir, "x.rch", good); ftype = "RCH6")
            @test f.dfn.name == "gwf-rcha"  # READASARRAYS variant, as in MF6
            @test length(find_blocks(f, "PERIOD")) == 2
            f = read_input_file(write_input(dir, "y.rch", "BEGIN options\nEND options\nBEGIN dimensions\n MAXBOUND 1\nEND dimensions\n"); dfn = "gwf-rch")
            @test f.dfn.name == "gwf-rch"
            # IMS is not read through IDM: NONLINEAR and LINEAR are optional.
            f = read_input_file(write_input(dir, "x.ims", "BEGIN options\nEND options\n"); ftype = "IMS6")
            @test f.dfn.name == "sln-ims"

            dis = "BEGIN options\nEND options\nBEGIN dimensions\n NLAY 1\nEND dimensions\n"
            cases = [
                ("unknown", "gwf-wel", "BEGIN options\nEND options\n\nBEGIN griddata\nEND griddata\n", 4, "unknown block GRIDDATA"),
                ("order", "gwf-wel", "BEGIN dimensions\nEND dimensions\nBEGIN options\nEND options\n", 3, "must come before block DIMENSIONS"),
                ("twice", "gwf-wel", "BEGIN options\nEND options\nBEGIN options\nEND options\n", 3, "more than once"),
                ("noper", "gwf-wel", "BEGIN dimensions\nEND dimensions\nBEGIN period\nEND period\n", 3, "missing stress period number"),
                ("badper", "gwf-wel", "BEGIN dimensions\nEND dimensions\nBEGIN period one\nEND period\n", 3, "stress period number"),
                ("zeroper", "gwf-wel", "BEGIN dimensions\nEND dimensions\nBEGIN period 0\nEND period\n", 3, "at least 1"),
                ("decper", "gwf-wel", "BEGIN dimensions\nEND dimensions\nBEGIN period 3\nEND period\nBEGIN period 2\nEND period\n", 5, "not increasing"),
                ("required", "gwf-dis", dis, 0, "required block GRIDDATA"),
                ("tastime", "utl-tas", "BEGIN attributes\nEND attributes\nBEGIN time x\nEND time\n", 3, "double precision"),
            ]
            for (name, dfn, text, line, msg) in cases
                path = write_input(dir, name, text)
                err = input_error_of(() -> read_input_file(path; dfn))
                @test err.file == path
                @test err.line == line
                @test occursin(msg, sprint(showerror, err))
            end
        end
    end
end
