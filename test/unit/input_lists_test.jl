# MF6 list and period input (T1.3). Uses `write_input` from input_tokenizer_test.jl and
# `input_error_of` from input_blocks_test.jl.
using ModflowGPU: MF6InputError, read_input_file, find_blocks, dis_dims, disv_dims, disu_dims,
    grid_dims, write_binary_array, DNODATA, ncelldim, user_node, cellid, valid_cellid,
    ListSchema, list_schema, list_column, package_options, PackageOptions, PeriodList, TsRef,
    nbound, read_list_block, read_list_periods, write_binary_list, StressPeriods,
    period_block, period_data, ArrayPeriod, array_nodes, read_array_periods, read_sto_periods,
    OcRecord, read_oc_periods, read_records, read_tdis_perioddata, read_period_data,
    ExchangeData, read_exchange_data, find_block

"Write package `name` (DFN `dfn`) with `text` into `dir` and read it."
pkg(dir, name, dfn, text) = read_input_file(write_input(dir, name, text); dfn, simdir = dir)

"OPTIONS + DIMENSIONS header of a list package."
header(maxbound; options = "", dims = "") =
    "BEGIN OPTIONS\n$options\nEND OPTIONS\nBEGIN DIMENSIONS\n  MAXBOUND $maxbound\n$dims\nEND DIMENSIONS\n"

"Read the list periods of a package file."
lists(dir, name, dfn, text, g) = read_list_periods(pkg(dir, name, dfn, text), g; simdir = dir)

@testset "List and period input" begin
    G = dis_dims(3, 4, 5)  # nlay 3, nrow 4, ncol 5

    @testset "cellids and user nodes" begin
        @test ncelldim(G) == 3 && ncelldim(disv_dims(2, 7)) == 2 && ncelldim(disu_dims(9)) == 1
        @test user_node(G, (1, 1, 1)) == 1
        @test user_node(G, (2, 3, 4)) == 20 + 10 + 4
        @test all(cellid(G, user_node(G, (k, i, j))) == (k, i, j) for k in 1:3, i in 1:4, j in 1:5)
        V = disv_dims(2, 7)
        @test user_node(V, (2, 3)) == 10 && cellid(V, 10) == (2, 3)
        U = disu_dims(9)
        @test user_node(U, (4,)) == 4 && cellid(U, 4) == (4,)
        @test valid_cellid(G, (3, 4, 5)) && !valid_cellid(G, (0, 1, 1)) && !valid_cellid(G, (1, 5, 1)) && !valid_cellid(G, (1, 1, 6))
        @test !valid_cellid(V, (2, 8)) && !valid_cellid(U, (10,))
    end

    @testset "cellid forms" begin
        mktempdir() do dir
            # DIS: lay row col
            s, sp = lists(dir, "a.chd", "gwf-chd", header(2) * "BEGIN PERIOD 1\n 1 1 1 10.0\n 3,4,5 , 20.5  # comment\nEND PERIOD\n", G)
            p = sp.data[1]
            @test s.columns[1].kind === :cellid && s.columns[1].width == 3
            @test p.nodeu == Int32[1, 60] && p.bound == [10.0 20.5]
            @test cellid(G, p.nodeu[2]) == (3, 4, 5)
            # DISV: lay cell2d
            V = disv_dims(2, 7)
            s, sp = lists(dir, "v.chd", "gwf-chd", header(2) * "BEGIN PERIOD 1\n 1 7 1.0\n 2 1 2.0\nEND PERIOD\n", V)
            @test s.columns[1].width == 2 && sp.data[1].nodeu == Int32[7, 8] && sp.data[1].bound == [1.0 2.0]
            # DISU: node
            U = disu_dims(9)
            s, sp = lists(dir, "u.wel", "gwf-wel", header(1) * "BEGIN PERIOD 1\n 9 -3.5\nEND PERIOD\n", U)
            @test s.columns[1].width == 1 && sp.data[1].nodeu == Int32[9] && sp.data[1].bound == fill(-3.5, 1, 1)
            # The cellid is checked against the grid shape (check_cellid).
            for (g, rec, msg) in (
                    (G, "1 5 1", "(1,5,1)"), (G, "0 1 1", "(0,1,1)"), (G, "1 1 6", "(1,1,6)"),
                    (V, "1 8", "(1,8)"), (U, "10", "(10)"),
                )
                f = pkg(dir, "bad.chd", "gwf-chd", header(2) * "BEGIN PERIOD 1\n $(join(fill("1", ncelldim(g)), " ")) 1.0\n $rec 2.0\nEND PERIOD\n")
                e = input_error_of(() -> read_list_periods(f, g))
                @test e.line == 10 && occursin("List entry 2 contains cellid $msg but this cellid is invalid for model with shape", e.msg)
            end
            # A cellid must be integers.
            f = pkg(dir, "bad.chd", "gwf-chd", header(1) * "BEGIN PERIOD 1\n 1 1.0 1 2.0\nEND PERIOD\n")
            @test occursin("could not read cellid as an integer from \"1.0\"", input_error_of(() -> read_list_periods(f, G)).msg)
        end
    end

    @testset "aux, boundnames and missing values" begin
        mktempdir() do dir
            text = header(4; options = "  AUXILIARY conc mult\n  BOUNDNAMES") * """
                BEGIN PERIOD 1
                  1 1 1 10.0 5.0 0.1 0.2 river_a
                  1 1 2 11.0 6.0 0.3 0.4 'two words'
                  1 1 3 12.0 7.0 0.5 0.6
                  1 1 4 13.0 8.0 0.7 0.8 abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz
                END PERIOD
                """
            s, sp = lists(dir, "a.ghb", "gwf-ghb", text, G)
            @test s.auxnames == ["CONC", "MULT"] && s.boundnames && !s.ts_active && s.nbound_rows == 2
            @test [c.name for c in s.columns] == ["cellid", "bhead", "cond", "aux", "boundname"]
            @test list_column(s, "COND").first == 2 && list_column(s, "aux").width == 2 && isnothing(list_column(s, "x"))
            p = sp.data[1]
            @test nbound(p) == 4
            @test p.bound == [10.0 11.0 12.0 13.0; 5.0 6.0 7.0 8.0]
            @test p.auxvar == [0.1 0.3 0.5 0.7; 0.2 0.4 0.6 0.8]
            @test p.boundname[1:3] == ["RIVER_A", "TWO WORDS", ""]
            @test p.boundname[4] == "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMN"  # LENBOUNDNAME = 40
            @test isempty(p.ts)
            # Without BOUNDNAMES the name is a trailing word and ignored.
            s, sp = lists(dir, "b.ghb", "gwf-ghb", header(1) * "BEGIN PERIOD 1\n 1 1 1 10.0 5.0 name\nEND PERIOD\n", G)
            @test isempty(sp.data[1].boundname) && sp.data[1].bound == reshape([10.0, 5.0], 2, 1)
            # A missing value (an aux column too) is an error.
            f = pkg(dir, "c.ghb", "gwf-ghb", header(1; options = "AUXILIARY c") * "BEGIN PERIOD 1\n 1 1 1 10.0 5.0\nEND PERIOD\n")
            e = input_error_of(() -> read_list_periods(f, G))
            @test e.line == 9 && occursin("missing aux", e.msg)
            f = pkg(dir, "d.riv", "gwf-riv", header(1) * "BEGIN PERIOD 1\n 1 1 1 10.0 5.0\nEND PERIOD\n")
            @test occursin("missing rbot", input_error_of(() -> read_list_periods(f, G)).msg)
            f = pkg(dir, "e.riv", "gwf-riv", header(1) * "BEGIN PERIOD 1\n 1 1\nEND PERIOD\n")
            @test occursin("missing cellid", input_error_of(() -> read_list_periods(f, G)).msg)
            # More records than MAXBOUND.
            f = pkg(dir, "f.wel", "gwf-wel", header(1) * "BEGIN PERIOD 1\n 1 1 1 1.0\n 1 1 2 1.0\nEND PERIOD\n")
            e = input_error_of(() -> read_list_periods(f, G))
            @test e.line == 10 && occursin("line count exceeds input dimension. Expected rows=1", e.msg)
            # MAXBOUND must be positive; AUXILIARY on one line only.
            f = pkg(dir, "g.wel", "gwf-wel", "BEGIN DIMENSIONS\n MAXBOUND 0\nEND DIMENSIONS\nBEGIN PERIOD 1\nEND PERIOD\n")
            @test occursin("MAXBOUND must be an integer greater than zero", input_error_of(() -> read_list_periods(f, G)).msg)
            f = pkg(dir, "h.wel", "gwf-wel", header(1; options = " AUXILIARY a\n AUXILIARY b"))
            @test occursin("auxiliary variables already specified", input_error_of(() -> package_options(f)).msg)
        end
    end

    @testset "EVT segments and the optional last column" begin
        mktempdir() do dir
            s, sp = lists(dir, "a.evt", "gwf-evt", header(1; dims = "NSEG 3") * "BEGIN PERIOD 1\n 1 1 1 50 4e-4 10 0.2 0.5 0.3 0.1\nEND PERIOD\n", G)
            @test [c.name => c.width for c in s.columns] == ["cellid" => 3, "surface" => 1, "rate" => 1, "depth" => 1, "pxdp" => 2, "petm" => 2]
            @test list_column(s, "petm").first == 6
            @test vec(sp.data[1].bound) == [50, 4.0e-4, 10, 0.2, 0.5, 0.3, 0.1]
            # NSEG 1: no pxdp/petm; SURF_RATE_SPECIFIED: petm0 (optional, last: TryGetDouble).
            text = header(3; options = "SURF_RATE_SPECIFIED", dims = "NSEG 1") *
                "BEGIN PERIOD 1\n 1 1 1 50 4e-4 10 0.7\n 1 1 2 50 4e-4 10\n 1 1 3 50 4e-4 10 # no petm0\nEND PERIOD\n"
            s, sp = lists(dir, "b.evt", "gwf-evt", text, G)
            @test [c.name for c in s.columns] == ["cellid", "surface", "rate", "depth", "petm0"] && s.columns[end].optional_last
            @test sp.data[1].bound[4, :] == [0.7, DNODATA, DNODATA]
            # The optional column is not last with BOUNDNAMES: then it is required.
            text = header(1; options = "SURF_RATE_SPECIFIED\nBOUNDNAMES", dims = "NSEG 1") * "BEGIN PERIOD 1\n 1 1 1 50 4e-4 10\nEND PERIOD\n"
            @test occursin("missing petm0", input_error_of(() -> lists(dir, "c.evt", "gwf-evt", text, G)).msg)
            # Vector columns (petm(nseg-1)) are never "optional last": GetDouble or, with
            # TS6, read_token. A series name in the last petm is a reference, a missing
            # value an error (with and without TS6).
            ts = "BEGIN OPTIONS\n TS6 FILEIN e.ts\nEND OPTIONS\nBEGIN DIMENSIONS\n MAXBOUND 1\n NSEG 3\nEND DIMENSIONS\n"
            s, sp = lists(dir, "d.evt", "gwf-evt", ts * "BEGIN PERIOD 1\n 1 1 1 50 4e-4 10 0.2 0.5 0.3 petmseries\nEND PERIOD\n", G)
            @test !any(c -> c.optional_last, s.columns)
            @test sp.data[1].ts == [TsRef(1, 7, false, :PETMSERIES)] && isnan(sp.data[1].bound[7, 1]) && sp.data[1].bound[6, 1] == 0.3
            text = "BEGIN PERIOD 1\n 1 1 1 50 4e-4 10 0.2 0.5 0.3\nEND PERIOD\n"
            @test occursin("missing petm", input_error_of(() -> lists(dir, "e.evt", "gwf-evt", ts * text, G)).msg)
            @test occursin("missing petm", input_error_of(() -> lists(dir, "f.evt", "gwf-evt", header(1; dims = "NSEG 3") * text, G)).msg)
            # petm0 with TS6 active: the time-series branch comes first, so it is required
            # and may be a series name.
            ts0 = "BEGIN OPTIONS\n TS6 FILEIN e.ts\n SURF_RATE_SPECIFIED\nEND OPTIONS\nBEGIN DIMENSIONS\n MAXBOUND 1\n NSEG 1\nEND DIMENSIONS\n"
            s, sp = lists(dir, "g.evt", "gwf-evt", ts0 * "BEGIN PERIOD 1\n 1 1 1 50 4e-4 10 p0series\nEND PERIOD\n", G)
            @test !s.columns[end].optional_last && sp.data[1].ts == [TsRef(1, 4, false, :P0SERIES)]
            @test occursin("missing petm0", input_error_of(() -> lists(dir, "h.evt", "gwf-evt", ts0 * "BEGIN PERIOD 1\n 1 1 1 50 4e-4 10\nEND PERIOD\n", G)).msg)
        end
    end

    @testset "time-series names" begin
        mktempdir() do dir
            ts = "BEGIN OPTIONS\n TS6 FILEIN x.ts\n AUXILIARY m\n BOUNDNAMES\nEND OPTIONS\n"
            text = ts * "BEGIN DIMENSIONS\n MAXBOUND 3\nEND DIMENSIONS\nBEGIN PERIOD 1\n 1 1 1 tides 2.0 well_a\n 1 1 2 -5.0 mult_ts\n 1 1 3 1.0d0 0.5\nEND PERIOD\n"
            s, sp = lists(dir, "a.wel", "gwf-wel", text, G)
            p = sp.data[1]
            @test s.ts_active
            @test p.ts == [TsRef(1, 1, false, :TIDES), TsRef(2, 1, true, :MULT_TS)]
            @test isnan(p.bound[1, 1]) && p.bound[1, 2:3] == [-5.0, 1.0] && isnan(p.auxvar[1, 2]) && p.auxvar[1, [1, 3]] == [2.0, 0.5]
            @test p.boundname == ["WELL_A", "", ""]
            # Without TS6 a name is an error; a cellid is never a time series.
            f = pkg(dir, "b.wel", "gwf-wel", header(1) * "BEGIN PERIOD 1\n 1 1 1 tides\nEND PERIOD\n")
            e = input_error_of(() -> read_list_periods(f, G))
            @test occursin("could not read q as a double precision real from \"tides\"", e.msg) && occursin("TS6 FILEIN", e.msg)
            f = pkg(dir, "c.wel", "gwf-wel", ts * "BEGIN DIMENSIONS\n MAXBOUND 1\nEND DIMENSIONS\nBEGIN PERIOD 1\n lay 1 1 1.0 2.0\nEND PERIOD\n")
            @test occursin("cellid", input_error_of(() -> read_list_periods(f, G)).msg)
            # EVT surface/rate/depth may be time series.
            s, sp = lists(dir, "d.evt", "gwf-evt", "BEGIN OPTIONS\n TS6 FILEIN e.ts\nEND OPTIONS\n" * "BEGIN DIMENSIONS\n MAXBOUND 1\n NSEG 1\nEND DIMENSIONS\nBEGIN PERIOD 1\n 1 1 1 surf rate 10\nEND PERIOD\n", G)
            @test sp.data[1].ts == [TsRef(1, 1, false, :SURF), TsRef(1, 2, false, :RATE)]
        end
    end

    @testset "OPEN/CLOSE lists" begin
        mktempdir() do dir
            # Text: a whole-block include (block parser); lines after it in the package
            # file belong to the block too.
            write_input(dir, "sub/list1.txt", "# list\n 1 1 1 1.0\n 1 1 2 2.0\n")
            s, sp = lists(dir, "a.chd", "gwf-chd", header(5) * "BEGIN PERIOD 1\n OPEN/CLOSE sub/list1.txt\n 1 1 3 3.0\nEND PERIOD\n", G)
            @test sp.data[1].nodeu == Int32[1, 2, 3] && sp.data[1].bound == [1.0 2.0 3.0]
            # Binary: cellid Int32, doubles and aux Float64, MAXBOUND records at most.
            o = PackageOptions(; auxnames = ["A"], maxbound = 3)
            sch = list_schema(ModflowGPU.dfn_definition("gwf-riv"), G, o)
            cid = Int32[1 2 3 3; 1 2 4 4; 1 3 5 1]
            bnd = [1.0 2 3 4; 10 20 30 40; 0.5 1.5 2.5 3.5]
            aux = [7.0 8 9 10]
            write_binary_list(joinpath(dir, "riv.bin"), sch, cid, bnd, aux)
            @test filesize(joinpath(dir, "riv.bin")) == 4 * (3 * 4 + 4 * 8)
            text = header(3; options = "AUXILIARY a") * "BEGIN PERIOD 2\n OPEN/CLOSE riv.bin (binary)\nEND PERIOD\n"
            s, sp = lists(dir, "b.riv", "gwf-riv", text, G)
            p = sp.data[1]
            @test nbound(p) == 3  # MAXBOUND
            @test p.nodeu == Int32[user_node(G, cid[:, i]) for i in 1:3]
            @test p.bound == bnd[:, 1:3] && p.auxvar == aux[:, 1:3]
            # An incomplete last record is dropped (read_from_binary, end of file).
            write(joinpath(dir, "short.bin"), read(joinpath(dir, "riv.bin"))[1:(44 + 20)])
            s, sp = lists(dir, "c.riv", "gwf-riv", header(4; options = "AUXILIARY a") * "BEGIN PERIOD 1\n OPEN/CLOSE short.bin (BINARY)\nEND PERIOD\n", G)
            @test nbound(sp.data[1]) == 1
            # Not with BOUNDNAMES; only line of the block; a cellid outside the grid.
            f = pkg(dir, "d.riv", "gwf-riv", header(3; options = "AUXILIARY a\nBOUNDNAMES") * "BEGIN PERIOD 1\n OPEN/CLOSE riv.bin (BINARY)\nEND PERIOD\n")
            @test occursin("BOUNDNAME", input_error_of(() -> read_list_periods(f, G; simdir = dir)).msg)
            f = pkg(dir, "e.riv", "gwf-riv", header(3; options = "AUXILIARY a") * "BEGIN PERIOD 1\n OPEN/CLOSE riv.bin (BINARY)\n 1 1 1 1 1 1 1\nEND PERIOD\n")
            @test occursin("END PERIOD", input_error_of(() -> read_list_periods(f, G; simdir = dir)).msg)
            f = pkg(dir, "f.riv", "gwf-riv", header(3; options = "AUXILIARY a") * "BEGIN PERIOD 1\n OPEN/CLOSE riv.bin (BINARY)\nEND PERIOD\n")
            @test occursin("List entry 2 contains cellid (2,2,3)", input_error_of(() -> read_list_periods(f, dis_dims(3, 1, 5); simdir = dir)).msg)
            f = pkg(dir, "g.riv", "gwf-riv", header(3) * "BEGIN PERIOD 1\n OPEN/CLOSE missing.bin (BINARY)\nEND PERIOD\n")
            @test occursin("does not exist", input_error_of(() -> read_list_periods(f, G; simdir = dir)).msg)
            # SFAC is not supported by MF6's list reader.
            write_input(dir, "l.txt", "1 1 1 1.0\n")
            f = pkg(dir, "h.wel", "gwf-wel", header(3) * "BEGIN PERIOD 1\n OPEN/CLOSE l.txt SFAC 2.0\nEND PERIOD\n")
            @test occursin("SFAC", input_error_of(() -> read_list_periods(f, G; simdir = dir)).msg)
        end
    end

    @testset "period persistence" begin
        mktempdir() do dir
            text = header(2) * """
                BEGIN PERIOD 2
                  1 1 1 -1.0
                  1 1 2 -2.0
                END PERIOD
                BEGIN PERIOD 4
                END PERIOD
                BEGIN PERIOD 6
                  2 1 1 -6.0
                END PERIOD
                """
            s, sp = lists(dir, "a.wel", "gwf-wel", text, G)
            @test sp.iper == [2, 4, 6]
            @test period_block(sp, 1) == 0 && isnothing(period_data(sp, 1))
            @test period_data(sp, 2) === period_data(sp, 3) === sp.data[1]
            @test period_data(sp, 3).bound == [-1.0 -2.0]
            @test nbound(period_data(sp, 4)) == 0 && nbound(period_data(sp, 5)) == 0  # an empty block clears
            @test period_data(sp, 6).bound == fill(-6.0, 1, 1) && period_data(sp, 100) === sp.data[3]
            # Period numbers must increase (checked by the block reader).
            f = write_input(dir, "b.wel", header(1) * "BEGIN PERIOD 3\nEND PERIOD\nBEGIN PERIOD 3\nEND PERIOD\n")
            @test occursin("not increasing", input_error_of(() -> read_input_file(f; dfn = "gwf-wel")).msg)
        end
    end

    @testset "READASARRAYS" begin
        mktempdir() do dir
            G2 = dis_dims(2, 2, 3)  # ncpl 6
            text = """
            BEGIN OPTIONS
              READASARRAYS
              AUXILIARY mult
            END OPTIONS
            BEGIN PERIOD 1
              RECHARGE LAYERED
                INTERNAL FACTOR 1e-3
                1 2 3
                4 5 6
            END PERIOD
            BEGIN PERIOD 3
              irch
                INTERNAL
                1 2 1 2 2 1
              mult
                CONSTANT 0.5
            END PERIOD
            BEGIN PERIOD 4
              recharge
                CONSTANT 0.0
            END PERIOD
            """
            f = pkg(dir, "a.rch", "gwf-rch", text)
            @test f.dfn.name == "gwf-rcha"  # variant from OPTIONS
            sp = read_array_periods(f, G2)
            @test sp.iper == [1, 3, 4]
            a1, a3, a4 = sp.data
            @test a1.values["recharge"] ≈ (1:6) .* 1.0e-3 && a1.layer == ones(Int32, 6) && a1.auxvar == zeros(1, 6)
            @test a1.given == ["recharge"] && isempty(a1.tas)
            @test array_nodes(a1, G2) == Int32.(1:6)
            # Arrays not given keep their values; IRCH applies from its block.
            @test a3.values["recharge"] === a1.values["recharge"] && a3.layer == Int32[1, 2, 1, 2, 2, 1]
            @test a3.auxvar == fill(0.5, 1, 6) && a1.auxvar == zeros(1, 6) && a3.given == ["irch", "MULT"]
            @test array_nodes(a3, G2) == Int32[1, 8, 3, 10, 11, 6]
            @test a4.values["recharge"] == zeros(6) && a4.layer === a3.layer && a4.auxvar === a3.auxvar
            @test period_data(sp, 2) === a1 && period_data(sp, 10) === a4
            @test read_period_data(f, G2) isa StressPeriods{ArrayPeriod}
            # DISV: NCPL values.
            sp = read_array_periods(pkg(dir, "v.rch", "gwf-rch", "BEGIN OPTIONS\n READASARRAYS\nEND OPTIONS\nBEGIN PERIOD 1\n RECHARGE\n  INTERNAL\n  1 2 3 4\nEND PERIOD\n"), disv_dims(3, 4))
            @test sp.data[1].values["recharge"] == [1.0, 2, 3, 4]
            # EVTA arrays.
            text = "BEGIN OPTIONS\n READASARRAYS\nEND OPTIONS\nBEGIN PERIOD 1\n SURFACE\n  CONSTANT 10\n RATE\n  CONSTANT 1e-3\n DEPTH\n  CONSTANT 2\n IEVT\n  CONSTANT 2\nEND PERIOD\n"
            f = pkg(dir, "a.evt", "gwf-evt", text)
            @test f.dfn.name == "gwf-evta"
            e = read_array_periods(f, G2).data[1]
            @test e.values["surface"] == fill(10.0, 6) && e.values["rate"] == fill(1.0e-3, 6) && e.values["depth"] == fill(2.0, 6) && e.layer == fill(Int32(2), 6)
            # TIMEARRAYSERIES: a reference, NaN values (needs TAS6).
            text = "BEGIN OPTIONS\n READASARRAYS\n TAS6 FILEIN r.tas\n AUXILIARY m\nEND OPTIONS\nBEGIN PERIOD 1\n RECHARGE TIMEARRAYSERIES rchts\n m TIMEARRAYSERIES mts\nEND PERIOD\n"
            a = read_array_periods(pkg(dir, "t.rch", "gwf-rch", text), G2).data[1]
            @test a.tas == Dict("recharge" => :RCHTS, "M" => :MTS) && all(isnan, a.values["recharge"]) && all(isnan, a.auxvar)
            text = "BEGIN OPTIONS\n READASARRAYS\nEND OPTIONS\nBEGIN PERIOD 1\n RECHARGE TIMEARRAYSERIES rchts\nEND PERIOD\n"
            @test occursin("TAS6", input_error_of(() -> read_array_periods(pkg(dir, "u.rch", "gwf-rch", text), G2)).msg)
            # Errors: layer outside the grid, unknown name, missing control record, DISU.
            text = "BEGIN OPTIONS\n READASARRAYS\nEND OPTIONS\nBEGIN PERIOD 1\n IRCH\n  CONSTANT 3\nEND PERIOD\n"
            @test occursin("Invalid layer number: 3", input_error_of(() -> read_array_periods(pkg(dir, "w.rch", "gwf-rch", text), G2)).msg)
            text = "BEGIN OPTIONS\n READASARRAYS\nEND OPTIONS\nBEGIN PERIOD 1\n RATE\n  CONSTANT 3\nEND PERIOD\n"
            @test occursin("\"RATE\" is not a PERIOD array", input_error_of(() -> read_array_periods(pkg(dir, "x.rch", "gwf-rch", text), G2)).msg)
            text = "BEGIN OPTIONS\n READASARRAYS\nEND OPTIONS\nBEGIN PERIOD 1\n RECHARGE\nEND PERIOD\n"
            @test occursin("missing control record", input_error_of(() -> read_array_periods(pkg(dir, "y.rch", "gwf-rch", text), G2)).msg)
            text = "BEGIN OPTIONS\n READASARRAYS\nEND OPTIONS\nBEGIN PERIOD 1\n RECHARGE\n  INTERNAL\n  1 2 3\nEND PERIOD\n"
            @test input_error_of(() -> read_array_periods(pkg(dir, "z.rch", "gwf-rch", text), G2)) isa MF6InputError
            @test occursin("not compatible", input_error_of(() -> read_array_periods(pkg(dir, "z.rch", "gwf-rch", text), disu_dims(6))).msg)
        end
    end

    @testset "STO, OC and TDIS" begin
        mktempdir() do dir
            gd = "BEGIN GRIDDATA\n ICONVERT\n  CONSTANT 0\nEND GRIDDATA\n"
            f = pkg(dir, "a.sto", "gwf-sto", gd * "BEGIN PERIOD 1\n STEADY-STATE\nEND PERIOD\nBEGIN PERIOD 2\nEND PERIOD\nBEGIN PERIOD 3\n transient\nEND PERIOD\n")
            sp = read_sto_periods(f)
            @test sp.iper == [1, 3] && sp.data == [:steady_state, :transient] && period_data(sp, 2) === :steady_state
            f = pkg(dir, "b.sto", "gwf-sto", gd * "BEGIN PERIOD 1\n STEADY-STATE\n TRANSIENT\nEND PERIOD\n")
            @test occursin("END PERIOD", input_error_of(() -> read_sto_periods(f)).msg)
            f = pkg(dir, "c.sto", "gwf-sto", gd * "BEGIN PERIOD 1\n STEADY\nEND PERIOD\n")
            @test occursin("STEADY-STATE or TRANSIENT", input_error_of(() -> read_sto_periods(f)).msg)

            text = "BEGIN PERIOD 1\n SAVE HEAD ALL\n print budget last\n SAVE BUDGET FREQUENCY 5\n PRINT HEAD STEPS 1 3 x\n SAVE HEAD FIRST\nEND PERIOD\nBEGIN PERIOD 3\nEND PERIOD\n"
            sp = read_oc_periods(pkg(dir, "a.oc", "gwf-oc", text))
            @test sp.iper == [1, 3]
            @test sp.data[1] == [
                OcRecord(:save, "HEAD", :all, Int[]), OcRecord(:print, "BUDGET", :last, Int[]),
                OcRecord(:save, "BUDGET", :frequency, [5]), OcRecord(:print, "HEAD", :steps, [1, 3]),
                OcRecord(:save, "HEAD", :first, Int[]),
            ]
            @test isempty(period_data(sp, 4))  # an empty block turns output off
            @test occursin("PRINT or SAVE", input_error_of(() -> read_oc_periods(pkg(dir, "b.oc", "gwf-oc", "BEGIN PERIOD 1\n WRITE HEAD ALL\nEND PERIOD\n"))).msg)
            @test occursin("ALL, STEPS", input_error_of(() -> read_oc_periods(pkg(dir, "c.oc", "gwf-oc", "BEGIN PERIOD 1\n SAVE HEAD\nEND PERIOD\n"))).msg)

            tdis = "BEGIN DIMENSIONS\n NPER 2\nEND DIMENSIONS\nBEGIN PERIODDATA\n 1.0 1 1.0\n 10.0d0 5 1.2 # comment\nEND PERIODDATA\n"
            t = read_tdis_perioddata(pkg(dir, "a.tdis", "sim-tdis", tdis))
            @test t.perlen == [1.0, 10.0] && t.nstp == [1, 5] && t.tsmult == [1.0, 1.2]
            bad = replace(tdis, "NPER 2" => "NPER 3")
            @test occursin("expected 3", input_error_of(() -> read_tdis_perioddata(pkg(dir, "b.tdis", "sim-tdis", bad))).msg)
            bad = replace(tdis, "NPER 2" => "NPER 1")
            @test occursin("Expected rows=1", input_error_of(() -> read_tdis_perioddata(pkg(dir, "c.tdis", "sim-tdis", bad))).msg)
            bad = replace(tdis, " 1.0 1 1.0" => " 1.0 1.5 1.0")
            @test occursin("nstp", input_error_of(() -> read_tdis_perioddata(pkg(dir, "d.tdis", "sim-tdis", bad))).msg)
            # Packages without supported period data.
            f = pkg(dir, "a.chdg", "gwf-chd", "BEGIN OPTIONS\n READARRAYGRID\nEND OPTIONS\nBEGIN PERIOD 1\nEND PERIOD\n")
            @test occursin("unsupported", input_error_of(() -> read_period_data(f, G)).msg)
        end
    end
end

@testset "GWF-GWF exchange data" begin
    mktempdir() do dir
        g1 = dis_dims(1, 3, 4)
        g2 = disv_dims(2, 5)
        text = """
        BEGIN OPTIONS
          AUXILIARY angldegx cdist
          BOUNDNAMES
        END OPTIONS
        BEGIN DIMENSIONS
          NEXG 3
        END DIMENSIONS
        BEGIN EXCHANGEDATA
          1 2 3  2 5  1  10.0 5.0 2.5  90.0 15.0 west
          1 3 4  1 1  0  1d0 2.0 3.0  0.0 3.0
          1 1 1  2 1  1  1.0 2.0 3.0  180 4.0 'b c'  # comment
        END EXCHANGEDATA
        """
        f = pkg(dir, "a.gwfgwf", "exg-gwfgwf", text)
        e = read_exchange_data(f, g1, g2)
        @test e.nodem1 == Int32[user_node(g1, (1, 2, 3)), 12, 1] && e.nodem2 == Int32[10, 1, 6]
        @test e.ihc == Int32[1, 0, 1] && e.cl1 == [10.0, 1.0, 1.0] && e.cl2 == [5.0, 2.0, 2.0] && e.hwva == [2.5, 3.0, 3.0]
        @test e.auxnames == ["ANGLDEGX", "CDIST"] && e.auxvar == [90.0 0.0 180.0; 15.0 3.0 4.0]
        @test e.boundname == ["WEST", "", "B C"]
        # Without aux and boundnames, trailing words are ignored.
        plain = "BEGIN DIMENSIONS\n NEXG 1\nEND DIMENSIONS\nBEGIN EXCHANGEDATA\n 1 1 1  1 1  1  1.0 2.0 3.0 extra\nEND EXCHANGEDATA\n"
        e = read_exchange_data(pkg(dir, "b.gwfgwf", "exg-gwfgwf", plain), g1, g2)
        @test size(e.auxvar) == (0, 1) && isempty(e.boundname) && e.hwva == [3.0]
        # Errors: record count, cellid outside its model, non-integer IHC, missing value.
        @test occursin("expected NEXG = 2", input_error_of(() -> read_exchange_data(pkg(dir, "c.gwfgwf", "exg-gwfgwf", replace(plain, "NEXG 1" => "NEXG 2")), g1, g2)).msg)
        two = replace(plain, " 1 1 1  1 1  1  1.0 2.0 3.0 extra\n" => " 1 1 1  1 1  1  1.0 2.0 3.0\n 1 1 2  1 2  1  1.0 2.0 3.0\n")
        @test occursin("Expected rows=1", input_error_of(() -> read_exchange_data(pkg(dir, "d.gwfgwf", "exg-gwfgwf", two), g1, g2)).msg)
        bad = replace(plain, "1 1 1  1 1  1" => "1 1 1  3 1  1")
        err = input_error_of(() -> read_exchange_data(pkg(dir, "e.gwfgwf", "exg-gwfgwf", bad), g1, g2))
        @test occursin("List entry 1 contains cellidm2 (3,1) but this cellid is invalid for model with shape (2,5)", err.msg)
        bad = replace(plain, "1 1 1  1 1  1" => "1 4 1  1 1  1")
        @test occursin("cellidm1 (1,4,1)", input_error_of(() -> read_exchange_data(pkg(dir, "f.gwfgwf", "exg-gwfgwf", bad), g1, g2)).msg)
        bad = replace(plain, "1  1.0 2.0 3.0" => "1.5  1.0 2.0 3.0")
        @test occursin("could not read ihc as an integer", input_error_of(() -> read_exchange_data(pkg(dir, "g.gwfgwf", "exg-gwfgwf", bad), g1, g2)).msg)
        bad = replace(plain, "1.0 2.0 3.0 extra" => "1.0 2.0")
        @test occursin("missing hwva", input_error_of(() -> read_exchange_data(pkg(dir, "h.gwfgwf", "exg-gwfgwf", bad), g1, g2)).msg)
        # The static record reader refuses cellid columns.
        f = pkg(dir, "b.gwfgwf", "exg-gwfgwf", plain)
        @test_throws "unsupported: cellid columns" read_records(find_block(f, "EXCHANGEDATA"), f.dfn)
    end
end

# Done-when test of T1.3: hand-checked values of the period data of advtidal, twri01 and
# fhb (read from the files in reference/mf6examples), and every PERIOD block of every
# in-scope example's GWF packages reads without error.
@testset "MF6 example period data" begin
    exdir = joinpath(REFERENCE_DIR, "mf6examples")
    if !isdir(exdir)
        @info "Skipping the MF6 example period-data tests: $exdir not found (validation/fetch_reference.sh)"
    else
        rd(ex, file, ftype) = read_input_file(joinpath(exdir, ex, file); ftype, simdir = joinpath(exdir, ex))
        dims(ex, file) = grid_dims(rd(ex, file, "DIS6"))

        @testset "advtidal" begin
            ex = "ex-gwf-advtidal"
            g = dims(ex, "$ex.dis")
            @test g.mshape == [3, 15, 10]
            s, sp = read_period_data(rd(ex, "$ex.ghb", "GHB6"), g)
            p = period_data(sp, 4)  # block 1 applies to all 4 periods
            @test sp.iper == [1] && p === sp.data[1] && nbound(p) == 30
            @test cellid(g, p.nodeu[1]) == (2, 1, 10) && p.nodeu[1] == 160 && p.bound[2, 1] == 15.0 && p.boundname[1] == "ESTUARY-L2"
            @test cellid(g, p.nodeu[16]) == (3, 1, 10) && p.bound[2, 16] == 1500.0 && p.boundname[16] == "ESTUARY-L3"
            @test length(p.ts) == 30 && all(t -> t.name === :TIDES && t.col == 1 && !t.aux, p.ts) && all(isnan, p.bound[1, :])

            s, sp = read_period_data(rd(ex, "$ex.wel", "WEL6"), g)
            @test sp.iper == [2, 3, 4] && isnothing(period_data(sp, 1))
            p = period_data(sp, 2)
            @test nbound(p) == 3 && p.nodeu[1:2] == Int32[user_node(g, (1, 12, 3)), user_node(g, (3, 5, 8))]
            @test p.bound[1, 1] == -50.0 && p.boundname == ["", "WELL_1", "WELL_2"]
            @test p.ts == [TsRef(2, 1, false, :WELL_1_RATE), TsRef(3, 1, false, :WELL_2_RATE)]
            p = period_data(sp, 3)
            @test nbound(p) == 2 && p.boundname == ["WELL_2", "WELL_1"]
            p = period_data(sp, 4)
            @test nbound(p) == 5 && cellid(g, p.nodeu[5]) == (1, 14, 6) && p.bound[1, 3:5] == [-10.0, -20.0, -40.0]

            s, sp = read_period_data(rd(ex, "$ex.riv", "RIV6"), g)
            p = sp.data[1]
            @test nbound(p) == 20 && cellid(g, p.nodeu[6]) == (1, 6, 6) && p.bound[2:3, 6] == [1006.0, 35.4]
            @test p.boundname[6] == "RIV1_C6" && p.boundname[1] == "" && p.ts[6] == TsRef(6, 1, false, :RIVER_STAGE_1)
            @test p.ts[11].name === :RIVER_STAGE_2 && p.boundname[11] == "RIV2_UPPER"

            s, sp = read_period_data(rd(ex, "$ex.rch1", "RCH6"), g)
            p = sp.data[1]
            @test s.auxnames == ["MULTIPLIER"] && nbound(p) == 84 && cellid(g, p.nodeu[3]) == (1, 1, 3)
            @test p.auxvar[1, 3] == 0.5 && p.ts[3] == TsRef(3, 1, false, :RCH_1)
            s, sp = read_period_data(rd(ex, "$ex.rch3", "RCH6"), g)
            @test sp.data[1].auxvar[1, 1:3] == [0.5, 0.5, 1.0] && sp.data[1].ts[1].name === :RCH_3

            s, sp = read_period_data(rd(ex, "$ex.evt", "EVT6"), g)
            p = sp.data[1]
            @test nbound(p) == 150 && cellid(g, p.nodeu[150]) == (1, 15, 10)
            @test p.bound[:, 150] == [50.0, 4.0e-4, 10.0, 0.2, 0.5, 0.3, 0.1]

            sp = read_period_data(rd(ex, "$ex.sto", "STO6"), g)
            @test [period_data(sp, k) for k in 1:4] == [:steady_state, :transient, :transient, :transient]
            sp = read_period_data(rd(ex, "$ex.oc", "OC6"), g)
            @test period_data(sp, 4) == [OcRecord(:save, "HEAD", :all, Int[]), OcRecord(:save, "BUDGET", :all, Int[])]
            t = read_tdis_perioddata(rd(ex, "$ex.tdis", "TDIS6"))
            @test t.perlen == [1.0, 10.0, 10.0, 10.0] && t.nstp == [1, 120, 120, 120] && t.tsmult == ones(4)
        end

        @testset "twri01" begin
            ex = "ex-gwf-twri01"
            g = dims(ex, "$ex.dis")
            @test g.mshape == [5, 15, 15]
            sp = read_period_data(rd(ex, "$ex.rcha", "RCH6"), g)
            a = period_data(sp, 1)
            @test a.values["recharge"] == fill(3.0e-8, 225) && a.layer == ones(Int32, 225) && array_nodes(a, g) == Int32.(1:225)
            s, sp = read_period_data(rd(ex, "$ex.chd", "CHD6"), g)
            p = sp.data[1]
            @test nbound(p) == 30 && cellid(g, p.nodeu[16]) == (3, 1, 1) && p.nodeu[16] == 451 && all(==(0.0), p.bound)
            s, sp = read_period_data(rd(ex, "$ex.drn", "DRN6"), g)
            p = sp.data[1]
            @test nbound(p) == 9 && p.nodeu[4] == user_node(g, (1, 8, 5)) == 110 && p.bound[:, 4] == [20.0, 1.0] && p.bound[:, 9] == [100.0, 1.0]
            s, sp = read_period_data(rd(ex, "$ex.wel", "WEL6"), g)
            p = sp.data[1]
            @test nbound(p) == 15 && p.nodeu[1:2] == Int32[971, 501] && all(==(-5.0), p.bound)
            @test read_tdis_perioddata(rd(ex, "$ex.tdis", "TDIS6")).perlen == [86400.0]
        end

        @testset "fhb" begin
            ex = "ex-gwf-fhb"
            g = dims(ex, "$ex.dis")
            @test g.mshape == [1, 3, 10]
            s, sp = read_period_data(rd(ex, "$ex.chd", "CHD6"), g)
            p = period_data(sp, 3)
            @test s.ts_active && p.nodeu == Int32[10, 20, 30] && p.ts == [TsRef(i, 1, false, :CHDHEAD) for i in 1:3]
            s, sp = read_period_data(rd(ex, "$ex.wel", "WEL6"), g)
            @test sp.data[1].nodeu == Int32[11] && sp.data[1].ts == [TsRef(1, 1, false, :FLOWRATE)]
            sp = read_period_data(rd(ex, "$ex.sto", "STO6"), g)
            @test sp.iper == [1] && period_data(sp, 3) === :transient
            t = read_tdis_perioddata(rd(ex, "$ex.tdis", "TDIS6"))
            @test t.perlen == [400.0, 200.0, 400.0] && t.nstp == [10, 4, 6]
        end

        @testset "all in-scope examples" begin
            npkg = 0
            nblock = 0
            for (name, simdir) in in_scope_examples(; tiers = 0:7)
                @testset "$name" begin
                    for (model, files) in model_package_files(simdir)
                        g = grid_dims(only(filter(f -> f.dfn.subcomponent in ("DIS", "DISV", "DISU"), files)))
                        for f in files
                            blocks = find_blocks(f, "PERIOD")
                            isempty(blocks) && continue
                            r = read_period_data(f, g; simdir)
                            sp = r isa Tuple ? r[2] : r
                            @test sp isa StressPeriods
                            npkg += 1
                            nblock += length(blocks)
                        end
                    end
                end
            end
            @test npkg > 100
            @info "MF6 examples: period data of $npkg packages, $nblock PERIOD blocks"
        end

        @testset "GWF-GWF exchanges" begin
            # Every GWF6-GWF6 exchange of the in-scope examples (Tier 7), read with the
            # grids of its two models (mfsim.nam EXCHANGES: type, file, model 1, model 2).
            exchanges = Dict{String, ExchangeData}()
            for (name, simdir) in in_scope_examples(; tiers = 0:7)
                sim = read_input_file(joinpath(simdir, "mfsim.nam"); ftype = "SIM", component = "SIM", simdir)
                blk = find_block(sim, "EXCHANGES")
                isnothing(blk) && continue
                grids = Dict(
                    lowercase(m) => grid_dims(only(filter(f -> f.dfn.subcomponent in ("DIS", "DISV", "DISU"), files)))
                        for (m, files) in model_package_files(simdir)
                )
                for l in blk.lines
                    keyword(l, 1) == "GWF6-GWF6" || continue
                    f = read_input_file(joinpath(simdir, l[2]); ftype = "GWF6-GWF6", component = "EXG", simdir)
                    e = read_exchange_data(f, grids[lowercase(l[3])], grids[lowercase(l[4])])
                    @test length(e.nodem1) == length(e.nodem2) == length(e.ihc) > 0
                    exchanges[name] = e
                end
            end
            @test issubset(
                ["ex-gwf-u1gwfgwf-s1", "ex-gwf-u1gwfgwf-s2", "ex-gwf-u1gwfgwf-s3", "ex-gwf-u1gwfgwf-s4", "ex-gwf-lgrv-lgr", "ex-gwf-spbc"],
                keys(exchanges),
            )
            # Hand-checked rows. u1gwfgwf-s1: outer 1×7×7 → inner 1×9×9, aux and boundnames.
            e = exchanges["ex-gwf-u1gwfgwf-s1"]
            @test length(e.ihc) == 36 && e.auxnames == ["ANGLDEGX", "CDIST"]
            @test (e.nodem1[1], e.nodem2[1], e.ihc[1]) == (16, 1, 1) && (e.cl1[1], e.cl2[1], e.hwva[1]) == (50.0, 16.66666667, 33.33333333)
            @test e.auxvar[:, 1] == [0.0, 74.53559925] && e.boundname[1] == "LEFT"
            @test (e.nodem1[10], e.nodem2[10]) == (38, 73) && e.auxvar[1, 10] == 90.0 && e.boundname[10] == "BOTTOM"
            # spbc: a model (190×1×100) exchanging with itself: column 1 to column 100.
            e = exchanges["ex-gwf-spbc"]
            @test length(e.ihc) == 190 && (e.nodem1[10], e.nodem2[10]) == (901, 1000)
            @test (e.cl1[10], e.cl2[10], e.hwva[10]) == (0.03, 0.03, 1.0) && e.auxvar[:, 10] == [90.0, 0.06] && isempty(e.boundname)
            # lgrv-lgr: parent 9×61×49 → child 25×90×78.
            e = exchanges["ex-gwf-lgrv-lgr"]
            @test length(e.ihc) == 8400 && (e.nodem1[2], e.nodem2[2]) == (18 * 49 + 16, 1)
            @test (e.cl1[2], e.cl2[2], e.hwva[2]) == (37.5, 12.5, 35.0) && e.auxvar[:, 2] == [270.0, 61.03277808]
        end
    end
end
