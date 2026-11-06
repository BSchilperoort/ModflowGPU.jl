# MF6 array input, READARRAY (T1.2). Uses `write_input` from input_tokenizer_test.jl and
# `input_error_of` from input_blocks_test.jl.
using ModflowGPU: MF6InputError, InputLine, read_input_file, find_block, dfn_definition,
    dfn_block, dfn_variable, GridDims, dis_dims, disv_dims, disu_dims, grid_dims, array_size,
    layered_shape, read_control_record, ArrayControl, list_read!, read_array!, read_grid_array,
    read_array_block, read_binary_array, write_binary_array, BinaryArrayHeader,
    BINARY_HEADER_BYTES

"All lines of `text` as records (blank lines kept, as in an OPEN/CLOSE file)."
array_lines(text; file = "test.txt") = [InputLine(s, file, i) for (i, s) in enumerate(split(text, '\n'))]

"Write a package file with the given GRIDDATA body and read its arrays."
function griddata(dir, dfn, g, body; name = "x." * split(dfn, '-')[2], extra = "", after = "")
    path = write_input(dir, name, extra * "BEGIN GRIDDATA\n" * body * "END GRIDDATA\n" * after)
    f = read_input_file(path; dfn, simdir = dir)
    return read_array_block(find_block(f, "GRIDDATA"), f.dfn, g; simdir = dir)
end

const DIS_2x3x4 = """
BEGIN DIMENSIONS
  NLAY 2
  NROW 3
  NCOL 4
END DIMENSIONS
"""

@testset "Array input" begin
    @testset "control records" begin
        l(s) = InputLine(s, "f", 1)
        c = read_control_record(Float64, l("constant 2.5d0"))
        @test c.form === :constant && c.constant == 2.5
        c = read_control_record(Int32, l("CONSTANT -3"))
        @test c.form === :constant && c.constant === Int32(-3)
        c = read_control_record(Float64, l("INTERNAL"))
        @test c.form === :internal && c.factor == 1.0 && c.iprn == 0 && !c.binary
        c = read_control_record(Float64, l("internal iprn 3 factor 1.0e-3  # comment FACTORS"))
        @test c.form === :internal && c.factor == 1.0e-3 && c.iprn == 3
        c = read_control_record(Float64, l("OPEN/CLOSE 'Data Dir/K.bin' FACTOR 2 (binary) IPRN -1"))
        @test c.form === :open_close && c.file == "Data Dir/K.bin" && c.factor == 2.0 && c.binary && c.iprn == -1
        c = read_control_record(Int32, l("open/close ib.txt factor 2"))
        @test c.form === :open_close && c.file == "ib.txt" && c.factor === Int32(2) && !c.binary
        # Errors: unknown form, missing or bad values, real FACTOR for an integer array.
        @test_throws MF6InputError read_control_record(Float64, l("CONSTNT 1.0"))
        @test_throws MF6InputError read_control_record(Float64, l("CONSTANT"))
        @test_throws MF6InputError read_control_record(Float64, l("CONSTANT abc"))
        @test_throws MF6InputError read_control_record(Int32, l("CONSTANT 1.0"))
        @test_throws MF6InputError read_control_record(Int32, l("CONSTANT 3000000000"))
        @test_throws MF6InputError read_control_record(Float64, l("INTERNAL FACTOR"))
        @test_throws MF6InputError read_control_record(Int32, l("INTERNAL FACTOR 1.5"))
        @test_throws MF6InputError read_control_record(Float64, l("INTERNAL IPRN x"))
        @test_throws MF6InputError read_control_record(Float64, l("OPEN/CLOSE"))
        e = input_error_of(() -> read_control_record(Float64, l("BOGUS 1"); name = "K"))
        @test occursin("CONSTANT, INTERNAL, or OPEN/CLOSE", e.msg) && occursin("K", e.msg)
    end

    @testset "list-directed values" begin
        rd(T, text, n) = (out = zeros(T, n); k = list_read!(out, 1, n, array_lines(text), 1, InputLine("", "f", 0)); (out, k))
        @test rd(Float64, "1 2 3\n4 5 6 7 8\nNEXT", 6) == ([1, 2, 3, 4, 5, 6], 3)  # rest of line 2 skipped
        @test rd(Float64, "3*1.0 2*2d0 5.5", 6)[1] == [1, 1, 1, 2, 2, 5.5]
        @test rd(Float64, "1.5+3,2.0D-1 , -3.\t.5", 4)[1] == [1500, 0.2, -3, 0.5]
        @test rd(Float64, "\n\n1 2\n\n3", 3) == ([1, 2, 3], 6)  # blank lines skipped
        @test rd(Float64, "1 2 3 # comment", 3)[1] == [1, 2, 3]  # words after the values are not read
        @test rd(Float64, "10*7", 4)[1] == fill(7.0, 4)  # repeat count longer than needed
        @test rd(Int32, "+1 -2 3*7", 5)[1] == Int32[1, -2, 7, 7, 7]
        @test eltype(rd(Int32, "1", 1)[1]) == Int32
        bad(T, text, n) = input_error_of(() -> rd(T, text, n))
        @test occursin("after 3 of 4 values", bad(Float64, "1 2 3", 4).msg)  # too few
        @test occursin("could not read \"#\"", bad(Float64, "1 2 # 3", 3).msg)
        @test occursin("null value", bad(Float64, "1,,3", 3).msg)
        @test occursin("null value", bad(Float64, ",2,3", 3).msg)
        @test occursin("null value", bad(Float64, "1 2*", 3).msg)
        @test occursin("'/'", bad(Float64, "1 2 / 9", 3).msg)
        @test occursin("repeat count", bad(Float64, "0*1 2", 2).msg)
        @test occursin("an integer", bad(Int32, "1 2 5.", 3).msg)
        @test occursin("an integer", bad(Int32, "1 2 1e2", 3).msg)
        @test occursin("an integer", bad(Int32, "1 99999999999", 2).msg)
        @test_throws MF6InputError rd(Float64, "1 2 0x10", 3)
        e = bad(Float64, "1 2\n3 x", 4)
        @test e.line == 2 && occursin("value 4 of 4", e.msg)
        # A `/` or comma after the last needed value is in the skipped rest of the line.
        @test rd(Float64, "1 2 3 / ,,", 3)[1] == [1, 2, 3]
    end

    @testset "shapes" begin
        g = dis_dims(2, 3, 4)
        @test g.mshape == [2, 3, 4] && g.dims["NCPL"] == 12 && g.dims["NODES"] == 24
        @test array_size("(nodes)", g) == [24]
        @test array_size("(ncol, nrow, nlay)", g) == [4, 3, 2]
        @test array_size("(ncol, nrow)", g) == [4, 3]
        @test array_size("(ncol*nrow; ncpl)", g) == [12]
        @test array_size("(ncol*nrow)", g) == [12]
        @test layered_shape(g) == (2, (4, 3))
        gv = disv_dims(3, 5)
        @test array_size("(ncpl, nlay)", gv) == [5, 3] && array_size("(nodes)", gv) == [15]
        @test layered_shape(gv) == (3, (5,))
        gu = disu_dims(7; nja = 20)
        @test array_size("(nja)", gu) == [20] && array_size("(nodes)", gu) == [7]
        @test layered_shape(gu) == (1, (7,))
        @test_throws ArgumentError array_size("(nrow)", gv)
    end

    @testset "INTERNAL and CONSTANT, FACTOR on double and integer arrays" begin
        mktempdir() do dir
            g = dis_dims(2, 3, 4)
            a = griddata(
                dir, "gwf-npf", g, """
                  icelltype
                    INTERNAL FACTOR 2
                      24*1
                  k
                    internal factor 0.5 iprn 1
                      1  2  3  4  5  6  7  8  9 10 11 12
                     13 14 15 16 17 18 19 20 21 22 23 24 99 99
                  k33
                    CONSTANT 4.5
                  k22
                    INTERNAL FACTOR 0.0  # factor 0 is ignored
                      24*3.0
                """,
            )
            @test a["icelltype"] == fill(Int32(2), 24) && eltype(a["icelltype"]) == Int32
            @test a["k"] == 0.5 .* (1:24)
            @test a["k33"] == fill(4.5, 24)
            @test a["k22"] == fill(3.0, 24)
            @test all(x -> x isa Vector, values(a))
        end
    end

    @testset "DIS layout: linear index is the user node number" begin
        mktempdir() do dir
            g = dis_dims(2, 3, 4)
            body = "  BOTM\n    INTERNAL\n" * join(("    " * join(((k - 1) * 12 + (i - 1) * 4 .+ (1:4)), " ") for k in 1:2 for i in 1:3), "\n") * "\n"
            a = griddata(dir, "gwf-dis", g, "  top\n    constant 10.0\n" * body; extra = DIS_2x3x4)
            botm = a["botm"]
            @test size(botm) == (4, 3, 2) && eltype(botm) == Float64
            @test all(botm[j, i, k] == (k - 1) * 12 + (i - 1) * 4 + j for j in 1:4, i in 1:3, k in 1:2)
            @test size(a["top"]) == (4, 3) && all(==(10.0), a["top"])
            # A 3-D array without LAYERED is one read: the values may be on one line.
            a = griddata(dir, "gwf-dis", g, "  top\n    constant 10.0\n  botm\n    INTERNAL\n    " * join(1:24, " ") * "\n"; extra = DIS_2x3x4)
            @test vec(a["botm"]) == 1:24
            # A 2-D array (ncol, nrow) is read row by row: each row starts on a new line.
            a = griddata(dir, "gwf-dis", g, "  top\n    INTERNAL\n    1 2 3 4 99\n    5 6 7 8\n    9 10 11 12\n  botm\n    constant 0\n"; extra = DIS_2x3x4)
            @test a["top"] == reshape(1.0:12.0, 4, 3)
            e = input_error_of(() -> griddata(dir, "gwf-dis", g, "  top\n    INTERNAL\n    1 2 3 4 5 6 7 8 9 10 11 12\n  botm\n    constant 0\n"; extra = DIS_2x3x4))
            @test occursin("TOP", e.msg)  # row 2 reads "botm"
        end
    end

    @testset "LAYERED" begin
        mktempdir() do dir
            g = dis_dims(2, 3, 4)
            write_input(dir, "ext/k2.txt", "13 14 15 16\n17 18 19 20 # comment after the values\n21 22 23 24\n")
            layered = griddata(
                dir, "gwf-npf", g, """
                  K LAYERED
                    INTERNAL FACTOR 1.0
                      1 2 3 4
                      5 6 7 8
                      9 10 11 12
                    OPEN/CLOSE ext/k2.txt
                  ICELLTYPE LAYERED
                    CONSTANT 1
                    INTERNAL
                      0 0 0 0
                      -1 -1 -1 -1
                      1 1 1 1
                """,
            )
            @test layered["k"] == 1:24
            @test layered["icelltype"] == Int32[fill(1, 12); fill(0, 4); fill(-1, 4); fill(1, 4)]
            # The same array without LAYERED.
            whole = griddata(dir, "gwf-npf", g, "  k\n    INTERNAL\n    " * join(1:24, " ") * "\n  icelltype\n    constant 0\n")
            @test whole["k"] == layered["k"]
            # LAYERED rows of a DIS layer start on new lines.
            e = input_error_of(() -> griddata(dir, "gwf-npf", g, "  K LAYERED\n    INTERNAL\n    " * join(1:12, " ") * "\n    CONSTANT 1\n"))
            @test occursin("K (layer 1)", e.msg)
            # Too few control records.
            e = input_error_of(() -> griddata(dir, "gwf-npf", g, "  K LAYERED\n    CONSTANT 1.0\n"))
            @test occursin("layer 2 of 2", e.msg)
            # LAYERED on an array that is not layered is ignored (LoadMf6File.f90).
            a = griddata(dir, "gwf-dis", g, "  top LAYERED\n    constant 3\n  botm layered\n    constant 2\n    constant 1\n"; extra = DIS_2x3x4)
            @test all(==(3.0), a["top"]) && a["botm"][:, :, 1] == fill(2.0, 4, 3) && a["botm"][:, :, 2] == fill(1.0, 4, 3)
            # DISV: botm (ncpl, nlay) without LAYERED is read one layer per row.
            gv = disv_dims(2, 3)
            disv_dims_block = "BEGIN DIMENSIONS\n  NLAY 2\n  NCPL 3\n  NVERT 4\nEND DIMENSIONS\n"
            a = griddata(
                dir, "gwf-disv", gv, "  top\n    constant 1\n  botm\n    internal\n    1 2 3\n    4 5 6\n  idomain layered\n    internal\n    1 1 1 0 0 0\n    constant 1\n";
                extra = disv_dims_block, after = "BEGIN VERTICES\nEND VERTICES\nBEGIN CELL2D\nEND CELL2D\n",
            )
            @test a["botm"] == [1.0 4.0; 2.0 5.0; 3.0 6.0] && size(a["top"]) == (3,)
            @test a["idomain"] == ones(Int32, 3, 2)
            # DISU: LAYERED is one layer of all nodes.
            gu = disu_dims(5; nja = 9)
            a = griddata(dir, "gwf-ic", gu, "  strt LAYERED\n    internal factor 2\n    1 2 3 4 5\n")
            @test a["strt"] == [2.0, 4.0, 6.0, 8.0, 10.0]
        end
    end

    @testset "OPEN/CLOSE text files" begin
        mktempdir() do dir
            g = dis_dims(1, 2, 3)
            write_input(dir, "sub/k.dat", "1 2 3\n4 5 6 7 8\n")
            write_input(dir, "sub/short.dat", "1 2 3\n4\n")
            write_input(dir, "sub/comment.dat", "# a comment line is an error in MF6\n1 2 3 4 5 6\n")
            write_input(dir, "sub/ib.dat", "1 2*0\n3*-1\n")
            write_input(dir, "sub/crlf.dat", "1 2 3\r\n4 5 6\r\n")
            a = griddata(dir, "gwf-npf", g, "  k\n    OPEN/CLOSE sub/k.dat FACTOR 10.\n  icelltype\n    open/close 'sub/ib.dat' factor -1\n  k33\n    open/close sub/crlf.dat\n")
            @test a["k"] == 10.0 .* (1:6)
            @test a["icelltype"] == Int32[-1, 0, 0, 1, 1, 1]
            @test a["k33"] == 1:6
            e = input_error_of(() -> griddata(dir, "gwf-npf", g, "  k\n    OPEN/CLOSE sub/short.dat\n"))
            @test occursin("end of file after 4 of 6", e.msg) && endswith(e.file, "short.dat")
            e = input_error_of(() -> griddata(dir, "gwf-npf", g, "  k\n    OPEN/CLOSE sub/comment.dat\n"))
            @test e.line == 1 && endswith(e.file, "comment.dat")
            e = input_error_of(() -> griddata(dir, "gwf-npf", g, "  k\n    OPEN/CLOSE sub/missing.dat\n"))
            @test occursin("not found", e.msg)
        end
    end

    @testset "binary files" begin
        mktempdir() do dir
            # Writer → reader round trip, both element types.
            x = collect(range(-1.5, 3.25; length = 7))
            p = write_binary_array(joinpath(dir, "x.bin"), x; kstp = 2, kper = 3, pertim = 1.5, totim = 10.0, text = "HEAD", m1 = 7, m2 = 1, m3 = 4)
            @test filesize(p) == BINARY_HEADER_BYTES + 7 * 8
            h, y = read_binary_array(p, Float64, 7)
            @test y == x && h == BinaryArrayHeader(; kstp = 2, kper = 3, pertim = 1.5, totim = 10.0, text = "HEAD", m1 = 7, m2 = 1, m3 = 4)
            ib = Int32[1, 0, -1, typemax(Int32)]
            write_binary_array(joinpath(dir, "ib.bin"), ib; text = "IDOMAIN")
            @test filesize(joinpath(dir, "ib.bin")) == BINARY_HEADER_BYTES + 4 * 4
            h, y = read_binary_array(joinpath(dir, "ib.bin"), Int32, 4)
            @test y == ib && eltype(y) == Int32 && h.m1 == 4 && h.text == "IDOMAIN"
            @test_throws ArgumentError write_binary_array(joinpath(dir, "t.bin"), x; text = "seventeen chars!!")
            e = input_error_of(() -> read_binary_array(p, Float64, 6))
            @test occursin("expected 100 bytes", e.msg) && occursin("found 108", e.msg)
            @test_throws MF6InputError read_binary_array(joinpath(dir, "ib.bin"), Float64, 4)

            # (BINARY) arrays in GRIDDATA: whole grid, LAYERED (one file per layer), FACTOR.
            g = dis_dims(2, 3, 4)
            k = reshape(collect(1.0:24.0), 4, 3, 2)
            write_binary_array(joinpath(dir, "k.bin"), k; text = "K")
            for lay in 1:2
                write_binary_array(joinpath(dir, "k$lay.bin"), k[:, :, lay]; m1 = 4, m2 = 3, m3 = lay)
            end
            write_binary_array(joinpath(dir, "ic.bin"), Int32.(repeat([1, 0], 12)))
            a = griddata(
                dir, "gwf-npf", g, """
                  k
                    OPEN/CLOSE k.bin (BINARY)
                  k33 LAYERED
                    OPEN/CLOSE k1.bin FACTOR 2.0 (BINARY) IPRN 1
                    OPEN/CLOSE k2.bin (BINARY)
                  icelltype
                    OPEN/CLOSE ic.bin (BINARY) FACTOR 3
                """,
            )
            @test a["k"] == 1:24
            @test a["k33"] == [2 .* (1:12); 13:24]
            @test a["icelltype"] == Int32.(repeat([3, 0], 12))
            # A whole-grid file where a layer is expected has the wrong size.
            e = input_error_of(() -> griddata(dir, "gwf-npf", g, "  k layered\n    open/close k.bin (binary)\n    constant 1\n"))
            @test occursin("unexpected file size", e.msg)
        end
    end

    @testset "DISU GRIDDATA and CONNECTIONDATA" begin
        mktempdir() do dir
            # Three cells in a row: 1 - 2 - 3 (nja = 3 diagonals + 4 connections).
            path = write_input(
                dir, "x.disu", """
                BEGIN DIMENSIONS
                  NODES 3
                  NJA 7
                END DIMENSIONS
                BEGIN GRIDDATA
                  top
                    constant 10.0
                  bot
                    internal
                    0.0 -1.0 -2.0
                  area
                    constant 100.0
                  idomain
                    internal
                    1 1 0
                END GRIDDATA
                BEGIN CONNECTIONDATA
                  iac
                    internal
                    2 3 2
                  ja
                    internal
                    1 2
                    2 1 3
                    3 2
                  ihc
                    internal
                    1 1 1 1 1 1 1
                  cl12
                    internal
                    0 5 0 5 5 0 5
                  hwva
                    internal factor 2
                    0 10 0 10 10 0 10
                END CONNECTIONDATA
                """,
            )
            f = read_input_file(path; dfn = "gwf-disu")
            g = grid_dims(f)
            @test g.mshape == [3] && g.dims["NJA"] == 7
            a = read_array_block(find_block(f, "GRIDDATA"), f.dfn, g)
            @test a["top"] == fill(10.0, 3) && a["bot"] == [0.0, -1.0, -2.0] && a["area"] == fill(100.0, 3)
            @test a["idomain"] == Int32[1, 1, 0]
            c = read_array_block(find_block(f, "CONNECTIONDATA"), f.dfn, g)
            @test c["iac"] == Int32[2, 3, 2] && eltype(c["iac"]) == Int32
            @test cumsum([1; c["iac"]]) == [1, 3, 6, 8]  # IA
            @test c["ja"] == Int32[1, 2, 2, 1, 3, 3, 2]
            @test c["ihc"] == ones(Int32, 7)
            @test c["cl12"] == [0, 5, 0, 5, 5, 0, 5]
            @test c["hwva"] == [0, 20, 0, 20, 20, 0, 20]
        end
    end

    @testset "GRIDDATA block errors" begin
        mktempdir() do dir
            g = dis_dims(1, 1, 3)
            # Too many values: the rest of the last line is ignored, a further line is not.
            a = griddata(dir, "gwf-ic", g, "  strt\n    internal\n    1 2 3 4\n")
            @test a["strt"] == [1, 2, 3]
            e = input_error_of(() -> griddata(dir, "gwf-ic", g, "  strt\n    internal\n    1 2 3\n    4\n"))
            @test occursin("input tag \"4\" not found", e.msg) && e.line == 5
            # Too few values: the END line is not data.
            e = input_error_of(() -> griddata(dir, "gwf-ic", g, "  strt\n    internal\n    1 2\n"))
            @test occursin("end of block after 2 of 3", e.msg)
            e = input_error_of(() -> griddata(dir, "gwf-npf", g, "  k\n    constant 1\n  k\n    constant 2\n"))
            @test occursin("more than once", e.msg)
            e = input_error_of(() -> griddata(dir, "gwf-npf", g, "  k\n"))
            @test occursin("missing control record", e.msg)
            e = input_error_of(() -> griddata(dir, "gwf-npf", g, "  k netcdf\n"))
            @test occursin("NETCDF", e.msg)
            e = input_error_of(() -> griddata(dir, "gwf-npf", g, "  k timearrayseries ts1\n    constant 1\n"))
            @test occursin("TIMEARRAYSERIES", e.msg)
            # grid_dims from a DIS file
            path = write_input(dir, "m.dis", DIS_2x3x4 * "BEGIN GRIDDATA\n  top\n    constant 1\n  botm\n    constant 0\nEND GRIDDATA\n")
            gd = grid_dims(read_input_file(path; dfn = "gwf-dis"))
            @test gd.mshape == [2, 3, 4] && gd.dims == dis_dims(2, 3, 4).dims
            path = write_input(dir, "m.disu", "BEGIN DIMENSIONS\n  NODES 5\n  NJA 13\nEND DIMENSIONS\nBEGIN GRIDDATA\n  top\n    constant 1\n  bot\n    constant 0\n  area\n    constant 1\nEND GRIDDATA\nBEGIN CONNECTIONDATA\nEND CONNECTIONDATA\n")
            gd = grid_dims(read_input_file(path; dfn = "gwf-disu"))
            @test gd.mshape == [5] && gd.dims["NJA"] == 13
            path = write_input(dir, "bad.dis", "BEGIN DIMENSIONS\n  NLAY 2\n  NROW 0\n  NCOL 4\nEND DIMENSIONS\nBEGIN GRIDDATA\nEND GRIDDATA\n")
            @test_throws MF6InputError grid_dims(read_input_file(path; dfn = "gwf-dis"))
        end
    end
end

# Every GRIDDATA array (and DISU CONNECTIONDATA array) of every in-scope MF6 example
# (validation/examples.toml) reads with the grid dimensions of its model; the number of
# active cells and the ICELLTYPE counts agree with the inventory (read by the independent
# reader validation/mf6_input.jl).
@testset "MF6 example arrays" begin
    examples = in_scope_examples(; tiers = 0:7)
    if isempty(examples)
        @info "Skipping the MF6 example array test: $(joinpath(REFERENCE_DIR, "mf6examples")) not found (validation/fetch_reference.sh)"
    else
        inventory = TOML.parsefile(joinpath(@__DIR__, "..", "..", "validation", "examples.toml"))["examples"]
        narrays = 0
        nvalues = 0
        nblocks = 0
        for (name, simdir) in examples
            @testset "$name" begin
                models = inventory[name]["models"]
                for (mname, files) in model_package_files(simdir)
                    dis = only(filter(f -> f.dfn.subcomponent in ("DIS", "DISV", "DISU"), files))
                    g = grid_dims(dis)
                    arrays = Dict{String, Array}()
                    for f in files, b in f.blocks
                        db = dfn_block(f.dfn, b.name)
                        (b.name == "PERIOD" || !any(v -> v.reader === :readarray, db.variables)) && continue
                        a = read_array_block(b, f.dfn, g; simdir)
                        nblocks += 1
                        for (vname, A) in a
                            v = dfn_variable(db, vname)
                            @test size(A) == Tuple(array_size(v.shape, g))
                            @test eltype(A) == (v.type === :double ? Float64 : Int32)
                            narrays += 1
                            nvalues += length(A)
                            arrays[f.dfn.subcomponent * "." * vname] = A
                        end
                    end
                    m = models[only(filter(k -> lowercase(k) == lowercase(mname), keys(models)))]
                    nodes = prod(g.mshape)
                    idomain = get(arrays, dis.dfn.subcomponent * ".idomain", nothing)
                    active = isnothing(idomain) ? trues(nodes) : vec(idomain) .> 0
                    @test count(active) == m["nactive"]
                    ict = vec(get(arrays, "NPF.icelltype", zeros(Int32, nodes)))
                    @test count(active .& (ict .== 0)) == m["npf"]["icelltype_confined"]
                    @test count(active .& (ict .> 0)) == m["npf"]["icelltype_positive"]
                    @test count(active .& (ict .< 0)) == m["npf"]["icelltype_negative"]
                end
            end
        end
        @info "MF6 example arrays: $(length(examples)) simulations, $nblocks blocks, $narrays arrays, $nvalues values"
    end
end

"""
    read_grb(path) -> Dict{String, Vector}

The arrays of an MF6 binary grid file (`.grb`): four 50-character header lines (GRID,
VERSION, NTXT, LENTXT), NTXT definitions of LENTXT characters (`NAME TYPE NDIM n dims`),
then the data (INTEGER as Int32, DOUBLE as Float64; scalars as one-element vectors).
"""
function read_grb(path)
    return open(path) do io
        hdr = [String(read(io, 50)) for _ in 1:4]
        ntxt = parse(Int, split(hdr[3])[2])
        lentxt = parse(Int, split(hdr[4])[2])
        defs = [split(String(read(io, lentxt))) for _ in 1:ntxt]
        d = Dict{String, Vector}()
        for df in defs
            ndim = parse(Int, df[4])
            n = ndim == 0 ? 1 : prod(parse.(Int, df[5:(4 + ndim)]))
            v = Vector{df[2] == "INTEGER" ? Int32 : Float64}(undef, n)
            read!(io, v)
            d[df[1]] = v
        end
        return d
    end
end

# The discretization arrays (TOP, BOTM/BOT, IDOMAIN, DISU IA/JA) and ICELLTYPE on active
# cells agree with the binary grid file MF6 wrote in the reference run of every in-scope
# example (reference/runs/<example>/raw, validation/run_reference.jl).
@testset "MF6 example arrays vs .grb" begin
    examples = in_scope_examples(; tiers = 0:7)
    runs = joinpath(REFERENCE_DIR, "runs")
    if isempty(examples) || !isdir(runs)
        @info "Skipping the .grb comparison: $(joinpath(REFERENCE_DIR, "mf6examples")) or $runs not found (validation/fetch_reference.sh, validation/run_reference.jl)"
    else
        inventory = TOML.parsefile(joinpath(@__DIR__, "..", "..", "validation", "examples.toml"))["examples"]
        ncompared = 0
        for (name, simdir) in examples
            rdir = joinpath(runs, name, "raw")
            if !isdir(rdir)
                @info "No reference run for $name: .grb comparison skipped"
                continue
            end
            @testset "$name" begin
                models = inventory[name]["models"]
                for (mname, files) in model_package_files(simdir)
                    m = models[only(filter(k -> lowercase(k) == lowercase(mname), keys(models)))]
                    haskey(m, "grb_file") || continue  # NOGRB
                    G = read_grb(joinpath(rdir, m["grb_file"]))
                    dis = only(filter(f -> f.dfn.subcomponent in ("DIS", "DISV", "DISU"), files))
                    g = grid_dims(dis)
                    A = read_array_block(find_block(dis, "GRIDDATA"), dis.dfn, g; simdir)
                    cd = find_block(dis, "CONNECTIONDATA")
                    isnothing(cd) || merge!(A, read_array_block(cd, dis.dfn, g; simdir))
                    npf = only(filter(f -> f.dfn.subcomponent == "NPF", files))
                    N = read_array_block(find_block(npf, "GRIDDATA"), npf.dfn, g; simdir)
                    nodes = prod(g.mshape)
                    idomain = haskey(A, "idomain") ? vec(A["idomain"]) : ones(Int32, nodes)
                    @test vec(A["top"]) == G["TOP"]
                    haskey(G, "BOTM") && @test vec(A["botm"]) == G["BOTM"]
                    haskey(G, "BOT") && @test vec(A["bot"]) == G["BOT"]
                    haskey(G, "IDOMAIN") && @test idomain == G["IDOMAIN"]
                    if haskey(A, "ja")
                        @test vec(A["ja"]) == G["JA"]
                        @test cumsum([1; A["iac"]]) == G["IA"]
                    end
                    active = idomain .> 0
                    ict = vec(get(N, "icelltype", zeros(Int32, nodes)))
                    @test ict[active] == G["ICELLTYPE"][active]
                    ncompared += 1
                end
            end
        end
        @info "MF6 example arrays vs .grb: $ncompared models compared"
    end
end
