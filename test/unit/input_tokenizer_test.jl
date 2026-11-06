# MF6 input tokenizer (T1.1): comment lines, URWORD word rules, numbers.
using ModflowGPU: MF6InputError, InputLine, is_comment_line, tokenize_line, read_input_lines,
    keyword, parse_double, parse_int, get_double, get_int, input_error

"Write `text` to `name` in `dir` and return the path."
function write_input(dir, name, text)
    path = joinpath(dir, name)
    mkpath(dirname(path))
    write(path, text)
    return path
end

@testset "Tokenizer" begin
    @testset "comment lines (u9rdcom)" begin
        for c in ("# x", "! x", "// x", "   # x", "\t! x", ",, # x", " , // x", "#", "", "   ", ",,", "\t")
            @test is_comment_line(c)
        end
        for c in ("a # b", "x", "/x", "/ /", "  BEGIN options", "1.0 ! c")
            @test !is_comment_line(c)
        end
    end

    @testset "words (URWORD)" begin
        @test tokenize_line("a b,c\td") == ["a", "b", "c", "d"]
        @test tokenize_line(",, a ,, b,") == ["a", "b"]
        @test tokenize_line("  \t ") == String[]
        # Case is preserved; keywords are compared in upper case.
        @test tokenize_line("BEGIN Options MyFile.TXT") == ["BEGIN", "Options", "MyFile.TXT"]
        # Quoted words keep blanks, commas and the other kind of quote; quotes are removed.
        @test tokenize_line("'my file.txt' x") == ["my file.txt", "x"]
        @test tokenize_line("\"a, b\" c") == ["a, b", "c"]
        @test tokenize_line("'WORD\"S' \"WORD'S\"") == ["WORD\"S", "WORD'S"]
        # A missing closing quote extends the word to the end of the line.
        @test tokenize_line("x 'abc def") == ["x", "abc def"]
        # After the closing quote the next word starts.
        @test tokenize_line("'ab'cd") == ["ab", "cd"]
        # A quote inside a word is an ordinary character.
        @test tokenize_line("ab'cd e") == ["ab'cd", "e"]
        @test tokenize_line("'' x") == ["", "x"]
        # Comment characters after the start of a line are ordinary words, as in MF6.
        @test tokenize_line("1 2 3 -1.0 # a comment") == ["1", "2", "3", "-1.0", "#", "a", "comment"]
        @test tokenize_line("välue ü") == ["välue", "ü"]
        @test eltype(tokenize_line("a")) == SubString{String}
    end

    @testset "numbers (list-directed read)" begin
        @test parse_double("1.0") == 1.0
        @test parse_double("  -2.5 ") == -2.5
        @test parse_double("1.0d0") == 1.0
        @test parse_double("1.5D-3") == 1.5e-3
        @test parse_double("2.5E+2") == 250.0
        @test parse_double("1.5+3") == 1500.0
        @test parse_double("-2.0-1") == -0.2
        @test parse_double(".5") == 0.5
        @test parse_double("5.") == 5.0
        @test parse_double("+3") == 3.0
        @test parse_double("7") == 7.0
        @test isnothing(parse_double("abc"))
        @test isnothing(parse_double("1.0x"))
        @test isnothing(parse_double(""))
        @test parse_int("5") == 5
        @test parse_int("+5") == 5
        @test parse_int("-12") == -12
        @test isnothing(parse_int("5.0"))
        @test isnothing(parse_int("1e3"))
        @test isnothing(parse_int("x"))
    end

    @testset "lines and errors" begin
        mktempdir() do dir
            path = write_input(
                dir, "a.txt",
                "# header\r\n\r\nBEGIN options\r\n  ! comment\r\n  SAVE_FLOWS\r\n  a  'b c'  # trailing\r\nEND options\r\n",
            )
            lines = read_input_lines(path)
            @test [l.line for l in lines] == [3, 5, 6, 7]
            @test all(l -> l.file == path, lines)
            @test lines[1].text == "BEGIN options"  # carriage return removed
            @test lines[3].tokens == ["a", "b c", "#", "trailing"]
            @test length(lines[3]) == 4
            @test lines[3][2] == "b c"
            @test collect(lines[2]) == ["SAVE_FLOWS"]
            @test keyword(lines[1], 2) == "OPTIONS"
            @test keyword(lines[1], 3) == ""

            l = InputLine(tokenize_line("x 12 3.5 abc 4.0"), "x 12 3.5 abc 4.0", path, 9)
            @test get_int(l, 2) == 12
            @test get_double(l, 3) == 3.5
            @test get_double(l, 2) == 12.0
            @test_throws MF6InputError get_int(l, 5)       # 4.0 is not an integer
            @test_throws MF6InputError get_double(l, 4)
            @test_throws MF6InputError get_int(l, 6)       # missing
            err = try
                get_int(l, 4; name = "NLAY")
            catch e
                e
            end
            @test err isa MF6InputError
            @test err.file == path && err.line == 9
            msg = sprint(showerror, err)
            @test occursin("$path:9:", msg)
            @test occursin("NLAY", msg)
            @test occursin("abc", msg)
            @test_throws MF6InputError input_error(l, "bad")

            bad = write_input(dir, "cr.txt", "BEGIN options\nA\rB\nEND options\n")
            err = try
                read_input_lines(bad)
            catch e
                e
            end
            @test err isa MF6InputError && err.line == 2
            @test_throws MF6InputError read_input_lines(joinpath(dir, "missing.txt"))
            @test sprint(showerror, MF6InputError("f.txt", 0, "m")) == "MF6InputError: f.txt: m"
        end
    end
end
