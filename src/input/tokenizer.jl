# Tokenizer for MF6 text input (host only).
#
# Ports the line and word rules of MF6:
# - `u9rdcom` and `get_line` (src/Utilities/InputOutput.f90): a line is a comment when,
#   after leading blanks, commas and tabs, it starts with `#`, `!` or `//`; blank lines
#   are skipped. Comment characters later in a line are NOT special: MF6 splits them into
#   words like any other text, and the readers ignore the words after the values they
#   need ("Comments can also be placed at the end of some input lines, after the
#   required information", mf6io.pdf, "Block and Keyword Input").
# - `URWORD` (src/Utilities/InputOutput.f90): words are separated by blanks, commas and
#   tabs; a word starting with `'` or `"` ends at the next quote of the same kind (it may
#   hold blanks and commas, and the other kind of quote) and the quotes are removed; a
#   missing closing quote extends the word to the end of the line. Case is preserved in
#   the tokens; keywords are compared in upper case (`keyword`).
# - Numbers are read like Fortran list-directed `read(word, *)`, which URWORD uses for
#   `ncode` 2 and 3 (`parse_int`, `parse_double`).

"""
    MF6InputError(file, line, msg)

An error in an MF6 input file. `showerror` prints `file:line: msg` (`line = 0` when the
error is not about a particular line).
"""
struct MF6InputError <: Exception
    file::String
    line::Int
    msg::String
end

function Base.showerror(io::IO, e::MF6InputError)
    print(io, "MF6InputError: ", e.file)
    e.line > 0 && print(io, ":", e.line)
    return print(io, ": ", e.msg)
end

"""
    InputLine

One non-comment line of an MF6 input file: its words (`tokens`, original case, quotes
removed; substrings of `text`), the line as read (`text`, without line terminator), the
file it was read from and its line number (1-based).
"""
struct InputLine
    tokens::Vector{SubString{String}}
    text::String
    file::String
    line::Int
end

Base.length(l::InputLine) = length(l.tokens)
Base.getindex(l::InputLine, i::Integer) = l.tokens[i]
Base.firstindex(::InputLine) = 1
Base.lastindex(l::InputLine) = length(l.tokens)
Base.iterate(l::InputLine, s...) = iterate(l.tokens, s...)
Base.eltype(::Type{InputLine}) = SubString{String}
Base.show(io::IO, l::InputLine) = print(io, "InputLine(", l.file, ":", l.line, ", ", repr(l.text), ")")

"""
    input_error(l::InputLine, msg)

Throw an `MF6InputError` for line `l`, quoting the line.
"""
input_error(l::InputLine, msg::AbstractString) =
    throw(MF6InputError(l.file, l.line, string(msg, " (line: \"", strip(l.text), "\")")))

# URWORD separators.
@inline _is_separator(c::Char) = c == ' ' || c == ',' || c == '\t'

"""
    is_comment_line(text) -> Bool

Whether MF6 skips `text`: blank, or (after leading blanks, commas and tabs) starting with
`#`, `!` or `//` (`u9rdcom`).
"""
function is_comment_line(text::AbstractString)
    i = findfirst(!_is_separator, text)
    isnothing(i) && return true
    c = text[i]
    c == '#' && return true
    c == '!' && return true
    return c == '/' && startswith(SubString(text, i), "//")
end

"""
    tokenize_line(text) -> Vector{SubString{String}}

Split one line into words with MF6's `URWORD` rules (see the file header). Quoted words
are returned without their quotes; an empty quoted word (`''`) gives an empty token.
Comment characters are only special at the start of a line (`is_comment_line`), so
`tokenize_line("a # b")` returns `["a", "#", "b"]`, as MF6 reads it.
"""
function tokenize_line(text::AbstractString)
    s = String(text)
    tokens = SubString{String}[]
    n = ncodeunits(s)
    i = 1
    while i <= n
        c = s[i]
        if _is_separator(c)
            i = nextind(s, i)
            continue
        end
        if c == '\'' || c == '"'
            start = nextind(s, i)
            j = start
            while j <= n && s[j] != c
                j = nextind(s, j)
            end
            # [start, j) is the word; a missing closing quote ends it at the line end.
            push!(tokens, SubString(s, start, prevind(s, j)))
            i = j <= n ? nextind(s, j) : j
        else
            j = i
            while j <= n && !_is_separator(s[j])
                j = nextind(s, j)
            end
            push!(tokens, SubString(s, i, prevind(s, j)))
            i = j
        end
    end
    return tokens
end

"""
    read_input_lines(path) -> Vector{InputLine}

Read the non-comment lines of a text file (`u9rdcom`). A trailing carriage return
(CRLF files) is removed; another carriage return in a line is an error, as in MF6's
`get_line`.
"""
function read_input_lines(path::AbstractString)
    isfile(path) || throw(MF6InputError(path, 0, "file not found"))
    file = String(path)
    lines = InputLine[]
    for (i, raw) in enumerate(eachline(file))
        text = endswith(raw, '\r') ? chop(raw) : raw
        if occursin('\r', text)
            throw(MF6InputError(file, i, "isolated carriage return in line"))
        end
        is_comment_line(text) && continue
        t = String(text)
        push!(lines, InputLine(tokenize_line(t), t, file, i))
    end
    return lines
end

"""
    keyword(l::InputLine, i) -> String

Token `i` of `l` in upper case (MF6 compares keywords in upper case, `URWORD` with
`ncode = 1`); `""` when the line has fewer tokens.
"""
keyword(l::InputLine, i::Integer) = i <= length(l.tokens) ? uppercase(l.tokens[i]) : ""

# Fortran list-directed reals: an exponent letter d/D/q/Q is the same as e/E, and an
# exponent may be written without a letter (`1.5+3`, `2.0-4`).
const _FORTRAN_EXPONENT = r"^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$"

"""
    parse_double(s) -> Union{Float64, Nothing}

Parse a real number like Fortran's list-directed `read(s, *) r` (as `URWORD` with
`ncode = 3`): also `1.0d0`, `1.0D-3` and `1.5+3`. Returns `nothing` when `s` is not a
number.
"""
function parse_double(s::AbstractString)
    t = replace(strip(s), r"[dDqQ]" => "e")
    v = tryparse(Float64, t)
    isnothing(v) || return v
    m = match(_FORTRAN_EXPONENT, t)
    isnothing(m) && return nothing
    return tryparse(Float64, string(m[1], "e", m[2]))
end

"""
    parse_int(s) -> Union{Int, Nothing}

Parse an integer like Fortran's list-directed `read(s, *) n` (as `URWORD` with
`ncode = 2`): an optional sign and digits only (`"5.0"` is not an integer). Returns
`nothing` when `s` is not an integer.
"""
parse_int(s::AbstractString) = tryparse(Int, strip(s))

"""
    get_double(l::InputLine, i; name = "value") -> Float64

Token `i` of `l` as a real number; an `MF6InputError` naming `name` and the line when it
is missing or not a number (`BlockParser%GetDouble`).
"""
function get_double(l::InputLine, i::Integer; name::AbstractString = "value")
    i <= length(l.tokens) || input_error(l, "missing $name (double precision)")
    v = parse_double(l.tokens[i])
    isnothing(v) && input_error(l, "could not read $name as a double precision real from \"$(l.tokens[i])\"")
    return v
end

"""
    get_int(l::InputLine, i; name = "value") -> Int

Token `i` of `l` as an integer; an `MF6InputError` naming `name` and the line when it is
missing or not an integer (`BlockParser%GetInteger`).
"""
function get_int(l::InputLine, i::Integer; name::AbstractString = "value")
    i <= length(l.tokens) || input_error(l, "missing $name (integer)")
    v = parse_int(l.tokens[i])
    isnothing(v) && input_error(l, "could not read $name as an integer from \"$(l.tokens[i])\"")
    return v
end
