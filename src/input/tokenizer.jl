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
#
# Memory: a file is read into one `String`; each `InputLine` is a view of it (`text`),
# and its words are found on demand (`eachtoken`, no allocation) instead of being
# stored, so large array blocks cost little more than the file itself.

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

One non-comment line of an MF6 input file: the line as read (`text`, a view of the file
contents without the line terminator), the file it was read from and its line number
(1-based). The words are not stored: `eachtoken(l)` iterates over them without
allocating, `tokens(l)` collects them, and `l[i]`, `length(l)`, `keyword(l, i)` find
them by scanning the line.
"""
struct InputLine
    text::SubString{String}
    file::String
    line::Int
end

InputLine(text::AbstractString, file::AbstractString, line::Integer) =
    InputLine(SubString(String(text)), String(file), Int(line))

Base.show(io::IO, l::InputLine) = print(io, "InputLine(", l.file, ":", l.line, ", ", repr(String(l.text)), ")")

"""
    input_error(l::InputLine, msg)

Throw an `MF6InputError` for line `l`, quoting the line.
"""
input_error(l::InputLine, msg::AbstractString) =
    throw(MF6InputError(l.file, l.line, string(msg, " (line: \"", strip(l.text), "\")")))

# URWORD separators: blank, comma, tab.
@inline _is_separator(c::UInt8) = c == UInt8(' ') || c == UInt8(',') || c == UInt8('\t')
@inline _is_separator(c::Char) = c == ' ' || c == ',' || c == '\t'

"""
    EachToken

Iterator over the words of a line (`eachtoken`); yields `SubString{String}`s of the
underlying string.
"""
struct EachToken
    s::String
    first::Int  # first byte of the line
    last::Int   # last byte of the line
end

"""
    eachtoken(l::InputLine)
    eachtoken(text::AbstractString)

Iterate over the words of a line with MF6's `URWORD` rules (see the file header),
without allocating. Quoted words are returned without their quotes; an empty quoted word
(`''`) gives an empty token.
"""
eachtoken(l::InputLine) = eachtoken(l.text)
eachtoken(s::SubString{String}) = EachToken(s.string, s.offset + 1, s.offset + s.ncodeunits)
eachtoken(s::String) = EachToken(s, 1, ncodeunits(s))
eachtoken(s::AbstractString) = eachtoken(String(s))

Base.IteratorSize(::Type{EachToken}) = Base.SizeUnknown()
Base.eltype(::Type{EachToken}) = SubString{String}

# Bytes are compared as ASCII: separators and quotes are ASCII, and UTF-8 continuation
# bytes never equal an ASCII byte, so every word starts and ends on a character boundary.
@inline function Base.iterate(it::EachToken, i::Int = it.first)
    s = it.s
    n = it.last
    while i <= n && _is_separator(codeunit(s, i))
        i += 1
    end
    i > n && return nothing
    c = codeunit(s, i)
    if c == UInt8('\'') || c == UInt8('"')
        start = i + 1
        j = start
        while j <= n && codeunit(s, j) != c
            j += 1
        end
        # [start, j) is the word; a missing closing quote ends it at the line end.
        tok = j > start ? SubString(s, start, prevind(s, j)) : SubString(s, 1, 0)
        return tok, j + 1
    end
    j = i
    while j <= n && !_is_separator(codeunit(s, j))
        j += 1
    end
    return SubString(s, i, prevind(s, j)), j
end

"""
    tokenize_line(text) -> Vector{SubString{String}}

The words of one line (`collect(eachtoken(text))`). Comment characters are only special
at the start of a line (`is_comment_line`), so `tokenize_line("a # b")` returns
`["a", "#", "b"]`, as MF6 reads it.
"""
tokenize_line(text::AbstractString) = collect(eachtoken(text))

"""
    tokens(l::InputLine) -> Vector{SubString{String}}

The words of `l`, collected (allocates; use `eachtoken` to avoid that).
"""
tokens(l::InputLine) = collect(eachtoken(l))

"""
    ntokens(l::InputLine) -> Int

The number of words of `l` (scans the line).
"""
ntokens(l::InputLine) = count(_ -> true, eachtoken(l))

Base.length(l::InputLine) = ntokens(l)
Base.iterate(l::InputLine, s...) = iterate(eachtoken(l), s...)
Base.IteratorSize(::Type{InputLine}) = Base.SizeUnknown()
Base.eltype(::Type{InputLine}) = SubString{String}

"""
    token(l::InputLine, i) -> Union{SubString{String}, Nothing}

Word `i` of `l`, or `nothing` when the line has fewer words (scans the line).
"""
function token(l::InputLine, i::Integer)
    k = 0
    for t in eachtoken(l)
        k += 1
        k == i && return t
    end
    return nothing
end

function Base.getindex(l::InputLine, i::Integer)
    t = token(l, i)
    isnothing(t) && throw(BoundsError(l, i))
    return t
end

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

# Byte version of is_comment_line on buf[a:b].
function _is_comment_bytes(buf::String, a::Int, b::Int)
    i = a
    while i <= b && _is_separator(codeunit(buf, i))
        i += 1
    end
    i > b && return true
    c = codeunit(buf, i)
    (c == UInt8('#') || c == UInt8('!')) && return true
    return c == UInt8('/') && i < b && codeunit(buf, i + 1) == UInt8('/')
end

"""
    read_input_lines(path) -> Vector{InputLine}

Read the non-comment lines of a text file (`u9rdcom`). The file is read into one string
and the lines are views of it. A trailing carriage return (CRLF files) is removed;
another carriage return in a line is an error, as in MF6's `get_line`.
"""
function read_input_lines(path::AbstractString)
    isfile(path) || throw(MF6InputError(path, 0, "file not found"))
    file = String(path)
    buf = read(file, String)
    n = ncodeunits(buf)
    lines = InputLine[]
    sizehint!(lines, count(==(UInt8('\n')), codeunits(buf)) + 1)
    i = 1
    lineno = 0
    while i <= n
        lineno += 1
        j = i
        cr = 0
        while j <= n && codeunit(buf, j) != UInt8('\n')
            codeunit(buf, j) == UInt8('\r') && (cr = j)
            j += 1
        end
        stop = j - 1
        if cr == stop && stop >= i
            stop -= 1
            cr = 0
            for k in i:stop
                codeunit(buf, k) == UInt8('\r') && (cr = k)
            end
        end
        cr > 0 && throw(MF6InputError(file, lineno, "isolated carriage return in line"))
        if !_is_comment_bytes(buf, i, stop)
            push!(lines, InputLine(SubString(buf, i, thisind(buf, stop)), file, lineno))
        end
        i = j + 1
    end
    return lines
end

"""
    keyword(l::InputLine, i) -> String

Word `i` of `l` in upper case (MF6 compares keywords in upper case, `URWORD` with
`ncode = 1`); `""` when the line has fewer words.
"""
function keyword(l::InputLine, i::Integer)
    t = token(l, i)
    return isnothing(t) ? "" : uppercase(t)
end

# Fortran list-directed reals: an exponent letter d/D/q/Q is the same as e/E, and an
# exponent may be written without a letter (`1.5+3`, `2.0-4`).
const _FORTRAN_EXPONENT = r"^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$"

# Julia accepts 0x/0b/0o prefixes (and hexadecimal floats); Fortran does not.
function _has_radix_prefix(s::AbstractString)
    i = firstindex(s)
    i <= lastindex(s) && s[i] in ('+', '-') && (i = nextind(s, i))
    i < lastindex(s) || return false
    return s[i] == '0' && s[nextind(s, i)] in ('x', 'X', 'b', 'B', 'o', 'O')
end

"""
    parse_double(s) -> Union{Float64, Nothing}

Parse a real number like Fortran's list-directed `read(s, *) r` (as `URWORD` with
`ncode = 3`): also `1.0d0`, `1.0D-3` and `1.5+3`; not `0x10`. Returns `nothing` when `s`
is not a number.
"""
function parse_double(s::AbstractString)
    _has_radix_prefix(s) && return nothing
    v = tryparse(Float64, s)
    isnothing(v) || return v
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
`ncode = 2`): an optional sign and digits only (`"5.0"` and `"0x10"` are not integers).
Returns `nothing` when `s` is not an integer.
"""
parse_int(s::AbstractString) = _has_radix_prefix(s) ? nothing : tryparse(Int, s)

"""
    get_double(l::InputLine, i; name = "value") -> Float64

Word `i` of `l` as a real number; an `MF6InputError` naming `name` and the line when it
is missing or not a number (`BlockParser%GetDouble`).
"""
function get_double(l::InputLine, i::Integer; name::AbstractString = "value")
    t = token(l, i)
    isnothing(t) && input_error(l, "missing $name (double precision)")
    return _double_or_error(l, t, name)
end

function _double_or_error(l::InputLine, t::AbstractString, name::AbstractString)
    v = parse_double(t)
    isnothing(v) && input_error(l, "could not read $name as a double precision real from \"$t\"")
    return v
end

"""
    get_int(l::InputLine, i; name = "value") -> Int

Word `i` of `l` as an integer; an `MF6InputError` naming `name` and the line when it is
missing or not an integer (`BlockParser%GetInteger`).
"""
function get_int(l::InputLine, i::Integer; name::AbstractString = "value")
    t = token(l, i)
    isnothing(t) && input_error(l, "missing $name (integer)")
    v = parse_int(t)
    isnothing(v) && input_error(l, "could not read $name as an integer from \"$t\"")
    return v
end

"""
    each_double(l::InputLine; first = 1, name = "value")

Iterate over the words of `l` from word `first` on as real numbers (`parse_double`),
without allocating; a word that is not a number throws an `MF6InputError` naming `name`.
For numbers on a list or keyword line (URWORD rules). Not for READARRAY data: array
values follow Fortran list-directed rules (repeat counts `3*1.0`, values spanning lines,
rows starting on new lines), which `list_read!` in `arrays.jl` implements.
"""
each_double(l::InputLine; first::Integer = 1, name::AbstractString = "value") =
    EachDouble(l, Int(first), String(name))

struct EachDouble
    line::InputLine
    first::Int
    name::String
end

Base.IteratorSize(::Type{EachDouble}) = Base.SizeUnknown()
Base.eltype(::Type{EachDouble}) = Float64

@inline function Base.iterate(it::EachDouble)
    toks = eachtoken(it.line)
    st = iterate(toks)
    for _ in 2:it.first
        isnothing(st) && return nothing
        st = iterate(toks, st[2])
    end
    isnothing(st) && return nothing
    return _double_or_error(it.line, st[1], it.name), st[2]
end

@inline function Base.iterate(it::EachDouble, i::Int)
    st = iterate(eachtoken(it.line), i)
    isnothing(st) && return nothing
    return _double_or_error(it.line, st[1], it.name), st[2]
end
