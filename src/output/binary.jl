# Shared helpers for the MF6 binary output files (heads, budgets, binary grid files).
#
# MF6 opens its binary output files as unformatted stream files (no Fortran record
# markers; `openfile` in InputOutput.f90 with `access='stream'`), so a file is a plain
# sequence of 4-byte integers, 8-byte reals and fixed-length character fields, in the
# byte order of the machine (little-endian on every platform MF6 runs on). The readers
# and writers below read and write little-endian bytes.

"""
    ModflowGPU.MF6BinaryFileError(path, offset, msg)

Error raised by the output readers ([`open_head_file`](@ref), [`open_budget_file`](@ref),
[`read_grb`](@ref)) when an MF6 binary output file is malformed or truncated: `offset` is
the byte offset in the file where the problem was found.
"""
struct MF6BinaryFileError <: Exception
    path::String
    offset::Int64
    msg::String
end

function Base.showerror(io::IO, e::MF6BinaryFileError)
    return print(io, "MF6BinaryFileError: ", e.path, " (byte ", e.offset, "): ", e.msg)
end

# Length of the character fields of the binary output files (TEXT, model and package
# names, auxiliary names): `character(len=16)` in ulasav/ubdsv1/ubdsv06.
const OUTPUT_TEXT_BYTES = 16

_read_le(io::IO, ::Type{T}) where {T} = ltoh(read(io, T))

function _read_text(io::IO, n::Integer = OUTPUT_TEXT_BYTES)
    b = read(io, n)
    length(b) == n || throw(EOFError())
    return String(b)
end

"""
    pad_text(s, n = 16; justify = :left) -> String

`s` as a fixed-length character field of `n` bytes, padded with blanks on the right
(`justify = :left`) or on the left (`justify = :right`). A string of exactly `n` bytes is
returned unchanged, so headers read from a file are written back byte for byte.
"""
function pad_text(s::AbstractString, n::Integer = OUTPUT_TEXT_BYTES; justify::Symbol = :left)
    m = ncodeunits(s)
    m <= n || throw(ArgumentError("text \"$s\" is longer than $n characters"))
    isascii(s) || throw(ArgumentError("text \"$s\" is not ASCII"))
    m == n && return String(s)
    justify === :left && return rpad(s, n)
    justify === :right && return lpad(s, n)
    throw(ArgumentError("justify must be :left or :right, got :$justify"))
end

# Write a Float64/Int32 array in file byte order.
function _write_le(io::IO, x::AbstractArray{T}) where {T <: Union{Int32, Float64}}
    if ENDIAN_BOM == 0x04030201 && x isa Union{DenseArray, SubArray{<:Any, <:Any, <:Array}}
        write(io, x)
    else
        for v in x
            write(io, htol(v))
        end
    end
    return nothing
end

# Read `length(out)` values into `out` starting at the current position.
function _read_le!(io::IO, out::DenseArray{T}) where {T <: Union{Int32, Float64}}
    read!(io, out)
    ENDIAN_BOM == 0x04030201 || (out .= ltoh.(out))
    return out
end

# Read `n` values of type `T` into `out[first:(first + n - 1)]`, without a temporary.
function _read_le!(io::IO, out::Array{T}, first::Integer, n::Integer) where {T <: Union{Int32, Float64}}
    checkbounds(out, first:(first + n - 1))
    GC.@preserve out unsafe_read(io, pointer(out, first), n * sizeof(T))
    ENDIAN_BOM == 0x04030201 || (@views out[first:(first + n - 1)] .= ltoh.(out[first:(first + n - 1)]))
    return out
end

# Unaligned little-endian load of a `T` at byte `off` (0-based) of `buf`.
@inline function _load_le(::Type{T}, buf::Vector{UInt8}, off::Integer) where {T}
    return ltoh(GC.@preserve buf unsafe_load(Ptr{T}(pointer(buf, off + 1))))
end
