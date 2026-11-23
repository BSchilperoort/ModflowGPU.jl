# The T6.14 test cases (group xt3d_var): each case is a test model of test/models with edits
# made at test time (`xt3d_var_case(name, dst)` writes the inputs to `dst`). Used by
# test/unit/xt3d_var_test.jl; also includable on its own (no dependencies).
const XT3D_VAR_MODELS = joinpath(@__DIR__, "..")

# Input files only (no outputs of an earlier run).
_xv_input(f) = !any(e -> endswith(f, e), (".hds", ".cbc", ".lst", ".txt", ".toml", ".grb", ".csv", "generate.jl", "variants.jl"))

_xv_model_file(f) = endswith(f, ".nam") && f != "mfsim.nam"

# NEWTON in the model name file (REWET removed: MF6 rejects it with NEWTON).
_xv_newton(f, t) = _xv_model_file(f) ? replace(t, "SAVE_FLOWS" => "SAVE_FLOWS\n  NEWTON", count = 1) :
    endswith(f, ".npf") ? replace(t, r"\n *REWET[^\n]*" => "") : t

# XT3D in the NPF file, with K22 and ANGLE1 (rotated anisotropy) when given.
function _xv_xt3d(; k22 = nothing, angle1 = nothing, rhs = false)
    return (f, t) -> begin
        endswith(f, ".npf") || return t
        t = replace(t, r"BEGIN options\n"i => "BEGIN options\n  XT3D$(rhs ? "  RHS" : "")\n", count = 1)
        add = (isnothing(k22) ? "" : "  k22\n    CONSTANT  $k22\n") * (isnothing(angle1) ? "" : "  angle1\n    CONSTANT  $angle1\n")
        return replace(t, r"END griddata"i => add * "END griddata", count = 1)
    end
end

# gnc_disv (T6.13) made convertible: TOP 12 (the water table in layer 1), ICELLTYPE and
# ICONVERT 1.
function _xv_gnc_conv(f, t)
    endswith(f, ".disv") && return replace(t, "top\n    CONSTANT  0.0" => "top\n    CONSTANT  12.0")
    endswith(f, ".npf") && return replace(t, "icelltype\n    CONSTANT  0" => "icelltype\n    CONSTANT  1")
    endswith(f, ".sto") && return replace(t, "iconvert\n    CONSTANT  0" => "iconvert\n    CONSTANT  1")
    return t
end

_xv_nognc(f, t) = _xv_model_file(f) ? replace(t, r"\n *GNC6 [^\n]*"i => "") : t

_xv_bicgstab(f, t) = endswith(f, ".ims") ? replace(t, r"LINEAR_ACCELERATION +CG"i => "LINEAR_ACCELERATION  BICGSTAB") : t

_xv_explicit(f, t) = endswith(f, ".gnc") ? replace(t, r"BEGIN options\n"i => "BEGIN options\n  EXPLICIT\n", count = 1) : t

# A package added to the model name file (`ftype`, file) and its file.
function _xv_package(ftype, file, text)
    return (f, t) -> _xv_model_file(f) ? replace(t, r"\n *OC6 "i => "\n  $ftype  $file  $(lowercase(ftype[1:3]))\n  OC6 ", count = 1) : t,
        file => text
end

# HFB6 of gnc_disv: coarse–coarse, coarse–fine (with GNC records), a vertical barrier and a
# negative HYDCHR (a multiplier).
const _XV_GNC_HFB = """
BEGIN options
END options

BEGIN dimensions
  MAXHFB  5
END dimensions

BEGIN period  1
  1  1  1  2  1.0e-3
  1  10  1  15  1.0e-3
  1  11  1  18  5.0e-3
  1  2  2  2  1.0e-2
  1  3  1  4  -0.5
END period  1
"""

# TVK6 of gnc_disv: K of a fine and a coarse cell in period 1, a coarse cell and K33 of a
# layer-2 cell in period 2.
const _XV_GNC_TVK = """
BEGIN options
END options

BEGIN period  1
  1  15  K  15.0
  1  9  K  1.0
END period  1

BEGIN period  2
  1  10  K  20.0
  2  38  K33  0.05
END period  2
"""

# HFB6 of test/models/xt3d (confined, ANGLE1–3).
const _XV_XT3D_HFB = """
BEGIN options
END options

BEGIN dimensions
  MAXHFB  6
END dimensions

BEGIN period  1
  1  2  3  1  2  4  1.0e-3
  1  3  3  1  3  4  1.0e-3
  1  4  3  1  4  4  2.0e-3
  2  1  2  2  2  2  -0.5
  1  5  6  2  5  6  1.0e-2
  3  4  4  3  4  5  0.0
END period  1
"""

# name => (source model directory under test/models, edits, extra files)
function xt3d_var_cases()
    hfbg = _xv_package("HFB6", "gnc_disv.hfb", _XV_GNC_HFB)
    hfbx = _xv_package("HFB6", "xt3d.hfb", _XV_XT3D_HFB)
    tvk(f, t) = endswith(f, ".npf") ? replace(t, r"BEGIN options\n"i => "BEGIN options\n  TVK6  FILEIN  gnc_disv.npf.tvk\n", count = 1) : t
    xdisv = _xv_xt3d(; k22 = 1.0, angle1 = 30.0)
    xhfb = _xv_xt3d(; k22 = 4.0, angle1 = 25.0)
    return [
        "xt3dnr_cont" => ("xt3d_var/xt3dnr_cont", [], []),
        "xt3dnr_conv" => ("xt3d_var/xt3dnr_conv", [], []),
        "xt3dnr_conv_std" => ("xt3d_var/xt3dnr_conv_std", [], []),
        "xt3d_conv" => ("xt3d_var/xt3d_conv", [], []),
        "xt3d_conv_newton" => ("xt3d_var/xt3d_conv", [_xv_newton], []),
        "xt3d_conv_rhs" => ("xt3d_var/xt3d_conv", [(f, t) -> endswith(f, ".npf") ? replace(t, "  XT3D\n" => "  XT3D  RHS\n") : t], []),
        "xt3d_disv_conv" => ("gnc/gnc_disv", [_xv_gnc_conv, _xv_nognc, xdisv], []),
        "xt3d_disv_conv_newton" => ("gnc/gnc_disv", [_xv_gnc_conv, _xv_nognc, xdisv, _xv_newton], []),
        "xt3d_hfb_conv" => ("hfb_conv", [xhfb], []),
        "xt3d_hfb_conv_newton" => ("hfb_conv", [xhfb, _xv_newton], []),
        "xt3d_hfb_confined" => ("xt3d", [hfbx[1]], [hfbx[2]]),
        "xt3d_tvk_conv" => ("tv/tv_conv", [_xv_xt3d(; angle1 = 20.0), _xv_bicgstab], []),
        "gnc_disv_conv" => ("gnc/gnc_disv", [_xv_gnc_conv], []),
        "gnc_disv_conv_newton" => ("gnc/gnc_disv", [_xv_gnc_conv, _xv_newton], []),
        "gnc_disv_conv_explicit" => ("gnc/gnc_disv", [_xv_gnc_conv, _xv_explicit], []),
        "gnc_disv_hfb" => ("gnc/gnc_disv", [_xv_gnc_conv, hfbg[1]], [hfbg[2]]),
        "gnc_disv_tvk" => ("gnc/gnc_disv", [_xv_gnc_conv, tvk], ["gnc_disv.npf.tvk" => _XV_GNC_TVK]),
    ]
end

"""
    xt3d_var_case(name, dst) -> dst

Write the inputs of the T6.14 test case `name` (see `xt3d_var_cases`) to the directory `dst`.
"""
function xt3d_var_case(name, dst)
    src, edits, extra = Dict(xt3d_var_cases())[name]
    srcdir = joinpath(XT3D_VAR_MODELS, src)
    mkpath(dst)
    for f in readdir(srcdir)
        _xv_input(f) || continue
        t = read(joinpath(srcdir, f), String)
        for e in edits
            t = e(f, t)
        end
        write(joinpath(dst, f), t)
    end
    for (f, t) in extra
        write(joinpath(dst, f), t)
    end
    chmod(dst, 0o755; recursive = true)
    return dst
end
