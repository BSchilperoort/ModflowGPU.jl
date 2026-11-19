# End-to-end parity of `ModflowGPU.run` (T2.8) against MF6 6.8.2 tight-closure runs,
# PLAN.md section 4:
#
# - models: the Tier 0–2 examples that load with `strict = true`, against their reference
#   run (`reference_rundir`: reference/runs/<example>/tight, or tight-relative when the
#   tight run did not converge, e.g. ex-gwf-fhb with its time series since T6.3;
#   ex-gwf-hwell-a, 153164 cells and 60 steps, runs with
#   the AMG preconditioner since T4.3, about 15 s on the CPU backend; MFGPU_TEST_HWELL=0
#   skips it), the confined Tier 3 examples ex-gwf-lgrv-gc and ex-gwf-lgrv-gr (T6.0;
#   `parity_examples`; -gr has 672525 cells, :single with AMG, MFGPU_TEST_LGRV_GR=0 skips
#   it), and the synthetic models test/models/ts_tas (T6.3: TS6 and TAS6 input) and
#   test/models/coupling_rde, run with mf6 at test time;
#   the Tier 5 examples (T6.4: DISV ex-gwf-u1disv, ex-gwf-curve-90, ex-gwf-curvilin) and
#   rad_disu_confined (ex-gwf-rad-disu with ICONVERT 0, `rad_disu_confined`, run with mf6
#   at test time, AMG preconditioner);
# - heads: every saved step of our HDS file (read back with our reader) vs MF6's:
#   `:double` max |Δh| < max(1e-6 m, 1e-7 × head range), `:single` < 1e-3 m;
# - budgets: every package list record and FLOW-JA-FACE/STO-SS of our CBC vs MF6's
#   (full precision), max |Δq| / total in < 1e-6 (`:double`) or 1e-4 (`:single`); the
#   listing tables (every term's rate in and out, incl. the CHD split, read back with
#   `read_listing_budgets`) vs MF6's, within the same criterion plus the listing rounding;
#   percent discrepancy < 0.01 % in `:single`;
# - outer and inner iterations are reported next to MF6's (mfsim.lst), not tested: the
#   preconditioner is the production default (`:auto` = red–black MC-ILU0 since T4.5;
#   MF6: ILU0 in natural order), `MFGPU_TEST_PRECONDITIONER` chooses another;
# - on a GPU backend, the run is repeated on the CPU backend: `:double` heads within 1e-10
#   relative, `:single` within 1e-4 m.
# The comparison helpers (`_run_parity`, ...) are in test/run_parity.jl.

# IMS variants of coupling_rde for the under-relaxation (sln_underrelax) and COMPLEXITY
# presets: (name, OPTIONS lines, NONLINEAR lines). COMPLEX enables backtracking (T6.2a):
# ModflowGPU must backtrack in the same outer iterations as MF6.
const _UR_VARIANTS = [
    ("simple", "", "UNDER_RELAXATION SIMPLE\n  UNDER_RELAXATION_GAMMA 0.8"),
    ("cooley", "", "UNDER_RELAXATION COOLEY\n  UNDER_RELAXATION_GAMMA 0.5"),
    ("dbd", "", "UNDER_RELAXATION DBD\n  UNDER_RELAXATION_THETA 0.7\n  UNDER_RELAXATION_KAPPA 0.1\n  UNDER_RELAXATION_GAMMA 0.2\n  UNDER_RELAXATION_MOMENTUM 0.001"),
    ("moderate", "COMPLEXITY MODERATE", ""),
    ("complex", "COMPLEXITY COMPLEX", ""),
]

function _ur_ims(opts, nonlin)
    return """
    BEGIN options
      PRINT_OPTION  SUMMARY
      $opts
    END options
    BEGIN nonlinear
      OUTER_DVCLOSE  1.0e-9
      OUTER_MAXIMUM  500
      $nonlin
    END nonlinear
    BEGIN linear
      INNER_MAXIMUM  1000
      INNER_DVCLOSE  1.0e-11
      INNER_RCLOSE  1.0e-9
      LINEAR_ACCELERATION  BICGSTAB
    END linear
    """
end

# Gated on the model name like the models of the loop below, so that the parallel driver
# (test/parallel.jl) runs it in one worker only.
@testset "Run parity vs MF6: under-relaxation and COMPLEXITY on coupling_rde" begin
    if !test_model_selected("coupling_rde")
    elseif !isfile(RUN_MF6)
        @info "Skipping the under-relaxation run parity test: mf6 not found at $RUN_MF6"
    else
        for (name, opts, nonlin) in _UR_VARIANTS
            mktempdir() do tmp
                rundir = joinpath(tmp, "rde_$name")
                cp(joinpath(@__DIR__, "..", "models", "coupling_rde"), rundir)
                write(joinpath(rundir, "rde.ims"), _ur_ims(opts, nonlin))
                out = read(Cmd(`$RUN_MF6`; dir = rundir), String)
                @test occursin("Normal termination", out)
                @testset "coupling_rde, $name" begin
                    _run_parity("coupling_rde ($name)", rundir; backtracking = name == "complex")
                end
            end
        end
    end
end

@testset "Run parity vs MF6 (heads, budgets, listing)" begin
    nmodels = Ref(0)
    _run_models() do name, rundir
        nmodels[] += 1
        @testset "$name" begin
            _run_parity(name, rundir; run_parity_options(name)...)
        end
    end
    @info "Run parity test: $(nmodels[]) model(s)"
end

# flopy reads our HDS and CBC files (opt-in: MFGPU_FLOPY=1, needs `uv`; like the flopy
# testset of test/unit/output_test.jl). Every value flopy returns equals our readers'.
if get(ENV, "MFGPU_FLOPY", "") == "1" && test_model_selected("coupling_rde")
    @testset "flopy reads ModflowGPU.run output (coupling_rde)" begin
        mktempdir() do out
            r = ModflowGPU.run(joinpath(@__DIR__, "..", "models", "coupling_rde"); backend = TEST_BACKEND, precision = :single, outdir = out)
            script = joinpath(@__DIR__, "..", "flopy_run_read.py")
            txt = read(`uv run --quiet --no-project --with flopy --with numpy python -I $script $(r.head_file) $(r.budget_file)`, String)
            res = Dict{String, Vector{String}}()
            for l in split(txt, '\n'; keepempty = false)
                w = split(l)
                res[w[1]] = w[2:end]
            end
            num(k) = parse.(Float64, res[k])
            hf = open_head_file(r.head_file)
            @test num("hds_times") == [s.totim for s in hf.steps]
            @test num("hds_kstpkper") == Float64.(vcat([[s.kstp - 1, s.kper - 1] for s in hf.steps]...))
            # flopy's (nlay, nrow, ncol) C-order array has our (ncol, nrow, nlay) memory order.
            @test num("hds_last") == vec(read_head(hf, length(hf)))
            bf = open_budget_file(r.budget_file)
            last = bf.records[end]
            names = unique(strip(x.text) for x in bf.records)
            @test res["cbc_names"] == names
            @test num("cbc_times") == unique(x.totim for x in bf.records)
            for n in names
                i = only(find_budget_records(bf; text = n, kstp = last.kstp, kper = last.kper))
                v = read_budget(bf, i)
                if v isa BudgetList
                    @test num("cbc_$(n)_node") == Float64.(v.id1)
                    @test num("cbc_$(n)_q") == v.q
                else
                    @test num("cbc_$n") == vec(v)
                end
            end
            @info "flopy read ModflowGPU's HDS ($(length(hf)) steps) and CBC ($(length(bf)) records, $(join(names, ", "))) of coupling_rde"
        end
    end
end
