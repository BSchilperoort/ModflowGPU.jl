# Simulation loader → SimulationSpec (T1.4). The example tests skip when reference/ is missing.
using ModflowGPU: load_simulation, SimulationSpec, GwfSpec, DisSpec, DisvSpec, DisuSpec, NpfSpec, StoSpec,
    IcSpec, OcSpec, BoundarySpec, boundary_kind, ImsSpec, ims_defaults, ExchangeSpec, TdisSpec, nper,
    model_spec, is_transient, MF6UnsupportedError, MF6InputError, read_options, read_input_file, find_block,
    dfn_definition, option, has_option, PeriodList, ArrayPeriod, TsRef, grid_dims, nodes_user, idomain_user,
    list_column, period_data, PrintFormat, SUPPORTED_FEATURES

"The error `f()` throws, which must be an `MF6UnsupportedError`."
function unsupported_error_of(f)
    try
        f()
    catch e
        e isa MF6UnsupportedError && return e
        rethrow()
    end
    @test false
    return nothing
end

# A small simulation with the coupling feature set (PLAN.md section 1.1): DIS 2×3×4 with
# one inactive cell, confined NPF, STO, IC, OC, CHD, WEL (TS6), RIV (AUXMULTNAME), DRN,
# RCH, EVT (FIXED_CELL), IMS COMPLEX/BICGSTAB/SIMPLE.
const BASE_SIM = Dict(
    "mfsim.nam" => """
        BEGIN options
        END options
        BEGIN timing
          TDIS6  sim.tdis
        END timing
        BEGIN models
          gwf6  gwf.nam  Gwf
        END models
        BEGIN exchanges
        END exchanges
        BEGIN solutiongroup  1
          ims6  sim.ims  gwf
        END solutiongroup  1
        """,
    "sim.tdis" => """
        BEGIN options
          TIME_UNITS  days
        END options
        BEGIN dimensions
          NPER  2
        END dimensions
        BEGIN perioddata
          1.0  1  1.0
          10.0  5  1.2
        END perioddata
        """,
    "gwf.nam" => """
        BEGIN options
          SAVE_FLOWS
        END options
        BEGIN packages
          DIS6  gwf.dis  dis
          NPF6  gwf.npf
          IC6  gwf.ic
          STO6  gwf.sto
          CHD6  gwf.chd
          WEL6  gwf.wel  wel_a
          WEL6  gwf2.wel
          RIV6  gwf.riv
          DRN6  gwf.drn
          RCH6  gwf.rch
          EVT6  gwf.evt
          OC6  gwf.oc
        END packages
        """,
    "gwf.dis" => """
        BEGIN options
          LENGTH_UNITS meters
          NOGRB
          XORIGIN 100.0
        END options
        BEGIN dimensions
          NLAY 2
          NROW 3
          NCOL 4
        END dimensions
        BEGIN griddata
          delr
            CONSTANT 10.0
          delc
            INTERNAL
            1.0 2.0 3.0
          top
            CONSTANT 50.0
          botm LAYERED
            CONSTANT 20.0
            CONSTANT 0.0
          idomain
            INTERNAL
            0 1 1 1
            1 1 1 1
            1 1 1 -1
            2 1 1 1
            1 1 1 1
            1 1 1 1
        END griddata
        """,
    "gwf.npf" => """
        BEGIN options
          SAVE_SPECIFIC_DISCHARGE
        END options
        BEGIN griddata
          icelltype
            CONSTANT 0
          k
            CONSTANT 5.0
          k33
            CONSTANT 0.5
        END griddata
        """,
    "gwf.ic" => "BEGIN griddata\n  strt\n    CONSTANT 30.0\nEND griddata\n",
    "gwf.sto" => """
        BEGIN options
          STORAGECOEFFICIENT
        END options
        BEGIN griddata
          iconvert
            CONSTANT 0
          ss
            CONSTANT 1.0e-3
        END griddata
        BEGIN period 1
          STEADY-STATE
        END period
        BEGIN period 2
          TRANSIENT
        END period
        """,
    "gwf.chd" => "BEGIN dimensions\n  MAXBOUND 2\nEND dimensions\nBEGIN period 1\n  1 1 2 40.0\n  2 3 4 35.0\nEND period\n",
    "gwf.wel" => """
        BEGIN options
          BOUNDNAMES
          TS6 FILEIN gwf.wel.ts
        END options
        BEGIN dimensions
          MAXBOUND 1
        END dimensions
        BEGIN period 2
          2 2 2 -2.5 well1
        END period
        """,
    "gwf.wel.ts" => "BEGIN attributes\n  NAME pumping\n  METHOD stepwise\nEND attributes\nBEGIN timeseries\n  0.0 -1.0\nEND timeseries\n",
    "gwf2.wel" => "BEGIN dimensions\n  MAXBOUND 1\nEND dimensions\nBEGIN period 1\n  2 1 1 -5.0\nEND period\n",
    "gwf.riv" => """
        BEGIN options
          AUXILIARY width mult
          AUXMULTNAME mult
          SAVE_FLOWS
        END options
        BEGIN dimensions
          MAXBOUND 1
        END dimensions
        BEGIN period 1
          1 2 2 45.0 100.0 40.0 5.0 0.5
        END period
        """,
    "gwf.drn" => "BEGIN dimensions\n  MAXBOUND 1\nEND dimensions\nBEGIN period 1\n  1 3 3 48.0 10.0\nEND period\n",
    "gwf.rch" => "BEGIN dimensions\n  MAXBOUND 2\nEND dimensions\nBEGIN period 1\n  1 2 1 0.001\n  1 2 3 0.002\nEND period\n",
    "gwf.evt" => """
        BEGIN options
          FIXED_CELL
        END options
        BEGIN dimensions
          MAXBOUND 1
          NSEG 1
        END dimensions
        BEGIN period 1
          1 3 1 50.0 0.002 2.0
        END period
        """,
    "gwf.oc" => """
        BEGIN options
          HEAD FILEOUT gwf.hds
          BUDGET FILEOUT gwf.cbc
          HEAD PRINT_FORMAT COLUMNS 10 WIDTH 15 DIGITS 6 GENERAL
        END options
        BEGIN period 1
          PRINT BUDGET ALL
          SAVE HEAD ALL
        END period
        """,
    "sim.ims" => """
        BEGIN options
          PRINT_OPTION SUMMARY
          COMPLEXITY COMPLEX
        END options
        BEGIN nonlinear
          OUTER_DVCLOSE 5.0
          OUTER_MAXIMUM 20
          UNDER_RELAXATION SIMPLE
          UNDER_RELAXATION_GAMMA 0.9
        END nonlinear
        BEGIN linear
          INNER_MAXIMUM 100
          INNER_DVCLOSE 0.01
          INNER_RCLOSE 10.0 STRICT
          LINEAR_ACCELERATION BICGSTAB
          RELAXATION_FACTOR 0.97
        END linear
        """,
)

"Write the simulation `BASE_SIM` with `changes` (file name => text, or `nothing` to drop a file) into `dir`."
function write_sim(dir, changes = Dict{String, Any}())
    files = merge(BASE_SIM, changes)
    for (name, text) in files
        isnothing(text) || write(joinpath(dir, name), text)
    end
    return dir
end

"Load a variant of `BASE_SIM` in a fresh directory."
function load_variant(changes = Dict{String, Any}(); strict = true)
    return mktempdir() do dir
        load_simulation(write_sim(dir, changes); strict)
    end
end

"Replace `old` by `new` in base file `name`."
edit(name, old, new) = Dict{String, Any}(name => replace(BASE_SIM[name], old => new))

@testset "Simulation loader" begin
    @testset "read_options (DFN records)" begin
        mktempdir() do dir
            f = read_input_file(
                write_input(
                    dir, "a.npf",
                    "BEGIN options\n  REWET WETFCT 1.5 IWETIT 2 IHDWET 1\n  XT3D RHS\n  VARIABLECV\n  ALTERNATIVE_CELL_AVERAGING amt-lmk\nEND options\nBEGIN griddata\n  icelltype\n    CONSTANT 0\n  k\n    CONSTANT 1.0\nEND griddata\n",
                ); dfn = "gwf-npf",
            )
            o = read_options(f, "OPTIONS")
            @test o.tags == ["REWET", "XT3D", "VARIABLECV", "ALTERNATIVE_CELL_AVERAGING"]
            @test option(o, "rewet", false) && option(o, "wetfct", 0.0) == 1.5 && option(o, "iwetit", 0) == 2 && option(o, "ihdwet", 0) == 1
            @test option(o, "xt3d", false) && option(o, "rhs", false)
            @test option(o, "variablecv", false) && !has_option(o, "dewatered")
            @test option(o, "alternative_cell_averaging", "") == "AMT-LMK"
            # OC: two records start with HEAD; the second word selects one.
            f = read_input_file(write_input(dir, "a.oc", "BEGIN options\n  HEAD PRINT_FORMAT COLUMNS 5 WIDTH 12 DIGITS 4 scientific\n  head fileout My.HDS\nEND options\nBEGIN period 1\nEND period\n"); dfn = "gwf-oc")
            o = read_options(f, "OPTIONS")
            @test option(o, "columns", 0) == 5 && option(o, "width", 0) == 12 && option(o, "digits", 0) == 4
            @test option(o, "format", "") == "SCIENTIFIC" && option(o, "headfile", "") == "My.HDS"
            # Several TS6 files; AUX alias.
            f = read_input_file(write_input(dir, "a.wel", "BEGIN options\n  AUX a b\n  TS6 FILEIN One.ts\n  TS6 FILEIN two.ts\nEND options\nBEGIN dimensions\n MAXBOUND 1\nEND dimensions\n"); dfn = "gwf-wel")
            o = read_options(f, "OPTIONS")
            @test option(o, "ts6_filename", String[]) == ["One.ts", "two.ts"] && option(o, "auxiliary", String[]) == ["A", "B"]
            # Errors: unknown tag, bad record, invalid value.
            for (text, msg) in (
                    ("FOO", "Input file tag not found: \"FOO\" in block \"OPTIONS\"."),
                    ("TS6 ex.ts", "Expecting record input tag \"FILEIN\" but instead found \"EX.TS\"."),
                    ("TS6", "block input record that includes keyword \"TS6\" is not properly formed"),
                )
                f = read_input_file(write_input(dir, "b.wel", "BEGIN options\n  $text\nEND options\nBEGIN dimensions\n MAXBOUND 1\nEND dimensions\n"); dfn = "gwf-wel")
                e = input_error_of(() -> read_options(f, "OPTIONS"))
                @test e.line == 2 && occursin(msg, e.msg)
            end
            f = read_input_file(write_input(dir, "b.npf", "BEGIN options\n  ALTERNATIVE_CELL_AVERAGING geometric\nEND options\nBEGIN griddata\nEND griddata\n"); dfn = "gwf-npf")
            @test occursin("Unrecognized input value for ALTERNATIVE_CELL_AVERAGING option.", input_error_of(() -> read_options(f, "OPTIONS")).msg)
        end
    end

    @testset "IMS defaults" begin
        s = ims_defaults(:simple)
        @test (s.outer_dvclose, s.outer_maximum, s.under_relaxation, s.inner_maximum, s.linear_acceleration) == (1.0e-3, 25, :none, 50, :cg)
        m = ims_defaults(:moderate)
        @test (m.outer_maximum, m.under_relaxation, m.under_relaxation_theta, m.relaxation_factor, m.linear_acceleration) == (50, :dbd, 0.9, 0.97, :bicgstab)
        c = ims_defaults(:complex)
        @test (c.outer_maximum, c.backtracking_number, c.inner_maximum, c.preconditioner_levels, c.number_orthogonalizations) == (100, 20, 500, 5, 2)
        @test c.inner_rclose == 0.1 && c.rclose_option === :infinity_norm
    end

    @testset "coupling feature set (strict)" begin
        mktempdir() do dir
            s = load_simulation(write_sim(dir))
            @test s isa SimulationSpec && s.simdir == abspath(dir)
            @test s.tdis.perlen == [1.0, 10.0] && s.tdis.nstp == Int32[1, 5] && s.tdis.tsmult == [1.0, 1.2]
            @test s.tdis.time_units === :days && nper(s.tdis) == 2
            m = model_spec(s, "gwf")
            @test m.name == "GWF" && m.save_flows && m.file == "gwf.nam"
            d = m.dis
            @test d isa DisSpec && (d.nlay, d.nrow, d.ncol) == (2, 3, 4)
            @test d.delr == fill(10.0, 4) && d.delc == [1.0, 2.0, 3.0] && size(d.top) == (4, 3) && size(d.botm) == (4, 3, 2)
            @test d.botm[:, :, 1] == fill(20.0, 4, 3) && d.idomain[1, 1, 1] == 0 && sum(d.idomain) == 22 && d.idomain[4, 3, 1] == -1 && d.idomain[1, 1, 2] == 2
            @test d.length_units === :meters && d.nogrb && d.xorigin == 100.0 && d.yorigin == 0.0
            @test nodes_user(d) == 24 && grid_dims(d).mshape == [2, 3, 4]
            @test m.npf.k == fill(5.0, 24) && m.npf.k33 == fill(0.5, 24) && isnothing(m.npf.k22) && m.npf.cell_averaging === :harmonic
            @test m.npf.save_specific_discharge
            @test m.ic.strt == fill(30.0, 24)
            @test m.sto.storagecoefficient && m.sto.ss == fill(1.0e-3, 24) && isnothing(m.sto.sy)
            @test !is_transient(m.sto, 1) && is_transient(m.sto, 2) && !is_transient(nothing, 1)
            @test m.oc.head_file == "gwf.hds" && m.oc.budget_file == "gwf.cbc"
            @test m.oc.head_print_format == PrintFormat(10, 15, 6, "GENERAL")
            @test length(period_data(m.oc.periods, 2)) == 2
            # Stress packages in MF6's order (GWF_MULTIPKG), default names WEL-2, CHD-1, ...
            @test [b.name for b in m.boundaries] == ["WEL_A", "WEL-2", "DRN-1", "RIV-1", "RCH-1", "EVT-1", "CHD-1"]
            @test [boundary_kind(b) for b in m.boundaries] == [:WEL, :WEL, :DRN, :RIV, :RCH, :EVT, :CHD]
            wel = m.boundaries[1]
            @test wel.boundnames && wel.ts_files == ["gwf.wel.ts"] && wel.periods.iper == [2]
            p = wel.periods.data[1]
            @test p.nodeu == Int32[12 + 4 + 2] && p.bound == fill(-2.5, 1, 1) && isempty(p.ts) && p.boundname == ["WELL1"]
            riv = m.boundaries[4]
            @test riv.auxnames == ["WIDTH", "MULT"] && riv.auxmultname == "MULT" && riv.auxmultcol == 2 && riv.save_flows
            @test riv.periods.data[1].bound == reshape([45.0, 100.0, 40.0], 3, 1) && riv.periods.data[1].auxvar == reshape([5.0, 0.5], 2, 1)
            @test list_column(riv.schema, "rbot").first == 3
            evt = m.boundaries[6]
            @test evt.fixed_cell && evt.nseg == 1 && evt.maxbound == 1 && !evt.surf_rate_specified
            chd = m.boundaries[7]
            @test chd.periods.data[1].nodeu == Int32[2, 12 + 8 + 4] && chd.periods.data[1].bound == [40.0 35.0]
            @test isempty(s.exchanges) && length(s.solution_groups) == 1 && s.solution_groups[1].mxiter == 1
            ims = s.solution_groups[1].solutions[1]
            @test ims.models == ["GWF"] && ims.complexity === :complex && ims.print_option === :summary
            @test ims.outer_dvclose == 5.0 && ims.outer_maximum == 20 && ims.under_relaxation === :simple && ims.under_relaxation_gamma == 0.9
            @test ims.inner_rclose == 10.0 && ims.rclose_option === :strict && ims.linear_acceleration === :bicgstab
            @test ims.relaxation_factor == 0.97 && ims.inner_maximum == 100 && ims.backtracking_number == 20  # COMPLEX default
            @test "INNER_RCLOSE" in ims.given && !("BACKTRACKING_NUMBER" in ims.given)
        end
    end

    @testset "period blocks after NPER are not read" begin
        changes = Dict{String, Any}("gwf.chd" => BASE_SIM["gwf.chd"] * "BEGIN period 3\n  9 9 9 1.0\nEND period\n")
        s = @test_logs (:warn, r"PERIOD 3 is ignored: the simulation has NPER = 2") match_mode = :any load_variant(changes)
        @test s.models[1].boundaries[end].periods.iper == [1]
    end

    @testset "input errors" begin
        mktempdir() do dir
            @test occursin("simulation directory not found", input_error_of(() -> load_simulation(joinpath(dir, "nope"))).msg)
            @test occursin("file not found", input_error_of(() -> load_simulation(dir)).msg)  # no mfsim.nam
        end
        cases = [
            # MF6 cellid_to_nlist: list cellid in an inactive cell (IDOMAIN 0).
            (edit("gwf.chd", "1 1 2 40.0", "1 1 1 40.0"), "Cell is outside active grid domain: (1,1,1)", "gwf.chd", 5),
            # IDOMAIN -1 (vertical pass-through) is not an active cell either.
            (edit("gwf.chd", "1 1 2 40.0", "1 3 4 40.0"), "Cell is outside active grid domain: (1,3,4)", "gwf.chd", 5),
            (Dict{String, Any}("gwf.drn" => nothing), "DRN6 input file not found", "gwf.drn", 0),
            (edit("gwf.nam", "  IC6  gwf.ic\n", ""), "Initial Conditions (IC6) package not specified.", "gwf.nam", 0),
            (edit("gwf.nam", "  DIS6  gwf.dis  dis\n", ""), "Discretization package not specified for model \"GWF\".", "gwf.nam", 0),
            (edit("gwf.nam", "  NPF6  gwf.npf\n", "  NPF6  gwf.npf\n  NPF6  gwf.npf\n"), "Multiple instances specified for model base package type [model=GWF, type=NPF6].", "gwf.nam", 0),
            (edit("gwf.nam", "  OC6", "  XYZ6  a.xyz\n  OC6"), "Model package type not supported [model=GWF, type=XYZ6].", "gwf.nam", 0),
            (edit("gwf.nam", "gwf2.wel", "gwf2.wel wel_a"), "Package name  already exists: WEL_A", "gwf.nam", 0),
            (edit("mfsim.nam", "ims6  sim.ims  gwf", "ims6  sim.ims  other"), "Invalid model name: OTHER", "mfsim.nam", 0),
            (edit("mfsim.nam", "solutiongroup  1\n  ims6", "solutiongroup  1\n  MXITER 3\n  ims6"), "MXITER is set to 3 but there is only one solution", "mfsim.nam", 0),
            (edit("mfsim.nam", "  ims6  sim.ims  gwf\n", ""), "There are no solutions for solution group 1", "mfsim.nam", 0),
            (edit("mfsim.nam", "gwf6  gwf.nam  Gwf", "gwf6  gwf.nam  averyveryverylongname"), "exceeds maximum length of 16", "mfsim.nam", 0),
            (edit("sim.tdis", "10.0  5  1.2", "10.0  0  1.2"), "Number of time steps less than one  for stress period 2", "sim.tdis", 0),
            (edit("sim.tdis", "10.0  5  1.2", "10.0  5  -1.0"), "TSMULT must be greater than 0.0  for stress period 2", "sim.tdis", 0),
            (edit("sim.tdis", "days", "fortnights"), "Unrecognized input value for TIME_UNITS option.", "sim.tdis", 2),
            (edit("gwf.npf", "  k\n", "  kk\n"), "input tag \"KK\" not found in block GRIDDATA", "gwf.npf", 0),
            (edit("gwf.npf", "  k33\n    CONSTANT 0.5\n", ""), "", "gwf.npf", -1),  # fine: no K33
            (edit("gwf.npf", "SAVE_SPECIFIC_DISCHARGE", "K22OVERK"), "K22OVERK option specified but K22 not specified.", "gwf.npf", 0),
            (edit("gwf.sto", "  ss\n    CONSTANT 1.0e-3\n", ""), "Error in GRIDDATA block: SS not found.", "gwf.sto", 0),
            (edit("gwf.sto", "CONSTANT 1.0e-3", "CONSTANT -1.0e-3"), "Error in SS DATA: SS value in cell (1,1,2) is less than zero", "gwf.sto", 0),
            (edit("gwf.oc", "SAVE HEAD ALL", "SAVE CONCENTRATION ALL"), "Input OC period block rtype not found:  \"CONCENTRATION\".", "gwf.oc", 8),
            (edit("gwf.riv", "AUXMULTNAME mult", "AUXMULTNAME factor"), "AUXMULTNAME was specified as FACTOR but no AUX variable found with this name.", "gwf.riv", 3),
            (edit("sim.ims", "UNDER_RELAXATION_GAMMA 0.9", "UNDER_RELAXATION_GAMMA 0.0"), "GAMMA must be greater than zero if SIMPLE under-relaxation is used.", "sim.ims", 0),
            (edit("sim.ims", "RELAXATION_FACTOR 0.97", "RELAX 0.97"), "Unknown IMSLINEAR keyword (RELAX).", "sim.ims", 16),
            (edit("sim.ims", "COMPLEXITY COMPLEX", "COMPLEXITY HARD"), "Unknown IMS COMPLEXITY option (HARD).", "sim.ims", 3),
        ]
        for (changes, msg, file, line) in cases
            if line == -1
                @test load_variant(changes) isa SimulationSpec
                continue
            end
            e = input_error_of(() -> load_variant(changes))
            ok = occursin(msg, e.msg) && basename(e.file) == file && (line == 0 || e.line == line)
            ok || @info "unexpected error" msg file line e
            @test ok
        end
        # The model is in no solution.
        changes = merge(
            edit("mfsim.nam", "gwf6  gwf.nam  Gwf\n", "gwf6  gwf.nam  Gwf\n  gwf6  gwf.nam  second\n"),
        )
        # T6.6: several models are supported (strict and not).
        for strict in (true, false)
            e = input_error_of(() -> load_variant(changes; strict))
            @test occursin("Model was not assigned to a solution: SECOND", e.msg)
        end
    end

    @testset "unsupported input" begin
        # Out of scope: always an error, also with strict = false.
        for strict in (true, false)
            e = unsupported_error_of(() -> load_variant(edit("gwf.nam", "  OC6", "  SFR6  gwf.sfr  sfr\n  OC6"); strict))
            @test e.msg == "unsupported: SFR6 package 'SFR' in model 'GWF'" && basename(e.file) == "gwf.nam"
            e = unsupported_error_of(() -> load_variant(edit("mfsim.nam", "END models", "  gwt6  t.nam  trans\nEND models"); strict))
            @test startswith(e.msg, "unsupported: GWT6 model 'TRANS'")
            e = unsupported_error_of(() -> load_variant(edit("gwf.riv", "SAVE_FLOWS", "MOVER"); strict))
            @test startswith(e.msg, "unsupported: RIV6 option MOVER")
        end
        # Not implemented yet: rejected with strict = true, read with strict = false.
        for (changes, msg, check) in (
                # XT3D: supported since T6.5.
                (edit("gwf.npf", "SAVE_SPECIFIC_DISCHARGE", "XT3D  RHS"), nothing, s -> s.models[1].npf.xt3d && s.models[1].npf.xt3d_rhs),
                # Convertible cells, convertible storage, AUTO_FLOW_REDUCE: supported since T6.1.
                (edit("gwf.npf", "icelltype\n    CONSTANT 0", "icelltype\n    CONSTANT 1"), nothing, s -> all(==(1), s.models[1].npf.icelltype)),
                (
                    edit("gwf.sto", "iconvert\n    CONSTANT 0\n", "iconvert\n    CONSTANT 1\n  sy\n    CONSTANT 0.2\n"),
                    nothing, s -> s.models[1].sto.sy == fill(0.2, 24),
                ),
                # Newton–Raphson: supported since T6.2 (BASE_SIM uses BICGSTAB).
                (edit("gwf.nam", "SAVE_FLOWS", "NEWTON UNDER_RELAXATION"), nothing, s -> s.models[1].newton && s.models[1].newton_under_relaxation),
                (edit("gwf.oc", "BUDGET FILEOUT gwf.cbc", "BUDGETCSV FILEOUT gwf.csv"), "unsupported: OC6 option BUDGETCSV", s -> s.models[1].oc.budgetcsv_file == "gwf.csv"),
                (edit("gwf.wel", "BOUNDNAMES", "BOUNDNAMES\n  AUTO_FLOW_REDUCE 0.1"), nothing, s -> s.models[1].boundaries[1].auto_flow_reduce == 0.1),
                (edit("sim.ims", "RELAXATION_FACTOR 0.97", "SCALING_METHOD L2NORM"), nothing, s -> s.solution_groups[1].solutions[1].scaling_method === :l2norm),
                (edit("sim.ims", "OUTER_MAXIMUM 20", "BACKTRACKING_NUMBER 5"), nothing, s -> s.solution_groups[1].solutions[1].backtracking_number == 5),
                # Time series: supported since T6.3.
                (
                    edit("gwf.wel", "2 2 2 -2.5 well1", "2 2 2 PUMPING well1"), nothing,
                    s -> s.models[1].boundaries[1].periods.data[1].ts == [TsRef(1, 1, false, :PUMPING)] && isnan(s.models[1].boundaries[1].periods.data[1].bound[1, 1]) &&
                        s.models[1].boundaries[1].timeseries[:PUMPING].values == [-1.0],
                ),
                (edit("sim.tdis", "TIME_UNITS  days", "TIME_UNITS  days\n  START_DATE_TIME 2020-01-01"), nothing, s -> s.tdis.start_date_time == "2020-01-01"),
            )
            if isnothing(msg)
                @test check(load_variant(changes))
            else
                e = unsupported_error_of(() -> load_variant(changes))
                ok = startswith(e.msg, msg)
                ok || @info "unexpected message" msg e.msg
                @test ok
                @test check(load_variant(changes; strict = false))
            end
        end
        # A feature added to SUPPORTED_FEATURES is accepted (how later cards extend the table).
        # (T6.1: :convertible is in the table; without it the model is rejected.)
        changes = edit("gwf.npf", "icelltype\n    CONSTANT 0", "icelltype\n    CONSTANT 1")
        @test load_variant(changes) isa SimulationSpec
        delete!(SUPPORTED_FEATURES, :convertible)
        try
            @test startswith(unsupported_error_of(() -> load_variant(changes)).msg, "unsupported: convertible cells")
        finally
            push!(SUPPORTED_FEATURES, :convertible)
        end
    end

    # MF6 examples (validation/examples.toml; reference/ from validation/fetch_reference.sh).
    examples = Dict(in_scope_examples())
    if isempty(examples)
        @info "Skipping the simulation loader example tests: $(joinpath(REFERENCE_DIR, "mf6examples")) not found (validation/fetch_reference.sh)"
    else
        exdir = joinpath(REFERENCE_DIR, "mf6examples")
        @testset "Tier 0-2 examples load (strict)" begin
            for name in ("ex-gwf-toth", "ex-gwf-hanic", "ex-gwf-hanir", "ex-gwf-hwell-a")
                @test load_simulation(examples[name]) isa SimulationSpec
            end
            # Spot checks against the input files.
            s = load_simulation(examples["ex-gwf-toth"])
            d = s.models[1].dis
            @test (d.nlay, d.nrow, d.ncol) == (200, 1, 100) && d.delr == fill(200.0, 100)
            @test [b.name for b in s.models[1].boundaries] == ["CHD_0"]
            s = load_simulation(examples["ex-gwf-hanir"])
            npf = s.models[1].npf
            @test npf.k22 == fill(0.01, nodes_user(s.models[1].dis)) && npf.angle1 == zeros(nodes_user(s.models[1].dis))
            @test isnothing(npf.k33) && npf.save_specific_discharge
            # fhb uses TS6 values (strict loading since T6.3).
            s = load_simulation(examples["ex-gwf-fhb"])
            @test s.tdis.perlen == [400.0, 200.0, 400.0] && s.tdis.nstp == Int32[10, 4, 6] && s.tdis.time_units === :days
            m = s.models[1]
            @test [b.name for b in m.boundaries] == ["WEL", "CHD"]
            @test m.boundaries[1].ts_files == ["ex-gwf-fhb.wel.ts"] && m.boundaries[2].ts_files == ["ex-gwf-fhb.chd.ts"]
            @test m.obs_files == ["ex-gwf-fhb.obs"]
            chd = m.boundaries[2].periods.data[1]
            @test chd.ts == [TsRef(i, 1, false, :CHDHEAD) for i in 1:3] && all(isnan, chd.bound)
            ts = m.boundaries[2].timeseries[:CHDHEAD]
            @test ts.method === :linearend && ts.times == [0.0, 307.0, 791.0, 1000.0] && ts.values == [0.0, 1.0, 5.0, 2.0]
            @test m.sto.storagecoefficient && is_transient(m.sto, 3)
            ims = s.solution_groups[1].solutions[1]
            @test ims.inner_rclose == 1.0e-6 && ims.rclose_option === :strict && ims.outer_dvclose == 1.0e-9
        end
        @testset "all in-scope examples load (strict = false)" begin
            for (name, simdir) in sort!(collect(examples))
                @testset "$name" begin
                    @test load_simulation(simdir; strict = false) isa SimulationSpec
                end
            end
            s = load_simulation(examples["ex-gwf-twri01"]; strict = false)
            m = s.models[1]
            @test m.npf.variablecv && m.npf.dewatered && m.npf.perched && m.npf.save_specific_discharge
            @test [b.name for b in m.boundaries] == ["WEL_0", "DRN_0", "RCHA_0", "CHD_0"]
            @test m.boundaries[3].readasarrays && m.boundaries[3].periods.data[1] isa ArrayPeriod
            @test s.tdis.time_units === :seconds
            s = load_simulation(examples["ex-gwf-u1gwfgwf-s3"]; strict = false)
            @test [m.name for m in s.models] == ["OUTER", "INNER"] && s.models[1].dis isa DisSpec
            x = s.exchanges[1]
            @test (x.model1, x.model2) == ("OUTER", "INNER") && x.xt3d && length(x.data.nodem1) > 0
            @test s.solution_groups[1].solutions[1].models == ["OUTER", "INNER"]
            @test load_simulation(examples["ex-gwf-rad-disu"]; strict = false).models[1].dis isa DisuSpec
            d = load_simulation(examples["ex-gwf-u1disv"]; strict = false).models[1].dis
            @test d isa DisvSpec && (d.ncpl, d.nvert) == (121, 148) && length(d.cell_vertices) == 121
        end
        @testset "in-scope examples with strict = true" begin
            # T6.1: the convertible examples load (twri01, advtidal, bcf2ss-p01a, ...); T6.6:
            # the multi-model examples (Tier 7) too. Nothing in scope fails any more.
            expected = Dict{String, String}()
            for (name, simdir) in sort!(collect(examples))
                @testset "$name" begin
                    if haskey(expected, name)
                        e = unsupported_error_of(() -> load_simulation(simdir))
                        @test startswith(e.msg, "unsupported: " * expected[name])
                    else
                        # Tier 0-2 without higher features, the confined Tier 3 size tests and
                        # the DISV examples without XT3D (T6.4).
                        @test name in (
                            "ex-gwf-toth", "ex-gwf-hanic", "ex-gwf-hanir", "ex-gwf-hwell-a", "ex-gwf-fhb", "ex-gwf-lgrv-gc", "ex-gwf-lgrv-gr",
                            "ex-gwf-u1disv", "ex-gwf-curve-90", "ex-gwf-curvilin",
                            # T6.1: convertible cells
                            "ex-gwf-advtidal", "ex-gwf-twri01", "ex-gwf-bcf2ss-p01a", "ex-gwf-bump-p01b", "ex-gwf-nwt-p02b",
                            "ex-gwf-maw-p03a", "ex-gwf-maw-p03c", "ex-gwf-rad-disu",
                            # T6.2: Newton–Raphson
                            "ex-gwf-bcf2ss-p02a", "ex-gwf-bump-p01a", "ex-gwf-bump-p01c", "ex-gwf-capture",
                            "ex-gwf-nwt-p02a", "ex-gwf-nwt-p03a", "ex-gwf-nwt-p03b", "ex-gwf-zaidel",
                            # T6.5: XT3D
                            "ex-gwf-hanix", "ex-gwf-whirl", "ex-gwf-u1disv-x", "ex-gwf-disvmesh",
                            # T6.6: several models, GWF-GWF exchanges
                            "ex-gwf-lgrv-lgr", "ex-gwf-spbc", "ex-gwf-u1gwfgwf-s1", "ex-gwf-u1gwfgwf-s2",
                            "ex-gwf-u1gwfgwf-s3", "ex-gwf-u1gwfgwf-s4",
                        )
                        @test load_simulation(simdir) isa SimulationSpec
                    end
                end
            end
        end
        @testset "out-of-scope examples name the package or model" begin
            for strict in (true, false)
                e = unsupported_error_of(() -> load_simulation(joinpath(exdir, "ex-gwf-drn-p01a"); strict))
                @test e.msg == "unsupported: SFR6 package 'SFR-1' in model 'EX-GWF-DRN-P01'"
                e = unsupported_error_of(() -> load_simulation(joinpath(exdir, "ex-gwt-henry-a"); strict))
                @test startswith(e.msg, "unsupported: GWT6 model 'TRANS'")
            end
        end
    end
end
