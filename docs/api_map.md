# API map of `src/`

Update this file in every card's merge; skip `src/input/dfn/definitions.jl` (generated).

One entry per file, in include order (`src/ModflowGPU.jl`): purpose, key types and
functions, and the test groups (`MFGPU_TEST_GROUPS`, see `test/runtests.jl`) that cover
it. Underscore-prefixed helpers are omitted.

## Backends and precision

- **backend_setup.jl**: backend registry, adapted from mGV. `select_backend` (name → KA
  backend), `register_backend!` (called by `ext/*Ext.jl`), `backend_options`,
  `backend_name`, `known_backend_package`, `pin_memory!` (extension hook), `BackendEntry`, `BACKEND_REGISTRY`, `KNOWN_BACKENDS`. Tests: backend.
- **ext/*Ext.jl** (AMDGPU, CUDA, Metal, oneAPI, OpenCL): `__init__` registers the backend;
  overrides `supports_float64` where needed. Tests: backend (GPU file with
  `MFGPU_TEST_BACKEND`).
- **precision.jl**: precision presets (PLAN.md 3.7). `resolve_precision(backend, preset)`
  → `(; T, Ts, Tp)`, `PRECISION_PRESETS`, `supports_float64` (extension hook). Tests: precision, kernel.

## MF6 input (host)

- **input/tokenizer.jl**: MF6 line and word rules (URWORD). `InputLine`, `MF6InputError`,
  `input_error`, `read_input_lines`, `is_comment_line`, `tokenize_line`, `tokens` /
  `ntokens` / `token` / `eachtoken` (non-allocating words), `keyword`, `parse_double` /
  `parse_int`, `get_double` / `get_int`, `each_double`. Tests: input_tokenizer.
- **input/dfn/schema.jl**: DFN schema types. `DfnVariable`, `DfnBlock`, `DfnDefinition`,
  `DFN_DEFINITIONS` (filled by the generated `definitions.jl`), `dfn_definition`,
  `dfn_block`, `dfn_variable`, `dfn_name`, `dfn_tag`. Tests: input_blocks.
- **input/dfn/generate.jl**: script that writes `definitions.jl` from the MF6 DFN files
  (`main`, `parse_dfn`); not loaded by the package.
- **input/blocks.jl**: block parser (BlockParser.f90), OPEN/CLOSE, plans of which blocks a
  file has. `InputBlock`, `InputFile`, `parse_blocks`, `read_input_file`, `find_block`,
  `find_blocks`, `block_number`, `ReadPlan` / `read_plan`, `select_blocks`. Tests:
  input_blocks, input_examples.
- **input/arrays.jl**: READARRAY (ArrayReaderBase.f90): control records, CONSTANT /
  INTERNAL / OPEN/CLOSE, LAYERED, binary arrays. `GridDims` (`dis_dims`, `disv_dims`,
  `disu_dims`, `grid_dims`), `array_size`, `layered_shape`, `ArrayControl`,
  `read_control_record`, `list_read!`, `read_array!`, `read_grid_array`,
  `read_array_block`, `BinaryArrayHeader`, `read_binary_array` / `write_binary_array`.
  Tests: input_arrays.
- **input/lists.jl**: list and period input (IDM): cellids, boundnames, time-series refs,
  binary lists, OC/STO/TDIS period records, exchanges. `ListColumn`, `ListSchema` /
  `list_schema`, `PackageOptions` / `package_options`, `cellid`, `user_node`,
  `valid_cellid`, `PeriodList`, `TsRef`, `StressPeriods`, `ArrayPeriod`,
  `read_list_block`, `read_list_periods`, `read_array_periods`, `read_period_data`,
  `read_oc_periods` (`OcRecord`), `read_sto_periods`, `read_tdis_perioddata`,
  `read_records`, `ExchangeData` / `read_exchange_data` / `exchange_schema`,
  `write_binary_list`. Tests: input_lists.
- **input/options.jl**: OPTIONS / DIMENSIONS / TIMING blocks through the DFN schema.
  `OptionSet`, `read_options`, `option`, `has_option`, `option_line`. Tests:
  simulation_loader.
- **input/supported.jl**: what the engine supports; the loader checks input against it.
  `MF6UnsupportedError`, `SUPPORTED_OPTIONS`, `SUPPORTED_VALUES`, `SUPPORTED_FEATURES`,
  `UNSUPPORTED_PACKAGES` / `UNSUPPORTED_MODELS`, `check_options`, `check_value`,
  `check_feature`. Tests: simulation_loader, input_lists.

## Model description (host specs, plain structs)

- **spec/grid.jl**: discretization specs (PLAN.md 3.2). `AbstractDisSpec`, `DisSpec`,
  `DisvSpec`, `DisuSpec`, `grid_dims`, `nodes_user`, `idomain_user`. Tests:
  simulation_loader, connectivity.
- **spec/packages.jl**: GWF package specs. `NpfSpec`, `StoSpec` (`is_transient`), `IcSpec`,
  `OcSpec`, `PrintFormat`, `BoundarySpec{Kind}`, `BOUNDARY_KINDS`. Tests:
  simulation_loader, npf.
- **spec/simulation.jl**: simulation-level specs. `TdisSpec` (`nper`), `ImsSpec`
  (`ims_defaults`), `SolutionGroupSpec`, `PackageEntry`, `GwfSpec`, `ExchangeSpec`,
  `SimulationSpec`, `model_spec`. Tests: simulation_loader.
- **input/simulation.jl**: simulation loader, mfsim.nam → `SimulationSpec` (TDIS, IMS,
  GWF name files, DIS/DISV/DISU, IC, NPF, STO, OC, boundaries, exchanges).
  `load_simulation`, `LENMODELNAME`, `LENPACKAGENAME`. Tests: simulation_loader (examples
  need `reference/`).

## Discretization and connectivity (host)

- **discretization/connectivity.jl**: MF6 CSR connectivity layout (PLAN.md 3.3).
  `Connectivity` (ia/ja/jas/isym/ihc, cl1/cl2/hwva/anglex, top/bot/area/xc/yc, reduced
  numbering), `build_connectivity`, `reduced_numbering`, `fill_isym!`, `fill_jas!`,
  `iajausr`, `getjaindex`, `connection_lengths`. Tests: connectivity.
- **discretization/dis.jl**: DIS → `Connectivity` (Dis.f90): `build_connectivity(::DisSpec)`
  with IDOMAIN 0 / pass-through cells, `check_dis_grid`. Tests: connectivity (parity vs
  libmf6 and `.grb`).

## GWF engine

- **gwf/offsets.jl**: heads as offsets from per-cell references (PLAN.md 3.7 rule 1).
  `reference_heads`, `reference_differences`, `reference_difference` (device-safe),
  `to_offsets` (Float64 head-like values → offsets, per node or per record). Tests: npf,
  boundaries.
- **gwf/npf.jl**: NPF, confined: saturated conductance per connection (NPF.f90). Device-safe
  `logmean`, `condmean`, `staggered_thkfrac`; host `hyeff` / `hy_eff` (anisotropic K),
  `NpfProperties` / `npf_properties`, `check_npf_options`, `calc_condsat`, `NpfSetup` /
  `npf_setup(::GwfSpec)`, device struct `NpfModel{T}`, `CCOND_*` averaging codes. Tests:
  npf (parity vs libmf6 CONDSAT).

## Boundary packages

- **boundaries/framework.jl**: boundary framework (PLAN.md 3.4; BoundaryPackage.f90
  `bnd_rp`/`bnd_fc`/`bnd_cq`). `BoundaryKind{K}`, `BoundaryPackage{K,T}` (device
  `nodelist`, `bound`, `hcof`, `rhs`, `simvals`, node→record map; host copies),
  `bound_columns` / `head_columns` (per kind), `check_boundary_spec` (rejects WEL
  AUTO_FLOW_REDUCE), `apply_period!` (period data, MULTIPLIER aux, head columns →
  offsets, RCH highest active cell), `highest_active`, `boundary_ibound` (CHD → IBOUND
  −1), `formulate!` / `formulate_record`, `add_to_system!`, `calc_simvals!`,
  `boundary_kind`. Tests: boundaries.
- **boundaries/chd.jl**: CHD (gwf-chd.f90; see `docs/mf6_notes.md`). `apply_chd_heads!`,
  `calc_chd_rate!`. Tests: boundaries (parity vs MF6 CBC).
- **boundaries/wel.jl**, **rch.jl**, **ghb.jl**: WEL, RCH (list and RCHA arrays, cell
  area), GHB formulations (`formulate_record` methods). Tests: boundaries.

## MF6 output files (host)

- **output/binary.jl**: shared binary I/O helpers (little-endian stream files).
  `MF6BinaryFileError`, `pad_text`, `OUTPUT_TEXT_BYTES`. Tests: output.
- **output/grb.jl**: binary grid files. `BinaryGridFile`, `GrbVariable`, `read_grb`,
  `write_grb`. Tests: output, input_arrays, connectivity.
- **output/layout.jl**: how arrays split into records. `OutputLayout`, `nodesuser`,
  `budget_dims`. Tests: output.
- **output/hds.jl**: head files. `HeadFile` / `open_head_file`, `HeadStep`, `head_kstpkper`,
  `head_times`, `find_head_step`, `read_head`, `head_output_values`, `write_head_record`,
  `write_head_file`, `HNOFLO`, `HDRY`. Tests: output.
- **output/cbc.jl**: cell-by-cell budget files. `BudgetFile` / `open_budget_file`,
  `BudgetRecord`, `BudgetList`, `budget_texts`, `budget_kstpkper`, `budget_times`,
  `find_budget_records`, `read_budget`, `write_budget_array`, `write_budget_list`,
  `write_budget_file`. Tests: output.
- **output/listing.jl**: listing-file budget tables. `ListingBudget`, `ListingBudgetTerm`,
  `read_listing_budgets`, `listing_rates`, `percent_discrepancy`. Tests: output.

## Test helpers (`test/`, always loaded)

- **testutils.jl**: `for_each_precision`, `write_input`, `input_error_of`.
- **input_files.jl**: `REFERENCE_DIR`, `in_scope_examples`, `simulation_input_files`,
  `model_package_files`.
- **libmf6_oracle.jl**: libmf6 subprocess oracle. `LIBMF6_ENV`, `libmf6_available`,
  `run_libmf6_script`, `libmf6_connectivity`, `write_dis_simulation` (synthetic DIS sims).
- **grids.jl**: `_synthetic_coupling_dis`, `_synthetic_passthrough_dis`, `_rel_close`.
