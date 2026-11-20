# API map of `src/`

Update this file in every card's merge; skip `src/input/dfn/definitions.jl` (generated).

One entry per file, in include order (`src/ModflowGPU.jl`): purpose, key types and
functions, and the test groups (`MFGPU_TEST_GROUPS`, see `test/runtests.jl`) that cover
it. Underscore-prefixed helpers are omitted.

## Public API (T8.2)

- **ModflowGPU.jl**: the `export` and `public` lists, by group (running, backends and
  precision, stepping, builder, output, backend authors). Exported: the user-facing names
  (`run_simulation`, `finalize_simulation`, the stepping calls, builders, readers);
  `public` only: `run` and `finalize` (aliases of `run_simulation` and
  `finalize_simulation`, same function objects; Base clash), `solve` (CommonSolve clash), `update` (CUDA.jl exports `update`),
  `device_bound!`, the error and component types (`MF6InputError`, `HeadStep`, ...), the
  backend hooks (`register_backend!`, `pin_memory!`, `free_memory!`, `host_mapped_zeros`,
  `free_device_arrays!`). Every public name has a docstring and is listed in exactly one
  `@docs` block of `docs/src/api/*.md` (`checkdocs = :public` in docs/make.jl). A new
  public name needs both. Tests: api (test/unit/api_test.jl: names vs the API pages,
  docstrings, no clash with Base exports, aliases).

## Backends and precision

- **backend_setup.jl**: backend registry, adapted from mGV. `select_backend` (name → KA
  backend), `register_backend!` (called by `ext/*Ext.jl`), `backend_options`,
  `backend_name`, `known_backend_package`, `_WG` (T4.5: static work-group size 256 of the element-wise kernels), `pin_memory!` (extension hook; returns a handle to keep, or `nothing`; CUDA, ROCm since T4.6), `free_memory!` (extension hook, T5.4: release a device array now; default `KernelAbstractions.unsafe_free!`, ROCm and CUDA methods), `free_device_arrays!` (T5.4: frees every device array reachable from an object; used by `finalize` and the AMG rebuild), `host_mapped_zeros`
  (extension hook, T3.1: device view of a host vector for reduction results; CPU, ROCm), `BackendEntry`, `BACKEND_REGISTRY`, `KNOWN_BACKENDS`; `_device_copy` (host array → new device array). Tests: backend.
- **ext/*Ext.jl** (AMDGPU, CUDA, Metal, oneAPI, OpenCL): `__init__` registers the backend;
  overrides `supports_float64` where needed; AMDGPUExt adds `host_mapped_zeros` and
  `pin_memory!` (registered host memory); AMDGPUExt and CUDAExt add `free_memory!` (T5.4). Tests: backend (GPU file with `MFGPU_TEST_BACKEND`).
- **precision.jl**: precision presets (PLAN.md 3.7). `resolve_precision(backend, preset)`
  → `(; T, Ts, Tp)` (`:double`, `:mixed` = Float64 with Float32 `Tp` since T4.4, `:single`; `:auto` = `:mixed` with Float64, else `:single`), `PRECISION_PRESETS`, `supports_float64` (extension hook). Tests: precision, kernel.
- **instrument.jl** (T4.1): optional timing of the hot loop. `@timed_section name expr`
  (one `Ref` load and a branch when off; no closure), `set_section_timer!` /
  `section_timer`, hooks `section_begin` / `section_end` (methods come from the timer type,
  e.g. benchmark/run.jl's `SyncTimer`). Sections: `assembly`, `linear_solve`
  (`precond_setup`, `precond_apply`, `spmv_dot`, `update_reduce`, `vector_update`, `true_residual`),
  `outer_reduction`, `underrelax` (picard.jl, krylov.jl); `period_data`,
  `start_timestep`, `flows`, `budget`, `output`, `mirrors` (stepping.jl). Tests: stepping.

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
- **input/timeseries.jl** (T6.3): TS6 and TAS6 files and their evaluation on the host in
  Float64 (TimeSeries.f90, TimeArraySeries.f90). `TimeSeries` / `read_time_series_file`
  (NAME(S), METHOD(S), SFAC(S), several series per file, DNODATA), `ts_value(s, t0, t1)`
  (`GetValue`: STEPWISE/LINEAR time-weighted average over the step, LINEAREND value at
  `t1`), `TimeArraySeries` / `read_time_array_series_file`, `tas_values!` (`GetAverageValues`).
  Tests: timeseries.
- **input/options.jl**: OPTIONS / DIMENSIONS / TIMING blocks through the DFN schema.
  `OptionSet`, `read_options`, `option`, `has_option`, `option_line`. Tests:
  simulation_loader.
- **input/supported.jl**: what the engine supports; the loader checks input against it.
  `MF6UnsupportedError`, `SUPPORTED_OPTIONS`, `SUPPORTED_VALUES`, `SUPPORTED_FEATURES`,
  `UNSUPPORTED_PACKAGES` / `UNSUPPORTED_MODELS`, `check_options`, `check_value`,
  `check_feature` (T6.4: `:disv`, `:disu` supported; DISV/DISU options; T6.1: `:convertible`,
  `:convertible_storage`, NPF THICKSTRT/VARIABLECV/DEWATERED/PERCHED/REWET, WEL
  AUTO_FLOW_REDUCE). Tests: simulation_loader, input_lists.
  T6.6: `:multi_model` and `:exchange` supported, `:multi_solution` (more than one solution)
  not; `SUPPORTED_OPTIONS` of `exg-gwfgwf`.

## Model description (host specs, plain structs)

- **spec/grid.jl**: discretization specs (PLAN.md 3.2). `AbstractDisSpec`, `DisSpec`,
  `DisvSpec`, `DisuSpec`, `grid_dims`, `nodes_user`, `idomain_user`. Tests:
  simulation_loader, connectivity.
- **input/obs.jl** (T6.7; included after spec/grid.jl): OBS6 input (Obs.f90
  `read_obs_options`/`read_obs_blocks`, Observe.f90 `ConstructObservation`, the ID
  processors of gwf-obs.f90 and `DefaultObsIdProcessor`). `ObsEntry` (name, type, user
  node(s) or boundname), `ObsOutputSpec` (one CONTINUOUS FILEOUT block, BINARY),
  `ObsSpec` (DIGITS, PRINT_INPUT), `OBS_TYPES` (per owner: `:GWF` HEAD/DRAWDOWN/FLOW-JA-FACE,
  package kinds incl. TO-MVR and WEL-REDUCTION), `OBSNODATA` (3.0e30), `read_obs_file(path,
  owner, dis)`. Loaded into `GwfSpec.obs` and `BoundarySpec.obs` by input/simulation.jl.
  Tests: obs.
  T6.6: owner `:GWFGWF` (`FLOW-JA-FACE`; `cells` holds the exchange number, or a boundary
  name), read for GWF-GWF exchanges.
- **spec/packages.jl**: GWF package specs. `NpfSpec`, `StoSpec` (`is_transient`), `IcSpec`,
  `OcSpec`, `PrintFormat`, `BoundarySpec{Kind}` (T6.3: `timeseries`, `timearrayseries`
  dictionaries by name, read by the loader, `_load_time_series` in input/simulation.jl,
  which also checks every name the period data use and, T6.1, rejects a TS6 file named twice), `BOUNDARY_KINDS`. Tests:
  simulation_loader, npf, timeseries.
- **spec/simulation.jl**: simulation-level specs. `TdisSpec` (`nper`), `ImsSpec`
  (`ims_defaults`), `SolutionGroupSpec`, `PackageEntry`, `GwfSpec`, `ExchangeSpec`,
  `SimulationSpec`, `model_spec`. Tests: simulation_loader.
  T6.6: `ExchangeSpec.obs` (the exchange's OBS6 file read, or `nothing`).
- **input/simulation.jl**: simulation loader, mfsim.nam → `SimulationSpec` (TDIS, IMS,
  GWF name files, DIS/DISV/DISU, IC, NPF, STO, OC, boundaries, exchanges).
  `load_simulation`, `LENMODELNAME`, `LENPACKAGENAME`. Tests: simulation_loader (examples
  need `reference/`).
  T6.6: GWF6-GWF6 options checked with `strict`, the exchange OBS6 file read and its
  exchange numbers / boundary names checked (`gwf_gwf_rp_obs`).

## Discretization and connectivity (host)

- **discretization/connectivity.jl**: MF6 CSR connectivity layout (PLAN.md 3.3).
  `Connectivity` (ia/ja/jas/isym/ihc, cl1/cl2/hwva/anglex, top/bot/area/xc/yc, reduced
  numbering; T6.4: `grid` = `:dis`/`:disv`/`:disu`, xc/yc empty for DISU without vertices),
  `build_connectivity`, `reduced_numbering`, `fill_isym!`, `fill_jas!`,
  `iajausr`, `getjaindex`, `connection_lengths`. Tests: connectivity.
- **discretization/dis.jl**: DIS → `Connectivity` (Dis.f90): `build_connectivity(::DisSpec)`
  with IDOMAIN 0 / pass-through cells, `check_dis_grid`. Tests: connectivity (parity vs
  libmf6 and `.grb`).
- **discretization/disv.jl** (T6.4): DISV → `Connectivity` (Disv.f90 `grid_finalize`,
  `connect`, `define_cellverts`; Connections.f90 `disvconnections`/`vertexconnect`;
  DisvGeom.f90 `cprops`, `shared_edge`, `anglex`): `build_connectivity(::DisvSpec)`,
  `check_disv_grid` (MF6 messages; returns the cell2d areas), `disv_anglex`;
  `_cell_vertex_csr` (closed vertex lists, the GRB IAVERT/JAVERT). Tests: disv_disu,
  connectivity (libmf6 oracle, `.grb`), npf (CONDSAT oracle).
- **discretization/disu.jl** (T6.4): DISU → `Connectivity` (Disu.f90 `disu_ck`,
  `grid_finalize`, `disu_face_normal`; Connections.f90 `disuconnections`, `con_finalize`):
  `build_connectivity(::DisuSpec)` (reduced grids: rows sorted, input arrays compressed),
  `check_disu_grid`, `disu_angldegx_errors` (MF6's `nangldegxerr` and warnings), `DNODATA`.
  Tests: disv_disu, connectivity, npf.

## In-memory models and the MF6 writer (host, T3.2)

- **builder.jl**: `SimulationSpec` from Julia arrays (grid arrays `(ncol, nrow[, nlay])`
  as ModflowCoupling, or scalar / per layer / by user node; boundary records as cell lists
  + value matrices `(nrecords, ncolumns)`), validated with `ArgumentError`s. `build_dis`,
  `build_npf`, `build_ic`, `build_sto`, `oc_record`, `build_oc`, `build_boundary` (list
  packages), `build_array_boundary` (RCHA/EVTA), `build_model` (package entries, default
  names and files `<model>.<ext>`, MF6 package order), `build_tdis`, `build_ims` (`given`
  = keywords passed), `build_simulation`; structural equality `spec_difference` /
  `spec_identical` (types, shapes, order, Float64 bits; `ignore` fields). Tests: builder.
- **mf6_writer.jl**: `write_simulation(dir, spec)`: MF6 input files of a spec (DIS grids,
  NPF, IC, STO, OC, `BOUNDARY_KINDS` lists and arrays, TDIS, IMS; generalised from
  ModflowCoupling's `mf6_writer.jl`), round trip `load_simulation` → identical spec.
  Tests: builder (round trip; mf6 runs the written model, needs `reference/`).

## GWF engine

- **gwf/offsets.jl**: heads as offsets from per-cell references (PLAN.md 3.7 rule 1).
  `reference_heads` (STRT for ICELLTYPE 0 with a clamp fallback for implausible STRT,
  `REFERENCE_FALLBACK_MIN_DISTANCE`; clamp for convertible cells), `reference_differences`, `reference_difference` (device-safe),
  `to_offsets` (Float64 head-like values → offsets, per node or per record). Tests: npf,
  boundaries.
- **gwf/npf.jl**: NPF, confined: saturated conductance per connection (NPF.f90). Device-safe
  `logmean`, `condmean`, `staggered_thkfrac`; host `hyeff` / `hy_eff` (anisotropic K),
  `NpfProperties` / `npf_properties`, `check_npf_options`, `calc_condsat`, `NpfSetup` /
  `npf_setup(::GwfSpec)`, device struct `NpfModel{T}` (`NpfModel(backend, T, setup)`, Adapt), `CCOND_*` averaging codes. T6.1:
  THICKSTRT (`NpfProperties.thickstart`, `sat0`, `calc_condsat` with initial saturations), `wetdry`,
  `NpfSetup.hyl`/`hyh`, reference heads clamped also for STO ICONVERT ≠ 0 cells, `NpfModel.cv`
  (`nothing` for confined models: the constant-conductance path; `npf_setup(...; options, conn)`). T6.4: connection
  normals from `anglex` by cos/sin on DISV/DISU (exact axes on DIS), the DISU ANGLDEGX
  checks of `prepcheck`/`npf_ar` (`_check_disu_angldegx`). Tests:
  npf (parity vs libmf6 CONDSAT, also DISV/DISU), disv_disu.
- **gwf/convertible.jl** (T6.1; included after solver/csr.jl): convertible cells and wetting/drying
  (gwf-npf.f90 `cf_default_flow`, `fc_default_flow`, `sgwf_npf_qcalc`, `sgwf_npf_wetdry`,
  `rewet_check`, `npf_ad`; GwfConductanceUtils.f90 `hcond`, `vcond`). Device struct
  `ConvertibleNpf{T}` (offsets `topoff`/`botoff`, `thk`, `sat`, per-connection `hyl`/`hyh`/cl/hwva,
  `ihcx`; options VARIABLECV, DEWATERED, PERCHED), `thksat`, `connection_terms` (row-kernel terms
  of one connection for `condsat` or a `ConvertibleNpf`, PERCHED included; used by every
  assembly/residual/flow kernel), `_cond_arg`, `npf_cf!` (`npf_sat_kernel!`), `WetDry` (state),
  `npf_rewet_kernel!` (Jacobi passes equal to MF6's sequential loop), `npf_rewet_apply_kernel!`,
  `npf_dry_kernel!`, `wetdry_device!` (T6.9: also returns the rewet flags), `npf_ad!`. Tests: convertible, assembly, run.
  T6.2 (Newton–Raphson; gwf-npf.f90 `fn_default_flow`, `highest_cell_saturation`,
  `sgwf_npf_thksat`; SmoothingFunctions.f90): `ConvertibleNpf.newton`/`ihighcellsat` (`ihcx`
  bit 8: bottoms differ by ≥ 0.01), `newton_saturation`, `newton_saturation_derivative`
  (`sQuadraticSaturation[Derivative]`, SATOMEGA 1e-6), `_newton_conductance` (upstream
  saturation × condsat; ties at the offsets' rounding level go to the higher-numbered cell),
  `_newton_jacobian`, `connection_terms(...; jac)` (the Jacobian term of the upstream cell in
  `a`/`dd`/`sb`; `q` unchanged); DEWATERED guard: a saturation below eps(T) is 0 (T6.1
  follow-up). Tests: newton (FD Jacobian), convertible, assembly (libmf6 oracle), run.
- **gwf/xt3d.jl** (T6.5; included after gwf/convertible.jl): XT3D for confined models
  (Xt3dAlgorithm.f90 `qconds`, `abhats`, `getrot`, `tranvc`, `abwts`; Xt3dInterface.f90
  `xt3d_load`, `xt3d_areas`, `xt3d_ac`, `xt3d_fcpc`/`xt3d_fc`, `xt3d_rhs`, `xt3d_flowja`;
  `connection_vector` of Dis/Disv/Disu.f90). `check_xt3d` (DISU without vertices/ANGLDEGX:
  MF6's errors; convertible cells or NEWTON: unsupported), host `Xt3dSetup` / `xt3d_setup`
  (Float64 coefficients as flux terms `c·(h_a − h_b)` per symmetric connection, the
  extended stencil `_xt3d_structure` with neighbours of neighbours, the constant XT3D matrix
  and `sb = −Σ a·Δr`), device `Xt3dNpf{T}` (Adapt), `CsrStructure(backend, ::Xt3dSetup)`,
  `_xt3d_flux` (compensated face flow), `xt3d_formulate_system!` (`xt3d_formulate_kernel!`:
  packages, storage, matrix row, flux-form residual; RHS option: matrix of the primary terms),
  `xt3d_flowja!` (`xt3d_flowja_kernel!`). Tests: xt3d (K-orthogonal reduction, libmf6 oracle,
  universal residual, RHS and test/models/xt3d run parity), run.
  T6.6: `xt3d_assemble` (the structure, matrix and `sb` from flux terms per connection;
  `xt3d_setup` ends with it, and the XT3D of a multi-model solution uses it).
- **gwf/exchange.jl** (T6.6; included after gwf/xt3d.jl): multi-model solutions and GWF-GWF
  exchanges, host setup in Float64 (NumericalSolution.f90 `sln_df`/`sln_connect`,
  exg-gwfgwf.f90 `validate_exchange`, `calc_cond_sat`, `condcalc`, `gwf_gwf_fc`; the
  interface model of XT3D on an exchange: SpatialModelConnection.f90, GridConnection.f90,
  GwfGwfConnection.f90). The models of the solution become one merged model: `ModelPart`
  (spec, own conn and NpfSetup, node/user offsets, `jamap` model CSR entry → merged entry,
  package range), `ExchangeSetup` (name `GWF-GWF_<id>`, model indices, nodes per model and
  merged, `pos` = merged entry of the flow into the model-1 cell, `condsat`),
  `SolutionSetup` (models, exchanges, merged `conn` (sorted rows, grid `:disu`) and
  `setup`, `exgentry` per CSR entry, `xt3d`, merged `ic` and `sto`); `solution_setup(sim)`
  (checks: confined models only, MF6's exchange errors), `_exchange_condsat`,
  `_merge_properties`, `_merge_sto`, `_solution_xt3d` (flux terms: conductance, each model's
  own XT3D, and `_interface_xt3d!` on `_interface_grid` (global coordinates, nozee false) for
  `_interface_connections`: the exchange connections and, with XT3D in the owner model, the
  connections of its exchange cells), `solution_colors` (per-model red–black made
  consistent across exchanges, else greedy), `ModelView` (one model of the merged model; the
  accessors `_vspec`, `_vconn`, `_vrange`, ... are in gwf/model.jl). Tests: exchange.
- **gwf/ic.jl**: IC (gwf-ic.f90 `ic_ar`). `initial_heads` (starting heads per reduced
  node as offsets, Float64 host or device copy in `T`). Tests: sto.
- **gwf/sto.jl**: storage (gwf-sto.f90 `sto_rp`/`sto_fc`/`sto_cq`/`sto_bd`,
  GwfStorageUtils.f90 `SsCapacity`/`SsTerms`/`SyTerms`). `storage_capacity`, `check_storage_spec`
  (rejects TVS6), T6.1: `ConvertibleStorage` (ICONVERT, offsets, `rho2`, SS_CONFINED_ONLY),
  `storage_saturation`, `storage_terms`, `StoragePackage.conv`/`sc2`/`strgsy` (STO-SY), `StoragePackage{T}`
  (device `rho = sc/delt`, `strgss`; Adapt), `apply_period!` (steady/transient), `set_timestep!`
  (`rho` from Float64 `delt`), `add_to_system!` (diagonal and `b`), `calc_storage_rates!`
  (STO-SS), `add_storage_flowja!`, `storage_budget` (rin/rout, host Float64). Tests: sto
  (parity: residual test on fhb with constant stresses, mf6 run at test time).
  T6.2 (`fn_default_sto`): `ConvertibleStorage.newton`, `newton_satthk` (thk·S without the
  division), `storage_terms(...; jac)` (Newton: quadratic saturations; with `jac` the SY term
  as `rho2·(thk·sn − thk·snold)` on the right-hand side, without it MF6's `SyTerms`),
  `storage_jacobian` (the diagonal Newton term). Tests: newton, convertible, run.

## Boundary packages

- **boundaries/framework.jl**: boundary framework (PLAN.md 3.4; BoundaryPackage.f90
  `bnd_rp`/`bnd_fc`/`bnd_cq`). `BoundaryKind{K}`, `BoundaryPackage{K,T}` (device
  `nodelist`, `bound`, `hcof`, `rhs`, `simvals`, node→record map; host copies; Adapt moves the device
  arrays and copies the host period state),
  `bound_columns` / `head_columns` / `mult_column` (per kind), `bound_layout` (per spec:
  list columns, aux variables, constants), `check_records` (`*_ck` hook),
  `check_boundary_spec` (rejects WEL AUTO_FLOW_REDUCE; EVT NSEG / READASARRAYS checks),
  `apply_period!` (period data, MULTIPLIER aux, checks, RCH/EVT highest active cell, head
  columns → offsets), `highest_active`, `boundary_ibound` (CHD → IBOUND
  −1), `formulate!` / `formulate_record`, `add_to_system!`, `calc_simvals!`,
  `boundary_kind`. Tests: boundaries.
  T6.1: `BoundaryPackage.icelltype` (cell-bottom checks of `riv_ck`/`drn_ck`/`ghb_ck`, which
  also check the raw conductance and the multiplier separately: `_check_cond_mult!`), `moved`
  (no period fast path after records were moved), `move_to_highest_active!` (RCH/EVT after
  IBOUND changes, from the cells given), `boundary_ibound(conn, packages, current)` (dry cells
  stay dry, `chd_rp`), `formulate!(...; kind)`. Tests: convertible, run.
  T3.3: `bound_user` (host values as given, before the multiplier) and `set_bound!` (stepping
  API writes: multiplier + offsets as period data, then copy to the device); tests: stepping.
  T4.6 (period-data path): a block with the cells in force keeps nodes and the node→record
  map (`_apply_period!` → `:unchanged` / `:values` / `:structure`); values are read from the
  PERIOD block through `BoundSource` (lazy `(ncol, n)` view, multiplier applied; what
  `check_records(kind, pkg, bound, n)` sees) and staged in one fused pass (Float64
  multiplier and offsets with per-record references `rrec`, one cast) into the pinned
  `staging` buffer, one `copyto!` (Float64 pass-through columns are copied directly);
  `bound_user!` fills `bound_user` on demand; `bound_version` counts value changes;
  `bound_values(pkg, r)` (Float64 `bound`, replaces the `bound_host` field). Tests:
  boundaries (fast path vs full application). T6.2: `newton_record` (the diagonal Newton
  term of a record, `*_fn`; 0 for every kind but `WelAfr` with `newton` and `DrnNewton`).
  T6.3 (time series): a PERIOD block with `TsRef`s or TIMEARRAYSERIES is applied through a
  working copy (`tsblock`, `_ts_working_copy`, `_block_data`) whose linked entries hold the
  series values of the step (`apply_period!(...; window)`); `advance_time_series!(pkg,
  window, r)` re-evaluates them per step and restages `bound` (multiplier and offsets in
  Float64, as period data; after a stepping API write the series replace their columns of
  `bound_user`); `tswindow`. Tests: timeseries, run, assembly, sto.
  T6.6: `BoundaryPackage.uoffset` (keyword `uoffset`: a package of a model of a merged
  model resolves its cells through the solution's user numbering, `_reduced(pkg, nu)`).
  T6.9: `BoundaryPackage.device_written` (set by `device_bound!`, cleared by a full host
  staging) and `tsscatter`; `advance_time_series!` writes only the entries linked to a series
  (`_linked_entries`, `_stage_linked!`, `bound_scatter_kernel!`) and checks the records
  (`check_records`, as `gwf_ad` → `bnd_ck`); `move_to_highest_active!` restages nothing for
  RCH and only the EVT surface of moved records (`bound_shift_kernel!` by the reference
  difference when written on the device). Tests: timeseries, riv_drn_evt.
- **boundaries/chd.jl**: CHD (gwf-chd.f90; see `docs/dev/mf6_notes.md`). `apply_chd_heads!`,
  `calc_chd_rate!`. Tests: boundaries (parity vs MF6 CBC).
  T6.6: `calc_chd_rate!(...; exg)` skips the exchange entries (`_is_exchange`).
- **boundaries/wel.jl**, **rch.jl**, **ghb.jl**: WEL, RCH (list and RCHA arrays, cell
  area), GHB formulations (`formulate_record` methods). Tests: boundaries. T6.1: WEL
  AUTO_FLOW_REDUCE (`WelAfr` record kind, `sq_saturation`, `record_kind`; tests: run, model
  test/models/afr), `ghb_ck` (`check_records`). T6.2: `record_kind(pkg, cv; newton)`,
  `WelAfr.newton` and `sq_saturation_derivative` (`wel_fn`); `sq_saturation(..., c1, c2)`.
- **boundaries/riv.jl**, **drn.jl**, **evt.jl** (T2.4): RIV (`riv_cf`, stage/rbot as
  offsets, `riv_ck`), DRN (`drn_cf` with AUXDEPTHNAME drainage depth, device-safe
  `linear_saturation` = `sQuadraticSaturation` with eps 0), EVT/EVTA (`evt_cf`: NSEG
  segments, PETM0 / SURF_RATE_SPECIFIED, `check_pxdp`; `bound` = surface, rate, depth,
  petm0, pxdp…, petm…). Tests: riv_drn_evt (parity: residual and CBC flows on
  test/models/coupling_rde, mf6 run at test time). T6.2: `DrnNewton` (DRN AUXDEPTHNAME in a
  Newton model: cubic scaling `sq_saturation(…, −1, 2)` and the `drn_fn` term; tests: newton,
  run on test/models/newton).

## Assembly and residual

- **solver/csr.jl** (T2.6): MF6-layout assembly and the flux-form residual (PLAN.md 3.3,
  3.7 rule 2; `npf_fc`, `bnd_fc`, `sto_fc`, `sln_ls`). Device structs `CsrStructure`
  (ia/ja/jas/isym, `idiag`, `inode`; Adapt) and `LinearSystem{T}` (`amat`, offset rhs
  `b' = b − A·r`, per-node boundary + storage terms `bdiag`/`brhs`); `reset_terms!`,
  `add_terms!` (boundary or storage package → `bdiag`/`brhs`), `assemble!` (CHD identity
  rows, small-diagonal fix, `symmetric` CG column move), `flux_residual!` (compensated row
  sums; optional `scale`/`bound` outputs for the universal residual test). Tests:
  assembly (parity: universal residual on Tier 0–2 + synthetic models, libmf6 first-solve
  AMAT/RHS oracle via validation/libmf6_first_solve.jl, cancellation test), also used by
  boundaries, sto, riv_drn_evt through `flux_residual` in test/residual.jl. Also `spmv!`
  (`y = A·x`, T2.7; tests: krylov). T4.5: `formulate_system!` (`formulate_kernel!`:
  boundary/storage terms, assembly and residual in one pass, bitwise as the separate
  kernels; `offdiag = false` skips the off-diagonal writes), `CsrStructure.irow` (row of
  each entry, for entry-parallel kernels); tests: picard, run, stepping.
  T4.7: `LinearSystem.diag` (contiguous copy of the diagonal, written by `assemble!` and
  `formulate_system!`); `formulate_system!` keywords `terms = false` (no `rhs`/`bdiag`/`brhs`
  writes, the solver path) and package tuples `(kind, bound, area, hcof, rhs, nodeptr,
  noderec)` whose records are formulated in the row kernel (`_package_terms`); tests:
  assembly. T6.2: `formulate_system!(...; jac)` (Newton terms of NPF, storage and packages
  on the diagonal/off-diagonals and the right-hand side; `bdiag`/`brhs` and the residual
  without them).

## Linear solvers

- **solver/krylov.jl** (T2.7): correction solve `A·δ = res` with zero initial guess
  (PLAN.md 3.7 rules 3–4; ImsLinearBase.f90 `ims_base_cg`, `ims_base_bcgs`,
  `ims_base_scale`, `ims_base_testcnvg`, `ims_base_epfact`). `KrylovSettings` (host; from
  an `ImsSpec`: LINEAR_ACCELERATION, INNER_*, rclose options, NUMBER_ORTHOGONALIZATIONS,
  SCALING_METHOD, `preconditioner`; ILU0/MILU0/ILUT → `preconditioner`, `:auto` = `:mc_ilu0` by default since T4.4; the replacement is logged once per session, at info level when ILUT/MILU come from COMPLEXITY only, T5.2), `ims_preconditioner`
  (`resolve_ipc`; T5.4: once-per-session messages keyed by message, `_log_id`, not by call site), `ims_testcnvg`, `ims_epfact`, `KrylovSolver{Ts,Tp,Ta}` (device
  workspace), `solve!(δ, solver, amat, res; kstp)` → `KrylovResult` (converged,
  iterations, status, dvmax, rmax, l2norm0, l2norm). Device reductions: `Reducer`,
  `accumulation_type`, `device_dot`, `device_maxabs`, `device_argmaxabs` (T3.1; fused
  map-reduce kernels `mapreduce_kernel!` / `finalize_reduction_kernel!`, `map_kernel!`;
  T4.5: `_reduce_async!` / `_reduce_stage1!` / `_finalize_async!` / `_reduced` (launch
  without waiting, read after one synchronisation) and `post` maps of the finalisation
  (`_SetRho`, `_AlphaPost`, `_OmegaPost`: rho, alpha, omega kept on the device in `sa`/`ss`),
  so CG and BiCGSTAB synchronise once per iteration (`KrylovSolver` fields `reducer2`,
  `reducer3`, `sa`, `ss`).
  T4.7: `solve!(...; diag, offdiag_id)` (setup from the contiguous diagonal; incremental
  preconditioner setup while `offdiag_id` is unchanged, `_precond_setup!` hook);
  single-pass reductions `mapreduce_last_kernel!` / `_reduce_last!` (the last work group
  finalises, atomic counter `Reducer.count`); `_NoScale` / `_op_factors` (SCALING_METHOD
  NONE: operator rows without reading `rowfac`/`colfac`); CG: direction fused into the
  SpMV (`_dir_apply_dot`, direction vectors `p`/`phat`), device convergence test
  `_CgStopPost` (`sa[4]`) so MC-ILU0 skips the last application.
  `_ArgMaxAbsOp` as the first op also returns the first index of the largest magnitude,
  `_mapreduce_loc!`; results land in host-mapped memory where `host_mapped_zeros` allows).
  Adapt rules on `KrylovSolver` and `Reducer` (a new one on the target). IMS mapping in
  docs/dev/mf6_notes.md, accuracy and timing in docs/dev/precision.md. Tests: krylov.
  `KrylovSettings` has `preconditioner` and `precond_options`, `KrylovSolver` a `precond`
  field. Non-Jacobi preconditioners run through `precond_apply!` between non-fused kernels
  (`_cg_init_nf`, `_cg_update_nf`, `_bcgs_direction_nf`, `_bcgs_half_nf`; section
  `precond_apply`; CG's `r·z` fused by `_precond_apply_dot!` where the preconditioner
  does); `_precond_record!` / `_precond_retry!` after a solve (a failed solve is repeated
  once after an AMG rebuild); `KrylovSolver(...; colors)` passes a host colouring.
  T6.2: in Newton–Raphson solves (`solve!(...; newton = true)`) BiCGSTAB restarts with
  `dhat = r` (`_bcgs_restart`) when `rho` vanishes at the rounding level of the vectors
  (`|rho| <= eps(Ts)·|dhat|·|r|`; ex-gwf-zaidel with the red–black ILU0, nwt-p02a `:single`).
- **solver/precond/precond.jl** (T4.2, T4.3): preconditioner interface. `AbstractPreconditioner`,
  `precond_setup!(P, solver, amat)` (per `solve!`, after scaling), `precond_apply!(z, P, r)`,
  `JacobiPreconditioner` (wraps the solver's `minv`; fused path), `PRECONDITIONERS`
  (`:jacobi`, `:chebyshev`, `:mc_sgs`, `:mc_ilu0`, `:amg`), `DEFAULT_PRECONDITIONER` /
  `resolve_preconditioner` (`:auto` → `:mc_ilu0`, T4.4), `build_preconditioner` (the
  constructor; `precond_options` as its keywords). Tests: krylov (unit/precond_test.jl), amg.
- **solver/precond/chebyshev.jl** (T4.2): `ChebyshevPreconditioner` (scaled operator copy in
  `Tp`, host interval `bounds`; options `degree`, `ratio`, `lanczos`, `safety`),
  `lanczos_extremes`. Tests: krylov.
- **solver/precond/multicolor.jl** (T4.2): host colouring `greedy_colors`,
  `dis_parity_colors` (red–black), `grid_colors` (used by `PicardSolver`),
  `is_valid_coloring`; `MulticolorPreconditioner` (`:sgs` / `:ilu0`; colour-ordered CSR copy
  in `Tp`, per-colour kernels for the ILU0 factorisation and the sweeps; option `relax`,
  MILU), `ncolors`. Red–black ILU0 (T4.5): entry-parallel copy `_mc_rb_copy`, pivots
  `_mc_rb_factor` through the transpose positions `ctr`, `r·z` fused into the two apply
  kernels (`_precond_apply_dot!`). T4.7: incremental red–black setup (`_precond_setup!`
  method, `_mc_rb_diag`, `_mc_rb_refactor`; fields `dprev`, `flag`, `key`): only rows whose
  diagonal changed and their second-colour neighbours are refactored, bitwise equal to a
  full setup. Tests: krylov.
- **solver/precond/amg.jl** (T4.3): smoothed-aggregation AMG. `AMGOptions`,
  `AMGPreconditioner` (host Float64 setup: `_strength`, `_aggregate`, `_tentative`,
  `_smoother_matrix`, threaded `_spgemm`, `_build_hierarchy`; device `AMGLevel`s in `Tp`),
  `AMGStats` / `amg_stats(solver)`; device recomputation of prolongators
  (`_update_prolongator!`) and Galerkin products (`galerkin_kernel!`,
  `galerkin_store_kernel!`, accumulated in Float64 where available; exempt in the
  kernel-hygiene test) when the matrix changed (row signatures, `_matrix_change!`);
  Gershgorin bounds for the Chebyshev/Jacobi smoother; V-cycle `_cycle!`, fused finest
  level `_fine_cycle1!`, dense coarsest inverse; T4.5: `_amg_apply!` (function barrier
  for the untyped `fine`), smoother coefficients cached per level (`_level_coefficients`);
  T5.4: a rebuild frees the previous hierarchy (`free_device_arrays!`). Tests: amg (also kernel_hygiene, run:
  ex-gwf-hwell-a).

## Time stepping, outer iterations, flows and budgets (T2.8)

- **timing.jl**: TDIS on the host (tdis.f90 `tdis_set_timestep`/`tdis_set_delt`).
  `TimeStep` (kper, kstp, delt, pertim, totim, endofperiod), `time_steps(tdis)`,
  `time_step_window(steps, i)` (T6.3: `(totimc, totimc + delt)` for the time series). Tests:
  picard, timeseries.
- **gwf/model.jl**: one model on a device. `GwfModel{T}` (spec, conn, NpfSetup, CSR,
  NpfModel, LinearSystem, boundary packages, STO, ibound, offset heads `h`/`hold`, `res`,
  `δ`, `flowja`; `offdiag_id` (T4.7, session-unique id of the last off-diagonal write); Adapt, e.g. `adapt(backend, m)`, T3.1), `apply_period!(::GwfModel, kper; window)` (once per period: package period
  data, STO, CHD ibound, rebuilt only when a CHD package has new cells), `advance_time_series!(::GwfModel, window)` (T6.3: every step, in `prepare_time_step`), `start_timestep!` (`rho`, `hold ← h`, CHD heads), `formulate!`
  (T6.1: `npf_cf!` and new off-diagonals every call for convertible models), `wetdry!` (T6.1:
  `sgwf_npf_wetdry` per outer iteration, IBOUND to the host, RCH/EVT moves), field `wd`
  (per-package formulation, then `formulate_system!`; off-diagonals rewritten only when
  `ibound` or `symmetric` changed, field `offdiag`, T4.5), `absolute_heads` (host Float64 `r + h'`; HDRY for dry cells, T6.1). Tests:
  picard, run. T6.2: fields `newton`, `nurbot` (`newton_ur_bottoms`: `npf_nur`'s `ibotnode`
  bottom as an offset, −Inf where it does not apply), `vol` (PTC), `delt`; `wd = nothing` with
  NEWTON (no drying); `formulate!(m; jac = m.newton)`. T6.5: fields `xt3d` (`Xt3dNpf` or
  `nothing`) and `scsr` (the system/solver CSR: `csr`, or the extended XT3D stencil; `sys`,
  the Krylov solver, the colouring and PTC use it; `csr` stays the connectivity for flows);
  `formulate!` calls `xt3d_formulate_system!` for XT3D models.
  T6.6: fields `solution` (`SolutionSetup` or `nothing`) and `exgentry` (device Bool per CSR
  entry), `GwfModel(backend, T, sol::SolutionSetup, nper)` (the merged model: packages with
  `uoffset`, merged storage and starting heads), `model_views`, the view accessors
  (`_vspec`, `_vconn`, `_vrange`, `_voffset`, `_vpackages`, `_vmodel`, `_vsetup`, `_vname`;
  a `GwfModel` is its own view), `absolute_heads(::ModelView)`.
  T6.9: field `conversions` (the `(kiter, wet, dry)` cell conversions of `wetdry!` in the
  current step, cleared by `start_timestep!`).
- **solver/underrelax.jl**: `sln_underrelax`. `UnderRelaxation` (SIMPLE, COOLEY host
  scalars, DBD device state), `underrelax!` (`relax_update_kernel!`,
  `dbd_update_kernel!`); Adapt. Tests: picard (vs a host port), run (variants vs MF6).
- **solver/backtracking.jl** (T6.2a): `sln_backtracking`. `BacktrackRecord` (kiter, flag,
  count, resin, resout: MF6's "Backtracking" row), `backtrack!` (the reduction loop on a
  `reformulate` callback; reusable for Newton, T6.2), `residual_l2norm` (`sln_l2norm`,
  Float64 accumulation), `backtrack_update!` (`backtrack_update_kernel!`,
  `apply_backtracking`), `save_hprev!` (`save_hprev_kernel!`: `xtemp` with HDRY for dry
  cells). Tests: picard (unit and vs mf6), run (coupling_rde COMPLEX).
- **solver/newton.jl** (T6.2): outer-iteration parts of Newton–Raphson. `PtcState`
  (`ptcdel`, `l2norm0`, kept across steps), `ptc_allowed` (IMS NO_PTC), `apply_ptc!`
  (`sln_calc_ptc`/`gwf_ptc`/`sln_ls`: one reduction of max |res|/V and the L2 norm, then
  `−1/ptcdel` on the active diagonal), `newton_ur!` (`npf_nur` + `sln_nur_has_converged` in one
  reduction). Tests: newton, run.
- **solver/picard.jl**: outer loop (`sln_ca`/`solve`; PLAN.md 3.7 rule 3). `PicardSettings`
  (from `ImsSpec`, `closures` overrides, Float32 floors `FLOAT32_CLOSURES`),
  `PicardSolver` (KrylovSolver, UnderRelaxation, Reducer, `hprev` for backtracking; Adapt), `solve_timestep!` →
  `OuterResult` (converged, outer and inner iterations, hncg and its node `lrch` from one
  argmax reduction, the dvclose/rclose used, `backtracking` records, `backtrack_count`;
  T6.2a: `_backtracking!` after the formulation when BACKTRACKING_NUMBER > 0); `OuterState` + `outer_iteration!` (one outer
  iteration, T3.3; `solve_timestep!` loops over it; T5.3/T6.0: in Float32, once a
  correction meets OUTER_DVCLOSE and only the inner test failed, `OuterState.noise_floor`
  holds INNER_DVCLOSE >= `OuterState.dvfloor` = eps32·max|h'| for the rest of the step;
  T6.1: `wetdry!` before each formulation, and with convertible cells the Float32 rclose
  floor is the maximum over the step's iterations; T6.5: also with XT3D, whose rclose scale
  is the sum of the face terms' magnitudes; the Krylov solver and its colouring use `m.scsr`).
  T6.2: Newton formulation (`jac`; with backtracking first the formulation without the
  Newton terms), PTC in steady-state Newton periods, NEWTON UNDER_RELAXATION after the
  update, `PicardSolver.ptc`; `FLOAT32_CLOSURES.dvclose_head_cap` (the head-resolution
  OUTER_DVCLOSE floor is at most 1 cm).
  Tests: picard, run, stepping.
  T6.6: the colouring of a merged model is `solution_colors`.
- **budget.jl**: `gwf_cq`/`gwf_bd`. `npf_flowja!` (one work item per entry, T4.5; T6.1 `connection_terms`, STO-SY term), `calc_flows!` (NPF, STO, packages with
  `bnd_cf` + `simvals`, CHD `calc_chd_rate!`, `csr_diagsum`), `BudgetTerm`, `ModelBudget`
  (`total_rate_in`/`_out`, `total_cum_in`/`_out`, `percent_discrepancy`,
  `cum_percent_discrepancy`), `BudgetAccumulator` (Adapt), `model_budget!` (device reductions,
  host cumulative volumes), `budget_text` (CBC text per package). Tests: picard, run.
  T6.6: `ModelBudget.model` (the model's name); `calc_flows!` passes `exgentry` to the CHD
  rate.

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
  `read_listing_budgets`, `listing_rates`, `percent_discrepancy`, `write_listing_budget`
  (`budget_ot` layout, T2.8). Tests: output, picard.
- **output/obs.jl** (T6.7): observation values and files (gwf-obs.f90 `gwf_obs_bd`,
  BoundaryPackage.f90 `bnd_rp_obs`/`bnd_bd_obs`, gwf-wel.f90 `wel_bd_obs`, Obs.f90
  `build_headers`/`write_obs_simvals`, ObsUtility.f90 `write_fmtd_obs`/`write_unfmtd_obs`).
  `ObservationOutput` (built by `ModelOutput` when the model has OBS6 input; FLOW-JA-FACE
  connections checked with `getjaindex`), `obs_rp!` (records per package observation, at
  the first step of every period, from `prepare_time_step`), `obs_gather!`
  (`obs_gather_kernel!`: one launch per source array, one device→host copy per step),
  `obs_values`, `write_observations` (from `_write_step`), `obs_format` (Fortran G editing
  as MF6's Intel build: G0 with 16 digits, G20.13 times, DIGITS), `ObservationFile` /
  `read_observations` (CSV and BINARY, ours and MF6's). Tests: obs, run (parity of the
  observation files, `_obs_difference` in test/run_parity.jl).
  T6.6: `ObservationOutput(::ModelView, ...)` (nodes, connections and packages of the
  merged model, `_obs_node`, `_obs_ja`).
- **output/oc.jl** (T2.8): output control and run output. `oc_selected` (OC PERIOD
  blocks, ALL/FIRST/LAST/FREQUENCY/STEPS), `ModelOutput` (open HDS/CBC/listing files),
  `write_heads`, `write_budget` (STO-SS, FLOW-JA-FACE, package lists as MF6),
  `write_head_listing` (PRINT HEAD); T6.7: `ModelOutput.obs`. Tests: picard, run.
  T6.9: `write_conversions` (`sgwf_npf_wdmsg`: the `GwfModel.conversions` of the step in the
  listing; tests: run, `_conversion_blocks` vs MF6's listing).
  T6.6: `ModelOutput`, `write_heads`, `write_budget(...; exflows)`, `write_head_listing`
  take a `GwfModel` or a `ModelView`; `model_flowja` (a model's FLOW-JA-FACE; the diagonal
  of a constant-head cell without its exchange flows), `_model_values`.
- **solution.jl** (T6.6; included after output/oc.jl): multi-model solutions at run time
  (exg-gwfgwf.f90 `gwf_gwf_calc_simvals`, `gwf_gwf_bd`, `gwf_gwf_chd_bd`,
  `gwf_gwf_bdsav_model`, `gwf_gwf_rp_obs`, `gwf_gwf_save_simvals`). `ExchangeFlows` /
  `exchange_flows!` (gather of the exchange flows from the merged FLOW-JA-FACE, one copy
  per step), `exchange_simvals`, `BudgetAccumulator(::ModelView)` and
  `model_budget!(acc, v, step, x)` (per model: STO-SS, packages, then FLOW-JA-FACE and
  FLOW-JA-FACE-CHD per exchange in MF6's order, `_exchange_rows`), `write_exchange_budget`
  (the IMETH 6 lists of exchanges with SAVE_FLOWS), `ExchangeObservations` /
  `write_exchange_observations` (exchange OBS6). Tests: exchange, run.

## Running a simulation

- **run.jl** (T2.8): `run_simulation(simdir or SimulationSpec; backend, precision,
  outdir, closures, write_output, preconditioner, precond_options)` (on the stepping API since T3.3) → `RunResult` (model, precision types, `StepResult`s,
  `ModelBudget`s, final heads, output paths); `ConvergenceError`. One model, one IMS
  solution. Tests: run (parity vs MF6 on Tier 0, ex-gwf-fhb, ts_tas, coupling_rde and IMS
  variants; flopy read-back with MFGPU_FLOPY=1). `_write_step` (step output) copies the
  heads to the host only when OC saves or prints them (T5.2). `run` is a `const` alias
  of `run_simulation` (T8.2). T6.7: `RunResult.obs_files`; `_write_step` writes the
  observations every step.
  T6.6: several models: `RunResult.models` (per model `name`, `nodes` (range in `heads`),
  the output files), `heads` and `budgets` of every model; `_listing_header` and
  `_write_step` take a view.

- **stepping.jl** (T3.3): libmf6/XMI-like stepping (PLAN.md 3.6). `Simulation`,
  `initialize`, `prepare_time_step`, `prepare_solve`, `solve` (one outer iteration),
  `finalize_solve`, `finalize_time_step`, `do_time_step`, `update`, `finalize_simulation`
  (T8.2; `finalize` is a `const` alias),
  `get_current_time` / `get_start_time` / `get_end_time` / `get_time_step`, `StepResult`;
  T5.4: `finalize_simulation(sim; free = true)` frees the device arrays (stage `:freed`; `run_simulation` uses
  `free = false`), `device_bound!` (the device BOUND write window between
  `prepare_time_step` and `prepare_solve`, enforced; `prepare_solve` checks the written
  records on the device: finite, RIV stage ≥ bottom and conductance ≥ 0,
  `device_bound_check_kernel!`);
  addresses `get_var_address`, `var_addresses`, `input_addresses`, `get_value_ptr` (host
  Float64 mirror, absolute; `device = true` for the engine array), `get_value`,
  `set_value` (X, BOUND). Mirror sync rules in the file header; BOUND and NODELIST
  mirrors are versioned (T4.6: copied only when the engine values changed, dirty check by
  memcmp). T5.2: device BOUND writes between `prepare_time_step` and `prepare_solve` (the
  contract ModflowCoupling's device-resident coupling uses; header of the file). Tests: stepping
  (T8.5: "addresses documented" checks that `var_addresses` of a model with every stress
  package matches the address table of `docs/src/coupling/stepping.md`; a new address
  needs a row there).
  T6.6: several models in one solution: `Simulation` fields `views`, `models` (names),
  `outputs`, `accs`, `exflows`, `exobs`; `initialize` builds the merged model
  (`solution_setup`) when the simulation has several models or exchanges; addresses of every
  model (`_find_view`, `_find_package(s, model, package)`, `_view_variable`: device X, XOLD,
  IBOUND as views, host NODELIST in the model's numbering).
- **bmi.jl** (T3.3, T8.9): BasicModelInterface methods on `Simulation`, all 41 functions
  of the spec. `BMI.initialize(Simulation, config_file)` (no keywords; `_bmi_config`
  reads the TOML configuration file, or takes a directory / mfsim.nam with defaults),
  `update`, `update_until` (step ends only), time, variables (`_bmi_units`: UDUNITS from
  LENGTH_UNITS / TIME_UNITS; `_bmi_var_grid`), getters returning `dest`, grids
  (`_bmi_grid`: 0 scalar, 1 model grid, 2 model nodes as points, 1 + 2i / 2 + 2i package
  records / BOUND as vectors; coordinates with XORIGIN, YORIGIN, ANGROT; unstructured
  faces and edges from the DISV/DISU vertices), `_bmi_unsupported` (the ArgumentError of
  unsupported functions). Tests: bmi (conformance on DIS, DISV, DISU), stepping.
  T6.6: the grids of every model, one model after the other (`_bmi_ngrids`,
  `_bmi_grid_base`, `_bmi_view_grid`).
- **mf6model.jl** (T8.10): `MF6Model`, libmf6's BMI/XMI as ModflowInterface.jl wraps it
  (`ModflowModel`), a handle (`source`, `settings`, `simulation`, `last_error`, `arrays`)
  around a `Simulation`; every call forwards to the stepping API (`_mf6call` keeps the
  error message for `get_last_bmi_error`). `_mf6_ptr`: host mirrors with libmf6 shapes
  (0-dim scalars, length-1 NODEUSER/NODEREDUCED dummies); grid 1 per mf6bmiGrid.f90
  (`_mf6_dis(m, f, grid, T)` = `confirm_grid_type`, failures throw); functions libmf6
  lacks throw (`_mf6_not_libmf6`). Simulation's BMI (bmi.jl) is separate and unchanged.
  Tests: bmi (`test/unit/mf6model_test.jl`: validation/bmi_script.jl on MF6Model and,
  via validation/libmf6_bmi.jl, on libmf6 6.3.0 and 6.8.2).
  T6.6: grid k = model k of the solution (libmf6's model ids).

## Test helpers (`test/`, always loaded)

- **test_groups.jl**: `TEST_GROUPS` (group => files; T6.6: group `exchange`,
  test/parity/exchange_test.jl: universal residual and libmf6 oracle of the Tier 7 merged
  models, exchange unit tests, a two-model simulation against mf6). **parallel.jl** (T6.8):
  `run_parallel_tests`, the `MFGPU_TEST_JOBS=N` driver (work units, `GROUP_WEIGHTS`,
  `GROUP_SPLITS` of run by model through `MFGPU_TEST_MODELS`, which accepts
  `!a,b` = all but those).
- **testutils.jl**: `for_each_precision`, `write_input`, `input_error_of`.
- **input_files.jl**: `REFERENCE_DIR`, `in_scope_examples`, `simulation_input_files`,
  `model_package_files`; `parity_examples` (T6.0: Tier 0–2 plus `EXTRA_PARITY_EXAMPLES`,
  ex-gwf-lgrv-gc/-gr; `MFGPU_TEST_LGRV_GR=0` drops -gr; T6.4: plus the Tier 5 examples),
  used by the assembly and run parity tests; `reference_rundir(name)` (T6.3: the converged
  reference run, tight or tight-relative, e.g. ex-gwf-fhb; T6.1: `relaxed` only for
  `RELAXED_REFERENCE_EXAMPLES`); T6.1: the Tier 3 convertible examples in
  `EXTRA_PARITY_EXAMPLES`; `rad_disu_confined` (T6.4:
  ex-gwf-rad-disu with ICONVERT 0, written from the reference run at test time).
  T6.6: Tier 7 (several models, exchanges) in `parity_examples`.
- **libmf6_oracle.jl**: libmf6 subprocess oracle. `LIBMF6_ENV`, `libmf6_available`,
  `run_libmf6_script` (validation/libmf6_connectivity.jl, libmf6_first_solve.jl), `libmf6_connectivity`, `write_dis_simulation` (synthetic DIS sims; DISV/DISU since T6.4),
  `_libmf6_first_solve` (T6.5: moved from the assembly test; SLN_1 IA/JA/AMAT/RHS/X0 after the first solve).
- **grids.jl**: `_synthetic_coupling_dis`, `_synthetic_passthrough_dis`, `_rel_close`,
  `_dis`, `_Lcg`/`_next!`; T6.4: `_dis_as_disv`, `_dis_as_disu` (the DISV/DISU description of
  a DIS grid, same user node numbers).
- **run_parity.jl** (T3.4, moved from parity/run_test.jl; T6.7: `_obs_outputs`,
  `_obs_difference` (observation files vs MF6's or the CPU run's), `single_obs_tol`): `_run_parity(name, rundir)`
  (ModflowGPU vs MF6 output in `rundir`: heads, CBC, listing tables, both presets, GPU vs
  CPU; returns the numbers per preset; `single_preconditioner` for `:single`, T6.0; `single_budget_tol` and list records matched by package, T6.4; T6.1: `double_budget_tol`, dry cells compared, `_rounding_bounds` for the `:single` budgets), `_mf6_iterations`, `_head_difference`,
  `_budget_difference`, `_listing_difference`, `RUN_MF6`; T6.5: `_null_space_model` (steady state
with WEL/RCH only: heads compared after removing the mean difference, `_head_difference(...; shift)`,
`_maxdiff`). Used by groups run, synthetic.
  T6.9: `_conversion_blocks` (cell-conversion messages of a listing), `_single_budget_replay`
  (`:single` budget vs a `:double` replay of the `:single` heads through the stepping API;
  `_rounding_bounds` also rounds the top/bottom offsets of convertible storage cells).
  T6.6: `_run_parity` on several models (each model's files, the exchange observation
  files; periodic exchange lists matched in file order in `_budget_difference`).
- **residual.jl**: `to_test_device`, `flux_residual` (device `flux_residual!` → (rel, rounding
  bound)), `residual_stats`, `host_flux_residual` (the T2.3 host loop, validates the kernel).
- **models/coupling_rde/**: synthetic coupling-like model (2 × 12 × 15, confined, steady +
  transient) with CHD, WEL, RCHA, RIV, DRN, EVT, EVTA; `generate.jl` writes it
  (riv_drn_evt parity test); test/unit/builder_test.jl builds it in memory.
- **models/wetdry/**, **models/disu_staggered/**, **models/afr/** (T6.1): synthetic models
  run with mf6 at test time (groups assembly, run; disu_staggered also bmi): wetting and drying (2×1×12, VARIABLECV
  DEWATERED, PERCHED, REWET, convertible storage, RCH moving to layer 2), DISU with staggered
  horizontal connections (IHC 2) and convertible cells, WEL AUTO_FLOW_REDUCE (fraction and
  FLOW_REDUCTION_LENGTH with AUTO_FLOW_REDUCE_AUXNAME).
- **models/xt3d/** (T6.5): synthetic XT3D model (3×6×7 DIS, uneven layers, an IDOMAIN 0 cell,
  K/K22/K33 with ANGLE1–3, CHD, WEL, GHB, RCH, steady + transient STO); `generate.jl` writes
  it; run with mf6 at test time (group xt3d; also the kernel-hygiene workload).
- **models/ts_tas/** (T6.3, replaces models/fhb_const): synthetic time-series model (1×5×10,
  3 transient periods): TS6 LINEAR/STEPWISE/LINEAREND with SFACS, DNODATA and several series
  per file (CHD heads, WEL rate and AUXMULTNAME multiplier, GHB conductance), TAS6 RCHA
  (LINEAR, zeroed by a later block) and EVTA (STEPWISE); run with mf6 at test time (groups
  assembly, run).

## Benchmarks (`benchmark/`, workspace project)

- **synthetic.jl** (T3.4): module `SyntheticModels`, the synthetic coupling-model
  generator (tests `include` it; also a script). `SYNTHETIC_PRESETS` (tiny, indus,
  continental, global5m, regional30s, global30s), `synthetic_fields` (counter-based noise
  `value_noise!`, land mask + ocean rim, DEM, log-normal K, priority-flood
  `flow_accumulation`), `synthetic_simulation` (builder API → `SimulationSpec`: CHD rim,
  RIV, DRN, EVT, RCH, STO; steady + 12 daily periods, `ntransient`; `output = :none` = OC without records for mf6), `synthetic_ims` (`:parity`,
  `:production`), `write_synthetic`, `synthetic_counts` / `scaled_counts`, device memory
  `device_bytes` / `memory_coefficients` / `estimate_device_bytes`, `synthetic_main`
  (benchmark driver). Tests: synthetic (unit + parity vs mf6). Results: docs/dev/scaling.md.
- **hardware.jl** (T0.4): hardware baseline (stream, SpMV, dot, atomics).
- **run.jl** (T4.1): benchmark harness: indus / continental / global5m, steady state + 30
  daily steps (`synthetic_simulation(...; ntransient)`), on a backend in `:double` and
  `:single`, with `precond=` preconditioners (wall time per step, iterations, pool peak memory, SpMV GB/s, TimerOutputs
  time split through `SyncTimer`) or with mf6 (serial, per-step times from stdout).
  T4.3: `precond=amg`, `amg=key:value,...` (`AMGOptions`), AMG hierarchy/setup table.
  Results: benchmark/results/<date>.md.

## GPU tests (`test/gpu/`, T3.1)

- **runtests.jl**: GPU entry, the full suite with `MFGPU_TEST_BACKEND` (default rocm);
  every parity test runs on the GPU, run and assembly parity also compare with the CPU.
- **backend_gpu_test.jl**: backend registry on the GPU (group backend).
- **kernel_hygiene_test.jl** + **kernel_hygiene_workload.jl**: rule 5 (group
  kernel_hygiene, ROCm only): a subprocess runs `:single` work under
  `AMDGPU.@device_code_llvm` with Float64 and with Float32 accumulation and reports the
  `double`s per kernel; every `gpu_*` kernel of the package must be compiled by the
  workload. Exempt with Float64 accumulation: `mapreduce_kernel!`,
  `finalize_reduction_kernel!`.
