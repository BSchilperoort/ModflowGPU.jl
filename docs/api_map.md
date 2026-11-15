# API map of `src/`

Update this file in every card's merge; skip `src/input/dfn/definitions.jl` (generated).

One entry per file, in include order (`src/ModflowGPU.jl`): purpose, key types and
functions, and the test groups (`MFGPU_TEST_GROUPS`, see `test/runtests.jl`) that cover
it. Underscore-prefixed helpers are omitted.

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
- **input/options.jl**: OPTIONS / DIMENSIONS / TIMING blocks through the DFN schema.
  `OptionSet`, `read_options`, `option`, `has_option`, `option_line`. Tests:
  simulation_loader.
- **input/supported.jl**: what the engine supports; the loader checks input against it.
  `MF6UnsupportedError`, `SUPPORTED_OPTIONS`, `SUPPORTED_VALUES`, `SUPPORTED_FEATURES`,
  `UNSUPPORTED_PACKAGES` / `UNSUPPORTED_MODELS`, `check_options`, `check_value`,
  `check_feature` (T6.4: `:disv`, `:disu` supported; DISV/DISU options). Tests: simulation_loader, input_lists.

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
  `npf_setup(::GwfSpec)`, device struct `NpfModel{T}` (`NpfModel(backend, T, setup)`, Adapt), `CCOND_*` averaging codes. T6.4: connection
  normals from `anglex` by cos/sin on DISV/DISU (exact axes on DIS), the DISU ANGLDEGX
  checks of `prepcheck`/`npf_ar` (`_check_disu_angldegx`). Tests:
  npf (parity vs libmf6 CONDSAT, also DISV/DISU), disv_disu.
- **gwf/ic.jl**: IC (gwf-ic.f90 `ic_ar`). `initial_heads` (starting heads per reduced
  node as offsets, Float64 host or device copy in `T`). Tests: sto.
- **gwf/sto.jl**: confined storage (gwf-sto.f90 `sto_rp`/`sto_fc`/`sto_cq`/`sto_bd`,
  GwfStorageUtils.f90 `SsCapacity`/`SsTerms`). `storage_capacity`, `check_storage_spec`
  (rejects convertible cells with a transient period, T6.1, and TVS6), `StoragePackage{T}`
  (device `rho = sc/delt`, `strgss`; Adapt), `apply_period!` (steady/transient), `set_timestep!`
  (`rho` from Float64 `delt`), `add_to_system!` (diagonal and `b`), `calc_storage_rates!`
  (STO-SS), `add_storage_flowja!`, `storage_budget` (rin/rout, host Float64). Tests: sto
  (parity: residual test on fhb with constant stresses, mf6 run at test time).

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
  boundaries (fast path vs full application).
- **boundaries/chd.jl**: CHD (gwf-chd.f90; see `docs/mf6_notes.md`). `apply_chd_heads!`,
  `calc_chd_rate!`. Tests: boundaries (parity vs MF6 CBC).
- **boundaries/wel.jl**, **rch.jl**, **ghb.jl**: WEL, RCH (list and RCHA arrays, cell
  area), GHB formulations (`formulate_record` methods). Tests: boundaries.
- **boundaries/riv.jl**, **drn.jl**, **evt.jl** (T2.4): RIV (`riv_cf`, stage/rbot as
  offsets, `riv_ck`), DRN (`drn_cf` with AUXDEPTHNAME drainage depth, device-safe
  `linear_saturation` = `sQuadraticSaturation` with eps 0), EVT/EVTA (`evt_cf`: NSEG
  segments, PETM0 / SURF_RATE_SPECIFIED, `check_pxdp`; `bound` = surface, rate, depth,
  petm0, pxdp…, petm…). Tests: riv_drn_evt (parity: residual and CBC flows on
  test/models/coupling_rde, mf6 run at test time).

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
  `_ArgMaxAbsOp` as the first op also returns the first index of the largest magnitude,
  `_mapreduce_loc!`; results land in host-mapped memory where `host_mapped_zeros` allows).
  Adapt rules on `KrylovSolver` and `Reducer` (a new one on the target). IMS mapping in
  docs/mf6_notes.md, accuracy and timing in docs/precision.md. Tests: krylov.
  `KrylovSettings` has `preconditioner` and `precond_options`, `KrylovSolver` a `precond`
  field. Non-Jacobi preconditioners run through `precond_apply!` between non-fused kernels
  (`_cg_init_nf`, `_cg_update_nf`, `_bcgs_direction_nf`, `_bcgs_half_nf`; section
  `precond_apply`; CG's `r·z` fused by `_precond_apply_dot!` where the preconditioner
  does); `_precond_record!` / `_precond_retry!` after a solve (a failed solve is repeated
  once after an AMG rebuild); `KrylovSolver(...; colors)` passes a host colouring.
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
  kernels (`_precond_apply_dot!`). Tests: krylov.
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
  `TimeStep` (kper, kstp, delt, pertim, totim, endofperiod), `time_steps(tdis)`. Tests:
  picard.
- **gwf/model.jl**: one model on a device. `GwfModel{T}` (spec, conn, NpfSetup, CSR,
  NpfModel, LinearSystem, boundary packages, STO, ibound, offset heads `h`/`hold`, `res`,
  `δ`, `flowja`; Adapt, e.g. `adapt(backend, m)`, T3.1), `apply_period!(::GwfModel, kper)` (once per period: package period
  data, STO, CHD ibound, rebuilt only when a CHD package has new cells), `start_timestep!` (`rho`, `hold ← h`, CHD heads), `formulate!`
  (per-package formulation, then `formulate_system!`; off-diagonals rewritten only when
  `ibound` or `symmetric` changed, field `offdiag`, T4.5), `absolute_heads` (host Float64 `r + h'`). Tests:
  picard, run.
- **solver/underrelax.jl**: `sln_underrelax`. `UnderRelaxation` (SIMPLE, COOLEY host
  scalars, DBD device state), `underrelax!` (`relax_update_kernel!`,
  `dbd_update_kernel!`); Adapt. Tests: picard (vs a host port), run (variants vs MF6).
- **solver/picard.jl**: outer loop (`sln_ca`/`solve`; PLAN.md 3.7 rule 3). `PicardSettings`
  (from `ImsSpec`, `closures` overrides, Float32 floors `FLOAT32_CLOSURES`, backtracking
  warning), `PicardSolver` (KrylovSolver, UnderRelaxation, Reducer; Adapt), `solve_timestep!` →
  `OuterResult` (converged, outer and inner iterations, hncg and its node `lrch` from one
  argmax reduction, the dvclose/rclose used); `OuterState` + `outer_iteration!` (one outer
  iteration, T3.3; `solve_timestep!` loops over it; T5.3/T6.0: in Float32, once a
  correction meets OUTER_DVCLOSE and only the inner test failed, `OuterState.noise_floor`
  holds INNER_DVCLOSE >= `OuterState.dvfloor` = eps32·max|h'| for the rest of the step).
  Tests: picard, run, stepping.
- **budget.jl**: `gwf_cq`/`gwf_bd`. `npf_flowja!` (one work item per entry, T4.5), `calc_flows!` (NPF, STO, packages with
  `bnd_cf` + `simvals`, CHD `calc_chd_rate!`, `csr_diagsum`), `BudgetTerm`, `ModelBudget`
  (`total_rate_in`/`_out`, `total_cum_in`/`_out`, `percent_discrepancy`,
  `cum_percent_discrepancy`), `BudgetAccumulator` (Adapt), `model_budget!` (device reductions,
  host cumulative volumes), `budget_text` (CBC text per package). Tests: picard, run.

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
- **output/oc.jl** (T2.8): output control and run output. `oc_selected` (OC PERIOD
  blocks, ALL/FIRST/LAST/FREQUENCY/STEPS), `ModelOutput` (open HDS/CBC/listing files),
  `write_heads`, `write_budget` (STO-SS, FLOW-JA-FACE, package lists as MF6),
  `write_head_listing` (PRINT HEAD). Tests: picard, run.

## Running a simulation

- **run.jl** (T2.8): `ModflowGPU.run(simdir or SimulationSpec; backend, precision,
  outdir, closures, write_output, preconditioner, precond_options)` (on the stepping API since T3.3) → `RunResult` (model, precision types, `StepResult`s,
  `ModelBudget`s, final heads, output paths); `ConvergenceError`. One model, one IMS
  solution. Tests: run (parity vs MF6 on Tier 0, fhb_const, coupling_rde and IMS
  variants; flopy read-back with MFGPU_FLOPY=1). `_write_step` (step output) copies the
  heads to the host only when OC saves or prints them (T5.2).

- **stepping.jl** (T3.3): libmf6/XMI-like stepping (PLAN.md 3.6). `Simulation`,
  `initialize`, `prepare_time_step`, `prepare_solve`, `solve` (one outer iteration),
  `finalize_solve`, `finalize_time_step`, `do_time_step`, `update`, `finalize`,
  `get_current_time` / `get_start_time` / `get_end_time` / `get_time_step`, `StepResult`;
  T5.4: `finalize(sim; free = true)` frees the device arrays (stage `:freed`; `run` uses
  `free = false`), `device_bound!` (the device BOUND write window between
  `prepare_time_step` and `prepare_solve`, enforced; `prepare_solve` checks the written
  records on the device: finite, RIV stage ≥ bottom and conductance ≥ 0,
  `device_bound_check_kernel!`);
  addresses `get_var_address`, `var_addresses`, `input_addresses`, `get_value_ptr` (host
  Float64 mirror, absolute; `device = true` for the engine array), `get_value`,
  `set_value` (X, BOUND). Mirror sync rules in the file header; BOUND and NODELIST
  mirrors are versioned (T4.6: copied only when the engine values changed, dirty check by
  memcmp). T5.2: device BOUND writes between `prepare_time_step` and `prepare_solve` (the
  contract ModflowCoupling's device-resident coupling uses; header of the file). Tests: stepping.
- **bmi.jl** (T3.3): BasicModelInterface methods on `Simulation` (`BMI.initialize`,
  `update`, `update_until`, time, variables, `get_value(_ptr)`, `set_value(_at_indices)`,
  grid 1 = model grid). Tests: stepping.

## Test helpers (`test/`, always loaded)

- **testutils.jl**: `for_each_precision`, `write_input`, `input_error_of`.
- **input_files.jl**: `REFERENCE_DIR`, `in_scope_examples`, `simulation_input_files`,
  `model_package_files`; `parity_examples` (T6.0: Tier 0–2 plus `EXTRA_PARITY_EXAMPLES`,
  ex-gwf-lgrv-gc/-gr; `MFGPU_TEST_LGRV_GR=0` drops -gr; T6.4: plus the Tier 5 examples),
  used by the assembly and run parity tests; `rad_disu_confined` (T6.4: ex-gwf-rad-disu with
  ICONVERT 0, written from the reference run at test time).
- **libmf6_oracle.jl**: libmf6 subprocess oracle. `LIBMF6_ENV`, `libmf6_available`,
  `run_libmf6_script` (validation/libmf6_connectivity.jl, libmf6_first_solve.jl), `libmf6_connectivity`, `write_dis_simulation` (synthetic DIS sims; DISV/DISU since T6.4).
- **grids.jl**: `_synthetic_coupling_dis`, `_synthetic_passthrough_dis`, `_rel_close`,
  `_dis`, `_Lcg`/`_next!`; T6.4: `_dis_as_disv`, `_dis_as_disu` (the DISV/DISU description of
  a DIS grid, same user node numbers).
- **run_parity.jl** (T3.4, moved from parity/run_test.jl): `_run_parity(name, rundir)`
  (ModflowGPU vs MF6 output in `rundir`: heads, CBC, listing tables, both presets, GPU vs
  CPU; returns the numbers per preset; `single_preconditioner` for `:single`, T6.0; `single_budget_tol` and list records matched by package, T6.4), `_mf6_iterations`, `_head_difference`,
  `_budget_difference`, `_listing_difference`, `RUN_MF6`. Used by groups run, synthetic.
- **residual.jl**: `to_test_device`, `flux_residual` (device `flux_residual!` → (rel, rounding
  bound)), `residual_stats`, `host_flux_residual` (the T2.3 host loop, validates the kernel).
- **models/coupling_rde/**: synthetic coupling-like model (2 × 12 × 15, confined, steady +
  transient) with CHD, WEL, RCHA, RIV, DRN, EVT, EVTA; `generate.jl` writes it
  (riv_drn_evt parity test); test/unit/builder_test.jl builds it in memory.
- **models/fhb_const/**: ex-gwf-fhb with the CHD/WEL time series replaced by constants per
  period (sto parity test).

## Benchmarks (`benchmark/`, workspace project)

- **synthetic.jl** (T3.4): module `SyntheticModels`, the synthetic coupling-model
  generator (tests `include` it; also a script). `SYNTHETIC_PRESETS` (tiny, indus,
  continental, global5m, regional30s, global30s), `synthetic_fields` (counter-based noise
  `value_noise!`, land mask + ocean rim, DEM, log-normal K, priority-flood
  `flow_accumulation`), `synthetic_simulation` (builder API → `SimulationSpec`: CHD rim,
  RIV, DRN, EVT, RCH, STO; steady + 12 daily periods, `ntransient`; `output = :none` = OC without records for mf6), `synthetic_ims` (`:parity`,
  `:production`), `write_synthetic`, `synthetic_counts` / `scaled_counts`, device memory
  `device_bytes` / `memory_coefficients` / `estimate_device_bytes`, `synthetic_main`
  (benchmark driver). Tests: synthetic (unit + parity vs mf6). Results: docs/scaling.md.
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
