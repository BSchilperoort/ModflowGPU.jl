# Global inputs of the coupled run (T5.5a checks, for T5.5b)

Checked on 2026-11-14 (card T5.5a) with `examples/global/check_inputs.jl` of the patched
ModflowCoupling (patch 0005), which reads every file through ModflowCoupling's input code
(coordinates matched within 1e-3°), builds the package parameters and compares the grids
of MODFLOW, the mGV parameters, the forcing and the routing file. The draft config for
T5.5b is `examples/global/global.toml` (patch 0005); the log of the run is quoted below.
Nothing in `/home/bart/data` was changed.

```sh
cd reference/Modflow-coupling
julia --project=examples/mgv_coupling examples/global/check_inputs.jl examples/global/global.toml \
    /home/bart/data/mGV/vic_global_5min_params_lossless.nc \
    /home/bart/data/mGV/WFDE5_v3.1_5arcmin_land_1979.zarr \
    /home/bart/data/mGV/routing/routing_global_5min.nc
```

## One grid for all inputs

Every MODFLOW file in `/home/bart/data/modflow/global_data`, the mGV parameter file
`vic_global_5min_params_lossless.nc`, the land-only forcing and the generated routing file
are on the same 4320 × 1680 grid: 5′ cells, longitude −179.958 … 179.958, latitude
−55.958 … 83.958 (84 N to 56 S), stored latitude-ascending (the MODFLOW `*_oc.nc` files
store `(lat, lon)`, read as `(lon, lat)`). Coordinates agree within 5.1e-6° (Float32
labels of the parameter file and the head files against Float64 ones). ModflowCoupling's
orientation audit passes (cell area follows cos(latitude), DEM correlates with the
steady head, bankfull discharge > 10 m³/s only on land). Unlike the Indus files
(`NOTES.md` in the data directory) no latitude labels are wrong here.

## File choices

| Role | File | Why |
| --- | --- | --- |
| `ibound` | `ibound_oc_nogl.nc` | VIC active domain without Greenland (`coupled_domain_rebuild.qa.json`): 2,188,551 land cells (idomain 1), 173,727 CHD cells (idomain 2, a 2-cell ring in the ocean, disjoint from land); the files below are built on this domain |
| `landmask` | `landmask_oc_nogl.nc` | equals the nogl domain |
| `catchment` | `ldd_oc_nogl.nc` | catchment ids (20,385), masked to the nogl domain (not flow directions, as `ldd_indus_updated.nc`) |
| `z0_floodplain`, `min_dem` | `Z0_floodplain_oc_v2clamped.nc`, `min_dem_oc_v2clamped.nc` | v2 surfaces on the nogl domain, finite on every land cell (`coupled_surfaces_v2.qa.json`; Z0 differs from the original by a median 2.8 m, p95 83.6 m) |
| `qbank` | `qbank_daily_1979_2019_natpcv_gwm_v1.nc` | the bankfull discharge of the coupled steady state (median 9.4 × the 1979–2010 file, rivers 3.1 × wider) |
| `cpsurface`, `cprate`, `cpdistinct` | `capillary_rise_oc_v2.nc` | **v2, not v3**: v3's `cprate` is v2's times cell_area / (DELR·DELC) (checked: max relative difference 5.7e-8; v3 `cprate_physical` equals v2 `cprate`), and ModflowCoupling's `evt_parameters` applies that factor itself, so v3 would apply it twice. `cpsurface` and `cpdistinct` are identical in v2 and v3 |
| `steady_head` | `initial_head_gv2p41_5min_nogl.nc` | steady state on the nogl domain, finite on every land cell, **but not on the CHD cells** (see below) |
| others | `dem_ini_oc.nc`, `aqdepth_ini_oc.nc`, `cell_area_oc.nc`, `ksat_log_oc.nc`, `ksat_l1/l2_conf_log_oc.nc`, `spe_yi_inp_oc.nc`, `riv_slope_oc.nc`, `KQ3_oc.nc` | single variants |

Not used: `ibound_oc.nc`, `landmask_oc.nc`, `ldd_oc.nc` (the domain with Greenland,
2,288,857 land cells; the v2 files have no values on the 100,306 Greenland cells),
`capillary_rise_oc.nc` and `_updated.nc` (earlier soil bundles), `capillary_rise_oc_v3.nc`
(above), `bankfull_discharge_1979_2010_vicwur5min.nc` and `bankfull_discharge_daily_1979_2018.nc`
(earlier qbank), `daily_bankfull_mf_ss_gwl_5minDRN.nc` (the earlier steady state, not on
the nogl domain), `Clone_05min_oc.*` and `conflayers_oc.nc` (not ModflowCoupling inputs).

## What does not line up

1. **CHD heads.** `initial_head_gv2p41_5min_nogl.nc` is NaN on all 173,727 CHD cells
   (its QA file says so: `ibound_cells_without_head = 173727`), and the earlier
   `daily_bankfull_mf_ss_gwl_5minDRN.nc` has a head on 1 of them. ModflowCoupling takes
   the CHD heads from `steady_head` and stops (`build_parameters`: "347454 CHD cell heads
   are not finite"). The ring lies in the ocean, so sea level (0 m) is the natural head:
   with 0 m on those cells every check passes. T5.5b needs one of: a steady-head file
   with 0 on the ring, or a ModflowCoupling option for the CHD head (not added here;
   owner decision).
2. **Greenland.** mGV's parameter file (and the land-only forcing) has 2,288,857 active
   cells: the nogl domain plus the 100,306 Greenland cells (lon −73.0 … −11.4, lat
   59.7 … 83.6; one of them is a MODFLOW CHD cell). Every MODFLOW land cell is an mGV
   cell. The coupling gives the Greenland cells zero baseflow and capillary rise
   (`grid_maps` warns); restricting mGV to the nogl domain would need a parameter file
   with a different `run_cell`.
3. **Missing values** on active land cells, handled by ModflowCoupling's defaults as on
   the Indus: `aqdepth` 39.5 % (200 m), `ksat_log` 2.9 %, `spe_yi` 3.7 %, `kq3` 3.0 %
   (no DRN record there), `cprate` 10.5 % NaN (no EVT record).
4. **Forcing.** `/media/bart/Data/Work/SWIFT/WFDE5_v3.1_5arcmin.zarr` (the per-variable
   gridded stores) was not mounted on 2026-11-14 and could not be checked.
   `/home/bart/data/mGV/WFDE5_v3.1_5arcmin_land_1979.zarr` is a land-only copy of it for
   1979 (attribute `source`), made for mGV (`scripts/convert_forcing_landonly.jl` of mGV):
   one group with all seven variables (`lwdown`, `prec`, `psurf`, `swdown`, `tair`, `vp`,
   `wind`) as int16 `(time, cell)` arrays with scale/offset, 365 days, and `lon_index` /
   `lat_index` of its 2,288,857 cells, which are exactly mGV's active cells. mGV reads it
   with one `forcing_file` entry (the per-variable entries default to it; variable names
   are mGV's defaults). It is the better input for T5.5b's one year: 5.0 GB on the
   internal disk against the gridded stores on an external one, and mGV's land-only path
   reads only the active cells. For more years, either more land-only years or the
   gridded stores with one entry per variable (`precipitation_file = ".../prec"`, …;
   `{year}` in the path when they are split by year).

5. **Routing.** The generated file (`benchmark/results/2026-11-14-T5.5a.md`, "Global
   routing file") has 2,286,186 HydroSHEDS network cells; 2,841 active mGV cells have no
   HydroSHEDS land data, and mGV stops on them ("Routing channel length is not positive
   and finite"), so they were added as one-cell basins. Final file: 2,289,027 cells, every
   mGV active cell covered, 170 cells outside mGV's domain (ignored by mGV).

## Packages and size (nogl domain, from the checks)

| | records |
| --- | --- |
| model nodes (2 layers) | 4,724,556 |
| CHD (per layer) | 173,727 |
| RCH | 2,362,278 (every active cell) |
| RIV | 2,162,135 (604,512 wider than 30 m, the rest drain-like) |
| DRN | 2,123,032 |
| EVT | 1,728,678 |

No package has two records on one cell (ModflowCoupling builds one record per cell), so the
device coupling's one-record-per-cell rule (T5.4, `record_map`) holds for these inputs.
The synthetic `global5m` of T5.5a (6.59 M nodes) is 40 % larger than this model.

## Derived inputs of the global run: north of 60°N masked (T5.5b)

PLAN.md decision 10: Greenland and the Arctic are not modelled; everything with a cell
centre north of 60°N is masked out in both models. Made on 2026-11-15 with
`examples/global/make_60N_inputs.jl` of the patched ModflowCoupling (patch 0006), which
copies each original and changes the copy (nothing that existed under `/home/bart/data`
was changed):

```sh
cd reference/Modflow-coupling
julia --project=examples/mgv_coupling examples/global/make_60N_inputs.jl /home/bart/data/mGV/global_60N
```

| File under `/home/bart/data/mGV/global_60N/` | From | Change |
| --- | --- | --- |
| `vic_global_5min_params_lossless_60N.nc` (1.38 GB) | `vic_global_5min_params_lossless.nc` | `run_cell` = 0 north of 60°N |
| `modflow/ibound_oc_nogl_60N.nc` | `ibound_oc_nogl.nc` | `ibound` = 0 north of 60°N (no-flow at the cut) |
| `modflow/landmask_oc_nogl_60N.nc` | `landmask_oc_nogl.nc` | 0 north of 60°N |
| `modflow/initial_head_gv2p41_5min_nogl_chd0_60N.nc` | `initial_head_gv2p41_5min_nogl.nc` | sea level (0 m) on the 123,983 remaining CHD ring cells, which had no head; NaN north of 60°N |

The configs of the run are `examples/global/global_60N.toml` (MODFLOW; these three files
by absolute path, the other inputs as in `global.toml`) and `mgv_global_60N.toml` (mGV).
`run_global.jl` also sets every other MODFLOW input to NaN north of 60°N in memory, so
that no package has a record there and the orientation audit (bankfull discharge only on
land) passes.

Not derived, because the models cut them themselves:

- **Routing**: mGV `main` builds its network on the full grid and keeps the active cells;
  a cell that flows into a removed cell becomes an outlet (`RoutingState`). So
  `routing_global_5min.nc` is used as it is.
- **Forcing**: the land-only store holds all 2,288,857 cells; mGV reads the rows of its
  active cells only (`land_rows`), so the 1979 store is used as it is.

Counts (`make_60N_inputs.jl` output):

| | before | removed north of 60°N | remaining |
| --- | --- | --- | --- |
| mGV active cells | 2,288,857 | 575,055 | 1,713,802 |
| MODFLOW land cells (idomain 1) | 2,188,551 | 474,809 | 1,713,742 |
| MODFLOW CHD ring cells (idomain 2) | 173,727 | 49,744 | 123,983 |

- Routing links cut: 646 links from cells south of 60°N into removed cells (they become
  outlets: the northward rivers end at the cut), 598 from removed cells into the remaining
  domain (the southward rivers lose that inflow).
- Channel length (as mGV sets it: `cell_dist`, or `sqrt(cell_area)` where that is not
  positive): shortest 1029 m and 146,207 channels below 3.6 km on the whole domain; south
  of 60°N the shortest is 4639 m and none is below 3.6 km, the length at which mGV's
  explicit routing (600 s sub-steps, 6 m/s cap) starts to create water (T5.5b card).
- The 60 mGV cells south of 60°N without a MODFLOW land cell get no feedback (as before).

**CHD heads at sea level.** The DEM (`dem_ini_oc.nc`, no datum attribute) has zero at
mean sea level as far as the data show: on the 57,972 land cells next to the CHD ring its
median is 16.5 m (p05 0.9 m, p25 5.5 m), and 1150 of them (1.98 %, 613 below −5 m) lie below
0 m, which is what a sea-level datum gives on low coasts (which coasts these are was not
checked); the lowest land cell is −413.9 m, about the level of the Dead Sea shore. An
offset datum would move the whole coastal distribution. These 1150 cells are below the ring head by
construction; that is real (they are below sea level), not an artefact. The provided
steady head lies below 0 m on 25.7 % of the coastal cells.

**Forcing of the gridded store.** `/media/bart/Data/Work/SWIFT/WFDE5_v3.1_5arcmin.zarr`
is one zarr v3 store with the seven variables as `(time, lat, lon)` int16 arrays
(chunks 24 × 240 × 240, Blosc zstd) with CF `scale_factor`, `add_offset` and `_FillValue`
attributes, 16,802 days (1979-01-01 to 2024-12-31), latitudes −55.958 … 83.958 (the
parameter grid). mGV `main` (42aebaa) opens it (dimension check, time axis), but its
gridded reader (`load_block!` for `ZarrForcingVar`) reads the int16 values into Float32
with `Zarr.readblock!` and applies neither the scale and offset nor the fill value (only
the land-only reader does): the first 24 days of `prec` come back as raw values
−32768 … −22161 instead of millimetres. So mGV `main` cannot use this store, and T5.5b
cycles the land-only 1979 store (the fallback). Using the full record needs either a
fix in mGV's gridded reader (owner's change; mGV is not patched here) or land-only
stores of more years (`scripts/convert_forcing_landonly.jl` of mGV; about 3.8 GB per
year for the masked domain, 175 GB for 1979–2024).
