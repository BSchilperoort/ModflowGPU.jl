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
