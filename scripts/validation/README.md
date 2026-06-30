# Catalog blinding validation scripts

These scripts validate the catalog-level blinding workflows against the
authoritative LSS implementations and, at the final layer, against
`desi-clustering` measurement products.

The BAO/AP scripts validate `CatalogBAOBlinder` against the LSS BAO/AP
catalog-blinding workflow. The RSD scripts validate `CatalogRSDBlinder` both in
isolation and in the combined BAO/AP + RSD saved-catalog ladder.

The comparisons are intentionally apples-to-apples:

- **BAO/AP LSS reference branch**: LSS BAO/AP redshift shifter and LSS
  catalog-making functions.
- **BAO/AP desiblind branch**: `CatalogBAOBlinder` used as the redshift shifter,
  then the same downstream LSS catalog-making functions.
- **RSD LSS reference branch**: LSS BAO/AP redshift shifter, LSS catalog-making
  functions, `IFFTrsd` reconstruction, and `LSS.apply_zshift_RSD`.
- **RSD desiblind branch**: `CatalogBAOBlinder`, the same downstream LSS
  catalog-making and reconstruction steps, then `CatalogRSDBlinder`.

The BAO/AP validation asks whether replacing the LSS BAO/AP redshift remapping
with `CatalogBAOBlinder` changes any downstream catalog or measurement product.
The RSD saved-catalog validation asks the same question for the combined
BAO/AP + RSD workflow.

## Environment

On Perlmutter:

```bash
cd /global/homes/u/uendert/repos/desi/desiblind
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module unload desi-clustering || true
export PYTHONPATH=/global/homes/u/uendert/repos/desi/desi-clustering:${PYTHONPATH:-}
```

Validation products are written under `$SCRATCH/desiblind_lss_validation/` by
default.

## BAO/AP validation ladder

Run the BAO/AP scripts in this order when rebuilding confidence from scratch.

1. Redshift remapping only:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_redshift_shift.py
   ```

2. LSS `n(z)` / `WEIGHT_SYS` compatibility:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_nz_weight.py
   ```

3. Saved full-catalog workflow through final saved `Z`, `n(z)`, `WEIGHT_FKP`,
   and `WEIGHT_SYS`:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_saved_catalog.py
   ```

4. LSS `mkclusdat` clustering-data generation:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_mkclusdat.py
   ```

5. LSS `mkclusran` clustering-random generation:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_mkclusran.py
   ```

6. GC split of clustering data and random catalogs:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_split_gc.py
   ```

7. Multi-random production-like validation. This builds the expensive full-data
   branch once and loops over random numbers:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_multi_randoms.py --rannums 0-17 --cleanup-random-files
   ```

   LSS convention: random 1 is used for `mknz_full` area normalization while
   `mkclusran` processes the requested random number.

8. Final statistics through `desi-clustering`:

   ```bash
   python scripts/validation/validate_catalog_bao_desi_clustering_stats.py \
     --stats mesh2_spectrum \
     --input-dir /pscratch/sd/u/uendert/desiblind_lss_validation/production-54906853-0
   ```

   For `particle2_correlation`, use a GPU interactive node because the
   `desi-clustering` correlation path uses `cucount.jax`.


## RSD validation ladder

Run the RSD scripts in this order when rebuilding confidence from scratch.
The first two isolate the RSD transform; the later scripts validate the combined
BAO/AP + RSD saved-catalog workflow and final measurements.

1. Direct RSD redshift shift against `LSS.blinding_tools.apply_zshift_RSD`:

   ```bash
   python scripts/validation/validate_catalog_rsd_lss_redshift_shift.py
   ```

2. Real-subset `IFFTrsd` reconstruction plus the same direct RSD shift check:

   ```bash
   python scripts/validation/validate_catalog_rsd_lss_reconstruction_shift.py
   ```

3. Combined BAO/AP + RSD saved-catalog ladder through `mkclusdat`, `mkclusran`,
   GC split, `IFFTrsd` reconstruction, and final RSD shift:

   ```bash
   python scripts/validation/validate_catalog_rsd_lss_saved_catalog.py
   ```

4. `desi-clustering` saved-catalog driver reconstruction backends:

   ```bash
   python scripts/validation/validate_catalog_rsd_desi_clustering_driver.py \
     --backends pyrecon jaxrecon
   ```

   The direct-`pyrecon` backend is the non-LSS, reference-compatible candidate.
   The `jaxrecon` backend is a speed/on-the-fly candidate; the driver matches
   pyrecon mesh-center and random-threshold conventions before running JAX.

5. Final Pk/xi measurements through `desi-clustering.compute_stats_from_options`:

   ```bash
   python scripts/validation/validate_catalog_rsd_desi_clustering_stats.py
   ```

6. Executed walkthrough notebook:

   ```bash
   scripts/validation/catalog_rsd_validation.ipynb
   ```

## RSD validation notebook

The RSD validation ladder is summarized in an executable notebook:

```bash
scripts/validation/catalog_rsd_validation.ipynb
```

It loads the RSD validation JSON summaries, checks the direct/reconstruction and
saved-catalog deltas, visualizes the reconstructed RSD displacement scaling, and
overlays the final `desi-clustering` Pk/xi products.

## RSD direct-transform validation

The first RSD validation step is intentionally narrow: compare the generic
`CatalogRSDBlinder` redshift transform against
`LSS.blinding_tools.apply_zshift_RSD` for the same observed clustering catalog,
reconstructed-realspace catalog, and derived `fgrowth_blind` value.

```bash
python scripts/validation/validate_catalog_rsd_lss_redshift_shift.py
```

By default this runs a self-contained toy IO comparison. To validate real LSS
products, pass both the observed clustering data catalog and the corresponding
reconstructed-realspace catalog:

```bash
python scripts/validation/validate_catalog_rsd_lss_redshift_shift.py \
  --data-catalog /path/to/LRG_NGC_clustering.dat.fits \
  --realspace-catalog /path/to/LRG_NGC_clustering.IFFTrsd.dat.fits
```

A real-subset reconstruction validation is also available. It draws
reproducible subsets from real LSS clustering data/random catalogs, runs the LSS
reconstruction helper in `convention='rsd'` mode, and then performs the same
LSS-vs-desiblind RSD redshift-shift comparison:

```bash
python scripts/validation/validate_catalog_rsd_lss_reconstruction_shift.py
```

For this script, prefer an interactive CPU node rather than a login node, even
for modest subsets.

Current small real-subset references:

- NGC, 2000 data rows / 10000 random rows:
  `/pscratch/sd/u/uendert/desiblind_lss_validation/rsd-recon-shift-20260626-201203-1234847/summary.json`
- SGC, 2000 data rows / 10000 random rows:
  `/pscratch/sd/u/uendert/desiblind_lss_validation/rsd-recon-shift-20260626-201300-1235312/summary.json`

Both have `max_abs_delta_Z = 0.0` for `LSS.apply_zshift_RSD` vs
`CatalogRSDBlinder` after the shared LSS reconstruction step.

The saved-catalog RSD validation extends this through BAO/AP saved full
catalogs, `mkclusdat`, `mkclusran`, GC splitting, `IFFTrsd` reconstruction, and
final RSD redshift shifting:

```bash
python scripts/validation/validate_catalog_rsd_lss_saved_catalog.py
```

Current small saved-catalog reference:

- NGC+SGC, 8000 full-data input rows / 20000 random rows:
  `/pscratch/sd/u/uendert/desiblind_lss_validation/rsd-saved-catalog-20260629-154659-1398399/summary.json`
- NGC+SGC, 50000 full-data input rows / 200000 random rows:
  `/pscratch/sd/u/uendert/desiblind_lss_validation/rsd-saved-catalog-20260629-160133-759367/summary.json`

This has `max_abs_delta_Z = 0.0` and zero deltas for the checked weight columns
for both final RSD-blinded NGC and SGC clustering data catalogs.

Current full-row one-random saved-catalog reference:

- full LRG data, full random 0, NGC+SGC, LSS reconstruction defaults:
  `/pscratch/sd/u/uendert/desiblind_lss_validation/rsd-saved-catalog-fullrow-ran0-lssrecon-20260630-132632-308997/summary.json`

This has `status = PASS`, `max_abs_delta_Z = 0.0` for both NGC and SGC, and
zero deltas for the checked weight columns.

The `desi-clustering` saved-catalog driver backend validation is run with:

```bash
python scripts/validation/validate_catalog_rsd_desi_clustering_driver.py \
  --backends pyrecon jaxrecon
```

Current driver-backend reference using the 2000-row NGC/SGC reconstruction
subsets:

- `/pscratch/sd/u/uendert/desiblind_lss_validation/rsd-desi-clustering-driver-20260630-155656-1022871/summary.json`

This has direct-`pyrecon` max final blinded-Z deltas of `2.46e-09` (NGC) and
`2.55e-09` (SGC) relative to the LSS/pyrecon reference. After matching pyrecon's
mesh-center and random-threshold conventions, `jaxrecon` matches at the same
level: `2.46e-09` (NGC) and `2.55e-09` (SGC). Latest matched-backend reference:
`/pscratch/sd/u/uendert/desiblind_lss_validation/rsd-desi-clustering-driver-20260630-185228-1058512/summary.json`.

Final measurement-layer checks through `desi-clustering` are run with:

```bash
python scripts/validation/validate_catalog_rsd_desi_clustering_stats.py
```

Current measurement-layer references using the larger saved-catalog RSD output:

- Pk / `mesh2_spectrum`, NGC+SGC, meshsize=256, boxsize=6000, kmax=0.13:
  `/pscratch/sd/u/uendert/desiblind_lss_validation/rsd-desi-clustering-stats-20260629-160601-1746031/summary.json`
- xi / `particle2_correlation`, NGC+SGC, smax=200, ds=5, nmu=40, 50000 staged random rows:
  `/pscratch/sd/u/uendert/desiblind_lss_validation/rsd-desi-clustering-stats-20260629-160817-905849/summary.json`

The Pk check has `max_abs_delta = 0.0`. The xi check has
`max_abs_delta = 3.25e-19`, consistent with numerical roundoff.

This is still a sampled validation ladder rather than a full production-scale
rerun, but it exercises the full intended integration path through saved
catalogs, RSD reconstruction, RSD blinding, and `desi-clustering` Pk/xi
measurement.

## NERSC interactive runs

Prefer interactive QoS for validation/test jobs when possible. Use a wrapper on
`$SCRATCH` rather than `/tmp`, because login-node `/tmp` is not visible from
compute nodes.

Example CPU run:

```bash
runner=$SCRATCH/desiblind_validation_run.sh
cat > "$runner" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
cd /global/homes/u/uendert/repos/desi/desiblind
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module unload desi-clustering || true
export PYTHONPATH=/global/homes/u/uendert/repos/desi/desi-clustering:${PYTHONPATH:-}
python scripts/validation/validate_catalog_bao_lss_multi_randoms.py --rannums 0-17 --cleanup-random-files
RUN
chmod +x "$runner"
srun --account=desi --qos=interactive --constraint=cpu --nodes=1 --ntasks=1 --cpus-per-task=32 --time=04:00:00 "$runner"
```

Example GPU run for `particle2_correlation`:

```bash
runner=$SCRATCH/desiblind_desi_clustering_xi_run.sh
cat > "$runner" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
cd /global/homes/u/uendert/repos/desi/desiblind
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
module unload desi-clustering || true
export PYTHONPATH=/global/homes/u/uendert/repos/desi/desi-clustering:${PYTHONPATH:-}
python scripts/validation/validate_catalog_bao_desi_clustering_stats.py \
  --stats particle2_correlation \
  --input-dir /pscratch/sd/u/uendert/desiblind_lss_validation/production-54906853-0 \
  --xi-smax 40 --xi-ds 10 --xi-nmu 8 --max-allowed-delta 1e-10
RUN
chmod +x "$runner"
srun --account=desi_g --qos=interactive --constraint='gpu&hbm80g' --nodes=1 --ntasks=1 --gpus-per-node=4 --cpus-per-task=32 --time=00:30:00 "$runner"
```

## What the final-statistics comparison means

`validate_catalog_bao_desi_clustering_stats.py` compares:

```text
LSS-produced blinded GC-split clustering catalogs
vs
CatalogBAOBlinder-produced blinded GC-split clustering catalogs
```

after both are measured by:

```python
clustering_statistics.compute_stats.compute_stats_from_options(...)
```

It then compares the resulting `desi-clustering` HDF5 outputs dataset by
dataset.

The final LSS clustering catalogs carry final `WEIGHT_FKP` but not `NX`; the
validation uses existing `WEIGHT_FKP` (`FKP_P0=None`) and does not add `NX`.
Its masking helper applies the usual `NX == 0` cut only if `NX` is present,
while preserving normal redshift and sky-region cuts.

Rendered validation walkthrough notebook:

```text
scripts/validation/catalog_bao_validation.ipynb
```

## Current production-like references

- Full random 0 split-GC output:
  `/pscratch/sd/u/uendert/desiblind_lss_validation/production-54906853-0/summary.json`
- Full randoms 1-17 multi-random output:
  `/pscratch/sd/u/uendert/desiblind_lss_validation/multi-randoms-20260623-213146-1005132/summary.json`
- Clean desi-clustering Pk output:
  `/pscratch/sd/u/uendert/desiblind_lss_validation/desi-clustering-stats-20260624-003911-726740/summary.json`
- BAO-range desi-clustering Pk output used by the rendered notebook plots
  (`meshsize=256`, `boxsize=6000`, `kmax=0.13`, `dk=0.005`):
  `/pscratch/sd/u/uendert/desiblind_lss_validation/desi-clustering-pk-bao-validated-20260625T130130Z/summary.json`
- Clean desi-clustering xi output:
  `/pscratch/sd/u/uendert/desiblind_lss_validation/desi-clustering-stats-20260624-004200-1585981/summary.json`
- Wide desi-clustering xi output used by the rendered notebook plots
  (`smax=200`, `ds=5`, `nmu=40`):
  `/pscratch/sd/u/uendert/desiblind_lss_validation/desi-clustering-xi-wide-20260625T114027Z/summary.json`
