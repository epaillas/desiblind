# Catalog BAO/AP validation scripts

These scripts validate `CatalogBAOBlinder` against the authoritative LSS BAO/AP
catalog-blinding workflow.

The comparisons are intentionally apples-to-apples:

- **LSS reference branch**: LSS BAO/AP redshift shifter and LSS catalog-making
  functions.
- **desiblind branch**: `CatalogBAOBlinder` used as the redshift shifter, then
  the same downstream LSS catalog-making functions.

The validation asks whether replacing the LSS BAO/AP redshift remapping with
`CatalogBAOBlinder` changes any downstream catalog or measurement product.

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

## Validation ladder

Run the scripts in this order when rebuilding confidence from scratch.

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
