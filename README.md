# desiblind

**Data-vector level blinding for DESI galaxy clustering statistics.**

## Overview

`desiblind` implements a blinding scheme for full-shape galaxy clustering analyses to prevent confirmation bias during data analysis. Before the data quality is approved and the analysis is frozen, all data vectors are shifted by a hidden offset derived from a fiducial cosmological model evaluated at a randomly drawn set of parameters. The true cosmological signal is concealed until the analysis pipeline is validated.

## Blinding Formalism

Blinded power spectrum multipoles are constructed as:

```
P_blind(k) = P(k) - W(k,k') P_ref(k') + W(k,k') P_shift(k')
```

where `P(k)` is the measured power spectrum, `W(k,k')` is the window function matrix, `P_ref(k')` is the theory evaluated at the reference (fiducial) cosmology, and `P_shift(k')` is the theory evaluated at a randomly drawn set of shifted cosmological parameters. The shift is deterministic but hidden from the analyst. There is also support for the Sugiyama bispectrum multipoles.

## Installation

```bash
pip install -e /path/to/desiblind/
```

The package depends on [`desilike`](https://github.com/cosmodesi/desilike). Install it first if it is not already present in your environment, or load the DESI `cosmodesi` environment at NERSC:

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
```


## Catalog-level BAO/AP blinding

`desiblind` also provides a generic BAO/AP catalog-level redshift remapping API.
DESI pipeline code or validation drivers should still handle catalog file
discovery, `LSScats/` naming, region splitting, and job orchestration; the
catalog blinder only handles the generic redshift-remapping transformation and
private hash-key parameter-bank convention.

```python
from desiblind import CatalogBAOBlinder

params = {'w0': -0.95, 'wa': 0.10}
blinded = CatalogBAOBlinder.apply_blinding('LRG3', catalog, parameters=params)
```

For LSS-style full-catalog validation, the source and destination redshift
columns can be specified separately:

```python
blinded = CatalogBAOBlinder.apply_blinding(
    'LRG3', catalog, parameters=params, input_zcol='Z_not4clus', output_zcol='Z'
)
```

Blind parameters can be saved in a private NumPy dictionary using the same
hash-key style as the summary-statistic shifts:

```python
CatalogBAOBlinder.write_blinded_parameters('LRG3', params, save_dir='private')
params = CatalogBAOBlinder.load_blinded_parameters('LRG3', save_dir='private')
```

Validation is organized by LSS workflow step, because LSS is the
reference implementation for catalog-level blinding:

1. Redshift-shift equivalence with
   `LSS.blinding_tools.apply_zshift_DE`. This is covered by the lightweight
   `tests/test_catalog.py` test that compares the cosmology formula directly.
   On NERSC, the same comparison can be run against the actual LSS function and
   a small real-catalog sample with:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_redshift_shift.py
   ```

2. LSS `n(z)` / `WEIGHT_SYS` compatibility after replacing only the redshift
   shifter with `CatalogBAOBlinder`:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_nz_weight.py
   ```

3. LSS-style saved full-catalog BAO/AP workflow through `WEIGHT_FKP`, shifted
   saved `Z`, output `n(z)`, and final `WEIGHT_SYS` update:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_saved_catalog.py
   ```

4. LSS `mkclusdat` clustering-data generation from the saved blinded full
   catalog:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_mkclusdat.py
   ```

5. LSS `mkclusran` clustering-random generation from the blinded clustering
   data catalog and a scratch random sample:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_mkclusran.py
   ```

6. LSS Galactic-cap splitting for the blinded clustering data and random
   catalogs:

   ```bash
   python scripts/validation/validate_catalog_bao_lss_split_gc.py
   ```

   These validation scripts write only to fresh directories under `$SCRATCH` by
   default.

The validation suite now covers the BAO/AP saved-catalog path through the same
LSS data/random catalog production steps. Detailed commands, NERSC interactive
examples, exact comparison definitions, and current production-like reference
outputs are documented in:

```text
scripts/validation/README.md
```

A rendered validation walkthrough notebook with the full validation ladder, exact
input/output file pairs, HDF5 comparisons, native smoke tests, Pk plots over the BAO-range k grid, final xi plots out to s=200, built-in xi multipoles, and pair-count residual plots is
available at:

```text
scripts/validation/catalog_bao_validation.ipynb
```

The rendered notebook uses BAO-range validation products for the plots: Pk with `meshsize=256`, `boxsize=6000`, `kmax=0.13`, `dk=0.005`, and xi with `smax=200`, `ds=5`, `nmu=40`. The Pk plot summary is `/pscratch/sd/u/uendert/desiblind_lss_validation/desi-clustering-pk-bao-validated-20260625T130130Z/summary.json`; the xi plot summary is `/pscratch/sd/u/uendert/desiblind_lss_validation/desi-clustering-xi-wide-20260625T114027Z/summary.json`.

Remaining work before production use includes optional random resampling modes
and any additional production packaging requested by the collaboration. The
catalog-level blinding machinery and validation should stay in `desiblind`; a
measurement pipeline such as `desi-clustering` should either read saved blinded
catalogs as normal inputs or keep only a narrow measurement-side bridge to
`CatalogBAOBlinder`. Future RSD and fNL catalog blinding should be treated as
saved-catalog workflows, not generic on-the-fly measurement options, because
they require reconstruction and heavier LSS processing.

## Examples

See `nb/blinding_example_new.ipynb` for data-vector power-spectrum/bispectrum
examples, and `nb/catalog_bao_blinding_example.ipynb` for the catalog-level
BAO/AP redshift-remapping API.

## Credits

- Alejandro Perez Fernandez and Jiamin Hou for the original idea and script development.
- Arnaud de Mattia for co-developing this code and all of its dependencies.