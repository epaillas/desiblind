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
Pipeline packages such as `desi-clustering` should still handle DESI catalog
file discovery, `LSScats/` naming, region splitting, and job orchestration; the
catalog blinder only handles the generic blinding transformation and private
hash-key parameter-bank convention.

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

Validation is split into two levels:

1. Core redshift-remapping equivalence with
   `LSS.blinding_tools.apply_zshift_DE`. This is covered by the lightweight
   `tests/test_catalog.py` test that compares the cosmology formula directly.
2. Full LSS workflow equivalence. This still needs to be run before production
   use, because the LSS script applies BAO/AP blinding to `Z_not4clus`, clips
   that input redshift column, recomputes `n(z)`, rescales `WEIGHT_SYS`, and
   then builds clustering data/random catalogs.

## Examples

See `nb/blinding_example_new.ipynb` for an up-to-date example of the API functionality.

## Credits

- Alejandro Perez Fernandez and Jiamin Hou for the original idea and script development.
- Arnaud de Mattia for co-developing this code and all of its dependencies.