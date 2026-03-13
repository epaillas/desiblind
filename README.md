# desiblind

**Data-vector level blinding for DESI galaxy clustering statistics.**

## Overview

`desiblind` implements a blinding scheme for full-shape galaxy clustering analyses to prevent confirmation bias during data analysis. Before the data quality is approved and the analysis is frozen, all data vectors are shifted by a hidden offset derived from a fiducial cosmological model evaluated at a randomly drawn set of parameters. The true cosmological signal is concealed until the analysis pipeline is validated.

## Blinding Formalism

Blinded power spectrum multipoles are constructed as:

```
P_blind(k) = P(k) - W(k,k') P_ref(k') + W(k,k') P_shift(k')
```

where `P(k)` is the measured power spectrum, `W(k,k')` is the window function matrix, `P_ref(k')` is the theory evaluated at the reference (fiducial) cosmology, and `P_shift(k')` is the theory evaluated at a randomly drawn set of shifted cosmological parameters. The shift is deterministic but hidden from the analyst.

## Installation

```bash
pip install -e /path/to/desiblind/
```

The package depends on [`desilike`](https://github.com/cosmodesi/desilike). Install it first if it is not already present in your environment, or load the DESI `cosmodesi` environment at NERSC:

```bash
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
```

## Quick Start

### Developer workflow (generating and saving shifts)

This workflow is used by the blinding committee to generate and store the hidden shifts. See the example notebooks for more details.

```python
from desiblind.blinding import TracerPowerSpectrumMultipolesBlinder

blinder = TracerPowerSpectrumMultipolesBlinder()

# Add observables (data vectors + covariance) for each tracer/redshift bin
blinder.add_observable('LRG_z0', data=my_observable_tree, covariance=my_cov)
blinder.add_observable('ELG_z1', data=my_observable_tree, covariance=my_cov)

# Generate 100 blinded realisations and store the hidden shifts
for i in range(100):
    blinded = compute_shift(...)   # evaluate theory at shifted params
    blinder.set_blinded_data('LRG_z0', blinded_data=blinded)

blinder.write_blinded_shifts(save_dir='/path/to/shifts/')
```

### User workflow (applying and removing blinding)

```python
from desiblind.blinding import Blinder

# Apply blinding to a data vector
blinded_data = Blinder.apply_blinding('LRG_z0', data=my_data)

# Remove blinding (requires explicit opt-in to prevent accidental unblinding)
unblinded_data = Blinder.remove_blinding('LRG_z0', data=blinded_data, force=True)
```

Both methods accept either a `numpy` array (with `k` and `ells` keyword arguments) or an `lsstypes.ObservableTree` object.

## API Reference

| Class / Method | Description |
|---|---|
| `Blinder` | Base class for blinding galaxy clustering observables |
| `Blinder.add_observable(name, data, covariance)` | Register a tracer/redshift-bin data vector |
| `Blinder.set_blinded_data(name, blinded_data)` | Store a blinded realisation (developer use only) |
| `Blinder.write_blinded_shifts(save_dir)` | Persist the hidden shifts to disk |
| `Blinder.apply_blinding(name, data)` | Apply the pre-computed shift to a data vector |
| `Blinder.remove_blinding(name, data, force=True)` | Remove the blinding shift (requires `force=True`) |
| `TracerPowerSpectrumMultipolesBlinder` | Subclass specialised for power spectrum multipoles; adds `plot_observables()` |
| `TracerBispectrumMultipolesBlinder` | Subclass specialised for bispectrum multipoles |

## Repository Layout

```
desiblind/          # Main Python package
  blinding.py       # Blinder and subclasses
  utils.py          # Shared utilities (plot style, etc.)
nb/                 # Jupyter notebooks
  blinding_example.ipynb   # End-to-end blinding example
scripts/            # Standalone analysis scripts
  fs_likelihood.py  # Full-shape likelihood helper
```
