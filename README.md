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

## Examples

See `nb/blinding_example_new.ipynb` for an up-to-date example of the API functionality.

## Credits

- Alejandro Perez Fernandez and Jiamin Hou for the original idea and script development.
- Arnaud de Mattia for co-developing this code and all of its dependencies.