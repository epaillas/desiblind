# desiblind

**Data-vector and catalog-level blinding for DESI galaxy clustering statistics.**

## Overview

`desiblind` implements blinding schemes for galaxy clustering analyses to prevent confirmation bias during data analysis. For full-shape data vectors, measured statistics are shifted by a hidden offset derived from a fiducial cosmological model evaluated at a randomly drawn set of parameters. For catalog-level BAO/RSD/fNL validation, DESI clustering catalogs can instead be blinded before the measurement is made.

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

The two implementation modules are:

- `desiblind.data_vector`: full-shape data-vector shifts for power-spectrum and bispectrum multipoles.
- `desiblind.catalog`: BAO/RSD/fNL catalog-level blinding before measuring statistics.

## Credits

- Alejandro Perez Fernandez and Jiamin Hou for the original idea and script development.
- Arnaud de Mattia for co-developing this code and all of its dependencies.
- Uendert Andrade for developing the catalog-level blinding.
