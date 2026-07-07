import hashlib
from pathlib import Path

import numpy as np

from .blinding import Blinder


CATALOG_FNL_PARAMETERS_FILENAME = 'catalog_fnl_blinding_parameters.npy'

TRACER_DEFAULTS = {
    'BGS': {'zeff': 0.25, 'bias': 1.8},
    'LRG': {'zeff': 0.8, 'bias': 2.0},
    'ELG': {'zeff': 1.1, 'bias': 1.3},
    'QSO': {'zeff': 1.6, 'bias': 2.3},
}


class CatalogFNLBlinder(Blinder):
    """Catalog-level local PNG/fNL weight blinding.

    This class implements the fNL catalog-level blinding operation used in the
    LSS catalog scripts. The physics operation is delegated to
    ``mockfactory.blinding.CutskyCatalogBlinding.png`` with
    ``method='data_weights'`` and ``shotnoise_correction=True`` by default.

    The class does not handle DESI filenames or final LSS random resampling.
    Callers provide data/random RDZ catalogs and decide how to propagate the
    returned internal fNL weight factor into the final saved catalog workflow.
    """

    observable_suffix = 'catalog_fnl'
    catalog_suffix = observable_suffix
    parameters_filename = CATALOG_FNL_PARAMETERS_FILENAME

    @classmethod
    def _get_bid(cls):
        rng = np.random.RandomState(seed=42)
        return int(rng.randint(0, cls.blinded_nmax))

    @classmethod
    def _get_internal_name(cls, name):
        """Return the statistic-qualified internal name used for private keys."""
        return cls._get_internal_observable_name(name)

    @classmethod
    def get_key(cls, name, bid=None):
        """Return the deterministic hash key for a bare canonical tracer-bin name."""
        if bid is None:
            bid = cls._get_bid()
        internal_name = cls._get_internal_name(name)
        return hashlib.sha256(f'{internal_name}_bid{int(bid)}'.encode()).hexdigest()

    @classmethod
    def _get_parameters_fn(cls, save_dir=None, parameters_fn=None):
        if parameters_fn is not None:
            return Path(parameters_fn)
        if save_dir is None:
            save_dir = '.'
        return Path(save_dir) / cls.parameters_filename

    @staticmethod
    def infer_tracer_defaults(tracer):
        """Return LSS-style default ``zeff`` and ``bias`` for a tracer name."""
        text = str(tracer).upper()
        if text.startswith('ELG'):
            key = 'ELG'
        elif text.startswith('LRG'):
            key = 'LRG'
        elif text.startswith('QSO'):
            key = 'QSO'
        elif text.startswith('BGS'):
            key = 'BGS'
        else:
            raise ValueError(f'Cannot infer fNL zeff/bias defaults for tracer {tracer!r}')
        return dict(TRACER_DEFAULTS[key])

    @staticmethod
    def generate_fnl_from_index(index, low=-15., high=15.):
        """Generate the LSS fNL blind value associated with a bank row index.

        LSS seeds NumPy's legacy random generator with the same selected w0/wa
        row index, then draws one uniform value in ``[-15, 15]``.
        """
        rng = np.random.RandomState(int(index))
        return float(rng.uniform(low=float(low), high=float(high), size=1)[0])

    @classmethod
    def _normalize_parameters(cls, parameters, tracer=None):
        parameters = dict(parameters or {})
        defaults = cls.infer_tracer_defaults(tracer) if tracer is not None else {}
        if 'zeff' not in parameters and 'z' in parameters:
            parameters['zeff'] = parameters['z']
        if 'zeff' not in parameters and 'zeff' in defaults:
            parameters['zeff'] = defaults['zeff']
        if 'bias' not in parameters and 'bias' in defaults:
            parameters['bias'] = defaults['bias']
        if 'fnl' not in parameters and 'index' in parameters:
            parameters['fnl'] = cls.generate_fnl_from_index(parameters['index'])

        missing = [name for name in ['fnl', 'zeff', 'bias'] if name not in parameters]
        if missing:
            raise ValueError(
                'Catalog fNL blinding parameters missing required key(s): '
                f'{missing}. Provide fnl/zeff/bias, or pass index plus tracer so LSS defaults can be inferred.'
            )

        normalized = dict(parameters)
        normalized['fnl'] = float(parameters['fnl'])
        normalized['zeff'] = float(parameters['zeff'])
        normalized['bias'] = float(parameters['bias'])
        normalized.setdefault('method', 'data_weights')
        normalized.setdefault('shotnoise_correction', True)
        normalized.setdefault('smoothing_radius', 30.)
        return normalized

    @classmethod
    def write_blinded_parameters(cls, name, parameters, save_dir=None, parameters_fn=None,
                                 bid=None, update=True, overwrite=False, tracer=None):
        """Write fNL blind parameters to a private hash-key parameter bank."""
        if bid is None:
            bid = cls._get_bid()
        key = cls.get_key(name, bid=bid)
        parameters = cls._normalize_parameters(parameters, tracer=tracer or name)
        save_fn = cls._get_parameters_fn(save_dir=save_dir, parameters_fn=parameters_fn)
        if save_fn.exists():
            if not update and not overwrite:
                raise FileExistsError(f'{save_fn} already exists; pass update=True or overwrite=True')
            bank = {} if overwrite else np.load(save_fn, allow_pickle=True).item()
        else:
            bank = {}
        if key in bank and not overwrite:
            raise FileExistsError(f'Blinding key {key} already exists in {save_fn}; pass overwrite=True only after checking this is safe')
        bank[key] = parameters
        save_fn.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_fn, bank)
        return save_fn

    @classmethod
    def load_blinded_parameters(cls, name, save_dir=None, parameters_fn=None, bid=None):
        """Load fNL blind parameters from a private hash-key parameter bank."""
        if bid is None:
            bid = cls._get_bid()
        internal_name = cls._get_internal_name(name)
        key = hashlib.sha256(f'{internal_name}_bid{int(bid)}'.encode()).hexdigest()
        save_fn = cls._get_parameters_fn(save_dir=save_dir, parameters_fn=parameters_fn)
        bank = np.load(save_fn, allow_pickle=True).item()
        if key not in bank:
            raise ValueError(f'Cannot find catalog fNL blinding parameters for {internal_name}')
        return cls._normalize_parameters(bank[key], tracer=name)

    @staticmethod
    def _column_names(catalog):
        if hasattr(catalog, 'colnames'):
            return list(catalog.colnames)
        columns = getattr(catalog, 'columns', None)
        if callable(columns):
            return list(columns())
        if columns is not None:
            return list(columns)
        if hasattr(catalog, 'keys'):
            return list(catalog.keys())
        names = getattr(getattr(catalog, 'dtype', None), 'names', None)
        return list(names or [])

    @staticmethod
    def _catalog_positions(catalog, racol='RA', deccol='DEC', zcol='Z'):
        return [
            np.asarray(catalog[racol], dtype='f8'),
            np.asarray(catalog[deccol], dtype='f8'),
            np.asarray(catalog[zcol], dtype='f8'),
        ]

    @staticmethod
    def _catalog_weights(catalog, weight_col='WEIGHT'):
        if weight_col is None:
            return None
        return np.asarray(catalog[weight_col], dtype='f8')

    @staticmethod
    def _concatenate_random_positions(randoms, racol='RA', deccol='DEC', zcol='Z'):
        random_list = list(randoms) if isinstance(randoms, (list, tuple)) else [randoms]
        return [np.concatenate([np.asarray(random[col], dtype='f8') for random in random_list])
                for col in [racol, deccol, zcol]]

    @staticmethod
    def _concatenate_random_weights(randoms, weight_col='WEIGHT'):
        if weight_col is None:
            return None
        random_list = list(randoms) if isinstance(randoms, (list, tuple)) else [randoms]
        return np.concatenate([np.asarray(random[weight_col], dtype='f8') for random in random_list])

    @classmethod
    def _build_mockfactory_blinding(cls, parameters, cosmo_fid='DESI', position_type='rdz',
                                    mpicomm=None, mpiroot=0, dtype=None):
        from mockfactory.blinding import CutskyCatalogBlinding, get_cosmo_blind

        cosmo_blind = get_cosmo_blind(cosmo_fid, z=parameters['zeff'])
        cosmo_blind._derived['fnl'] = parameters['fnl']
        return CutskyCatalogBlinding(
            cosmo_fid=cosmo_fid, cosmo_blind=cosmo_blind, bias=parameters['bias'],
            z=parameters['zeff'], position_type=position_type, mpicomm=mpicomm,
            mpiroot=mpiroot, dtype=dtype,
        )

    @classmethod
    def compute_blinded_data_weights(cls, data_positions, data_weights, randoms_positions,
                                     randoms_weights, parameters, *, tracer=None, cosmo_fid='DESI',
                                     position_type='rdz', mpicomm=None, mpiroot=0, dtype=None,
                                     recon='IterativeFFTReconstruction', smoothing_radius=None,
                                     method=None, shotnoise_correction=None, **kwargs):
        """Return fNL-blinded data weights matching the LSS mockfactory call."""
        parameters = cls._normalize_parameters(parameters, tracer=tracer)
        method = parameters.get('method') if method is None else method
        shotnoise_correction = parameters.get('shotnoise_correction') if shotnoise_correction is None else shotnoise_correction
        smoothing_radius = parameters.get('smoothing_radius') if smoothing_radius is None else smoothing_radius
        if method != 'data_weights':
            raise ValueError('CatalogFNLBlinder currently supports the LSS integrated method="data_weights" path')

        data_weights = None if data_weights is None else np.asarray(data_weights, dtype='f8')
        if parameters['fnl'] == 0.:
            return None if data_weights is None else data_weights.copy()

        blinding = cls._build_mockfactory_blinding(
            parameters, cosmo_fid=cosmo_fid, position_type=position_type,
            mpicomm=mpicomm, mpiroot=mpiroot, dtype=dtype,
        )
        return blinding.png(
            data_positions, data_weights=data_weights,
            randoms_positions=randoms_positions, randoms_weights=randoms_weights,
            method=method, shotnoise_correction=bool(shotnoise_correction),
            recon=recon, smoothing_radius=float(smoothing_radius), **kwargs,
        )

    @classmethod
    def compute_weight_factor(cls, data_positions, data_weights, randoms_positions,
                              randoms_weights, parameters, **kwargs):
        """Return ``new_data_weights / data_weights`` for fNL blinding."""
        data_weights = np.asarray(data_weights, dtype='f8')
        new_data_weights = cls.compute_blinded_data_weights(
            data_positions, data_weights, randoms_positions, randoms_weights, parameters, **kwargs,
        )
        return np.asarray(new_data_weights, dtype='f8') / data_weights

    @classmethod
    def attrs(cls, parameters, tracer=None):
        parameters = cls._normalize_parameters(parameters, tracer=tracer)
        return {
            'desiblind_catalog_blinding': cls._get_observable_suffix(),
            'catalog_fnl_fnl': parameters['fnl'],
            'catalog_fnl_zeff': parameters['zeff'],
            'catalog_fnl_bias': parameters['bias'],
            'catalog_fnl_method': parameters['method'],
            'catalog_fnl_shotnoise_correction': bool(parameters['shotnoise_correction']),
        }

    @classmethod
    def apply_to_catalog(cls, data_catalog, random_catalog, parameters, *, tracer=None,
                         racol='RA', deccol='DEC', zcol='Z', weight_col='WEIGHT',
                         output_weight_col=None, random_weight_col='WEIGHT',
                         update_weight_comp=True, weight_comp_col='WEIGHT_COMP',
                         return_weight_factor=False,
                         copy=True, **kwargs):
        """Apply fNL data-weight blinding to a catalog-like object.

        This follows the integrated LSS path: update ``WEIGHT`` and, when
        present, fold the same factor into ``WEIGHT_COMP``. The fNL factor is
        returned for diagnostics when requested, but is not stored as a
        persistent catalog column.
        """
        parameters = cls._normalize_parameters(parameters, tracer=tracer)
        output_weight_col = weight_col if output_weight_col is None else output_weight_col
        new = data_catalog.copy() if copy else data_catalog

        data_positions = cls._catalog_positions(new, racol=racol, deccol=deccol, zcol=zcol)
        data_weights = cls._catalog_weights(new, weight_col=weight_col)
        random_positions = None if random_catalog is None else cls._concatenate_random_positions(
            random_catalog, racol=racol, deccol=deccol, zcol=zcol,
        )
        random_weights = None if random_catalog is None else cls._concatenate_random_weights(
            random_catalog, weight_col=random_weight_col,
        )
        new_data_weights = cls.compute_blinded_data_weights(
            data_positions, data_weights, random_positions, random_weights,
            parameters, tracer=tracer, **kwargs,
        )
        weight_factor = np.asarray(new_data_weights, dtype='f8') / np.asarray(data_weights, dtype='f8')
        new[output_weight_col] = new_data_weights
        columns = set(cls._column_names(new))
        if update_weight_comp and weight_comp_col in columns:
            new[weight_comp_col] = np.asarray(new[weight_comp_col]) * weight_factor
        if hasattr(new, 'attrs'):
            new.attrs.update(cls.attrs(parameters, tracer=tracer))
        if return_weight_factor:
            return new, weight_factor
        return new

    @classmethod
    def apply_blinding(cls, name, data, randoms, parameters=None, save_dir=None,
                       parameters_fn=None, bid=None, **kwargs):
        """Apply catalog fNL blinding to a catalog-like object."""
        cls._get_public_observable_name(name)
        if parameters is None:
            parameters = cls.load_blinded_parameters(name, save_dir=save_dir, parameters_fn=parameters_fn, bid=bid)
        else:
            parameters = cls._normalize_parameters(parameters, tracer=name)
        return cls.apply_to_catalog(data, randoms, parameters=parameters, tracer=name, **kwargs)
