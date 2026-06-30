from pathlib import Path
import hashlib

import numpy as np

from .blinding import Blinder


CATALOG_PARAMETERS_FILENAME = 'catalog_blinding_parameters.npy'


class CatalogBAOBlinder(Blinder):
    """Catalog-level BAO/AP redshift blinding.

    This class contains the generic catalog operation only. DESI catalog file
    discovery, LSScats naming, region splitting, and job orchestration should
    live in pipeline packages such as ``desi-clustering``.
    """

    observable_suffix = 'catalog_bao'
    # Backward-compatible public label used in catalog attrs.
    catalog_suffix = observable_suffix
    parameters_filename = CATALOG_PARAMETERS_FILENAME

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
        """Return the deterministic hash key for a bare canonical tracer-bin name.

        The naming convention intentionally mirrors the summary-statistic
        blinders: public APIs take bare canonical names such as ``'LRG3'`` and
        the blinder appends its observable suffix internally before hashing.
        """
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
    def _normalize_parameters(parameters):
        parameters = dict(parameters or {})
        missing = [name for name in ['w0', 'wa'] if name not in parameters]
        if missing:
            raise ValueError(f'Catalog BAO blinding parameters missing required key(s): {missing}')
        parameters['w0'] = float(parameters['w0'])
        parameters['wa'] = float(parameters['wa'])
        return parameters

    @classmethod
    def write_blinded_parameters(cls, name, parameters, save_dir=None, parameters_fn=None,
                                 bid=None, update=True, overwrite=False):
        """Write BAO/AP blind parameters to a private hash-key parameter bank.

        The saved format follows the summary-statistic blinding convention in
        :mod:`desiblind.blinding`: a NumPy ``.npy`` file containing a dictionary
        with hashed namespaces as keys.
        """
        if bid is None:
            bid = cls._get_bid()
        key = cls.get_key(name, bid=bid)
        parameters = cls._normalize_parameters(parameters)
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
        """Load BAO/AP blind parameters from a private hash-key parameter bank."""
        if bid is None:
            bid = cls._get_bid()
        internal_name = cls._get_internal_name(name)
        key = hashlib.sha256(f'{internal_name}_bid{int(bid)}'.encode()).hexdigest()
        save_fn = cls._get_parameters_fn(save_dir=save_dir, parameters_fn=parameters_fn)
        bank = np.load(save_fn, allow_pickle=True).item()
        public_name = cls._get_public_name_from_internal(internal_name)
        legacy_key = hashlib.sha256(f'{public_name}_bid{int(bid)}'.encode()).hexdigest()
        if key not in bank and legacy_key in bank:
            raise ValueError(
                f'Parameter bank stores legacy unsuffixed keys for {public_name}; regenerate it with '
                f'catalog-qualified {cls._get_observable_suffix()} names.'
            )
        if key not in bank:
            raise ValueError(f'Cannot find catalog BAO blinding parameters for {internal_name}')
        return cls._normalize_parameters(bank[key])

    @staticmethod
    def _get_fiducial_cosmology(cosmo_fid='DESI'):
        if isinstance(cosmo_fid, str):
            if cosmo_fid != 'DESI':
                raise ValueError(f'Unknown fiducial cosmology {cosmo_fid!r}; pass a cosmology object instead')
            from cosmoprimo.fiducial import DESI
            return DESI()
        return cosmo_fid

    @classmethod
    def transform_redshift(cls, z, w0, wa, cosmo_fid='DESI', inverse=False):
        """Return BAO/AP-blinded redshifts.

        The forward transform matches the LSS catalog-level BAO/AP convention
        in ``LSS.blinding_tools.apply_zshift_DE``: compute distances in the
        blinded ``w0``/``wa`` cosmology and convert them back to redshifts with
        the fiducial DESI distance-redshift relation.

        Parameters
        ----------
        z : array_like
            Input redshifts.
        w0, wa : float
            Dark-energy parameters for the blinded cosmology.
        cosmo_fid : str, cosmology object, default='DESI'
            Fiducial cosmology. The default matches the LSS catalog blinding
            convention.
        inverse : bool, default=False
            If ``True``, apply the inverse mapping.
        """
        from cosmoprimo.utils import DistanceToRedshift

        cosmo_fid = cls._get_fiducial_cosmology(cosmo_fid=cosmo_fid)
        cosmo_blind = cosmo_fid.clone(w0_fld=float(w0), wa_fld=float(wa))

        scalar = np.ndim(z) == 0
        zin = np.asarray(z, dtype='f8')
        zout = zin.copy()
        mask = np.isfinite(zin)
        if np.any(mask):
            if inverse:
                distance = cosmo_fid.comoving_radial_distance(zin[mask])
                zout[mask] = DistanceToRedshift(cosmo_blind.comoving_radial_distance)(distance)
            else:
                distance = cosmo_blind.comoving_radial_distance(zin[mask])
                zout[mask] = DistanceToRedshift(cosmo_fid.comoving_radial_distance)(distance)
        if scalar:
            return float(zout)
        return zout

    @classmethod
    def apply_to_catalog(cls, catalog, parameters, zcol='Z', input_zcol=None, output_zcol=None,
                         copy=True, inverse=False):
        """Apply BAO/AP redshift blinding to a catalog-like object.

        The object is expected to behave like a ``mockfactory.Catalog``: support
        ``copy()``, column access, and column assignment.

        Parameters
        ----------
        zcol : str, default='Z'
            Backward-compatible shortcut used when ``input_zcol`` and
            ``output_zcol`` are not provided.
        input_zcol : str, optional
            Column containing the redshifts to transform. If omitted, use
            ``zcol``. This makes the LSS convention explicit: its full-catalog
            BAO/AP script transforms ``Z_not4clus``.
        output_zcol : str, optional
            Column receiving the transformed redshifts. If omitted, use
            ``zcol``. This makes the LSS convention explicit: its full-catalog
            BAO/AP script writes the transformed values into ``Z``.
        """
        parameters = cls._normalize_parameters(parameters)
        input_zcol = zcol if input_zcol is None else input_zcol
        output_zcol = zcol if output_zcol is None else output_zcol
        new = catalog.copy() if copy else catalog
        new[output_zcol] = cls.transform_redshift(new[input_zcol], w0=parameters['w0'], wa=parameters['wa'], inverse=inverse)
        if hasattr(new, 'attrs'):
            new.attrs.update({'desiblind_catalog_blinding': cls._get_observable_suffix()})
        return new

    @staticmethod
    def _is_redshift_array(data):
        return np.isscalar(data) or isinstance(data, (np.ndarray, list, tuple))

    @classmethod
    def apply_blinding(cls, name, data, parameters=None, save_dir=None, parameters_fn=None,
                       bid=None, zcol='Z', input_zcol=None, output_zcol=None, copy=True):
        """Apply catalog BAO/AP blinding to redshifts or a catalog-like object."""
        # Validate naming consistently with the summary-statistic blinders even
        # when explicit parameters are provided and no parameter-bank lookup is
        # needed.
        cls._get_public_observable_name(name)
        if parameters is None:
            parameters = cls.load_blinded_parameters(name, save_dir=save_dir, parameters_fn=parameters_fn, bid=bid)
        if cls._is_redshift_array(data):
            parameters = cls._normalize_parameters(parameters)
            return cls.transform_redshift(data, w0=parameters['w0'], wa=parameters['wa'])
        return cls.apply_to_catalog(data, parameters=parameters, zcol=zcol, input_zcol=input_zcol,
                                    output_zcol=output_zcol, copy=copy)

    @classmethod
    def remove_blinding(cls, name, data, parameters=None, save_dir=None, parameters_fn=None,
                        bid=None, zcol='Z', input_zcol=None, output_zcol=None, copy=True, force=False):
        """Remove catalog BAO/AP blinding from redshifts or a catalog-like object."""
        if not force:
            raise ValueError('Are you sure you want to unblind? If so, provide "force=True"')
        cls._get_public_observable_name(name)
        if parameters is None:
            parameters = cls.load_blinded_parameters(name, save_dir=save_dir, parameters_fn=parameters_fn, bid=bid)
        if cls._is_redshift_array(data):
            parameters = cls._normalize_parameters(parameters)
            return cls.transform_redshift(data, w0=parameters['w0'], wa=parameters['wa'], inverse=True)
        return cls.apply_to_catalog(data, parameters=parameters, zcol=zcol, input_zcol=input_zcol,
                                    output_zcol=output_zcol, copy=copy, inverse=True)
