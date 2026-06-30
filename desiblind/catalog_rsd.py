import hashlib
from pathlib import Path

import numpy as np

from .blinding import Blinder


CATALOG_RSD_PARAMETERS_FILENAME = 'catalog_rsd_blinding_parameters.npy'


class CatalogRSDBlinder(Blinder):
    """Catalog-level RSD redshift blinding.

    This class implements the generic redshift shift used by the LSS
    catalog-level RSD blinding step. It does not run reconstruction or handle
    DESI catalog filenames; callers must provide the observed redshift catalog
    and the reconstructed-realspace catalog.

    By default, ``fgrowth_blind`` is derived from the blinded ``w0``/``wa``
    cosmology following the DESI catalog-level blinding prescription. Passing an
    explicit ``fgrowth_blind`` is supported only as a validation/testing escape
    hatch.
    """

    observable_suffix = 'catalog_rsd'
    catalog_suffix = observable_suffix
    parameters_filename = CATALOG_RSD_PARAMETERS_FILENAME

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
    def _get_fiducial_cosmology(cosmo_fid='DESI'):
        if isinstance(cosmo_fid, str):
            if cosmo_fid != 'DESI':
                raise ValueError(f'Unknown fiducial cosmology {cosmo_fid!r}; pass a cosmology object instead')
            from cosmoprimo.fiducial import DESI
            return DESI()
        return cosmo_fid

    @classmethod
    def compute_fgrowth_blind(cls, w0, wa, z=None, bias=None, fiducial_f=0.8,
                              max_df_fraction=0.1, cosmo_fid='DESI', zeff=None):
        """Return the RSD growth-rate value derived from blinded ``w0``/``wa``.

        This follows the LSS pipeline formula used in
        ``LSS/scripts/main/apply_blinding_*``. In the DESI DR1-style catalog
        blinding scheme, ``fgrowth_blind`` is not an independent primary blind:
        it is derived from the same blinded ``w0``/``wa`` cosmology used for
        BAO/AP blinding. The value is chosen so the RSD Kaiser monopole factor
        compensates the isotropic AP/volume amplitude factor from the BAO/AP
        redshift remapping, in the same simplified linear model used by LSS.

        The fractional shift relative to ``fiducial_f`` is clipped to
        ``max_df_fraction`` by default, matching the LSS implementation. If the
        unclipped compensating value exceeds this cap, the compensation is
        intentionally incomplete rather than applying an overly large RSD shift.
        """
        cosmo_fid = cls._get_fiducial_cosmology(cosmo_fid=cosmo_fid)
        cosmo_blind = cosmo_fid.clone(w0_fld=float(w0), wa_fld=float(wa))

        if z is None:
            z = zeff
        if z is None:
            raise ValueError('compute_fgrowth_blind requires z or zeff')
        if bias is None:
            raise ValueError('compute_fgrowth_blind requires bias')
        z = float(z)
        bias = float(bias)
        fiducial_f = float(fiducial_f)

        dm_fid = cosmo_fid.comoving_angular_distance(z)
        dh_fid = 1. / cosmo_fid.hubble_function(z)
        dm_blind = cosmo_blind.comoving_angular_distance(z)
        dh_blind = 1. / cosmo_blind.hubble_function(z)
        volume_factor = (dm_blind**2 * dh_blind) / (dm_fid**2 * dh_fid)

        a = 0.2 / bias**2
        b = 2. / (3. * bias)
        c = 1. - (1. + 0.2 * (fiducial_f / bias)**2 + 2. / 3. * fiducial_f / bias) / volume_factor
        discriminant = b**2 - 4. * a * c
        if discriminant < 0:
            raise ValueError(f'Negative discriminant while computing fgrowth_blind: {discriminant}')
        fgrowth_blind = (-b + np.sqrt(discriminant)) / (2. * a)

        if max_df_fraction is not None:
            df_fraction = (fgrowth_blind - fiducial_f) / fiducial_f
            if abs(df_fraction) > max_df_fraction:
                df_fraction = float(max_df_fraction) * df_fraction / abs(df_fraction)
                fgrowth_blind = (1. + df_fraction) * fiducial_f
        return float(fgrowth_blind)

    @classmethod
    def _normalize_parameters(cls, parameters):
        parameters = dict(parameters or {})
        fiducial_f = float(parameters.get('fiducial_f', parameters.get('fgrowth_fid', 0.8)))
        if 'fgrowth_blind' in parameters:
            # Validation/testing override: the DESI DR1-style path derives this
            # value from w0/wa/zeff/bias via compute_fgrowth_blind().
            fgrowth_blind = float(parameters['fgrowth_blind'])
            normalized = dict(parameters)
            normalized['fiducial_f'] = fiducial_f
            normalized['fgrowth_fid'] = fiducial_f
            normalized['fgrowth_blind'] = fgrowth_blind
            return normalized

        missing = [name for name in ['w0', 'wa', 'zeff', 'bias'] if name not in parameters]
        if missing:
            raise ValueError(
                'Catalog RSD blinding parameters missing required key(s): '
                f'{missing}. Provide w0/wa/zeff/bias so fgrowth_blind can be derived, '
                'or pass fgrowth_blind explicitly for validation tests.'
            )
        normalized = dict(parameters)
        normalized['w0'] = float(parameters['w0'])
        normalized['wa'] = float(parameters['wa'])
        normalized['zeff'] = float(parameters['zeff'])
        normalized['bias'] = float(parameters['bias'])
        normalized['fiducial_f'] = fiducial_f
        normalized['fgrowth_fid'] = fiducial_f
        normalized['max_df_fraction'] = parameters.get('max_df_fraction', 0.1)
        normalized['fgrowth_blind'] = cls.compute_fgrowth_blind(
            w0=normalized['w0'], wa=normalized['wa'], z=normalized['zeff'],
            bias=normalized['bias'], fiducial_f=normalized['fiducial_f'],
            max_df_fraction=normalized['max_df_fraction'],
        )
        return normalized

    @classmethod
    def write_blinded_parameters(cls, name, parameters, save_dir=None, parameters_fn=None,
                                 bid=None, update=True, overwrite=False):
        """Write RSD blind parameters to a private hash-key parameter bank."""
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
        """Load RSD blind parameters from a private hash-key parameter bank."""
        if bid is None:
            bid = cls._get_bid()
        internal_name = cls._get_internal_name(name)
        key = hashlib.sha256(f'{internal_name}_bid{int(bid)}'.encode()).hexdigest()
        save_fn = cls._get_parameters_fn(save_dir=save_dir, parameters_fn=parameters_fn)
        bank = np.load(save_fn, allow_pickle=True).item()
        if key not in bank:
            raise ValueError(f'Cannot find catalog RSD blinding parameters for {internal_name}')
        return cls._normalize_parameters(bank[key])

    @staticmethod
    def _is_redshift_array(data):
        return np.isscalar(data) or isinstance(data, (np.ndarray, list, tuple))

    @classmethod
    def transform_redshift(cls, z_observed, z_realspace, fgrowth_fid=0.8,
                           fgrowth_blind=0.9, cosmo_fid='DESI'):
        """Return RSD-blinded redshifts matching ``LSS.apply_zshift_RSD``.

        ``z_observed`` is the redshift-space catalog redshift and ``z_realspace``
        is the reconstructed-realspace redshift produced by the LSS
        reconstruction step.
        """
        cosmo_fid = cls._get_fiducial_cosmology(cosmo_fid=cosmo_fid)
        from cosmoprimo.utils import DistanceToRedshift

        scalar = np.ndim(z_observed) == 0 and np.ndim(z_realspace) == 0
        z_observed = np.asarray(z_observed, dtype='f8')
        z_realspace = np.asarray(z_realspace, dtype='f8')
        z_observed, z_realspace = np.broadcast_arrays(z_observed, z_realspace)
        zout = z_observed.copy()
        mask = np.isfinite(z_observed) & np.isfinite(z_realspace)
        if np.any(mask):
            distance_observed = cosmo_fid.comoving_radial_distance(z_observed[mask])
            distance_realspace = cosmo_fid.comoving_radial_distance(z_realspace[mask])
            distance_blind = distance_realspace + (float(fgrowth_blind) / float(fgrowth_fid)) * (
                distance_observed - distance_realspace
            )
            zout[mask] = DistanceToRedshift(cosmo_fid.comoving_radial_distance)(distance_blind)
        if scalar:
            return float(zout)
        return zout

    @classmethod
    def attrs(cls, parameters):
        parameters = cls._normalize_parameters(parameters)
        out = {
            'desiblind_catalog_blinding': cls._get_observable_suffix(),
            'catalog_rsd_fgrowth_fid': parameters['fgrowth_fid'],
            'catalog_rsd_fgrowth_blind': parameters['fgrowth_blind'],
        }
        for key in ['w0', 'wa', 'zeff', 'bias', 'max_df_fraction']:
            if key in parameters:
                out[f'catalog_rsd_{key}'] = parameters[key]
        return out

    @classmethod
    def apply_to_catalog(cls, data_catalog, realspace_catalog, parameters, zcol='Z',
                         realspace_zcol=None, output_zcol=None, copy=True):
        """Apply RSD redshift blinding to a catalog-like object."""
        parameters = cls._normalize_parameters(parameters)
        realspace_zcol = zcol if realspace_zcol is None else realspace_zcol
        output_zcol = zcol if output_zcol is None else output_zcol
        new = data_catalog.copy() if copy else data_catalog
        new[output_zcol] = cls.transform_redshift(
            new[zcol], realspace_catalog[realspace_zcol],
            fgrowth_fid=parameters['fgrowth_fid'],
            fgrowth_blind=parameters['fgrowth_blind'],
        )
        if hasattr(new, 'attrs'):
            new.attrs.update(cls.attrs(parameters))
        return new

    @classmethod
    def apply_blinding(cls, name, data, realspace_data, parameters=None,
                       save_dir=None, parameters_fn=None, bid=None, zcol='Z',
                       realspace_zcol=None, output_zcol=None, copy=True):
        """Apply catalog RSD blinding to redshift arrays or a catalog-like object."""
        cls._get_public_observable_name(name)
        if parameters is None:
            parameters = cls.load_blinded_parameters(name, save_dir=save_dir, parameters_fn=parameters_fn, bid=bid)
        else:
            parameters = cls._normalize_parameters(parameters)
        if cls._is_redshift_array(data):
            return cls.transform_redshift(
                data, realspace_data,
                fgrowth_fid=parameters['fgrowth_fid'],
                fgrowth_blind=parameters['fgrowth_blind'],
            )
        return cls.apply_to_catalog(data, realspace_data, parameters=parameters, zcol=zcol,
                                    realspace_zcol=realspace_zcol, output_zcol=output_zcol, copy=copy)
