from pathlib import Path
import hashlib

import numpy as np

from .blinding import Blinder


CATALOG_PARAMETERS_FILENAME = 'catalog_blinding_parameters.npy'
DEFAULT_LSS_W0WA_BANK = '/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/w0wa_initvalues_zeffcombined_1000realisations.txt'


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
    def alpha_shift_stats(cls, parameters, zrange=(0.4, 2.1), nz=100, cosmo_fid='DESI'):
        """Return BAO/AP alpha-shift diagnostics for ``parameters``.

        This follows the Andrade et al. / DESI Y1 LSS selection convention used
        to make ``w0wa_initvalues_zeffcombined_1000realisations.txt``: compare
        the blinded dark-energy cosmology to the fiducial/template DESI
        cosmology over ``0.4 < z < 2.1`` and require both alpha shifts to stay
        within 3% of unity.

        Returns a dictionary with ``max_abs_alpha_parallel_minus_one`` and
        ``max_abs_alpha_perp_minus_one`` plus the sampled alpha arrays.
        """
        parameters = cls._normalize_parameters(parameters)
        zmin, zmax = map(float, zrange)
        if zmin >= zmax:
            raise ValueError(f'zrange must be increasing, got {zrange}')
        z = np.linspace(zmin, zmax, int(nz))
        cosmo_fid = cls._get_fiducial_cosmology(cosmo_fid=cosmo_fid)
        cosmo_blind = cosmo_fid.clone(w0_fld=parameters['w0'], wa_fld=parameters['wa'])

        # DESI fiducial and template cosmologies are the same here, but keep the
        # rs_drag factors explicit to mirror the validation/plotting notebook.
        rs_fid = cosmo_fid.rs_drag / cosmo_fid.h
        rs_template = rs_fid
        da_fid = cosmo_fid.angular_diameter_distance(z) / cosmo_fid.h
        da_blind = cosmo_blind.angular_diameter_distance(z) / cosmo_blind.h
        h_fid = cosmo_fid.hubble_function(z)
        h_blind = cosmo_blind.hubble_function(z)

        alpha_parallel = (h_blind * rs_template) / (h_fid * rs_fid)
        alpha_perp = (da_fid / rs_fid) / (da_blind / rs_template)
        return {
            'z': z,
            'alpha_parallel': np.asarray(alpha_parallel),
            'alpha_perp': np.asarray(alpha_perp),
            'max_abs_alpha_parallel_minus_one': float(np.max(np.abs(alpha_parallel - 1.))),
            'max_abs_alpha_perp_minus_one': float(np.max(np.abs(alpha_perp - 1.))),
            'w0_plus_wa': float(parameters['w0'] + parameters['wa']),
        }

    @classmethod
    def validate_alpha_shift(cls, parameters, zrange=(0.4, 2.1), max_alpha_shift=0.03,
                             nz=100, cosmo_fid='DESI'):
        """Validate that BAO/AP parameters lie inside the DESI 3% alpha mask.

        Raises
        ------
        ValueError
            If ``w0 + wa > 0`` or either alpha shift exceeds
            ``max_alpha_shift`` over ``zrange``.
        """
        parameters = cls._normalize_parameters(parameters)
        stats = cls.alpha_shift_stats(parameters, zrange=zrange, nz=nz, cosmo_fid=cosmo_fid)
        failures = []
        if stats['w0_plus_wa'] > 0.:
            failures.append(f"w0 + wa = {stats['w0_plus_wa']:.6g} > 0")
        if stats['max_abs_alpha_parallel_minus_one'] >= max_alpha_shift:
            failures.append(
                f"max |alpha_parallel - 1| = {stats['max_abs_alpha_parallel_minus_one']:.6g} >= {max_alpha_shift}"
            )
        if stats['max_abs_alpha_perp_minus_one'] >= max_alpha_shift:
            failures.append(
                f"max |alpha_perp - 1| = {stats['max_abs_alpha_perp_minus_one']:.6g} >= {max_alpha_shift}"
            )
        if failures:
            raise ValueError(
                'Catalog BAO/AP blinding parameters are outside the allowed DESI 3% alpha-shift region: '
                + '; '.join(failures)
            )
        return stats

    @staticmethod
    def _normalize_parameter_range(name, values):
        values = tuple(map(float, values))
        if len(values) != 2:
            raise ValueError(f'{name} must contain exactly two values, got {values}')
        low, high = values
        if low >= high:
            raise ValueError(f'{name} must be increasing, got {values}')
        return low, high

    @classmethod
    def generate_parameters(cls, seed, w0_range=(-1.2, -0.8), wa_range=(-0.8, 0.8),
                            max_attempts=10000, validate_alpha_shift=True,
                            alpha_zrange=(0.4, 2.1), max_alpha_shift=0.03,
                            alpha_nz=100, cosmo_fid='DESI'):
        """Generate a self-contained BAO/AP blinding ``(w0, wa)`` pair.

        The draw is from independent uniform ``w0`` and ``wa`` ranges, followed
        by rejection on the physically required ``w0 + wa <= 0`` condition and,
        by default, the DESI 3 percent BAO/AP alpha-shift mask. This is the
        native ``desiblind`` bank-creation path and does not read the historical
        LSS ``w0wa`` text bank.

        Returns
        -------
        parameters : dict
            Generated ``{'w0': ..., 'wa': ...}`` pair.
        metadata : dict
            Private audit metadata with source, seed, ranges, and attempt count.
        """
        if seed is None:
            raise ValueError('generate_parameters requires an explicit seed for reproducibility')
        w0_range = cls._normalize_parameter_range('w0_range', w0_range)
        wa_range = cls._normalize_parameter_range('wa_range', wa_range)
        max_attempts = int(max_attempts)
        if max_attempts < 1:
            raise ValueError(f'max_attempts must be >= 1, got {max_attempts}')

        rng = np.random.default_rng(int(seed))
        last_error = None
        for attempt in range(1, max_attempts + 1):
            parameters = {
                'w0': float(rng.uniform(*w0_range)),
                'wa': float(rng.uniform(*wa_range)),
            }
            if parameters['w0'] + parameters['wa'] > 0.:
                last_error = 'w0 + wa > 0'
                continue
            if validate_alpha_shift:
                try:
                    cls.validate_alpha_shift(
                        parameters, zrange=alpha_zrange, max_alpha_shift=max_alpha_shift,
                        nz=alpha_nz, cosmo_fid=cosmo_fid,
                    )
                except ValueError as exc:
                    last_error = str(exc)
                    continue
            metadata = {
                'parameter_source': 'desiblind_generated',
                'generator': 'uniform_rejection',
                'seed': int(seed),
                'w0_range': list(w0_range),
                'wa_range': list(wa_range),
                'max_attempts': max_attempts,
                'accepted_attempt': attempt,
                'validated_alpha_shift': bool(validate_alpha_shift),
            }
            return parameters, metadata
        raise RuntimeError(
            f'Could not generate valid catalog BAO/AP parameters after {max_attempts} attempts. '
            f'Last rejection reason: {last_error}'
        )

    @classmethod
    def load_lss_parameter_bank(cls, parameters_fn=DEFAULT_LSS_W0WA_BANK):
        """Load an LSS-style plain text ``w0 wa`` parameter bank."""
        bank = np.loadtxt(parameters_fn)
        bank = np.atleast_2d(bank)
        if bank.shape[1] != 2:
            raise ValueError(f'LSS w0/wa bank must have two columns, got shape {bank.shape}')
        return bank

    @classmethod
    def load_lss_parameters(cls, parameters_fn=DEFAULT_LSS_W0WA_BANK, index=None, filerow=None):
        """Load one ``(w0, wa)`` pair from an LSS-style parameter bank.

        Parameters
        ----------
        parameters_fn : str, Path
            Plain-text file with two columns: ``w0`` and ``wa``.
        index : int, optional
            Row index to load.
        filerow : str, Path, optional
            File containing the row index, matching the historical LSS
            ``filerow.txt`` pattern. Exactly one of ``index`` or ``filerow``
            must be provided.
        """
        if (index is None) == (filerow is None):
            raise ValueError('Provide exactly one of index or filerow for LSS-style BAO parameters')
        if filerow is not None:
            index = int(np.loadtxt(filerow))
        bank = cls.load_lss_parameter_bank(parameters_fn)
        index = int(index)
        if index < 0 or index >= len(bank):
            raise IndexError(f'LSS parameter index {index} out of range for bank with {len(bank)} rows')
        w0, wa = bank[index]
        return {'w0': float(w0), 'wa': float(wa), 'index': index, 'parameters_fn': str(parameters_fn)}

    @classmethod
    def load_parameters(cls, name=None, parameters=None, parameters_fn=None, save_dir=None, bid=None,
                        source=None, lss_parameters_fn=None, lss_parameter_index=None, lss_filerow=None,
                        validate_alpha_shift=True, alpha_zrange=(0.4, 2.1), max_alpha_shift=0.03,
                        alpha_nz=100):
        """Load BAO/AP parameters from explicit, desiblind-bank, or LSS-bank sources.

        The preferred production source is the desiblind hashed parameter bank
        (``source='desiblind'``), consistent with summary-statistic blinding.
        LSS-style plain text banks are supported for historical compatibility.
        """
        if source is None:
            if parameters is not None:
                source = 'explicit'
            elif parameters_fn is not None or save_dir is not None:
                source = 'desiblind'
            elif lss_parameters_fn is not None or lss_parameter_index is not None or lss_filerow is not None:
                source = 'lss'
            else:
                raise ValueError('Cannot infer BAO/AP parameter source')
        source = str(source).lower().replace('-', '_')
        metadata = {'parameter_source': source}
        if source == 'explicit':
            loaded = cls._normalize_parameters(parameters)
        elif source in {'desiblind', 'hash', 'hashed', 'parameter_bank'}:
            if name is None:
                raise ValueError('desiblind hashed parameter banks require a canonical tracer-bin name')
            loaded = cls.load_blinded_parameters(name, save_dir=save_dir, parameters_fn=parameters_fn, bid=bid)
            metadata.update({'name': name, 'parameters_fn': None if parameters_fn is None else str(parameters_fn),
                             'save_dir': None if save_dir is None else str(save_dir), 'bid': cls._get_bid() if bid is None else int(bid)})
            source = metadata['parameter_source'] = 'desiblind'
        elif source in {'lss', 'lss_bank', 'lss_filerow'}:
            if lss_parameters_fn is None:
                lss_parameters_fn = DEFAULT_LSS_W0WA_BANK
            loaded = cls.load_lss_parameters(parameters_fn=lss_parameters_fn, index=lss_parameter_index, filerow=lss_filerow)
            metadata.update({'parameters_fn': loaded.pop('parameters_fn'), 'index': loaded.pop('index'),
                             'filerow': None if lss_filerow is None else str(lss_filerow)})
            source = metadata['parameter_source'] = 'lss'
        else:
            raise ValueError(f'Unknown catalog BAO parameter source {source!r}')

        loaded = cls._normalize_parameters(loaded)
        if validate_alpha_shift:
            stats = cls.validate_alpha_shift(loaded, zrange=alpha_zrange, max_alpha_shift=max_alpha_shift, nz=alpha_nz)
            metadata.update({
                'alpha_zrange': tuple(map(float, alpha_zrange)),
                'max_alpha_shift': float(max_alpha_shift),
                'max_abs_alpha_parallel_minus_one': stats['max_abs_alpha_parallel_minus_one'],
                'max_abs_alpha_perp_minus_one': stats['max_abs_alpha_perp_minus_one'],
            })
        else:
            metadata.update({'alpha_validation': False})
        return loaded, metadata

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
