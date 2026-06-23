"""Catalog-level blinding for DESI clustering catalogs.

This module owns the in-memory catalog blinding operations used by DESI LSS
analysis scripts.  Heavy catalog dependencies are imported lazily so that the
standard data-vector blinding API remains lightweight.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import warnings
from pathlib import Path

import numpy as np

from .data_vector import SHIFTS_DIR


logger = logging.getLogger("desiblind.catalog")

CATALOG_PARAMETERS_FILENAME = "catalog_blinding.npy"
LEGACY_CATALOG_PARAMETERS_FILENAMES = ("catalog_blinding_2026_06.npy",)
CATALOG_INTERNAL_NAME = "catalog_blinding"
DEFAULT_AP_ZRANGE = (0.1, 2.1)
DEFAULT_AP_MAX_SHIFT = 0.03
DEFAULT_MAX_DF_FRACTION = 0.1
DEFAULT_FNL_LIMITS = (-20.0, 20.0)

# Fallback values inherited from the LSS catalog-blinding scripts.  Production
# callers can override these with analysis-specific values.
LSS_DEFAULT_ZEFF = {"BGS": 0.25, "LRG": 0.8, "LGE": 0.8, "ELG": 1.1, "QSO": 1.6}
LSS_DEFAULT_BIAS = {"BGS": 1.8, "LRG": 2.0, "LGE": 2.0, "ELG": 1.3, "QSO": 2.3}


def _make_tuple(value):
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _to_builtin(value):
    """Return a plain-Python representation suitable for attrs/YAML records."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(val) for val in value]
    return value


def _simple_tracer(tracer):
    """Map DESI catalog tracer names to the blinding fallback tracer family."""
    if isinstance(tracer, (list, tuple)):
        tracer = "_".join(map(str, tracer))
    text = str(tracer).upper()
    if "BGS" in text:
        return "BGS"
    if "LRG" in text:
        return "LRG"
    if "LGE" in text:
        return "LGE"
    if "ELG" in text:
        return "ELG"
    if "QSO" in text:
        return "QSO"
    raise ValueError(f"Unknown DESI catalog tracer {tracer!r}.")


def _catalog_parameter_key(bid):
    return hashlib.sha256(f"{CATALOG_INTERNAL_NAME}_bid{int(bid)}".encode()).hexdigest()


def _load_w0wa_table(w0wa_table):
    table = np.loadtxt(w0wa_table)
    table = np.asarray(table)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"w0/wa table must have at least two columns, got shape {table.shape}.")
    return table


def _ap_shift_summary(w0, wa, zrange=DEFAULT_AP_ZRANGE, nz=256):
    from cosmoprimo.fiducial import DESI

    z = np.linspace(float(zrange[0]), float(zrange[1]), int(nz))
    cosmo_fid = DESI()
    cosmo_blind = cosmo_fid.clone(w0_fld=float(w0), wa_fld=float(wa))
    dm_fid = cosmo_fid.comoving_angular_distance(z)
    dh_fid = 1.0 / cosmo_fid.hubble_function(z)
    dm_blind = cosmo_blind.comoving_angular_distance(z)
    dh_blind = 1.0 / cosmo_blind.hubble_function(z)
    alpha_perp = dm_blind / dm_fid
    alpha_parallel = dh_blind / dh_fid
    return {
        "alpha_perp_min": float(np.min(alpha_perp)),
        "alpha_perp_max": float(np.max(alpha_perp)),
        "alpha_parallel_min": float(np.min(alpha_parallel)),
        "alpha_parallel_max": float(np.max(alpha_parallel)),
        "max_abs_ap_shift": float(np.max(np.abs(np.concatenate([alpha_perp - 1.0, alpha_parallel - 1.0])))),
    }


def compute_fgrowth_blind(
    w0,
    wa,
    tracer="LRG",
    zeff=None,
    bias=None,
    fiducial_f=0.8,
    max_df_fraction=DEFAULT_MAX_DF_FRACTION,
    cap=True,
):
    """Return the RSD growth-rate blind implied by the AP volume factor."""
    from cosmoprimo.fiducial import DESI

    stracer = _simple_tracer(tracer)
    zeff = LSS_DEFAULT_ZEFF[stracer] if zeff is None else float(zeff)
    bias = LSS_DEFAULT_BIAS[stracer] if bias is None else float(bias)
    fiducial_f = float(fiducial_f)

    cosmo_fid = DESI()
    cosmo_shift = cosmo_fid.clone(w0_fld=float(w0), wa_fld=float(wa))

    dm_fid = cosmo_fid.comoving_angular_distance(zeff)
    dh_fid = 1.0 / cosmo_fid.hubble_function(zeff)
    dm_shift = cosmo_shift.comoving_angular_distance(zeff)
    dh_shift = 1.0 / cosmo_shift.hubble_function(zeff)
    vol_fac = (dm_shift**2 * dh_shift) / (dm_fid**2 * dh_fid)

    aa = 0.2 / bias**2
    bb = 2.0 / (3.0 * bias)
    cc = 1.0 - (1.0 + 0.2 * (fiducial_f / bias) ** 2.0 + 2.0 / 3.0 * fiducial_f / bias) / vol_fac
    f_shift = (-bb + np.sqrt(bb**2.0 - 4.0 * aa * cc)) / (2.0 * aa)

    df_fraction = (f_shift - fiducial_f) / fiducial_f
    if abs(df_fraction) > max_df_fraction:
        if not cap:
            raise ValueError(
                f"RSD f-growth shift {df_fraction:.4f} exceeds max_df_fraction={max_df_fraction}."
            )
        df_fraction = max_df_fraction * df_fraction / abs(df_fraction)
        f_shift = (1.0 + df_fraction) * fiducial_f
    return float(f_shift)


def _validate_candidate(
    candidate,
    modes=("bao", "rsd", "fnl"),
    tracer="LRG",
    zeff=None,
    bias=None,
    fiducial_f=0.8,
    ap_zrange=DEFAULT_AP_ZRANGE,
    ap_max_shift=DEFAULT_AP_MAX_SHIFT,
    max_df_fraction=DEFAULT_MAX_DF_FRACTION,
    fnl_limits=DEFAULT_FNL_LIMITS,
):
    modes = TracerCatalogBlinder.get_blinding_modes(modes)
    needs_w0wa = any(mode in modes for mode in ["bao", "rsd"])
    if needs_w0wa:
        for name in ["w0", "wa"]:
            if name not in candidate:
                raise ValueError(f"Catalog blinding candidate is missing {name!r}.")
        ap_summary = _ap_shift_summary(candidate["w0"], candidate["wa"], zrange=ap_zrange)
        if ap_summary["max_abs_ap_shift"] > ap_max_shift:
            raise ValueError(
                f"AP shift {ap_summary['max_abs_ap_shift']:.4f} exceeds limit {ap_max_shift:.4f} "
                f"for w0={candidate['w0']}, wa={candidate['wa']}."
            )
    if "rsd" in modes:
        fgrowth = compute_fgrowth_blind(
            candidate["w0"],
            candidate["wa"],
            tracer=tracer,
            zeff=zeff,
            bias=bias,
            fiducial_f=fiducial_f,
            max_df_fraction=max_df_fraction,
            cap=False,
        )
        df_fraction = abs((fgrowth - float(fiducial_f)) / float(fiducial_f))
        if df_fraction > max_df_fraction:
            raise ValueError(f"RSD f-growth shift {df_fraction:.4f} exceeds limit {max_df_fraction:.4f}.")
    if "fnl" in modes:
        if "fnl_blind" not in candidate:
            raise ValueError("Catalog blinding candidate is missing 'fnl_blind'.")
        low, high = map(float, fnl_limits)
        if not low <= float(candidate["fnl_blind"]) <= high:
            raise ValueError(f"fNL value {candidate['fnl_blind']} is outside [{low}, {high}].")


def make_catalog_parameter_candidate(w0, wa, fnl_blind, source=None):
    """Return one serializable catalog-blinding parameter candidate."""
    return {
        "w0": float(w0),
        "wa": float(wa),
        "fnl_blind": float(fnl_blind),
        "source": _to_builtin(source or {}),
    }


def generate_catalog_parameters(
    parameters_fn=None,
    save_dir=None,
    w0=None,
    wa=None,
    fnl_blind=None,
    w0wa_table=None,
    rows=None,
    seed=42,
    nrealizations=100,
    fnl_limits=DEFAULT_FNL_LIMITS,
    modes=("bao", "rsd", "fnl"),
    overwrite=False,
    validate=True,
    validation_tracer="LRG",
    **validation_options,
):
    """Generate the sealed ``.npy`` secret file used by catalog blinding.

    If ``w0`` and ``wa`` are provided, the same explicit values are stored in
    every hidden candidate.  If ``w0wa_table`` is provided, candidates are drawn
    from the selected rows or sampled deterministically from the table.
    """
    fn = TracerCatalogBlinder.get_parameters_fn(parameters_fn=parameters_fn, save_dir=save_dir)
    if fn.exists() and not overwrite:
        raise FileExistsError(f"{fn} already exists; pass overwrite=True to replace it.")

    nrealizations = int(nrealizations)
    rng = np.random.RandomState(int(seed))
    low_fnl, high_fnl = map(float, fnl_limits)

    if (w0 is None) != (wa is None):
        raise ValueError("Explicit generation requires both w0 and wa.")
    if w0 is None:
        if w0wa_table is None:
            raise ValueError("Provide either explicit w0/wa values or a w0wa_table.")
        table = _load_w0wa_table(w0wa_table)
        if rows is None:
            selected_rows = rng.choice(len(table), size=nrealizations, replace=len(table) < nrealizations)
        else:
            selected_rows = np.asarray(rows, dtype=int)
            if len(selected_rows) == 1:
                selected_rows = np.repeat(selected_rows, nrealizations)
            if len(selected_rows) < nrealizations:
                raise ValueError(f"Need at least {nrealizations} rows or one row to repeat.")
            selected_rows = selected_rows[:nrealizations]
        w0wa = table[selected_rows, :2]
    else:
        selected_rows = [None] * nrealizations
        w0wa = np.repeat([[float(w0), float(wa)]], nrealizations, axis=0)

    if fnl_blind is None:
        fnls = rng.uniform(low=low_fnl, high=high_fnl, size=nrealizations)
    else:
        fnls = np.repeat(float(fnl_blind), nrealizations)

    candidates = {}
    for bid, ((candidate_w0, candidate_wa), candidate_fnl, row) in enumerate(zip(w0wa, fnls, selected_rows)):
        source = {"generator": "desiblind.catalog.generate_catalog_parameters", "bid": int(bid), "seed": int(seed)}
        if w0wa_table is not None:
            source.update({"w0wa_table": str(w0wa_table), "row": None if row is None else int(row)})
        else:
            source.update({"explicit": True})
        candidate = make_catalog_parameter_candidate(candidate_w0, candidate_wa, candidate_fnl, source=source)
        if validate:
            _validate_candidate(
                candidate,
                modes=modes,
                tracer=validation_tracer,
                fnl_limits=fnl_limits,
                **validation_options,
            )
        candidates[_catalog_parameter_key(bid)] = candidate

    fn.parent.mkdir(parents=True, exist_ok=True)
    np.save(fn, candidates)
    return fn


class TracerCatalogBlinder:
    """Apply DESI catalog-level blinding to in-memory catalog objects.

    BAO/AP redshift remapping has an inverse helper for validation. RSD and
    fNL catalog-level blinding are apply-only; regenerate from the original
    catalogs instead.
    """

    blinded_nmax = 100
    parameters_filename = CATALOG_PARAMETERS_FILENAME
    legacy_parameters_filenames = LEGACY_CATALOG_PARAMETERS_FILENAMES

    @classmethod
    def _get_bid(cls):
        rng = np.random.RandomState(seed=42)
        return int(rng.randint(0, cls.blinded_nmax))

    @classmethod
    def get_parameters_fn(cls, save_dir=None, parameters_fn=None):
        if parameters_fn is not None:
            return Path(parameters_fn)
        if save_dir is None:
            save_dir = SHIFTS_DIR
        return Path(save_dir) / cls.parameters_filename

    @classmethod
    def _find_legacy_parameters_fn(cls, save_dir=None):
        if save_dir is None:
            save_dir = SHIFTS_DIR
        save_dir = Path(save_dir)
        for filename in cls.legacy_parameters_filenames:
            fn = save_dir / filename
            if fn.exists():
                return fn
        return None

    @classmethod
    def get_blinding_modes(cls, modes):
        aliases = {"ap": "bao", "bao_ap": "bao", "png": "fnl", "local_png": "fnl"}
        normalized = []
        for mode in _make_tuple(modes):
            public = aliases.get(str(mode).lower(), str(mode).lower())
            if public not in ["bao", "rsd", "fnl"]:
                raise ValueError(f"Unknown catalog blinding mode {mode!r}.")
            if public not in normalized:
                normalized.append(public)
        return tuple(normalized)

    @classmethod
    def _load_secret_candidate(cls, save_dir=None, parameters_fn=None):
        fn = cls.get_parameters_fn(save_dir=save_dir, parameters_fn=parameters_fn)
        if parameters_fn is None and not fn.exists():
            legacy_fn = cls._find_legacy_parameters_fn(save_dir=save_dir)
            if legacy_fn is not None:
                warnings.warn(
                    f"Loading legacy catalog blinding parameter file {legacy_fn.name!r}. "
                    f"Pass parameters_fn explicitly or rename/copy it to {cls.parameters_filename!r}.",
                    FutureWarning,
                    stacklevel=2,
                )
                fn = legacy_fn
        candidates = np.load(fn, allow_pickle=True).item()
        bid = cls._get_bid()
        key = _catalog_parameter_key(bid)
        try:
            candidate = dict(candidates[key])
        except KeyError as exc:
            raise KeyError(f"Catalog blinding parameters in {fn} are missing key for bid={bid}.") from exc
        return fn, key, bid, candidate

    @classmethod
    def load_parameters(
        cls,
        modes=("bao",),
        tracer="LRG",
        save_dir=None,
        parameters_fn=None,
        metadata="sealed",
        validate=True,
        **options,
    ):
        """Load and resolve catalog-level blinding parameters for one tracer."""
        modes = cls.get_blinding_modes(modes)
        if not modes:
            return None
        fn, key, bid, candidate = cls._load_secret_candidate(save_dir=save_dir, parameters_fn=parameters_fn)
        params = {
            "modes": modes,
            "metadata": metadata,
            "parameter_mode": "secret_file",
            "parameters_fn": str(fn),
            "parameters_file_sha256": hashlib.sha256(fn.read_bytes()).hexdigest(),
            "parameters_key": key,
            "bid": int(bid),
            "options": dict(options),
        }
        for name in ["w0", "wa", "fnl_blind"]:
            if name in candidate:
                params[name] = float(candidate[name])
        if "source" in candidate:
            params["source"] = _to_builtin(candidate["source"])

        fiducial_f = float(options.get("fiducial_f", 0.8))
        if validate:
            _validate_candidate(
                candidate,
                modes=modes,
                tracer=tracer,
                zeff=options.get("zeff", None),
                bias=options.get("bias", None),
                fiducial_f=fiducial_f,
                ap_zrange=options.get("ap_zrange", DEFAULT_AP_ZRANGE),
                ap_max_shift=options.get("ap_max_shift", DEFAULT_AP_MAX_SHIFT),
                max_df_fraction=options.get("max_df_fraction", DEFAULT_MAX_DF_FRACTION),
                fnl_limits=options.get("fnl_limits", DEFAULT_FNL_LIMITS),
            )

        if "rsd" in modes:
            params["fiducial_f"] = fiducial_f
            params["fgrowth_blind"] = compute_fgrowth_blind(
                params["w0"],
                params["wa"],
                tracer=tracer,
                zeff=options.get("zeff", None),
                bias=options.get("bias", None),
                fiducial_f=fiducial_f,
                max_df_fraction=options.get("max_df_fraction", DEFAULT_MAX_DF_FRACTION),
                cap=options.get("cap_fgrowth", True),
            )
        return params

    @classmethod
    def from_options(cls, options, tracer="LRG", **kwargs):
        """Resolve parameters from a desi-clustering style ``blinding`` dict."""
        if not options:
            return None
        options = dict(options)
        modes = options.pop("modes", options.pop("mode", options.pop("kind", "bao")))
        save_dir = options.pop("save_dir", None)
        parameters_fn = options.pop("parameters_fn", None)
        metadata = options.pop("metadata", "sealed")
        validate = options.pop("validate", True)
        options.update(kwargs)
        return cls.load_parameters(
            modes=modes,
            tracer=tracer,
            save_dir=save_dir,
            parameters_fn=parameters_fn,
            metadata=metadata,
            validate=validate,
            **options,
        )

    @classmethod
    def blinding_attrs(cls, params):
        """Return attrs to attach to blinded catalogs and measurements."""
        if not params:
            return {}
        modes = cls.get_blinding_modes(params.get("modes", ("bao",)))
        attrs = {
            "catalog_blinding": ",".join(modes),
            "catalog_blinding_parameter_mode": params.get("parameter_mode", "secret_file"),
            "catalog_blinding_metadata": params.get("metadata", "sealed"),
        }
        if params.get("parameters_fn", None):
            attrs["catalog_blinding_parameters_file"] = Path(params["parameters_fn"]).name
        if params.get("parameters_key", None):
            attrs["catalog_blinding_parameters_key"] = hashlib.sha256(params["parameters_key"].encode()).hexdigest()[:16]
        if params.get("parameters_file_sha256", None):
            attrs["catalog_blinding_parameters_file_sha256"] = params["parameters_file_sha256"][:16]
        if params.get("metadata", "sealed") == "open":
            for key in ["w0", "wa", "fgrowth_blind", "fiducial_f", "fnl_blind", "bid", "source"]:
                if key in params and params[key] is not None:
                    attrs[f"catalog_blinding_{key}"] = _to_builtin(params[key])
        return attrs

    @classmethod
    def output_version(cls, version, params_or_options):
        if not params_or_options:
            return version
        modes = params_or_options.get("modes", ("bao",))
        modes = cls.get_blinding_modes(modes)
        if not modes:
            return version
        options = params_or_options.get("options", params_or_options)
        suffix = options.get("output_version_suffix", None)
        if suffix is False:
            return version
        if suffix is None:
            suffix = "{}-blinded".format("-".join(modes))
        suffix = str(suffix).strip("-")
        if not suffix:
            return version
        if version is None:
            return suffix
        if suffix in str(version):
            return version
        return f"{version}-{suffix}"

    @classmethod
    def output_version_from_options(cls, version, options):
        if not options:
            return version
        options = dict(options)
        modes = cls.get_blinding_modes(options.get("modes", options.get("mode", options.get("kind", "bao"))))
        if not modes:
            return version
        return cls.output_version(version, {"modes": modes, "options": options})

    @classmethod
    def _copy_catalog_with_attrs(cls, catalog, **columns):
        new = catalog.copy()
        new.attrs.update(getattr(catalog, "attrs", {}))
        for name, value in columns.items():
            new[name] = value
        return new

    @classmethod
    def _with_attrs(cls, catalog, params):
        attrs = cls.blinding_attrs(params)
        if attrs:
            catalog.attrs.update(attrs)
        return catalog

    @staticmethod
    def _get_fiducial_cosmology(cosmo_fid="DESI"):
        if isinstance(cosmo_fid, str):
            if cosmo_fid != "DESI":
                raise ValueError(f"Unknown fiducial cosmology {cosmo_fid!r}; pass a cosmology object instead.")
            from cosmoprimo.fiducial import DESI

            return DESI()
        return cosmo_fid

    @classmethod
    def transform_redshift(cls, z, w0, wa, cosmo_fid="DESI", inverse=False):
        """Return BAO/AP-remapped redshifts for a blinded ``w0``/``wa`` cosmology."""
        from cosmoprimo.utils import DistanceToRedshift

        cosmo_fid = cls._get_fiducial_cosmology(cosmo_fid=cosmo_fid)
        cosmo_blind = cosmo_fid.clone(w0_fld=float(w0), wa_fld=float(wa))

        scalar = np.ndim(z) == 0
        zin = np.asarray(z, dtype="f8")
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
    def _require_bao_parameters(cls, params):
        if not params:
            return None
        modes = params.get("modes", None)
        if modes is not None and "bao" not in cls.get_blinding_modes(modes):
            return None
        missing = [name for name in ["w0", "wa"] if name not in params]
        if missing:
            raise ValueError(f"Catalog BAO blinding parameters missing required key(s): {missing}.")
        return params

    @classmethod
    def _apply_bao_redshift_transform(
        cls,
        catalog,
        params,
        zcol="Z",
        input_zcol=None,
        output_zcol=None,
        copy=True,
        inverse=False,
    ):
        input_zcol = zcol if input_zcol is None else input_zcol
        output_zcol = zcol if output_zcol is None else output_zcol
        z_shift = cls.transform_redshift(catalog[input_zcol], w0=params["w0"], wa=params["wa"], inverse=inverse)
        if copy:
            new = cls._copy_catalog_with_attrs(catalog, **{output_zcol: z_shift})
        else:
            new = catalog
            new[output_zcol] = z_shift
        return cls._with_attrs(new, params)

    @classmethod
    def apply_bao_blinding(cls, catalog, params, zcol="Z", input_zcol=None, output_zcol=None, copy=True):
        """Apply AP/BAO redshift remapping to one raw catalog."""
        params = cls._require_bao_parameters(params)
        if params is None:
            return catalog
        return cls._apply_bao_redshift_transform(
            catalog,
            params,
            zcol=zcol,
            input_zcol=input_zcol,
            output_zcol=output_zcol,
            copy=copy,
        )

    @classmethod
    def apply_bao_blinding_to_catalogs(cls, catalogs, params, zcol="Z", input_zcol=None, output_zcol=None, copy=True):
        """Apply AP/BAO redshift remapping to one catalog or a list of catalogs."""
        if isinstance(catalogs, (list, tuple)):
            return [
                cls.apply_bao_blinding(
                    catalog,
                    params,
                    zcol=zcol,
                    input_zcol=input_zcol,
                    output_zcol=output_zcol,
                    copy=copy,
                )
                for catalog in catalogs
            ]
        return cls.apply_bao_blinding(
            catalogs,
            params,
            zcol=zcol,
            input_zcol=input_zcol,
            output_zcol=output_zcol,
            copy=copy,
        )

    @classmethod
    def remove_bao_blinding(
        cls,
        catalog,
        params,
        zcol="Z",
        input_zcol=None,
        output_zcol=None,
        copy=True,
        force=False,
    ):
        """Remove AP/BAO redshift remapping from one raw catalog."""
        if not force:
            raise ValueError('Are you sure you want to unblind? If so, provide "force=True"')
        params = cls._require_bao_parameters(params)
        if params is None:
            return catalog
        return cls._apply_bao_redshift_transform(
            catalog,
            params,
            zcol=zcol,
            input_zcol=input_zcol,
            output_zcol=output_zcol,
            copy=copy,
            inverse=True,
        )

    @classmethod
    def remove_bao_blinding_from_catalogs(
        cls,
        catalogs,
        params,
        zcol="Z",
        input_zcol=None,
        output_zcol=None,
        copy=True,
        force=False,
    ):
        """Remove AP/BAO redshift remapping from one catalog or a list of catalogs."""
        if isinstance(catalogs, (list, tuple)):
            return [
                cls.remove_bao_blinding(
                    catalog,
                    params,
                    zcol=zcol,
                    input_zcol=input_zcol,
                    output_zcol=output_zcol,
                    copy=copy,
                    force=force,
                )
                for catalog in catalogs
            ]
        return cls.remove_bao_blinding(
            catalogs,
            params,
            zcol=zcol,
            input_zcol=input_zcol,
            output_zcol=output_zcol,
            copy=copy,
            force=force,
        )

    @staticmethod
    def _positions_to_rdz(positions):
        from cosmoprimo.fiducial import DESI
        from cosmoprimo.utils import DistanceToRedshift
        from mockfactory import cartesian_to_sky

        distance, ra, dec = cartesian_to_sky(positions)
        redshift = DistanceToRedshift(DESI().comoving_radial_distance)(distance)
        return ra, dec, redshift

    @classmethod
    def apply_rsd_blinding(cls, data, randoms, params, tracer="LRG"):
        """Apply catalog-level RSD blinding to prepared data positions."""
        if not params or "rsd" not in params["modes"]:
            return data
        from cosmoprimo.fiducial import DESI
        from mockfactory import Catalog
        from mockfactory.blinding import CutskyCatalogBlinding

        options = params.get("options", {})
        stracer = _simple_tracer(tracer)
        zeff = options.get("zeff", LSS_DEFAULT_ZEFF[stracer])
        bias = options.get("bias", LSS_DEFAULT_BIAS[stracer])

        cosmo_blind = DESI()
        cosmo_blind._derived["f"] = params["fgrowth_blind"]
        crandoms = Catalog.concatenate(randoms)
        blinding = CutskyCatalogBlinding(
            cosmo_fid="DESI",
            cosmo_blind=cosmo_blind,
            bias=bias,
            z=zeff,
            position_type="pos",
            mpicomm=data.mpicomm,
        )
        positions = blinding.rsd(
            data["POSITION"],
            data_weights=data["INDWEIGHT"],
            randoms_positions=crandoms["POSITION"],
            randoms_weights=crandoms["INDWEIGHT"],
            recon=options.get("rsd_recon", "IterativeFFTReconstruction"),
            smoothing_radius=options.get("rsd_smoothing_radius", 15.0),
            **options.get("rsd_kwargs", {}),
        )
        ra, dec, redshift = cls._positions_to_rdz(positions)
        new = cls._copy_catalog_with_attrs(data, POSITION=positions, RA=ra, DEC=dec, Z=redshift)
        return cls._with_attrs(new, params)

    @classmethod
    def apply_fnl_blinding(cls, data, randoms, params, tracer="LRG"):
        """Apply local-PNG/fNL data-weight blinding to prepared catalogs."""
        if not params or "fnl" not in params["modes"]:
            return data, randoms
        from cosmoprimo.fiducial import DESI
        from mockfactory import Catalog
        from mockfactory.blinding import CutskyCatalogBlinding

        options = params.get("options", {})
        stracer = _simple_tracer(tracer)
        zeff = options.get("zeff", LSS_DEFAULT_ZEFF[stracer])
        bias = options.get("bias", LSS_DEFAULT_BIAS[stracer])
        method = options.get("fnl_method", "data_weights")
        if method != "data_weights":
            raise NotImplementedError(
                'Only fnl_method="data_weights" is currently validated for DESI catalog blinding.'
            )

        cosmo_blind = DESI()
        cosmo_blind._derived["fnl"] = params["fnl_blind"]
        crandoms = Catalog.concatenate(randoms)
        blinding = CutskyCatalogBlinding(
            cosmo_fid="DESI",
            cosmo_blind=cosmo_blind,
            bias=bias,
            z=zeff,
            position_type="pos",
            mpicomm=data.mpicomm,
        )
        new_weights = blinding.png(
            data["POSITION"],
            data_weights=data["INDWEIGHT"],
            randoms_positions=crandoms["POSITION"],
            randoms_weights=crandoms["INDWEIGHT"],
            method=method,
            shotnoise_correction=options.get("fnl_shotnoise_correction", True),
            smoothing_radius=options.get("fnl_smoothing_radius", 30.0),
            **options.get("fnl_kwargs", {}),
        )
        blind_weight = new_weights / data["INDWEIGHT"]
        new = cls._copy_catalog_with_attrs(data, INDWEIGHT=new_weights, WEIGHT_BLIND=blind_weight)
        return cls._with_attrs(new, params), randoms


def _parse_rows(rows):
    if rows is None:
        return None
    parsed = []
    for value in rows:
        parsed.extend(int(item) for item in str(value).split(",") if item != "")
    return parsed


def collect_argparser():
    parser = argparse.ArgumentParser(description="Generate DESI catalog-level blinding secret parameters.")
    parser.add_argument("--parameters-fn", default=None, help="Explicit .npy parameter filename; recommended for production.")
    parser.add_argument("--save-dir", default=None, help=f"Directory for the generic default {CATALOG_PARAMETERS_FILENAME}.")
    parser.add_argument("--w0", type=float, default=None, help="Explicit global w0 value.")
    parser.add_argument("--wa", type=float, default=None, help="Explicit global wa value.")
    parser.add_argument("--fnl-blind", type=float, default=None, help="Explicit global fNL value.")
    parser.add_argument("--w0wa-table", default=None, help="Text table with w0 and wa in the first two columns.")
    parser.add_argument("--rows", nargs="*", default=None, help="Rows to select from --w0wa-table; comma-separated values allowed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nrealizations", type=int, default=100)
    parser.add_argument("--modes", nargs="+", default=["bao", "rsd", "fnl"], choices=["bao", "ap", "rsd", "fnl", "png"])
    parser.add_argument("--fnl-limits", nargs=2, type=float, default=DEFAULT_FNL_LIMITS)
    parser.add_argument("--validation-tracer", default="LRG")
    parser.add_argument("--zeff", type=float, default=None)
    parser.add_argument("--bias", type=float, default=None)
    parser.add_argument("--fiducial-f", type=float, default=0.8)
    parser.add_argument("--ap-max-shift", type=float, default=DEFAULT_AP_MAX_SHIFT)
    parser.add_argument("--max-df-fraction", type=float, default=DEFAULT_MAX_DF_FRACTION)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    return parser


def main(args=None):
    parser = collect_argparser()
    ns = parser.parse_args(args=args)
    fn = generate_catalog_parameters(
        parameters_fn=ns.parameters_fn,
        save_dir=ns.save_dir,
        w0=ns.w0,
        wa=ns.wa,
        fnl_blind=ns.fnl_blind,
        w0wa_table=ns.w0wa_table,
        rows=_parse_rows(ns.rows),
        seed=ns.seed,
        nrealizations=ns.nrealizations,
        fnl_limits=tuple(ns.fnl_limits),
        modes=ns.modes,
        overwrite=ns.overwrite,
        validate=not ns.no_validate,
        validation_tracer=ns.validation_tracer,
        zeff=ns.zeff,
        bias=ns.bias,
        fiducial_f=ns.fiducial_f,
        ap_max_shift=ns.ap_max_shift,
        max_df_fraction=ns.max_df_fraction,
    )
    logger.info("Wrote %s", fn)
    return fn


if __name__ == "__main__":
    main()
