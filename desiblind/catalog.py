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
    """Return ``value`` as a tuple while treating strings as atomic values."""
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
    """Return the hidden dictionary key for one catalog-blinding realization."""
    return hashlib.sha256(f"{CATALOG_INTERNAL_NAME}_bid{int(bid)}".encode()).hexdigest()


def _load_w0wa_table(w0wa_table):
    """Load a text table whose first two columns are ``w0`` and ``wa``."""
    table = np.loadtxt(w0wa_table)
    table = np.asarray(table)
    if table.ndim != 2 or table.shape[1] < 2:
        raise ValueError(f"w0/wa table must have at least two columns, got shape {table.shape}.")
    return table


def _ap_shift_summary(w0, wa, zrange=DEFAULT_AP_ZRANGE, nz=256):
    """Summarize AP scaling shifts for one candidate dark-energy model."""
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


def _get_from_cosmo(cosmo, name, z=None):
    """Return a scalar cosmological quantity, including derived f/fNL values."""

    def check_z():
        """Require an explicit redshift for redshift-dependent quantities."""
        if z is None:
            raise ValueError("z is None.")

    if name.startswith("omega"):
        return _get_from_cosmo(cosmo, "O" + name[1:], z=z) * cosmo.h**2
    if name.startswith("Omega"):
        if z is None:
            name = name[:5] + "0" + name[5:]
            return getattr(cosmo, name)
        check_z()
        return getattr(cosmo, name)(z)
    if name in getattr(cosmo, "_derived", {}):
        return cosmo._derived[name]
    if name == "fsigma8":
        check_z()
        return cosmo.get_fourier().sigma8_z(z, of="theta_cb")
    if name == "sigma8":
        check_z()
        return cosmo.get_fourier().sigma8_z(z, of="delta_cb")
    if name == "f":
        return _get_from_cosmo(cosmo, "fsigma8", z=z) / _get_from_cosmo(cosmo, "sigma8", z=z)
    if name == "fnl":
        return 0.0
    return getattr(cosmo, name)


def _pop_recon_kwargs(kwargs, default_cellsize):
    """Split user reconstruction options into mesh and jax-recon keywords."""
    kwargs = dict(kwargs)
    if not any(name in kwargs for name in ["nmesh", "meshsize", "cellsize"]):
        kwargs["cellsize"] = default_cellsize

    mesh_kwargs = {}
    if "nmesh" in kwargs:
        mesh_kwargs["meshsize"] = kwargs.pop("nmesh")
    for name in [
        "meshsize",
        "boxsize",
        "boxcenter",
        "cellsize",
        "boxpad",
        "check",
        "approximate",
        "dtype",
        "primes",
        "divisors",
        "sharding_mesh",
        "fft_backend",
    ]:
        if name in kwargs:
            mesh_kwargs[name] = kwargs.pop(name)

    recon_kwargs = {}
    for name, default in [
        ("los", None),
        ("resampler", "cic"),
        ("halo_add", 0),
        ("threshold_randoms", ("noise", 0.01)),
        ("niterations", 3),
    ]:
        recon_kwargs[name] = kwargs.pop(name, default)

    if kwargs:
        raise TypeError(f"Unknown reconstruction keyword argument(s): {', '.join(sorted(kwargs))}")
    return mesh_kwargs, recon_kwargs


def _get_reconstruction_class(recon):
    """Return a jax-recon reconstruction class from a class or class name."""
    from jaxrecon import zeldovich

    if not isinstance(recon, str):
        return recon
    try:
        return getattr(zeldovich, recon)
    except AttributeError as exc:
        raise ValueError(f"Unknown jax-recon reconstruction {recon!r}.") from exc


def _gather_to_all(array, mpicomm=None):
    """Gather an MPI-scattered array and broadcast the gathered copy."""
    if array is None or mpicomm is None or getattr(mpicomm, "size", 1) == 1:
        return array
    import mpytools as mpy

    array = mpy.gather(array, mpicomm=mpicomm, mpiroot=0)
    if mpicomm.rank == 0:
        array = np.asarray(array)
    return mpicomm.bcast(array, root=0)


def _make_particle_field(positions, weights=None, attrs=None, mpicomm=None):
    """Build a jaxpower particle field after gathering MPI-scattered arrays."""
    from jaxpower import ParticleField

    positions = _gather_to_all(positions, mpicomm=mpicomm)
    weights = _gather_to_all(weights, mpicomm=mpicomm)
    return ParticleField(positions, weights, attrs=attrs)


def _build_reconstruction(
    data_positions,
    data_weights=None,
    randoms_positions=None,
    randoms_weights=None,
    f=None,
    bias=None,
    smoothing_radius=15.0,
    dtype=None,
    mpicomm=None,
    default_cellsize=7.0,
    recon="IterativeFFTReconstruction",
    **kwargs,
):
    """Build the jax-recon object and mesh attributes used by RSD/fNL blinding."""
    from jaxpower import FKPField, get_mesh_attrs

    reconstruction_algorithm = _get_reconstruction_class(recon)
    mesh_kwargs, recon_kwargs = _pop_recon_kwargs(kwargs, default_cellsize=default_cellsize)
    if dtype is not None:
        mesh_kwargs.setdefault("dtype", dtype)

    data_positions_all = _gather_to_all(data_positions, mpicomm=mpicomm)
    randoms_positions_all = _gather_to_all(randoms_positions, mpicomm=mpicomm)
    positions = [pos for pos in [data_positions_all, randoms_positions_all] if pos is not None]
    attrs = get_mesh_attrs(*positions, **mesh_kwargs)

    data = _make_particle_field(data_positions, data_weights, attrs=attrs, mpicomm=mpicomm)
    randoms = None
    if randoms_positions is not None:
        randoms = _make_particle_field(randoms_positions, randoms_weights, attrs=attrs, mpicomm=mpicomm)
    particles = FKPField(data, randoms, attrs=attrs) if randoms is not None else data
    kwargs_recon = {
        "resampler": recon_kwargs["resampler"],
        "halo_add": recon_kwargs["halo_add"],
        "smoothing_radius": smoothing_radius,
        "threshold_randoms": recon_kwargs["threshold_randoms"],
    }
    if reconstruction_algorithm.__name__ in ["IterativeFFTReconstruction", "IterativeFFTParticleReconstruction"]:
        kwargs_recon["niterations"] = recon_kwargs["niterations"]
    reconstruction = reconstruction_algorithm(
        particles,
        growth_rate=f,
        bias=bias,
        los=recon_kwargs["los"],
        **kwargs_recon,
    )
    return reconstruction, attrs, recon_kwargs


def _paint_particles(positions, weights=None, attrs=None, resampler="cic", halo_add=0, mpicomm=None):
    """Paint particle positions and optional weights onto a real mesh."""
    particles = _make_particle_field(positions, weights=weights, attrs=attrs, mpicomm=mpicomm)
    mesh = particles.paint(resampler=resampler, compensate=False, interlacing=0, halo_add=halo_add, out="real")
    return mesh, particles


def _get_threshold_randoms(randoms, threshold_randoms=0.01):
    """Resolve a random-catalog density threshold for density-contrast meshes."""
    if randoms is None or threshold_randoms is None:
        return None
    if isinstance(threshold_randoms, tuple):
        threshold_method, threshold_value = threshold_randoms
    else:
        threshold_method, threshold_value = "noise", threshold_randoms
    if threshold_method not in ["noise", "mean"]:
        raise ValueError('threshold_randoms method must be "noise" or "mean".')
    if threshold_method == "noise":
        return threshold_value * (randoms.weights**2).sum() / randoms.sum()
    return threshold_value * randoms.sum() / randoms.size


def _density_contrast(mesh_data, mesh_randoms=None, randoms=None, bias=1.0, smoothing_radius=15.0, threshold_randoms=0.01):
    """Estimate the bias-scaled density contrast used for PNG blinding."""
    from jaxrecon.zeldovich import estimate_mesh_delta

    threshold_randoms = _get_threshold_randoms(randoms, threshold_randoms=threshold_randoms)
    return (
        estimate_mesh_delta(
            mesh_data,
            mesh_randoms=mesh_randoms,
            threshold_randoms=threshold_randoms,
            smoothing_radius=smoothing_radius,
        )
        / bias
    )


def _apply_png_transfer(mesh, bfnl, tk):
    """Apply PNG transfer function to a jaxpower complex mesh."""
    import jax.numpy as jnp

    k = sum(np.asarray(kk) ** 2 for kk in mesh.attrs.kcoords(sparse=True)) ** 0.5
    transfer = np.zeros(k.shape, dtype=np.asarray(mesh.value).real.dtype)
    nonzero = k != 0.0
    transfer[nonzero] = bfnl / tk(k[nonzero])
    return mesh * jnp.asarray(transfer)


def _replace_mesh_zeros(mesh):
    """Replace exact zero mesh cells by one to avoid division by zero."""
    import jax.numpy as jnp

    return mesh.clone(value=jnp.where(mesh.value == 0.0, 1.0, mesh.value))


def _smooth_mesh(mesh, smoothing_radius=15.0):
    """Return a Gaussian-smoothed copy of a jaxpower mesh."""
    from jaxrecon.zeldovich import kernel_gaussian

    return (mesh.r2c() * kernel_gaussian(mesh.attrs, smoothing_radius=smoothing_radius)).c2r()


def _read_mesh(mesh, positions, resampler="cic", halo_add=0):
    """Read a mesh field at particle positions as a NumPy array."""
    return np.asarray(mesh.read(positions, resampler=resampler, compensate=False, halo_add=halo_add))


def _gradient_shifts(mesh, positions, resampler="cic", halo_add=0):
    """Read gradient displacements from a complex mesh at particle positions."""
    import jax.numpy as jnp

    kcoords = mesh.attrs.kcoords(sparse=True)
    k2 = sum(kk**2 for kk in kcoords)
    k2 = jnp.where(k2 == 0.0, 1.0, k2)
    disps = []
    for iaxis in range(mesh.attrs.ndim):
        psi = (mesh * (1j * kcoords[iaxis] / k2)).c2r()
        disps.append(_read_mesh(psi, positions, resampler=resampler, halo_add=halo_add))
    return np.column_stack(disps)


def _apply_rsd_jax_blinding(
    data_positions,
    data_weights=None,
    randoms_positions=None,
    randoms_weights=None,
    cosmo_fid=None,
    fgrowth_blind=None,
    bias=None,
    z=None,
    recon="IterativeFFTReconstruction",
    smoothing_radius=15.0,
    dtype=None,
    mpicomm=None,
    **kwargs,
):
    """Apply RSD blinding using desiblind-owned jax-recon logic."""
    data_positions = np.asarray(data_positions)
    f_fid = _get_from_cosmo(cosmo_fid, "f", z=z)
    reconstruction, attrs, recon_kwargs = _build_reconstruction(
        data_positions,
        data_weights=data_weights,
        randoms_positions=randoms_positions,
        randoms_weights=randoms_weights,
        f=f_fid,
        bias=bias,
        smoothing_radius=smoothing_radius,
        dtype=dtype,
        mpicomm=mpicomm,
        default_cellsize=7.0,
        recon=recon,
        **kwargs,
    )
    del attrs, recon_kwargs
    shifts = np.asarray(reconstruction.read_shifts(data_positions, field="rsd"))
    return data_positions + (float(fgrowth_blind) / f_fid - 1.0) * shifts


def _apply_fnl_jax_blinding(
    data_positions,
    data_weights=None,
    randoms_positions=None,
    randoms_weights=None,
    cosmo_fid=None,
    fnl_blind=0.0,
    bias=None,
    z=None,
    method="data_weights",
    recon="IterativeFFTReconstruction",
    smoothing_radius=30.0,
    shotnoise_correction=False,
    dtype=None,
    mpicomm=None,
    **kwargs,
):
    """Apply local-PNG/fNL blinding using desiblind-owned jax-recon logic."""
    import mpytools as mpy

    available_methods = ["data_weights", "randoms_weights", "data_positions", "randoms_positions"]
    if method not in available_methods:
        raise ValueError(f"blinding method {method} must be one of {available_methods}.")

    data_positions = np.asarray(data_positions)
    randoms_positions = None if randoms_positions is None else np.asarray(randoms_positions)
    f_fid = _get_from_cosmo(cosmo_fid, "f", z=z)
    reconstruction, attrs, recon_kwargs = _build_reconstruction(
        data_positions,
        data_weights=data_weights,
        randoms_positions=randoms_positions,
        randoms_weights=randoms_weights,
        f=f_fid,
        bias=bias,
        smoothing_radius=smoothing_radius,
        dtype=dtype,
        mpicomm=mpicomm,
        default_cellsize=15.0,
        recon=recon,
        **kwargs,
    )
    resampler, halo_add = recon_kwargs["resampler"], recon_kwargs["halo_add"]
    threshold_randoms = recon_kwargs["threshold_randoms"]
    sigma1 = smoothing_radius
    shifts = np.asarray(reconstruction.read_shifts(data_positions, field="rsd"))
    shifted_positions = data_positions - shifts
    mesh_data, _ = _paint_particles(
        shifted_positions,
        weights=data_weights,
        attrs=attrs,
        resampler=resampler,
        halo_add=halo_add,
        mpicomm=mpicomm,
    )
    mesh_randoms, randoms = None, None
    if randoms_positions is not None:
        mesh_randoms, randoms = _paint_particles(
            randoms_positions,
            weights=randoms_weights,
            attrs=attrs,
            resampler=resampler,
            halo_add=halo_add,
            mpicomm=mpicomm,
        )

    if "weights" not in method and shotnoise_correction:
        raise ValueError("No shot noise correction when blinding is based on particle shifts.")

    mesh_delta = _density_contrast(
        mesh_data,
        mesh_randoms=mesh_randoms,
        randoms=randoms,
        bias=bias,
        smoothing_radius=smoothing_radius,
        threshold_randoms=threshold_randoms,
    )
    sigma2 = smoothing_radius
    mesh = mesh_delta.r2c()
    bfnl = 2.0 * 1.686 * (bias - 1.0) * float(fnl_blind)

    pk_prim = cosmo_fid.get_primordial().pk_interpolator(mode="scalar")
    pk_lin = cosmo_fid.get_fourier().pk_interpolator(of="theta_cb").to_1d(z=z)

    def tk(k):
        """Return the transfer factor connecting primordial and linear power."""
        pphi_prim = 9.0 / 25.0 * 2.0 * np.pi**2 / k**3 * pk_prim(k) / cosmo_fid.h**3
        return (pk_lin(k) / pphi_prim) ** 0.5

    mesh = _apply_png_transfer(mesh, bfnl, tk)

    if shotnoise_correction:

        def s1(k):
            """Return the first Gaussian smoothing kernel value."""
            return np.exp(-0.5 * k**2 * sigma1**2)

        def s2(k):
            """Return the second Gaussian smoothing kernel value."""
            return np.exp(-0.5 * k**2 * sigma2**2)

        sum_w2, _ = _paint_particles(
            data_positions,
            weights=data_weights * data_weights if data_weights is not None else None,
            attrs=attrs,
            resampler=resampler,
            halo_add=halo_add,
            mpicomm=mpicomm,
        )

        sum_wd, _ = _paint_particles(
            data_positions,
            weights=data_weights,
            attrs=attrs,
            resampler=resampler,
            halo_add=halo_add,
            mpicomm=mpicomm,
        )

        if randoms_positions is not None:
            mesh_nbar, _ = _paint_particles(
                randoms_positions,
                weights=randoms_weights,
                attrs=attrs,
                resampler=resampler,
                halo_add=halo_add,
                mpicomm=mpicomm,
            )
            alpha = mpy.csum(
                data_weights if data_weights is not None else len(data_positions),
                mpicomm=mpicomm,
            ) / mpy.csum(
                randoms_weights if randoms_weights is not None else len(randoms_positions),
                mpicomm=mpicomm,
            )
            nbar = alpha / np.prod(np.asarray(attrs.cellsize)) * mesh_nbar
        else:
            nbar = mpy.csum(
                data_weights if data_weights is not None else len(data_positions),
                mpicomm=mpicomm,
            ) / np.prod(np.asarray(attrs.boxsize))

        sum_w2 = _replace_mesh_zeros(sum_w2)
        inv_shotnoise = sum_wd * nbar / sum_w2
        inv_shotnoise = _smooth_mesh(inv_shotnoise, smoothing_radius=smoothing_radius)

        mu_pivot = 0.6
        k_pivot = 4e-3 if bfnl >= 0.0 else 8e-3
        if "data" in method:
            shotnoise = 1.0 / _read_mesh(inv_shotnoise, data_positions, resampler=resampler, halo_add=halo_add)
        elif "randoms" in method:
            shotnoise = 1.0 / _read_mesh(inv_shotnoise, randoms_positions, resampler=resampler, halo_add=halo_add)
        else:
            shotnoise = 0.0

        mask = s1(pk_lin.k) > 1e-4
        sigma_d_2 = pk_lin.clone(k=pk_lin.k[mask], pk=(s1(pk_lin.k) ** 2 * pk_lin(pk_lin.k))[mask]).sigma_d() ** 2

        x_tilde = (bias + f_fid * mu_pivot**2) * (bias + (1.0 - s1(k_pivot)) * f_fid * mu_pivot**2)
        x_tilde *= s2(k_pivot) * pk_lin(k_pivot)
        x_tilde += s2(k_pivot) * shotnoise * np.exp(-0.5 * k_pivot**2 * mu_pivot**2 * f_fid**2 * sigma_d_2)
        y_tilde = (bias + (1.0 - s1(k_pivot)) * f_fid * mu_pivot**2) ** 2
        y_tilde *= s2(k_pivot) ** 2 * pk_lin(k_pivot)
        y_tilde += s2(k_pivot) ** 2 * shotnoise
        expected_pivot = 2.0 * bfnl / tk(k_pivot) * bias * (bias + f_fid * mu_pivot**2) * pk_lin(k_pivot)
        expected_pivot += (bfnl / tk(k_pivot)) ** 2 * bias**2 * pk_lin(k_pivot)

        shotnoise_factor = (-x_tilde + np.sqrt(x_tilde**2 + y_tilde * expected_pivot)) / y_tilde / (bfnl / tk(k_pivot))
    else:
        shotnoise_factor = 1.0

    if "weights" in method:
        mesh = mesh.c2r()
        if "data" in method:
            weights = _read_mesh(mesh, data_positions, resampler=resampler, halo_add=halo_add)
            weights = (1.0 if data_weights is None else data_weights) * (1.0 + shotnoise_factor * weights)
        elif "randoms" in method:
            weights = _read_mesh(mesh, randoms_positions, resampler=resampler, halo_add=halo_add)
            weights = (1.0 if randoms_weights is None else randoms_weights) * (1.0 - shotnoise_factor * weights)
        return weights

    positions = data_positions if "data" in method else randoms_positions
    shifts = _gradient_shifts(mesh, positions, resampler=resampler, halo_add=halo_add)
    shifts -= mpy.cmean(shifts, mpicomm=mpicomm)
    return positions + (shifts if "data" in method else -shifts)


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
    """Validate one sealed catalog-blinding candidate against safety limits."""
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
        """Return the deterministic blinded realization id used in production."""
        rng = np.random.RandomState(seed=42)
        return int(rng.randint(0, cls.blinded_nmax))

    @classmethod
    def get_parameters_fn(cls, save_dir=None, parameters_fn=None):
        """Return the catalog-blinding parameter filename to load or write."""
        if parameters_fn is not None:
            return Path(parameters_fn)
        if save_dir is None:
            save_dir = SHIFTS_DIR
        return Path(save_dir) / cls.parameters_filename

    @classmethod
    def _find_legacy_parameters_fn(cls, save_dir=None):
        """Return the first legacy parameter file present in ``save_dir``."""
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
        """Normalize user-facing catalog blinding mode names and aliases."""
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
        """Load the sealed parameter candidate selected by the hidden bid."""
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
        """Return an output version string with the configured blind suffix."""
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
        """Return an output version string using an unresolved options dict."""
        if not options:
            return version
        options = dict(options)
        modes = cls.get_blinding_modes(options.get("modes", options.get("mode", options.get("kind", "bao"))))
        if not modes:
            return version
        return cls.output_version(version, {"modes": modes, "options": options})

    @classmethod
    def _copy_catalog_with_attrs(cls, catalog, **columns):
        """Copy a catalog, preserve attrs, and replace selected columns."""
        new = catalog.copy()
        new.attrs.update(getattr(catalog, "attrs", {}))
        for name, value in columns.items():
            new[name] = value
        return new

    @classmethod
    def _with_attrs(cls, catalog, params):
        """Attach catalog-blinding attrs to a catalog-like object in place."""
        attrs = cls.blinding_attrs(params)
        if attrs:
            catalog.attrs.update(attrs)
        return catalog

    @staticmethod
    def _get_fiducial_cosmology(cosmo_fid="DESI"):
        """Return the fiducial cosmology object used by catalog blinding."""
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
        """Return BAO params when active, validating required AP keys."""
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
        """Apply the forward or inverse AP redshift transform to one catalog."""
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
        """Convert Cartesian positions to RA, DEC, and DESI-fiducial redshift."""
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

        options = params.get("options", {})
        stracer = _simple_tracer(tracer)
        zeff = options.get("zeff", LSS_DEFAULT_ZEFF[stracer])
        bias = options.get("bias", LSS_DEFAULT_BIAS[stracer])

        crandoms = Catalog.concatenate(randoms)
        rsd_kwargs = dict(options.get("rsd_kwargs", {}))
        recon = rsd_kwargs.pop("recon", options.get("rsd_recon", "IterativeFFTReconstruction"))
        positions = _apply_rsd_jax_blinding(
            data["POSITION"],
            data_weights=data["INDWEIGHT"],
            randoms_positions=crandoms["POSITION"],
            randoms_weights=crandoms["INDWEIGHT"],
            cosmo_fid=DESI(),
            fgrowth_blind=params["fgrowth_blind"],
            bias=bias,
            z=zeff,
            recon=recon,
            smoothing_radius=options.get("rsd_smoothing_radius", 15.0),
            dtype=options.get("dtype", None),
            mpicomm=getattr(data, "mpicomm", None),
            **rsd_kwargs,
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

        options = params.get("options", {})
        stracer = _simple_tracer(tracer)
        zeff = options.get("zeff", LSS_DEFAULT_ZEFF[stracer])
        bias = options.get("bias", LSS_DEFAULT_BIAS[stracer])
        method = options.get("fnl_method", "data_weights")
        if method != "data_weights":
            raise NotImplementedError(
                'Only fnl_method="data_weights" is currently validated for DESI catalog blinding.'
            )

        crandoms = Catalog.concatenate(randoms)
        fnl_kwargs = dict(options.get("fnl_kwargs", {}))
        recon = fnl_kwargs.pop("recon", options.get("fnl_recon", "IterativeFFTReconstruction"))
        new_weights = _apply_fnl_jax_blinding(
            data["POSITION"],
            data_weights=data["INDWEIGHT"],
            randoms_positions=crandoms["POSITION"],
            randoms_weights=crandoms["INDWEIGHT"],
            cosmo_fid=DESI(),
            fnl_blind=params["fnl_blind"],
            bias=bias,
            z=zeff,
            method=method,
            recon=recon,
            shotnoise_correction=options.get("fnl_shotnoise_correction", True),
            smoothing_radius=options.get("fnl_smoothing_radius", 30.0),
            dtype=options.get("dtype", None),
            mpicomm=getattr(data, "mpicomm", None),
            **fnl_kwargs,
        )
        blind_weight = new_weights / data["INDWEIGHT"]
        new = cls._copy_catalog_with_attrs(data, INDWEIGHT=new_weights, WEIGHT_BLIND=blind_weight)
        return cls._with_attrs(new, params), randoms


def _parse_rows(rows):
    """Parse comma-separated and repeated CLI row selectors."""
    if rows is None:
        return None
    parsed = []
    for value in rows:
        parsed.extend(int(item) for item in str(value).split(",") if item != "")
    return parsed


def collect_argparser():
    """Build the command-line parser for catalog parameter generation."""
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
    """CLI entry point for generating sealed catalog-blinding parameters."""
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
