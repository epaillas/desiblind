"""Validate RSD blinding on real LSS catalogs after running reconstruction.

This is the first real-catalog RSD validation rung. It takes an observed
clustering data catalog and matching random catalog, draws deterministic real
subsets, runs the LSS reconstruction helper in ``convention='rsd'`` mode to
produce the reconstructed-realspace data catalog, then compares:

- ``LSS.blinding_tools.apply_zshift_RSD``
- ``desiblind.CatalogRSDBlinder``

for the same observed/reconstructed catalogs and the same derived
``fgrowth_blind``.

The reconstruction step is shared here; this script validates the RSD redshift
shift on a real reconstructed LSS product, not the full production RSD ladder.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
# This validation is CPU-only; avoid noisy JAX CUDA plugin initialization on CPU nodes.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from pathlib import Path
import sys

import fitsio
import numpy as np
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from desiblind import CatalogRSDBlinder


DEFAULT_LSS_REPO = '/global/homes/u/uendert/repos/desi/LSS'
DEFAULT_BASE = '/pscratch/sd/u/uendert/desiblind_lss_validation/production-54906853-0'
DEFAULT_DATA = f'{DEFAULT_BASE}/lss_NGC_clustering.dat.fits'
DEFAULT_RANDOMS = f'{DEFAULT_BASE}/lss_NGC_0_clustering.ran.fits'
DEFAULT_PARAMETERS = {'w0': -0.95, 'wa': 0.10, 'zeff': 0.8, 'bias': 2.0, 'fiducial_f': 0.8}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run LSS reconstruction on a real subset and validate CatalogRSDBlinder against LSS.apply_zshift_RSD.'
    )
    parser.add_argument('--lss-repo', default=os.environ.get('LSS_REPO', DEFAULT_LSS_REPO))
    parser.add_argument('--output-dir', default=None,
                        help='Directory for scratch outputs. Default: $SCRATCH/desiblind_lss_validation/rsd-recon-shift-<timestamp>-<pid>.')
    parser.add_argument('--data-catalog', default=DEFAULT_DATA)
    parser.add_argument('--random-catalog', default=DEFAULT_RANDOMS)
    parser.add_argument('--nrows', type=int, default=5000,
                        help='Number of data rows sampled across the input catalog.')
    parser.add_argument('--random-nrows', type=int, default=25000,
                        help='Number of random rows sampled across the input random catalog.')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--tracer-name', default='LRG3')
    parser.add_argument('--tracer-type', default='LRG')
    parser.add_argument('--region', default='NGC')
    parser.add_argument('--w0', type=float, default=DEFAULT_PARAMETERS['w0'])
    parser.add_argument('--wa', type=float, default=DEFAULT_PARAMETERS['wa'])
    parser.add_argument('--zeff', type=float, default=DEFAULT_PARAMETERS['zeff'])
    parser.add_argument('--bias', type=float, default=DEFAULT_PARAMETERS['bias'])
    parser.add_argument('--fiducial-f', type=float, default=DEFAULT_PARAMETERS['fiducial_f'])
    parser.add_argument('--max-df-fraction', type=float, default=0.1)
    parser.add_argument('--zmin', type=float, default=0.4)
    parser.add_argument('--zmax', type=float, default=1.1)
    parser.add_argument('--reconstruction', choices=['iterative_fft', 'multigrid'], default='iterative_fft')
    parser.add_argument('--nmesh', type=int, default=64)
    parser.add_argument('--boxsize', type=float, default=6000.,
                        help='Cartesian reconstruction box size. A large explicit box is safer for sky-wide validation subsets.')
    parser.add_argument('--cellsize', type=float, default=7.)
    parser.add_argument('--smoothing-radius', type=float, default=15.)
    parser.add_argument('--nthreads', type=int, default=8)
    parser.add_argument('--zcol', default='Z')
    return parser.parse_args()


def add_lss_to_path(lss_repo):
    lss_py = Path(lss_repo).expanduser().resolve(strict=False) / 'py'
    if lss_py.exists():
        sys.path.insert(0, str(lss_py))
    try:
        import LSS.blinding_tools as lss_blind
        import LSS.common_tools as common
        import LSS.recon_tools as rectools
    except ImportError as exc:
        raise RuntimeError(
            f'Could not import LSS modules. Pass --lss-repo or set LSS_REPO. Tried: {lss_py}'
        ) from exc
    return lss_blind, common, rectools, lss_py


def make_output_dir(output_dir=None):
    if output_dir is None:
        base = Path(os.environ.get('SCRATCH', '.')) / 'desiblind_lss_validation'
        stamp = datetime.now(timezone.utc).strftime('rsd-recon-shift-%Y%m%d-%H%M%S')
        output_dir = base / f'{stamp}-{os.getpid()}'
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def table_nrows(filename):
    return int(fitsio.read_header(str(filename), ext='LSS')['NAXIS2'])


def sample_rows(filename, nrows, seed, label):
    filename = Path(filename).expanduser().resolve(strict=False)
    if not filename.exists():
        raise FileNotFoundError(filename)
    total = table_nrows(filename)
    if nrows <= 0 or nrows >= total:
        rows = None
    else:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(total, size=nrows, replace=False))
    table = Table(fitsio.read(str(filename), ext='LSS', rows=rows))
    return table, filename, total, 0 if rows is None else len(rows)


def filter_zrange(table, zmin, zmax, zcol='Z'):
    mask = (np.asarray(table[zcol]) >= zmin) & (np.asarray(table[zcol]) < zmax)
    return table[mask]


def write_lss_table(common, table, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    common.write_LSS(Table(table), str(filename))
    return filename


def read_any_fits_table(filename):
    filename = str(filename)
    try:
        return Table(fitsio.read(filename, ext='LSS'))
    except Exception:
        return Table.read(filename)


def run_reconstruction(rectools, output_dir, data_fn, random_fn, args):
    from LSS.tabulated_cosmo import TabulatedDESI
    from pyrecon import IterativeFFTReconstruction, MultiGridReconstruction, setup_logging

    reconstruction_cls = IterativeFFTReconstruction if args.reconstruction == 'iterative_fft' else MultiGridReconstruction
    setup_logging()
    distance = TabulatedDESI().comoving_radial_distance
    data_rec_fn = output_dir / f'{args.tracer_type}_{args.region}_clustering.IFFTrsd.dat.fits'
    random_rec_fn = output_dir / f'{args.tracer_type}_{args.region}_0_clustering.IFFTrsd.ran.fits'
    rectools.run_reconstruction(
        reconstruction_cls,
        distance,
        str(data_fn),
        str(random_fn),
        str(data_rec_fn),
        str(random_rec_fn),
        f=args.fiducial_f,
        bias=args.bias,
        convention='rsd',
        dtype='f8',
        zlim=(args.zmin, args.zmax),
        boxsize=args.boxsize,
        nmesh=args.nmesh,
        cellsize=args.cellsize,
        smoothing_radius=args.smoothing_radius,
        nthreads=args.nthreads,
    )
    return data_rec_fn


def run_lss_apply_zshift_rsd(lss_blind, data, realspace, output_fn, params, zcol):
    lss_data = data.copy()
    lss_realspace = realspace.copy()
    lss_blind.apply_zshift_RSD(
        lss_data, lss_realspace, str(output_fn),
        fgrowth_fid=params['fiducial_f'], fgrowth_blind=params['fgrowth_blind'], zcol=zcol,
    )
    return read_any_fits_table(output_fn)


def assert_outputs_match(desiblind_output, lss_output, zcol='Z'):
    if len(desiblind_output) != len(lss_output):
        raise AssertionError(f'row count differs: {len(desiblind_output)} != {len(lss_output)}')
    common_columns = [column for column in desiblind_output.colnames if column in lss_output.colnames]
    checked = []
    for column in common_columns:
        if column == zcol:
            continue
        left = np.asarray(desiblind_output[column])
        right = np.asarray(lss_output[column])
        if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
            np.testing.assert_allclose(left, right, rtol=0, atol=0, err_msg=column)
        else:
            np.testing.assert_array_equal(left, right, err_msg=column)
        checked.append(column)
    dz = np.asarray(desiblind_output[zcol]) - np.asarray(lss_output[zcol])
    np.testing.assert_allclose(dz, 0., rtol=0, atol=1e-12, err_msg=zcol)
    return {
        'rows': int(len(lss_output)),
        'checked_non_z_columns': checked,
        f'max_abs_delta_{zcol}': float(np.max(np.abs(dz))) if len(dz) else 0.,
        f'{zcol}_range': [float(np.min(desiblind_output[zcol])), float(np.max(desiblind_output[zcol]))]
            if len(desiblind_output) else None,
    }


def main():
    args = parse_args()
    lss_blind, common, rectools, lss_py = add_lss_to_path(args.lss_repo)
    output_dir = make_output_dir(args.output_dir)

    data, data_input, data_total, data_sampled = sample_rows(args.data_catalog, args.nrows, args.seed, 'data')
    randoms, random_input, random_total, random_sampled = sample_rows(
        args.random_catalog, args.random_nrows, args.seed + 1, 'randoms'
    )
    data = filter_zrange(data, args.zmin, args.zmax, zcol=args.zcol)
    randoms = filter_zrange(randoms, args.zmin, args.zmax, zcol=args.zcol)
    if len(data) == 0 or len(randoms) == 0:
        raise ValueError(f'empty data/random sample after z cut: data={len(data)} randoms={len(randoms)}')

    data_fn = write_lss_table(common, data, output_dir / f'{args.tracer_type}_{args.region}_clustering.dat.fits')
    random_fn = write_lss_table(common, randoms, output_dir / f'{args.tracer_type}_{args.region}_0_clustering.ran.fits')
    data_rec_fn = run_reconstruction(rectools, output_dir, data_fn, random_fn, args)

    observed = read_any_fits_table(data_fn)
    realspace = read_any_fits_table(data_rec_fn)
    if len(observed) != len(realspace):
        raise AssertionError(f'observed/reconstructed row count differs: {len(observed)} != {len(realspace)}')

    params = {
        'w0': args.w0,
        'wa': args.wa,
        'zeff': args.zeff,
        'bias': args.bias,
        'fiducial_f': args.fiducial_f,
        'max_df_fraction': args.max_df_fraction,
    }
    params['fgrowth_blind'] = CatalogRSDBlinder.compute_fgrowth_blind(**params)

    lss_output_fn = output_dir / f'{args.tracer_type}_{args.region}_clustering.lss_rsd_blinded.dat.fits'
    lss_output = run_lss_apply_zshift_rsd(lss_blind, observed, realspace, lss_output_fn, params, args.zcol)
    desiblind_output = CatalogRSDBlinder.apply_blinding(
        args.tracer_name, observed, realspace, parameters=params, zcol=args.zcol, output_zcol=args.zcol
    )
    comparison = assert_outputs_match(desiblind_output, lss_output, zcol=args.zcol)

    summary = {
        'validation': 'catalog_rsd_lss_reconstruction_shift',
        'status': 'PASS',
        'lss_py': str(lss_py),
        'output_dir': str(output_dir),
        'inputs': {
            'data_catalog': str(data_input),
            'random_catalog': str(random_input),
            'data_total_rows': data_total,
            'random_total_rows': random_total,
            'data_sampled_rows': data_sampled,
            'random_sampled_rows': random_sampled,
            'data_rows_after_zcut': int(len(data)),
            'random_rows_after_zcut': int(len(randoms)),
        },
        'parameters': params,
        'reconstruction': {
            'method': args.reconstruction,
            'nmesh': args.nmesh,
            'boxsize': args.boxsize,
            'cellsize': args.cellsize,
            'smoothing_radius': args.smoothing_radius,
            'nthreads': args.nthreads,
            'zrange': [args.zmin, args.zmax],
        },
        'files': {
            'observed_sample': str(data_fn),
            'random_sample': str(random_fn),
            'realspace_reconstruction': str(data_rec_fn),
            'lss_rsd_blinded': str(lss_output_fn),
        },
        'comparison': comparison,
    }
    summary_fn = output_dir / 'summary.json'
    summary_fn.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print('catalog_rsd_lss_reconstruction_shift_validation=PASS')
    print(f'fgrowth_blind={params["fgrowth_blind"]}')
    print(f'output_dir={output_dir}')
    print(f'summary={summary_fn}')
    print(f'rows={comparison["rows"]} max_abs_delta_{args.zcol}={comparison[f"max_abs_delta_{args.zcol}"]}')


if __name__ == '__main__':
    main()
