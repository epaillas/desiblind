"""Validate desi-clustering catalog RSD driver reconstruction backends.

This validation consumes small real LSS/pyrecon RSD reference products and runs
``desi-clustering``'s saved-catalog blinding driver with computed realspace
catalogs. It is complementary to the LSS benchmark validations:

- ``pyrecon`` backend: direct pyrecon from desi-clustering, no LSS import;
  expected to reproduce the LSS/pyrecon reference up to numerical precision.
- ``jaxrecon`` backend: JAX-native speed/on-the-fly candidate. The driver
  matches pyrecon mesh/threshold conventions before running JAX reconstruction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from pathlib import Path
import sys
import time

import fitsio
import numpy as np


DEFAULT_DESI_CLUSTERING_REPO = Path('/global/homes/u/uendert/repos/desi/desi-clustering')
DEFAULT_REFERENCE_DIRS = {
    'NGC': Path('/pscratch/sd/u/uendert/desiblind_lss_validation/rsd-recon-shift-20260626-201203-1234847'),
    'SGC': Path('/pscratch/sd/u/uendert/desiblind_lss_validation/rsd-recon-shift-20260626-201300-1235312'),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate desi-clustering catalog RSD driver reconstruction backends against LSS/pyrecon reference products.'
    )
    parser.add_argument('--desi-clustering-repo', default=str(DEFAULT_DESI_CLUSTERING_REPO))
    parser.add_argument('--output-dir', default=None,
                        help='Scratch output directory. Default: $SCRATCH/desiblind_lss_validation/rsd-desi-clustering-driver-<timestamp>-<pid>.')
    parser.add_argument('--backends', nargs='+', default=['pyrecon', 'jaxrecon'], choices=['pyrecon', 'jaxrecon'])
    parser.add_argument('--regions', nargs='+', default=['NGC', 'SGC'], choices=['NGC', 'SGC'])
    parser.add_argument('--ngc-reference-dir', default=str(DEFAULT_REFERENCE_DIRS['NGC']))
    parser.add_argument('--sgc-reference-dir', default=str(DEFAULT_REFERENCE_DIRS['SGC']))
    parser.add_argument('--tracer-name', default='LRG3')
    parser.add_argument('--bias', type=float, default=2.0)
    parser.add_argument('--fiducial-f', type=float, default=0.8)
    parser.add_argument('--fgrowth-blind', type=float, default=0.8800000000000001)
    parser.add_argument('--recon-method', choices=['iterative_fft', 'multigrid'], default='iterative_fft')
    parser.add_argument('--recon-boxsize', type=float, default=6000.)
    parser.add_argument('--recon-meshsize', type=int, default=64)
    parser.add_argument('--recon-cellsize', type=float, default=7.)
    parser.add_argument('--recon-smoothing-radius', type=float, default=15.)
    parser.add_argument('--recon-threshold-randoms', type=float, default=0.01)
    parser.add_argument('--recon-threshold-randoms-method', choices=['mean', 'noise'], default='mean')
    parser.add_argument('--recon-growth-rate', type=float, default=0.8)
    parser.add_argument('--recon-nthreads', type=int, default=16)
    parser.add_argument('--pyrecon-max-allowed-blinded-delta', type=float, default=1e-7,
                        help='Hard tolerance for final blinded-Z agreement for the pyrecon backend.')
    parser.add_argument('--jaxrecon-max-allowed-blinded-delta', type=float, default=1e-7,
                        help='Hard tolerance for final blinded-Z agreement for the jaxrecon backend.')
    return parser.parse_args()


def make_output_dir(output_dir=None):
    if output_dir is None:
        base = Path(os.environ.get('SCRATCH', '.')) / 'desiblind_lss_validation'
        stamp = datetime.now(timezone.utc).strftime('rsd-desi-clustering-driver-%Y%m%d-%H%M%S')
        output_dir = base / f'{stamp}-{os.getpid()}'
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def add_desi_clustering_to_path(repo):
    repo = Path(repo).expanduser().resolve(strict=True)
    sys.path.insert(0, str(repo))
    return repo


def reference_dir(args, region):
    value = args.ngc_reference_dir if region == 'NGC' else args.sgc_reference_dir
    return Path(value).expanduser().resolve(strict=True)


def reference_files(ref_dir, region):
    return {
        'observed': ref_dir / f'LRG_{region}_clustering.dat.fits',
        'random': ref_dir / f'LRG_{region}_0_clustering.ran.fits',
        'realspace': ref_dir / f'LRG_{region}_clustering.IFFTrsd.dat.fits',
        'lss_blinded': ref_dir / f'LRG_{region}_clustering.lss_rsd_blinded.dat.fits',
    }


def stats(delta):
    delta = np.asarray(delta)
    abs_delta = np.abs(delta)
    return {
        'max_abs': float(np.max(abs_delta)),
        'mean_abs': float(np.mean(abs_delta)),
        'rms': float(np.sqrt(np.mean(delta**2))),
        'p50_abs': float(np.percentile(abs_delta, 50)),
        'p95_abs': float(np.percentile(abs_delta, 95)),
        'p99_abs': float(np.percentile(abs_delta, 99)),
    }


def run_backend(args, output_dir, backend, region, driver):
    ref = reference_dir(args, region)
    files = reference_files(ref, region)
    for name, filename in files.items():
        if not filename.exists():
            raise FileNotFoundError(f'{name}: {filename}')

    realspace_out = output_dir / backend / f'LRG_{region}_clustering.{backend}.IFFTrsd.dat.fits'
    blinded_out = output_dir / backend / f'LRG_{region}_clustering.{backend}.rsd_blinded.dat.fits'
    summary_out = output_dir / backend / f'summary_{region}.json'
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    namespace = argparse.Namespace(
        input_catalog=str(files['observed']),
        output_catalog=str(blinded_out),
        realspace_catalog=None,
        random_catalog=str(files['random']),
        run_pyrecon=backend == 'pyrecon',
        run_jaxrecon=backend == 'jaxrecon',
        save_realspace_catalog=str(realspace_out),
        fits_ext='LSS',
        modes=['rsd'],
        tracer_name=args.tracer_name,
        input_zcol='Z',
        output_zcol='Z',
        realspace_zcol='Z',
        w0=None,
        wa=None,
        zeff=None,
        bias=args.bias,
        fiducial_f=args.fiducial_f,
        recon_bias=None,
        recon_method=args.recon_method,
        recon_smoothing_radius=args.recon_smoothing_radius,
        recon_growth_rate=args.recon_growth_rate,
        recon_cellsize=args.recon_cellsize,
        recon_meshsize=args.recon_meshsize,
        recon_boxsize=args.recon_boxsize,
        recon_boxcenter=None,
        recon_threshold_randoms=args.recon_threshold_randoms,
        recon_threshold_randoms_method=args.recon_threshold_randoms_method,
        recon_nthreads=args.recon_nthreads,
        recon_weight_col='WEIGHT',
        fgrowth_blind=args.fgrowth_blind,
        max_df_fraction=0.1,
        summary_file=str(summary_out),
        clobber=True,
    )

    t0 = time.time()
    driver_summary = driver.run_from_args(namespace)
    runtime = time.time() - t0

    ref_real = fitsio.read(str(files['realspace']), ext='LSS')
    ref_blind = fitsio.read(str(files['lss_blinded']), ext='LSS')
    trial_real = fitsio.read(str(realspace_out), ext='LSS')
    trial_blind = fitsio.read(str(blinded_out), ext='LSS')

    comparison = {
        'backend': backend,
        'region': region,
        'rows': int(len(trial_blind)),
        'runtime_seconds': runtime,
        'reference_dir': str(ref),
        'files': {name: str(filename) for name, filename in files.items()} | {
            'trial_realspace': str(realspace_out),
            'trial_blinded': str(blinded_out),
            'driver_summary': str(summary_out),
        },
        'driver_summary': driver_summary,
        'realspace_delta_Z_vs_lss': stats(trial_real['Z'] - ref_real['Z']),
        'blinded_delta_Z_vs_lss': stats(trial_blind['Z'] - ref_blind['Z']),
    }
    compare_out = output_dir / backend / f'compare_{region}.json'
    compare_out.write_text(json.dumps(comparison, indent=2, sort_keys=True))
    comparison['comparison_file'] = str(compare_out)
    return comparison


def tolerance_for(args, backend):
    if backend == 'pyrecon':
        return args.pyrecon_max_allowed_blinded_delta
    return args.jaxrecon_max_allowed_blinded_delta


def main():
    args = parse_args()
    output_dir = make_output_dir(args.output_dir)
    desi_clustering_repo = add_desi_clustering_to_path(args.desi_clustering_repo)

    from clustering_statistics import catalog_blinding_driver as driver

    results = {}
    hard_failures = []
    for backend in args.backends:
        results[backend] = {}
        tol = tolerance_for(args, backend)
        for region in args.regions:
            result = run_backend(args, output_dir, backend, region, driver)
            results[backend][region] = result
            max_delta = result['blinded_delta_Z_vs_lss']['max_abs']
            if tol is not None and max_delta > tol:
                hard_failures.append({
                    'backend': backend,
                    'region': region,
                    'max_abs_delta': max_delta,
                    'tolerance': tol,
                })

    summary = {
        'validation': 'catalog_rsd_desi_clustering_driver',
        'desi_clustering_repo': str(desi_clustering_repo),
        'output_dir': str(output_dir),
        'backends': args.backends,
        'regions': args.regions,
        'reconstruction': {
            'method': args.recon_method,
            'boxsize': args.recon_boxsize,
            'meshsize': args.recon_meshsize,
            'cellsize': args.recon_cellsize,
            'smoothing_radius': args.recon_smoothing_radius,
            'growth_rate': args.recon_growth_rate,
            'threshold_randoms': args.recon_threshold_randoms,
            'threshold_randoms_method': args.recon_threshold_randoms_method,
            'nthreads': args.recon_nthreads,
        },
        'pyrecon_max_allowed_blinded_delta': args.pyrecon_max_allowed_blinded_delta,
        'jaxrecon_max_allowed_blinded_delta': args.jaxrecon_max_allowed_blinded_delta,
        'results': results,
        'hard_failures': hard_failures,
    }
    summary_fn = output_dir / 'summary.json'
    summary_fn.write_text(json.dumps(summary, indent=2, sort_keys=True))

    if hard_failures:
        raise AssertionError(f'desi-clustering RSD driver validation failures: {hard_failures}')

    print('catalog_rsd_desi_clustering_driver_validation=PASS')
    print(f'output_dir={output_dir}')
    print(f'summary={summary_fn}')
    for backend, backend_results in results.items():
        for region, result in backend_results.items():
            print(
                f'{backend} {region}: rows={result["rows"]} '
                f'runtime_seconds={result["runtime_seconds"]:.3f} '
                f'max_abs_blinded_delta={result["blinded_delta_Z_vs_lss"]["max_abs"]}'
            )


if __name__ == '__main__':
    main()
