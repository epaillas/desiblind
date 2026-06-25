"""Validate final clustering statistics through desi-clustering.

This consumes already-produced matched LSS/desiblind clustering catalogs and
feeds them through ``desi-clustering``'s ``compute_stats_from_options`` driver.
The purpose is integration validation of the measurement layer, not another
standalone pycorr/pypower check.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path
import sys

import numpy as np


DEFAULT_INPUT_DIR = Path('/pscratch/sd/u/uendert/desiblind_lss_validation/production-54906853-0')
DEFAULT_DESI_CLUSTERING_REPO = Path('/global/homes/u/uendert/repos/desi/desi-clustering')


def parse_args():
    parser = argparse.ArgumentParser(description='Run desi-clustering final-stat validation on matched LSS/desiblind catalogs.')
    parser.add_argument('--input-dir', default=str(DEFAULT_INPUT_DIR),
                        help='Directory containing lss/desiblind split-GC clustering catalogs.')
    parser.add_argument('--desi-clustering-repo', default=str(DEFAULT_DESI_CLUSTERING_REPO),
                        help='Local desi-clustering checkout to import.')
    parser.add_argument('--output-dir', default=None,
                        help='Scratch output directory. Default: $SCRATCH/desiblind_lss_validation/desi-clustering-stats-<timestamp>-<pid>.')
    parser.add_argument('--tracer', default='LRG')
    parser.add_argument('--regions', nargs='+', default=['NGC', 'SGC'], choices=['NGC', 'SGC'])
    parser.add_argument('--rannum', type=int, default=0,
                        help='Random number to use for the measurement validation.')
    parser.add_argument('--stats', nargs='+', default=['mesh2_spectrum'],
                        choices=['mesh2_spectrum', 'particle2_correlation'],
                        help='desi-clustering statistics to compute.')
    parser.add_argument('--zmin', type=float, default=0.4)
    parser.add_argument('--zmax', type=float, default=1.1)
    parser.add_argument('--weight', default='default-FKP',
                        help='desi-clustering weight option. default-FKP uses WEIGHT * existing WEIGHT_FKP when --use-existing-fkp is set.')
    parser.add_argument('--use-existing-fkp', action='store_true', default=True,
                        help='Use existing WEIGHT_FKP column by passing FKP_P0=None.')
    parser.add_argument('--recompute-fkp-p0', type=float, default=None,
                        help='If set, recompute WEIGHT_FKP with this P0 instead of using existing WEIGHT_FKP.')
    parser.add_argument('--meshsize', type=int, default=256,
                        help='Mesh size for mesh2_spectrum validation. Use a larger value for production-like Pk.')
    parser.add_argument('--boxsize', type=float, default=None,
                        help='Optional cubic boxsize for mesh2_spectrum validation. Useful with meshsize to control k_Nyquist for BAO-range plots.')
    parser.add_argument('--cellsize', type=float, default=None,
                        help='Optional mesh cellsize for mesh2_spectrum validation.')
    parser.add_argument('--k-step', type=float, default=0.005,
                        help='k-bin step for mesh2_spectrum validation.')
    parser.add_argument('--k-min', type=float, default=None,
                        help='Optional minimum k edge for mesh2_spectrum validation.')
    parser.add_argument('--k-max', type=float, default=None,
                        help='Optional maximum k edge for mesh2_spectrum validation; useful for BAO-range Pk plots.')
    parser.add_argument('--ells', nargs='+', type=int, default=[0, 2, 4])
    parser.add_argument('--xi-smax', type=float, default=80.)
    parser.add_argument('--xi-ds', type=float, default=4.)
    parser.add_argument('--xi-nmu', type=int, default=40)
    parser.add_argument('--sample-data-rows', type=int, default=0,
                        help='If >0, copy only the first N rows from each data catalog into staged inputs.')
    parser.add_argument('--sample-random-rows', type=int, default=0,
                        help='If >0, copy only the first N rows from each random catalog into staged inputs.')
    parser.add_argument('--max-allowed-delta', type=float, default=1e-12,
                        help='Fail if any compared numeric dataset differs by more than this absolute tolerance.')
    return parser.parse_args()


def make_output_dir(output_dir=None):
    if output_dir is None:
        base = Path(os.environ.get('SCRATCH', '.')) / 'desiblind_lss_validation'
        stamp = datetime.now(timezone.utc).strftime('desi-clustering-stats-%Y%m%d-%H%M%S')
        output_dir = base / f'{stamp}-{os.getpid()}'
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def copy_or_link(src, dst, max_rows=0):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if max_rows and max_rows > 0:
        import fitsio
        data = fitsio.read(src, rows=np.arange(max_rows))
        fitsio.write(dst, data, clobber=True)
    else:
        os.symlink(src, dst)


def stage_catalogs(input_dir, output_dir, tracer='LRG', regions=('NGC', 'SGC'), rannum=0,
                   sample_data_rows=0, sample_random_rows=0):
    input_dir = Path(input_dir).expanduser().resolve(strict=True)
    staged = {}
    for branch in ['lss', 'desiblind']:
        cat_dir = output_dir / 'catalogs' / branch
        cat_dir.mkdir(parents=True, exist_ok=True)
        staged[branch] = str(cat_dir)
        for region in regions:
            src_data = input_dir / f'{branch}_{region}_clustering.dat.fits'
            dst_data = cat_dir / f'{tracer}_{region}_clustering.dat.fits'
            src_random = input_dir / f'{branch}_{region}_{rannum}_clustering.ran.fits'
            dst_random = cat_dir / f'{tracer}_{region}_{rannum}_clustering.ran.fits'
            for src in [src_data, src_random]:
                if not src.exists():
                    raise FileNotFoundError(src)
            copy_or_link(src_data, dst_data, sample_data_rows)
            copy_or_link(src_random, dst_random, sample_random_rows)
    return staged


def add_desi_clustering_to_path(repo):
    repo = Path(repo).expanduser().resolve(strict=True)
    sys.path.insert(0, str(repo))
    return repo


def make_mask_catalog_for_final_clustering(tools):
    """Return a mask function for already-final clustering catalogs.

    ``desi-clustering``'s default mask routine assumes an ``NX`` column is
    available so it can drop objects with ``NX == 0`` when it is working from
    catalogs where FKP weights may be recomputed. The LSS final clustering
    catalogs used here already contain the final ``WEIGHT_FKP`` column and do
    not carry ``NX``. For this validation, we keep the default redshift and
    sky-region cuts, and apply the ``NX == 0`` cut only if an ``NX`` column is
    actually present. No synthetic catalog columns are added.
    """

    def mask_catalog(catalog, kind, zrange=None, region=None):
        mask = catalog.trues()
        if 'fibered' in kind and 'data' in kind:
            mask = catalog['LOCATION_ASSIGNED']
        if kind in ['data', 'randoms'] and zrange is not None:
            mask_z = (catalog['Z'] >= zrange[0]) & (catalog['Z'] < zrange[1])
            mask &= mask_z
            if 'NX' in catalog:
                mask &= ~(mask_z & (catalog['NX'] == 0))
        if region is not None:
            mask &= tools.select_region(catalog['RA'], catalog['DEC'], region)
        return catalog[mask]

    return mask_catalog


def run_desi_clustering_stats(cat_dir, stats_dir, branch, region, args, tools, compute_stats_from_options):
    get_stats_fn = functools.partial(tools.get_stats_fn, stats_dir=stats_dir, extra=branch)
    mask_catalog = make_mask_catalog_for_final_clustering(tools)
    prepare_catalog = functools.partial(tools.prepare_catalog, mask_catalog=mask_catalog)

    fkp_p0 = args.recompute_fkp_p0
    if args.use_existing_fkp and args.recompute_fkp_p0 is None:
        fkp_p0 = None

    edges = {'step': args.k_step}
    if args.k_min is not None:
        edges['min'] = args.k_min
    if args.k_max is not None:
        edges['max'] = args.k_max

    mesh_options = {
        'mattrs': {'meshsize': args.meshsize},
        'edges': edges,
        'ells': tuple(args.ells),
        'norm': {'cellsize': 10.},
        'cut': None,
        'auw': None,
    }
    if args.boxsize is not None:
        mesh_options['mattrs']['boxsize'] = args.boxsize
    if args.cellsize is not None:
        mesh_options['mattrs']['cellsize'] = args.cellsize

    xi_edges_s = np.arange(0., args.xi_smax + 0.5 * args.xi_ds, args.xi_ds)
    xi_edges_mu = np.linspace(-1., 1., args.xi_nmu + 1)
    correlation_options = {
        'battrs': {'s': xi_edges_s, 'mu': (xi_edges_mu, 'midpoint')},
        'cut': None,
        'auw': None,
        'split_randoms': False,
    }

    options = {
        'catalog': {
            'tracer': args.tracer,
            'version': None,
            'cat_dir': str(cat_dir),
            'region': region,
            'zrange': [(args.zmin, args.zmax)],
            'weight': args.weight,
            'nran': [args.rannum],
            'ext': 'fits',
            'FKP_P0': fkp_p0,
        },
        'mesh2_spectrum': mesh_options,
        'particle2_correlation': correlation_options,
    }
    cache = {}
    compute_stats_from_options(args.stats, analysis='full_shape', get_stats_fn=get_stats_fn,
                               prepare_catalog=prepare_catalog, mask_catalog=mask_catalog,
                               cache=cache, **options)


def list_stat_files(stats_dir):
    return sorted(Path(stats_dir).rglob('*.h5'))


def compare_hdf5_numeric(lhs, rhs):
    import h5py
    out = {'max_abs_delta': 0.0, 'datasets': {}}
    with h5py.File(lhs, 'r') as fl, h5py.File(rhs, 'r') as fr:
        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset):
                return
            if name not in fr:
                raise KeyError(f'{name} missing from {rhs}')
            a = obj[()]
            b = fr[name][()]
            if getattr(a, 'shape', None) != getattr(b, 'shape', None):
                raise ValueError(f'shape mismatch for {name}: {getattr(a, "shape", None)} != {getattr(b, "shape", None)}')
            if np.issubdtype(np.asarray(a).dtype, np.number):
                delta = float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) if np.size(a) else 0.0
                out['datasets'][name] = delta
                out['max_abs_delta'] = max(out['max_abs_delta'], delta)
            else:
                equal = np.array_equal(a, b)
                out['datasets'][name] = 0.0 if equal else 'non_numeric_mismatch'
                if not equal:
                    raise ValueError(f'non-numeric mismatch for {name}')
        fl.visititems(visit)
    return out


def compare_outputs(output_dir):
    comparisons = {}
    for region_dir in sorted((output_dir / 'stats').glob('*')):
        if not region_dir.is_dir():
            continue
        region = region_dir.name
        lss_files = list_stat_files(region_dir / 'lss')
        desi_files = list_stat_files(region_dir / 'desiblind')
        lss_by_name = {fn.name: fn for fn in lss_files}
        desi_by_name = {fn.name: fn for fn in desi_files}
        # The branch name is encoded through stats_extra, so normalize names.
        normalized_lss = {name.replace('_lss.h5', '.h5'): fn for name, fn in lss_by_name.items()}
        normalized_desi = {name.replace('_desiblind.h5', '.h5'): fn for name, fn in desi_by_name.items()}
        if set(normalized_lss) != set(normalized_desi):
            raise ValueError(f'Output file mismatch for {region}: {set(normalized_lss)} != {set(normalized_desi)}')
        comparisons[region] = {}
        for name in sorted(normalized_lss):
            comparisons[region][name] = compare_hdf5_numeric(normalized_lss[name], normalized_desi[name])
    return comparisons


def describe_input_files(input_dir, tracer='LRG', regions=('NGC', 'SGC'), rannum=0):
    input_dir = Path(input_dir).expanduser().resolve(strict=True)
    out = {}
    for branch in ['lss', 'desiblind']:
        out[branch] = {}
        for region in regions:
            out[branch][region] = {
                'data': str(input_dir / f'{branch}_{region}_clustering.dat.fits'),
                'randoms': str(input_dir / f'{branch}_{region}_{rannum}_clustering.ran.fits'),
            }
    return out



def main():
    args = parse_args()
    output_dir = make_output_dir(args.output_dir)
    add_desi_clustering_to_path(args.desi_clustering_repo)

    from clustering_statistics import tools
    from clustering_statistics.compute_stats import compute_stats_from_options
    from clustering_statistics.tools import setup_logging

    setup_logging()
    staged = stage_catalogs(args.input_dir, output_dir, tracer=args.tracer, regions=args.regions,
                            rannum=args.rannum, sample_data_rows=args.sample_data_rows,
                            sample_random_rows=args.sample_random_rows)

    for region in args.regions:
        for branch, cat_dir in staged.items():
            stats_dir = output_dir / 'stats' / region / branch
            run_desi_clustering_stats(cat_dir, stats_dir, branch, region, args, tools, compute_stats_from_options)

    comparisons = compare_outputs(output_dir)
    max_delta = max((comp['max_abs_delta'] for files in comparisons.values() for comp in files.values()), default=0.0)
    if max_delta > args.max_allowed_delta:
        raise AssertionError(f'Max statistic delta {max_delta} exceeds tolerance {args.max_allowed_delta}')
    summary = {
        'input_dir': str(Path(args.input_dir).expanduser().resolve(strict=True)),
        'desi_clustering_repo': str(Path(args.desi_clustering_repo).expanduser().resolve(strict=True)),
        'output_dir': str(output_dir),
        'tracer': args.tracer,
        'regions': args.regions,
        'rannum': args.rannum,
        'stats': args.stats,
        'zrange': [args.zmin, args.zmax],
        'weight': args.weight,
        'sample_data_rows': args.sample_data_rows,
        'sample_random_rows': args.sample_random_rows,
        'inputs': describe_input_files(args.input_dir, tracer=args.tracer, regions=args.regions, rannum=args.rannum),
        'catalog_dirs': staged,
        'comparisons': comparisons,
        'max_abs_delta': max_delta,
        'max_allowed_delta': args.max_allowed_delta,
    }
    summary_fn = output_dir / 'summary.json'
    summary_fn.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print('catalog_bao_desi_clustering_stats_validation=PASS')
    print(f'output_dir={output_dir}')
    print(f'summary={summary_fn}')
    print(f'max_abs_delta={max_delta}')
    print(f'max_allowed_delta={args.max_allowed_delta}')
    for region, files in comparisons.items():
        for name, comp in files.items():
            print(f'{region} {name} max_abs_delta={comp["max_abs_delta"]}')


if __name__ == '__main__':
    main()
