"""Validate final RSD-blinded clustering statistics through desi-clustering.

This consumes matched LSS/desiblind RSD-blinded clustering catalogs produced by
``validate_catalog_rsd_lss_saved_catalog.py`` and feeds them through
``desi-clustering``'s ``compute_stats_from_options`` driver. The purpose is
measurement-layer integration validation, not a standalone pycorr/pypower check.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import validate_catalog_bao_desi_clustering_stats as stats_validation


DEFAULT_INPUT_DIR = Path('/pscratch/sd/u/uendert/desiblind_lss_validation/rsd-saved-catalog-20260629-160133-759367')
DEFAULT_DESI_CLUSTERING_REPO = stats_validation.DEFAULT_DESI_CLUSTERING_REPO


def parse_args():
    parser = argparse.ArgumentParser(description='Run desi-clustering final-stat validation on matched LSS/desiblind RSD-blinded catalogs.')
    parser.add_argument('--input-dir', default=str(DEFAULT_INPUT_DIR),
                        help='Directory containing lss/desiblind split-GC RSD-blinded clustering catalogs.')
    parser.add_argument('--desi-clustering-repo', default=str(DEFAULT_DESI_CLUSTERING_REPO),
                        help='Local desi-clustering checkout to import.')
    parser.add_argument('--output-dir', default=None,
                        help='Scratch output directory. Default: $SCRATCH/desiblind_lss_validation/rsd-desi-clustering-stats-<timestamp>-<pid>.')
    parser.add_argument('--tracer', default='LRG')
    parser.add_argument('--regions', nargs='+', default=['NGC', 'SGC'], choices=['NGC', 'SGC'])
    parser.add_argument('--rannum', type=int, default=0)
    parser.add_argument('--stats', nargs='+', default=['mesh2_spectrum'],
                        choices=['mesh2_spectrum', 'particle2_correlation'])
    parser.add_argument('--zmin', type=float, default=0.4)
    parser.add_argument('--zmax', type=float, default=1.1)
    parser.add_argument('--weight', default='default-FKP')
    parser.add_argument('--use-existing-fkp', action='store_true', default=True)
    parser.add_argument('--recompute-fkp-p0', type=float, default=None)
    parser.add_argument('--meshsize', type=int, default=256)
    parser.add_argument('--boxsize', type=float, default=None)
    parser.add_argument('--cellsize', type=float, default=None)
    parser.add_argument('--k-step', type=float, default=0.005)
    parser.add_argument('--k-min', type=float, default=None)
    parser.add_argument('--k-max', type=float, default=None)
    parser.add_argument('--ells', nargs='+', type=int, default=[0, 2, 4])
    parser.add_argument('--xi-smax', type=float, default=200.)
    parser.add_argument('--xi-ds', type=float, default=5.)
    parser.add_argument('--xi-nmu', type=int, default=40)
    parser.add_argument('--sample-data-rows', type=int, default=0,
                        help='If >0, copy only the first N rows from each data catalog into staged inputs.')
    parser.add_argument('--sample-random-rows', type=int, default=0,
                        help='If >0, copy only the first N rows from each random catalog into staged inputs.')
    parser.add_argument('--max-allowed-delta', type=float, default=1e-12)
    return parser.parse_args()


def make_output_dir(output_dir=None):
    if output_dir is None:
        base = Path(os.environ.get('SCRATCH', '.')) / 'desiblind_lss_validation'
        stamp = datetime.now(timezone.utc).strftime('rsd-desi-clustering-stats-%Y%m%d-%H%M%S')
        output_dir = base / f'{stamp}-{os.getpid()}'
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def describe_input_files(input_dir, tracer='LRG', regions=('NGC', 'SGC'), rannum=0):
    return stats_validation.describe_input_files(input_dir, tracer=tracer, regions=regions, rannum=rannum)


def main():
    args = parse_args()
    output_dir = make_output_dir(args.output_dir)
    stats_validation.add_desi_clustering_to_path(args.desi_clustering_repo)

    from clustering_statistics import tools
    from clustering_statistics.compute_stats import compute_stats_from_options
    from clustering_statistics.tools import setup_logging

    setup_logging()
    staged = stats_validation.stage_catalogs(
        args.input_dir, output_dir, tracer=args.tracer, regions=args.regions,
        rannum=args.rannum, sample_data_rows=args.sample_data_rows,
        sample_random_rows=args.sample_random_rows,
    )

    for region in args.regions:
        for branch, cat_dir in staged.items():
            stats_dir = output_dir / 'stats' / region / branch
            stats_validation.run_desi_clustering_stats(cat_dir, stats_dir, branch, region, args, tools, compute_stats_from_options)

    comparisons = stats_validation.compare_outputs(output_dir)
    max_delta = max((comp['max_abs_delta'] for files in comparisons.values() for comp in files.values()), default=0.0)
    if max_delta > args.max_allowed_delta:
        raise AssertionError(f'Max statistic delta {max_delta} exceeds tolerance {args.max_allowed_delta}')
    summary = {
        'validation': 'catalog_rsd_desi_clustering_stats',
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

    print('catalog_rsd_desi_clustering_stats_validation=PASS')
    print(f'output_dir={output_dir}')
    print(f'summary={summary_fn}')
    print(f'max_abs_delta={max_delta}')


if __name__ == '__main__':
    main()
