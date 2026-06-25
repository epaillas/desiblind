"""Validate LSS ``mkclusdat`` after saved full-catalog BAO/AP blinding.

This validation extends ``validate_catalog_bao_lss_saved_catalog.py`` by taking
both saved full-catalog outputs and running the actual LSS
``LSS.main.cattools.mkclusdat`` step on each branch. It checks that replacing
only the LSS BAO/AP redshift shifter with ``CatalogBAOBlinder`` gives identical
clustering data catalogs.

It does not run ``mkclusran``/random resampling; that is the next validation
layer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import fitsio
import numpy as np
from astropy.table import Table

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import validate_catalog_bao_lss_saved_catalog as saved


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate LSS mkclusdat after LSS-style saved full-catalog BAO/AP blinding.'
    )
    parser.add_argument('--lss-repo', default=os.environ.get('LSS_REPO', saved.DEFAULT_LSS_REPO))
    parser.add_argument('--output-dir', default=None,
                        help='Directory for scratch outputs. Default: $SCRATCH/desiblind_lss_validation/mkclusdat-<timestamp>-<pid>.')
    parser.add_argument('--real-catalog', default=saved.DEFAULT_REAL_CATALOG)
    parser.add_argument('--random-catalog', default=saved.DEFAULT_RANDOM_CATALOG)
    parser.add_argument('--nrows', type=int, default=20000)
    parser.add_argument('--w0', type=float, default=saved.DEFAULT_PARAMETERS['w0'])
    parser.add_argument('--wa', type=float, default=saved.DEFAULT_PARAMETERS['wa'])
    parser.add_argument('--tracer-name', default='LRG3')
    parser.add_argument('--tracer-type', default='LRG')
    parser.add_argument('--input-zcol', default='Z_not4clus')
    parser.add_argument('--output-zcol', default='Z')
    parser.add_argument('--zmin', type=float, default=None)
    parser.add_argument('--zmax', type=float, default=None)
    parser.add_argument('--dz', type=float, default=None)
    parser.add_argument('--randens', type=float, default=2500.)
    parser.add_argument('--p0', type=float, default=None)
    parser.add_argument('--clip-min', type=float, default=0.01)
    parser.add_argument('--clip-max', type=float, default=3.6)
    parser.add_argument('--wsyscol', default=None)
    parser.add_argument('--skip-add-fkp', action='store_true')
    parser.add_argument('--compmd', choices=['dat', 'ran'], default='ran',
                        help='Completeness mode passed to LSS mkclusdat. LSS BAO script default is ran.')
    parser.add_argument('--dchi2', type=float, default=None,
                        help='dchi2 passed to LSS mkclusdat. Default: LSS.globals.main(...).dchi2 when available, else mkclusdat default behavior.')
    return parser.parse_args()


def make_output_dir(output_dir=None):
    if output_dir is None:
        base = Path(os.environ.get('SCRATCH', '.')) / 'desiblind_lss_validation'
        stamp = datetime.now(timezone.utc).strftime('mkclusdat-%Y%m%d-%H%M%S')
        output_dir = base / f'{stamp}-{os.getpid()}'
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def get_default_dchi2(tracer_type):
    try:
        from LSS.globals import main as lss_main
        return lss_main(tracer_type, survey='Y1').dchi2
    except Exception:
        return None


def run_mkclusdat(ct, prefix, tracer_type, zmin, zmax, compmd, dchi2):
    kwargs = dict(tp=tracer_type, zmin=zmin, zmax=zmax, compmd=compmd, splitNS='n')
    if dchi2 is not None:
        kwargs['dchi2'] = dchi2
    ct.mkclusdat(str(prefix), **kwargs)
    output_fn = Path(f'{prefix}_clustering.dat.fits')
    if not output_fn.exists():
        raise FileNotFoundError(output_fn)
    return output_fn


def compare_tables(lss_fn, desiblind_fn):
    lss = Table(fitsio.read(str(lss_fn), ext='LSS'))
    desi = Table(fitsio.read(str(desiblind_fn), ext='LSS'))
    if len(lss) != len(desi):
        raise AssertionError(f'row count differs: {len(lss)} != {len(desi)}')
    if set(lss.colnames) != set(desi.colnames):
        raise AssertionError(f'column sets differ: {set(lss.colnames) ^ set(desi.colnames)}')

    deltas = {}
    for column in lss.colnames:
        left = np.asarray(lss[column])
        right = np.asarray(desi[column])
        if np.issubdtype(left.dtype, np.number):
            np.testing.assert_allclose(right, left, rtol=0, atol=1e-12, err_msg=column)
            deltas[column] = float(np.nanmax(np.abs(right - left))) if left.size else 0.0
        else:
            if not np.array_equal(right, left):
                raise AssertionError(f'{column} differs')
    return {
        'rows': int(len(lss)),
        'columns': list(lss.colnames),
        'column_max_abs_delta': deltas,
        'max_abs_delta_Z': deltas.get('Z'),
        'max_abs_delta_WEIGHT': deltas.get('WEIGHT'),
        'max_abs_delta_WEIGHT_SYS': deltas.get('WEIGHT_SYS'),
        'max_abs_delta_WEIGHT_FKP': deltas.get('WEIGHT_FKP'),
    }


def main():
    args = parse_args()
    if args.output_zcol != 'Z':
        raise ValueError('LSS.blinding_tools.apply_zshift_DE always writes to Z; keep --output-zcol=Z.')

    tracer_type = saved.simple_tracer(args.tracer_type)
    zmin, zmax = (args.zmin, args.zmax)
    if zmin is None or zmax is None:
        default_zmin, default_zmax = saved.ZRANGE_BY_TRACER[tracer_type]
        zmin = default_zmin if zmin is None else zmin
        zmax = default_zmax if zmax is None else zmax
    dz = args.dz if args.dz is not None else saved.DZ_BY_TRACER.get(tracer_type, 0.01)
    p0 = args.p0 if args.p0 is not None else saved.P0_BY_TRACER[tracer_type]

    params = {'w0': args.w0, 'wa': args.wa}
    lss_blind, common, lss_py = saved.add_lss_to_path(args.lss_repo)
    import LSS.main.cattools as ct

    output_dir = make_output_dir(args.output_dir)
    random_fn = Path(args.random_catalog).expanduser().resolve(strict=False)
    if not random_fn.exists():
        raise FileNotFoundError(random_fn)
    sample, sample_fn = saved.read_sample(args.real_catalog, args.nrows, args.input_zcol)

    lss_result = saved.run_saved_workflow(
        'lss', sample, common, lss_blind, output_dir, random_fn, params, args.tracer_name,
        tracer_type, dz, zmin, zmax, args.randens, p0, args.input_zcol, args.output_zcol,
        args.clip_min, args.clip_max, wsyscol=args.wsyscol, skip_add_fkp=args.skip_add_fkp,
    )
    desiblind_result = saved.run_saved_workflow(
        'desiblind', sample, common, lss_blind, output_dir, random_fn, params, args.tracer_name,
        tracer_type, dz, zmin, zmax, args.randens, p0, args.input_zcol, args.output_zcol,
        args.clip_min, args.clip_max, wsyscol=args.wsyscol, skip_add_fkp=args.skip_add_fkp,
    )
    saved_comparison = saved.compare_saved_catalogs(lss_result, desiblind_result, output_zcol=args.output_zcol)

    dchi2 = args.dchi2 if args.dchi2 is not None else get_default_dchi2(tracer_type)
    lss_clus_fn = run_mkclusdat(ct, output_dir / 'lss', tracer_type, zmin, zmax, args.compmd, dchi2)
    desiblind_clus_fn = run_mkclusdat(ct, output_dir / 'desiblind', tracer_type, zmin, zmax, args.compmd, dchi2)
    clustering_comparison = compare_tables(lss_clus_fn, desiblind_clus_fn)

    summary = {
        'parameters': params,
        'tracer_name': args.tracer_name,
        'tracer_type': tracer_type,
        'lss_py': str(lss_py),
        'real_catalog': str(sample_fn),
        'random_catalog': str(random_fn),
        'output_dir': str(output_dir),
        'nrows': args.nrows,
        'zrange': [zmin, zmax],
        'dz': dz,
        'p0': p0,
        'dchi2': dchi2,
        'compmd': args.compmd,
        'saved_catalog_comparison': saved_comparison,
        'clustering_comparison': clustering_comparison,
        'files': {
            'lss_full': lss_result['saved_full_fn'],
            'desiblind_full': desiblind_result['saved_full_fn'],
            'lss_clustering': str(lss_clus_fn),
            'desiblind_clustering': str(desiblind_clus_fn),
        },
    }
    summary_fn = output_dir / 'summary.json'
    summary_fn.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print('catalog_bao_lss_mkclusdat_validation=PASS')
    print(f'output_dir={output_dir}')
    print(f'summary={summary_fn}')
    print(
        f"rows={clustering_comparison['rows']} "
        f"max_abs_delta_Z={clustering_comparison.get('max_abs_delta_Z', 'NA')} "
        f"max_abs_delta_WEIGHT={clustering_comparison.get('max_abs_delta_WEIGHT', 'NA')} "
        f"max_abs_delta_WEIGHT_SYS={clustering_comparison.get('max_abs_delta_WEIGHT_SYS', 'NA')} "
        f"max_abs_delta_WEIGHT_FKP={clustering_comparison.get('max_abs_delta_WEIGHT_FKP', 'NA')}"
    )


if __name__ == '__main__':
    main()
