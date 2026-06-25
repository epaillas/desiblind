"""Validate LSS Galactic-cap splitting after the BAO/AP saved-catalog pipeline.

This script runs the LSS-equivalent BAO/AP chain through saved full catalogs,
``mkclusdat``, ``mkclusran``, and then the same Galactic-cap split used at the
end of ``LSS/scripts/main/apply_blinding_main_fromfile_fcomp.py``. It compares
LSS and desiblind branches for the split clustering data and random catalogs.

It is still a scratch/sample validation, not a production-scale full-random run.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import fitsio
from astropy.table import Table

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import validate_catalog_bao_lss_mkclusdat as mkclusdat_validation
import validate_catalog_bao_lss_mkclusran as mkclusran_validation
import validate_catalog_bao_lss_saved_catalog as saved


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate LSS GC split after LSS-style BAO/AP saved-catalog, mkclusdat, and mkclusran steps.'
    )
    parser.add_argument('--lss-repo', default=os.environ.get('LSS_REPO', saved.DEFAULT_LSS_REPO))
    parser.add_argument('--output-dir', default=None,
                        help='Directory for scratch outputs. Default: $SCRATCH/desiblind_lss_validation/split-gc-<timestamp>-<pid>.')
    parser.add_argument('--real-catalog', default=saved.DEFAULT_REAL_CATALOG)
    parser.add_argument('--random-catalog', default=saved.DEFAULT_RANDOM_CATALOG,
                        help='Full random catalog used as the mkclusran input sample.')
    parser.add_argument('--nz-random-catalog', default=None,
                        help='Random catalog used for LSS mknz_full area normalization. Defaults to --random-catalog. In the LSS BAO script this is typically the _1_full random even when mkclusran processes another random number.')
    parser.add_argument('--nrows', type=int, default=20000)
    parser.add_argument('--random-nrows', type=int, default=50000)
    parser.add_argument('--rannum', type=int, default=1)
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
    parser.add_argument('--compmd', choices=['dat', 'ran'], default='ran')
    parser.add_argument('--dchi2', type=float, default=None)
    parser.add_argument('--use-map-veto', default='_HPmapcut')
    return parser.parse_args()


def make_output_dir(output_dir=None):
    if output_dir is None:
        base = Path(os.environ.get('SCRATCH', '.')) / 'desiblind_lss_validation'
        stamp = datetime.now(timezone.utc).strftime('split-gc-%Y%m%d-%H%M%S')
        output_dir = base / f'{stamp}-{os.getpid()}'
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def split_gc_file(common, input_fn, output_prefix, app):
    """Mirror the splitGC helper in apply_blinding_main_fromfile_fcomp.py."""
    table = Table(fitsio.read(str(input_fn), ext='LSS'))
    mask_ngc = common.splitGC(table)
    output_ngc = Path(f'{output_prefix}NGC_{app}')
    output_sgc = Path(f'{output_prefix}SGC_{app}')
    common.write_LSS_scratchcp(table[mask_ngc], str(output_ngc))
    common.write_LSS_scratchcp(table[~mask_ngc], str(output_sgc))
    return output_ngc, output_sgc


def compare_pair(lss_fn, desiblind_fn):
    return mkclusdat_validation.compare_tables(lss_fn, desiblind_fn)


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
    random_prefix, random_sample_fn, random_rows = mkclusran_validation.write_random_sample(
        common, args.random_catalog, output_dir, args.rannum, args.use_map_veto, args.random_nrows
    )
    sample, sample_fn = mkclusran_validation.read_balanced_sample(args.real_catalog, args.nrows, args.input_zcol)
    nz_random_catalog = args.nz_random_catalog or args.random_catalog
    random_fn_for_nz = Path(nz_random_catalog).expanduser().resolve(strict=False)
    if not random_fn_for_nz.exists():
        raise FileNotFoundError(random_fn_for_nz)

    lss_result = saved.run_saved_workflow(
        'lss', sample, common, lss_blind, output_dir, random_fn_for_nz, params, args.tracer_name,
        tracer_type, dz, zmin, zmax, args.randens, p0, args.input_zcol, args.output_zcol,
        args.clip_min, args.clip_max, wsyscol=args.wsyscol, skip_add_fkp=args.skip_add_fkp,
    )
    desiblind_result = saved.run_saved_workflow(
        'desiblind', sample, common, lss_blind, output_dir, random_fn_for_nz, params, args.tracer_name,
        tracer_type, dz, zmin, zmax, args.randens, p0, args.input_zcol, args.output_zcol,
        args.clip_min, args.clip_max, wsyscol=args.wsyscol, skip_add_fkp=args.skip_add_fkp,
    )
    saved_comparison = saved.compare_saved_catalogs(lss_result, desiblind_result, output_zcol=args.output_zcol)

    dchi2 = args.dchi2 if args.dchi2 is not None else mkclusdat_validation.get_default_dchi2(tracer_type)
    lss_clus_fn = mkclusdat_validation.run_mkclusdat(ct, output_dir / 'lss', tracer_type, zmin, zmax, args.compmd, dchi2)
    desiblind_clus_fn = mkclusdat_validation.run_mkclusdat(ct, output_dir / 'desiblind', tracer_type, zmin, zmax, args.compmd, dchi2)
    clustering_data_comparison = mkclusdat_validation.compare_tables(lss_clus_fn, desiblind_clus_fn)

    lss_random_fn = mkclusran_validation.run_mkclusran(ct, random_prefix, output_dir / 'lss_', args.rannum,
                                                       lss_clus_fn, args.compmd, tracer_type, args.use_map_veto)
    desiblind_random_fn = mkclusran_validation.run_mkclusran(ct, random_prefix, output_dir / 'desiblind_', args.rannum,
                                                             desiblind_clus_fn, args.compmd, tracer_type, args.use_map_veto)
    clustering_random_comparison = mkclusdat_validation.compare_tables(lss_random_fn, desiblind_random_fn)

    data_app = 'clustering.dat.fits'
    random_app = f'{args.rannum}_clustering.ran.fits'
    lss_data_ngc, lss_data_sgc = split_gc_file(common, lss_clus_fn, output_dir / 'lss_', data_app)
    desi_data_ngc, desi_data_sgc = split_gc_file(common, desiblind_clus_fn, output_dir / 'desiblind_', data_app)
    lss_ran_ngc, lss_ran_sgc = split_gc_file(common, lss_random_fn, output_dir / 'lss_', random_app)
    desi_ran_ngc, desi_ran_sgc = split_gc_file(common, desiblind_random_fn, output_dir / 'desiblind_', random_app)

    split_comparison = {
        'data_NGC': compare_pair(lss_data_ngc, desi_data_ngc),
        'data_SGC': compare_pair(lss_data_sgc, desi_data_sgc),
        'random_NGC': compare_pair(lss_ran_ngc, desi_ran_ngc),
        'random_SGC': compare_pair(lss_ran_sgc, desi_ran_sgc),
    }

    summary = {
        'parameters': params,
        'tracer_name': args.tracer_name,
        'tracer_type': tracer_type,
        'lss_py': str(lss_py),
        'real_catalog': str(sample_fn),
        'random_catalog_for_nz': str(random_fn_for_nz),
        'random_catalog_for_mkclusran': str(Path(args.random_catalog).expanduser().resolve(strict=False)),
        'random_sample_file': str(random_sample_fn),
        'random_rows': random_rows,
        'output_dir': str(output_dir),
        'nrows': args.nrows,
        'zrange': [zmin, zmax],
        'dz': dz,
        'p0': p0,
        'dchi2': dchi2,
        'compmd': args.compmd,
        'rannum': args.rannum,
        'saved_catalog_comparison': saved_comparison,
        'clustering_data_comparison': clustering_data_comparison,
        'clustering_random_comparison': clustering_random_comparison,
        'split_gc_comparison': split_comparison,
        'files': {
            'lss_full': lss_result['saved_full_fn'],
            'desiblind_full': desiblind_result['saved_full_fn'],
            'lss_clustering_data': str(lss_clus_fn),
            'desiblind_clustering_data': str(desiblind_clus_fn),
            'lss_clustering_random': str(lss_random_fn),
            'desiblind_clustering_random': str(desiblind_random_fn),
            'lss_data_NGC': str(lss_data_ngc),
            'desiblind_data_NGC': str(desi_data_ngc),
            'lss_data_SGC': str(lss_data_sgc),
            'desiblind_data_SGC': str(desi_data_sgc),
            'lss_random_NGC': str(lss_ran_ngc),
            'desiblind_random_NGC': str(desi_ran_ngc),
            'lss_random_SGC': str(lss_ran_sgc),
            'desiblind_random_SGC': str(desi_ran_sgc),
        },
    }
    summary_fn = output_dir / 'summary.json'
    summary_fn.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print('catalog_bao_lss_split_gc_validation=PASS')
    print(f'output_dir={output_dir}')
    print(f'summary={summary_fn}')
    for name, comparison in split_comparison.items():
        print(
            f"{name}: rows={comparison['rows']} "
            f"max_abs_delta_Z={comparison.get('max_abs_delta_Z', 'NA')} "
            f"max_abs_delta_WEIGHT={comparison.get('max_abs_delta_WEIGHT', 'NA')} "
            f"max_abs_delta_WEIGHT_SYS={comparison.get('max_abs_delta_WEIGHT_SYS', 'NA')} "
            f"max_abs_delta_WEIGHT_FKP={comparison.get('max_abs_delta_WEIGHT_FKP', 'NA')}"
        )


if __name__ == '__main__':
    main()
