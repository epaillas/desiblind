"""Validate the LSS-equivalent BAO/AP saved-catalog pipeline for many randoms.

This is the production-oriented companion to
``validate_catalog_bao_lss_split_gc.py``. It builds the LSS and desiblind full
catalog / clustering-data branches once, then loops over one or more random
catalogs with actual LSS ``mkclusran`` and Galactic-cap splitting. This avoids
recomputing the expensive full-data branch for every random file.
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

import validate_catalog_bao_lss_mkclusdat as mkclusdat_validation
import validate_catalog_bao_lss_mkclusran as mkclusran_validation
import validate_catalog_bao_lss_saved_catalog as saved
import validate_catalog_bao_lss_split_gc as split_gc_validation


def parse_rannums(tokens):
    values = []
    for token in tokens:
        for piece in str(token).split(','):
            piece = piece.strip()
            if not piece:
                continue
            if '-' in piece:
                start, stop = piece.split('-', 1)
                values.extend(range(int(start), int(stop) + 1))
            else:
                values.append(int(piece))
    return sorted(dict.fromkeys(values))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate LSS-equivalent BAO/AP saved-catalog pipeline for multiple random files.'
    )
    parser.add_argument('--lss-repo', default=os.environ.get('LSS_REPO', saved.DEFAULT_LSS_REPO))
    parser.add_argument('--output-dir', default=None,
                        help='Directory for scratch outputs. Default: $SCRATCH/desiblind_lss_validation/multi-randoms-<timestamp>-<pid>.')
    parser.add_argument('--version-dir', default='/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1')
    parser.add_argument('--real-catalog', default=None,
                        help='Full data catalog. Default: <version-dir>/<tracer>_full_HPmapcut.dat.fits')
    parser.add_argument('--nz-random-catalog', default=None,
                        help='Random catalog used by LSS mknz_full for area normalization. Default: <version-dir>/<tracer>_1_full_HPmapcut.ran.fits')
    parser.add_argument('--random-catalog-template', default=None,
                        help='Template for mkclusran random inputs. Use {rannum}. Default: <version-dir>/<tracer>_{rannum}_full_HPmapcut.ran.fits')
    parser.add_argument('--rannums', nargs='+', default=['0-17'],
                        help='Random numbers to validate. Accepts space/comma separated values and ranges, e.g. "1-17" or "0 1 2".')
    parser.add_argument('--nrows', type=int, default=0,
                        help='Data rows to read; 0 means full catalog.')
    parser.add_argument('--random-nrows', type=int, default=0,
                        help='Random rows to read for each random; 0 means full random catalog.')
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
    parser.add_argument('--cleanup-random-files', action='store_true',
                        help='After each random passes and summary is updated, remove large per-random FITS outputs.')
    return parser.parse_args()


def make_output_dir(output_dir=None):
    if output_dir is None:
        base = Path(os.environ.get('SCRATCH', '.')) / 'desiblind_lss_validation'
        stamp = datetime.now(timezone.utc).strftime('multi-randoms-%Y%m%d-%H%M%S')
        output_dir = base / f'{stamp}-{os.getpid()}'
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def default_real_catalog(version_dir, tracer_type):
    return Path(version_dir) / f'{tracer_type}_full_HPmapcut.dat.fits'


def default_nz_random_catalog(version_dir, tracer_type):
    return Path(version_dir) / f'{tracer_type}_1_full_HPmapcut.ran.fits'


def default_random_catalog(version_dir, tracer_type, rannum):
    return Path(version_dir) / f'{tracer_type}_{rannum}_full_HPmapcut.ran.fits'


def random_catalog_from_template(template, rannum):
    return Path(str(template).format(rannum=rannum))


def unlink_existing(paths):
    removed = []
    for path in paths:
        path = Path(path)
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def main():
    args = parse_args()
    if args.output_zcol != 'Z':
        raise ValueError('LSS.blinding_tools.apply_zshift_DE always writes to Z; keep --output-zcol=Z.')

    tracer_type = saved.simple_tracer(args.tracer_type)
    version_dir = Path(args.version_dir).expanduser().resolve(strict=False)
    real_catalog = Path(args.real_catalog).expanduser().resolve(strict=False) if args.real_catalog else default_real_catalog(version_dir, tracer_type)
    nz_random_catalog = (Path(args.nz_random_catalog).expanduser().resolve(strict=False)
                         if args.nz_random_catalog else default_nz_random_catalog(version_dir, tracer_type))
    if not real_catalog.exists():
        raise FileNotFoundError(real_catalog)
    if not nz_random_catalog.exists():
        raise FileNotFoundError(nz_random_catalog)

    rannums = parse_rannums(args.rannums)
    if not rannums:
        raise ValueError('No random numbers requested')

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
    sample, sample_fn = mkclusran_validation.read_balanced_sample(real_catalog, args.nrows, args.input_zcol)

    lss_result = saved.run_saved_workflow(
        'lss', sample, common, lss_blind, output_dir, nz_random_catalog, params, args.tracer_name,
        tracer_type, dz, zmin, zmax, args.randens, p0, args.input_zcol, args.output_zcol,
        args.clip_min, args.clip_max, wsyscol=args.wsyscol, skip_add_fkp=args.skip_add_fkp,
    )
    desiblind_result = saved.run_saved_workflow(
        'desiblind', sample, common, lss_blind, output_dir, nz_random_catalog, params, args.tracer_name,
        tracer_type, dz, zmin, zmax, args.randens, p0, args.input_zcol, args.output_zcol,
        args.clip_min, args.clip_max, wsyscol=args.wsyscol, skip_add_fkp=args.skip_add_fkp,
    )
    saved_comparison = saved.compare_saved_catalogs(lss_result, desiblind_result, output_zcol=args.output_zcol)

    dchi2 = args.dchi2 if args.dchi2 is not None else mkclusdat_validation.get_default_dchi2(tracer_type)
    lss_clus_fn = mkclusdat_validation.run_mkclusdat(ct, output_dir / 'lss', tracer_type, zmin, zmax, args.compmd, dchi2)
    desiblind_clus_fn = mkclusdat_validation.run_mkclusdat(ct, output_dir / 'desiblind', tracer_type, zmin, zmax, args.compmd, dchi2)
    clustering_data_comparison = mkclusdat_validation.compare_tables(lss_clus_fn, desiblind_clus_fn)

    summary = {
        'parameters': params,
        'tracer_name': args.tracer_name,
        'tracer_type': tracer_type,
        'lss_py': str(lss_py),
        'real_catalog': str(sample_fn),
        'random_catalog_for_nz': str(nz_random_catalog),
        'output_dir': str(output_dir),
        'nrows': args.nrows,
        'random_nrows': args.random_nrows,
        'rannums': rannums,
        'zrange': [zmin, zmax],
        'dz': dz,
        'p0': p0,
        'dchi2': dchi2,
        'compmd': args.compmd,
        'saved_catalog_comparison': saved_comparison,
        'clustering_data_comparison': clustering_data_comparison,
        'files': {
            'lss_full': lss_result['saved_full_fn'],
            'desiblind_full': desiblind_result['saved_full_fn'],
            'lss_clustering_data': str(lss_clus_fn),
            'desiblind_clustering_data': str(desiblind_clus_fn),
        },
        'randoms': {},
    }
    summary_fn = output_dir / 'summary.json'
    summary_fn.write_text(json.dumps(summary, indent=2, sort_keys=True))

    for rannum in rannums:
        random_catalog = (random_catalog_from_template(args.random_catalog_template, rannum)
                          if args.random_catalog_template else default_random_catalog(version_dir, tracer_type, rannum))
        if not random_catalog.exists():
            raise FileNotFoundError(random_catalog)

        random_prefix, random_sample_fn, random_rows = mkclusran_validation.write_random_sample(
            common, random_catalog, output_dir, rannum, args.use_map_veto, args.random_nrows
        )
        lss_random_fn = mkclusran_validation.run_mkclusran(
            ct, random_prefix, output_dir / 'lss_', rannum, lss_clus_fn, args.compmd, tracer_type, args.use_map_veto
        )
        desiblind_random_fn = mkclusran_validation.run_mkclusran(
            ct, random_prefix, output_dir / 'desiblind_', rannum, desiblind_clus_fn, args.compmd, tracer_type, args.use_map_veto
        )
        clustering_random_comparison = mkclusdat_validation.compare_tables(lss_random_fn, desiblind_random_fn)

        random_app = f'{rannum}_clustering.ran.fits'
        lss_ran_ngc, lss_ran_sgc = split_gc_validation.split_gc_file(common, lss_random_fn, output_dir / 'lss_', random_app)
        desi_ran_ngc, desi_ran_sgc = split_gc_validation.split_gc_file(common, desiblind_random_fn, output_dir / 'desiblind_', random_app)
        split_random_comparison = {
            'random_NGC': mkclusdat_validation.compare_tables(lss_ran_ngc, desi_ran_ngc),
            'random_SGC': mkclusdat_validation.compare_tables(lss_ran_sgc, desi_ran_sgc),
        }

        random_summary = {
            'random_catalog_for_mkclusran': str(random_catalog),
            'random_sample_file': str(random_sample_fn),
            'random_rows': random_rows,
            'clustering_random_comparison': clustering_random_comparison,
            'split_gc_comparison': split_random_comparison,
            'files': {
                'lss_clustering_random': str(lss_random_fn),
                'desiblind_clustering_random': str(desiblind_random_fn),
                'lss_random_NGC': str(lss_ran_ngc),
                'desiblind_random_NGC': str(desi_ran_ngc),
                'lss_random_SGC': str(lss_ran_sgc),
                'desiblind_random_SGC': str(desi_ran_sgc),
            },
        }

        if args.cleanup_random_files:
            random_summary['removed_files'] = unlink_existing([
                random_sample_fn,
                lss_random_fn, desiblind_random_fn,
                lss_ran_ngc, lss_ran_sgc, desi_ran_ngc, desi_ran_sgc,
            ])

        summary['randoms'][str(rannum)] = random_summary
        summary_fn.write_text(json.dumps(summary, indent=2, sort_keys=True))

        print(
            f"random={rannum} PASS rows={clustering_random_comparison['rows']} "
            f"max_abs_delta_Z={clustering_random_comparison.get('max_abs_delta_Z', 'NA')} "
            f"max_abs_delta_WEIGHT={clustering_random_comparison.get('max_abs_delta_WEIGHT', 'NA')} "
            f"max_abs_delta_WEIGHT_SYS={clustering_random_comparison.get('max_abs_delta_WEIGHT_SYS', 'NA')} "
            f"max_abs_delta_WEIGHT_FKP={clustering_random_comparison.get('max_abs_delta_WEIGHT_FKP', 'NA')}"
        )

    print('catalog_bao_lss_multi_randoms_validation=PASS')
    print(f'output_dir={output_dir}')
    print(f'summary={summary_fn}')
    print(f'rannums={rannums}')


if __name__ == '__main__':
    main()
