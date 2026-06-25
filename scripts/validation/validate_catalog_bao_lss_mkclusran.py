"""Validate LSS ``mkclusran`` after saved full-catalog BAO/AP blinding.

This validation extends the saved-catalog and ``mkclusdat`` validations by
running the actual LSS ``LSS.main.cattools.mkclusran`` random-catalog generation
step on the LSS and desiblind branches. It checks that replacing only the
BAO/AP redshift shifter with ``CatalogBAOBlinder`` gives identical clustering
random catalogs for the same scratch random input sample and deterministic LSS
random seed.

It does not validate the later optional RSD/fNL/reconstruction workflows.
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

import validate_catalog_bao_lss_mkclusdat as mkclusdat_validation
import validate_catalog_bao_lss_saved_catalog as saved


DEFAULT_RCOLS = ['Z', 'WEIGHT', 'WEIGHT_SYS', 'WEIGHT_COMP', 'WEIGHT_ZFAIL',
                 'WEIGHT_FKP', 'TARGETID_DATA', 'WEIGHT_SN']
RANDOM_INPUT_COLUMNS = ['RA', 'DEC', 'TARGETID', 'TILEID', 'NTILE', 'PHOTSYS', 'FRAC_TLOBS_TILES']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate LSS mkclusran after LSS-style saved full-catalog BAO/AP blinding.'
    )
    parser.add_argument('--lss-repo', default=os.environ.get('LSS_REPO', saved.DEFAULT_LSS_REPO))
    parser.add_argument('--output-dir', default=None,
                        help='Directory for scratch outputs. Default: $SCRATCH/desiblind_lss_validation/mkclusran-<timestamp>-<pid>.')
    parser.add_argument('--real-catalog', default=saved.DEFAULT_REAL_CATALOG)
    parser.add_argument('--random-catalog', default=saved.DEFAULT_RANDOM_CATALOG,
                        help='Full random catalog used as the mkclusran input sample.')
    parser.add_argument('--nz-random-catalog', default=None,
                        help='Random catalog used for LSS mknz_full area normalization. Defaults to --random-catalog. In the LSS BAO script this is typically the _1_full random even when mkclusran processes another random number.')
    parser.add_argument('--nrows', type=int, default=20000,
                        help='Number of data rows copied into each scratch input branch.')
    parser.add_argument('--random-nrows', type=int, default=50000,
                        help='Number of random rows copied into the scratch random input.')
    parser.add_argument('--rannum', type=int, default=1,
                        help='Random catalog number and deterministic seed used by LSS mkclusran.')
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
    parser.add_argument('--use-map-veto', default='_HPmapcut',
                        help='Suffix used in the scratch random input filename, matching LSS mkclusran use_map_veto.')
    return parser.parse_args()


def make_output_dir(output_dir=None):
    if output_dir is None:
        base = Path(os.environ.get('SCRATCH', '.')) / 'desiblind_lss_validation'
        stamp = datetime.now(timezone.utc).strftime('mkclusran-%Y%m%d-%H%M%S')
        output_dir = base / f'{stamp}-{os.getpid()}'
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def select_balanced_photsys_rows(filename, nrows):
    """Return row indices with both PHOTSYS=N and PHOTSYS=S represented.

    LSS mkclusran normalizes N/S random weights separately. If a tiny scratch
    sample contains only one PHOTSYS region, the untouched LSS code hits 0/0 in
    the empty region and fills weights with NaN. For this validation we select a
    small but balanced sample so the LSS N/S normalization is meaningful.
    """
    if nrows is None or nrows <= 0:
        return None
    photsys = np.asarray(fitsio.read(str(filename), ext='LSS', columns=['PHOTSYS'])['PHOTSYS']).astype(str)
    counts = {'N': int(np.sum(photsys == 'N')), 'S': int(np.sum(photsys == 'S'))}
    if not counts['N'] or not counts['S']:
        raise ValueError(f'{filename} does not contain both PHOTSYS=N and PHOTSYS=S: {counts}')
    n_n = min(counts['N'], nrows // 2)
    n_s = min(counts['S'], nrows - n_n)
    # If one side was capped, put the remaining rows on the other side.
    remaining = nrows - n_n - n_s
    if remaining > 0:
        extra_n = min(counts['N'] - n_n, remaining)
        n_n += extra_n
        remaining -= extra_n
    if remaining > 0:
        extra_s = min(counts['S'] - n_s, remaining)
        n_s += extra_s
        remaining -= extra_s
    rows = np.concatenate([np.flatnonzero(photsys == 'N')[:n_n], np.flatnonzero(photsys == 'S')[:n_s]])
    rows.sort()
    return rows


def read_balanced_sample(filename, nrows, input_zcol):
    filename = Path(filename).expanduser().resolve(strict=False)
    if not filename.exists():
        raise FileNotFoundError(filename)
    rows = select_balanced_photsys_rows(filename, nrows)
    table = Table(fitsio.read(str(filename), ext='LSS', rows=rows))
    if input_zcol not in table.colnames:
        raise ValueError(f'{filename} does not contain {input_zcol!r}')
    return table, filename


def write_random_sample(common, random_catalog, output_dir, rannum, use_map_veto, random_nrows):
    random_catalog = Path(random_catalog).expanduser().resolve(strict=False)
    if not random_catalog.exists():
        raise FileNotFoundError(random_catalog)
    rows = select_balanced_photsys_rows(random_catalog, random_nrows)
    table = Table(fitsio.read(str(random_catalog), ext='LSS', rows=rows, columns=RANDOM_INPUT_COLUMNS))
    prefix = output_dir / 'random_input_'
    random_fn = Path(f'{prefix}{rannum}_full{use_map_veto}.ran.fits')
    saved.write_lss(common, table, random_fn)
    return prefix, random_fn, int(len(table))


def run_mkclusran(ct, random_prefix, output_prefix, rannum, clus_fn, compmd, tracer_type, use_map_veto):
    clus_array = fitsio.read(str(clus_fn), ext='LSS')
    ct.mkclusran(
        str(random_prefix), str(output_prefix), rannum,
        rcols=list(DEFAULT_RCOLS), compmd=compmd, clus_arrays=[clus_array],
        use_map_veto=use_map_veto, tp=tracer_type,
    )
    output_fn = Path(f'{output_prefix}{rannum}_clustering.ran.fits')
    if not output_fn.exists():
        raise FileNotFoundError(output_fn)
    return output_fn


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
    random_prefix, random_sample_fn, random_rows = write_random_sample(
        common, args.random_catalog, output_dir, args.rannum, args.use_map_veto, args.random_nrows
    )
    sample, sample_fn = read_balanced_sample(args.real_catalog, args.nrows, args.input_zcol)

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
    clustering_comparison = mkclusdat_validation.compare_tables(lss_clus_fn, desiblind_clus_fn)

    lss_random_fn = run_mkclusran(ct, random_prefix, output_dir / 'lss_', args.rannum,
                                  lss_clus_fn, args.compmd, tracer_type, args.use_map_veto)
    desiblind_random_fn = run_mkclusran(ct, random_prefix, output_dir / 'desiblind_', args.rannum,
                                        desiblind_clus_fn, args.compmd, tracer_type, args.use_map_veto)
    random_comparison = mkclusdat_validation.compare_tables(lss_random_fn, desiblind_random_fn)

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
        'clustering_data_comparison': clustering_comparison,
        'clustering_random_comparison': random_comparison,
        'files': {
            'lss_full': lss_result['saved_full_fn'],
            'desiblind_full': desiblind_result['saved_full_fn'],
            'lss_clustering_data': str(lss_clus_fn),
            'desiblind_clustering_data': str(desiblind_clus_fn),
            'lss_clustering_random': str(lss_random_fn),
            'desiblind_clustering_random': str(desiblind_random_fn),
        },
    }
    summary_fn = output_dir / 'summary.json'
    summary_fn.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print('catalog_bao_lss_mkclusran_validation=PASS')
    print(f'output_dir={output_dir}')
    print(f'summary={summary_fn}')
    print(
        f"rows={random_comparison['rows']} "
        f"max_abs_delta_Z={random_comparison.get('max_abs_delta_Z', 'NA')} "
        f"max_abs_delta_WEIGHT={random_comparison.get('max_abs_delta_WEIGHT', 'NA')} "
        f"max_abs_delta_WEIGHT_SYS={random_comparison.get('max_abs_delta_WEIGHT_SYS', 'NA')} "
        f"max_abs_delta_WEIGHT_FKP={random_comparison.get('max_abs_delta_WEIGHT_FKP', 'NA')}"
    )


if __name__ == '__main__':
    main()
