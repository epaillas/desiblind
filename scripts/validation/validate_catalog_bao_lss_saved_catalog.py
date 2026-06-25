"""Validate the LSS-style saved full-catalog BAO/AP blinding workflow.

This script is intentionally environment-dependent and belongs under
``scripts/validation`` rather than ``tests``. It uses LSS as the reference
implementation, writes only to a fresh scratch directory by default, and checks
that replacing the LSS redshift shifter with ``CatalogBAOBlinder`` gives the
same saved blinded full catalog through the LSS BAO data workflow:

1. compute input n(z) from the unblinded full data catalog;
2. add input ``WEIGHT_FKP`` when missing, as the LSS script does;
3. clip ``Z_not4clus`` only before the BAO/AP redshift shift;
4. write the shifted full catalog to disk;
5. recompute output n(z) from shifted ``Z``;
6. apply ``WEIGHT_SYS *= nz_in / nz_out`` and save the final full catalog.

It does not run ``mkclusdat``/``mkclusran`` yet; those are later saved-catalog
workflow validation layers.
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

# Prefer this checkout over any packaged desiblind in the shared environment.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from desiblind import CatalogBAOBlinder


DEFAULT_LSS_REPO = '/global/homes/u/uendert/repos/desi/LSS'
DEFAULT_REAL_CATALOG = '/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1/LRG_full_HPmapcut.dat.fits'
DEFAULT_RANDOM_CATALOG = '/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1/LRG_1_full_HPmapcut.ran.fits'
DEFAULT_PARAMETERS = {'w0': -0.95, 'wa': 0.10}
P0_BY_TRACER = {'BGS': 7000., 'LRG': 10000., 'LGE': 10000., 'ELG': 4000., 'QSO': 6000.}
ZRANGE_BY_TRACER = {'BGS': (0.1, 0.4), 'LRG': (0.4, 1.1), 'LGE': (0.4, 1.1), 'ELG': (0.8, 1.6), 'QSO': (0.8, 3.5)}
DZ_BY_TRACER = {'QSO': 0.02}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate the saved full-catalog BAO/AP workflow against LSS, replacing only the redshift shifter.'
    )
    parser.add_argument('--lss-repo', default=os.environ.get('LSS_REPO', DEFAULT_LSS_REPO))
    parser.add_argument('--output-dir', default=None,
                        help='Directory for scratch outputs. Default: $SCRATCH/desiblind_lss_validation/saved-catalog-<timestamp>-<pid>.')
    parser.add_argument('--real-catalog', default=DEFAULT_REAL_CATALOG)
    parser.add_argument('--random-catalog', default=DEFAULT_RANDOM_CATALOG)
    parser.add_argument('--nrows', type=int, default=20000,
                        help='Number of real-catalog rows copied into each scratch input branch.')
    parser.add_argument('--w0', type=float, default=DEFAULT_PARAMETERS['w0'])
    parser.add_argument('--wa', type=float, default=DEFAULT_PARAMETERS['wa'])
    parser.add_argument('--tracer-name', default='LRG3',
                        help='Bare canonical tracer-bin name passed to CatalogBAOBlinder.')
    parser.add_argument('--tracer-type', default='LRG',
                        help='Tracer type passed to LSS common.mknz_full/addFKPfull, e.g. LRG.')
    parser.add_argument('--input-zcol', default='Z_not4clus')
    parser.add_argument('--output-zcol', default='Z')
    parser.add_argument('--zmin', type=float, default=None)
    parser.add_argument('--zmax', type=float, default=None)
    parser.add_argument('--dz', type=float, default=None)
    parser.add_argument('--randens', type=float, default=2500.)
    parser.add_argument('--p0', type=float, default=None,
                        help='P0 for LSS addFKPfull when WEIGHT_FKP is missing. Default follows tracer type.')
    parser.add_argument('--clip-min', type=float, default=0.01)
    parser.add_argument('--clip-max', type=float, default=3.6)
    parser.add_argument('--wsyscol', default=None,
                        help='Column copied to WEIGHT_SYS if WEIGHT_SYS is missing, matching LSS --wsyscol behavior.')
    parser.add_argument('--skip-add-fkp', action='store_true',
                        help='Do not mirror the LSS addFKPfull step when WEIGHT_FKP is missing.')
    return parser.parse_args()


def simple_tracer(tracer):
    tracer = tracer.upper()
    if tracer.startswith('LRG'):
        return 'LRG'
    if tracer.startswith('ELG'):
        return 'ELG'
    if tracer.startswith('QSO'):
        return 'QSO'
    if tracer.startswith('BGS'):
        return 'BGS'
    return tracer[:3]


def add_lss_to_path(lss_repo):
    lss_py = Path(lss_repo).expanduser().resolve(strict=False) / 'py'
    if lss_py.exists():
        sys.path.insert(0, str(lss_py))
    try:
        import LSS.blinding_tools as lss_blind
        import LSS.common_tools as common
    except ImportError as exc:
        raise RuntimeError(
            f'Could not import LSS modules. Pass --lss-repo or set LSS_REPO. Tried: {lss_py}'
        ) from exc
    return lss_blind, common, lss_py


def make_output_dir(output_dir=None):
    if output_dir is None:
        base = Path(os.environ.get('SCRATCH', '.')) / 'desiblind_lss_validation'
        stamp = datetime.now(timezone.utc).strftime('saved-catalog-%Y%m%d-%H%M%S')
        output_dir = base / f'{stamp}-{os.getpid()}'
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def read_sample(filename, nrows, input_zcol):
    filename = Path(filename).expanduser().resolve(strict=False)
    if not filename.exists():
        raise FileNotFoundError(filename)
    rows = np.arange(nrows) if nrows is not None and nrows > 0 else None
    table = Table(fitsio.read(str(filename), ext='LSS', rows=rows))
    if input_zcol not in table.colnames:
        raise ValueError(f'{filename} does not contain {input_zcol!r}')
    return table, filename


def write_lss(common, table, filename):
    filename = Path(filename)
    if filename.exists():
        filename.unlink()
    common.write_LSS(table, str(filename))
    return filename


def compute_nz(common, data_fn, random_fn, tracer_type, dz, zmin, zmax, randens, zcol):
    return common.mknz_full(
        str(data_fn), str(random_fn), tracer_type, bs=dz, zmin=zmin, zmax=zmax,
        randens=randens, write='n', md='data', zcol=zcol,
    )


def add_fkp_if_needed(common, data_fn, nz_in, tracer_type, dz, zmin, zmax, p0, input_zcol, skip_add_fkp=False):
    cols = list(fitsio.read(str(data_fn)).dtype.names)
    added = False
    if 'WEIGHT_FKP' not in cols and not skip_add_fkp:
        common.addFKPfull(str(data_fn), nz_in, tracer_type, bs=dz, zmin=zmin, zmax=zmax,
                          P0=p0, md='data', zcol=input_zcol)
        added = True
    return added


def apply_lss_weight_sys_update(common, shifted_fn, nz_in, nz_out, zmin, zmax, dz, output_zcol, wsyscol=None):
    table = Table(fitsio.read(str(shifted_fn), ext='LSS'))
    if 'WEIGHT_SYS' not in table.colnames:
        if wsyscol is not None:
            table['WEIGHT_SYS'] = np.copy(table[wsyscol])
        else:
            table['WEIGHT_SYS'] = np.ones(len(table))

    z = np.asarray(table[output_zcol])
    zind = ((z - zmin) / dz).astype(int)
    gz = np.asarray(table['ZWARN']) != 999999
    zr = (z > zmin) & (z < zmax)
    mask = gz & zr

    weights = np.ones(len(table))
    weights[mask] = nz_in[zind[mask]] / nz_out[zind[mask]]
    table['WEIGHT_SYS'] *= weights
    write_lss(common, table, shifted_fn)
    return shifted_fn


def run_saved_workflow(branch, sample, common, lss_blind, output_dir, random_fn, params, tracer_name,
                       tracer_type, dz, zmin, zmax, randens, p0, input_zcol, output_zcol,
                       clip_min, clip_max, wsyscol=None, skip_add_fkp=False):
    input_fn = write_lss(common, sample, output_dir / f'{branch}_input_full_sample.dat.fits')

    # LSS computes nz_in before clipping Z_not4clus, and addFKPfull also uses this un-clipped input catalog.
    nz_in = compute_nz(common, input_fn, random_fn, tracer_type, dz, zmin, zmax, randens, input_zcol)
    added_fkp = add_fkp_if_needed(common, input_fn, nz_in, tracer_type, dz, zmin, zmax, p0, input_zcol,
                                  skip_add_fkp=skip_add_fkp)

    data = Table(fitsio.read(str(input_fn), ext='LSS'))
    data[input_zcol] = np.clip(data[input_zcol], clip_min, clip_max)
    shifted_fn = output_dir / f'{branch}_full.dat.fits'

    if branch == 'lss':
        lss_blind.apply_zshift_DE(data, str(shifted_fn), w0=params['w0'], wa=params['wa'], zcol=input_zcol)
    elif branch == 'desiblind':
        shifted = CatalogBAOBlinder.apply_blinding(
            tracer_name, data, parameters=params, input_zcol=input_zcol, output_zcol=output_zcol
        )
        write_lss(common, shifted, shifted_fn)
    else:
        raise ValueError(branch)

    nz_out = compute_nz(common, shifted_fn, random_fn, tracer_type, dz, zmin, zmax, randens, output_zcol)
    apply_lss_weight_sys_update(common, shifted_fn, nz_in, nz_out, zmin, zmax, dz, output_zcol, wsyscol=wsyscol)
    return {
        'input_fn': str(input_fn),
        'saved_full_fn': str(shifted_fn),
        'nz_in': nz_in,
        'nz_out': nz_out,
        'added_fkp': added_fkp,
    }


def compare_saved_catalogs(lss_result, desiblind_result, output_zcol):
    lss = Table(fitsio.read(lss_result['saved_full_fn'], ext='LSS'))
    desi = Table(fitsio.read(desiblind_result['saved_full_fn'], ext='LSS'))
    if len(lss) != len(desi):
        raise AssertionError(f'row count differs: {len(lss)} != {len(desi)}')

    columns = ['TARGETID', 'RA', 'DEC', 'Z_not4clus', output_zcol, 'WEIGHT_SYS', 'WEIGHT_FKP']
    column_deltas = {}
    for column in columns:
        if column in lss.colnames and column in desi.colnames:
            atol = 1e-12 if np.issubdtype(np.asarray(lss[column]).dtype, np.floating) else 0
            np.testing.assert_allclose(np.asarray(desi[column]), np.asarray(lss[column]), rtol=0, atol=atol, err_msg=column)
            if np.issubdtype(np.asarray(lss[column]).dtype, np.number):
                column_deltas[column] = float(np.nanmax(np.abs(np.asarray(desi[column]) - np.asarray(lss[column]))))

    np.testing.assert_allclose(desiblind_result['nz_in'], lss_result['nz_in'], rtol=0, atol=1e-12, err_msg='nz_in')
    np.testing.assert_allclose(desiblind_result['nz_out'], lss_result['nz_out'], rtol=0, atol=1e-12, err_msg='nz_out')

    comparison = {
        'rows': int(len(lss)),
        'max_abs_delta_nz_in': float(np.nanmax(np.abs(desiblind_result['nz_in'] - lss_result['nz_in']))),
        'max_abs_delta_nz_out': float(np.nanmax(np.abs(desiblind_result['nz_out'] - lss_result['nz_out']))),
        'column_max_abs_delta': column_deltas,
    }
    if output_zcol in column_deltas:
        comparison[f'max_abs_delta_{output_zcol}'] = column_deltas[output_zcol]
    if 'WEIGHT_SYS' in column_deltas:
        comparison['max_abs_delta_WEIGHT_SYS'] = column_deltas['WEIGHT_SYS']
    if 'WEIGHT_FKP' in column_deltas:
        comparison['max_abs_delta_WEIGHT_FKP'] = column_deltas['WEIGHT_FKP']
    return comparison


def main():
    args = parse_args()
    if args.output_zcol != 'Z':
        raise ValueError('LSS.blinding_tools.apply_zshift_DE always writes to Z; keep --output-zcol=Z.')

    tracer_type = simple_tracer(args.tracer_type)
    zmin, zmax = (args.zmin, args.zmax)
    if zmin is None or zmax is None:
        default_zmin, default_zmax = ZRANGE_BY_TRACER[tracer_type]
        zmin = default_zmin if zmin is None else zmin
        zmax = default_zmax if zmax is None else zmax
    dz = args.dz if args.dz is not None else DZ_BY_TRACER.get(tracer_type, 0.01)
    p0 = args.p0 if args.p0 is not None else P0_BY_TRACER[tracer_type]

    params = {'w0': args.w0, 'wa': args.wa}
    lss_blind, common, lss_py = add_lss_to_path(args.lss_repo)
    output_dir = make_output_dir(args.output_dir)
    random_fn = Path(args.random_catalog).expanduser().resolve(strict=False)
    if not random_fn.exists():
        raise FileNotFoundError(random_fn)

    sample, sample_fn = read_sample(args.real_catalog, args.nrows, args.input_zcol)

    lss_result = run_saved_workflow(
        'lss', sample, common, lss_blind, output_dir, random_fn, params, args.tracer_name,
        tracer_type, dz, zmin, zmax, args.randens, p0, args.input_zcol, args.output_zcol,
        args.clip_min, args.clip_max, wsyscol=args.wsyscol, skip_add_fkp=args.skip_add_fkp,
    )
    desiblind_result = run_saved_workflow(
        'desiblind', sample, common, lss_blind, output_dir, random_fn, params, args.tracer_name,
        tracer_type, dz, zmin, zmax, args.randens, p0, args.input_zcol, args.output_zcol,
        args.clip_min, args.clip_max, wsyscol=args.wsyscol, skip_add_fkp=args.skip_add_fkp,
    )

    comparison = compare_saved_catalogs(lss_result, desiblind_result, output_zcol=args.output_zcol)
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
        'input_zcol': args.input_zcol,
        'output_zcol': args.output_zcol,
        'clip': [args.clip_min, args.clip_max],
        'skip_add_fkp': args.skip_add_fkp,
        'lss': {key: value for key, value in lss_result.items() if not isinstance(value, np.ndarray)},
        'desiblind': {key: value for key, value in desiblind_result.items() if not isinstance(value, np.ndarray)},
        'comparison': comparison,
    }
    summary_fn = output_dir / 'summary.json'
    summary_fn.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print('catalog_bao_lss_saved_catalog_validation=PASS')
    print(f'output_dir={output_dir}')
    print(f'summary={summary_fn}')
    print(
        f"rows={comparison['rows']} "
        f"max_abs_delta_{args.output_zcol}={comparison.get(f'max_abs_delta_{args.output_zcol}', 'NA')} "
        f"max_abs_delta_WEIGHT_SYS={comparison.get('max_abs_delta_WEIGHT_SYS', 'NA')} "
        f"max_abs_delta_WEIGHT_FKP={comparison.get('max_abs_delta_WEIGHT_FKP', 'NA')} "
        f"max_abs_delta_nz_in={comparison['max_abs_delta_nz_in']} "
        f"max_abs_delta_nz_out={comparison['max_abs_delta_nz_out']}"
    )


if __name__ == '__main__':
    main()
