"""Validate LSS-style BAO/AP n(z) and WEIGHT_SYS reweighting after redshift shift.

This validation is intentionally separate from unit tests. It requires an LSS
checkout plus DESI/NERSC catalog paths, writes only to a fresh scratch output
directory, and checks that replacing the LSS redshift remapper with
``CatalogBAOBlinder`` leaves the subsequent LSS n(z)-ratio/WEIGHT_SYS update
unchanged on a small full-catalog sample.
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate CatalogBAOBlinder through the LSS n(z)/WEIGHT_SYS BAO workflow step.'
    )
    parser.add_argument('--lss-repo', default=os.environ.get('LSS_REPO', DEFAULT_LSS_REPO))
    parser.add_argument('--output-dir', default=None,
                        help='Directory for scratch outputs. Default: $SCRATCH/desiblind_lss_validation/nz-weight-<timestamp>-<pid>.')
    parser.add_argument('--real-catalog', default=DEFAULT_REAL_CATALOG)
    parser.add_argument('--random-catalog', default=DEFAULT_RANDOM_CATALOG,
                        help='Full random catalog used by LSS common.mknz_full for area normalization.')
    parser.add_argument('--nrows', type=int, default=20000,
                        help='Number of real-catalog rows to copy into the scratch sample.')
    parser.add_argument('--w0', type=float, default=DEFAULT_PARAMETERS['w0'])
    parser.add_argument('--wa', type=float, default=DEFAULT_PARAMETERS['wa'])
    parser.add_argument('--tracer-name', default='LRG3')
    parser.add_argument('--tracer-type', default='LRG',
                        help='Tracer type passed to LSS.common_tools.mknz_full, e.g. LRG.')
    parser.add_argument('--input-zcol', default='Z_not4clus')
    parser.add_argument('--output-zcol', default='Z')
    parser.add_argument('--zmin', type=float, default=0.4)
    parser.add_argument('--zmax', type=float, default=1.1)
    parser.add_argument('--dz', type=float, default=0.01)
    parser.add_argument('--randens', type=float, default=2500.)
    parser.add_argument('--clip-min', type=float, default=0.01)
    parser.add_argument('--clip-max', type=float, default=3.6)
    return parser.parse_args()


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
        stamp = datetime.now(timezone.utc).strftime('nz-weight-%Y%m%d-%H%M%S')
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


def write_table(filename, table):
    filename = Path(filename)
    if filename.exists():
        filename.unlink()
    fitsio.write(str(filename), table.as_array(), extname='LSS', clobber=True)
    return filename


def compute_nz(common, data_fn, random_fn, tracer_type, dz, zmin, zmax, randens, zcol):
    return common.mknz_full(
        str(data_fn), str(random_fn), tracer_type, bs=dz, zmin=zmin, zmax=zmax,
        randens=randens, write='n', md='data', zcol=zcol,
    )


def lss_shift(lss_blind, table, output_fn, params, input_zcol):
    lss_table = table.copy()
    # Match LSS apply_blinding_main_fromfile_fcomp.py: clipping is done after
    # nz_in/addFKPfull and immediately before apply_zshift_DE.
    lss_table[input_zcol] = np.clip(lss_table[input_zcol], 0.01, 3.6)
    lss_blind.apply_zshift_DE(lss_table, str(output_fn), w0=params['w0'], wa=params['wa'], zcol=input_zcol)
    return output_fn


def desiblind_shift(table, output_fn, params, tracer_name, input_zcol, output_zcol):
    desiblind_table = table.copy()
    # Match LSS apply_blinding_main_fromfile_fcomp.py: clipping is done after
    # nz_in/addFKPfull and immediately before the BAO/AP redshift shift.
    desiblind_table[input_zcol] = np.clip(desiblind_table[input_zcol], 0.01, 3.6)
    shifted = CatalogBAOBlinder.apply_blinding(
        tracer_name, desiblind_table, parameters=params, input_zcol=input_zcol, output_zcol=output_zcol
    )
    return write_table(output_fn, shifted)


def apply_lss_weight_sys_update(input_fn, output_fn, nz_in, nz_out, zmin, zmax, dz, output_zcol):
    """Apply the WEIGHT_SYS update lines from LSS apply_blinding_main_fromfile_fcomp.py."""
    table = Table(fitsio.read(str(input_fn), ext='LSS'))
    if 'WEIGHT_SYS' not in table.colnames:
        table['WEIGHT_SYS'] = np.ones(len(table))

    z = np.asarray(table[output_zcol])
    zind = ((z - zmin) / dz).astype(int)
    gz = np.asarray(table['ZWARN']) != 999999
    zr = (z > zmin) & (z < zmax)
    mask = gz & zr

    weights = np.ones(len(table))
    weights[mask] = nz_in[zind[mask]] / nz_out[zind[mask]]
    table['WEIGHT_SYS'] *= weights
    write_table(output_fn, table)
    return output_fn


def compare_outputs(lss_fn, desiblind_fn, output_zcol):
    lss = Table(fitsio.read(str(lss_fn), ext='LSS'))
    desi = Table(fitsio.read(str(desiblind_fn), ext='LSS'))
    if len(lss) != len(desi):
        raise AssertionError(f'row count differs: {len(lss)} != {len(desi)}')

    for column in ['TARGETID', 'RA', 'DEC', 'Z_not4clus', output_zcol, 'WEIGHT_SYS']:
        if column in lss.colnames and column in desi.colnames:
            np.testing.assert_allclose(
                np.asarray(desi[column]), np.asarray(lss[column]),
                rtol=0, atol=1e-12 if column in [output_zcol, 'WEIGHT_SYS'] else 0,
                err_msg=column,
            )

    dz = np.asarray(desi[output_zcol]) - np.asarray(lss[output_zcol])
    dw = np.asarray(desi['WEIGHT_SYS']) - np.asarray(lss['WEIGHT_SYS'])
    return {
        'rows': int(len(lss)),
        f'max_abs_delta_{output_zcol}': float(np.nanmax(np.abs(dz))),
        'max_abs_delta_WEIGHT_SYS': float(np.nanmax(np.abs(dw))),
        'finite_WEIGHT_SYS_fraction': float(np.mean(np.isfinite(desi['WEIGHT_SYS']))),
        f'{output_zcol}_range': [float(np.nanmin(desi[output_zcol])), float(np.nanmax(desi[output_zcol]))],
        'WEIGHT_SYS_range': [float(np.nanmin(desi['WEIGHT_SYS'])), float(np.nanmax(desi['WEIGHT_SYS']))],
    }


def main():
    args = parse_args()
    if args.output_zcol != 'Z':
        raise ValueError('LSS.blinding_tools.apply_zshift_DE always writes to Z; keep --output-zcol=Z.')

    params = {'w0': args.w0, 'wa': args.wa}
    lss_blind, common, lss_py = add_lss_to_path(args.lss_repo)
    output_dir = make_output_dir(args.output_dir)
    random_fn = Path(args.random_catalog).expanduser().resolve(strict=False)
    if not random_fn.exists():
        raise FileNotFoundError(random_fn)

    sample, sample_fn = read_sample(args.real_catalog, args.nrows, args.input_zcol)
    input_fn = write_table(output_dir / 'input_full_sample.dat.fits', sample)

    nz_in = compute_nz(common, input_fn, random_fn, args.tracer_type, args.dz, args.zmin, args.zmax, args.randens, args.input_zcol)

    lss_shift_fn = lss_shift(lss_blind, sample, output_dir / 'lss_shifted_full_sample.dat.fits', params, args.input_zcol)
    desi_shift_fn = desiblind_shift(sample, output_dir / 'desiblind_shifted_full_sample.dat.fits', params,
                                    args.tracer_name, args.input_zcol, args.output_zcol)

    nz_out_lss = compute_nz(common, lss_shift_fn, random_fn, args.tracer_type, args.dz, args.zmin, args.zmax, args.randens, args.output_zcol)
    nz_out_desi = compute_nz(common, desi_shift_fn, random_fn, args.tracer_type, args.dz, args.zmin, args.zmax, args.randens, args.output_zcol)
    np.testing.assert_allclose(nz_out_desi, nz_out_lss, rtol=0, atol=1e-12, err_msg='nz_out')

    lss_final_fn = apply_lss_weight_sys_update(output_dir / 'lss_shifted_full_sample.dat.fits',
                                               output_dir / 'lss_weighted_full_sample.dat.fits',
                                               nz_in, nz_out_lss, args.zmin, args.zmax, args.dz, args.output_zcol)
    desi_final_fn = apply_lss_weight_sys_update(output_dir / 'desiblind_shifted_full_sample.dat.fits',
                                                output_dir / 'desiblind_weighted_full_sample.dat.fits',
                                                nz_in, nz_out_desi, args.zmin, args.zmax, args.dz, args.output_zcol)

    comparison = compare_outputs(lss_final_fn, desi_final_fn, output_zcol=args.output_zcol)
    comparison['max_abs_delta_nz_out'] = float(np.nanmax(np.abs(nz_out_desi - nz_out_lss)))

    summary = {
        'parameters': params,
        'lss_py': str(lss_py),
        'real_catalog': str(sample_fn),
        'random_catalog': str(random_fn),
        'output_dir': str(output_dir),
        'nrows': args.nrows,
        'zrange': [args.zmin, args.zmax],
        'dz': args.dz,
        'input_zcol': args.input_zcol,
        'output_zcol': args.output_zcol,
        'comparison': comparison,
        'files': {
            'input': str(input_fn),
            'lss_shifted': str(lss_shift_fn),
            'desiblind_shifted': str(desi_shift_fn),
            'lss_weighted': str(lss_final_fn),
            'desiblind_weighted': str(desi_final_fn),
        },
    }
    summary_fn = output_dir / 'summary.json'
    summary_fn.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print('catalog_bao_lss_nz_weight_validation=PASS')
    print(f'output_dir={output_dir}')
    print(f'summary={summary_fn}')
    print(
        f"rows={comparison['rows']} "
        f"max_abs_delta_{args.output_zcol}={comparison[f'max_abs_delta_{args.output_zcol}']} "
        f"max_abs_delta_WEIGHT_SYS={comparison['max_abs_delta_WEIGHT_SYS']} "
        f"max_abs_delta_nz_out={comparison['max_abs_delta_nz_out']}"
    )


if __name__ == '__main__':
    main()
