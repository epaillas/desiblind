"""Validate catalog-level RSD blinding against the LSS RSD redshift-shift step.

This compares ``desiblind.CatalogRSDBlinder`` to
``LSS.blinding_tools.apply_zshift_RSD`` for the same observed clustering catalog,
reconstructed-realspace catalog, and derived ``fgrowth_blind`` value.

It is a direct transformation validator, not the full reconstruction pipeline.
The default toy validation is self-contained. Pass ``--data-catalog`` and
``--realspace-catalog`` to validate real LSS clustering products.
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
DEFAULT_PARAMETERS = {'w0': -0.95, 'wa': 0.10, 'zeff': 0.8, 'bias': 2.0, 'fiducial_f': 0.8}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate desiblind CatalogRSDBlinder against LSS.blinding_tools.apply_zshift_RSD.'
    )
    parser.add_argument('--lss-repo', default=os.environ.get('LSS_REPO', DEFAULT_LSS_REPO))
    parser.add_argument('--output-dir', default=None,
                        help='Directory for scratch validation outputs. Default: $SCRATCH/desiblind_lss_validation/rsd-redshift-shift-<timestamp>-<pid>.')
    parser.add_argument('--tracer-name', default='LRG3')
    parser.add_argument('--w0', type=float, default=DEFAULT_PARAMETERS['w0'])
    parser.add_argument('--wa', type=float, default=DEFAULT_PARAMETERS['wa'])
    parser.add_argument('--zeff', type=float, default=DEFAULT_PARAMETERS['zeff'])
    parser.add_argument('--bias', type=float, default=DEFAULT_PARAMETERS['bias'])
    parser.add_argument('--fiducial-f', type=float, default=DEFAULT_PARAMETERS['fiducial_f'])
    parser.add_argument('--max-df-fraction', type=float, default=0.1)
    parser.add_argument('--fgrowth-blind', type=float, default=None,
                        help='Optional testing override. By default fgrowth_blind is derived from w0/wa.')
    parser.add_argument('--data-catalog', default=None,
                        help='Observed clustering data catalog passed to LSS.apply_zshift_RSD.')
    parser.add_argument('--realspace-catalog', default=None,
                        help='Reconstructed-realspace data catalog passed to LSS.apply_zshift_RSD.')
    parser.add_argument('--nrows', type=int, default=0,
                        help='Rows to read from real catalogs. Default 0 means all rows.')
    parser.add_argument('--zcol', default='Z')
    parser.add_argument('--output-zcol', default='Z')
    return parser.parse_args()


def add_lss_to_path(lss_repo):
    lss_py = Path(lss_repo).expanduser().resolve(strict=False) / 'py'
    if lss_py.exists():
        sys.path.insert(0, str(lss_py))
    try:
        import LSS.blinding_tools as lss_blind
    except ImportError as exc:
        raise RuntimeError(
            f'Could not import LSS.blinding_tools. Pass --lss-repo or set LSS_REPO. Tried: {lss_py}'
        ) from exc
    return lss_blind, lss_py


def make_output_dir(output_dir=None):
    if output_dir is None:
        base = Path(os.environ.get('SCRATCH', '.')) / 'desiblind_lss_validation'
        stamp = datetime.now(timezone.utc).strftime('rsd-redshift-shift-%Y%m%d-%H%M%S')
        output_dir = base / f'{stamp}-{os.getpid()}'
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def make_toy_catalogs(zcol='Z'):
    data = Table()
    data['TARGETID'] = np.array([101, 102, 103, 104], dtype='i8')
    data['RA'] = np.array([10., 20., 30., 40.])
    data['DEC'] = np.array([-5., 0., 5., 10.])
    data[zcol] = np.array([0.76, 0.82, 0.95, 1.05])
    data['WEIGHT'] = np.ones(4)

    realspace = data.copy()
    realspace[zcol] = np.array([0.75, 0.80, 0.94, 1.02])
    return data, realspace


def read_catalog_pair(data_catalog, realspace_catalog, nrows):
    data_catalog = Path(data_catalog).expanduser().resolve(strict=False)
    realspace_catalog = Path(realspace_catalog).expanduser().resolve(strict=False)
    if not data_catalog.exists():
        raise FileNotFoundError(data_catalog)
    if not realspace_catalog.exists():
        raise FileNotFoundError(realspace_catalog)
    rows = np.arange(nrows) if nrows and nrows > 0 else None
    data = Table(fitsio.read(str(data_catalog), ext='LSS', rows=rows))
    realspace = Table(fitsio.read(str(realspace_catalog), ext='LSS', rows=rows))
    return data, realspace, data_catalog, realspace_catalog


def run_lss_apply_zshift_rsd(lss_blind, data, realspace, output_fn, params, zcol):
    lss_data = data.copy()
    lss_realspace = realspace.copy()
    lss_blind.apply_zshift_RSD(
        lss_data, lss_realspace, str(output_fn),
        fgrowth_fid=params['fiducial_f'], fgrowth_blind=params['fgrowth_blind'], zcol=zcol,
    )
    return Table(fitsio.read(str(output_fn), ext='LSS'))


def run_desiblind(data, realspace, params, tracer_name, zcol, output_zcol):
    return CatalogRSDBlinder.apply_blinding(
        tracer_name, data, realspace, parameters=params, zcol=zcol, output_zcol=output_zcol,
    )


def _assert_column_equal(label, column, left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
        np.testing.assert_allclose(left, right, rtol=0, atol=0, err_msg=f'{label}: {column}')
    else:
        np.testing.assert_array_equal(left, right, err_msg=f'{label}: {column}')


def compare_outputs(label, desiblind_output, lss_output, zcol='Z'):
    if len(desiblind_output) != len(lss_output):
        raise AssertionError(f'{label}: row count differs: {len(desiblind_output)} != {len(lss_output)}')
    common_columns = [column for column in desiblind_output.colnames if column in lss_output.colnames]
    for column in common_columns:
        if column == zcol:
            continue
        _assert_column_equal(label, column, desiblind_output[column], lss_output[column])
    np.testing.assert_allclose(
        np.asarray(desiblind_output[zcol]), np.asarray(lss_output[zcol]),
        rtol=0, atol=1e-12, err_msg=f'{label}: {zcol}'
    )
    return {
        'label': label,
        'rows': int(len(lss_output)),
        'checked_non_z_columns': [column for column in common_columns if column != zcol],
        f'max_abs_delta_{zcol}': float(np.max(np.abs(
            np.asarray(desiblind_output[zcol]) - np.asarray(lss_output[zcol])
        ))),
        f'{zcol}_range': [float(np.min(desiblind_output[zcol])), float(np.max(desiblind_output[zcol]))],
    }


def validate_pair(label, data, realspace, lss_blind, output_dir, params, tracer_name, zcol, output_zcol):
    lss_fn = output_dir / f'{label}_lss_apply_zshift_RSD.fits'
    lss_output = run_lss_apply_zshift_rsd(lss_blind, data, realspace, lss_fn, params=params, zcol=zcol)
    desiblind_output = run_desiblind(data, realspace, params=params, tracer_name=tracer_name,
                                     zcol=zcol, output_zcol=output_zcol)
    result = compare_outputs(label, desiblind_output, lss_output, zcol=output_zcol)
    result['lss_output'] = str(lss_fn)
    return result


def main():
    args = parse_args()
    if args.output_zcol != 'Z':
        raise ValueError('LSS.blinding_tools.apply_zshift_RSD always writes to Z; keep --output-zcol=Z.')
    lss_blind, lss_py = add_lss_to_path(args.lss_repo)
    output_dir = make_output_dir(args.output_dir)

    params = {
        'w0': args.w0,
        'wa': args.wa,
        'zeff': args.zeff,
        'bias': args.bias,
        'fiducial_f': args.fiducial_f,
        'max_df_fraction': args.max_df_fraction,
    }
    if args.fgrowth_blind is not None:
        params['fgrowth_blind'] = args.fgrowth_blind
    else:
        params['fgrowth_blind'] = CatalogRSDBlinder.compute_fgrowth_blind(
            w0=args.w0, wa=args.wa, z=args.zeff, bias=args.bias,
            fiducial_f=args.fiducial_f, max_df_fraction=args.max_df_fraction,
        )

    results = {
        'parameters': params,
        'tracer_name': args.tracer_name,
        'zcol': args.zcol,
        'lss_py': str(lss_py),
        'output_dir': str(output_dir),
        'validations': [],
    }

    toy_data, toy_realspace = make_toy_catalogs(zcol=args.zcol)
    results['validations'].append(validate_pair(
        'toy', toy_data, toy_realspace, lss_blind=lss_blind, output_dir=output_dir,
        params=params, tracer_name=args.tracer_name, zcol=args.zcol, output_zcol=args.output_zcol,
    ))

    if args.data_catalog or args.realspace_catalog:
        if not (args.data_catalog and args.realspace_catalog):
            raise ValueError('Pass both --data-catalog and --realspace-catalog for real validation.')
        real_data, real_realspace, data_fn, realspace_fn = read_catalog_pair(
            args.data_catalog, args.realspace_catalog, args.nrows
        )
        real_result = validate_pair(
            'real_sample', real_data, real_realspace, lss_blind=lss_blind, output_dir=output_dir,
            params=params, tracer_name=args.tracer_name, zcol=args.zcol, output_zcol=args.output_zcol,
        )
        real_result['data_catalog'] = str(data_fn)
        real_result['realspace_catalog'] = str(realspace_fn)
        results['validations'].append(real_result)

    summary_fn = output_dir / 'summary.json'
    summary_fn.write_text(json.dumps(results, indent=2, sort_keys=True))

    print('catalog_rsd_lss_redshift_shift_validation=PASS')
    print(f'fgrowth_blind={params["fgrowth_blind"]}')
    print(f'output_dir={output_dir}')
    print(f'summary={summary_fn}')
    for result in results['validations']:
        print(
            f"{result['label']}: rows={result['rows']} "
            f"max_abs_delta_{args.output_zcol}={result[f'max_abs_delta_{args.output_zcol}']} "
            f"lss_output={result['lss_output']}"
        )


if __name__ == '__main__':
    main()
