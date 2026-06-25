"""Validate catalog-level BAO/AP blinding against the LSS LSS redshift-shift step.

This is an environment-dependent validation script, not a package unit test. It
requires an LSS checkout and, for the real-catalog check, DESI catalog paths
available on NERSC. Outputs are written to a fresh directory under ``$SCRATCH``
by default.
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

# Prefer the checkout containing this validation script over any packaged
# desiblind module that may be loaded by the shared cosmodesi environment.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from desiblind import CatalogBAOBlinder


DEFAULT_LSS_REPO = '/global/homes/u/uendert/repos/desi/LSS'
DEFAULT_REAL_CATALOG = '/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/LSScats/v2.1/LRG_full_HPmapcut.dat.fits'
DEFAULT_PARAMETERS = {'w0': -0.95, 'wa': 0.10}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate desiblind CatalogBAOBlinder against LSS.blinding_tools.apply_zshift_DE.'
    )
    parser.add_argument('--lss-repo', default=os.environ.get('LSS_REPO', DEFAULT_LSS_REPO),
                        help='Path to an LSS checkout. Its py/ directory is added to PYTHONPATH before importing LSS.')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for scratch validation outputs. Default: $SCRATCH/desiblind_lss_validation/redshift-shift-<timestamp>-<pid>.')
    parser.add_argument('--w0', type=float, default=DEFAULT_PARAMETERS['w0'])
    parser.add_argument('--wa', type=float, default=DEFAULT_PARAMETERS['wa'])
    parser.add_argument('--tracer-name', default='LRG3',
                        help='Bare canonical tracer-bin name passed to CatalogBAOBlinder.')
    parser.add_argument('--real-catalog', default=DEFAULT_REAL_CATALOG,
                        help='LSS full catalog used for the real-sample validation.')
    parser.add_argument('--nrows', type=int, default=2000,
                        help='Number of rows to read from the real catalog.')
    parser.add_argument('--skip-real', action='store_true',
                        help='Only run the toy validation, skipping the DESI real-catalog sample.')
    parser.add_argument('--input-zcol', default='Z_not4clus',
                        help='LSS source redshift column for catalog BAO/AP blinding.')
    parser.add_argument('--output-zcol', default='Z',
                        help='Destination redshift column for catalog BAO/AP blinding.')
    parser.add_argument('--clip-min', type=float, default=0.01,
                        help='Minimum source redshift for the LSS-style pre-call clipping.')
    parser.add_argument('--clip-max', type=float, default=3.6,
                        help='Maximum source redshift for the LSS-style pre-call clipping.')
    parser.add_argument('--no-clip', action='store_true',
                        help='Do not clip the source redshift column before validation.')
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
        stamp = datetime.now(timezone.utc).strftime('redshift-shift-%Y%m%d-%H%M%S')
        output_dir = base / f'{stamp}-{os.getpid()}'
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def make_toy_catalog(input_zcol='Z_not4clus'):
    catalog = Table()
    catalog['TARGETID'] = np.array([101, 102, 103, 104], dtype='i8')
    catalog['RA'] = np.array([10., 20., 30., 40.])
    catalog['DEC'] = np.array([-5., 0., 5., 10.])
    # Deliberately make Z and Z_not4clus different to test source -> destination behavior.
    catalog['Z'] = np.array([0.41, 0.81, 1.01, 1.21])
    catalog[input_zcol] = np.array([0.40, 0.80, 1.00, 1.20])
    catalog['WEIGHT'] = np.ones(4)
    return catalog


def clip_source_redshift(catalog, input_zcol, clip_min, clip_max, do_clip=True):
    catalog = catalog.copy()
    if do_clip:
        catalog[input_zcol] = np.clip(catalog[input_zcol], clip_min, clip_max)
    return catalog


def run_lss_apply_zshift_de(lss_blind, catalog, output_fn, params, input_zcol):
    lss_catalog = catalog.copy()
    lss_blind.apply_zshift_DE(lss_catalog, str(output_fn), w0=params['w0'], wa=params['wa'], zcol=input_zcol)
    return Table(fitsio.read(str(output_fn), ext='LSS'))


def run_desiblind(catalog, params, tracer_name, input_zcol, output_zcol):
    return CatalogBAOBlinder.apply_blinding(
        tracer_name, catalog, parameters=params, input_zcol=input_zcol, output_zcol=output_zcol
    )


def assert_matching_outputs(label, desiblind_output, lss_output, input_zcol, output_zcol):
    if len(desiblind_output) != len(lss_output):
        raise AssertionError(f'{label}: row count differs: {len(desiblind_output)} != {len(lss_output)}')

    for column in ['TARGETID', 'RA', 'DEC', input_zcol]:
        if column in desiblind_output.colnames and column in lss_output.colnames:
            np.testing.assert_allclose(
                np.asarray(desiblind_output[column]), np.asarray(lss_output[column]),
                rtol=0, atol=0, err_msg=f'{label}: {column}'
            )

    np.testing.assert_allclose(
        np.asarray(desiblind_output[output_zcol]), np.asarray(lss_output[output_zcol]),
        rtol=0, atol=1e-12, err_msg=f'{label}: {output_zcol}'
    )

    return {
        'label': label,
        'rows': int(len(lss_output)),
        f'max_abs_delta_{output_zcol}': float(np.max(np.abs(
            np.asarray(desiblind_output[output_zcol]) - np.asarray(lss_output[output_zcol])
        ))),
        f'{input_zcol}_range': [float(np.min(desiblind_output[input_zcol])), float(np.max(desiblind_output[input_zcol]))]
            if input_zcol in desiblind_output.colnames else None,
        f'{output_zcol}_range': [float(np.min(desiblind_output[output_zcol])), float(np.max(desiblind_output[output_zcol]))],
    }


def validate_catalog(label, catalog, lss_blind, output_dir, params, tracer_name,
                     input_zcol, output_zcol, clip_min, clip_max, do_clip):
    prepared = clip_source_redshift(catalog, input_zcol, clip_min, clip_max, do_clip=do_clip)
    lss_fn = output_dir / f'{label}_lss_apply_zshift_DE.fits'
    lss_output = run_lss_apply_zshift_de(lss_blind, prepared, lss_fn, params=params, input_zcol=input_zcol)
    desiblind_output = run_desiblind(prepared, params=params, tracer_name=tracer_name,
                                     input_zcol=input_zcol, output_zcol=output_zcol)
    result = assert_matching_outputs(label, desiblind_output, lss_output, input_zcol=input_zcol, output_zcol=output_zcol)
    result['lss_output'] = str(lss_fn)
    return result


def read_real_catalog(filename, input_zcol, nrows):
    filename = Path(filename).expanduser().resolve(strict=False)
    if not filename.exists():
        raise FileNotFoundError(filename)
    rows = np.arange(nrows) if nrows is not None and nrows > 0 else None
    catalog = Table(fitsio.read(str(filename), ext='LSS', rows=rows))
    if input_zcol not in catalog.colnames:
        raise ValueError(f'Real catalog {filename} does not contain {input_zcol!r}')
    return catalog, filename


def main():
    args = parse_args()
    if args.output_zcol != 'Z':
        raise ValueError('LSS.blinding_tools.apply_zshift_DE always writes the transformed redshifts to Z; keep --output-zcol=Z for this validation.')
    params = {'w0': args.w0, 'wa': args.wa}
    lss_blind, lss_py = add_lss_to_path(args.lss_repo)
    output_dir = make_output_dir(args.output_dir)
    do_clip = not args.no_clip

    results = {
        'parameters': params,
        'tracer_name': args.tracer_name,
        'input_zcol': args.input_zcol,
        'output_zcol': args.output_zcol,
        'clip': None if not do_clip else [args.clip_min, args.clip_max],
        'lss_py': str(lss_py),
        'output_dir': str(output_dir),
        'validations': [],
    }

    toy = make_toy_catalog(input_zcol=args.input_zcol)
    results['validations'].append(validate_catalog(
        'toy', toy, lss_blind=lss_blind, output_dir=output_dir, params=params,
        tracer_name=args.tracer_name, input_zcol=args.input_zcol, output_zcol=args.output_zcol,
        clip_min=args.clip_min, clip_max=args.clip_max, do_clip=do_clip,
    ))

    if not args.skip_real:
        real, real_fn = read_real_catalog(args.real_catalog, input_zcol=args.input_zcol, nrows=args.nrows)
        real_result = validate_catalog(
            'real_sample', real, lss_blind=lss_blind, output_dir=output_dir, params=params,
            tracer_name=args.tracer_name, input_zcol=args.input_zcol, output_zcol=args.output_zcol,
            clip_min=args.clip_min, clip_max=args.clip_max, do_clip=do_clip,
        )
        real_result['real_catalog'] = str(real_fn)
        results['validations'].append(real_result)

    summary_fn = output_dir / 'summary.json'
    summary_fn.write_text(json.dumps(results, indent=2, sort_keys=True))

    print('catalog_bao_lss_redshift_shift_validation=PASS')
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
