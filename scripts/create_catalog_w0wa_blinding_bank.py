#!/usr/bin/env python
"""Create private desiblind catalog-level w0/wa blinding banks.

The shared closed catalog blinding draw is a hidden ``(w0, wa)`` cosmology.
BAO/AP uses that pair directly for the distance/redshift remapping. RSD uses
that same pair and derives ``fgrowth_blind`` for each tracer bin from
``w0``, ``wa``, ``zeff`` and ``bias``.

Examples
--------

Create a BAO/AP bank from a native desiblind random draw::

    python scripts/create_catalog_w0wa_blinding_bank.py \
        --output /private/path/catalog_bao_blinding_parameters.npy \
        --bid 42 \
        --generate --seed 12345

Create both BAO/AP and RSD banks from the same generated draw::

    python scripts/create_catalog_w0wa_blinding_bank.py \
        --effects bao rsd \
        --bao-output /private/path/catalog_bao_blinding_parameters.npy \
        --rsd-output /private/path/catalog_rsd_blinding_parameters.npy \
        --bid 42 \
        --generate --seed 12345 \
        --rsd-bin LRG1:0.50:2.0 \
        --rsd-bin LRG2:0.70:2.0 \
        --record-fn /private/path/catalog_w0wa_blinding_record.json

Seed a private bank from a historical LSS row index for DR1/Y1 compatibility::

    python scripts/create_catalog_w0wa_blinding_bank.py \
        --output /private/path/catalog_bao_blinding_parameters.npy \
        --bid 42 \
        --lss-index 281

By default, the script validates the DESI 3 percent BAO/AP alpha-shift mask
before writing anything.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Allow running directly from a source checkout without installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from desiblind import CatalogBAOBlinder, CatalogRSDBlinder
from desiblind.catalog_bao import DEFAULT_LSS_W0WA_BANK

DEFAULT_BAO_TRACER_BINS = ('BGS1', 'LRG1', 'LRG2', 'LRG3', 'ELG1', 'ELG2', 'QSO1')
VALID_EFFECTS = ('bao', 'rsd')


def parse_zrange(text: str) -> tuple[float, float]:
    parts = text.replace(',', ' ').split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f'zrange must contain two floats, got {text!r}')
    return (float(parts[0]), float(parts[1]))


def parse_range(text: str) -> tuple[float, float]:
    parts = text.replace(',', ' ').split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f'range must contain two floats, got {text!r}')
    low, high = map(float, parts)
    if low >= high:
        raise argparse.ArgumentTypeError(f'range must be increasing, got {text!r}')
    return low, high


def parse_rsd_bin(text: str) -> dict:
    """Parse NAME:ZEFF:BIAS[:FIDUCIAL_F[:MAX_DF_FRACTION]]."""
    parts = text.split(':')
    if len(parts) not in (3, 4, 5):
        raise argparse.ArgumentTypeError(
            'RSD bin must be NAME:ZEFF:BIAS[:FIDUCIAL_F[:MAX_DF_FRACTION]], '
            f'got {text!r}'
        )
    name = parts[0].strip()
    if not name:
        raise argparse.ArgumentTypeError(f'RSD bin name is empty in {text!r}')
    try:
        zeff = float(parts[1])
        bias = float(parts[2])
        fiducial_f = float(parts[3]) if len(parts) >= 4 else 0.8
        max_df_fraction = float(parts[4]) if len(parts) == 5 else 0.1
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'Invalid numeric value in RSD bin {text!r}') from exc
    return {
        'name': name,
        'zeff': zeff,
        'bias': bias,
        'fiducial_f': fiducial_f,
        'max_df_fraction': max_df_fraction,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Create private desiblind catalog-level w0/wa blinding banks.')
    parser.add_argument('--effects', nargs='+', choices=VALID_EFFECTS, default=['bao'],
                        help='Which effect-specific banks to write. Default: bao.')
    parser.add_argument('--output', '--parameters-fn', type=Path, dest='output', default=None,
                        help='Backward-compatible alias for --bao-output when writing a BAO/AP bank.')
    parser.add_argument('--bao-output', type=Path, default=None,
                        help='Output BAO/AP .npy bank file, typically catalog_bao_blinding_parameters.npy.')
    parser.add_argument('--rsd-output', type=Path, default=None,
                        help='Output RSD .npy bank file, typically catalog_rsd_blinding_parameters.npy.')
    parser.add_argument('--bid', type=int, default=None,
                        help='Blinding ID. If omitted, use desiblind deterministic default.')
    parser.add_argument('--tracer-bins', '--bao-tracer-bins', nargs='+', default=list(DEFAULT_BAO_TRACER_BINS),
                        dest='bao_tracer_bins', help='Canonical BAO/AP tracer-bin names to write.')
    parser.add_argument('--rsd-bin', action='append', type=parse_rsd_bin, default=[],
                        help='RSD tracer metadata: NAME:ZEFF:BIAS[:FIDUCIAL_F[:MAX_DF_FRACTION]]. Repeat as needed.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Delete existing requested output bank(s) before writing. Without this, existing keys are refused.')
    parser.add_argument('--dry-run', action='store_true', help='Validate and print planned keys without writing any bank.')

    source = parser.add_argument_group('w0/wa source')
    source.add_argument('--generate', action='store_true',
                        help='Generate a native desiblind w0/wa draw; does not read the historical LSS bank.')
    source.add_argument('--seed', type=int, default=None,
                        help='Random seed for --generate. Required for reproducible closed banks.')
    source.add_argument('--w0-range', type=parse_range, default=(-1.2, -0.8),
                        help='Uniform generation range for w0, e.g. --w0-range=-1.2,-0.8.')
    source.add_argument('--wa-range', type=parse_range, default=(-0.8, 0.8),
                        help='Uniform generation range for wa, e.g. --wa-range=-0.8,0.8.')
    source.add_argument('--max-attempts', type=int, default=10000,
                        help='Maximum rejection-sampling attempts for --generate.')
    source.add_argument('--w0', type=float, default=None, help='Explicit w0 value.')
    source.add_argument('--wa', type=float, default=None, help='Explicit wa value.')
    source.add_argument('--lss-w0wa-bank', '--lss-parameters-fn', type=Path, default=Path(DEFAULT_LSS_W0WA_BANK),
                        dest='lss_parameters_fn', help='Historical LSS plain-text two-column w0/wa bank.')
    source.add_argument('--lss-index', '--lss-parameter-index', '--parameter-index', type=int, default=None,
                        dest='lss_index', help='Row index in the historical LSS w0/wa bank.')
    source.add_argument('--lss-filerow', '--filerow', type=Path, default=None,
                        help='File containing row index in the historical LSS w0/wa bank.')

    validation = parser.add_argument_group('alpha-shift validation')
    validation.add_argument('--no-validate-alpha-shift', action='store_true',
                            help='Disable the 3 percent alpha-shift validation. Use only for debugging.')
    validation.add_argument('--alpha-zrange', type=parse_zrange, default=(0.4, 2.1),
                            help='Validation redshift range, e.g. 0.4,2.1.')
    validation.add_argument('--max-alpha-shift', type=float, default=0.03,
                            help='Maximum allowed |alpha - 1|.')
    validation.add_argument('--alpha-nz', type=int, default=100,
                            help='Number of redshift samples for validation.')

    record = parser.add_argument_group('private record')
    record.add_argument('--record-fn', type=Path, default=None,
                        help='Optional private JSON record containing the chosen parameters and validation stats.')
    record.add_argument('--show-parameters', action='store_true',
                        help='Print w0/wa and alpha/RSD diagnostics to stdout. Avoid this in shared logs.')
    record.add_argument('--chmod', default=None,
                        help='Optional octal permissions for output and record, e.g. 600 or 660.')
    return parser.parse_args()


def determine_parameters(args):
    has_generate = bool(args.generate)
    has_explicit = args.w0 is not None or args.wa is not None
    has_lss = args.lss_index is not None or args.lss_filerow is not None
    selected = sum([has_generate, has_explicit, has_lss])
    if selected != 1:
        raise ValueError(
            'Choose exactly one w0/wa source: --generate, explicit --w0/--wa, '
            'or historical LSS --lss-index/--lss-filerow.'
        )
    if has_generate:
        return CatalogBAOBlinder.generate_parameters(
            seed=args.seed,
            w0_range=args.w0_range,
            wa_range=args.wa_range,
            max_attempts=args.max_attempts,
            validate_alpha_shift=not args.no_validate_alpha_shift,
            alpha_zrange=args.alpha_zrange,
            max_alpha_shift=args.max_alpha_shift,
            alpha_nz=args.alpha_nz,
        )
    if has_explicit:
        if args.w0 is None or args.wa is None:
            raise ValueError('Explicit source requires both --w0 and --wa.')
        return {'w0': float(args.w0), 'wa': float(args.wa)}, {'parameter_source': 'explicit'}
    loaded = CatalogBAOBlinder.load_lss_parameters(
        parameters_fn=args.lss_parameters_fn,
        index=args.lss_index,
        filerow=args.lss_filerow,
    )
    metadata = {
        'parameter_source': 'lss',
        'lss_parameters_fn': loaded.pop('parameters_fn'),
        'lss_index': loaded.pop('index'),
        'lss_filerow': None if args.lss_filerow is None else str(args.lss_filerow),
    }
    return {'w0': loaded['w0'], 'wa': loaded['wa']}, metadata


def maybe_chmod(path: Path, chmod: str | None):
    if chmod is None:
        return
    os.chmod(path, int(str(chmod), 8))


def resolve_outputs(args):
    effects = tuple(dict.fromkeys(args.effects))
    if args.output is not None and args.bao_output is not None and args.output != args.bao_output:
        raise ValueError('Use either --output or --bao-output for BAO/AP, not two different paths.')
    bao_output = args.bao_output or args.output
    if 'bao' in effects and bao_output is None:
        raise ValueError('Writing a BAO/AP bank requires --output or --bao-output.')
    if 'rsd' in effects and args.rsd_output is None:
        raise ValueError('Writing an RSD bank requires --rsd-output.')
    if 'rsd' in effects and not args.rsd_bin:
        raise ValueError('Writing an RSD bank requires at least one --rsd-bin NAME:ZEFF:BIAS entry.')
    return effects, {'bao': bao_output, 'rsd': args.rsd_output}


def key_preview(blinder, name: str, bid: int) -> str:
    return blinder.get_key(name, bid=bid)[:16]


def rsd_parameters_from_w0wa(parameters, rsd_bin):
    out = {
        'w0': parameters['w0'],
        'wa': parameters['wa'],
        'zeff': rsd_bin['zeff'],
        'bias': rsd_bin['bias'],
        'fiducial_f': rsd_bin['fiducial_f'],
        'max_df_fraction': rsd_bin['max_df_fraction'],
    }
    return CatalogRSDBlinder._normalize_parameters(out)


def remove_if_overwrite(path: Path, overwrite: bool):
    if path.exists() and overwrite:
        path.unlink()


def main():
    args = parse_args()
    effects, outputs = resolve_outputs(args)
    bid = CatalogBAOBlinder._get_bid() if args.bid is None else int(args.bid)
    parameters, source_metadata = determine_parameters(args)

    if args.no_validate_alpha_shift:
        alpha_stats = None
    else:
        alpha_stats = CatalogBAOBlinder.validate_alpha_shift(
            parameters,
            zrange=args.alpha_zrange,
            max_alpha_shift=args.max_alpha_shift,
            nz=args.alpha_nz,
        )

    rsd_rows = []
    if 'rsd' in effects:
        for item in args.rsd_bin:
            normalized = rsd_parameters_from_w0wa(parameters, item)
            rsd_rows.append({'name': item['name'], 'parameters': normalized})

    print('Catalog w0/wa blinding bank plan')
    print('================================')
    print(f'effects     : {" ".join(effects)}')
    if 'bao' in effects:
        print(f'bao output  : {outputs["bao"]}')
    if 'rsd' in effects:
        print(f'rsd output  : {outputs["rsd"]}')
    print(f'bid         : {bid}')
    print(f'source      : {source_metadata["parameter_source"]}')
    print(f'validate    : {not args.no_validate_alpha_shift}')
    if args.show_parameters:
        print(f'w0          : {parameters["w0"]}')
        print(f'wa          : {parameters["wa"]}')
        if alpha_stats is not None:
            print(f'max |apar-1|: {alpha_stats["max_abs_alpha_parallel_minus_one"]}')
            print(f'max |aperp-1|: {alpha_stats["max_abs_alpha_perp_minus_one"]}')

    if 'bao' in effects:
        print('BAO/AP hash key previews:')
        for name in args.bao_tracer_bins:
            print(f'  {name:6s} -> {key_preview(CatalogBAOBlinder, name, bid)}...')
    if 'rsd' in effects:
        print('RSD hash key previews:')
        for row in rsd_rows:
            name = row['name']
            params = row['parameters']
            if args.show_parameters:
                detail = f" zeff={params['zeff']} bias={params['bias']} fgrowth_blind={params['fgrowth_blind']}"
            else:
                detail = f" zeff={params['zeff']} bias={params['bias']}"
            print(f'  {name:6s} -> {key_preview(CatalogRSDBlinder, name, bid)}...{detail}')

    record = {
        'bid': bid,
        'effects': list(effects),
        'outputs': {key: None if value is None else str(value) for key, value in outputs.items() if key in effects},
        'parameters': parameters,
        'source': source_metadata,
        'alpha_validation': False if args.no_validate_alpha_shift else {
            'alpha_zrange': list(args.alpha_zrange),
            'max_alpha_shift': args.max_alpha_shift,
            'alpha_nz': args.alpha_nz,
            'max_abs_alpha_parallel_minus_one': None if alpha_stats is None else alpha_stats['max_abs_alpha_parallel_minus_one'],
            'max_abs_alpha_perp_minus_one': None if alpha_stats is None else alpha_stats['max_abs_alpha_perp_minus_one'],
        },
    }
    if 'bao' in effects:
        record['bao'] = {'tracer_bins': list(args.bao_tracer_bins)}
    if 'rsd' in effects:
        record['rsd'] = {
            'tracer_bins': [
                {'name': row['name'], 'parameters': row['parameters']} for row in rsd_rows
            ]
        }

    if args.dry_run:
        print('dry-run: no files written')
        if args.record_fn is not None:
            print(f'dry-run: record not written: {args.record_fn}')
        return

    if 'bao' in effects:
        path = outputs['bao']
        remove_if_overwrite(path, args.overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
        for name in args.bao_tracer_bins:
            CatalogBAOBlinder.write_blinded_parameters(name, parameters, parameters_fn=path, bid=bid, update=True)
        maybe_chmod(path, args.chmod)
        print(f'wrote BAO/AP bank: {path}')

    if 'rsd' in effects:
        path = outputs['rsd']
        remove_if_overwrite(path, args.overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
        for row in rsd_rows:
            CatalogRSDBlinder.write_blinded_parameters(
                row['name'], row['parameters'], parameters_fn=path, bid=bid, update=True
            )
        maybe_chmod(path, args.chmod)
        print(f'wrote RSD bank   : {path}')

    if args.record_fn is not None:
        args.record_fn.parent.mkdir(parents=True, exist_ok=True)
        args.record_fn.write_text(json.dumps(record, indent=2, sort_keys=True))
        maybe_chmod(args.record_fn, args.chmod)
        print(f'wrote record    : {args.record_fn}')


if __name__ == '__main__':
    main()
