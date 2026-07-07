"""Validate CatalogFNLBlinder against the core LSS fNL blinding call.

The LSS fNL scripts implement the physical fNL operation through
``mockfactory.blinding.CutskyCatalogBlinding.png`` with ``method='data_weights'``
and ``shotnoise_correction=True``. This script runs that LSS-style call directly
and compares it to ``desiblind.CatalogFNLBlinder`` on the same synthetic cutsky
catalogs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mockfactory import Catalog
from mockfactory.blinding import CutskyCatalogBlinding, get_cosmo_blind

from desiblind import CatalogFNLBlinder


def make_catalogs(seed=42, ndata=80, nrandom=400):
    rng = np.random.default_rng(seed)
    data = Catalog({
        'RA': rng.uniform(150., 170., ndata),
        'DEC': rng.uniform(5., 25., ndata),
        'Z': rng.uniform(0.65, 0.85, ndata),
        'WEIGHT': rng.uniform(0.8, 1.2, ndata),
        'WEIGHT_COMP': rng.uniform(0.9, 1.1, ndata),
    })
    randoms = Catalog({
        'RA': rng.uniform(150., 170., nrandom),
        'DEC': rng.uniform(5., 25., nrandom),
        'Z': rng.uniform(0.65, 0.85, nrandom),
        'WEIGHT': np.ones(nrandom),
    })
    return data, randoms


def positions(catalog):
    return [np.asarray(catalog[col], dtype='f8') for col in ['RA', 'DEC', 'Z']]


def lss_core_apply_fnl(data, randoms, *, fnl, zeff, bias, nmesh=16):
    """Core fNL block from LSS scripts, without file IO."""
    cosmo_blind = get_cosmo_blind('DESI', z=zeff)
    cosmo_blind._derived['fnl'] = float(fnl)
    blinding = CutskyCatalogBlinding(
        cosmo_fid='DESI', cosmo_blind=cosmo_blind, bias=float(bias), z=float(zeff),
        position_type='rdz', mpicomm=None, mpiroot=0,
    )
    new_data_weights = blinding.png(
        positions(data), data_weights=np.asarray(data['WEIGHT'], dtype='f8'),
        randoms_positions=positions(randoms), randoms_weights=np.asarray(randoms['WEIGHT'], dtype='f8'),
        method='data_weights', shotnoise_correction=True, nmesh=int(nmesh), smoothing_radius=30.,
    )
    out = data.copy()
    factor = np.asarray(new_data_weights, dtype='f8') / np.asarray(data['WEIGHT'], dtype='f8')
    out['WEIGHT'] = new_data_weights
    out['WEIGHT_COMP'] = np.asarray(out['WEIGHT_COMP'], dtype='f8') * factor
    return out, factor


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=None, help='Optional JSON summary path.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ndata', type=int, default=80)
    parser.add_argument('--nrandom', type=int, default=400)
    parser.add_argument('--fnl', type=float, default=5.)
    parser.add_argument('--zeff', type=float, default=0.8)
    parser.add_argument('--bias', type=float, default=2.0)
    parser.add_argument('--nmesh', type=int, default=16)
    ns = parser.parse_args(args=args)

    data, randoms = make_catalogs(seed=ns.seed, ndata=ns.ndata, nrandom=ns.nrandom)
    params = {'fnl': ns.fnl, 'zeff': ns.zeff, 'bias': ns.bias}
    lss_out, lss_factor = lss_core_apply_fnl(data, randoms, nmesh=ns.nmesh, **params)
    desiblind_out, desiblind_factor = CatalogFNLBlinder.apply_blinding(
        'LRG3', data, randoms, parameters=params, return_weight_factor=True,
        nmesh=ns.nmesh, smoothing_radius=30.,
    )

    summary = {
        'passed': True,
        'parameters': params,
        'seed': ns.seed,
        'ndata': ns.ndata,
        'nrandom': ns.nrandom,
        'nmesh': ns.nmesh,
        'max_abs_delta_weight': float(np.max(np.abs(np.asarray(desiblind_out['WEIGHT']) - np.asarray(lss_out['WEIGHT'])))),
        'max_abs_delta_weight_comp': float(np.max(np.abs(np.asarray(desiblind_out['WEIGHT_COMP']) - np.asarray(lss_out['WEIGHT_COMP'])))),
        'max_abs_delta_factor': float(np.max(np.abs(np.asarray(desiblind_factor) - np.asarray(lss_factor)))),
        'factor_min': float(np.min(desiblind_factor)),
        'factor_mean': float(np.mean(desiblind_factor)),
        'factor_max': float(np.max(desiblind_factor)),
    }
    tolerance = 1e-12
    summary['passed'] = (
        summary['max_abs_delta_weight'] <= tolerance and
        summary['max_abs_delta_weight_comp'] <= tolerance and
        summary['max_abs_delta_factor'] <= tolerance
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if ns.output is not None:
        output = Path(ns.output).expanduser().resolve(strict=False)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + '\n')
    if not summary['passed']:
        raise SystemExit('CatalogFNLBlinder does not match LSS core fNL call')
    return summary


if __name__ == '__main__':
    main()
