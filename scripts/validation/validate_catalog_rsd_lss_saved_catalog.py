"""Validate the saved-catalog RSD blinding ladder against LSS.

This validation extends the BAO/AP saved-catalog ladder through clustering
catalog production, GC splitting, RSD reconstruction, and the final RSD redshift
shift. It compares two branches:

- LSS branch:       LSS BAO/AP shifter + LSS.apply_zshift_RSD
- desiblind branch: CatalogBAOBlinder + CatalogRSDBlinder

Both branches use the same LSS downstream machinery for n(z), mkclusdat,
mkclusran, GC splitting, and reconstruction. The validation target is that the
final RSD-blinded clustering data catalogs match column-by-column.

This is a validation driver, not a package unit test. Run it on a NERSC CPU node
for anything beyond tiny samples.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
# CPU-only validation; avoid noisy JAX CUDA plugin initialization on CPU nodes.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
from pathlib import Path
import sys

import fitsio
import numpy as np
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from desiblind import CatalogRSDBlinder

import validate_catalog_bao_lss_mkclusdat as mkclusdat_validation
import validate_catalog_bao_lss_mkclusran as mkclusran_validation
import validate_catalog_bao_lss_saved_catalog as saved
import validate_catalog_bao_lss_split_gc as split_gc_validation


def parse_args():
    parser = argparse.ArgumentParser(
        description='Validate desiblind RSD saved-catalog workflow against LSS after BAO/AP, mkclus, split-GC, and reconstruction.'
    )
    parser.add_argument('--lss-repo', default=os.environ.get('LSS_REPO', saved.DEFAULT_LSS_REPO))
    parser.add_argument('--output-dir', default=None,
                        help='Directory for scratch outputs. Default: $SCRATCH/desiblind_lss_validation/rsd-saved-catalog-<timestamp>-<pid>.')
    parser.add_argument('--real-catalog', default=saved.DEFAULT_REAL_CATALOG)
    parser.add_argument('--random-catalog', default=saved.DEFAULT_RANDOM_CATALOG,
                        help='Full random catalog used as the mkclusran input sample.')
    parser.add_argument('--nz-random-catalog', default=None,
                        help='Random catalog used for LSS mknz_full area normalization. Defaults to --random-catalog.')
    parser.add_argument('--nrows', type=int, default=20000)
    parser.add_argument('--random-nrows', type=int, default=50000)
    parser.add_argument('--rannum', type=int, default=1)
    parser.add_argument('--w0', type=float, default=saved.DEFAULT_PARAMETERS['w0'])
    parser.add_argument('--wa', type=float, default=saved.DEFAULT_PARAMETERS['wa'])
    parser.add_argument('--zeff', type=float, default=0.8)
    parser.add_argument('--bias', type=float, default=None,
                        help='Bias for fgrowth_blind and reconstruction. Default: LSS.recon_tools.get_f_bias(tracer)[1].')
    parser.add_argument('--fiducial-f', type=float, default=None,
                        help='Fiducial growth rate. Default: LSS.recon_tools.get_f_bias(tracer)[0].')
    parser.add_argument('--max-df-fraction', type=float, default=0.1)
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
    parser.add_argument('--regions', nargs='+', default=['NGC', 'SGC'], choices=['NGC', 'SGC'],
                        help='Galactic caps to run through reconstruction/RSD validation.')
    parser.add_argument('--reconstruction', choices=['iterative_fft', 'multigrid'], default='iterative_fft')
    parser.add_argument('--lss-reconstruction-defaults', action='store_true',
                        help='Do not pass explicit nmesh/boxsize to LSS.recon_tools.run_reconstruction; let pyrecon choose the mesh from the catalog and cellsize, matching the LSS script style.')
    parser.add_argument('--nmesh', type=int, default=64)
    parser.add_argument('--boxsize', type=float, default=6000.,
                        help='Explicit Cartesian reconstruction box size. For validation subsets this prevents tiny nmesh*cellsize boxes.')
    parser.add_argument('--cellsize', type=float, default=7.)
    parser.add_argument('--smoothing-radius', type=float, default=15.)
    parser.add_argument('--nthreads', type=int, default=8)
    return parser.parse_args()


def make_output_dir(output_dir=None):
    if output_dir is None:
        base = Path(os.environ.get('SCRATCH', '.')) / 'desiblind_lss_validation'
        stamp = datetime.now(timezone.utc).strftime('rsd-saved-catalog-%Y%m%d-%H%M%S')
        output_dir = base / f'{stamp}-{os.getpid()}'
    output_dir = Path(output_dir).expanduser().resolve(strict=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def add_lss_modules(lss_repo):
    lss_blind, common, lss_py = saved.add_lss_to_path(lss_repo)
    try:
        import LSS.main.cattools as ct
        import LSS.recon_tools as rectools
    except ImportError as exc:
        raise RuntimeError(f'Could not import LSS cattools/recon_tools from {lss_py}') from exc
    return lss_blind, common, ct, rectools, lss_py


def get_reconstruction_class(name):
    if name == 'iterative_fft':
        from pyrecon import IterativeFFTReconstruction
        return IterativeFFTReconstruction, 'IFFTrsd'
    from pyrecon import MultiGridReconstruction
    return MultiGridReconstruction, 'MGrsd'


def run_saved_bao_split_ladder(args, output_dir, lss_blind, common, ct, tracer_type, zmin, zmax, dz, p0):
    params_bao = {'w0': args.w0, 'wa': args.wa}
    random_prefix, random_sample_fn, random_rows = mkclusran_validation.write_random_sample(
        common, args.random_catalog, output_dir, args.rannum, args.use_map_veto, args.random_nrows
    )
    sample, sample_fn = mkclusran_validation.read_balanced_sample(args.real_catalog, args.nrows, args.input_zcol)
    nz_random_catalog = args.nz_random_catalog or args.random_catalog
    random_fn_for_nz = Path(nz_random_catalog).expanduser().resolve(strict=False)
    if not random_fn_for_nz.exists():
        raise FileNotFoundError(random_fn_for_nz)

    lss_result = saved.run_saved_workflow(
        'lss', sample, common, lss_blind, output_dir, random_fn_for_nz, params_bao, args.tracer_name,
        tracer_type, dz, zmin, zmax, args.randens, p0, args.input_zcol, args.output_zcol,
        args.clip_min, args.clip_max, wsyscol=args.wsyscol, skip_add_fkp=args.skip_add_fkp,
    )
    desiblind_result = saved.run_saved_workflow(
        'desiblind', sample, common, lss_blind, output_dir, random_fn_for_nz, params_bao, args.tracer_name,
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
    lss_data_ngc, lss_data_sgc = split_gc_validation.split_gc_file(common, lss_clus_fn, output_dir / 'lss_', data_app)
    desi_data_ngc, desi_data_sgc = split_gc_validation.split_gc_file(common, desiblind_clus_fn, output_dir / 'desiblind_', data_app)
    lss_ran_ngc, lss_ran_sgc = split_gc_validation.split_gc_file(common, lss_random_fn, output_dir / 'lss_', random_app)
    desi_ran_ngc, desi_ran_sgc = split_gc_validation.split_gc_file(common, desiblind_random_fn, output_dir / 'desiblind_', random_app)

    split_files = {
        'NGC': {
            'lss_data': lss_data_ngc,
            'desiblind_data': desi_data_ngc,
            'lss_random': lss_ran_ngc,
            'desiblind_random': desi_ran_ngc,
        },
        'SGC': {
            'lss_data': lss_data_sgc,
            'desiblind_data': desi_data_sgc,
            'lss_random': lss_ran_sgc,
            'desiblind_random': desi_ran_sgc,
        },
    }
    split_comparison = {
        'data_NGC': mkclusdat_validation.compare_tables(lss_data_ngc, desi_data_ngc),
        'data_SGC': mkclusdat_validation.compare_tables(lss_data_sgc, desi_data_sgc),
        'random_NGC': mkclusdat_validation.compare_tables(lss_ran_ngc, desi_ran_ngc),
        'random_SGC': mkclusdat_validation.compare_tables(lss_ran_sgc, desi_ran_sgc),
    }

    return {
        'params_bao': params_bao,
        'random_prefix': str(random_prefix),
        'random_sample_fn': str(random_sample_fn),
        'random_rows': random_rows,
        'sample_fn': str(sample_fn),
        'random_fn_for_nz': str(random_fn_for_nz),
        'lss_result': lss_result,
        'desiblind_result': desiblind_result,
        'saved_comparison': saved_comparison,
        'clustering_data_comparison': clustering_data_comparison,
        'clustering_random_comparison': clustering_random_comparison,
        'split_comparison': split_comparison,
        'split_files': split_files,
        'dchi2': dchi2,
        'files': {
            'lss_full': lss_result['saved_full_fn'],
            'desiblind_full': desiblind_result['saved_full_fn'],
            'lss_clustering_data': str(lss_clus_fn),
            'desiblind_clustering_data': str(desiblind_clus_fn),
            'lss_clustering_random': str(lss_random_fn),
            'desiblind_clustering_random': str(desiblind_random_fn),
        },
    }


def run_reconstruction(rectools, reconstruction_cls, data_fn, random_fn, data_rec_fn, random_rec_fn,
                       fiducial_f, bias, zmin, zmax, args):
    from LSS.tabulated_cosmo import TabulatedDESI
    from pyrecon import setup_logging

    setup_logging()
    distance = TabulatedDESI().comoving_radial_distance
    reconstruction_kwargs = {
        'cellsize': args.cellsize,
        'smoothing_radius': args.smoothing_radius,
        'nthreads': args.nthreads,
    }
    if not args.lss_reconstruction_defaults:
        reconstruction_kwargs.update({'boxsize': args.boxsize, 'nmesh': args.nmesh})

    rectools.run_reconstruction(
        reconstruction_cls,
        distance,
        str(data_fn),
        str(random_fn),
        str(data_rec_fn),
        str(random_rec_fn),
        f=fiducial_f,
        bias=bias,
        convention='rsd',
        dtype='f8',
        zlim=(zmin, zmax),
        **reconstruction_kwargs,
    )
    if not Path(data_rec_fn).exists():
        raise FileNotFoundError(data_rec_fn)
    return Path(data_rec_fn)


def apply_lss_rsd(lss_blind, data_fn, realspace_fn, output_fn, params_rsd):
    data = Table(fitsio.read(str(data_fn), ext='LSS'))
    data_real = Table(fitsio.read(str(realspace_fn), ext='LSS'))
    lss_blind.apply_zshift_RSD(
        data, data_real, str(output_fn),
        fgrowth_fid=params_rsd['fiducial_f'], fgrowth_blind=params_rsd['fgrowth_blind'], zcol='Z',
    )
    return Path(output_fn)


def apply_desiblind_rsd(data_fn, realspace_fn, output_fn, tracer_name, params_rsd, common):
    data = Table(fitsio.read(str(data_fn), ext='LSS'))
    data_real = Table(fitsio.read(str(realspace_fn), ext='LSS'))
    blinded = CatalogRSDBlinder.apply_blinding(
        tracer_name, data, data_real, parameters=params_rsd, zcol='Z', output_zcol='Z'
    )
    saved.write_lss(common, blinded, output_fn)
    return Path(output_fn)


def compare_final_rsd(lss_fn, desiblind_fn):
    return mkclusdat_validation.compare_tables(lss_fn, desiblind_fn)


def run_region_rsd(region, split_files, lss_blind, common, rectools, reconstruction_cls, rec_label,
                   tracer_type, params_rsd, zmin, zmax, args, output_dir):
    files = split_files[region]
    branch_results = {}
    for branch in ['lss', 'desiblind']:
        data_fn = Path(files[f'{branch}_data'])
        random_fn = Path(files[f'{branch}_random'])
        data_rec_fn = output_dir / f'{branch}_{tracer_type}_{region}_clustering.{rec_label}.dat.fits'
        random_rec_fn = output_dir / f'{branch}_{tracer_type}_{region}_{args.rannum}_clustering.{rec_label}.ran.fits'
        run_reconstruction(
            rectools, reconstruction_cls, data_fn, random_fn, data_rec_fn, random_rec_fn,
            params_rsd['fiducial_f'], params_rsd['bias'], zmin, zmax, args,
        )
        if branch == 'lss':
            output_fn = data_fn
            apply_lss_rsd(lss_blind, data_fn, data_rec_fn, output_fn, params_rsd)
        else:
            output_fn = data_fn
            apply_desiblind_rsd(data_fn, data_rec_fn, output_fn, args.tracer_name, params_rsd, common)
        branch_results[branch] = {
            'input_data': str(data_fn),
            'input_random': str(random_fn),
            'realspace_reconstruction': str(data_rec_fn),
            'rsd_blinded_data': str(output_fn),
        }
    comparison = compare_final_rsd(branch_results['lss']['rsd_blinded_data'], branch_results['desiblind']['rsd_blinded_data'])
    return {
        'branches': branch_results,
        'comparison': comparison,
    }


def main():
    args = parse_args()
    if args.output_zcol != 'Z':
        raise ValueError('LSS blinding functions write transformed redshifts to Z; keep --output-zcol=Z.')

    tracer_type = saved.simple_tracer(args.tracer_type)
    zmin, zmax = (args.zmin, args.zmax)
    if zmin is None or zmax is None:
        default_zmin, default_zmax = saved.ZRANGE_BY_TRACER[tracer_type]
        zmin = default_zmin if zmin is None else zmin
        zmax = default_zmax if zmax is None else zmax
    dz = args.dz if args.dz is not None else saved.DZ_BY_TRACER.get(tracer_type, 0.01)
    p0 = args.p0 if args.p0 is not None else saved.P0_BY_TRACER[tracer_type]

    lss_blind, common, ct, rectools, lss_py = add_lss_modules(args.lss_repo)
    default_f, default_bias = rectools.get_f_bias(tracer_type)
    fiducial_f = float(args.fiducial_f if args.fiducial_f is not None else default_f)
    bias = float(args.bias if args.bias is not None else default_bias)
    params_rsd = {
        'w0': args.w0,
        'wa': args.wa,
        'zeff': args.zeff,
        'bias': bias,
        'fiducial_f': fiducial_f,
        'max_df_fraction': args.max_df_fraction,
    }
    params_rsd['fgrowth_blind'] = CatalogRSDBlinder.compute_fgrowth_blind(**params_rsd)

    reconstruction_cls, rec_label = get_reconstruction_class(args.reconstruction)
    output_dir = make_output_dir(args.output_dir)

    ladder = run_saved_bao_split_ladder(args, output_dir, lss_blind, common, ct, tracer_type, zmin, zmax, dz, p0)

    rsd_region_results = {}
    for region in args.regions:
        rsd_region_results[region] = run_region_rsd(
            region, ladder['split_files'], lss_blind, common, rectools, reconstruction_cls, rec_label,
            tracer_type, params_rsd, zmin, zmax, args, output_dir,
        )

    summary = {
        'validation': 'catalog_rsd_lss_saved_catalog',
        'status': 'PASS',
        'parameters_bao': ladder['params_bao'],
        'parameters_rsd': params_rsd,
        'tracer_name': args.tracer_name,
        'tracer_type': tracer_type,
        'lss_py': str(lss_py),
        'real_catalog': ladder['sample_fn'],
        'random_catalog_for_nz': ladder['random_fn_for_nz'],
        'random_catalog_for_mkclusran': str(Path(args.random_catalog).expanduser().resolve(strict=False)),
        'random_sample_file': ladder['random_sample_fn'],
        'random_rows': ladder['random_rows'],
        'output_dir': str(output_dir),
        'nrows': args.nrows,
        'zrange': [zmin, zmax],
        'dz': dz,
        'p0': p0,
        'dchi2': ladder['dchi2'],
        'compmd': args.compmd,
        'rannum': args.rannum,
        'reconstruction': {
            'method': args.reconstruction,
            'label': rec_label,
            'convention': 'rsd',
            'lss_reconstruction_defaults': args.lss_reconstruction_defaults,
            'nmesh': None if args.lss_reconstruction_defaults else args.nmesh,
            'boxsize': None if args.lss_reconstruction_defaults else args.boxsize,
            'cellsize': args.cellsize,
            'smoothing_radius': args.smoothing_radius,
            'nthreads': args.nthreads,
        },
        'pre_rsd': {
            'saved_catalog_comparison': ladder['saved_comparison'],
            'clustering_data_comparison': ladder['clustering_data_comparison'],
            'clustering_random_comparison': ladder['clustering_random_comparison'],
            'split_gc_comparison': ladder['split_comparison'],
        },
        'rsd_regions': rsd_region_results,
        'files': ladder['files'],
    }
    summary_fn = output_dir / 'summary.json'
    summary_fn.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print('catalog_rsd_lss_saved_catalog_validation=PASS')
    print(f'fgrowth_blind={params_rsd["fgrowth_blind"]}')
    print(f'output_dir={output_dir}')
    print(f'summary={summary_fn}')
    for region, result in rsd_region_results.items():
        comparison = result['comparison']
        print(
            f"{region}: rows={comparison['rows']} "
            f"max_abs_delta_Z={comparison.get('max_abs_delta_Z', 'NA')} "
            f"max_abs_delta_WEIGHT={comparison.get('max_abs_delta_WEIGHT', 'NA')} "
            f"max_abs_delta_WEIGHT_SYS={comparison.get('max_abs_delta_WEIGHT_SYS', 'NA')} "
            f"max_abs_delta_WEIGHT_FKP={comparison.get('max_abs_delta_WEIGHT_FKP', 'NA')}"
        )


if __name__ == '__main__':
    main()
