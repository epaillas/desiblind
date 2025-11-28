import itertools
from pathlib import Path
import jax

import lsstypes as types

from desilike import setup_logging


def compute_theory_for_covariance_mesh2_spectrum(output_fn, spectrum_fns, window_fn, klim=(0., 0.3)):
    import lsstypes as types
    from jaxpower import (ParticleField, MeshAttrs, compute_spectrum2_covariance)
    mean = types.mean([types.read(fn) for fn in spectrum_fns])
    window = types.read(window_fn)

    mattrs = MeshAttrs(**{name: mean.attrs[name] for name in ['boxsize', 'boxcenter', 'meshsize']})
    covariance = compute_spectrum2_covariance(mattrs, mean)

    sl = slice(0, None, 5)  # rebin to dk = 0.001 h/Mpc
    oklim = (0., 0.35)  # fitted k-range, no need to go to higher k
    smooth = mean.map(lambda pole: pole.clone(k=pole.coords('k', center='mid_if_edges'))).select(k=klim)
    mean = mean.select(k=sl).select(k=oklim)
    window = window.at.observable.select(k=sl).at.observable.select(k=oklim).at.theory.select(k=(0., 1.1 * oklim[1]))
    covariance = covariance.at.observable.select(k=sl).at.observable.select(k=oklim)

    from desilike.theories.galaxy_clustering import FixedPowerSpectrumTemplate, REPTVelocileptorsTracerPowerSpectrumMultipoles
    from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
    from desilike.likelihoods import ObservablesGaussianLikelihood
    from desilike.profilers import MinuitProfiler

    template = FixedPowerSpectrumTemplate(fiducial='DESI', z=window.theory.get(ells=0).z)
    theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template)
    observable = TracerPowerSpectrumMultipolesObservable(data=mean.value(concatenate=True), wmatrix=window.value(), ells=mean.ells, k=[pole.coords('k') for pole in mean], kin=window.theory.get(ells=0).coords('k'), ellsin=window.theory.ells, theory=theory)
    likelihood = ObservablesGaussianLikelihood(observable, covariance=covariance.value())

    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize()
    profiles.save('profiles.npy')
    print(profiles.bestfit.choice(index='argmax', input=True))
    # theory.init.update(k=smooth.get(0).coords('k'))
    # poles = theory(**profiles.bestfit.choice(index='argmax', input=True))
    # smooth = smooth.clone(value=poles.ravel())
    # if output_fn is not None and jax.process_index() == 0:
    #     smooth.write(output_fn)
    return smooth

def get_measurement_fn(kind='mesh2_spectrum_poles', version='dr2-v2', recon=None, tracer='LRG', region='NGC', zrange=(0.8, 1.1), cut=None, auw=None, weight_type='default', **kwargs):
    base_dir = Path(f'/global/cfs/projectdirs/desi/mocks/cai/mock-challenge-cutsky-dr2/')
    base_dir = base_dir / (f'unblinded_data_{recon}' if recon else 'unblinded_data')
    if cut: cut = '_thetacut'
    else: cut = ''
    if auw: auw = '_auw'
    else: auw = ''
    return str(base_dir / f'{version}/{kind}_{tracer}_z{zrange[0]:.1f}-{zrange[1]:.1f}_{region}_{weight_type}{auw}{cut}.h5')

def get_cubic_measurement_fn(kind='mesh2_spectrum_poles', version='abacus-2ndgen', recon=None, tracer='LRG', zsnap=0.9, flavor='', imock=0, los='z', **kwargs):
    if imock is None or los is None:
        import glob
        return sorted(glob.glob(get_cubic_measurement_fn(kind=kind, version=version, recon=recon, tracer=tracer, zsnap=zsnap, flavor=flavor, imock='*' if imock is None else imock, los='*' if los is None else los)))
    base_dir = Path(f'/global/cfs/projectdirs/desi/mocks/cai/mock-benchmark-dr2/summary_statistics')
    base_dir = base_dir / (f'cubic_{recon}' if recon else 'cubic')
    if flavor: flavor = '_' + flavor
    base_name = f'{version}/{kind}_{tracer}{flavor}_z{zsnap:.3f}_los-{los}'
    base_name = f'{base_name}_{imock}.h5' if 'window' not in kind else f'{base_name}.h5'
    return str(base_dir / base_name)

def get_zsnap_from_z(tracer, z, version='abacus-2ndgen'):
    """Return zsnapshot from redshift."""
    import numpy as np
    tracer = tracer.upper()[:3]
    zrange = {}
    if version in ['abacus-2ndgen', 'ezmock-dr1', 'poisson', 'gaussian']:
        if tracer == 'BGS':
            zrange[0.200] = (0.1, 0.4)
        elif tracer == 'LRG':
            zrange[0.500] = (0.4, 0.6)
            zrange[0.800] = (0.6, 1.1)
        elif tracer == 'ELG':
            zrange[0.950] = (0.8, 1.1)
            zrange[1.325] = (1.1, 1.6)
        elif tracer == 'QSO':
            zrange[1.400] = (0.8, 2.1)
        else:
            raise ValueError('unknown tracer {}'.format(tracer))
    if version == 'abacus-hf':
        if tracer == 'BGS':
            zrange[0.300] = (0.1, 0.4)
        elif tracer == 'LRG':
            zrange[0.500] = (0.4, 0.6)
            zrange[0.725] = (0.6, 0.8)
            zrange[0.950] = (0.8, 1.1)
        elif tracer == 'ELG':
            zrange[0.950] = (0.8, 1.1)
            zrange[1.475] = (1.1, 1.6)
        elif tracer == 'QSO':
            zrange[1.400] = (0.8, 2.1)
        else:
            raise ValueError('unknown tracer {}'.format(tracer))
    if version == 'abacus-hf-v2':
        if tracer == 'BGS':
            zrange[0.300] = (0.1, 0.4)
        elif tracer == 'LRG':
            zrange[0.500] = (0.4, 0.6)
            zrange[0.725] = (0.6, 0.8)
            zrange[0.950] = (0.8, 1.1)
        elif tracer == 'ELG':
            zrange[0.950] = (0.8, 1.1)
            zrange[1.475] = (1.1, 1.6)
        elif tracer == 'QSO':
            zrange[1.550] = (0.8, 2.1)
        else:
            raise ValueError('unknown tracer {}'.format(tracer))
    if version == 'uchuu-hf':
        if tracer == 'BGS':
            zrange[0.190] = (0.1, 0.4)
        elif tracer == 'LRG':
            zrange[0.490] = (0.4, 0.6)
            zrange[0.700] = (0.6, 0.8)
            zrange[0.940] = (0.8, 1.1)
        elif tracer == 'ELG':
            zrange[0.940] = (0.8, 1.1)
            zrange[1.430] = (1.1, 1.6)
        elif tracer == 'QSO':
            zrange[1.430] = (0.8, 2.1)
        else:
            raise ValueError('unknown tracer {}'.format(tracer))
    z = np.array(z)
    for zsnap, zrange in zrange.items():
        if np.all((z >= zrange[0]) & (z <= zrange[1])):
            return zsnap
    raise ValueError(f'input z not found in any snapshot {z}')


if __name__ == '__main__':

    setup_logging()

    tracers = [
        # ('BGS', (0.1, 0.4)),
        ('LRG', (0.4, 0.6)),
        # ('LRG', (0.6, 0.8)),
        # ('LRG', (0.8, 1.1)),
        # ('ELG_LOPnotqso', (0.8, 1.1)),
        # ('ELG_LOPnotqso', (1.1, 1.6)), 
        # ('QSO', (0.8, 2.1))
    ]
    regions = ['GCcomb']
    versions = ['dr2-v2']
    weight_types = ['default_fkp']

    for (tracer, zrange), region, version, weight_type in itertools.product(tracers, regions, versions, weight_types):
        if 'BGS' in tracer:
            tracer = 'BGS_BRIGHT-21.5' if 'dr1' in version else 'BGS_BRIGHT-21.35'
        catalog_args = dict(version=version, region=region, tracer=tracer, zrange=zrange, weight_type=weight_type)

        window_fn = get_measurement_fn(**catalog_args, kind='window_mesh2_spectrum_poles')
        window = types.read(window_fn)
        kmax = window.observable.get(0).edges('k').max()
        cubic_tracer = {'BGS': 'BGS-21.35', 'ELG': 'ELG_LOP'}.get(tracer[:3], tracer[:3])
        version = 'abacus-hf-v2'
        zsnap = get_zsnap_from_z(cubic_tracer, zrange, version=version)
        flavor = {'ELG': 'base_conf_nfwexp'}.get(tracer[:3], 'base')
        cubic_catalog_args = {'version': version, 'tracer': cubic_tracer, 'zsnap': zsnap, 'flavor': flavor, 'los': 'z'}
        spectrum_fns = get_cubic_measurement_fn(imock=None, **cubic_catalog_args, kind='mesh2_spectrum_poles')
        window_fn = get_cubic_measurement_fn(**cubic_catalog_args, kind='window_mesh2_spectrum_poles')
        output_fn = 'profiles.npy'
        compute_theory_for_covariance_mesh2_spectrum(output_fn, spectrum_fns, window_fn, klim=(0., kmax))