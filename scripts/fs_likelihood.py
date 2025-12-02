from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate, REPTVelocileptorsTracerPowerSpectrumMultipoles
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.emulators import EmulatedCalculator
from desilike.samplers import EmceeSampler
from desilike.profilers import MinuitProfiler
from desilike.samples import Profiles
from desilike import setup_logging

import lsstypes as types

from pathlib import Path
import numpy as np


def get_tracer_label(tracer):
    return tracer.split('_')[0].replace('+', 'plus')

def get_synthetic_data(statistic='mesh2_spectrum_poles', tracer='LRG', zmin=0.4, zmax=0.6,
    region='GCcomb', ells=[0, 2, 4], weights='default_fkp', kmin=0.0, kmax=0.3, rebin=5):
    """
    Synthetic data from Abacus-HF mocks for testing.
    """
    
    dirname = Path('/global/cfs/projectdirs/desi/mocks/cai/mock-challenge-cutsky-dr2/unblinded_data/dr2-v2/')

    covariance = types.read(dirname / f'covariance_{statistic}_{tracer}_z{zmin}-{zmax}_{region}_{weights}.h5')
    observable = covariance.observable.select(k=slice(0, None, rebin)).select(k=(kmin, kmax)).get(ells=ells)

    covariance = covariance.at.observable.match(observable)

    window = types.read(dirname / f'window_{statistic}_{tracer}_z{zmin}-{zmax}_{region}_{weights}.h5')
    window = window.at.observable.match(observable)
    window = window.at.theory.select(k=(0, 0.35))

    return observable, covariance, window


def DESIFSLikelihood(tracers=None, cosmo=None, klim=(0.02, 0.2), solve='.auto'):
    """Create DESI FS likelihood."""

    if cosmo is None:
        cosmo = DirectPowerSpectrumTemplate(fiducial='DESI').cosmo
        cosmo.init.params['sigma8_m'] = {'derived': True, 'latex': '\sigma_8'}  # derive sigma_8
        cosmo.init.params['tau_reio'].update(fixed=True)
        cosmo.init.params['omega_b'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.02218, 'scale': (3.025e-7)**0.5})
        cosmo.init.update(engine='class')

    this_zrange = []
    for tracer, iz, zrange in list_zrange:
        tracer_label = get_tracer_label(tracer)
        namespace = '{tracer}_z{iz}'.format(tracer=tracer_label, iz=iz)
        if tracers is not None and namespace.lower() not in tracers: continue
        this_zrange.append((tracer, iz, zrange, namespace))

    observables = []
    likelihoods = []

    for tracer, iz, zrange, namespace in this_zrange:
        print('Adding DESI FS likelihood for tracer {}, zrange {}, namespace {}'.format(tracer, zrange, namespace))

        data, covariance, window = get_synthetic_data(
            tracer=tracer,
            zmin=zrange[0],
            zmax=zrange[1],
            region='GCcomb',
            ells=[0, 2, 4],
            weights='default_fkp',
            kmin=0.0,
            kmax=0.3,
            rebin=5
        )

        template = DirectPowerSpectrumTemplate(z=window.theory.get(ells=0).z, fiducial='DESI', cosmo=cosmo)
        theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, tracer=tracer_label, prior_basis='physical', freedom='max')
        observable = TracerPowerSpectrumMultipolesObservable(
            data=data,
            theory=theory,
            covariance=covariance,
            wmatrix=window,
        )

        # Compute or swap in PT emulator
        emu_fn = Path(f'emulator_fs_{namespace}.npy')

        if emu_fn.exists():
            calculator = EmulatedCalculator.load(emu_fn)
            # Update emulator with cosmo
            if cosmo is not None:
                for param in cosmo.init.params:
                    if param in calculator.init.params:
                        calculator.init.params.set(param)
            theory.init.update(pt=calculator)
        else:
            observable()  # to set up k-ranges for the emulator
            from desilike.emulators import Emulator, TaylorEmulatorEngine
            theory = observable.wmatrix.theory
            #theory.init.update(ells=(0, 2, 4))  # train emulator on all multipoles
            emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(method='finite', order=4))
            emulator.set_samples()
            emulator.fit()
            emulator.save(emu_fn)

        # Update namespace of bias parameters (to have one parameter per tracer / z-bin)
        for param in theory.init.params:
            # Update latex just to have better labels
            param.update(namespace=namespace,
                         latex=param.latex(namespace=r'\mathrm{{{}}}, {:d}'.format(tracer_label, iz), inline=False))
        

        likelihood = ObservablesGaussianLikelihood(observable, name=namespace)
        likelihoods.append(likelihood)

    #likelihood = sum(likelihoods)  # likelihood is a callable that returns the log-posterior
    if len(likelihoods) > 1: likelihood = sum(likelihoods)
    else: likelihood = likelihoods[0]  # to avoid duplicate loglikelihood derived parameter in cobaya when using multiple likelihoods independently

    if solve:
        for param in likelihood.all_params.select(basename=['alpha*', 'sn*', 'c*']):
            if param.varied: param.update(derived=solve)

        if likelihood.mpicomm.rank == 0:
            likelihood.log_info('Use analytic marginalization for {}.'.format(likelihood.all_params.names(solved=True)))


    return likelihood

def run_mcmc(likelihood):
    save_dir = '/global/cfs/cdirs/desicollab/users/epaillas/code/desiblind/scripts/chains/'
    save_fn = [Path(save_dir) / f'chain_abacushf_fs_{ichain:d}.npy' for ichain in range(4)]
    sampler = EmceeSampler(likelihood, nwalkers=64, save_fn=save_fn, seed=42)
    chains = sampler.run(min_iterations=200, check={'max_eigen_gr': 0.03})

def run_profiler(likelihood):
    save_dir = '/global/cfs/cdirs/desicollab/users/epaillas/code/desiblind/scripts/profiles/'
    save_fn = Path(save_dir) / f'profiles_abacushf_fs.npy'
    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize()
    print(profiles.to_stats(tablefmt='pretty'))
    profiles.save(save_fn)


if __name__ == '__main__':

    setup_logging()


    list_zrange = [
        ('BGS_BRIGHT-21.35', 0, (0.1, 0.4)), 
        ('LRG', 0, (0.4, 0.6)),
        ('LRG', 1, (0.6, 0.8)),
        ('LRG', 2, (0.8, 1.1)),
        ('ELG_LOPnotqso', 1, (1.1, 1.6)),
        ('QSO', 0, (0.8, 2.1)),
        # ('Lya', 0, (1.8, 4.2))
    ]

    tracers = [
        'bgs_z0',
        'lrg_z0',
        'lrg_z1',
        'lrg_z2',
        'elg_z1',
        'qso_z0'
    ]

    klim = (0.02, 0.2)

    likelihood = DESIFSLikelihood(
        tracers=tracers,
        cosmo=None,
        klim=klim,
        solve='.auto',
    )

    # run_mcmc(likelihood)
    run_profiler(likelihood)

    profiles = Profiles.load('profiles/profiles_abacushf_fs.npy')
    likelihood(**profiles.bestfit.choice(index='argmax', input=True))
    for tracer, likelihood in zip(tracers, likelihood.likelihoods):
        observable = likelihood.observables[0]
        observable.plot(fn=f'fig/plot_bestfit_{tracer}.png')