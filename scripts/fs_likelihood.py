from pathlib import Path
import numpy as np
import argparse

from desilike.theories.galaxy_clustering import DirectPowerSpectrumTemplate
from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable
from desilike.likelihoods import ObservablesGaussianLikelihood
from desilike.emulators import EmulatedCalculator
from desilike.samplers import EmceeSampler, MCMCSampler
from desilike.profilers import MinuitProfiler
from desilike.samples import Profiles
from desilike import setup_logging

import lsstypes as types


REPO_ROOT = Path(__file__).resolve().parents[1]
data_dir = REPO_ROOT / 'data'
output_dir = data_dir / 'fs_likelihood'
emulator_dir = output_dir / 'emulators' / 'ns'
fig_dir = REPO_ROOT / 'fig'
DEFAULT_MEASUREMENTS_DIR = Path('/global/cfs/projectdirs/desi/mocks/cai/mock-challenge-cutsky-dr2/unblinded_data/dr2-v2')


def get_tracer_zrange(name):
    list_zrange = {'BGS_z0': ('BGS_BRIGHT-21.35', (0.1, 0.4)), 
                   'LRG_z0': ('LRG', (0.4, 0.6)),
                   'LRG_z1': ('LRG', (0.6, 0.8)),
                   'LRG_z2': ('LRG', (0.8, 1.1)),
                   'ELG_z1': ('ELG_LOPnotqso', (1.1, 1.6)),
                   'QSO_z0': ('QSO', (0.8, 2.1))}
    if name is None:
        return list_zrange
    return list_zrange[name]


def get_synthetic_data(statistic='mesh2_spectrum_poles', tracer='LRG', zrange=(0.4, 0.6),
    region='GCcomb', ells=[0, 2, 4], weights='default_fkp', klim=(0., 0.3), rebin=5,
    measurements_dir=DEFAULT_MEASUREMENTS_DIR):
    """
    Synthetic data from Abacus-HF mocks for testing.
    """
    dirname = Path(measurements_dir)

    zmin, zmax = zrange
    covariance = types.read(dirname / f'covariance_{statistic}_{tracer}_z{zmin}-{zmax}_{region}_{weights}.h5')
    observable = covariance.observable.select(k=slice(0, None, rebin)).select(k=klim).get(ells=ells)

    covariance = covariance.at.observable.match(observable)

    window = types.read(dirname / f'window_{statistic}_{tracer}_z{zmin}-{zmax}_{region}_{weights}.h5')
    window = window.at.observable.match(observable)
    window = window.at.theory.select(k=(0, 0.35))

    return observable, covariance, window


def get_theory(cosmo=None, z=1., tracer=None, theory_model='reptvelocileptors'):
    """Return a DESI full-shape theory model.

    Parameters
    ----------
    theory_model : str, default='reptvelocileptors'
        Perturbation theory backend. Supported values match desi-clustering:
        ``'reptvelocileptors'`` and ``'folpsD'``.
    """
    from desilike.theories.galaxy_clustering import (
        DirectPowerSpectrumTemplate,
        FOLPSv2TracerPowerSpectrumMultipoles,
        REPTVelocileptorsTracerPowerSpectrumMultipoles,
    )
    from desilike.theories import Cosmoprimo

    if cosmo is None:
        cosmo = DirectPowerSpectrumTemplate(fiducial='DESI').cosmo
        cosmo.init.params['sigma8_m'] = {'derived': True, 'latex': r'\sigma_8'}  # derive sigma_8
        cosmo.init.params['tau_reio'].update(fixed=True)
        cosmo.init.params['omega_b'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.02218, 'scale': (3.025e-7)**0.5})
        cosmo.init.params['n_s'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.042})
        cosmo.init.update(engine='class')

    template = DirectPowerSpectrumTemplate(z=z, fiducial='DESI', cosmo=cosmo)
    tracer_label = tracer[:3].upper()
    if theory_model == 'reptvelocileptors':
        theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(
            template=template,
            tracer=tracer_label,
            prior_basis='physical',
            freedom='max',
        )
    elif theory_model == 'folpsD':
        theory = FOLPSv2TracerPowerSpectrumMultipoles(
            template=template,
            tracer=tracer_label,
            damping='lor',
            prior_basis='physical_aap',
            b3_coev=True,
            A_full=False,
        )
    else:
        raise ValueError(f'Unsupported theory model {theory_model!r}')
    return theory


def DESIFSLikelihood(tracers=None, cosmo=None, klim=(0.02, 0.2), solve='.auto',
                     theory_model='reptvelocileptors',
                     measurements_dir=DEFAULT_MEASUREMENTS_DIR):
    """Create DESI FS likelihood."""

    if tracers is None:
        tracers = list(get_tracer_zrange(Name))
  
    observables = []
    likelihoods = []
    if cosmo is None:
        cosmo = DirectPowerSpectrumTemplate(fiducial='DESI').cosmo
        cosmo.init.params['sigma8_m'] = {'derived': True, 'latex': r'\sigma_8'}  # derive sigma_8
        cosmo.init.params['tau_reio'].update(fixed=True)
        cosmo.init.params['omega_b'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.02218, 'scale': (3.025e-7)**0.5})
        cosmo.init.params['n_s'].update(fixed=False, prior={'dist': 'norm', 'loc': 0.9649, 'scale': 0.042})
        cosmo.init.update(engine='class')

    for namespace in tracers:
        tracer, zrange = get_tracer_zrange(namespace)
        print('Adding DESI FS likelihood for tracer {}, zrange {}, namespace {}'.format(tracer, zrange, namespace))

        data, covariance, window = get_synthetic_data(
            tracer=tracer,
            zrange=zrange,
            region='GCcomb',
            ells=[0, 2, 4],
            weights='default_fkp',
            klim=klim,
            rebin=5,
            measurements_dir=measurements_dir,
        )
        tracer_label = tracer.split('_')[0]
        # To reproduce previous bug:
        # tracer_label = 'QSO'  # FIXME
        theory = get_theory(cosmo=cosmo, z=window.theory.get(ells=0).z, tracer=tracer_label,
                            theory_model=theory_model)
        observable = TracerPowerSpectrumMultipolesObservable(
            data=data,
            theory=theory,
            covariance=covariance,
            window=window,
        )

        # Compute or swap in PT emulator
        emu_fn = emulator_dir / f'emulator_fs_{namespace}.npy'

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
            # Use the original desilike calculator; observable.window.theory is only an lsstypes tree.
            #theory.init.update(ells=(0, 2, 4))  # train emulator on all multipoles
            emulator = Emulator(theory.pt, engine=TaylorEmulatorEngine(method='finite', order=4))
            emulator.set_samples()
            emulator.fit()
            emulator.save(emu_fn)

        # Update namespace of bias parameters (to have one parameter per tracer / z-bin)
        for param in theory.init.params:
            # Update latex just to have better labels
            iz = int(namespace.split('_')[-1][1:])
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


def get_fit_fn(kind='profiles', sampler_name='emcee', theory_model='reptvelocileptors'):
    if theory_model == 'reptvelocileptors':
        fit_root = Path(output_dir)
    else:
        fit_root = Path(output_dir) / theory_model

    if kind == 'profiles':
        save_dir = fit_root / 'profiles/solve-best-ns-kmax0.3'
        save_fn = save_dir / f'profiles_abacushf_fs.npy'
    elif kind == 'chains':
        save_dir = fit_root / f'chains/solve-best-ns-kmax0.3/{sampler_name}'
        save_fn = [save_dir / f'chain_abacushf_fs_{ichain:d}.npy' for ichain in range(4)]
    return save_fn


def run_sampler(likelihood, sampler_name='emcee', theory_model='reptvelocileptors'):
    save_fn = get_fit_fn('chains', sampler_name=sampler_name, theory_model=theory_model)
    save_fn[0].parent.mkdir(parents=True, exist_ok=True)
    chain_fn = None
    if sampler_name == 'emcee':
        sampler = EmceeSampler(likelihood, chain_fn=chain_fn, nwalkers=64, save_fn=save_fn, seed=42)
    elif sampler_name == 'mcmc':
        sampler = MCMCSampler(likelihood, chains=chain_fn, save_fn=save_fn, oversample_power=0., seed=42)
    chains = sampler.run(min_iterations=200, check={'max_eigen_gr': 0.03})


def run_profiler(likelihood, theory_model='reptvelocileptors'):
    save_fn = get_fit_fn('profiles', theory_model=theory_model)
    save_fn.parent.mkdir(parents=True, exist_ok=True)
    profiler = MinuitProfiler(likelihood, seed=42)
    profiles = profiler.maximize()
    print(profiles.to_stats(tablefmt='pretty'))
    profiles.save(save_fn)
    return profiles


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--todo", type=str, nargs='+', default=['profile', 'sample'],)
    parser.add_argument("--sampler", type=str, default='emcee',)
    parser.add_argument("--theory-model", type=str, default='reptvelocileptors',
                        choices=['reptvelocileptors', 'folpsD'])
    parser.add_argument("--measurements-dir", type=Path, default=DEFAULT_MEASUREMENTS_DIR)

    args = parser.parse_args()

    todo = args.todo
    setup_logging()

    tracers = ['BGS_z0', 'LRG_z0', 'LRG_z1', 'LRG_z2', 'ELG_z1', 'QSO_z0']
    likelihood = DESIFSLikelihood(
            tracers=tracers,
            cosmo=None,
            klim=(0., 0.3),
            solve='.best',
            theory_model=args.theory_model,
            measurements_dir=args.measurements_dir,
        )

    if 'sample' in todo:
        run_sampler(likelihood, sampler_name=args.sampler, theory_model=args.theory_model)

    if 'profile' in todo:
        profiles = run_profiler(likelihood, theory_model=args.theory_model)
        profiles = Profiles.load(get_fit_fn('profiles', theory_model=args.theory_model))
        fig_dir.mkdir(parents=True, exist_ok=True)
        likelihood(**profiles.bestfit.choice(index='argmax', input=True))
        for tracer, likelihood in zip(tracers, likelihood.likelihoods):
            observable = likelihood.observables[0]
            observable.plot(fn=fig_dir / f'plot_bestfit_{tracer}.png')
