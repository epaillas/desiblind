"""Standalone power-spectrum blinding workflow.

This script is a scriptified version of ``nb/blinding_example.ipynb``. It keeps
the same scientific flow while moving the notebook narrative into runnable code:

1. Load the best-fit full-shape parameters used as the reference cosmology.
2. Build reference observables for each tracer and compare them to the input
   synthetic data vector.
3. Draw blinded cosmologies from a Gaussian approximation to the posterior.
4. Evaluate a blinded observable for each draw and store the resulting shifts.
5. Save per-tracer plots and persist the hidden shifts to disk.
6. Demonstrate the high-level apply/remove blinding API on one tracer.

The blinding formalism is

    P_blind(k) = P(k) - W(k, k') P_ref(k') + W(k, k') P_shift(k'),

where ``P_ref`` is evaluated at the reference cosmology and ``P_shift`` at a
hidden shifted cosmology drawn from the posterior.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from desilike.samples import Chain, Profiles

from desiblind import TracerPowerSpectrumMultipolesBlinder

from fs_likelihood import get_fit_fn, get_synthetic_data, get_theory, get_tracer_zrange


DEFAULT_TRACERS = ['BGS_z0', 'LRG_z0', 'LRG_z1', 'LRG_z2', 'ELG_z1', 'QSO_z0']
DEFAULT_ELLS = [0, 2, 4]
DEFAULT_COSMO_PARAMS = ['h', 'omega_cdm', 'omega_b', 'logA', 'n_s']


def sample_from_gaussian(mean, covariance, size=1, seed=42):
    """Sample from a multivariate Gaussian distribution."""
    rng = np.random.RandomState(seed=seed)
    return rng.multivariate_normal(mean, covariance, size=size)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Blind DESI synthetic power-spectrum multipoles and reproduce the '
            'plots from nb/blinding_example.ipynb.'
        )
    )
    parser.add_argument('--tracers', nargs='+', default=DEFAULT_TRACERS,
                        help='Tracer namespaces to blind, e.g. LRG_z0 ELG_z1.')
    parser.add_argument('--ells', nargs='+', type=int, default=DEFAULT_ELLS,
                        help='Multipoles to load and plot.')
    parser.add_argument('--klim', nargs=2, type=float, default=(0.0, 0.3), metavar=('KMIN', 'KMAX'),
                        help='k-range passed to get_synthetic_data.')
    parser.add_argument('--rebin', type=int, default=5,
                        help='Rebin factor passed to get_synthetic_data.')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of blinded realizations to generate per tracer.')
    parser.add_argument('--sampler', default='mcmc',
                        help='Sampler subdirectory used by fs_likelihood.get_fit_fn.')
    parser.add_argument('--chain-slice', nargs=2, type=int, default=(1, 4), metavar=('START', 'STOP'),
                        help='Slice of chain files to concatenate, matching notebook defaults.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for Gaussian posterior sampling.')
    parser.add_argument('--plot-dir', type=Path, default=Path(__file__).resolve().parent / 'fig',
                        help='Directory where output plots will be written.')
    parser.add_argument('--shifts-dir', type=Path, default=Path(__file__).resolve().parent / 'data' / 'blinding',
                        help='Directory where shifts_blinding.npy will be written.')
    parser.add_argument('--demo-name', default='LRG_z0',
                        help='Tracer namespace used for the apply/remove blinding check.')
    parser.add_argument('--skip-write-shifts', action='store_true',
                        help='Skip writing shifts_blinding.npy to disk.')
    parser.add_argument('--skip-demo-apply-remove', action='store_true',
                        help='Skip the final apply/remove blinding validation.')
    return parser.parse_args()


def load_bestfit():
    """Load the reference parameter dictionary from the joint full-shape fit."""
    profiles = Profiles.load(get_fit_fn('profiles'))
    bestfit = profiles.bestfit.choice(index='argmax', input=True)
    return {name: float(value) for name, value in bestfit.items()}


def split_bestfit_params(bestfit):
    """Split cosmological and tracer-specific nuisance parameters."""
    cosmo_params = {name: value for name, value in bestfit.items() if '.' not in name}
    nuisance_params = {name: value for name, value in bestfit.items() if name not in cosmo_params}
    return cosmo_params, nuisance_params


def load_chain(sampler_name, chain_slice):
    """Concatenate the selected chain files."""
    start, stop = chain_slice
    chain_fns = get_fit_fn('chains', sampler_name=sampler_name)[start:stop]
    return Chain.concatenate([Chain.load(fn).ravel()[::1] for fn in chain_fns])


def build_tracer_observable(namespace, cosmo_params, nuisance_params, ells, klim, rebin):
    """Build the reference observable for one tracer."""
    tracer, zrange = get_tracer_zrange(namespace)
    data, covariance, window = get_synthetic_data(
        tracer=tracer,
        zrange=zrange,
        region='GCcomb',
        ells=ells,
        weights='default_fkp',
        klim=tuple(klim),
        rebin=rebin,
    )
    theory = get_theory(z=window.theory.get(ells=0).z, tracer=tracer)
    theory.init.update(
        k=window.theory.get(ells=0).coords('k'),
        shotnoise=data.get(ells=0).values('shotnoise').mean(),
    )
    params = cosmo_params | {
        param.split('.')[-1]: value
        for param, value in nuisance_params.items()
        if param.startswith(namespace)
    }
    spectrum = theory(**params)
    observable = window.dot(np.ravel(spectrum), return_type=None, zpt=False)
    return data, covariance, window, theory, observable


def save_reference_plot(namespace, observable, data, covariance, output_dir):
    """Save the notebook's theory-vs-data diagnostic plot for one tracer."""
    fig, ax = plt.subplots(figsize=(4, 3))
    for ill, ell in enumerate(observable.ells):
        color = f'C{ill:d}'
        theory_pole = observable.get(ells=ell)
        ax.plot(k := theory_pole.coords('k'), k * theory_pole.value(), color=color, linestyle='--')
        data_pole = data.get(ells=ell)
        std = covariance.at.observable.get(ells=ell).std()
        ax.errorbar(k := data_pole.coords('k'), k * data_pole.value(), k * std,
                    color=color, ms=2.5, ls='none', elinewidth=1.0)
    ax.set_xlabel(r'$k\,[h{\rm Mpc}^{-1}]$', fontsize=14)
    ax.set_ylabel(r'$k P_\ell(k)\,[h^{-2}{\rm Mpc}^{2}]$', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / f'reference_{namespace}.png', dpi=300)
    plt.close(fig)


def get_blinding_samples(bestfit, chain, num_samples, seed):
    """Sample blinded cosmologies from the posterior covariance used in the notebook."""
    nuisance_param_names = [name for name in chain.params(input=True).keys() if name not in DEFAULT_COSMO_PARAMS]
    samples = sample_from_gaussian(
        mean=[bestfit[name] for name in DEFAULT_COSMO_PARAMS],
        covariance=chain.covariance(DEFAULT_COSMO_PARAMS) * 4,
        size=num_samples,
        seed=seed,
    )
    return [
        dict(zip(DEFAULT_COSMO_PARAMS, sample)) | {name: bestfit[name] for name in nuisance_param_names}
        for sample in samples
    ]


def add_reference_observables(blinder, tracers, cosmo_params, nuisance_params, ells, klim, rebin, plot_dir):
    """Build and register one reference observable per tracer."""
    for namespace in tracers:
        data, covariance, _, _, observable = build_tracer_observable(
            namespace=namespace,
            cosmo_params=cosmo_params,
            nuisance_params=nuisance_params,
            ells=ells,
            klim=klim,
            rebin=rebin,
        )
        save_reference_plot(namespace, observable, data, covariance, plot_dir)
        blinder.add_observable(name=namespace, data=observable, covariance=covariance)


def generate_blinded_realizations(blinder, tracers, samples, ells, klim, rebin):
    """Generate blinded observables by re-evaluating theory at shifted parameters."""
    for namespace in tracers:
        tracer, zrange = get_tracer_zrange(namespace)
        data, _, window = get_synthetic_data(
            tracer=tracer,
            zrange=zrange,
            region='GCcomb',
            ells=ells,
            weights='default_fkp',
            klim=tuple(klim),
            rebin=rebin,
        )
        theory = get_theory(z=window.theory.get(ells=0).z, tracer=tracer)
        theory.init.update(
            k=window.theory.get(ells=0).coords('k'),
            shotnoise=data.get(ells=0).values('shotnoise').mean(),
        )

        for sample in samples:
            nuisance_names = [name for name in sample.keys() if name.startswith(namespace)]
            params = {
                name.split('.')[-1]: sample[name]
                for name in [*DEFAULT_COSMO_PARAMS, *nuisance_names]
            }
            spectrum = theory(**params)
            observable = window.dot(np.ravel(spectrum), return_type=None, zpt=False)
            blinder.set_blinded_data(namespace, blinded_data=observable)


def save_blinded_plots(blinder, tracers, plot_dir, num_samples):
    """Save the notebook's blinded-shift plots for each tracer."""
    for namespace in tracers:
        fig, _ = blinder.plot_observables(
            name=namespace,
            show_blinded=True,
            blinded_ids=list(range(num_samples)),
        )
        fig.savefig(plot_dir / f'blinding_shifts_{namespace}.png', dpi=300)
        plt.close(fig)


def run_apply_remove_demo(name, ells, shifts_dir, klim, rebin):
    """Validate the high-level API on one tracer by applying and removing the saved shifts."""
    tracer, zrange = get_tracer_zrange(name)
    observable, _, _ = get_synthetic_data(
        tracer=tracer,
        zrange=zrange,
        region='GCcomb',
        ells=ells,
        weights='default_fkp',
        klim=tuple(klim),
        rebin=rebin,
    )
    poles = [observable.get(ell) for ell in ells]
    k = poles[0].coords('k')
    pole_values = [pole.value() for pole in poles]

    blinded_data = TracerPowerSpectrumMultipolesBlinder.apply_blinding(
        name=name,
        k=k,
        data=pole_values,
        ells=ells,
        save_dir=shifts_dir,
    )
    blinded_tree = TracerPowerSpectrumMultipolesBlinder.apply_blinding(
        name=name,
        data=observable,
        save_dir=shifts_dir,
    )
    if not np.allclose(blinded_tree.value(), np.ravel(blinded_data)):
        raise RuntimeError('ObservableTree and array blinding results do not match.')

    unblinded_data = TracerPowerSpectrumMultipolesBlinder.remove_blinding(
        name=name,
        k=k,
        data=blinded_data,
        ells=ells,
        save_dir=shifts_dir,
        force=True,
    )
    if not np.allclose(np.ravel(unblinded_data), np.ravel(pole_values)):
        raise RuntimeError('remove_blinding did not recover the original data vector.')


def main():
    args = parse_args()
    if not args.skip_write_shifts and args.num_samples < TracerPowerSpectrumMultipolesBlinder.blinded_nmax:
        raise ValueError(
            f'--num-samples must be at least {TracerPowerSpectrumMultipolesBlinder.blinded_nmax} '
            'when writing blinded shifts.'
        )
    if args.skip_write_shifts and not args.skip_demo_apply_remove:
        raise ValueError('Cannot run the apply/remove demo when --skip-write-shifts is set.')
    if not args.skip_demo_apply_remove and args.demo_name not in args.tracers:
        raise ValueError('--demo-name must be included in --tracers for a fresh run.')

    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.shifts_dir.mkdir(parents=True, exist_ok=True)

    bestfit = load_bestfit()
    cosmo_params, nuisance_params = split_bestfit_params(bestfit)
    blinder = TracerPowerSpectrumMultipolesBlinder()

    add_reference_observables(
        blinder=blinder,
        tracers=args.tracers,
        cosmo_params=cosmo_params,
        nuisance_params=nuisance_params,
        ells=args.ells,
        klim=args.klim,
        rebin=args.rebin,
        plot_dir=args.plot_dir,
    )

    chain = load_chain(args.sampler, args.chain_slice)
    samples = get_blinding_samples(bestfit, chain, args.num_samples, args.seed)
    generate_blinded_realizations(
        blinder=blinder,
        tracers=args.tracers,
        samples=samples,
        ells=args.ells,
        klim=args.klim,
        rebin=args.rebin,
    )
    save_blinded_plots(blinder, args.tracers, args.plot_dir, args.num_samples)

    if not args.skip_write_shifts:
        blinder.write_blinded_shifts(save_dir=args.shifts_dir)

    if not args.skip_demo_apply_remove:
        run_apply_remove_demo(
            name=args.demo_name,
            ells=args.ells,
            shifts_dir=args.shifts_dir,
            klim=args.klim,
            rebin=args.rebin,
        )


if __name__ == '__main__':
    main()
