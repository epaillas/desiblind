from desilike.theories.galaxy_clustering import (DirectPowerSpectrumTemplate,
                                                 FOLPSAXTracerPowerSpectrumMultipoles,
                                                 REPTVelocileptorsTracerPowerSpectrumMultipoles)
from desilike.theories import Cosmoprimo
from desilike.samples import Profiles
from cosmoprimo.fiducial import AbacusSummit
from cosmoprimo import Cosmology

from desiblind import TracerPowerSpectrumMultipolesBlinder

import lsstypes as types

from pathlib import Path
from jaxpower import read
import numpy as np
import matplotlib.pyplot as plt


def read_spectrum(kmin=0.02, kmax=0.2, rebin=1):
    data_dir = '/pscratch/sd/e/epaillas/desi/gqc-y3-growth/mock_challenge/abacus/secondgen/spectrum/c000_ph000/seed0'
    data_fn = Path(data_dir) / 'mesh2_spectrum_poles_c000.h5'
    data = read(data_fn)
    data = data.select(k=slice(0, None, rebin)).select(k=(kmin, kmax))
    poles = [data.get(ell) for ell in (0, 2, 4)]
    k = poles[0].coords('k')
    return k, poles


def get_theory(kmin=0.02, kmax=0.2):
    cosmo = Cosmoprimo(fiducial='DESI', engine='class', lensing=True)
    template = DirectPowerSpectrumTemplate(z=0.5, cosmo=cosmo)
    theory = REPTVelocileptorsTracerPowerSpectrumMultipoles(template=template, ells=(0, 2), k=k,
                                                  tracer='LRG', prior_basis='physical')
    return theory


def get_synthetic_data(statistic='mesh2_spectrum_poles', tracer='LRG', zmin=0.4, zmax=0.6,
    region='GCcomb', ells=[0, 2], weights='default_fkp', kmin=0.0, kmax=0.2, rebin=5):
    dirname = Path('/global/cfs/projectdirs/desi/mocks/cai/mock-challenge-cutsky-dr2/unblinded_data/dr2-v2/')

    covariance = types.read(
        dirname / f'covariance_{statistic}_{tracer}_z{zmin}-{zmax}_{region}_{weights}.h5'
    )

    # Synthetic data vector (REPTVelocileptorsTracerPowerSpectrumMultipoles fit to Abacus HF v2, multiplied by the window matrix)
    observable = covariance.observable.select(k=slice(0, None, rebin)).select(k=(kmin, kmax)).get(ells=ells)  # rebin, select, get monopole and quadrupole

    poles = [observable.get(ell) for ell in ells]
    k = poles[0].coords('k')
    poles = [pole.value() for pole in poles]
    # print(pole.values('shotnoise'))  # shotnoise

    covariance = covariance.at.observable.match(observable)
    covariance = covariance.value()

    window = types.read(dirname / f'window_{statistic}_{tracer}_z{zmin}-{zmax}_{region}_{region}.h5')
    window = window.at.observable.match(observable)  # rebin along observable axis
    window = window.at.theory.select(k=(0, 0.3))  # select 0. < k < 0.3 on the theory axis

    return k, poles, covariance, window

def plot_synthetic_data():
    k, poles, covariance = get_synthetic_data()
    error = np.sqrt(np.diag(covariance))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.errorbar(k, k * poles[0], k * error[0:len(k)], marker='o', ms=2.5, ls='none', elinewidth=1.0)
    ax.errorbar(k, k * poles[1], k * error[len(k):2*len(k)], marker='o', ms=2.5, ls='none', elinewidth=1.0)

    ax.set_xlabel(r'$k\,[h{\rm Mpc}^{-1}]$', fontsize=13)
    ax.set_ylabel(r'$k P_\ell(k)\,[h^{-2}{\rm Mpc}^{2}]$', fontsize=13)
    plt.tight_layout()
    plt.savefig('example_synthetic_data.png', dpi=300)


if __name__ == '__main__':

    k, poles, covariance, window = get_synthetic_data()

    cosmo_ref = AbacusSummit(0)
    # cosmo_shift = AbacusSummit(1)
    cosmo_shift = cosmo_ref.clone(logA=2)


    profiles = Profiles.load('profiles.npy')
    nuisance = profiles.bestfit.choice(index='argmax', input=True)

    theory = get_theory(kmin=0.02, kmax=0.2)

    blinder = TracerPowerSpectrumMultipolesBlinder(
        k=k,
        data=poles,
        theory=theory,
        window=window,
        covariance=covariance,
        cosmo_ref=cosmo_ref,
        cosmo_shift=cosmo_shift,
    )

    blinder.set_theory_vectors(nuisance)
    blinder.blind()
    blinder.unblind()

    blinder.plot(save_fn='blinding_example.png')