from desilike.theories.galaxy_clustering import (DirectPowerSpectrumTemplate,
                                                 FOLPSAXTracerPowerSpectrumMultipoles,
                                                 FOLPSAXTracerCorrelationFunctionMultipoles)
from desilike.observables.galaxy_clustering import (TracerPowerSpectrumMultipolesObservable,
                                                    TracerCorrelationFunctionMultipolesObservable)
from desilike.theories import Cosmoprimo
from cosmoprimo.fiducial import AbacusSummit

from pathlib import Path
from jaxpower import read
import matplotlib.pyplot as plt



class Blinder:
    def __init__(self, data, theory, cosmo_ref, cosmo_shift, param_cov=None):
        self.cosmo_ref = cosmo_ref
        self.cosmo_shift = cosmo_shift

        self.data = data
        self.theory = theory
        self.param_cov = param_cov

        self.set_params(None)

    def set_params(self, params):
        self.params = params or [
            'omega_cdm',
        ]

    def set_theory_vectors(self):
        params_ref = {param: self.cosmo_ref[param] for param in self.params}
        self.theory_ref = self.theory(**params_ref)

        params_shift = {param: self.cosmo_shift[param] for param in self.params}
        self.theory_shift = self.theory(**params_shift)

    def blind(self):
        params_ref = {param: self.cosmo_ref[param] for param in self.params}
        theory_ref = self.theory(**params_ref)

        params_shift = {param: self.cosmo_shift[param] for param in self.params}
        theory_shift = self.theory(**params_shift)

        blinded_data = {}
        for ell in self.data:
            slf.delta = theory_shift[ell] - theory_ref[ell]
            blinded_data[ell] = self.data[ell] + delta

        return blinded_data

    def unblind(self):
        params_ref = {param: self.cosmo_ref[param] for param in self.params}
        theory_ref = self.theory(**params_ref)

        params_shift = {param: self.cosmo_shift[param] for param in self.params}
        theory_shift = self.theory(**params_shift)

        unblinded_data = {}
        for ell in self.blinded_data:
            delta = theory_shift[ell] - theory_ref[ell]
            unblinded_data[ell] = self.blinded_data[ell] - delta

        return unblinded_data


class TracerPowerSpectrumMultipolesBlinder(Blinder):
    def __init__(self, k, ells=(0, 2), **kwargs):
        super().__init__(**kwargs)

        self.theory.k = k
        self.ells = ells

    def plot(self, save_fn: str = None):
        import matplotlib.pyplot as plt

        k = self.theory.k

        fig, ax = plt.subplots(figsize=(5, 4))

        for ell in self.ells:
            ax.plot(k, k * self.data[ell], label=f'data ell={ell}', marker='o', ms=3, ls='none')
            ax.plot(k, k * self.theory_ref[ell], label=f'ref model ell={ell}', ls='--')
            ax.plot(k, k * self.theory_shift[ell], label=f'shifted model ell={ell}', ls='-.')

        ax.legend()
        ax.set_xlabel(r'$k\,[h{\rm Mpc}^{-1}]$', fontsize=15)
        ax.set_ylabel(r'$k P_\ell(k)\,[h^{-2}{\rm Mpc}^{2}]$', fontsize=15)
        if save_fn is not None:
            plt.savefig(save_fn, dpi=300)
        return fig, ax

class TracerBispectrumMultipolesBlinder(Blinder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



def read_spectrum(kmin=0.02, kmax=0.2, rebin=1):
    data_dir = '/pscratch/sd/e/epaillas/desi/gqc-y3-growth/mock_challenge/abacus/secondgen/spectrum/c000_ph000/seed0'
    data_fn = Path(data_dir) / 'mesh2_spectrum_poles_c000.h5'
    data = read(data_fn)
    data = data.select(k=slice(0, None, rebin)).select(k=(kmin, kmax))
    poles = [data.get(ell) for ell in (0, 2, 4)]
    k = poles[0].coords('k')
    return k, poles


def get_theory_observable(kmin=0.02, kmax=0.2):
    cosmo = Cosmoprimo(fiducial='DESI', engine='class', lensing=True)
    template = DirectPowerSpectrumTemplate(z=0.5, cosmo=cosmo)
    theory = FOLPSAXTracerPowerSpectrumMultipoles(template=template, ells=(0, 2), k=k,
                                                  tracer='LRG', prior_basis='physical')
    observable = TracerPowerSpectrumMultipolesObservable(data=poles,
                                                         k=k, klim={0: (k.min(), k.max()),
                                                                    2: (k.min(), k.max())},)
    return theory, observable


if __name__ == '__main__':

    cosmo_ref = AbacusSummit(0)
    cosmo_shift = AbacusSummit(1)

    k, poles = read_spectrum()

    theory, observable = get_theory_observable(kmin=0.02, kmax=0.2)

    blinder = TracerPowerSpectrumMultipolesBlinder(
        k=k,
        data=poles,
        theory=theory,
        cosmo_ref=cosmo_ref,
        cosmo_shift=cosmo_shift,
    )

    blinder.set_theory_vectors()

    blinder.plot(save_fn='blinding_example.png')





