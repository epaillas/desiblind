from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from desiblind.utils import set_plot_style


class Blinder:
    def __init__(self, data, theory, covariance, cosmo_ref, cosmo_shift, window=None, param_cov=None):
        self.cosmo_ref = cosmo_ref
        self.cosmo_shift = cosmo_shift

        self.data = data
        self.theory = theory
        self.covariance = covariance
        self.window = window
        self.param_cov = param_cov

        self.set_params(None)

    def set_params(self, params):
        self.params = params or [
            'omega_b',
            'omega_cdm',
            'logA',
            'h',
            'n_s',
        ]

    def set_theory_vectors(self, nuisance_params=None):
        params_ref = {param: self.cosmo_ref[param] for param in self.params}
        if nuisance_params is not None:
            params_ref.update(nuisance_params)
        self.theory_ref = self.theory(**params_ref)

        params_shift = {param: self.cosmo_shift[param] for param in self.params}
        if nuisance_params is not None:
            params_shift.update(nuisance_params)
        self.theory_shift = self.theory(**params_shift)

    def blind(self):
        self.blinded_data = []
        for i, ell in enumerate(self.ells):
            self.blinded_data.append(self.data[i] - self.theory_ref[i] + self.theory_shift[i])
        return self.blinded_data

    def unblind(self):
        self.unblinded_data = []
        for i, ell in enumerate(self.ells):
            self.unblinded_data.append(self.blinded_data[i] + self.theory_ref[i] - self.theory_shift[i])
        return self.unblinded_data
        return unblinded_data


class TracerPowerSpectrumMultipolesBlinder(Blinder):
    def __init__(self, k, ells=(0, 2), **kwargs):
        super().__init__(**kwargs)

        self.theory.k = k
        self.ells = ells

    @set_plot_style
    def plot(self, save_fn: str = None):
        import matplotlib.pyplot as plt

        k = self.theory.k
        error = np.sqrt(np.diag(self.covariance))

        fig, ax = plt.subplots(figsize=(5, 4))

        for i, ell in enumerate(self.ells):
            ax.errorbar(k, k * self.data[i], yerr=k * error[i * len(k):(i + 1) * len(k)],
            label=f'original data' if i == 0 else None, marker='o', ms=2.5, ls='none', elinewidth=1.0, color=f'C{i}')
            ax.plot(k, k * self.theory_ref[i], label=f'reference model' if i == 0 else None, ls='--', color=f'C{i}')
            ax.plot(k, k * self.theory_shift[ell], label=f'shifted model' if i == 0 else None, ls=':', color=f'C{i}')
            ax.plot(k, k * self.blinded_data[i], label=f'blinded data' if i == 0 else None, ls='-', color=f'C{i}')
            ax.plot(k, k * self.unblinded_data[i], label=f'unblinded data' if i == 0 else None, ls='-.', color=f'C{i}')

        ax.legend()
        ax.set_xlabel(r'$k\,[h{\rm Mpc}^{-1}]$', fontsize=15)
        ax.set_ylabel(r'$k P_\ell(k)\,[h^{-2}{\rm Mpc}^{2}]$', fontsize=15)
        plt.tight_layout()
        if save_fn is not None:
            plt.savefig(save_fn, dpi=300)
        return fig, ax

class TracerBispectrumMultipolesBlinder(Blinder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)