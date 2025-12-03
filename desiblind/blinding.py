from pathlib import Path
import itertools
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

try:
    import lsstypes as types
    from lsstypes import ObservableTree
except ImportError:
    ObservableTree = None

from desiblind.utils import set_plot_style


SHIFTS_DIR = '/global/cfs/cdirs/desicollab/users/epaillas/y3-growth/dump/'


class Observable:
    """
    Class to hold information about a galaxy clustering observable.
    """
    def __init__(self, name, data, covariance=None):
        """
        Parameters
        ----------
        name : str
            Name of the observable.
        data : ObservableTree
            Reference data vector.
        covariance : CovarianceMatrix
            Covariance matrix.
        """
        self.name = name
        self.covariance = covariance
        self.reference_data = data
        self.blinded_data = []
        

class Blinder:
    """
    Class to handle blinding of galaxy clustering observables.
    """
    blinded_nmax = 100

    def __init__(self):
        self.observables = []

    # def get_convolved_theory(self, theory, window, params={}):
    #     """
    #     Convolve the theory evaluated at given params with the window function.
    #     """
    #     pred = theory(params)
    #     pred = window.dot(np.ravel(pred), return_type=None, zpt=False)
    #     pred = [pred.get(ell).value() for ell in theory.ells]
    #     return pred

    def add_observable(self, name, data, covariance=None):
        observable = Observable(name, data, covariance=covariance)
        self.observables.append(observable)

    @classmethod
    def __get_bid(cls):
        rng = np.random.RandomState(seed=42)
        return rng.randint(0, cls.blinded_nmax)

    def set_blinded_data(self, name: str = None, blinded_data=None):
        """
        Method to apply the blinding to the data, only meant to be used
        by the developers implementing the blinding procedure. General users
        should use the classmethods `apply_blinding` and `remove_blinding`.
        """
        for observable in self.observables:
            if name is not None and observable.name != name:
                continue
            if not isinstance(blinded_data, ObservableTree):
                blinded_data = observable.reference_data.clone(value=np.ravel(blinded_data))
            observable.blinded_data.append(blinded_data)

    def write_blinded_shifts(self, save_dir: Path | str = SHIFTS_DIR):
        """
        Saves the shifts applied during blinding (obtained with the set_blinded_data function) to a file for later unblinding.
        """
        cout = {}
        for observable in self.observables:
            assert len(observable.blinded_data) >= self.blinded_nmax, f'Please generate at least {self.blinded_nmax} blinded data vectors'
            for bid, blinded_data in enumerate(observable.blinded_data):
                shifts = {}
                for ell in blinded_data.ells:
                    k = blinded_data.get(ell).coords('k')
                    diff = blinded_data.get(ell).value() - observable.reference_data.get(ell).value()
                    shifts[ell] = (k, diff)
                namespace = hashlib.sha256(f'{observable.name}_bid{bid}'.encode()).hexdigest()
                cout[namespace] = shifts
        save_fn = Path(save_dir) / 'shifts_blinding.npy'
        np.save(save_fn, cout)

    @classmethod
    def apply_blinding(
        cls,
        name: str,
        data: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        Apply blinding from a saved shifts file. This method can be used
        by general users to blind their data, without the need of instantiating
        the Blinder class.

        Parameters
        ----------
        name : str
            Name of the tracer - redshift bin, e.g. 'LRG_z0'.
        data : list, ObservableTree
            If list, list of multipoles; in this case also provide corresponding ``ells`` and ``k``.
            Else, an ObservableTree from lsstypes.

        Returns
        -------
        blinded_data : np.ndarray or ObservableTree
        """
        save_fn = Path(SHIFTS_DIR) / 'shifts_blinding.npy'
        shifts_dict = np.load(save_fn, allow_pickle=True).item()
        key = hashlib.sha256(f'{name}_bid{cls.__get_bid()}'.encode()).hexdigest()
        if key not in shifts_dict:
            raise ValueError(f'Cannot find the blinding value for {name}')
        shifts = shifts_dict[key]

        def get_blinded_data(k, ells, data):
            blinded_data = []
            for ill, ell in enumerate(ells):
                shift_values = InterpolatedUnivariateSpline(shifts[ell][0], shifts[ell][1], k=3)(k[ill])
                blinded_data.append(data[ill] + shift_values)
            return blinded_data
        
        if isinstance(data, ObservableTree):
            blinded_data = get_blinded_data([pole.coords('k') for pole in data], data.ells, [pole.value() for pole in data])
            return data.clone(value=np.ravel(blinded_data))
        else:
            k, ells = kwargs['k'], kwargs['ells']
            blinded_data = get_blinded_data([k] * len(ells), ells, data)
            return np.array(blinded_data)

    @classmethod
    def remove_blinding(
        cls,
        name: str,
        data: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """
        Remove blinding from a saved shifts file. This method can be used
        by general users to blind their data, without the need of instantiating
        the Blinder class.

        Parameters
        ----------
        name : str
            Name of the tracer - redshift bin, e.g. 'LRG_z0'.
        data : list, ObservableTree
            If list, list of multipoles; in this case also provide corresponding ``ells`` and ``k``.
            Else, an ObservableTree from lsstypes.

        Returns
        -------
        unblinded_data : np.ndarray or ObservableTree
        """
        if not kwargs.get('force', False):
            raise ValueError('Are you sure you want to unblind? If so, provide "force=True"')
        save_fn = Path(SHIFTS_DIR) / 'shifts_blinding.npy'
        shifts_dict = np.load(save_fn, allow_pickle=True).item()
        key = hashlib.sha256(f'{name}_bid{cls.__get_bid()}'.encode()).hexdigest()
        if key not in shifts_dict:
            raise ValueError(f'Cannot find the blinding value for {name}')
        shifts = shifts_dict[key]

        def get_unblinded_data(k, data):
            unblinded_data = []
            for ill, ell in enumerate(ells):
                shift_values = InterpolatedUnivariateSpline(shifts[ell][0], shifts[ell][1], k=3)(k[ill])
                unblinded_data.append(data[ill] - shift_values)
            return unblinded_data
        
        if isinstance(data, ObservableTree):
            unblinded_data = get_unblinded_data([pole.coords('k') for pole in data], [pole.value() for pole in data])
            return data.clone(value=np.ravel(unblinded_data))
        else:
            k, ells = kwargs['k'], kwargs['ells']
            unblinded_data = get_unblinded_data([k] * len(ells), data)
            return np.array(unblinded_data)


class TracerPowerSpectrumMultipolesBlinder(Blinder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @set_plot_style
    def plot_observables(self, name: str = None, show_reference: bool = True, show_blinded: bool = False,
        blinded_ids: list[int] = [0]):
        """
        Plot the observed data and theory vectors, with options to show
        the reference theory and (multiple) blinded data vectors.
        """
        # TODO use KP3 color scheme
        fig, ax = plt.subplots(figsize=(5, 4))
        markers = itertools.cycle(('o', 'x', '+', 's'))
        linestyles = itertools.cycle(('--', ':', '-.'))

        for observable in self.observables:
            if name is not None and observable.name != name:
                continue
            reference_data = observable.reference_data

            if show_blinded:
                for ibid, bid in enumerate(blinded_ids):
                    blinded_data = observable.blinded_data[bid]
                    for ill, ell in enumerate(blinded_data.ells):
                        pole = blinded_data.get(ells=ell)
                        ax.plot(k:=pole.coords('k'), k * pole.value(), ls='--', color=f'C{ill}', lw=1.0,
                            label=f'blinded data' if ill == 0 and ibid == 0 else None, zorder=1)

            marker = next(markers)
            if show_reference:
                for ill, ell in enumerate(reference_data.ells):
                    std = observable.covariance.at.observable.get(ells=ell).std()
                    pole = reference_data.get(ells=ell)
                    ax.errorbar(k:=pole.coords('k'), k * pole.value(), yerr=k * std, color=f'C{ill}',
                    label=observable.name if ill == 0 else None, marker=marker, ms=2.5, ls='none', elinewidth=1.0,
                    zorder=2, mfc='k', mew=0.5)

        ax.legend()
        ax.set_xlabel(r'$k\,[h{\rm Mpc}^{-1}]$', fontsize=14)
        ax.set_ylabel(r'$k P_\ell(k)\,[h^{-2}{\rm Mpc}^{2}]$', fontsize=14)
        plt.tight_layout()
        return fig, ax


class TracerBispectrumMultipolesBlinder(Blinder):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)