from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from desilike.observables.galaxy_clustering import TracerPowerSpectrumMultipolesObservable

from desiblind.utils import set_plot_style


SHIFTS_DIR = '/global/cfs/cdirs/desicollab/users/epaillas/y3-growth/dump/'
CHOSEN_BID = 0  # Change this to the desired blinded data ID


class Observable:
    """
    Class to hold information about a galaxy clustering observable.
    """
    def __init__(self, name, k, data, theory, covariance, window,
        reference_params: dict = None, ells: list = None):
        self.name = name
        self.k = k
        self.data = data
        self.theory = theory
        self.covariance = covariance
        self.window = window
        self.reference_params = reference_params

        if ells is not None:
            self.ells = ells
        else:
            self.ells = theory.ells

        self.blinded_data = []
        self.delta_theory = []
        self.blinded_data_id = 0
        

class Blinder:
    """
    Class to handle blinding of galaxy clustering observables.
    """
    def __init__(self):
        self.observables = []

    def get_convolved_theory(self, theory, window, params={}):
        """
        Convolve the theory evaluated at given params with the window function.
        """
        pred = theory(params)
        pred = window.dot(np.ravel(pred), return_type=None, zpt=False)
        pred = [pred.get(ell).value() for ell in theory.ells]
        return pred

    def add_observable(self, name, k, data, theory, covariance, window,
        reference_params: dict = None):
        observable = Observable(name, k, data, theory, covariance, window,
            reference_params=reference_params)
        observable.reference_params = reference_params
        setattr(self, name, observable)
        self.observables.append(observable)

    def _apply_blinding(self, shifted_params: dict, name: str = None):
        """
        Method to apply the blinding to the data, only meant to be used
        by the developers implementing the blinding procedure. General users
        should use the classmethods `apply_blinding` and `remove_blinding`.
        """
        for observable in self.observables:
            if name is not None and observable.name != name:
                continue

            # Compute reference theory
            reference_theory = self.get_convolved_theory(
                observable.theory,
                observable.window,
                params=observable.reference_params
            )

            # keep reference params that are not shifted
            for param in observable.reference_params:
                if param not in shifted_params:
                    shifted_params[param] = observable.reference_params[param]

            # Compute shifts
            shifted_theory = self.get_convolved_theory(
                observable.theory,
                observable.window,
                params=shifted_params
            )

            # Apply blinding
            delta_theory = []
            blinded_data = []
            for i, ell in enumerate(observable.ells):
                delta_theory.append(shifted_theory[i] - reference_theory[i])
                blinded_data.append(observable.data[i] + delta_theory[i])
            observable.blinded_data.append(blinded_data)
            observable.delta_theory.append(delta_theory)

    def save_blinded_data(self, save_dir: Path | str = SHIFTS_DIR):
        """
        Saves the shifts applied during blinding (obtained with the
        _apply_blinding function) to a file for later unblinding.
        """
        cout = {}
        for observable in self.observables:
            for bid, delta_theory in enumerate(observable.delta_theory):
                shifts = []
                for i, ell in enumerate(observable.ells):
                    shifts.append(InterpolatedUnivariateSpline(observable.k, delta_theory[i], k=3))
                namespace = hash(f'{observable.name}_bid{bid}')
                cout[namespace] = shifts
        save_fn = Path(save_dir) / 'shifts_blinding.npy'
        np.save(save_fn, cout)

    @classmethod
    def apply_blinding(
        cls,
        name: str,
        data: np.ndarray,
        k: np.ndarray,
        ells: list = [0, 2, 4]
    ) -> np.ndarray:
        """
        Apply blinding from a saved shifts file. This method can be used
        by general users to blind their data, without the need of instantiating
        the Blinder class.
        """
        save_fn = Path(SHIFTS_DIR) / 'shifts_blinding.npy'
        shifts_dict = np.load(save_fn, allow_pickle=True).item()
        key = hash(f'{name}_bid{CHOSEN_BID}')
        shifts = shifts_dict[key]

        blinded_data = []
        for i, ell in enumerate(ells):
            shift_fn = shifts[i]
            shift_values = shift_fn(k)
            blinded_data.append(data[i] + shift_values)
        return np.array(blinded_data)

    @classmethod
    def remove_blinding(
        cls,
        name: str,
        data: np.ndarray,
        k: np.ndarray,
        ells: list = [0, 2, 4]
    ) -> np.ndarray:
        """
        Remove blinding from a saved shifts file. This method can be used
        by general users to unblind their data, without the need of instantiating
        the Blinder class.
        """
        save_fn = Path(SHIFTS_DIR) / 'shifts_blinding.npy'
        shifts_dict = np.load(save_fn, allow_pickle=True).item()
        key = hash(f'{name}_bid{CHOSEN_BID}')
        shifts = shifts_dict[key]

        unblinded_data = []
        for i, ell in enumerate(ells):
            shift_fn = shifts[i]
            shift_values = shift_fn(k)
            unblinded_data.append(data[i] - shift_values)
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

        import matplotlib.pyplot as plt
        import itertools

        fig, ax = plt.subplots(figsize=(5, 4))
        markers = itertools.cycle(('o', 'x', '+', 's'))
        linestyles = itertools.cycle(('--', ':', '-.'))

        for observable in self.observables:
            if name is not None and observable.name != name:
                continue
            k = observable.k
            data = observable.data
            error = np.sqrt(np.diag(observable.covariance))

            if show_reference:
                theory = self.get_convolved_theory(
                    observable.theory,
                    observable.window,
                    params=observable.reference_params
                )

            if show_blinded:
                for ibid, bid in enumerate(blinded_ids):
                    blinded_data = observable.blinded_data[bid]
                    for iell, ell in enumerate(observable.ells):
                        ax.plot(k, k * blinded_data[iell], ls='--', color=f'C{iell}', lw=1.0,
                            label=f'blinded data' if iell == 0  and ibid == 0 else None)

            marker = next(markers)

            for i, ell in enumerate(observable.ells):
                ax.errorbar(k, k * data[i], yerr=k * error[i * len(k):(i + 1) * len(k)], color=f'C{i}',
                label=observable.name if i == 0 else None, marker=marker, ms=2.5, ls='none', elinewidth=1.0)

                if show_reference:
                    ax.plot(k, k * theory[i], ls='-', color=f'C{i}', lw=1.0,
                        label=f'reference theory' if i == 0 else None)

        ax.legend()
        ax.set_xlabel(r'$k\,[h{\rm Mpc}^{-1}]$', fontsize=14)
        ax.set_ylabel(r'$k P_\ell(k)\,[h^{-2}{\rm Mpc}^{2}]$', fontsize=14)
        plt.tight_layout()
        return fig, ax


class TracerBispectrumMultipolesBlinder(Blinder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)