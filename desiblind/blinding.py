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
from desiblind.tracers import normalize_canonical_tracerbin_name, to_tracerbin_name


SHIFTS_DIR = '/global/cfs/cdirs/desicollab/users/epaillas/y3-growth/dump/'
SHIFTS_FILENAME = 'shifts_blinding_2026_04.npy'


def is_observable_tree(data) -> bool:
    """Return True when ``data`` is an lsstypes ObservableTree."""
    return ObservableTree is not None and isinstance(data, ObservableTree)


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
    observable_suffix = None

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

    @classmethod
    def _get_observable_suffix(cls) -> str:
        if cls.observable_suffix is None:
            raise NotImplementedError(f'{cls.__name__} must define observable_suffix.')
        return cls.observable_suffix

    @classmethod
    def _suggest_public_name(cls, name: str) -> str:
        suffix = f'_{cls._get_observable_suffix()}'
        candidate = to_tracerbin_name(name)
        if candidate.endswith(suffix):
            candidate = candidate[:-len(suffix)]
        try:
            return normalize_canonical_tracerbin_name(candidate)
        except ValueError:
            return 'LRG1'

    @classmethod
    def _get_public_observable_name(cls, name: str) -> str:
        text = str(name)
        suffix = f'_{cls._get_observable_suffix()}'
        if text.endswith(suffix):
            suggestion = cls._suggest_public_name(text[:-len(suffix)])
            raise ValueError(
                f'Observable name "{name}" must be a bare canonical tracer-bin name like "{suggestion}"; '
                f'do not include the "{cls._get_observable_suffix()}" suffix.'
            )
        try:
            return normalize_canonical_tracerbin_name(text)
        except ValueError as exc:
            suggestion = cls._suggest_public_name(text)
            raise ValueError(
                f'Observable name "{name}" must be a bare canonical tracer-bin name like "{suggestion}".'
            ) from exc

    @classmethod
    def _get_internal_observable_name(cls, name: str) -> str:
        return f'{cls._get_public_observable_name(name)}_{cls._get_observable_suffix()}'

    @classmethod
    def _get_public_name_from_internal(cls, name: str) -> str:
        suffix = f'_{cls._get_observable_suffix()}'
        text = str(name)
        if text.endswith(suffix):
            return text[:-len(suffix)]
        return text

    def add_observable(self, name, data, covariance=None):
        observable = Observable(self._get_internal_observable_name(name), data, covariance=covariance)
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
        target_name = None if name is None else self._get_internal_observable_name(name)
        for observable in self.observables:
            if target_name is not None and observable.name != target_name:
                continue
            if not is_observable_tree(blinded_data):
                blinded_data = observable.reference_data.clone(value=self._flatten_multipoles(blinded_data))
            observable.blinded_data.append(blinded_data)

    @staticmethod
    def _get_shifts_fn(save_dir: Path | str | None = None, shifts_fn: Path | str | None = None) -> Path:
        """Resolve the shifts file path from either a directory or a full filename."""
        if shifts_fn is not None:
            return Path(shifts_fn)
        if save_dir is None:
            save_dir = SHIFTS_DIR
        return Path(save_dir) / SHIFTS_FILENAME

    def write_blinded_shifts(self, save_dir: Path | str = SHIFTS_DIR, shifts_fn: Path | str | None = None):
        """
        Saves the shifts applied during blinding (obtained with the set_blinded_data function) to a file for later unblinding.
        """
        cout = {}
        for observable in self.observables:
            assert len(observable.blinded_data) >= self.blinded_nmax, f'Please generate at least {self.blinded_nmax} blinded data vectors'
            for bid, blinded_data in enumerate(observable.blinded_data):
                shifts = {}
                for ell in blinded_data.ells:
                    pole = blinded_data.get(ells=ell)
                    reference = observable.reference_data.get(ells=ell)
                    shifts[ell] = self._serialize_shift(pole, reference)
                namespace = hashlib.sha256(f'{observable.name}_bid{bid}'.encode()).hexdigest()
                cout[namespace] = shifts
        save_fn = self._get_shifts_fn(save_dir=save_dir, shifts_fn=shifts_fn)
        save_fn.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_fn, cout)

    @staticmethod
    def _flatten_multipoles(values) -> np.ndarray:
        """Flatten a list of multipoles, allowing for ragged per-multipole lengths."""
        if isinstance(values, np.ndarray) and values.dtype != object:
            return np.ravel(values)
        return np.concatenate([np.ravel(np.asarray(value)) for value in values])

    @staticmethod
    def _serialize_shift(pole, reference):
        """Serialize one observable multipole shift."""
        return pole.coords('k'), pole.value() - reference.value()

    @classmethod
    def _evaluate_shift(cls, ell, coordinates, shift):
        """Evaluate one saved shift on the requested coordinates."""
        shift_coords, shift_values = shift
        return InterpolatedUnivariateSpline(shift_coords, shift_values, k=3)(coordinates)

    @staticmethod
    def _normalize_values(data, ells):
        if len(ells) == 1 and not (
            isinstance(data, (list, tuple))
            and len(data) == 1
            and np.asarray(data[0]).ndim > 0
        ):
            return [np.asarray(data)]
        values = [np.asarray(value) for value in data]
        if len(values) != len(ells):
            raise ValueError(f'Expected {len(ells)} multipoles, received {len(values)}.')
        return values

    @staticmethod
    def _normalize_coordinates(k, ells):
        if isinstance(k, np.ndarray):
            return [np.asarray(k)] * len(ells)
        if len(ells) == 1 and not (
            isinstance(k, (list, tuple))
            and len(k) == 1
            and np.asarray(k[0]).ndim > 0
        ):
            return [np.asarray(k)]
        if isinstance(k, (list, tuple)) and len(k) == len(ells):
            return [np.asarray(coords) for coords in k]
        return [np.asarray(k)] * len(ells)

    @classmethod
    def _get_multipoles(cls, data, **kwargs):
        if is_observable_tree(data):
            ells = list(data.ells)
            poles = [data.get(ells=ell) for ell in ells]
            coords = [pole.coords('k') for pole in poles]
            values = [pole.value() for pole in poles]
            return ells, coords, values, True

        if 'ells' not in kwargs or 'k' not in kwargs:
            raise ValueError('Array input requires both "ells" and "k".')
        ells = list(kwargs['ells'])
        values = cls._normalize_values(data, ells)
        coords = cls._normalize_coordinates(kwargs['k'], ells)
        if len(coords) != len(ells):
            raise ValueError(f'Expected {len(ells)} coordinate arrays, received {len(coords)}.')
        return ells, coords, values, False

    @staticmethod
    def _restore_output(multipoles, prototype, is_tree):
        if is_tree:
            return prototype.clone(value=Blinder._flatten_multipoles(multipoles))
        if len(multipoles) == 1 and isinstance(prototype, np.ndarray) and prototype.dtype != object:
            return np.asarray(multipoles[0])
        if isinstance(prototype, tuple):
            return tuple(multipoles)
        if isinstance(prototype, list):
            try:
                output = np.array(multipoles)
            except ValueError:
                return multipoles
            if output.dtype != object:
                return output
            return multipoles
        try:
            return np.array(multipoles)
        except ValueError:
            return multipoles

    @classmethod
    def _apply_shift_operation(
        cls,
        name: str,
        data,
        sign: float,
        save_dir: Path | str | None = None,
        shifts_fn: Path | str | None = None,
        **kwargs,
    ):
        public_name = cls._get_public_observable_name(name)
        internal_name = f'{public_name}_{cls._get_observable_suffix()}'
        save_fn = cls._get_shifts_fn(save_dir=save_dir, shifts_fn=shifts_fn)
        shifts_dict = np.load(save_fn, allow_pickle=True).item()
        bid = cls.__get_bid()
        key = hashlib.sha256(f'{internal_name}_bid{bid}'.encode()).hexdigest()
        legacy_key = hashlib.sha256(f'{public_name}_bid{bid}'.encode()).hexdigest()
        if key not in shifts_dict and legacy_key in shifts_dict:
            raise ValueError(
                f'Shifts file stores legacy unsuffixed keys for {public_name}; regenerate it with '
                f'statistic-qualified {cls._get_observable_suffix()} names.'
            )
        if key not in shifts_dict:
            raise ValueError(f'Cannot find the blinding value for {internal_name}')
        shifts = shifts_dict[key]

        ells, coords, values, is_tree = cls._get_multipoles(data, **kwargs)
        blinded_data = []
        for ell, coord, value in zip(ells, coords, values, strict=True):
            if ell not in shifts:
                raise ValueError(f'Cannot find a saved shift for multipole {ell} in {internal_name}.')
            shift_values = cls._evaluate_shift(ell, coord, shifts[ell])
            blinded_data.append(np.asarray(value) + sign * np.asarray(shift_values))
        return cls._restore_output(blinded_data, prototype=data, is_tree=is_tree)

    @classmethod
    def apply_blinding(
        cls,
        name: str,
        data: np.ndarray,
        save_dir: Path | str | None = None,
        shifts_fn: Path | str | None = None,
        **kwargs
    ) -> np.ndarray:
        """
        Apply blinding from a saved shifts file. This method can be used
        by general users to blind their data, without the need of instantiating
        the Blinder class.

        Parameters
        ----------
        name : str
            Bare canonical tracer-bin name used in the saved shifts file, e.g.
            'LRG1'. The blinder appends the statistic suffix internally.
        data : list, ObservableTree
            If list, list of multipoles; in this case also provide corresponding ``ells`` and ``k``.
            Else, an ObservableTree from lsstypes.

        Returns
        -------
        blinded_data : np.ndarray or ObservableTree
        """
        return cls._apply_shift_operation(
            name=name,
            data=data,
            sign=1.0,
            save_dir=save_dir,
            shifts_fn=shifts_fn,
            **kwargs,
        )

    @classmethod
    def remove_blinding(
        cls,
        name: str,
        data: np.ndarray,
        save_dir: Path | str | None = None,
        shifts_fn: Path | str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Remove blinding from a saved shifts file. This method can be used
        by general users to blind their data, without the need of instantiating
        the Blinder class.

        Parameters
        ----------
        name : str
            Bare canonical tracer-bin name used in the saved shifts file, e.g.
            'LRG1'. The blinder appends the statistic suffix internally.
        data : list, ObservableTree
            If list, list of multipoles; in this case also provide corresponding ``ells`` and ``k``.
            Else, an ObservableTree from lsstypes.

        Returns
        -------
        unblinded_data : np.ndarray or ObservableTree
        """
        if not kwargs.get('force', False):
            raise ValueError('Are you sure you want to unblind? If so, provide "force=True"')
        kwargs = {key: value for key, value in kwargs.items() if key != 'force'}
        return cls._apply_shift_operation(
            name=name,
            data=data,
            sign=-1.0,
            save_dir=save_dir,
            shifts_fn=shifts_fn,
            **kwargs,
        )


class TracerPowerSpectrumMultipolesBlinder(Blinder):

    observable_suffix = 'mesh2_spectrum'

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
        target_name = None if name is None else self._get_internal_observable_name(name)

        for observable in self.observables:
            if target_name is not None and observable.name != target_name:
                continue
            reference_data = observable.reference_data
            display_name = self._get_public_name_from_internal(observable.name)

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
                    label=display_name if ill == 0 else None, marker=marker, ms=2.5, ls='none', elinewidth=1.0,
                    zorder=2, mfc='k', mew=0.5)

        ax.legend()
        ax.set_xlabel(r'$k\,[h{\rm Mpc}^{-1}]$', fontsize=14)
        ax.set_ylabel(r'$k P_\ell(k)\,[h^{-2}{\rm Mpc}^{2}]$', fontsize=14)
        plt.tight_layout()
        return fig, ax


class TracerBispectrumMultipolesBlinder(Blinder):

    observable_suffix = 'mesh3_spectrum'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _get_diagonal_sugiyama_coordinate(ell, coordinates, label):
        """Return the 1D diagonal coordinate for Sugiyama-diagonal bispectrum poles."""
        coordinates = np.asarray(coordinates)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                f'Bispectrum blinding only supports diagonal Sugiyama basis multipoles; '
                f'{label} coordinates for multipole {ell} must have shape (n, 2), received {coordinates.shape}.'
            )
        if not np.allclose(coordinates[:, 1], coordinates[:, 0], rtol=0.0, atol=1e-12):
            raise ValueError(
                f'Bispectrum blinding only supports diagonal Sugiyama basis multipoles; '
                f'{label} coordinates for multipole {ell} are not diagonal.'
            )
        return coordinates[:, 0]

    @classmethod
    def _evaluate_shift(cls, ell, coordinates, shift):
        """Evaluate bispectrum shifts on the diagonal Sugiyama coordinate."""
        shift_coords, shift_values = shift
        shift_k = cls._get_diagonal_sugiyama_coordinate(ell, shift_coords, label='saved')
        input_k = cls._get_diagonal_sugiyama_coordinate(ell, coordinates, label='input')
        return InterpolatedUnivariateSpline(shift_k, shift_values, k=3)(input_k)
