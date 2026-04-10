import argparse
import copy
from functools import wraps

import numpy as np
import matplotlib.pyplot as plt


def set_plot_style(func):
    """Decorator to set the plotting style to acm standard."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        return func(*args, **kwargs)
    return wrapper


def replace_select_upper_limit(limits, new_max):
    """Return a selection range with the same lower bound / step and a new upper bound."""
    if not isinstance(limits, (list, tuple)) or len(limits) < 2:
        raise ValueError(f'Expected a list/tuple range like [kmin, kmax, step], received {limits!r}.')
    updated = list(limits)
    updated[1] = float(new_max)
    if isinstance(limits, tuple):
        return tuple(updated)
    return updated


def resolve_eval_kmax_overrides(eval_kmax=None, eval_kmax_mesh2=None, eval_kmax_mesh3=None):
    """Resolve the per-stat kmax overrides used when rebuilding observables."""
    default = None if eval_kmax is None else float(eval_kmax)
    return {
        'mesh2_spectrum': default if eval_kmax_mesh2 is None else float(eval_kmax_mesh2),
        'mesh3_spectrum': default if eval_kmax_mesh3 is None else float(eval_kmax_mesh3),
    }


def clear_eval_kmax_overrides(args):
    """Return a shallow copy of args with evaluation-range overrides cleared."""
    cleared = argparse.Namespace(**vars(args))
    for name in ('eval_kmax', 'eval_kmax_mesh2', 'eval_kmax_mesh3'):
        if hasattr(cleared, name):
            setattr(cleared, name, None)
    return cleared


def clear_eval_ells_overrides(args):
    """Return a shallow copy of args with evaluation-multipole overrides cleared."""
    cleared = argparse.Namespace(**vars(args))
    if hasattr(cleared, 'eval_ells_mesh2'):
        setattr(cleared, 'eval_ells_mesh2', None)
    return cleared


def apply_eval_kmax_override(options, eval_kmax_by_stat):
    """Override the upper k limit used when rebuilding observables for blinding."""
    for likelihood in options['likelihoods']:
        for observable in likelihood['observables']:
            new_max = eval_kmax_by_stat.get(observable['stat']['kind'])
            if new_max is None:
                continue
            for select in observable['stat'].get('select', []):
                if 'k' not in select:
                    continue
                select['k'] = replace_select_upper_limit(select['k'], new_max)
    return options


def apply_eval_ells_override(options, eval_ells_mesh2=None):
    """Override the power-spectrum multipoles used when rebuilding observables for blinding."""
    if eval_ells_mesh2 is None:
        return options
    eval_ells_mesh2 = tuple(dict.fromkeys(int(ell) for ell in eval_ells_mesh2))
    for likelihood in options['likelihoods']:
        for observable in likelihood['observables']:
            if observable['stat']['kind'] != 'mesh2_spectrum':
                continue
            selects = observable['stat'].get('select', [])
            existing_by_ell = {}
            passthrough = []
            for select in selects:
                ell = select.get('ells')
                if isinstance(ell, (int, np.integer)):
                    existing_by_ell[int(ell)] = copy.deepcopy(select)
                else:
                    passthrough.append(copy.deepcopy(select))
            if existing_by_ell:
                template_select = existing_by_ell[max(existing_by_ell)]
                updated_selects = []
                for ell in eval_ells_mesh2:
                    updated = copy.deepcopy(existing_by_ell.get(ell, template_select))
                    updated['ells'] = ell
                    updated_selects.append(updated)
                observable['stat']['select'] = updated_selects + passthrough
            elif selects:
                template_select = copy.deepcopy(selects[-1])
                observable['stat']['select'] = [
                    copy.deepcopy(template_select) | {'ells': ell}
                    for ell in eval_ells_mesh2
                ]
            else:
                observable['stat']['select'] = [{'ells': ell} for ell in eval_ells_mesh2]
            observable.setdefault('theory', {}).setdefault('options', {})['ells'] = eval_ells_mesh2
    return options


def _format_observable_label(label):
    """Return a compact string description for one observable leaf label."""
    return ', '.join(f'{name}={value!r}' for name, value in label.items())


def get_observable_k_support(observable):
    """Return the largest k center and upper edge available for each observable leaf."""
    support = []
    for label in observable.labels():
        pole = observable.get(**label)
        coords = np.asarray(pole.coords('k'))
        edges = np.asarray(pole.edges('k'))
        if coords.size == 0 or edges.size == 0:
            continue
        if edges.shape[-1] != 2:
            raise ValueError(f'Expected k edges with trailing shape (..., 2), received {edges.shape!r}.')
        support.append({
            'label': label,
            'max_center': float(np.nanmax(coords)),
            'max_edge': float(np.nanmax(edges[..., 1])),
        })
    if not support:
        raise ValueError('Could not determine the available k support for the rebuilt observable.')
    return support


def validate_eval_kmax(states, eval_kmax_by_stat):
    """Raise if the rebuilt measurement grid does not reach the requested evaluation kmax."""
    for state in states:
        eval_kmax = eval_kmax_by_stat.get(state['stat'])
        if eval_kmax is None:
            continue
        for support in get_observable_k_support(state['data']):
            if support['max_edge'] + 1e-12 < eval_kmax:
                label = _format_observable_label(support['label'])
                raise ValueError(
                    f'kmax={eval_kmax} exceeds the available measurement/window grid for '
                    f'{state["output_name"]} ({label}); the last bin center is k={support["max_center"]:.6g} '
                    f'and the supported upper bin edge is k={support["max_edge"]:.6g}.'
                )
