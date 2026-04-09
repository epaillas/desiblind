import re


LEGACY_TO_TRACERBIN = {
    'BGS_z0': 'BGS1',
    'LRG_z0': 'LRG1',
    'LRG_z1': 'LRG2',
    'LRG_z2': 'LRG3',
    'ELG_z0': 'ELG1',
    'ELG_z1': 'ELG2',
    'QSO_z0': 'QSO1',
}
TRACERBIN_TO_LEGACY = {value: key for key, value in LEGACY_TO_TRACERBIN.items()}
_TRACER_NAME_RE = re.compile(r'^(?P<prefix>[A-Za-z]+)(?:_z(?P<legacy>\d+)|(?P<tracerbin>\d+))$')
_PREFIXED_TRACER_NAME_RE = re.compile(r'[A-Za-z]+(?:_z\d+|\d+)')


def _normalize_case(name):
    match = _TRACER_NAME_RE.match(str(name))
    if match is None:
        return str(name)
    prefix = match.group('prefix').upper()
    legacy = match.group('legacy')
    tracerbin = match.group('tracerbin')
    if legacy is not None:
        return f'{prefix}_z{int(legacy)}'
    return f'{prefix}{int(tracerbin)}'


def _split_prefixed_tracer_name(name):
    text = str(name)
    tokens = []
    pos = 0
    while True:
        match = _PREFIXED_TRACER_NAME_RE.match(text, pos)
        if match is None:
            break
        tokens.append(match.group(0))
        pos = match.end()
        if pos < len(text) and text[pos] == 'x':
            pos += 1
            continue
        break
    if not tokens:
        return None, text
    if pos < len(text) and text[pos].isalnum():
        return None, text
    return tokens, text[pos:]


def _convert_prefixed_tracer_name(name, converter):
    tokens, suffix = _split_prefixed_tracer_name(name)
    if tokens is None:
        return str(name)
    return 'x'.join(converter(token) for token in tokens) + suffix


def to_legacy_tracer_name(name):
    """Return the legacy ``TRACER_zN`` form for a tracer or observable name."""
    return _convert_prefixed_tracer_name(
        name,
        lambda token: TRACERBIN_TO_LEGACY.get(_normalize_case(token), _normalize_case(token)),
    )


def to_tracerbin_name(name):
    """Return the ``TRACER{i+1}`` form for a tracer or observable name."""
    return _convert_prefixed_tracer_name(
        name,
        lambda token: LEGACY_TO_TRACERBIN.get(_normalize_case(token), _normalize_case(token)),
    )


def normalize_canonical_tracerbin_name(name):
    """Return a bare canonical tracer-bin name, rejecting legacy aliases and suffixed observables."""
    tokens, suffix = _split_prefixed_tracer_name(name)
    if tokens is None or suffix:
        raise ValueError(f'{name!r} is not a bare tracer name.')
    normalized = 'x'.join(_normalize_case(token) for token in tokens)
    if to_tracerbin_name(normalized) != normalized:
        raise ValueError(f'{name!r} is not a canonical tracer-bin name.')
    return normalized


def get_tracer_name_aliases(name):
    """Return recognized aliases for a tracer or observable name."""
    legacy = to_legacy_tracer_name(name)
    tracerbin = to_tracerbin_name(name)
    aliases = [tracerbin]
    if legacy != tracerbin:
        aliases.append(legacy)
    return tuple(aliases)


def normalize_tracer_names(names, output='legacy'):
    """Normalize an iterable of tracer names to one convention."""
    if output not in {'legacy', 'tracerbin'}:
        raise ValueError(f'Unknown tracer name output convention: {output}')
    converter = to_legacy_tracer_name if output == 'legacy' else to_tracerbin_name
    return [converter(name) for name in names]
