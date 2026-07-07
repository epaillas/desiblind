import tempfile

import numpy as np
import pytest
from mockfactory import Catalog

from desiblind import CatalogFNLBlinder


PARAMS = {'fnl': 7.5, 'zeff': 0.8, 'bias': 2.0}


def _data_catalog():
    return Catalog({
        'RA': np.array([10., 20., 30.]),
        'DEC': np.array([0., 5., 10.]),
        'Z': np.array([0.80, 0.95, 1.10]),
        'WEIGHT': np.array([1.0, 2.0, 4.0]),
        'WEIGHT_COMP': np.array([0.5, 0.75, 1.25]),
    })


def _random_catalog():
    return Catalog({
        'RA': np.array([12., 22., 32., 42.]),
        'DEC': np.array([1., 6., 11., 16.]),
        'Z': np.array([0.81, 0.96, 1.08, 1.18]),
        'WEIGHT': np.array([1.0, 1.1, 1.2, 1.3]),
    })


class FakeMockfactoryBlinding:
    def __init__(self, factor=1.05):
        self.factor = factor
        self.calls = []

    def png(self, data_positions, data_weights=None, randoms_positions=None,
            randoms_weights=None, method=None, shotnoise_correction=None, **kwargs):
        self.calls.append({
            'data_positions': data_positions,
            'data_weights': data_weights,
            'randoms_positions': randoms_positions,
            'randoms_weights': randoms_weights,
            'method': method,
            'shotnoise_correction': shotnoise_correction,
            'kwargs': kwargs,
        })
        return np.asarray(data_weights) * self.factor


def test_catalog_fnl_generate_from_index_matches_lss_random_state():
    index = 17
    expected = np.random.RandomState(index).uniform(low=-15, high=15, size=1)[0]
    actual = CatalogFNLBlinder.generate_fnl_from_index(index)
    assert actual == expected


def test_catalog_fnl_infers_lss_tracer_defaults():
    assert CatalogFNLBlinder.infer_tracer_defaults('LRG') == {'zeff': 0.8, 'bias': 2.0}
    assert CatalogFNLBlinder.infer_tracer_defaults('ELG_LOPnotqso') == {'zeff': 1.1, 'bias': 1.3}
    assert CatalogFNLBlinder.infer_tracer_defaults('QSO') == {'zeff': 1.6, 'bias': 2.3}
    assert CatalogFNLBlinder.infer_tracer_defaults('BGS_BRIGHT-21.5') == {'zeff': 0.25, 'bias': 1.8}


def test_catalog_fnl_normalizes_index_with_tracer_defaults():
    params = CatalogFNLBlinder._normalize_parameters({'index': 3}, tracer='LRG3')
    assert params['fnl'] == CatalogFNLBlinder.generate_fnl_from_index(3)
    assert params['zeff'] == 0.8
    assert params['bias'] == 2.0
    assert params['method'] == 'data_weights'
    assert params['shotnoise_correction'] is True


def test_catalog_fnl_name_normalization_matches_blinder_convention():
    assert CatalogFNLBlinder._get_internal_name('LRG3') == 'LRG3_catalog_fnl'
    with pytest.raises(ValueError, match='bare canonical'):
        CatalogFNLBlinder.get_key('LRG3_catalog_fnl')


def test_catalog_fnl_compute_weight_factor_fnl_zero_short_circuit():
    data = _data_catalog()
    randoms = _random_catalog()
    factor = CatalogFNLBlinder.compute_weight_factor(
        CatalogFNLBlinder._catalog_positions(data), data['WEIGHT'],
        CatalogFNLBlinder._catalog_positions(randoms), randoms['WEIGHT'],
        {'fnl': 0.0, 'zeff': 0.8, 'bias': 2.0},
    )
    np.testing.assert_allclose(factor, np.ones(len(data)))


def test_catalog_fnl_apply_to_catalog_uses_mockfactory_png_and_keeps_factor_internal(monkeypatch):
    fake = FakeMockfactoryBlinding(factor=1.05)

    def fake_build(cls, parameters, **kwargs):
        assert parameters['fnl'] == PARAMS['fnl']
        assert parameters['zeff'] == PARAMS['zeff']
        assert parameters['bias'] == PARAMS['bias']
        assert kwargs['position_type'] == 'rdz'
        return fake

    monkeypatch.setattr(CatalogFNLBlinder, '_build_mockfactory_blinding', classmethod(fake_build))
    data = _data_catalog()
    randoms = _random_catalog()
    blinded, factor = CatalogFNLBlinder.apply_blinding(
        'LRG3', data, randoms, parameters=PARAMS, return_weight_factor=True,
    )

    assert blinded is not data
    np.testing.assert_allclose(factor, np.full(len(data), 1.05))
    np.testing.assert_allclose(blinded['WEIGHT'], data['WEIGHT'] * 1.05)
    np.testing.assert_allclose(blinded['WEIGHT_COMP'], data['WEIGHT_COMP'] * 1.05)
    assert set(blinded.columns()) == set(data.columns())
    assert blinded.attrs['desiblind_catalog_blinding'] == 'catalog_fnl'
    assert blinded.attrs['catalog_fnl_fnl'] == PARAMS['fnl']

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call['method'] == 'data_weights'
    assert call['shotnoise_correction'] is True
    np.testing.assert_allclose(call['data_positions'][0], data['RA'])
    np.testing.assert_allclose(call['randoms_positions'][2], randoms['Z'])
    np.testing.assert_allclose(call['randoms_weights'], randoms['WEIGHT'])


def test_catalog_fnl_parameter_bank_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_fn = CatalogFNLBlinder.write_blinded_parameters('LRG3', PARAMS, save_dir=tmpdir)
        key = CatalogFNLBlinder.get_key('LRG3')
        bank = np.load(save_fn, allow_pickle=True).item()
        assert key in bank
        assert bank[key]['fnl'] == PARAMS['fnl']
        assert bank[key]['zeff'] == PARAMS['zeff']
        assert bank[key]['bias'] == PARAMS['bias']

        loaded = CatalogFNLBlinder.load_blinded_parameters('LRG3', save_dir=tmpdir)
        assert loaded['fnl'] == PARAMS['fnl']
        assert loaded['zeff'] == PARAMS['zeff']
        assert loaded['bias'] == PARAMS['bias']

        with pytest.raises(FileExistsError):
            CatalogFNLBlinder.write_blinded_parameters('LRG3', PARAMS, save_dir=tmpdir)


def test_catalog_fnl_apply_to_astropy_table_column_handling(monkeypatch):
    from astropy.table import Table

    fake = FakeMockfactoryBlinding(factor=1.02)

    def fake_build(cls, parameters, **kwargs):
        return fake

    monkeypatch.setattr(CatalogFNLBlinder, '_build_mockfactory_blinding', classmethod(fake_build))
    data = Table({
        'RA': np.array([10., 20.]),
        'DEC': np.array([0., 5.]),
        'Z': np.array([0.8, 0.9]),
        'WEIGHT': np.array([1.0, 2.0]),
        'WEIGHT_COMP': np.array([0.5, 0.75]),
    })
    randoms = Table({
        'RA': np.array([11., 21.]),
        'DEC': np.array([1., 6.]),
        'Z': np.array([0.81, 0.91]),
        'WEIGHT': np.ones(2),
    })
    blinded = CatalogFNLBlinder.apply_blinding('LRG3', data, randoms, parameters=PARAMS)
    np.testing.assert_allclose(blinded['WEIGHT'], data['WEIGHT'] * 1.02)
    np.testing.assert_allclose(blinded['WEIGHT_COMP'], data['WEIGHT_COMP'] * 1.02)
    assert set(blinded.colnames) == set(data.colnames)
