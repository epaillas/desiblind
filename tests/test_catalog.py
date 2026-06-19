import hashlib
import tempfile

import numpy as np
import pytest
from mockfactory import Catalog

from desiblind import CatalogBAOBlinder


PARAMS = {'w0': -0.95, 'wa': 0.10}


def _lss_apply_zshift_de_formula(z, w0, wa):
    """Core formula from LSS.blinding_tools.apply_zshift_DE, without IO."""
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.utils import DistanceToRedshift

    cosmo_fid = DESI()
    cosmo_blind = cosmo_fid.clone(w0_fld=w0, wa_fld=wa)
    z = np.asarray(z, dtype='f8')
    out = z.copy()
    sel = z * 0 == 0
    out[sel] = DistanceToRedshift(cosmo_fid.comoving_radial_distance)(
        cosmo_blind.comoving_radial_distance(z[sel])
    )
    return out


def test_catalog_bao_redshift_transform_inverse():
    z = np.array([0.8, 0.95, 1.1, np.nan])
    z_blinded = CatalogBAOBlinder.transform_redshift(z, **PARAMS)
    assert z_blinded.shape == z.shape
    assert np.isnan(z_blinded[-1])
    assert not np.allclose(z_blinded[:-1], z[:-1])

    z_unblinded = CatalogBAOBlinder.transform_redshift(z_blinded, inverse=True, **PARAMS)
    np.testing.assert_allclose(z_unblinded[:-1], z[:-1], rtol=0, atol=1e-8)
    assert np.isnan(z_unblinded[-1])


def test_catalog_bao_matches_lss_apply_zshift_de_formula():
    z = np.array([0.4, 0.8, 1.1, np.nan])
    expected = _lss_apply_zshift_de_formula(z, **PARAMS)
    actual = CatalogBAOBlinder.transform_redshift(z, **PARAMS)
    np.testing.assert_allclose(actual[:-1], expected[:-1], rtol=0, atol=1e-12)
    assert np.isnan(actual[-1])


def test_catalog_bao_name_normalization_matches_blinder_convention():
    assert CatalogBAOBlinder._get_internal_name('LRG3') == 'LRG3_catalog_bao'
    with pytest.raises(ValueError, match='bare canonical'):
        CatalogBAOBlinder.get_key('LRG_z2')
    with pytest.raises(ValueError, match='do not include'):
        CatalogBAOBlinder.get_key('LRG3_catalog_bao')


def test_catalog_bao_apply_to_mockfactory_catalog():
    catalog = Catalog({
        'RA': np.array([10., 20., 30.]),
        'DEC': np.array([0., 5., 10.]),
        'Z': np.array([0.8, 0.95, 1.1]),
        'WEIGHT': np.ones(3),
    })
    blinded = CatalogBAOBlinder.apply_blinding('LRG3', catalog, parameters=PARAMS)

    assert blinded is not catalog
    np.testing.assert_allclose(blinded['RA'], catalog['RA'])
    np.testing.assert_allclose(blinded['DEC'], catalog['DEC'])
    assert not np.allclose(blinded['Z'], catalog['Z'])
    assert blinded.attrs['desiblind_catalog_blinding'] == 'catalog_bao'

    unblinded = CatalogBAOBlinder.remove_blinding('LRG3', blinded, parameters=PARAMS, force=True)
    np.testing.assert_allclose(unblinded['Z'], catalog['Z'], rtol=0, atol=1e-8)


def test_catalog_bao_lss_input_output_redshift_columns():
    catalog = Catalog({
        'RA': np.array([10., 20., 30.]),
        'DEC': np.array([0., 5., 10.]),
        'Z': np.array([0.81, 0.96, 1.11]),
        'Z_not4clus': np.array([0.8, 0.95, 1.1]),
        'WEIGHT': np.ones(3),
    })
    blinded = CatalogBAOBlinder.apply_blinding(
        'LRG3', catalog, parameters=PARAMS, input_zcol='Z_not4clus', output_zcol='Z'
    )

    expected = _lss_apply_zshift_de_formula(catalog['Z_not4clus'], **PARAMS)
    np.testing.assert_allclose(blinded['Z'], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(blinded['Z_not4clus'], catalog['Z_not4clus'])
    assert not np.allclose(blinded['Z'], catalog['Z'])


def test_catalog_bao_parameter_bank_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_fn = CatalogBAOBlinder.write_blinded_parameters('LRG3', PARAMS, save_dir=tmpdir)
        key = CatalogBAOBlinder.get_key('LRG3')
        bank = np.load(save_fn, allow_pickle=True).item()
        assert key in bank
        assert bank[key] == PARAMS

        loaded = CatalogBAOBlinder.load_blinded_parameters('LRG3', save_dir=tmpdir)
        assert loaded == PARAMS

        with pytest.raises(FileExistsError):
            CatalogBAOBlinder.write_blinded_parameters('LRG3', PARAMS, save_dir=tmpdir)

        CatalogBAOBlinder.write_blinded_parameters('LRG3', PARAMS | {'w0': -0.97}, save_dir=tmpdir, overwrite=True)
        loaded = CatalogBAOBlinder.load_blinded_parameters('LRG3', save_dir=tmpdir)
        assert loaded['w0'] == -0.97


def test_catalog_bao_parameter_bank_rejects_legacy_unsuffixed_key():
    with tempfile.TemporaryDirectory() as tmpdir:
        key = hashlib.sha256(f'LRG3_bid{CatalogBAOBlinder._get_bid()}'.encode()).hexdigest()
        parameters_fn = f'{tmpdir}/legacy.npy'
        np.save(parameters_fn, {key: PARAMS})
        with pytest.raises(ValueError, match='legacy unsuffixed keys'):
            CatalogBAOBlinder.load_blinded_parameters('LRG3', parameters_fn=parameters_fn)
