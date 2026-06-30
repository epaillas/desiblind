import tempfile

import numpy as np
from mockfactory import Catalog

from desiblind import CatalogRSDBlinder


PARAMS = {'w0': -0.95, 'wa': 0.10, 'zeff': 0.8, 'bias': 2.0, 'fiducial_f': 0.8}


def _lss_compute_fgrowth_blind(w0, wa, z=None, bias=None, fiducial_f=0.8, max_df_fraction=0.1, zeff=None):
    """Formula used in LSS/scripts/main/apply_blinding_* for fgrowth_blind."""
    from cosmoprimo.fiducial import DESI

    if z is None:
        z = zeff
    cosmo_fid = DESI()
    cosmo_shift = cosmo_fid.clone(w0_fld=w0, wa_fld=wa)
    dm_fid = cosmo_fid.comoving_angular_distance(z)
    dh_fid = 1. / cosmo_fid.hubble_function(z)
    dm_shift = cosmo_shift.comoving_angular_distance(z)
    dh_shift = 1. / cosmo_shift.hubble_function(z)
    vol_fac = (dm_shift**2 * dh_shift) / (dm_fid**2 * dh_fid)

    a = 0.2 / bias**2
    b = 2. / (3. * bias)
    c = 1. - (1. + 0.2 * (fiducial_f / bias)**2 + 2. / 3. * fiducial_f / bias) / vol_fac
    f_shift = (-b + np.sqrt(b**2 - 4. * a * c)) / (2. * a)
    dfper = (f_shift - fiducial_f) / fiducial_f
    if abs(dfper) > max_df_fraction:
        dfper = max_df_fraction * dfper / abs(dfper)
        f_shift = (1. + dfper) * fiducial_f
    return float(f_shift)


def _lss_apply_zshift_rsd_formula(z_observed, z_realspace, fgrowth_fid=0.8, fgrowth_blind=0.9):
    """Core formula from LSS.blinding_tools.apply_zshift_RSD, without IO."""
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.utils import DistanceToRedshift

    cosmo_fid = DESI()
    dis_fid = cosmo_fid.comoving_radial_distance
    dis_original = dis_fid(z_observed)
    dis_realspace = dis_fid(z_realspace)
    dis_blind = dis_realspace + (fgrowth_blind / fgrowth_fid) * (dis_original - dis_realspace)
    return DistanceToRedshift(dis_fid)(dis_blind)


def test_catalog_rsd_compute_fgrowth_blind_matches_lss_formula():
    expected = _lss_compute_fgrowth_blind(**PARAMS)
    actual = CatalogRSDBlinder.compute_fgrowth_blind(**PARAMS)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


def test_catalog_rsd_transform_redshift_matches_lss_formula():
    fgrowth_blind = CatalogRSDBlinder.compute_fgrowth_blind(**PARAMS)
    z_observed = np.array([0.76, 0.82, 0.95, 1.05])
    z_realspace = np.array([0.75, 0.80, 0.94, 1.02])
    expected = _lss_apply_zshift_rsd_formula(
        z_observed, z_realspace, fgrowth_fid=PARAMS['fiducial_f'], fgrowth_blind=fgrowth_blind
    )
    actual = CatalogRSDBlinder.transform_redshift(
        z_observed, z_realspace, fgrowth_fid=PARAMS['fiducial_f'], fgrowth_blind=fgrowth_blind
    )
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_catalog_rsd_apply_to_mockfactory_catalog():
    data = Catalog({
        'RA': np.array([10., 20., 30.]),
        'DEC': np.array([0., 5., 10.]),
        'Z': np.array([0.80, 0.95, 1.10]),
        'WEIGHT': np.ones(3),
    })
    realspace = Catalog({
        'RA': np.array([10., 20., 30.]),
        'DEC': np.array([0., 5., 10.]),
        'Z': np.array([0.79, 0.93, 1.08]),
    })
    blinded = CatalogRSDBlinder.apply_blinding('LRG3', data, realspace, parameters=PARAMS)
    expected = CatalogRSDBlinder.transform_redshift(
        data['Z'], realspace['Z'],
        fgrowth_fid=PARAMS['fiducial_f'],
        fgrowth_blind=CatalogRSDBlinder.compute_fgrowth_blind(**PARAMS),
    )

    assert blinded is not data
    np.testing.assert_allclose(blinded['RA'], data['RA'])
    np.testing.assert_allclose(blinded['DEC'], data['DEC'])
    np.testing.assert_allclose(blinded['Z'], expected, rtol=0, atol=1e-12)
    assert blinded.attrs['desiblind_catalog_blinding'] == 'catalog_rsd'
    assert 'catalog_rsd_fgrowth_blind' in blinded.attrs


def test_catalog_rsd_parameter_bank_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        save_fn = CatalogRSDBlinder.write_blinded_parameters('LRG3', PARAMS, save_dir=tmpdir)
        key = CatalogRSDBlinder.get_key('LRG3')
        bank = np.load(save_fn, allow_pickle=True).item()
        assert key in bank
        assert bank[key]['w0'] == PARAMS['w0']
        assert bank[key]['fgrowth_blind'] == CatalogRSDBlinder.compute_fgrowth_blind(**PARAMS)

        loaded = CatalogRSDBlinder.load_blinded_parameters('LRG3', save_dir=tmpdir)
        assert loaded['w0'] == PARAMS['w0']
        assert loaded['fgrowth_blind'] == bank[key]['fgrowth_blind']
