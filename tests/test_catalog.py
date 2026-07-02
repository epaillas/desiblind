import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from mockfactory import Catalog

from desiblind import CatalogBAOBlinder, CatalogRSDBlinder


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


def test_catalog_bao_alpha_shift_validation():
    stats = CatalogBAOBlinder.validate_alpha_shift(PARAMS)
    assert stats['max_abs_alpha_parallel_minus_one'] < 0.03
    assert stats['max_abs_alpha_perp_minus_one'] < 0.03

    with pytest.raises(ValueError, match='outside the allowed DESI 3% alpha-shift region'):
        CatalogBAOBlinder.validate_alpha_shift({'w0': -0.90, 'wa': 0.26})

    with pytest.raises(ValueError, match=r'w0 \+ wa'):
        CatalogBAOBlinder.validate_alpha_shift({'w0': -0.20, 'wa': 0.30})



def test_catalog_bao_generate_parameters():
    parameters, metadata = CatalogBAOBlinder.generate_parameters(
        seed=123,
        w0_range=(-0.96, -0.94),
        wa_range=(0.08, 0.12),
        max_attempts=10,
    )
    assert metadata['parameter_source'] == 'desiblind_generated'
    assert metadata['generator'] == 'uniform_rejection'
    assert metadata['seed'] == 123
    assert metadata['accepted_attempt'] >= 1
    assert -0.96 <= parameters['w0'] <= -0.94
    assert 0.08 <= parameters['wa'] <= 0.12
    CatalogBAOBlinder.validate_alpha_shift(parameters)

    with pytest.raises(ValueError, match='explicit seed'):
        CatalogBAOBlinder.generate_parameters(seed=None)

    with pytest.raises(RuntimeError, match='Could not generate valid'):
        CatalogBAOBlinder.generate_parameters(
            seed=123,
            w0_range=(-0.20, -0.10),
            wa_range=(0.30, 0.40),
            max_attempts=3,
        )

def test_catalog_bao_lss_parameter_bank_loader(tmp_path):
    bank_fn = tmp_path / 'w0wa.txt'
    np.savetxt(bank_fn, np.array([[-0.95, 0.10], [-0.90, 0.03]]))
    filerow = tmp_path / 'filerow.txt'
    filerow.write_text('1\n')

    loaded = CatalogBAOBlinder.load_lss_parameters(parameters_fn=bank_fn, index=0)
    assert loaded['w0'] == -0.95
    assert loaded['wa'] == 0.10
    assert loaded['index'] == 0

    loaded = CatalogBAOBlinder.load_lss_parameters(parameters_fn=bank_fn, filerow=filerow)
    assert loaded['w0'] == -0.90
    assert loaded['wa'] == 0.03
    assert loaded['index'] == 1

    with pytest.raises(ValueError, match='exactly one'):
        CatalogBAOBlinder.load_lss_parameters(parameters_fn=bank_fn)


def test_catalog_bao_load_parameters_sources(tmp_path):
    bank_fn = tmp_path / 'catalog_bank.npy'
    CatalogBAOBlinder.write_blinded_parameters('LRG1', PARAMS, parameters_fn=bank_fn)

    parameters, metadata = CatalogBAOBlinder.load_parameters(
        name='LRG1', parameters_fn=bank_fn, source='desiblind', bid=CatalogBAOBlinder._get_bid()
    )
    assert parameters == PARAMS
    assert metadata['parameter_source'] == 'desiblind'
    assert metadata['name'] == 'LRG1'

    lss_bank_fn = tmp_path / 'w0wa.txt'
    np.savetxt(lss_bank_fn, np.array([[-0.95, 0.10]]))
    parameters, metadata = CatalogBAOBlinder.load_parameters(
        source='lss', lss_parameters_fn=lss_bank_fn, lss_parameter_index=0
    )
    assert parameters == PARAMS
    assert metadata['parameter_source'] == 'lss'
    assert metadata['index'] == 0

    parameters, metadata = CatalogBAOBlinder.load_parameters(parameters=PARAMS)
    assert parameters == PARAMS
    assert metadata['parameter_source'] == 'explicit'



def test_create_catalog_w0wa_blinding_bank_script(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'create_catalog_w0wa_blinding_bank.py'
    bank_fn = tmp_path / 'catalog_blinding_parameters.npy'
    record_fn = tmp_path / 'private_record.json'

    subprocess.run(
        [
            sys.executable,
            str(script),
            '--output', str(bank_fn),
            '--bid', '7',
            '--w0', '-0.95',
            '--wa', '0.10',
            '--tracer-bins', 'LRG1', 'ELG1',
            '--record-fn', str(record_fn),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )

    assert bank_fn.exists()
    assert record_fn.exists()
    assert CatalogBAOBlinder.load_blinded_parameters('LRG1', parameters_fn=bank_fn, bid=7) == PARAMS
    assert CatalogBAOBlinder.load_blinded_parameters('ELG1', parameters_fn=bank_fn, bid=7) == PARAMS
    record = json.loads(record_fn.read_text())
    assert record['bid'] == 7
    assert record['parameters'] == PARAMS
    assert record['source']['parameter_source'] == 'explicit'
    assert record['effects'] == ['bao']
    assert record['outputs']['bao'] == str(bank_fn)

    generated_bank_fn = tmp_path / 'generated_catalog_bao_blinding_parameters.npy'
    generated_rsd_bank_fn = tmp_path / 'generated_catalog_rsd_blinding_parameters.npy'
    generated_record_fn = tmp_path / 'generated_private_record.json'
    subprocess.run(
        [
            sys.executable,
            str(script),
            '--effects', 'bao', 'rsd',
            '--bao-output', str(generated_bank_fn),
            '--rsd-output', str(generated_rsd_bank_fn),
            '--bid', '8',
            '--generate',
            '--seed', '123',
            '--w0-range=-0.96,-0.94',
            '--wa-range=0.08,0.12',
            '--tracer-bins', 'LRG1',
            '--rsd-bin', 'LRG1:0.8:2.0',
            '--record-fn', str(generated_record_fn),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    generated_record = json.loads(generated_record_fn.read_text())
    assert generated_record['effects'] == ['bao', 'rsd']
    assert generated_record['source']['parameter_source'] == 'desiblind_generated'
    assert generated_record['source']['seed'] == 123
    generated_params = generated_record['parameters']
    assert CatalogBAOBlinder.load_blinded_parameters('LRG1', parameters_fn=generated_bank_fn, bid=8) == generated_params
    CatalogBAOBlinder.validate_alpha_shift(generated_params)
    rsd_params = CatalogRSDBlinder.load_blinded_parameters('LRG1', parameters_fn=generated_rsd_bank_fn, bid=8)
    assert rsd_params['w0'] == generated_params['w0']
    assert rsd_params['wa'] == generated_params['wa']
    assert rsd_params['zeff'] == 0.8
    assert rsd_params['bias'] == 2.0
    assert rsd_params['fgrowth_blind'] == generated_record['rsd']['tracer_bins'][0]['parameters']['fgrowth_blind']

    dry_run_fn = tmp_path / 'dryrun.npy'
    subprocess.run(
        [
            sys.executable,
            str(script),
            '--output', str(dry_run_fn),
            '--bid', '7',
            '--w0', '-0.95',
            '--wa', '0.10',
            '--dry-run',
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert not dry_run_fn.exists()
