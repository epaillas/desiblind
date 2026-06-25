import builtins
import sys
import types

import numpy as np
import pytest

import desiblind.catalog as catalog_module
from desiblind.catalog import (
    TracerCatalogBlinder,
    _catalog_parameter_key,
    generate_catalog_parameters,
)


class ToyCatalog(dict):
    def __init__(self, **columns):
        super().__init__(columns)
        self.attrs = {}
        self.mpicomm = None

    def copy(self):
        new = type(self)(**{key: np.array(value, copy=True) for key, value in self.items()})
        new.attrs = dict(self.attrs)
        new.mpicomm = self.mpicomm
        return new


def test_catalog_secret_loading_is_deterministic(tmp_path):
    fn = tmp_path / "analysis_catalog_blinding.npy"
    generate_catalog_parameters(
        parameters_fn=fn,
        w0=-0.99,
        wa=0.05,
        fnl_blind=7.0,
        validate=False,
        overwrite=True,
    )

    params = TracerCatalogBlinder.load_parameters(
        modes=["bao", "fnl"],
        parameters_fn=fn,
        metadata="sealed",
        validate=False,
    )
    bid = np.random.RandomState(seed=42).randint(0, TracerCatalogBlinder.blinded_nmax)
    assert params["bid"] == bid
    assert params["parameters_key"] == _catalog_parameter_key(bid)
    assert params["w0"] == pytest.approx(-0.99)
    assert params["wa"] == pytest.approx(0.05)
    assert params["fnl_blind"] == pytest.approx(7.0)


def test_catalog_default_filename_and_save_dir_generation(tmp_path):
    fn = TracerCatalogBlinder.get_parameters_fn(save_dir=tmp_path)
    assert fn == tmp_path / "catalog_blinding.npy"

    written = generate_catalog_parameters(
        save_dir=tmp_path,
        w0=-0.99,
        wa=0.05,
        fnl_blind=7.0,
        validate=False,
        overwrite=True,
    )
    assert written == fn
    assert written.exists()


def test_catalog_legacy_filename_fallback_warns(tmp_path):
    legacy_fn = tmp_path / "catalog_blinding_2026_06.npy"
    generate_catalog_parameters(
        parameters_fn=legacy_fn,
        w0=-0.99,
        wa=0.05,
        fnl_blind=7.0,
        validate=False,
        overwrite=True,
    )

    with pytest.warns(FutureWarning, match="legacy catalog blinding parameter file"):
        params = TracerCatalogBlinder.load_parameters(
            modes=["bao"],
            save_dir=tmp_path,
            metadata="sealed",
            validate=False,
        )
    assert params["parameters_fn"] == str(legacy_fn)


def test_catalog_generic_default_takes_precedence_over_legacy(tmp_path):
    legacy_fn = tmp_path / "catalog_blinding_2026_06.npy"
    generic_fn = tmp_path / "catalog_blinding.npy"
    generate_catalog_parameters(
        parameters_fn=legacy_fn,
        w0=-0.99,
        wa=0.05,
        fnl_blind=7.0,
        validate=False,
        overwrite=True,
    )
    generate_catalog_parameters(
        parameters_fn=generic_fn,
        w0=-0.98,
        wa=0.04,
        fnl_blind=6.0,
        validate=False,
        overwrite=True,
    )

    params = TracerCatalogBlinder.load_parameters(
        modes=["bao", "fnl"],
        save_dir=tmp_path,
        metadata="sealed",
        validate=False,
    )
    assert params["parameters_fn"] == str(generic_fn)
    assert params["w0"] == pytest.approx(-0.98)
    assert params["fnl_blind"] == pytest.approx(6.0)


def test_catalog_metadata_and_output_suffix_do_not_expose_sealed_values(tmp_path):
    fn = tmp_path / "catalog.npy"
    generate_catalog_parameters(
        parameters_fn=fn,
        w0=-0.99,
        wa=0.05,
        fnl_blind=7.0,
        validate=False,
        overwrite=True,
    )
    sealed = TracerCatalogBlinder.load_parameters(
        modes=["ap", "png"],
        parameters_fn=fn,
        metadata="sealed",
        validate=False,
    )
    sealed_attrs = TracerCatalogBlinder.blinding_attrs(sealed)
    assert sealed_attrs["catalog_blinding"] == "bao,fnl"
    assert "catalog_blinding_w0" not in sealed_attrs
    assert "catalog_blinding_fnl_blind" not in sealed_attrs
    assert sealed_attrs["catalog_blinding_parameters_file"] == "catalog.npy"
    assert "catalog_blinding_parameters_file_sha256" in sealed_attrs

    opened = TracerCatalogBlinder.load_parameters(
        modes=["bao", "fnl"],
        parameters_fn=fn,
        metadata="open",
        validate=False,
    )
    opened_attrs = TracerCatalogBlinder.blinding_attrs(opened)
    assert opened_attrs["catalog_blinding_w0"] == pytest.approx(-0.99)
    assert opened_attrs["catalog_blinding_fnl_blind"] == pytest.approx(7.0)
    assert TracerCatalogBlinder.output_version("data-dr2-v2", sealed) == "data-dr2-v2-bao-fnl-blinded"


def test_bao_redshift_remapping_matches_direct_cosmoprimo():
    pytest.importorskip("cosmoprimo")
    from cosmoprimo.fiducial import DESI
    from cosmoprimo.utils import DistanceToRedshift

    catalog = ToyCatalog(Z=np.array([0.4, 0.8, np.nan]))
    params = {
        "modes": ("bao",),
        "w0": -0.99,
        "wa": 0.05,
        "metadata": "sealed",
        "parameter_mode": "secret_file",
        "parameters_fn": "catalog.npy",
        "parameters_key": "hidden",
    }
    blinded = TracerCatalogBlinder.apply_bao_blinding(catalog, params)

    cosmo_fid = DESI()
    cosmo_blind = cosmo_fid.clone(w0_fld=params["w0"], wa_fld=params["wa"])
    expected = catalog["Z"].copy()
    mask = np.isfinite(expected)
    expected[mask] = DistanceToRedshift(cosmo_fid.comoving_radial_distance)(
        cosmo_blind.comoving_radial_distance(expected[mask])
    )
    actual = TracerCatalogBlinder.transform_redshift(catalog["Z"], w0=params["w0"], wa=params["wa"])
    np.testing.assert_allclose(actual[mask], expected[mask], rtol=0, atol=1e-12)
    assert np.allclose(blinded["Z"][mask], expected[mask])
    assert np.isnan(blinded["Z"][-1])


def test_bao_redshift_transform_inverse_and_scalar():
    pytest.importorskip("cosmoprimo")

    z = np.array([0.4, 0.8, 1.1, np.nan])
    blinded = TracerCatalogBlinder.transform_redshift(z, w0=-0.99, wa=0.05)
    assert blinded.shape == z.shape
    assert not np.allclose(blinded[:-1], z[:-1])
    assert np.isnan(blinded[-1])

    unblinded = TracerCatalogBlinder.transform_redshift(blinded, w0=-0.99, wa=0.05, inverse=True)
    np.testing.assert_allclose(unblinded[:-1], z[:-1], rtol=0, atol=1e-8)
    assert np.isnan(unblinded[-1])

    scalar = TracerCatalogBlinder.transform_redshift(0.8, w0=-0.99, wa=0.05)
    assert isinstance(scalar, float)


def test_bao_redshift_remapping_supports_input_and_output_columns():
    pytest.importorskip("cosmoprimo")

    catalog = ToyCatalog(
        Z=np.array([0.41, 0.81, 1.11]),
        Z_not4clus=np.array([0.4, 0.8, 1.1]),
    )
    params = {
        "modes": ("bao",),
        "w0": -0.99,
        "wa": 0.05,
        "metadata": "sealed",
        "parameter_mode": "secret_file",
        "parameters_fn": "catalog.npy",
        "parameters_key": "hidden",
    }
    blinded = TracerCatalogBlinder.apply_bao_blinding(
        catalog,
        params,
        input_zcol="Z_not4clus",
        output_zcol="Z",
    )

    expected = TracerCatalogBlinder.transform_redshift(catalog["Z_not4clus"], w0=params["w0"], wa=params["wa"])
    np.testing.assert_allclose(blinded["Z"], expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(blinded["Z_not4clus"], catalog["Z_not4clus"])
    assert not np.allclose(blinded["Z"], catalog["Z"])


def test_bao_redshift_remapping_accepts_direct_parameters_and_alias_modes():
    pytest.importorskip("cosmoprimo")

    catalog = ToyCatalog(Z=np.array([0.4, 0.8, 1.1]))
    params = {"modes": ("ap",), "w0": -0.99, "wa": 0.05}
    blinded = TracerCatalogBlinder.apply_bao_blinding(catalog, params)
    expected = TracerCatalogBlinder.apply_bao_blinding(catalog, {"w0": -0.99, "wa": 0.05})

    np.testing.assert_allclose(blinded["Z"], expected["Z"], rtol=0, atol=1e-12)
    assert blinded.attrs["catalog_blinding"] == "bao"
    assert "catalog_blinding_parameters_file" not in expected.attrs


def test_bao_blinding_removal_requires_force_and_roundtrips():
    pytest.importorskip("cosmoprimo")

    catalog = ToyCatalog(Z=np.array([0.4, 0.8, 1.1, np.nan]))
    params = {
        "modes": ("bao",),
        "w0": -0.99,
        "wa": 0.05,
        "metadata": "sealed",
        "parameter_mode": "secret_file",
        "parameters_fn": "catalog.npy",
        "parameters_key": "hidden",
    }
    with pytest.raises(ValueError, match="force=True"):
        TracerCatalogBlinder.remove_bao_blinding(catalog, params)

    blinded = TracerCatalogBlinder.apply_bao_blinding(catalog, params)
    unblinded = TracerCatalogBlinder.remove_bao_blinding(blinded, params, force=True)
    mask = np.isfinite(catalog["Z"])
    np.testing.assert_allclose(unblinded["Z"][mask], catalog["Z"][mask], rtol=0, atol=1e-8)
    assert np.isnan(unblinded["Z"][-1])


def _make_jax_blinding_case():
    from cosmoprimo.fiducial import DESI
    from mockfactory import Catalog

    z, bias, seed = 1.0, 2.0, 11
    cosmo = DESI()
    fourier = cosmo.get_fourier()
    f_fid = fourier.sigma8_z(z, of="theta_cb") / fourier.sigma8_z(z, of="delta_cb")
    fgrowth_blind = 0.8 * f_fid
    fnl_blind = 10.0

    rng = np.random.RandomState(seed)
    boxcenter = np.array([0.0, 0.0, cosmo.comoving_radial_distance(z)])
    data_positions = rng.uniform(-50.0, 50.0, size=(300, 3)) + boxcenter
    randoms_positions = rng.uniform(-50.0, 50.0, size=(1500, 3)) + boxcenter
    data_weights = np.clip(1.0 + 0.1 * rng.normal(size=len(data_positions)), 0.5, 1.5)
    randoms_weights = np.ones(len(randoms_positions))
    mesh_kwargs = {"nmesh": 24, "boxsize": 140.0, "boxcenter": boxcenter}

    data = Catalog(
        {
            "POSITION": data_positions,
            "INDWEIGHT": data_weights,
            "RA": np.zeros(len(data_positions)),
            "DEC": np.zeros(len(data_positions)),
            "Z": np.zeros(len(data_positions)),
        }
    )
    randoms = [
        Catalog(
            {
                "POSITION": randoms_positions,
                "INDWEIGHT": randoms_weights,
            }
        )
    ]
    params = {
        "modes": ("rsd", "fnl"),
        "fgrowth_blind": fgrowth_blind,
        "fnl_blind": fnl_blind,
        "metadata": "sealed",
        "parameter_mode": "secret_file",
        "parameters_fn": "catalog.npy",
        "parameters_key": "hidden",
        "options": {
            "zeff": z,
            "bias": bias,
            "rsd_kwargs": dict(mesh_kwargs),
            "fnl_kwargs": dict(mesh_kwargs),
            "fnl_shotnoise_correction": True,
        },
    }
    return cosmo, data, randoms, params, mesh_kwargs


def test_rsd_and_fnl_use_desiblind_jax_helpers_not_mockfactory_blinding(monkeypatch):
    calls = []

    class FakeDESI:
        def __init__(self):
            self._derived = {}

        def comoving_radial_distance(self, z):
            return z

    class FakeCatalogOps:
        @staticmethod
        def concatenate(catalogs):
            return ToyCatalog(
                POSITION=np.concatenate([catalog["POSITION"] for catalog in catalogs]),
                INDWEIGHT=np.concatenate([catalog["INDWEIGHT"] for catalog in catalogs]),
            )

    def cartesian_to_sky(positions):
        distance = np.sqrt(np.sum(positions**2, axis=1))
        return distance, np.zeros_like(distance), np.zeros_like(distance)

    class DistanceToRedshift:
        def __init__(self, distance):
            self.distance = distance

        def __call__(self, distance):
            return np.asarray(distance)

    cosmoprimo = types.ModuleType("cosmoprimo")
    fiducial = types.ModuleType("cosmoprimo.fiducial")
    fiducial.DESI = FakeDESI
    utils = types.ModuleType("cosmoprimo.utils")
    utils.DistanceToRedshift = DistanceToRedshift
    mockfactory = types.ModuleType("mockfactory")
    mockfactory.Catalog = FakeCatalogOps
    mockfactory.cartesian_to_sky = cartesian_to_sky

    def fake_rsd_blinding(data_positions, data_weights=None, randoms_positions=None, randoms_weights=None, **kwargs):
        calls.append(("rsd", kwargs, data_weights, randoms_positions, randoms_weights))
        return data_positions + 1.0

    def fake_fnl_blinding(data_positions, data_weights=None, randoms_positions=None, randoms_weights=None, **kwargs):
        calls.append(("fnl", kwargs, data_weights, randoms_positions, randoms_weights))
        return data_weights * 2.0

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mockfactory.blinding" or name.startswith("mockfactory.blinding."):
            raise AssertionError("production catalog blinding must not import mockfactory.blinding")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(sys.modules, "cosmoprimo", cosmoprimo)
    monkeypatch.setitem(sys.modules, "cosmoprimo.fiducial", fiducial)
    monkeypatch.setitem(sys.modules, "cosmoprimo.utils", utils)
    monkeypatch.setitem(sys.modules, "mockfactory", mockfactory)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(catalog_module, "_apply_rsd_jax_blinding", fake_rsd_blinding)
    monkeypatch.setattr(catalog_module, "_apply_fnl_jax_blinding", fake_fnl_blinding)

    data = ToyCatalog(
        POSITION=np.ones((2, 3)),
        INDWEIGHT=np.array([1.0, 2.0]),
        RA=np.zeros(2),
        DEC=np.zeros(2),
        Z=np.zeros(2),
    )
    randoms = [ToyCatalog(POSITION=np.zeros((3, 3)), INDWEIGHT=np.ones(3))]
    params = {
        "modes": ("rsd", "fnl"),
        "fgrowth_blind": 0.86,
        "fnl_blind": 5.0,
        "metadata": "sealed",
        "parameter_mode": "secret_file",
        "parameters_fn": "catalog.npy",
        "parameters_key": "hidden",
        "options": {"zeff": 0.9, "bias": 2.1, "rsd_smoothing_radius": 12.0, "fnl_smoothing_radius": 25.0},
    }

    shifted = TracerCatalogBlinder.apply_rsd_blinding(data, randoms, params, tracer="LRG")
    weighted, returned_randoms = TracerCatalogBlinder.apply_fnl_blinding(shifted, randoms, params, tracer="LRG")

    assert np.allclose(shifted["POSITION"], data["POSITION"] + 1.0)
    assert np.allclose(weighted["INDWEIGHT"], data["INDWEIGHT"] * 2.0)
    assert np.allclose(weighted["WEIGHT_BLIND"], 2.0)
    assert returned_randoms is randoms
    assert calls[0][0] == "rsd"
    assert calls[0][1]["bias"] == pytest.approx(2.1)
    assert calls[0][1]["z"] == pytest.approx(0.9)
    assert calls[0][1]["recon"] == "IterativeFFTReconstruction"
    assert calls[0][1]["smoothing_radius"] == pytest.approx(12.0)
    assert calls[1][0] == "fnl"
    assert calls[1][1]["method"] == "data_weights"
    assert calls[1][1]["smoothing_radius"] == pytest.approx(25.0)


def test_jax_rsd_and_fnl_match_mockfactory_reference():
    from jax import config

    config.update("jax_enable_x64", True)
    pytest.importorskip("jaxrecon")
    pytest.importorskip("jaxpower")
    pytest.importorskip("mockfactory")
    from cosmoprimo.fiducial import DESI
    from mockfactory import Catalog
    from mockfactory.blinding import CutskyCatalogBlinding

    cosmo, data, randoms, params, mesh_kwargs = _make_jax_blinding_case()
    options = params["options"]
    crandoms = Catalog.concatenate(randoms)

    cosmo_blind = DESI()
    cosmo_blind._derived["f"] = params["fgrowth_blind"]
    cosmo_blind._derived["fnl"] = params["fnl_blind"]
    reference = CutskyCatalogBlinding(
        cosmo_fid=cosmo,
        cosmo_blind=cosmo_blind,
        bias=options["bias"],
        z=options["zeff"],
        position_type="pos",
        mpicomm=data.mpicomm,
    )

    expected_positions = reference.rsd(
        data["POSITION"],
        data_weights=data["INDWEIGHT"],
        randoms_positions=crandoms["POSITION"],
        randoms_weights=crandoms["INDWEIGHT"],
        smoothing_radius=options.get("rsd_smoothing_radius", 15.0),
        **mesh_kwargs,
    )
    actual_rsd = TracerCatalogBlinder.apply_rsd_blinding(data, randoms, params, tracer="LRG")
    np.testing.assert_allclose(actual_rsd["POSITION"], expected_positions, rtol=1e-10, atol=1e-10)

    expected_weights = reference.png(
        data["POSITION"],
        data_weights=data["INDWEIGHT"],
        randoms_positions=crandoms["POSITION"],
        randoms_weights=crandoms["INDWEIGHT"],
        method="data_weights",
        shotnoise_correction=options["fnl_shotnoise_correction"],
        smoothing_radius=options.get("fnl_smoothing_radius", 30.0),
        **mesh_kwargs,
    )
    actual_fnl, returned_randoms = TracerCatalogBlinder.apply_fnl_blinding(data, randoms, params, tracer="LRG")
    np.testing.assert_allclose(actual_fnl["INDWEIGHT"], expected_weights, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual_fnl["WEIGHT_BLIND"], expected_weights / data["INDWEIGHT"], rtol=1e-10, atol=1e-10)
    assert returned_randoms is randoms


def test_plane_parallel_reconstruction_blinding_smoke():
    from jax import config

    config.update("jax_enable_x64", True)
    pytest.importorskip("jaxrecon")
    pytest.importorskip("jaxpower")
    pytest.importorskip("mockfactory")

    cosmo, data, randoms, params, mesh_kwargs = _make_jax_blinding_case()
    del cosmo
    params["options"]["rsd_recon"] = "PlaneParallelFFTReconstruction"
    params["options"]["rsd_kwargs"] = dict(mesh_kwargs, los="z")
    params["options"]["fnl_recon"] = "PlaneParallelFFTReconstruction"
    params["options"]["fnl_kwargs"] = dict(mesh_kwargs, los="z")

    rsd_data = TracerCatalogBlinder.apply_rsd_blinding(data, randoms, params, tracer="LRG")
    assert rsd_data["POSITION"].shape == data["POSITION"].shape
    assert np.all(np.isfinite(rsd_data["POSITION"]))

    fnl_data, returned_randoms = TracerCatalogBlinder.apply_fnl_blinding(data, randoms, params, tracer="LRG")
    assert fnl_data["INDWEIGHT"].shape == data["INDWEIGHT"].shape
    assert np.all(np.isfinite(fnl_data["INDWEIGHT"]))
    assert returned_randoms is randoms


def test_invalid_reconstruction_name_raises():
    pytest.importorskip("jaxrecon")

    with pytest.raises(ValueError, match="Unknown jax-recon reconstruction"):
        catalog_module._get_reconstruction_class("NotAReconstruction")
