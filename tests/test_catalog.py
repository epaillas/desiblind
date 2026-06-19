import sys
import types

import numpy as np
import pytest

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
    fn = tmp_path / "catalog_blinding_2026_06.npy"
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
    assert np.allclose(blinded["Z"][mask], expected[mask])
    assert np.isnan(blinded["Z"][-1])


def test_rsd_and_fnl_delegate_to_mockfactory(monkeypatch):
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

    class FakeCutskyCatalogBlinding:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def rsd(self, positions, **kwargs):
            calls.append(("rsd", kwargs))
            return positions + 1.0

        def png(self, positions, **kwargs):
            calls.append(("png", kwargs))
            return kwargs["data_weights"] * 2.0

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
    mockfactory_blinding = types.ModuleType("mockfactory.blinding")
    mockfactory_blinding.CutskyCatalogBlinding = FakeCutskyCatalogBlinding

    monkeypatch.setitem(sys.modules, "cosmoprimo", cosmoprimo)
    monkeypatch.setitem(sys.modules, "cosmoprimo.fiducial", fiducial)
    monkeypatch.setitem(sys.modules, "cosmoprimo.utils", utils)
    monkeypatch.setitem(sys.modules, "mockfactory", mockfactory)
    monkeypatch.setitem(sys.modules, "mockfactory.blinding", mockfactory_blinding)

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
    assert calls[0][1]["bias"] == pytest.approx(2.1)
    assert calls[0][1]["z"] == pytest.approx(0.9)
    assert calls[1][0] == "rsd"
    assert calls[1][1]["smoothing_radius"] == pytest.approx(12.0)
    assert calls[3][0] == "png"
    assert calls[3][1]["method"] == "data_weights"
    assert calls[3][1]["smoothing_radius"] == pytest.approx(25.0)
