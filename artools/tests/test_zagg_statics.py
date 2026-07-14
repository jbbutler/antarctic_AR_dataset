import numpy as np
import pytest
import xarray as xr

from artools.zagg_statics import LAT_SLICE, load_zagg_statics, prepare_static_files


@pytest.fixture
def catalogs_dir(tmp_path):
    '''Miniature stand-ins for the two static source files.'''
    lat = np.array([-88.0, -70.000001, -40.0, -20.0])  # spans the LAT_SLICE cut
    lon = np.array([0.0, 0.625])
    coords = {'lat': lat, 'lon': lon}
    basins = xr.Dataset(
        {'Zwallybasins': (('lat', 'lon'), np.array([[0, 3], [1, 0], [0, 2], [4, 4]]))},
        coords=coords,
    )
    basins.to_netcdf(tmp_path / 'AIS_Full_basins_Zwally_MERRA2grid_new.nc')
    areas = xr.Dataset(
        {'cell_area': (('lat', 'lon'), np.full((4, 2), 9.0e8))}, coords=coords
    )
    areas.to_netcdf(tmp_path / 'MERRA2_gridarea.nc')
    return tmp_path


def test_prepare_static_files_normalizes(catalogs_dir):
    paths = prepare_static_files(catalogs_dir)

    assert set(paths) == {'ais_mask', 'cell_areas'}  # no climatology requested
    ais = xr.open_dataarray(paths['ais_mask'])
    assert ais.name == 'ais_mask'
    assert ais.dtype == np.int8
    # subset to the working latitude window and 0/1-valued
    assert float(ais.lat.min()) >= LAT_SLICE.start and float(ais.lat.max()) <= LAT_SLICE.stop
    assert set(np.unique(ais.values)) <= {0, 1}
    # coordinates rounded to 5 decimals (loading_utils convention)
    assert -70.0 in ais.lat.values

    areas = xr.open_dataarray(paths['cell_areas'])
    assert areas.name == 'cell_areas'
    assert areas.shape == ais.shape


def test_prepare_static_files_resumes_without_overwrite(catalogs_dir):
    import os

    paths = prepare_static_files(catalogs_dir)
    before = os.path.getmtime(paths['ais_mask'])
    prepare_static_files(catalogs_dir)
    assert os.path.getmtime(paths['ais_mask']) == before
    prepare_static_files(catalogs_dir, overwrite=True)
    assert os.path.getmtime(paths['ais_mask']) >= before


def test_load_zagg_statics_roundtrip(catalogs_dir):
    prepare_static_files(catalogs_dir)
    statics = load_zagg_statics(catalogs_dir, with_climatology=False)
    assert set(statics) == {'ais_mask', 'cell_areas'}
    assert isinstance(statics['ais_mask'], xr.DataArray)
    assert isinstance(statics['cell_areas'], xr.DataArray)


def test_build_climatology_monthly_means(catalogs_dir, monkeypatch):
    import earthaccess

    from artools.zagg_statics import build_climatology

    time = np.array(
        ['1980-01-01', '1980-02-01', '1981-01-01', '1981-02-01'], dtype='datetime64[ns]'
    )
    t2m = xr.DataArray(
        np.array([[[270.0]], [[250.0]], [[280.0]], [[260.0]]]),
        dims=('time', 'lat', 'lon'),
        coords={'time': time, 'lat': [-70.0], 'lon': [0.0]},
    )
    monthly = xr.Dataset({'T2M': t2m, 'TQV': t2m * 0.01})

    monkeypatch.setattr(earthaccess, 'login', lambda: None)
    monkeypatch.setattr(earthaccess, 'search_data', lambda **k: ['g1', 'g2'])
    monkeypatch.setattr(earthaccess, 'open', lambda granules: granules)
    monkeypatch.setattr(xr, 'open_mfdataset', lambda handles: monthly)

    out = catalogs_dir / 'static' / 'clim.nc'
    clim = build_climatology(('1980-01-01', '1981-12-31'), out)

    assert list(clim.dims) == ['month', 'lat', 'lon'] or set(clim.dims) == {'month', 'lat', 'lon'}
    # month means average the two years
    assert clim['T2M'].sel(month=1).item() == pytest.approx(275.0)
    assert clim['T2M'].sel(month=2).item() == pytest.approx(255.0)
    assert out.exists()


def test_zagg_configs_validate():
    zagg = pytest.importorskip('zagg')
    from pathlib import Path

    from zagg.config import load_config

    configs = Path(__file__).resolve().parents[2] / 'configs'
    for name in ('ar_attributes.yaml', 'ar_precip.yaml'):
        cfg = load_config(configs / name)  # load_config runs validate_config
        assert cfg.pipeline == {'type': 'temporal'}


def test_zagg_config_capabilities_resolve():
    # every spatial_func/reducer/mask/trigger named in the configs must be a
    # zagg built-in -- the no-user-code-on-Lambda rule means a typo would
    # otherwise surface per-worker after the fan-out
    zagg = pytest.importorskip('zagg')
    import yaml
    from pathlib import Path

    import zagg.temporal  # noqa: F401  (seeds the registries)
    from zagg import registry

    configs = Path(__file__).resolve().parents[2] / 'configs'
    for name in ('ar_attributes.yaml', 'ar_precip.yaml'):
        with open(configs / name) as f:
            spec_block = yaml.safe_load(f)['aggregation']['variables']
        for spec in spec_block.values():
            registry.get_spatial_func(spec['spatial_func'])
            registry.get_reducer(spec['temporal_reducer'])
            registry.get_mask_provider(spec.get('mask', 'ais'))
            if spec.get('trigger'):
                registry.get_event_trigger(spec['trigger'])
