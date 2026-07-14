import os

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from artools.zagg_events import export_events, parse_hyperparams

CATALOG_FNAME = 'epsspace0.5_epstime12_minpts5_nreppts10_seed12345.h5'


def _storm_mask(start, n_times, seed=0):
    '''A tiny (time, lat, lon) binary mask with real 3-hourly datetimes.'''
    rng = np.random.default_rng(seed)
    time = pd.date_range(start, periods=n_times, freq='3h')
    data = rng.integers(0, 2, size=(n_times, 2, 3)).astype('int8')
    data[0, 0, 0] = 1  # never an all-zero storm
    return xr.DataArray(
        data,
        dims=('time', 'lat', 'lon'),
        coords={'time': time, 'lat': [-70.0, -69.5], 'lon': [0.0, 0.625, 1.25]},
    )


@pytest.fixture
def catalog_h5(tmp_path):
    '''A synthetic catalog written through the same to_hdf path the real
    catalogs use: float non-contiguous index, data_array + is_landfalling.'''
    storms = {
        1.0: _storm_mask('1980-01-02', 4, seed=1),
        3.0: _storm_mask('1980-02-10T06', 2, seed=3),
        7.0: _storm_mask('1980-03-01T12', 8, seed=7),
    }
    catalog = pd.DataFrame(
        {
            'data_array': pd.Series(storms, dtype=object),
            'is_landfalling': pd.Series({1.0: True, 3.0: False, 7.0: True}),
        }
    )
    path = tmp_path / CATALOG_FNAME
    catalog.to_hdf(path, key='df')
    return path


def test_parse_hyperparams():
    params = parse_hyperparams(CATALOG_FNAME)
    assert params == {
        'eps_space': 0.5,
        'eps_time': 12.0,
        'min_pts': 5,
        'n_rep_pts': 10,
        'seed': 12345,
    }


def test_parse_hyperparams_fractional_epstime():
    params = parse_hyperparams('epsspace0.5_epstime18.0_minpts5_nreppts10_seed12345.h5')
    assert params['eps_time'] == 18.0


def test_parse_hyperparams_rejects_unencoded_name():
    with pytest.raises(ValueError, match='hyperparameters'):
        parse_hyperparams('storms_final_v2.h5')


def test_export_writes_masks_and_catalog(catalog_h5):
    events, parquet_path = export_events(catalog_h5)

    assert parquet_path.exists()
    # default layout: <catalog dir>/derived/<stem>/
    assert parquet_path.parent == catalog_h5.parent / 'derived' / catalog_h5.stem
    assert list(events['event_key']) == ['storm_00001', 'storm_00003', 'storm_00007']
    assert list(events['storm_id']) == [1, 3, 7]
    assert list(events['is_landfalling']) == [True, False, True]
    # hyperparameter provenance rides on every row
    assert set(events['seed']) == {12345}
    assert set(events['eps_space']) == {0.5}
    assert set(events['catalog_file']) == {CATALOG_FNAME}

    for _, row in events.iterrows():
        raw = xr.open_dataarray(row['mask_path'])
        assert raw.name == 'mask'
        assert raw.dims == ('time', 'lat', 'lon')
        assert raw.dtype == np.int8


def test_export_masks_roundtrip_values(catalog_h5):
    events, _ = export_events(catalog_h5)
    source = pd.read_hdf(catalog_h5)
    row = events.iloc[0]
    raw = xr.open_dataarray(row['mask_path'])
    np.testing.assert_array_equal(raw.values, source.loc[1.0, 'data_array'].values)


def test_export_augmented_mask_semantics(catalog_h5):
    # the augmented mask must equal augment_storm_da's output: time axis
    # extended exactly one day past the storm end, values a 24h rolling max
    from artools.attribute_utils import augment_storm_da

    events, _ = export_events(catalog_h5)
    source = pd.read_hdf(catalog_h5)
    for _, row in events.iterrows():
        raw = source.loc[float(row['storm_id']), 'data_array']
        augmented = xr.open_dataarray(row['mask_augmented_path'])
        expected = augment_storm_da(raw)
        assert augmented['time'].values[-1] == np.datetime64(row['t_end_augmented'])
        np.testing.assert_array_equal(augmented.values, expected.astype('int8').values)
        # augmented covers the raw footprint at every raw timestep
        overlap = augmented.sel(time=raw['time'].values)
        assert bool((overlap >= raw).all())


def test_export_catalog_time_and_bbox_columns(catalog_h5):
    events, _ = export_events(catalog_h5)
    row = events.set_index('storm_id').loc[1]
    raw = pd.read_hdf(catalog_h5).loc[1.0, 'data_array']
    assert row['t_start'] == raw['time'].values[0]
    assert row['t_end'] == raw['time'].values[-1]
    assert row['t_end_augmented'] == raw['time'].values[-1] + np.timedelta64(1, 'D')
    assert row['n_timesteps'] == raw.sizes['time']
    assert row['lat_min'] == float(raw['lat'].min())
    assert row['lon_max'] == float(raw['lon'].max())


def test_export_storm_subset(catalog_h5):
    events, _ = export_events(catalog_h5, storms=[3])
    assert list(events['event_key']) == ['storm_00003']


def test_export_skips_existing_unless_overwrite(catalog_h5):
    events, _ = export_events(catalog_h5)
    mask_path = events.iloc[0]['mask_path']
    before = os.path.getmtime(mask_path)

    events2, _ = export_events(catalog_h5)  # resume: no rewrite
    assert os.path.getmtime(mask_path) == before
    assert len(events2) == 3  # catalog still covers every storm

    export_events(catalog_h5, overwrite=True)
    assert os.path.getmtime(mask_path) >= before


def test_export_parquet_roundtrip(catalog_h5):
    events, parquet_path = export_events(catalog_h5)
    back = pd.read_parquet(parquet_path)
    assert list(back['event_key']) == list(events['event_key'])
    assert list(back.columns) == list(events.columns)
