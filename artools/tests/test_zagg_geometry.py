import numpy as np
import pandas as pd
import pytest
import xarray as xr

from artools.zagg_geometry import assemble_attribute_table, geometry_attributes


@pytest.fixture
def exported(tmp_path):
    '''One tiny exported storm (mask file + catalog row) and statics.

    Grid: 2 lat x 2 lon; the southern row (-70.0) is AIS. The storm covers
    the full grid at t0 and only the northern row at t1.
    '''
    lat = np.array([-70.0, -69.5])
    lon = np.array([0.0, 0.625])
    time = np.array(['1980-01-02T00', '1980-01-02T03'], dtype='datetime64[ns]')
    data = np.array(
        [[[1, 1], [1, 1]], [[0, 0], [1, 1]]], dtype='int8'
    )
    mask = xr.DataArray(
        data, dims=('time', 'lat', 'lon'), coords={'time': time, 'lat': lat, 'lon': lon},
        name='mask',
    )
    mask_path = tmp_path / 'storm_00001.nc'
    mask.to_netcdf(mask_path)

    events = pd.DataFrame(
        [
            {
                'event_key': 'storm_00001',
                'storm_id': 1,
                'is_landfalling': True,
                'catalog_file': 'epsspace0.5_epstime12_minpts5_nreppts10_seed12345.h5',
                'eps_space': 0.5, 'eps_time': 12.0, 'min_pts': 5,
                'n_rep_pts': 10, 'seed': 12345,
                'mask_path': str(mask_path),
            }
        ]
    )
    coords = {'lat': lat, 'lon': lon}
    statics = {
        'ais_mask': xr.DataArray(
            np.array([[1, 1], [0, 0]], dtype='int8'), dims=('lat', 'lon'), coords=coords
        ),
        'cell_areas': xr.DataArray(
            np.full((2, 2), 4.0e6), dims=('lat', 'lon'), coords=coords  # 4 km^2 cells
        ),
    }
    return events, statics


def test_geometry_attributes_notebook_columns(exported):
    events, statics = exported
    geometry = geometry_attributes(events, statics)

    (row,) = geometry.to_dict('records')
    assert row['event_key'] == 'storm_00001'
    # t0 covers 4 cells of 4 km^2 -> max 16; AIS-masked mean over 2 timesteps:
    # (8 + 0) / 2 = 4 km^2
    assert row['max_area'] == 16
    assert row['mean_landfalling_area'] == 4
    # 3h blocks: 8 km^2 * 3h / 24 = 1 km^2-day at t0, 0 at t1
    assert row['cumulative_landfalling_area'] == 1
    assert row['duration'] == 6  # two 3-hourly steps
    assert row['start_date'] == np.datetime64('1980-01-02T00')
    assert row['end_date'] == np.datetime64('1980-01-02T03')
    assert row['max_south_extent'] == -70.0
    assert row['region'] in {'West', 'East 1', 'East 2'}
    assert 'trajectory' not in geometry.columns


def test_geometry_trajectory_opt_in(exported):
    events, statics = exported
    geometry = geometry_attributes(events, statics, include_trajectory=True)
    trajectory = geometry.iloc[0]['trajectory']
    assert list(trajectory.columns) == ['time', 'avg_lon', 'avg_lat']
    assert len(trajectory) == 2


def test_assemble_attribute_table_joins_and_writes(exported, tmp_path):
    events, statics = exported
    geometry = geometry_attributes(events, statics)

    fields = pd.DataFrame(
        [{'event_key': 'storm_00001', 'max_T2m_ais': 271.5, 'timesteps_processed': 2}]
    )
    precip_path = tmp_path / 'precip.parquet'
    pd.DataFrame(
        [{'event_key': 'storm_00001', 'cumulative_rainfall_ais': 1.25e9}]
    ).to_parquet(precip_path)

    out = tmp_path / 'table.parquet'
    table = assemble_attribute_table(events, geometry, [fields, precip_path], out_path=out)

    (row,) = table.to_dict('records')
    assert row['storm_id'] == 1
    assert row['seed'] == 12345
    assert row['max_area'] == 16
    assert row['max_T2m_ais'] == 271.5
    assert row['cumulative_rainfall_ais'] == 1.25e9
    assert 'timesteps_processed' not in table.columns
    assert 'mask_path' not in table.columns
    back = pd.read_parquet(out)
    assert list(back['event_key']) == ['storm_00001']


def test_assemble_attribute_table_missing_zagg_rows_are_nan(exported):
    events, statics = exported
    geometry = geometry_attributes(events, statics)
    empty = pd.DataFrame({'event_key': [], 'max_T2m_ais': []})
    table = assemble_attribute_table(events, geometry, [empty])
    assert np.isnan(table.iloc[0]['max_T2m_ais'])


def test_assemble_attribute_table_rejects_column_collisions(exported):
    events, statics = exported
    geometry = geometry_attributes(events, statics)
    clashing = pd.DataFrame([{'event_key': 'storm_00001', 'max_area': 0.0}])
    with pytest.raises(ValueError, match='max_area'):
        assemble_attribute_table(events, geometry, [clashing])
