from artools.attribute_utils import *
import xarray as xr
import numpy as np

def test_is_landfalling1():
    ar_da = xr.DataArray([[1, 0],[1, 0]], dims=('lat', 'lon'), coords={'lat': [1,2], 'lon': [1,2]})
    ais_mask = xr.DataArray([[1, 1], [1, 0]], dims=('lat', 'lon'), coords={'lat': [1,2], 'lon': [1,2]})
    result = is_landfalling(ar_da, ais_mask)

    assert result == True

def test_is_landfalling2():
    ar_da = xr.DataArray([[0, 0],[0, 1]], dims=('lat', 'lon'), coords={'lat': [1,2], 'lon': [1,2]})
    ais_mask = xr.DataArray([[1, 1], [1, 0]], dims=('lat', 'lon'), coords={'lat': [1,2], 'lon': [1,2]})
    result = is_landfalling(ar_da, ais_mask)

    assert result == False

def test_compute_max_area1():

    data = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 0]]])
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [0, 1], 'lat': [1.0001, 2.0001], 'lon': [0.9999, 1.9999]})

    # test that _align_storm_coords also works
    cell_areas = xr.DataArray([[9*(10**6), 9*(10**6)], [8*(10**6), 8*(10**6)]],
                              dims=('lat', 'lon'),
                              coords={'lat': [1,2], 'lon': [1,2]})

    result = compute_max_area(ar_da, cell_areas)

    assert result == 18

def test_compute_max_area2():

    data = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 0]]])
    # test that _align_storm_coords also works
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [0, 1], 'lat': [1.0001, 2.0001], 'lon': [0.9999, 1.9999]})
    
    cell_areas = xr.DataArray([[9*(10**6), 9*(10**6)], [8*(10**6), 8*(10**6)]],
                              dims=('lat', 'lon'),
                              coords={'lat': [1,2], 'lon': [1,2]})
    ais_mask = xr.DataArray([[0, 1], [1, 0]], 
                            dims=('lat', 'lon'), 
                            coords={'lat': [1,2], 'lon': [1,2]})
    result = compute_max_area(ar_da, cell_areas, ais_mask)
    
    assert result == 9

def test_compute_max_southward_extent():
    data = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 1]]])
    # test that _align_storm_coords also works
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [0, 1], 'lat': [1, 2], 'lon': [1, 2]})
    result = compute_max_southward_extent(ar_da)

    assert result == 1

def test_compute_mean_area1():

    data = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 1]]])
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [0, 1], 'lat': [1.0001, 2.0001], 'lon': [0.9999, 1.9999]})

    # test that _align_storm_coords also works
    cell_areas = xr.DataArray([[9*(10**6), 9*(10**6)], [8*(10**6), 8*(10**6)]],
                              dims=('lat', 'lon'),
                              coords={'lat': [1,2], 'lon': [1,2]})

    result = compute_mean_area(ar_da, cell_areas)

    assert result == 17.5

def test_compute_mean_area2():

    data = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 1]]])
    # test that _align_storm_coords also works
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [0, 1], 'lat': [1.0001, 2.0001], 'lon': [0.9999, 1.9999]})
    
    cell_areas = xr.DataArray([[9*(10**6), 9*(10**6)], [8*(10**6), 8*(10**6)]],
                              dims=('lat', 'lon'),
                              coords={'lat': [1,2], 'lon': [1,2]})
    ais_mask = xr.DataArray([[0, 1], [1, 0]], 
                            dims=('lat', 'lon'), 
                            coords={'lat': [1,2], 'lon': [1,2]})
    result = compute_mean_area(ar_da, cell_areas, ais_mask)

    assert result == 4.5

def test_compute_cumulative_spacetime1():

    data = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 1]]])
    # test that _align_storm_coords also works
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [3, 6], 'lat': [1.0001, 2.0001], 'lon': [0.9999, 1.9999]})
    
    cell_areas = xr.DataArray([[9*(10**6), 9*(10**6)], [8*(10**6), 8*(10**6)]],
                              dims=('lat', 'lon'),
                              coords={'lat': [1,2], 'lon': [1,2]})

    result = compute_cumulative_spacetime(ar_da, cell_areas)

    assert result == 4.375

def test_compute_cumulative_spacetime2():

    data = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 1]]])
    # test that _align_storm_coords also works
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [3, 6], 'lat': [1.0001, 2.0001], 'lon': [0.9999, 1.9999]})
    
    cell_areas = xr.DataArray([[9*(10**6), 9*(10**6)], [8*(10**6), 8*(10**6)]],
                              dims=('lat', 'lon'),
                              coords={'lat': [1,2], 'lon': [1,2]})
    ais_mask = xr.DataArray([[0, 1], [1, 0]], 
                            dims=('lat', 'lon'), 
                            coords={'lat': [1,2], 'lon': [1,2]})
    result = compute_cumulative_spacetime(ar_da, cell_areas, ais_mask)

    assert result == 1.125

    

    

    
    

    
    
    