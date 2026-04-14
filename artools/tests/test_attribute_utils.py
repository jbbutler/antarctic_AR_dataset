from artools.attribute_utils import *
import xarray as xr
import numpy as np
from unittest.mock import patch

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

def test_compute_duration():

    data = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 1]]])
    # test that _align_storm_coords also works
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [3, 6], 
                                 'lat': [1.0001, 2.0001], 
                                 'lon': [0.9999, 1.9999]})
    
    ais_mask = xr.DataArray([[0, 1], [1, 0]], 
                            dims=('lat', 'lon'), 
                            coords={'lat': [1,2], 'lon': [1,2]})

    result = compute_duration(ar_da, ais_da=ais_mask)

    assert result == 3

def test_compute_duration2():

    data = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 1]]])
    # test that _align_storm_coords also works
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [3, 6], 
                                 'lat': [1.0001, 2.0001], 
                                 'lon': [0.9999, 1.9999]})

    result = compute_duration(ar_da)

    assert result == 6

def test_add_startdate():

    data = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 1]]])
    # test that _align_storm_coords also works
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [3, 6], 
                                 'lat': [1.0001, 2.0001], 
                                 'lon': [0.9999, 1.9999]})
    ais_mask = xr.DataArray([[0, 1], [1, 0]], 
                            dims=('lat', 'lon'), 
                            coords={'lat': [1,2], 'lon': [1,2]})

    result = add_start_date(ar_da, ais_da=ais_mask)

    assert result == 6

def test_add_startdate2():

    data = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 1]]])
    # test that _align_storm_coords also works
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [3, 6], 
                                 'lat': [1.0001, 2.0001], 
                                 'lon': [0.9999, 1.9999]})
    result = add_start_date(ar_da)

    assert result == 3

def test_end_enddate():

    data = np.array([[[1, 1], [0, 0]], [[1, 0], [0, 1]]])
    # test that _align_storm_coords also works
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [3, 6], 
                                 'lat': [1.0001, 2.0001], 
                                 'lon': [0.9999, 1.9999]})

    ais_mask = xr.DataArray([[0, 1], [1, 0]], 
                            dims=('lat', 'lon'), 
                            coords={'lat': [1,2], 'lon': [1,2]})
    result = add_end_date(ar_da, ais_mask)

    assert result == 3

def test_end_enddate2():

    data = np.array([[[1, 0], [0, 0]], [[1, 1], [0, 1]]])
    # test that _align_storm_coords also works
    ar_da = xr.DataArray(data,
                         dims=('time', 'lat', 'lon'),
                         coords={'time': [3, 6], 
                                 'lat': [1.0001, 2.0001], 
                                 'lon': [0.9999, 1.9999]})
    result = add_end_date(ar_da)

    assert result == 6

@patch('artools.attribute_utils.compute_cumulative_spacetime')
def test_find_landfalling_region(mock_compute):
    # mock output of region_mask looping
    # each output is a fake region's CLA
    mock_compute.side_effect = [15.5, 85.2, 10.1]
    
    # dummy inputs to find_landfalling_region
    # a dummy storm and dummy cell areas
    dummy_ar_da = 'fake_ar_da'
    dummy_cell_areas = 'fake_cell_areas'
    
    toy_region_masks = {'Region 1': 'mask_1',      
                        'Region 2': 'mask_2', 
                        'Region 3': 'mask_3'}
    
    winner = find_landfalling_region(dummy_ar_da, dummy_cell_areas, toy_region_masks)
    
    assert winner == 'Region 2'

def test_find_region_masks():

    toy_lons = [-170, -50, 0, 50, 170]

    # our dummy ais da will just consist of a line of longitudes with
    # fixed latitude. This region mask code is latitude agnostic, so it
    # suffices
    ais_da_toy = xr.DataArray([1,1,1,1,1], coords={'lon': toy_lons}, dims=['lon'])

    # test two regions: one which crosses the dateline and one
    # which doesn't
    toy_region_defs = {'Normal_Region': [-60, 60],
                       'Dateline_Region': [150, -150]}
    
    # run the function
    masks = find_region_masks(toy_region_defs, ais_da_toy)
    
    # check that both regions were added to the dictionary
    assert 'Normal_Region' in masks
    assert 'Dateline_Region' in masks
    
    # Verify normal region is subsetted
    # Lons: [-170, -50, 0, 50, 170]
    # Expected: False, True, True, True, False
    np.testing.assert_array_equal(masks['Normal_Region'].values,
                                  [False, True, True, True, False])
    
    # Verify the dateline crossing region is correct
    # Lons: [-170, -50, 0, 50, 170]
    # Expected: True, False, False, False, True
    np.testing.assert_array_equal(masks['Dateline_Region'].values,
                                  [True, False, False, False, True])

#def test_extract_trajectory():

#def test_compute_cumulative():

#def test_compute_max_intensity():

#def test_compute_min_SLP():

#def test_compute_max_SLPgrad():

#def test_compute_avg_landfalling_minomega():

#def test_compute_max_elevation_grad():

#def test_compute_max_landfalling_wind():

#def test_compute_avg_landfalling_wind():

#def test_compute_average():

#def test_augment_storm_dat():




    



    

    

    
    

    
    
    