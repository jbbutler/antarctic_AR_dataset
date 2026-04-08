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