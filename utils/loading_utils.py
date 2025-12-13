'''
A module with functions to load up any datasets that may be useful in other computations,
including the Wille (2024) catalogs, and the AIS points.

Jimmy Butler
October 2025
'''

from pathlib import Path
import xarray as xr
import os
import numpy as np
import earthaccess
import ray

cur_path = Path(__file__)
home_dir = Path(cur_path).parents[1]

def load_ais(points=False):
    '''
    Load up the AIS mask.

    Inputs:
        points (boolean): if True, gives a list of coordinate cells that correspond to the AIS.
            By default, loads up the binary valued xarray.DataArray mask.
    Outputs:
        Depends on points, as above.
    '''

    # Load up the AIS mask
    mask_path = home_dir/'data/antarctic_masks/AIS_Full_basins_Zwally_MERRA2grid_new.nc'
    full_ais_mask = xr.open_dataset(mask_path).Zwallybasins > 0
    # grab only points in the Southern Ocean area
    ais_mask = full_ais_mask.sel(lat=slice(-86, -39))

    if points:
        # get ais points
        ais_mask_lats = ais_mask.lat[np.where(ais_mask.to_numpy())[0]].to_numpy()
        ais_mask_lons = ais_mask.lon[np.where(ais_mask.to_numpy())[1]].to_numpy()
        ais_pts = set(zip(ais_mask_lats, ais_mask_lons))

        return ais_pts

    return ais_mask

def load_cell_areas():
    '''
    Load up the xarray.DataArray with the grid cell areas.

    Inputs:
        None
    Outputs:
        cell_areas (xarray.DataArray): the DataArray in our region of interest with the area of
            each grid cell provided
    '''

    areas_path = home_dir/'data/area/MERRA2_gridarea.nc'
    cell_areas = xr.open_dataset(areas_path)
    cell_areas = cell_areas.cell_area

    return cell_areas

def grab_MERRA2_granules(storm_da, data_doi):
    '''
    Grab a list of data granules from a specific MERRA-2 dataset for an AR,
        specifically pointers to granules stored in Amazon S3 bucket.

    Inputs:
        storm_da (xarray.DataArray) the AR's binary mask
        data_doi (str): the doi of the MERRA-2 dataset

    Outputs:
        list of granule pointers
    '''
    first = np.min(storm_da.time.dt.date.to_numpy())
    last = np.max(storm_da.time.dt.date.to_numpy())
    # stream the data only between those two dates
    granule_lst = earthaccess.search_data(doi=data_doi, 
                                  temporal=(f'{first.year}-{first.month}-{first.day}', 
                                            f'{last.year}-{last.month}-{last.day}'))

    return granule_lst

@ray.remote
class EarthdataGatekeeper:
    '''
    A Ray Actor that makes the open requests to NASA's servers sequentially so that we don't get
        rate limited by NASA.
    '''
    def __init__(self):
        self.auth = earthaccess.login()
    
    def get_granule_pointers(self, granule_lst):
        return earthaccess.open(granule_lst, show_progress=False)
