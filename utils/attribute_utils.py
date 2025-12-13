'''
Module with functions to compute attributes of storms.
Generally, these functions are lower-level helper functions
that compute an AR quantity assuming the climate variable
DataArrays have already been loaded.

Jimmy Butler
October 2025
'''

import xarray as xr
import pandas as pd
import numpy as np
from .general_utils import average_angle

def is_landfalling(ar_da, ais_mask):
    '''
    Function to determine if a given AR has made landfall on the AIS.

    Inputs:
        ar_da (xarray.DataArray): binary valued DataArray representing the ARs footprint

    Outputs:
        is_landfalling (boolean): whether the AR intersected the AIS at any point in time
    '''

    is_landfalling = (ais_mask*ar_da).any().values

    return is_landfalling

def compute_max_area(ar_da, area_da, ais_da=None):
    '''
    A function that, given a binary mask DataArray for a storm, computes the max area occupied over lifetime

    Inputs:
        ar_da (xarray.DataArray): the binary valued DataArray for that storm
        ais_da (xarray.DataArray): if provided, find max area occupied over AIS (default: over AIS + ocean)
    Outputs:
        max_area (float): the area in km^2
    '''

    # just to be safe, round coordinates so that -0 matches with 0 degrees, in case that appears
    ar_da_rounded = ar_da.assign_coords(lat=ar_da.lat.round(5), lon=ar_da.lon.round(5))
    
    if ais_da is not None:
        storm_ais_mask = ais_da.sel(lat=ar_da_rounded.lat, lon=ar_da_rounded.lon)
        storm_da_subset = ar_da_rounded.where(storm_ais_mask, 0)
    else:
        storm_da_subset = ar_da_rounded.copy()
    
    grid_area_storm = area_da.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)
    max_area = float(storm_da_subset.dot(grid_area_storm).max().values/(1000**2))
    
    return max_area

def compute_max_southward_extent(ar_da):
    '''
    A function that, given a binary mask DataArray for a storm, computes the lowest latitude it occupied

    Inputs:
        ar_da (xarray.DataArray): the binary valued DataArray for that storm
    Outputs:
        (float): the lowest latitude in degrees
    '''
    ar_da_rounded = ar_da.assign_coords(lat=ar_da.lat.round(5), lon=ar_da.lon.round(5))
    return np.min(ar_da.lat.values)

def compute_mean_area(ar_da, cell_areas, ais_da=None):
    '''
    A function that, given a binary mask DataArray for a storm, computes the mean area occupied over lifetime

    Inputs:
        ar_da (xarray.DataArray): the binary valued DataArray for that storm
        ais_da (xarray.DataArray): if provided, find mean area occupied over AIS (default: over AIS + ocean)
    Outputs:
        mean_area (float): the area in km^2
    '''
    
    ar_da_rounded = ar_da.assign_coords(lat=ar_da.lat.round(5), lon=ar_da.lon.round(5))
    
    if ais_da is not None:
        storm_ais_mask = ais_da.sel(lat=ar_da_rounded.lat, lon=ar_da_rounded.lon)
        storm_da_subset = ar_da_rounded.where(storm_ais_mask, 0)
    else:
        storm_da_subset = ar_da_rounded.copy()
    
    grid_area_storm = cell_areas.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)
    mean_area = float(storm_da_subset.dot(grid_area_storm).mean().values/(1000**2))
    return mean_area

def compute_cumulative_spacetime(ar_da, cell_areas, ais_da=None):
    '''
    A function that, given a binary mask DataArray for a storm, computes the cumulative amount
        of space and time the storm spent over the AIS (measured in km^2 x days)

    Inputs:
        ar_da (xarray.DataArray): the binary valued DataArray for that storm
        ais_da (xarray.DataArray): if provided, find quantity occupied over AIS (default: over AIS + ocean)
    Outputs:
        cumulative_area (float): the cumulative area in km^2 x days
    '''
    
    ar_da_rounded = ar_da.assign_coords(lat=ar_da.lat.round(5), lon=ar_da.lon.round(5))
    
    if ais_da is not None:
        storm_ais_mask = ais_da.sel(lat=ar_da_rounded.lat, lon=ar_da_rounded.lon)
        storm_da_subset = ar_da_rounded.where(storm_ais_mask, 0)
    else:
        storm_da_subset = ar_da_rounded.copy()
    
    grid_area_storm = cell_areas.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)
    cumulative_area = float((3*storm_da_subset.dot(grid_area_storm)).sum().values/((1000**2)*24)) #3 comes from 3 hourly blocks of Wille catalog
    return cumulative_area

def compute_duration(ar_da):
    '''
    Returns the duration of a storm. Note if the storm only occupies one 3 hourly time step, that storm's duration is 3 hours.

    Inputs:
        ar_da (xarray.DataArray): the binary mask for the storm
    Outputs:
        days (np.timedelta64): number of hours
    '''
    days = (ar_da.time.max() - ar_da.time.min()).values.astype('timedelta64[h]').astype(int) + np.timedelta64(3, 'h')
    return days

def add_start_date(ar_da):
    '''
    Returns the start date of a storm.

    Inputs:
        ar_da (xarray.DataArray): the binary mask for the storm
    Outputs:
        start (np.datetime64): starting time (3 hourly level)
    '''
    start = ar_da.time.min().values
    return start

def add_end_date(ar_da):
    '''
    Returns the end date of a storm.

    Inputs:
        ar_da (xarray.DataArray): the binary mask for the storm
    Outputs:
        end (np.datetime64): ending time (3 hourly level)
    '''
    end = ar_da.time.max().values
    return end

def find_landfalling_region(ar_da, cell_areas, region_masks):
    '''
    Finding the region in which the storm makes landfall. Regions provided by David Mikolajczyk
    at UW Madison, with additional regions for the RIS, FRIS, and completing the connection between
    QMC and END. Landfalling region determined as which one it has the most CLA over.

    Inputs:
        ar_da (xarray.DataArray): the binary mask DataArray
        region_masks (dictionary): keys are strings indicating region names, values are xarray.DataArray
            masks indicating pixels for those regions
    Outputs:
        winning_region (string): the string name of the region it spends most area x time over
    '''

    region_CLA = {}
    for label, mask in region_masks.items():
        region_CLA[label] = compute_cumulative_spacetime(ar_da, cell_areas=cell_areas, ais_da=mask)

    region_CLA = pd.Series(region_CLA)
    winning_region = region_CLA.idxmax()

    return winning_region

def find_region_masks(region_defs, ais_da):
    '''
    Helper function for the above find_landfalling_region function. Given longitude bounds for each region,
    find binary masks for each of these from the AIS mask
    
    Inputs:
        region_defs (dictionary): dictionary whose keys are strings indicating names of regions and
            values are lists with lower and upper longitude bounds for the regions
        ais_da (xarray.DataArray): the binary mask for the AIS
    Output:
        region_masks (dictionary): keys are names of regions (strings), and values are DataArray masks for
            just that region
    '''

    region_masks = {}

    for label, bound in region_defs.items():
        if bound[0] > bound[1]: # if we're crossing the dateline
            region_masks[label] = ais_da.where(((ais_da.lon > bound[0]) & (ais_da.lon <= 180)) | ((ais_da.lon >= -180) & (ais_da.lon < bound[1])), False)
        else: # normal case
            region_masks[label] = ais_da.where((ais_da.lon > bound[0]) & (ais_da.lon < bound[1]), False)

    return region_masks

def extract_trajectory(ar_da):
    '''
    Given an AR's binary valued DataArray, return a curve representing the path of the AR 
    through space and time (given as the spatial cluster centroid at each time step)

    Inputs:
        ar_da (xarray.DataArray): the binary valued DataArray
    Output:
        trajectory_df (pandas.DataFrame): a DataFrame where each row is the average 
            lat/lon of the storm at a particular time.
    '''
    times = ar_da.time.values

    avg_lons = []
    avg_lats = []

    for time in times:
        
        time_slice = ar_da.sel(time=time)
        inds = np.argwhere(time_slice.values == 1)
        storm_lats = time_slice.lat[inds[:,0]]
        storm_lons = time_slice.lon[inds[:,1]]

        time_slice_coords = pd.DataFrame({'lats':storm_lats, 'lons':storm_lons})
        time_slice_coords.name = '1'
        
        avg_angle = average_angle(time_slice_coords)

        avg_lons.append(avg_angle[2])
        avg_lats.append(avg_angle[1])

    trajectory_df = pd.DataFrame({'time': times, 'avg_lon': avg_lons, 'avg_lat': avg_lats})

    return trajectory_df

def compute_cumulative(storm_da, var_da, area_da, ais_da=None):
    '''
    Compute the cumulative amount of quantity underneath the footprint of the AR, considering
        spatial size of each grid cell. Example: cumulative snowfall due to AR

    Inputs:
        storm_da (xarray.DataArray): the binary AR mask
        var_da (xarray.DataArray): the DataArray with the atmospheric quantity we would like to compute on
        area_da (xarray.DataArray): the DataArray with cell areas
        ais_da (xarra.DataArray): the binary mask for the AIS (if supplied, compute cumulative only over AIS)

    Outputs:
        cumulative_storm_val (float): cumulative quantity underneath footprint of storm
    '''
    if ais_da is not None:
        storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon)
        storm_da_subset = storm_da.where(storm_ais_mask, 0)
    else:
        storm_da_subset = storm_da.copy() # just so we can work with the name storm_da_subset later

    var_da_subset = var_da.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)

    storm_cell_areas = area_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    amt_per_3hr = storm_cell_areas.dot((storm_da_subset*var_da_subset))
    cumulative_storm_val = float((amt_per_3hr).sum())

    return cumulative_storm_val

def compute_max_intensity(storm_da, var_da, area_da, ais_da=None):
    '''
    Compute the maximum intensity of some quantity underneath the footprint of the AR, throuhgout its lifetime.

    Inputs:
        storm_da (xarray.DataArray): the binary AR mask
        var_da (xarray.DataArray): the DataArray with the atmospheric quantity we would like to compute on
        area_da (xarray.DataArray): the DataArray with cell areas
        ais_da (xarra.DataArray): the binary mask for the AIS (if supplied, compute max only over AIS)

    Outputs:
        max_intensity_val (float): the value of the maximum intensity
    '''
    if ais_da is not None:
        storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon)
        storm_da_subset = storm_da.where(storm_ais_mask, 0)
    else:
        storm_da_subset = storm_da.copy() # just so we can work with the name storm_da_subset later
        
    var_da_subset = var_da.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)
    max_intensity_val = float((storm_da_subset*var_da_subset).max())

    return max_intensity_val

def compute_min_SLP(storm_da, var_da, area_da, ais_da):
    '''
    Compute the minimum SLP over the ocean at the time of first landfall.

    Inputs:
        storm_da (xarray.DataArray): the binary AR mask
        var_da (xarray.DataArray): the DataArray with the atmospheric quantity we would like to compute on
        area_da (xarray.DataArray): the DataArray with cell areas
        ais_da (xarra.DataArray): the binary mask for the AIS (if supplied, compute only over AIS)

    Outputs:
        min_slp (float): the minimum SLP value over the ocean
    '''

    # grab the intersection of storm with ocean and AIS, respectively
    storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    storm_ocean_mask = np.logical_not(storm_ais_mask)
    storm_da_ais = storm_da.where(storm_ais_mask, 0)
    storm_da_ocean = storm_da.where(storm_ocean_mask, 0)
    
    var_da_subset = var_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    first_landfall = np.min(storm_da.time[storm_da_ais.any(dim=['lat', 'lon'])].values)
    
    first_day = (storm_da_ocean*var_da_subset).sel(time=first_landfall).values
    min_slp = np.min(first_day[first_day > 0], initial=99999999) # if there's no portion over the ocean, fill in a missing value

    return min_slp

def compute_max_SLPgrad(storm_da, var_da, area_da, ais_da):
    '''
    Compute the  maximum SLP pressure gradient over ocean at time of first landfall.

    Inputs:
        storm_da (xarray.DataArray): the binary AR mask
        var_da (xarray.DataArray): the DataArray with the atmospheric quantity we would like to compute on
        area_da (xarray.DataArray): the DataArray with cell areas
        ais_da (xarra.DataArray): the binary mask for the AIS (if supplied, compute max only over AIS)

    Outputs:
        max_grad (float): the max SLP gradient value
    '''

    storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    storm_ocean_mask = np.logical_not(storm_ais_mask)
    storm_da_ais = storm_da.where(storm_ais_mask, 0)
    storm_da_ocean = storm_da.where(storm_ocean_mask, 0)

    var_da_subset = var_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    first_landfall = np.min(storm_da.time[storm_da_ais.any(dim=['lat', 'lon'])].values)

    var_da_subset_landfall = var_da_subset.sel(time=first_landfall)
    storm_da_ocean_landfall = storm_da_ocean.sel(time=first_landfall)

    # return -1 if no points over the ocean at first landfalling time
    if (storm_da_ocean_landfall == 0).all().values:
        return -1
    ## compute pressure gradient
    # convert to radians
    rads = var_da_subset_landfall.assign_coords(lon=np.radians(var_da_subset_landfall.lon), lat=np.radians(var_da_subset_landfall.lat))
    # partials in the latitude direction (spherical coordinates)
    r = 6378 # radius of Earth in km
    lat_partials = rads.differentiate('lat')/r
    # partials in the longitudinal direction (spherical coordinates)
    lon_partials = rads.differentiate('lon')/(np.sin(rads.lat)*r)
    
    magnitude = np.sqrt(lon_partials**2 + lat_partials**2)
    max_grad = np.max(magnitude.values*storm_da_ocean_landfall.values)

    return float(max_grad)

def compute_avg_landfalling_minomega(storm_da, var_da, area_da, ais_da):
    '''
    Function to compute the landfalling omega. Computed by finding the minimum
    omega at each grid cell at the time of first landfall, and then finding
    a spatial average of all of these minimum omegas. Minimum omega because
    upward lift is defined as negative. This is motivated by Baiman 2023,
    Figure 11, which shows that at landfall, omega varies with location and the height
    of the atmosphere, so we should consider both different levels of the atmopshere
    and spatial patterns.

    Inputs:
        storm_da (xarray.DataArray): the storm binary DataArray
        var_da (xarray.DataArray): the omega DataArray, with 42 different pressure levels
        area_da (xarray.DataArray): the data array with areas of grid cells
        ais_da (xarray.DataArray): binary mask DataArray indicating where the AIS is
        
    Outputs:
        omega_agg (float): the aggregate landfalling omega for that storm
    '''
    
    storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    storm_da_ais = storm_da.where(storm_ais_mask, 0)
    first_landfall = np.min(storm_da.time[storm_da_ais.any(dim=['lat', 'lon'])].values)
    storm_cell_areas = area_da.sel(lat=storm_da.lat, lon=storm_da.lon)

    var_da_subset = var_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    var_da_subset_landfall = var_da_subset.sel(time=first_landfall)
    var_da_agg = var_da_subset_landfall.min('lev')
    
    storm_da_landfall = storm_da_ais.sel(time=first_landfall)
    
    tot_area = storm_da_landfall.dot(storm_cell_areas)
    avg_min_omega = (storm_cell_areas.dot(storm_da_landfall*var_da_agg)/tot_area).values

    return float(avg_min_omega)

def compute_max_landfalling_wind(storm_da, var_da, area_da, ais_da):
    '''
    Compute the max landfalling 850 hPa over the ocean, at the time of first landfall.

    Inputs:
        storm_da (xarray.DataArray): the binary AR mask
        var_da (xarray.DataArray): the DataArray with the atmospheric quantity we would like to compute on
        area_da (xarray.DataArray): the DataArray with cell areas
        ais_da (xarra.DataArray): the binary mask for the AIS (if supplied, compute max only over AIS)

    Outputs:
        max_wind (float): the maximum 850 hPa wind over ocean at time of first landfall
    '''

    storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    storm_ocean_mask = np.logical_not(storm_ais_mask)
    storm_da_ais = storm_da.where(storm_ais_mask, 0)
    storm_da_ocean = storm_da.where(storm_ocean_mask, 0)
    first_landfall = np.min(storm_da.time[storm_da_ais.any(dim=['lat', 'lon'])].values)

    var_da_subset = var_da.sel(lat=storm_da.lat, lon=storm_da.lon)

    # if no portion of the storm over ocean at time of first landfall, give placeholder missing -1
    storm_da_ocean_landfall = storm_da_ocean.sel(time=first_landfall)
    if (storm_da_ocean_landfall == 0).all().values:
        return -1
    
    first_day = (storm_da_ocean*var_da_subset).sel(time=first_landfall).values
    # for some reason, 850hPa wind is null over the AIS. So, we take the max over all ocean
    # points, ignoring an NaNs
    # if the 850hPa wind is still null over all of those points, return a placeholder missing value
    max_wind = np.nanmax(first_day, initial=-999999)

    return max_wind

def compute_avg_landfalling_wind(storm_da, var_da, area_da, ais_da):
    '''
    Compute the avg landfalling 850 hPa over the ocean, at the time of first landfall.

    Inputs:
        storm_da (xarray.DataArray): the binary AR mask
        var_da (xarray.DataArray): the DataArray with the atmospheric quantity we would like to compute on
        area_da (xarray.DataArray): the DataArray with cell areas
        ais_da (xarra.DataArray): the binary mask for the AIS (if supplied, compute max only over AIS)

    Outputs:
        avg_wind (float): the average 850 hPa wind over ocean at time of first landfall
    '''

    storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    storm_ocean_mask = np.logical_not(storm_ais_mask)
    storm_da_ais = storm_da.where(storm_ais_mask, 0)
    storm_da_ocean = storm_da.where(storm_ocean_mask, 0)
    first_landfall = np.min(storm_da.time[storm_da_ais.any(dim=['lat', 'lon'])].values)

    storm_da_ocean_landfall = storm_da_ocean.sel(time=first_landfall)
    if (storm_da_ocean_landfall == 0).all().values:
        return -1

    var_da_subset = var_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    # for some reason, 850hPa wind is null for values in and around the AIS
    # so, we compute over the ocean, where the 850hPa wind is also not null
    notnull = var_da_subset.notnull()
    # fill null locations with 0; will be excluded from computation anyway
    var_da_subset = var_da_subset.fillna(0)
    storm_da_ocean_notnull = storm_da_ocean*notnull
    storm_cell_areas = area_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    tot_area = storm_da_ocean_notnull.dot(storm_cell_areas)
    
    avg_wind = (storm_cell_areas.dot(storm_da_ocean_notnull*var_da_subset)/tot_area).sel(time=first_landfall).values

    return avg_wind

def compute_average(storm_da, var_da, area_da, ais_da=None):
    '''
    Compute a spatial average of some quantity underneath the footprint of the AR.

    Inputs:
        storm_da (xarray.DataArray): the binary AR mask
        var_da (xarray.DataArray): the DataArray with the atmospheric quantity we would like to compute on
        area_da (xarray.DataArray): the DataArray with cell areas
        ais_da (xarra.DataArray): the binary mask for the AIS (if supplied, compute avg only over AIS)

    Outputs:
        avg_storm_val (float): the spatial average underneath the footprint, averaged over time as well
    '''
    
    if ais_da is not None:
        storm_ais_mask = ais_da.sel(lat=storm_da.lat, lon=storm_da.lon)
        storm_da_subset = storm_da.where(storm_ais_mask, 0)
    else:
        storm_da_subset = storm_da.copy()

    var_da_subset = var_da.sel(lat=storm_da_subset.lat, lon=storm_da_subset.lon)

    storm_cell_areas = area_da.sel(lat=storm_da.lat, lon=storm_da.lon)
    tot_area = storm_da_subset.dot(storm_cell_areas)
    avg_storm_val = float((storm_cell_areas.dot(storm_da_subset*var_da_subset)/tot_area).mean())

    return avg_storm_val

def augment_storm_da(storm_da):
    '''
    For any grid cell which had AR conditions, extend AR conditions to all grid cells 24 hours later.
        To be used as input for functions computing cumulative precipitation, as it is generally accepted
        that precip that falls up to 24 hours after the AR footprint leaves a cell is still attributable to
        the AR.

    Inputs:
        storm_da (xarray.DataArray): the original AR binary mask to be augmented

    Outputs:
        augmented_da (xarray.DataArray): the DataArray with mask extended on the time axis for 24 hours
    '''
    
    start = storm_da.time.values[0]
    end = storm_da.time.values[-1] + np.timedelta64(1, 'D')
    full_dates=pd.date_range(start, end, freq='3h')
    
    unincluded_times = set(np.array(full_dates)) - set(storm_da.time.values)
    
    unincluded_array = np.zeros((len(unincluded_times), storm_da.shape[1], storm_da.shape[2]))
    unincluded_coords = {'time' : np.array(list(unincluded_times)), 'lat': storm_da.lat.values, 'lon': storm_da.lon.values}
    unincluded_da = xr.DataArray(unincluded_array, coords=unincluded_coords)
    
    augmented_da = xr.concat([storm_da, unincluded_da], dim='time')
    augmented_da = augmented_da.rolling(time=8, min_periods=1).max()
    
    return augmented_da
