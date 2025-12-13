'''
Script with some helper functions that are more generally-purposed and are useful to call up
    in the other utils scripts.

Jimmy Butler
November 2025
'''

import numpy as np
import pandas as pd

def arctan(x, y):
    '''
    Fucntion to find the arctan of y/x. Had to make this function because by default,
        the argument to arctan is positive, numpy will not know whether it's because both
        sides of the triangle were negative or both were positive, so it will just output
        an angle in the positive quadrant, for example. So, this function returns an angle
        in the correct quadrant depending on the signs of y and x.

    Inputs:
        x (float): side 1 length
        y (float): side 2 length
    Returns:
        (float): the arctan of the ratio, in radians
    '''
    if y/x > 0:
        if x > 0:
            return(np.arctan(y/x))
        else:
            return(np.arctan(y/x)-np.pi)
    else:
        if x > 0:
            return(np.arctan(y/x))
        else:
            return(np.pi+np.arctan(y/x))
    

def average_angle(subdf):
    '''
    Given some dataframe of latitudes and longitudes, finds the average latitude and longitude.
        This is done by converting from spherical coordinates to cartesian (unit vector), so each
        coordinate is represented by a unit vector in cartesian coordinates, and then averaging these
        unit vectors and reconverting back to spherical. Follows the wikipedia article on circular mean.

    Inputs:
        subdf (pandas.DataFrame): a pandas DataFrame with a column of 'lats' and 'lons'
    Outputs:
        (3-tuple): the name of the dataframe (cluster label in our case), the average lat, and the average lon
    '''
    lats = np.radians(subdf.lats)
    lons = np.radians(subdf.lons)

    x = np.cos(lats)*np.cos(lons)
    y = np.cos(lats)*np.sin(lons)
    z = np.sin(lats)
    
    avg_x = np.mean(x)
    avg_y = np.mean(y)
    avg_z = np.mean(z)

    avg_lat = np.arcsin(avg_z)
    avg_lon = arctan(avg_x, avg_y)

    return (subdf.name, np.degrees(avg_lat), np.degrees(avg_lon))

