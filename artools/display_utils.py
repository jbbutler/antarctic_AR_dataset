'''
Module with functions that aid in the display of the AR catalogs, from constructing thumbnails
to plotting steographic projections of Antarctica.

Jimmy Butler
November 2025
'''

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.path as mpath
import matplotlib.colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

import matplotlib.patheffects as pe

from matplotlib import animation
import matplotlib
from matplotlib.cm import prism

from tqdm.auto import tqdm

from io import BytesIO
import base64

def construct_thumbnail(storm):
    '''
    Make a thumbnail image for a particular AR storm.

    Inputs:
        storm (xarray.DataArray): a binary-valued dataarray indicating pixels associated with this storm

    Outputs:
        imgstr (string): an html image tag for the thumbnail
    '''
    rep_time = storm.sel(time=storm.sum(dim=['lat', 'lon']).idxmax())
    fig, ax = plt.subplots(1)
    ax.imshow(rep_time.to_numpy())
    ax.invert_yaxis()
    ax.axis('off')
    fig.set_size_inches((0.5,0.5))
    plt.close()

    # following strategy taken from the following stackexchange
    # https://stackoverflow.com/questions/47038538/insert-matplotlib-images-into-a-pandas-dataframe
    figfile = BytesIO()
    fig.savefig(figfile, format='png', pad_inches=0, bbox_inches='tight')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode()
    imgstr = '<img src="data:image/png;base64,{}" />'.format(figdata_png)

    return imgstr

def construct_trajectory_thumbnail(traj_df):
    '''
    Make a thumbnail image for a storm's trajectory.

    Inputs:
        traj_df (pd.DataFrame or None): dataframe with 'lat', 'lon', and 'time' columns

    Outputs:
        imgstr (string): an html image tag for the trajectory thumbnail
    '''

    fig, ax = plt.subplots(1)
    
    # plot longitude on the x-axis and latitude on the y-axis
    ax.plot(traj_df['avg_lon'], traj_df['avg_lat'], color='black', linewidth=2)
    ax.plot(traj_df['avg_lon'].iloc[0], traj_df['avg_lat'].iloc[0], marker='o', color='red', markersize=4)
    ax.plot(traj_df['avg_lon'].iloc[-1], traj_df['avg_lat'].iloc[-1], marker='o', color='red', markersize=4)

    ax.axis('off')
    fig.set_size_inches((0.5, 0.5))
    plt.close()

    # Convert to base64 html string
    figfile = BytesIO()
    fig.savefig(figfile, format='png', pad_inches=0, bbox_inches='tight')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode()
    imgstr = '<img src="data:image/png;base64,{}" />'.format(figdata_png)

    return imgstr

def display_catalog(catalog_df, nrows=None):
    '''
    Render the catalog as a standard pd.DataFrame, but show the data_array 
    and trajectory of each AR as user-friendly thumbnails. Behaves as pd.DataFrame.head().
    By default, pandas seeks to print a string representation of each AR's xarray.DataArray,
    which can be very slow to print. This function allows for AR DataFrames to be printed
    in a much more readable manner.

    Inputs:
        catalog_df (pandas.DataFrame): a DataFrame with a data_array column consisting of binary-valued DataArrays
            for each AR
        nrows (int): optionally, how many rows of the DataFrame to display
        
    Outputs:
        df_to_display (pandas.DataFrame): a more user friendly DataFrame
    '''
    # create a dictionary of formatters
    formatters = {'data_array': construct_thumbnail}
    
    # add trajectory formatter if the column exists
    if 'trajectory' in catalog_df.columns:
        formatters['trajectory'] = construct_trajectory_thumbnail

    # filter by rows if specified
    df_to_display = catalog_df.head(nrows) if nrows else catalog_df

    # apply the formatters dictionary
    return df_to_display.style.format(formatters).format_index(precision=0)


def format_polar_axis(ax):
    '''
    Helper function with all of the routines to generate a stereographic projection of Antarctica.

    Inputs:
        ax (cartopy.GeoAxes): a GeoAxes map object to draw Antarctica on, modified in-place
    '''
    ax.set_extent([-180, 180, -90, -39], ccrs.PlateCarree())
    # add ice shelf lines and polygons, coastlines
    ice_shelf_poly = cfeature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', 
                                                  '50m', edgecolor='none', facecolor='lightcyan')
    ax.add_feature(ice_shelf_poly, linewidth=3)
    ice_shelf_line = cfeature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_lines', 
                                                  '50m', edgecolor='black', facecolor='none')
    ax.add_feature(ice_shelf_line, linewidth=1, zorder=13)
    ax.coastlines(resolution='110m', linewidth=1, zorder=32)

    # add grid lines
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.gridlines(alpha=0.5, zorder=33)

def plot_stormtime_frame(ax, time_pt, stormtime_df, color_mapping=None):
    '''
    Plot all of the ARs present around Antarctica at a particular time slice. If time slice
    is not in the DataFrame of AR times, a blank map of Antarctica is returned.

    Inputs:
        ax (cartopy.GeoAxes): the axis on which to plot the ARs and Antarctica (modified in-place)
        time_pt (datetime64[ns]): the time slice to plot
        stormtime_df (pd.DataFrame): a DataFrame where each row corresponds to an AR at a
            particular point in time. Contains columns with lists of latitudes and longitudes
            of each AR time.
        color_mapping (dict): mapping from cluster/AR number to color, used for plotting
    '''
    if color_mapping is None:
        ar_colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd',
                     '#8c564b', '#c51b8a', '#7f7f7f', '#1b9e77']
        unique_clusters = list(dict.fromkeys(stormtime_df['label']))
        color_mapping = {unique_clusters[j]: ar_colors[j % len(ar_colors)]
                         for j in range(len(unique_clusters))}
    if (time_pt == stormtime_df.time).any():
        dat = stormtime_df[stormtime_df['time'] == time_pt]
        n_clusts = dat.shape[0]
        for i in range(n_clusts):
            cluster = int(dat['label'].iloc[i])
            lon = np.asarray(dat['lon'].iloc[i])
            lat = np.asarray(dat['lat'].iloc[i])
            ax.scatter(lon, lat, transform=ccrs.PlateCarree(),
                       s=1, color=color_mapping[cluster], zorder=30)
            # in-image label at the AR centroid (replaces the legend so the
            # label follows each AR around the grid)
            # a little lon shift so it doesn't overlay all the points
            # compute averages
            lon = np.radians(lon)
            lat = np.radians(lat)
            x = np.cos(lat)*np.cos(lon)
            y = np.cos(lat)*np.sin(lon)
            z = np.sin(lat)

            avg_x = np.mean(x)
            avg_y = np.mean(y)
            avg_z = np.mean(z)

            avg_lat = np.degrees(np.arcsin(avg_z))
            avg_lon = np.degrees(np.atan2(avg_y, avg_x))
            
            txt = ax.text(avg_lon+10, avg_lat+10, str(cluster),
                          transform=ccrs.PlateCarree(), fontsize=8,
                          fontweight='bold', ha='center', va='center',
                          color='black',
                          bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                    edgecolor='0.3', linewidth=0.6),
                          zorder=100)
            txt.set_path_effects([pe.Stroke(linewidth=1.8, foreground='white'),
                                  pe.Normal()])
    format_polar_axis(ax)
    time_ts = pd.Timestamp(time_pt)
    ax.set_title(f'{time_ts.month}/{time_ts.day}/{time_ts.year}, {time_ts.hour}:00')


def plot_stormtime_grid(start_time, end_time, stormtime_df, freq, ncols, figsize=(6,6)):
    '''
    Plots side-by-side frames of storm events at a specified frequency within a specified time window.
    
    Inputs:
        start_time (str or pd.Timestamp): The beginning of your event (for example, '2005-01-01 00:00')
        end_time (str or pd.Timestamp): The end of your event (for example, '2005-01-02 12:00')
        stormtime_df (pd.DataFrame): dataframe in stormtime format (see plot_stormtime_frame)
        freq (str): the frequency at which to plot frames
        ncols (int): number of columns in the subplot grid

    Outputs:
        fig (matplotlib.figure.Figure), axes (cartopy.mpl.geoaxes.GeoAxes)
    '''
    times_to_plot = pd.date_range(start=start_time, end=end_time, freq=freq)

    n_frames = len(times_to_plot)
    nrows = int(np.ceil(n_frames / ncols))

    # generate a consistent color mapping
    ar_colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd',
                 '#8c564b', '#c51b8a', '#7f7f7f', '#1b9e77']
    unique_clusters = list(dict.fromkeys(stormtime_df['label']))
    color_mapping = {unique_clusters[j]: ar_colors[j % len(ar_colors)]
                     for j in range(len(unique_clusters))}

    # create the figure and axes grid with the stereographic projection
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize[0], figsize[1]),
        subplot_kw=dict(projection=ccrs.Stereographic(central_longitude=0., central_latitude=-90.)))

    # flatten axes for easy iteration
    axes_flat = axes.flatten() if n_frames > 1 else [axes]
    # plot each time point
    for i, time_pt in enumerate(times_to_plot):
        plot_stormtime_frame(axes_flat[i], time_pt, stormtime_df, color_mapping=color_mapping)

    # hide any unused axes in the grid
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')

    fig.tight_layout()
    return fig, axes


def plot_eulerian_frame(ax, time_pt, data_array):
    '''
    Plot unlabelled AR pixels at a particular point in time. Used to plot
    AR pixels from the Wille et al. 2021 vIVT threshold catalog.

    Inputs: 
        ax (matplotlib.axes.Axes): the axis object to generate the plot on (modified in-place)
        time_pt (np.datetime64): the time to plot
        data_array (xarray.DataArray): the binary valued DataArray
    '''
    dat = data_array.sel(time=time_pt)
    
    # mask out zeros so they remain transparent over the map
    dat_masked = dat.where(dat > 0)
    
    # plot the binary values
    ax.pcolormesh(dat.lon, dat.lat, dat_masked, transform=ccrs.PlateCarree(), 
                  cmap='Blues', vmin=0, vmax=1, zorder=30)
    
    format_polar_axis(ax)
    
    # ensure time formatting works smoothly with numpy.datetime64 from xarray
    time_val = time_pt.values if hasattr(time_pt, 'values') else time_pt
    time_ts = pd.Timestamp(time_val)
    ax.set_title(f'{time_ts.month}/{time_ts.day}/{time_ts.year}, {time_ts.hour}:00')


def make_movie(stormtime_df, title, save_path):
    '''
    Create and save an animation showing labelled ARs from the storm catalog.
    Can save as either mp4 or gif.

    Inputs:
        stormtime_df (pd.DataFrame): the DataFrame of storms to plot, in stormtime format
        title (string): the title to plot on the figure
        save_path (string): where to save the figure, along with the filename. Extension must be mp4 or gif.

    Outputs:
        ani (matplotlib.animation.FuncAnimation): the animation object
    '''
    movie_times = pd.date_range(start=stormtime_df.time.min(), end=stormtime_df.time.max(), freq='3h')
    unique_clusters = stormtime_df['label'].unique()
    
    prism_cmap = plt.get_cmap('prism')
    color_mapping = {unique_clusters[j]: prism_cmap(j/len(unique_clusters)) for j in range(len(unique_clusters))}

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(projection=ccrs.Stereographic(central_longitude=0., central_latitude=-90.)))
    plot_stormtime_frame(ax, movie_times[0], stormtime_df, color_mapping)
    fig.suptitle(title, fontsize=16)

    def update_img(i):
        ax.clear()
        plot_stormtime_frame(ax, movie_times[i], stormtime_df, color_mapping)

    ani = animation.FuncAnimation(fig, update_img, frames=len(movie_times))
    print(f"Saving animation to {save_path}...")
    filetype = save_path.split('.')[-1]

    if filetype == 'mp4':  
        with tqdm(total=len(movie_times)) as pbar:
            ani.save(save_path, writer='ffmpeg', progress_callback=lambda i, n: pbar.update(1))
    elif filetype == 'gif':
        with tqdm(total=len(movie_times)) as pbar:
            ani.save(save_path, progress_callback=lambda i, n: pbar.update(1))
    else:
        raise Exception('Unsupported file extension for animation.')
    
    plt.close(fig)
    return ani


def make_eulerian_movie(data_array, title, save_path):
    '''
    Create and save an animation showing the AR pixels from an eulerian pixelwise threshold catalog.
    Can save as either mp4 or gif.

    Inputs:
        data_array (xarray.DataArray): the binary valued dataarray indicating AR pixels
        title (string): the title to show over the movie
        save_path (string): where to save the movie, along with the filename. Only mp4 and gif supported.
    '''
    movie_times = data_array.time.values
    
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(projection=ccrs.Stereographic(central_longitude=0., central_latitude=-90.)))
    plot_eulerian_frame(ax, movie_times[0], data_array)
    fig.suptitle(title, fontsize=16)

    def update_img(i):
        ax.clear()
        plot_eulerian_frame(ax, movie_times[i], data_array)

    ani = animation.FuncAnimation(fig, update_img, frames=len(movie_times))
    print(f"Saving animation to {save_path}...")
    filetype = save_path.split('.')[-1]

    if filetype == 'mp4':  
        with tqdm(total=len(movie_times)) as pbar:
            ani.save(save_path, writer='ffmpeg', progress_callback=lambda i, n: pbar.update(1))
    elif filetype == 'gif':
        with tqdm(total=len(movie_times)) as pbar:
            ani.save(save_path, progress_callback=lambda i, n: pbar.update(1))
    else:
        raise Exception('Unsupported file extension for animation.')
    
    plt.close(fig)
    return ani