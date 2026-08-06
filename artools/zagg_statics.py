'''
Module to prepare the static fields the zagg AR configs consume.

zagg's temporal worker reads three static fields (englacial/zagg#213): the
AIS mask (drives the ais/ocean mask providers and the first_landfall
trigger), grid-cell areas (the area-weighted spatial functions), and the
monthly climatology (the anomaly transform). The source files ship with the
AR catalogs dataset; this module normalizes them into single-variable
NetCDFs under <catalogs dir>/static/ — the layout the zagg configs point at
and mirror_to_s3 uploads — and loads them as in-memory statics for the local
backend. Coordinates are rounded to 5 decimals throughout, matching the
loading_utils convention, so `.sel` against storm-mask coordinates matches
exactly.

Jimmy Butler & co.
July 2026
'''

import logging
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)

# the sel(lat=...) window the whole pipeline works in (Southern Ocean + AIS)
LAT_SLICE = slice(-86, -39)

# monthly-mean single-level diagnostics (T2M, TQV) for the anomaly baseline
CLIMATOLOGY_DOI = '10.5067/5ESKGQTZG7FO'  # M2TMNXSLV


def _rounded(da):
    return da.assign_coords(lat=da.lat.round(5), lon=da.lon.round(5))


def prepare_static_files(catalogs_dir, *, climatology_time_range=None, overwrite=False):
    '''
    Normalize the static source files into <catalogs_dir>/static/.

    Writes ais_mask.nc (0/1 int8 from the Zwally basins file) and
    cell_areas.nc (m^2), both subset to LAT_SLICE with rounded coordinates;
    builds merra2_monthly_climatology.nc when a time range is given (it needs
    an Earthdata login and streams ~monthly granules, so it is opt-in).

    Inputs:
        catalogs_dir (string or Path): the AR catalogs directory (holds
            AIS_Full_basins_Zwally_MERRA2grid_new.nc and MERRA2_gridarea.nc)
        climatology_time_range (tuple of str): (start, end) YYYY-MM-DD
            baseline for the monthly climatology; None skips building it
        overwrite (boolean): re-write outputs that already exist

    Outputs:
        paths (dictionary): {static_name: Path} for every file present
    '''
    catalogs_dir = Path(catalogs_dir)
    static_dir = catalogs_dir / 'static'
    static_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    ais_path = static_dir / 'ais_mask.nc'
    if overwrite or not ais_path.exists():
        basins = xr.open_dataset(catalogs_dir / 'AIS_Full_basins_Zwally_MERRA2grid_new.nc')
        ais = (basins['Zwallybasins'] > 0).sel(lat=LAT_SLICE)
        _rounded(ais).astype('int8').rename('ais_mask').to_netcdf(ais_path)
    paths['ais_mask'] = ais_path

    areas_path = static_dir / 'cell_areas.nc'
    if overwrite or not areas_path.exists():
        areas = xr.open_dataset(catalogs_dir / 'MERRA2_gridarea.nc')['cell_area']
        areas = areas.sel(lat=LAT_SLICE)
        _rounded(areas).rename('cell_areas').to_netcdf(areas_path)
    paths['cell_areas'] = areas_path

    clim_path = static_dir / 'merra2_monthly_climatology.nc'
    if climatology_time_range is not None and (overwrite or not clim_path.exists()):
        build_climatology(climatology_time_range, clim_path)
    if clim_path.exists():
        paths['climatology'] = clim_path

    return paths


def build_climatology(time_range, out_path, variables=('T2M', 'TQV')):
    '''
    Build the monthly climatology the anomaly attributes subtract.

    Streams the MERRA-2 monthly single-level means (M2TMNXSLV) over the
    baseline range through earthaccess and averages by calendar month —
    the same baseline construction the reference pipeline groups against
    (climatology[var] indexed by a month dimension).

    Inputs:
        time_range (tuple of str): (start, end) YYYY-MM-DD baseline period
        out_path (string or Path): NetCDF destination
        variables (tuple of str): fields to keep

    Outputs:
        climatology (xr.Dataset): month-indexed means of `variables`
    '''
    import earthaccess

    earthaccess.login()
    granules = earthaccess.search_data(doi=CLIMATOLOGY_DOI, temporal=time_range, count=-1)
    logger.info('building climatology from %d monthly granules', len(granules))
    # parallel opens: serial metadata reads over https make multi-decade
    # baselines pathologically slow (dask is a core dependency)
    ds = xr.open_mfdataset(earthaccess.open(granules), parallel=True)[list(variables)]
    ds = ds.sel(lat=LAT_SLICE)
    climatology = _rounded(ds).groupby('time.month').mean('time').compute()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    climatology.to_netcdf(out_path)
    return climatology


def load_zagg_statics(catalogs_dir, *, with_climatology=True):
    '''
    Load the prepared statics as the in-memory dict zagg's local backend needs.

    Inputs:
        catalogs_dir (string or Path): the AR catalogs directory (run
            prepare_static_files first)
        with_climatology (boolean): include the climatology (required by the
            anomaly attributes; ar_precip runs don't need it)

    Outputs:
        static_data (dictionary): {'ais_mask': DataArray, 'cell_areas':
            DataArray, 'climatology': Dataset?} ready for local_event_tuples
    '''
    static_dir = Path(catalogs_dir) / 'static'
    statics = {
        'ais_mask': xr.open_dataarray(static_dir / 'ais_mask.nc'),
        'cell_areas': xr.open_dataarray(static_dir / 'cell_areas.nc'),
    }
    if with_climatology:
        statics['climatology'] = xr.open_dataset(
            static_dir / 'merra2_monthly_climatology.nc'
        )
    return statics
