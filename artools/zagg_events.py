'''
Module to export a monolithic AR catalog into per-storm zagg event files.

zagg's temporal pipeline (englacial/zagg#213) consumes events by URI, one mask
per file. This module explodes one catalog HDF5 (a pandas DataFrame with a
`data_array` mask column, as written by the clustering pipeline and read by
loading_utils.load_catalog) into that layout: per-storm raw and 24-hour
augmented masks as single-variable NetCDFs, plus a parquet event catalog with
time ranges, bounding boxes, and the ST-DBSCAN hyperparameter provenance
parsed from the catalog filename. Precipitation attributes aggregate under the
augmented mask (see attribute_utils.augment_storm_da), so both masks ship for
every storm; zagg runs once per mask family and the outputs join on event_key.

Jimmy Butler & co.
July 2026
'''

import re
from pathlib import Path

import numpy as np
import pandas as pd

# augment_storm_da is imported lazily inside export_events: module import must
# not pull the streaming stack (ray/earthaccess) that attribute_utils drags in.

# Catalog filenames encode the clustering hyperparameters, e.g.
# epsspace0.5_epstime12_minpts5_nreppts10_seed12345.h5
_FNAME_RE = re.compile(
    r'epsspace(?P<eps_space>[\d.]+)_epstime(?P<eps_time>[\d.]+)'
    r'_minpts(?P<min_pts>\d+)_nreppts(?P<n_rep_pts>\d+)_seed(?P<seed>\d+)'
)


def parse_hyperparams(fname):
    '''
    Parse the ST-DBSCAN hyperparameters encoded in a catalog filename.

    Inputs:
        fname (string or Path): catalog filename, e.g.
            'epsspace0.5_epstime12_minpts5_nreppts10_seed12345.h5'

    Outputs:
        params (dictionary): eps_space and eps_time as floats; min_pts,
            n_rep_pts, and seed as ints
    '''
    match = _FNAME_RE.search(Path(fname).stem)
    if not match:
        raise ValueError(
            f'catalog filename {fname!r} does not encode hyperparameters '
            '(expected epsspace*_epstime*_minpts*_nreppts*_seed*)'
        )
    groups = match.groupdict()
    return {
        'eps_space': float(groups['eps_space']),
        'eps_time': float(groups['eps_time']),
        'min_pts': int(groups['min_pts']),
        'n_rep_pts': int(groups['n_rep_pts']),
        'seed': int(groups['seed']),
    }


def export_events(catalog_path, out_dir=None, storms=None, overwrite=False):
    '''
    Explode one catalog HDF5 into per-storm zagg event files.

    Writes, under out_dir:
        masks/storm_<id>.nc            the raw binary mask (time, lat, lon)
        masks_augmented/storm_<id>.nc  the 24-hour lookahead mask
        events.parquet                 one row per storm (the event catalog)

    Masks are single-variable files (variable name 'mask', int8) so zagg's
    reader opens them directly as DataArrays. Existing mask files are kept
    unless overwrite is set, so an interrupted export resumes cheaply; the
    parquet is always rewritten to cover the requested storms. Resumed files
    are trusted as-written: the parquet rows are recomputed from the source
    h5 while the on-disk masks are left untouched, so if the mask-writing code
    changes between runs (e.g. augment_storm_da) the two can silently diverge.
    After changing mask-writing code, re-run with overwrite=True.

    Inputs:
        catalog_path (string or Path): path to a catalog .h5 (the same file
            format loading_utils.load_catalog downloads from HuggingFace)
        out_dir (string or Path): output root; defaults to
            <catalog dir>/derived/<catalog stem>/
        storms (iterable): storm ids (catalog index values) to export;
            default is every storm in the catalog
        overwrite (boolean): if True, re-write mask files that already exist

    Outputs:
        events (pd.DataFrame): the event catalog, one row per storm
        parquet_path (Path): where events was written
    '''
    catalog_path = Path(catalog_path)
    hyperparams = parse_hyperparams(catalog_path.name)
    # same read path as loading_utils.load_catalog, minus the HF download
    catalog = pd.read_hdf(catalog_path)

    if out_dir is None:
        out_dir = catalog_path.parent / 'derived' / catalog_path.stem
    out_dir = Path(out_dir)
    masks_dir = out_dir / 'masks'
    augmented_dir = out_dir / 'masks_augmented'
    masks_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir.mkdir(parents=True, exist_ok=True)

    if storms is not None:
        catalog = catalog.loc[list(storms)]

    from .attribute_utils import augment_storm_da

    rows = []
    for storm_id, row in catalog.iterrows():
        event_key = f'storm_{int(storm_id):05d}'
        raw = row['data_array']
        raw_path = masks_dir / f'{event_key}.nc'
        augmented_path = augmented_dir / f'{event_key}.nc'

        if overwrite or not (raw_path.exists() and augmented_path.exists()):
            augmented = augment_storm_da(raw)
            raw.rename('mask').astype('int8').to_netcdf(raw_path)
            augmented.rename('mask').astype('int8').to_netcdf(augmented_path)

        times = raw['time'].values
        rows.append({
            'event_key': event_key,
            'storm_id': int(storm_id),
            'is_landfalling': bool(row.get('is_landfalling', False)),
            't_start': times[0],
            't_end': times[-1],
            # augment_storm_da extends the time axis exactly one day past the
            # storm's last timestep (the precip attribution window)
            't_end_augmented': times[-1] + np.timedelta64(1, 'D'),
            'n_timesteps': int(raw.sizes['time']),
            'lat_min': float(raw['lat'].min()),
            'lat_max': float(raw['lat'].max()),
            'lon_min': float(raw['lon'].min()),
            'lon_max': float(raw['lon'].max()),
            'mask_path': str(raw_path),
            'mask_augmented_path': str(augmented_path),
            'catalog_file': catalog_path.name,
            **hyperparams,
        })

    events = pd.DataFrame(rows)
    parquet_path = out_dir / 'events.parquet'
    events.to_parquet(parquet_path)
    return events, parquet_path
