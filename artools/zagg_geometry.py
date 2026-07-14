'''
Module for the mask-geometry storm attributes and the final table assembly.

The geometry attributes (areas, duration, dates, southward extent, landfall
region, trajectory) derive from the storm masks and static fields alone — no
MERRA-2 streaming — so they stay client-side (englacial/zagg#213) while zagg
computes the field attributes. This module mirrors the dataset_construction
notebook's column construction, reading per-storm masks exported by
zagg_events instead of the monolithic catalog, and joins the geometry columns
with the zagg Parquet outputs into the final storm attribute table.

Jimmy Butler & co.
July 2026
'''

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# longitude sectors from the dataset_construction notebook ('East 2' wraps the
# dateline; find_region_masks handles the wrap)
DEFAULT_REGION_DEFS = {
    'West': [-150, -30],
    'East 1': [-30, 75],
    'East 2': [75, -150],
}


def geometry_attributes(
    events, statics, *, region_defs=None, include_trajectory=False
):
    '''
    The mask-only attributes for a set of exported storms.

    One row per storm, with the dataset_construction notebook's columns:
    max_area, mean_landfalling_area, cumulative_landfalling_area (all int,
    km^2 / km^2 / km^2·days), duration (hours), start_date, end_date,
    max_south_extent (deg lat), and region (the landfalling longitude
    sector). region is pd.NA for non-landfalling storms — the sector
    cumulative-spacetime values are all 0 there, so idxmax would return an
    arbitrary label; the landfalling-area columns stay 0 (well-defined) for
    those storms. trajectory (a per-timestep DataFrame) is an object column
    that cannot ride into Parquet, so it is opt-in for in-memory use only.

    Inputs:
        events (pd.DataFrame): rows from zagg_events.export_events
        statics (dictionary): from zagg_statics.load_zagg_statics
            (needs ais_mask and cell_areas; climatology is not used)
        region_defs (dictionary): {label: [lon_west, lon_east]} sectors;
            defaults to the notebook's three-sector split
        include_trajectory (boolean): add the trajectory object column

    Outputs:
        geometry (pd.DataFrame): event_key + geometry columns
    '''
    # lazy: attribute_utils drags in the streaming stack (ray/earthaccess)
    from .attribute_utils import (
        add_end_date,
        add_start_date,
        compute_cumulative_spacetime,
        compute_duration,
        compute_max_area,
        compute_max_southward_extent,
        compute_mean_area,
        extract_trajectory,
        find_landfalling_region,
        find_region_masks,
        is_landfalling,
    )

    ais_mask = statics['ais_mask'].astype(bool)
    cell_areas = statics['cell_areas']
    region_masks = find_region_masks(region_defs or DEFAULT_REGION_DEFS, ais_mask)

    rows = []
    for _, event in events.iterrows():
        mask = xr.open_dataarray(event['mask_path'])
        mask = mask.assign_coords(lat=mask.lat.round(5), lon=mask.lon.round(5))
        # region is only well-defined for landfalling storms: the sector CLAs
        # are all 0 otherwise and idxmax returns an arbitrary label. Prefer the
        # catalog's is_landfalling flag; fall back to a mask∩AIS test when the
        # column is absent.
        if 'is_landfalling' in event.index:
            landfalling = bool(event['is_landfalling'])
        else:
            landfalling = bool(is_landfalling(mask, ais_mask))
        row = {
            'event_key': event['event_key'],
            'max_area': int(compute_max_area(mask, cell_areas)),
            'mean_landfalling_area': int(compute_mean_area(mask, cell_areas, ais_mask)),
            'cumulative_landfalling_area': int(
                compute_cumulative_spacetime(mask, cell_areas, ais_mask)
            ),
            'duration': compute_duration(mask),
            'start_date': add_start_date(mask),
            'end_date': add_end_date(mask),
            'max_south_extent': compute_max_southward_extent(mask),
            'region': (
                find_landfalling_region(mask, cell_areas, region_masks)
                if landfalling
                else pd.NA
            ),
        }
        if include_trajectory:
            row['trajectory'] = extract_trajectory(mask)
        rows.append(row)
    return pd.DataFrame(rows)


def assemble_attribute_table(events, geometry, zagg_outputs, out_path=None):
    '''
    Join geometry and zagg field/precip outputs into the final storm table.

    zagg writes flat Parquet — event_key plus one column per attribute (its
    per-run bookkeeping columns, e.g. timesteps_processed, are dropped here) —
    so the assembly is a left-merge of every piece onto the event catalog.
    Storms a zagg run errored on surface as NaN attribute rows, not dropped
    rows.

    Inputs:
        events (pd.DataFrame): rows from zagg_events.export_events (supplies
            event_key, storm_id, is_landfalling, and hyperparameter columns)
        geometry (pd.DataFrame): from geometry_attributes
        zagg_outputs (iterable): Parquet paths (or DataFrames) of zagg runs,
            e.g. the ar_attributes and ar_precip results
        out_path (string or Path): optional Parquet destination

    Outputs:
        table (pd.DataFrame): one row per storm
    '''
    keep = ['event_key', 'storm_id', 'is_landfalling', 'catalog_file',
            'eps_space', 'eps_time', 'min_pts', 'n_rep_pts', 'seed']
    table = events[[c for c in keep if c in events.columns]].copy()
    # geometry comes from our own one-row-per-storm loop, but the 1:1 guard is
    # cheap insurance; zagg outputs must be unique per key (m:1) so a duplicated
    # event_key fails loudly instead of silently fanning out storm rows.
    table = table.merge(geometry, on='event_key', how='left', validate='1:1')
    for output in zagg_outputs:
        frame = output if isinstance(output, pd.DataFrame) else pd.read_parquet(output)
        frame = frame.drop(columns=['timesteps_processed'], errors='ignore')
        overlap = set(table.columns) & set(frame.columns) - {'event_key'}
        if overlap:
            raise ValueError(f'zagg output would overwrite existing columns: {sorted(overlap)}')
        table = table.merge(frame, on='event_key', how='left', validate='m:1')
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(out_path)
        logger.info('wrote %d-storm attribute table to %s', len(table), out_path)
    return table
