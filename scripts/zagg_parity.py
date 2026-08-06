'''
Parity harness: zagg temporal pipeline vs the attribute_utils reference.

Runs a handful of storms through BOTH compute paths — zagg's local backend
(configs/ar_attributes.yaml + configs/ar_precip.yaml, englacial/zagg#213) and
the reference attribute_utils kernels invoked the way
compute_attributes_streaming does — over the SAME opened MERRA-2 datasets,
then prints a per-attribute comparison table. Needs an Earthdata login
(~/.netrc) and the zagg extra installed; it streams real granules, so it is a
pre-merge validation script, not a CI test.

Usage:
    python scripts/zagg_parity.py [--catalog PATH] [--storms 1 3 4 ...]
        [--n-storms 6] [--catalogs-dir ../antarctic_AR_catalogs] [--rtol 1e-5]

Exit code 0 when every finite attribute agrees within rtol (NaNs must match
as NaNs), 1 otherwise.

Jimmy Butler & co.
July 2026
'''

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from artools.attribute_utils import (  # noqa: E402
    augment_storm_da,
    compute_average,
    compute_avg_landfalling_minomega,
    compute_avg_landfalling_wind,
    compute_cumulative,
    compute_max_intensity,
    compute_max_landfalling_wind,
    compute_max_SLPgrad,
    compute_min_SLP,
)
from artools.zagg_events import export_events  # noqa: E402
from artools.zagg_granules import build_granule_index, local_event_tuples  # noqa: E402
from artools.zagg_statics import load_zagg_statics, prepare_static_files  # noqa: E402

logger = logging.getLogger('zagg_parity')

DEFAULT_CATALOG = 'epsspace0.5_epstime12_minpts5_nreppts10_seed12345.h5'
FIELD_COLLECTIONS = ['merra2_slv', 'merra2_flx', 'merra2_v850', 'merra2_omega']


def _pick_storms(catalog_path, n):
    '''First n-1 landfalling storm ids plus one non-landfalling (NaN checks).'''
    catalog = pd.read_hdf(catalog_path)
    landfalling = catalog.index[catalog['is_landfalling']][: n - 1]
    ashore = catalog.index[~catalog['is_landfalling']][:1]
    return [int(i) for i in list(landfalling) + list(ashore)]


def _monthly_anomaly(var_da, climatology, name):
    '''The reference anomaly construction (compute_attributes_streaming).'''
    anomaly = var_da.groupby('time.month') - climatology[name]
    return anomaly.drop_vars('month')


def reference_attributes(storm_da, collections, statics):
    '''
    The 14 field attributes via the attribute_utils kernels.

    `collections` are the SAME raw lazy datasets the zagg tuples carry; the
    reference transforms (half-hour shift, per-3h precip totals, negation for
    southward-positive vIVT/v850) are applied here exactly as
    compute_attributes_streaming / compute_precip_summaries do.
    '''
    storm = storm_da.assign_coords(lat=storm_da.lat.round(5), lon=storm_da.lon.round(5))
    areas = statics['cell_areas']
    ais = statics['ais_mask'].astype(bool)
    climatology = statics['climatology']

    def _var(collection, name, half_hour):
        ds = collections[collection]
        if half_hour:
            ds = ds.assign_coords(time=ds.time - np.timedelta64(30, 'm'))
        da = ds[name].sel(lat=storm.lat, lon=storm.lon, time=storm.time)
        return da.compute()

    out = {}
    t2m = _var('merra2_slv', 'T2M', half_hour=False)
    tqv = _var('merra2_slv', 'TQV', half_hour=False)
    slp = _var('merra2_slv', 'SLP', half_hour=False)
    vflx = -_var('merra2_flx', 'VFLXQV', half_hour=True)
    v850 = -_var('merra2_v850', 'V850', half_hour=True)

    out['max_T2m_ais'] = compute_max_intensity(storm, t2m, areas, ais)
    out['max_T2M_anomaly_ais'] = compute_max_intensity(
        storm, _monthly_anomaly(t2m, climatology, 'T2M'), areas, ais
    )
    out['max_IWV_ais'] = compute_max_intensity(storm, tqv, areas, ais)
    out['max_IWV_anomaly_ais'] = compute_max_intensity(
        storm, _monthly_anomaly(tqv, climatology, 'TQV'), areas, ais
    )
    out['avg_vIVT_ais'] = compute_average(storm, vflx, areas, ais)
    out['max_vIVT_ais'] = compute_max_intensity(storm, vflx, areas, ais)
    out['max_vIVT'] = compute_max_intensity(storm, vflx, areas, None)

    # The landfall-capture kernels assume a landfalling storm (the notebook
    # filtered on is_landfalling before computing them; np.min of an empty
    # landfall selection raises). zagg's trigger yields NaN for the same
    # storms, so the reference is NaN here by the pipeline's own semantics.
    landfalls = bool(storm.where(ais.sel(lat=storm.lat, lon=storm.lon), 0).any())
    if not landfalls:
        for name in ('min_SLP', 'max_SLP_gradient', 'max_landfalling_v850hPa',
                     'avg_landfalling_v850hPa', 'avg_landfalling_minomega'):
            out[name] = float('nan')
        return out

    out['min_SLP'] = compute_min_SLP(storm, slp, areas, ais)
    out['max_SLP_gradient'] = compute_max_SLPgrad(storm, slp, areas, ais)
    out['max_landfalling_v850hPa'] = compute_max_landfalling_wind(storm, v850, areas, ais)
    out['avg_landfalling_v850hPa'] = compute_avg_landfalling_wind(storm, v850, areas, ais)

    # OMEGA is on levels and 3-hourly; the kernel min-selects over lev itself
    omega_ds = collections['merra2_omega']
    omega = omega_ds['OMEGA'].sel(lat=storm.lat, lon=storm.lon, time=storm.time).compute()
    out['avg_landfalling_minomega'] = compute_avg_landfalling_minomega(storm, omega, areas, ais)
    return out


def reference_precip(storm_da, collections, statics):
    '''The two precip attributes, mirroring compute_precip_summaries.'''
    storm = storm_da.assign_coords(lat=storm_da.lat.round(5), lon=storm_da.lon.round(5))
    augmented = augment_storm_da(storm)
    areas = statics['cell_areas']
    ais = statics['ais_mask'].astype(bool)

    ds = collections['merra2_precip']
    ds = ds.assign_coords(time=ds.time - np.timedelta64(30, 'm'))
    totals = (ds[['PRECCU', 'PRECLS', 'PRECSN']] * 3600).resample(time='3h').sum()
    totals = totals.sel(lat=augmented.lat, lon=augmented.lon, time=augmented.time).compute()
    rainfall = totals['PRECCU'] + totals['PRECLS']
    return {
        'cumulative_rainfall_ais': compute_cumulative(augmented, rainfall, areas, ais),
        'cumulative_snowfall_ais': compute_cumulative(augmented, totals['PRECSN'], areas, ais),
    }


def _zagg_rows(config_path, tuples, workers=2):
    import zagg
    from zagg.config import load_config

    summary = zagg.agg(load_config(config_path), events=tuples, max_workers=workers)
    if summary['events_error']:
        raise RuntimeError(f'zagg run had {summary["events_error"]} event errors')
    return {r['event_key']: r['results'] for r in summary['results']}


def _compare(rows):
    '''rows: [(event_key, attribute, zagg_value, ref_value)] -> (table, ok).'''
    records, ok = [], True
    for key, attr, z, r in rows:
        both_nan = isinstance(z, float) and isinstance(r, float) and math.isnan(z) and math.isnan(r)
        if both_nan:
            status, rel = 'ok (NaN)', 0.0
        elif z is None or r is None:
            status, rel, ok = 'MISSING', float('inf'), False
        else:
            rel = abs(z - r) / max(abs(r), 1e-30)
            status = 'ok' if rel <= ARGS.rtol else 'FAIL'
            ok &= rel <= ARGS.rtol
        records.append({'event': key, 'attribute': attr, 'zagg': z, 'reference': r,
                        'rel_diff': rel, 'status': status})
    return pd.DataFrame(records), ok


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    catalog_path = Path(ARGS.catalogs_dir) / ARGS.catalog
    storm_ids = ARGS.storms or _pick_storms(catalog_path, ARGS.n_storms)
    logger.info('storms: %s', storm_ids)

    events, _ = export_events(catalog_path, storms=storm_ids)
    prepare_static_files(ARGS.catalogs_dir)
    statics = load_zagg_statics(ARGS.catalogs_dir)

    span = (
        str(pd.Timestamp(events['t_start'].min()).date()),
        str(pd.Timestamp(events['t_end_augmented'].max()).date()),
    )
    index = build_granule_index(span)

    raw_tuples = local_event_tuples(events, index, FIELD_COLLECTIONS, statics)
    aug_tuples = local_event_tuples(
        events, index, ['merra2_precip'], statics, augmented=True
    )

    logger.info('running zagg local backend (fields)...')
    zagg_fields = _zagg_rows(REPO / 'configs' / 'ar_attributes.yaml', raw_tuples)
    logger.info('running zagg local backend (precip)...')
    zagg_precip = _zagg_rows(REPO / 'configs' / 'ar_precip.yaml', aug_tuples)

    logger.info('running attribute_utils reference...')
    rows = []
    raw_by_key = {t[0]: t for t in raw_tuples}
    aug_by_key = {t[0]: t for t in aug_tuples}
    for _, event in events.iterrows():
        key = event['event_key']
        storm = xr.open_dataarray(event['mask_path'])
        ref = reference_attributes(storm, raw_by_key[key][2], statics)
        ref.update(reference_precip(storm, aug_by_key[key][2], statics))
        merged_zagg = {**zagg_fields.get(key, {}), **zagg_precip.get(key, {})}
        for attr, ref_value in ref.items():
            rows.append((key, attr, merged_zagg.get(attr), ref_value))

    table, ok = _compare(rows)
    pd.set_option('display.width', 200)
    print(table.to_string(index=False, float_format=lambda v: f'{v:.6g}'))
    n_fail = int((table['status'] == 'FAIL').sum() + (table['status'] == 'MISSING').sum())
    print(f'\n{"PARITY OK" if ok else f"PARITY FAILED ({n_fail} mismatches)"} '
          f'({len(rows)} comparisons, rtol={ARGS.rtol})')
    return 0 if ok else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--catalog', default=DEFAULT_CATALOG)
    parser.add_argument('--catalogs-dir', default=str(REPO.parent / 'antarctic_AR_catalogs'))
    parser.add_argument('--storms', type=int, nargs='*', default=None)
    parser.add_argument('--n-storms', type=int, default=6)
    parser.add_argument('--rtol', type=float, default=1e-5)
    ARGS = parser.parse_args()
    sys.exit(main())
